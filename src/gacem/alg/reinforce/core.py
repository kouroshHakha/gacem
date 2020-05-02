from typing import Tuple, cast
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import abc

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from gacem.models.made import MADE

import pdb

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def advantages(rewards, lperc=0.1):
    """
    Computes the relative advantages of rewards with simple ranking, top l percent get 1 and the
    rest a linearly decayed weight from 1 to -1
    """
    assert 0 < lperc and lperc < 1
    rewards_sorted = np.sort(rewards)[::-1]
    l = int(lperc * len(rewards))
    # select the reward only if it's negative
    rew_l = min(rewards_sorted[l], 0)
    weights = np.zeros_like(rewards)
    weights[rewards >= rew_l] = 1
    cond = rewards < rew_l
    # leave the remining weights 0 if there are not that many of them
    if np.sum(cond) > 2:
        weights[cond] = compute_centered_ranks(rewards[cond])
    return weights

def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y = 2 * (y / (x.size - 1)) - 1
    return y


class ARActor(nn.Module, abc.ABC):

    @abc.abstractmethod
    def get_logp(self, actions):
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, nsteps: int) -> torch.Tensor:
        raise NotImplementedError



class MADEBoxActor(ARActor, MADE):

    def __init__(self, action_space, hidden_sizes=(64, 64), fixed_sigma=None, nr_mix=40, bsize=16):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.ndim = action_space.shape[0]
        self.nr_mix = nr_mix
        self.action_space = action_space
        self.fixed_sigma = fixed_sigma
        self.bsize=bsize
        nparams_per_dim_mix = 2 if fixed_sigma else 3
        super().__init__(self.ndim, hidden_sizes, self.ndim * nparams_per_dim_mix * nr_mix,
                         natural_ordering=True)

    def _sample_actions(self, num: int, return_logp):
        aspace = self.action_space
        total_niter = -(-num // self.bsize)
        actions_list = []
        for iter_cnt in range(total_niter):
            if iter_cnt == total_niter - 1:
                bsize = num - iter_cnt * self.bsize
            else:
                bsize = self.bsize

            actions = torch.zeros(bsize, self.ndim, device=self.device)
            for i in range(self.ndim):
                coeffs, means, stds = self.get_dist_params(actions, index=i) # shape ~ (batch, nrmix)

                sel = coeffs.multinomial(num_samples=1)  # shape ~ (batch, 1)
                mean = means.gather(1, sel.view(-1,1))
                std = stds.gather(1, sel.view(-1,1))

                dist = Normal(mean, std)
                a_i = dist.sample()

                actions[..., i] = a_i.clamp(aspace.low[i].item(), aspace.high[i].item()).squeeze()

            actions_list.append(actions)

        nactions = torch.cat(actions_list, dim=0)
        logps = self.get_logp(nactions)

        if return_logp:
            return nactions, logps
        return nactions

    def get_dist_params(self, actions, index=None):
        actions = actions.to(self.device)
        ndim = self.ndim
        out_flat = self(actions)

        nparams_per_dim_mix = 2 if self.fixed_sigma else 3
        sigma_fixed = self.fixed_sigma is not None

        log_coeffs_flat = [out_flat[..., i::ndim * nparams_per_dim_mix] for i in range(ndim)]
        log_coeffs = torch.stack(log_coeffs_flat, dim=-2)
        coeffs = log_coeffs.softmax(dim=-1)

        means_flat = [out_flat[..., i+ndim::ndim*nparams_per_dim_mix] for i in range(ndim)]
        means = torch.stack(means_flat, dim=-2)

        if sigma_fixed:
            stds = torch.ones_like(means) * self.fixed_sigma
        else:
            log_sigma_flat = [out_flat[..., i+2*ndim::ndim*3] for i in range(ndim)]
            log_sigma = torch.stack(log_sigma_flat, dim=-2)
            # put a cap on the value of output so that it does not blow up
            # log_sigma = torch.min(log_sigma, torch.ones_like(log_sigma).to(self.device) * 50)
            # put a bottom on the value of output so that it does not diminish and becomes zero
            # to avoid numerical nan problems
            log_sigma = torch.max(log_sigma, torch.ones_like(log_sigma).to(self.device) * (-20))
            stds = log_sigma.exp()

        if index is None:
            return coeffs, means, stds

        return coeffs[..., index, :], means[..., index, :], stds[..., index, :]


    def get_logp(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        coeffs, means, stds = self.get_dist_params(actions)
        dist = Normal(means, stds)
        broadcasted_acts = actions[..., None] + torch.zeros_like(dist.loc)
        logp_i = ((coeffs * dist.log_prob(broadcasted_acts).exp()).sum(-1) + 2**(-self.ndim)).log()
        # average logp per dimension
        logp = logp_i.mean(-1)
        raw_log_prob = dist.log_prob(broadcasted_acts).mean()
        if logp.mean() == np.float('-inf') or torch.any(torch.isnan(raw_log_prob)):
            print('logp.mean = ', logp.mean())
            print('raw_log_prob = ', raw_log_prob)
            pdb.set_trace()
        return logp


    def act(self, nsteps: int, return_logp=True):
        with torch.no_grad():
            return self._sample_actions(nsteps, return_logp)



class Actor(nn.Module):

    def __init__(self, observation_space, action_space, **policy_kwargs):
        super().__init__()

        # only support Box action space for now
        assert isinstance(action_space, Box)
        # only support flattened action spaces for now
        assert len(action_space.shape) == 1

        obs_dim = observation_space.shape[0]

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Device = {self.device}')

        self.pi: ARActor = cast(ARActor, MADEBoxActor(action_space, **policy_kwargs))

        # move modules to cuda if necessary
        self.to(self.device)
        self.pi.to(self.device)


    def act(self, obs):
        # we only use obs input to determine the nsteps
        obs = obs.to(self.device)
        nsteps = 1 if obs.shape == 1 else obs.shape[0]
        return self.pi.act(nsteps)

