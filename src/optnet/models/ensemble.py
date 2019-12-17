"""
This module implements an ensemble model for cont_autoreg model
"""
from typing import Tuple, List

import random

import numpy as np
import torch
import torch.nn as nn

from ..torch.dist import cdf_logistic, cdf_normal, cdf_uniform


class Ensemble(nn.Module):

    def __init__(self,
                 modules: List[nn.Module],
                 ndim: int,
                 base_fn: str,
                 delta: float,
                 beta: float,
                 seed: int = 10,
                 ):
        # randomize models in ensemble using different seeds
        super(Ensemble).__init__()

        self.modules = modules

        num_modules = len(modules)
        self.devices = []
        if not torch.cuda.is_available():
            self.devices = [torch.device('cpu')] * num_modules
        else:
            for i in range(num_modules):
                device = torch.device(f'cuda:{i % num_modules}')
                self.devices.append(device)

        # initialize models, even if they are initialized already
        for i, (model, device) in enumerate(zip(self.modules, self.devices)):
            model.to(device)
            if torch.cuda.is_available():
                # noinspection PyUnresolvedReferences
                with torch.cuda.device(device.index):
                    print(f'device {torch.cuda.current_device()} initialized')
                    self.set_torch_seed(seed + 10 * i)
            model.apply(self.init_weights)

        self.ndim = ndim
        self.base_fn = base_fn
        self.delta = delta
        self.beta = beta

    @classmethod
    def init_weights(cls, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.uniform_(m.bias, a=-0.01, b=0.01)

    @classmethod
    def set_torch_seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def forward(self, x):
        y_list = [model(x.to(device)) for model, device in zip(self.modules, self.devices)]
        y = torch.stack(y_list, -1)
        return y.mean(-1)

    def get_probs(self, xin: torch.Tensor):
        """Given an input tensor (N, dim) computes the probabilities across the cardinality space
        of each dimension. prob.shape = (N, dim, K) where K is number of possible values for each
        dimension. Assume that xin is normalized to [-1,1], and delta is given."""
        n_modules = len(self.modules)
        rnd_index = random.randint(n_modules)

        model = self.modules[rnd_index]
        device = self.devices[rnd_index]
        xin = xin.to(device)
        xhat = model(xin)

        # xhat=  self(xin)
        # device = self.devices[0]

        delta = self.delta
        dim = self.ndim

        coeffs = torch.stack([xhat[..., i::dim*3] for i in range(dim)], dim=-2)
        # Note: coeffs was previously interpreted as log_coeffs
        # interpreting outputs if NN as log is dangerous, can result in Nan's.
        # solution: here they should be positive and should add up to 1, sounds familiar? softmax!
        coeffs_norm = coeffs.softmax(dim=-1)

        eps = torch.tensor(1e-15)
        xb = xin[..., None] + torch.zeros(coeffs.shape, device=device)

        if self.base_fn in ['logistic', 'normal']:
            means = torch.stack([xhat[..., i+dim::dim*3] for i in range(dim)], dim=-2)
            log_sigma = torch.stack([xhat[..., i+2*dim::dim*3] for i in range(dim)], dim=-2)
            # put a cap on the value of output so that it does not blow up
            log_sigma = torch.min(log_sigma, torch.ones(log_sigma.shape).to(device) * 50)
            # put a bottom on the value of output so that it does not diminish and becomes zero
            log_sigma = torch.max(log_sigma, torch.ones(log_sigma.shape).to(device) * (-40))
            sigma = log_sigma.exp()

            if self.base_fn == 'logistic':
                plus_cdf = cdf_logistic(xb + delta / 2, means, sigma)
                minus_cdf = cdf_logistic(xb - delta / 2, means, sigma)
            else:
                plus_cdf = cdf_normal(xb + delta / 2, means, sigma)
                minus_cdf = cdf_normal(xb - delta / 2, means, sigma)
        elif self.base_fn == 'uniform':
            center = torch.stack([xhat[..., i+dim::dim*3] for i in range(dim)], dim=-2)
            # normalize center between [-1,1] to cover all the space
            center = 2 * (center - center.min()) / (center.max() - center.min() + eps) - 1
            log_delta = torch.stack([xhat[..., i+2*dim::dim*3] for i in range(dim)], dim=-2)
            # put a cap on the value of output so that it does not blow up
            log_delta = torch.min(log_delta, torch.ones(log_delta.shape) * 50)
            bdelta = log_delta.exp()
            a = center - bdelta / 2
            b = center + bdelta / 2
            plus_cdf = cdf_uniform(xb + delta / 2, a, b)
            minus_cdf = cdf_uniform(xb - delta / 2, a, b)
        else:
            raise ValueError(f'unsupported base_fn = {self.base_fn}')

        # -1 is mapped to (-inf, -1+d/2], 1 is mapped to [1-d/2, inf), and other 'i's are mapped to
        # [i-d/2, i+d/2)n
        probs_nonedge = plus_cdf - minus_cdf
        probs_right_edge = torch.tensor(1) - minus_cdf
        probs_left_edge = plus_cdf

        l_cond = xb <= (-1 + delta / 2)
        r_cond = xb >= (1 - delta / 2)
        n_cond = ~(l_cond | r_cond)
        cdfs = probs_left_edge * l_cond + probs_right_edge * r_cond + probs_nonedge * n_cond

        probs = (coeffs_norm * cdfs).sum(-1)

        return probs

    def get_nll(self, xin: torch.Tensor, weights=None):
        """Given an input tensor computes the average negative likelihood of observing the inputs"""
        probs = self.get_probs(xin)
        eps_tens = 1e-15
        logp_vec = (probs + eps_tens).log10().sum(-1)

        if weights is None:
            min_obj = -logp_vec.mean(-1)
        else:
            pos_ind = (weights > 0).float()
            # neg_ind = 1 - pos_ind

            obj_term  = - weights.data
            ent_term = (self.beta * (1 + logp_vec)).data

            main_obj = obj_term * logp_vec
            ent_obj = ent_term * logp_vec

            npos = pos_ind.sum(-1)
            npos = 1 if npos == 0 else npos
            pos_main_obj = (main_obj * pos_ind).sum(-1) / npos
            pos_ent_obj = (ent_obj * pos_ind).sum(-1) / npos

            # nneg = neg_ind.sum(-1)
            # nneg = 1 if nneg == 0 else nneg
            # neg_main_obj = (main_obj * neg_ind).sum(-1) / nneg
            # neg_ent_obj = (ent_obj * neg_ind).sum(-1) / nneg

            min_obj = (pos_main_obj + pos_ent_obj) / self.ndim

        return min_obj

    def sample_model(self, nsamples: int, bsize: int,
                     input_vectors_norm: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """samples the current model nsamples times and returns both normalized samples i.e
        between [-1, 1] and sample indices

        Parameters
        ----------
        nsamples: int
            number of samples
        bsize: int
            batch size
        input_vectors_norm: List[np.ndarray]
            list of input vector normalized values (i.e. [-1, 1])

        Returns
        -------
        samples: Tuple[torch.Tensor, torch.Tensor]
            normalized samples / sample indices
        """
        self.eval()

        # synchronize operations on one device
        device = self.devices[0]
        dim = self.ndim
        # GPU friendly sampling
        total_niter = -(-nsamples // bsize)
        xsample_list, xsample_ind_list = [], []
        for iter_cnt in range(total_niter):
            if iter_cnt == total_niter - 1:
                bsize = nsamples - iter_cnt * bsize
            else:
                bsize = bsize
            xsample = torch.zeros(bsize, dim, device=device)
            xsample_ind = torch.zeros(bsize, dim, device=device)
            for i in range(dim):
                n = len(input_vectors_norm[i])
                xin = torch.zeros(bsize, n, dim, device=device)
                if i >= 1:
                    xin = torch.stack([xsample] * n, dim=-2)
                in_torch = torch.from_numpy(input_vectors_norm[i]).to(device)
                xin[..., i] = torch.stack([in_torch] * bsize)
                xin_reshaped = xin.view((bsize * n, dim))
                probs_reshaped = self.get_probs(xin_reshaped)
                probs = probs_reshaped.view((bsize, n, dim))
                xi_ind = self.sample_probs(probs, i)  # ith x index
                xsample[:, i] = xin[..., i][range(bsize), xi_ind]
                xsample_ind[:, i] = xi_ind
            xsample_ind_list.append(xsample_ind)
            xsample_list.append(xsample)

        xsample = torch.cat(xsample_list, dim=0)
        xsample_ind = torch.cat(xsample_ind_list, dim=0)
        return xsample, xsample_ind

    @classmethod
    def sample_probs(cls, probs: torch.Tensor, index: int):
        """Given a probability distribution tensor (shape = (N, D, K)) returns 1 sample from the
        distribution probs[:, index, :], the output is indices"""
        p = probs[..., index]
        sample = p.multinomial(num_samples=1).squeeze(-1)
        return sample
