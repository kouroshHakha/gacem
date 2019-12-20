"""
This module implements an ensemble model for cont_autoreg model
"""
from typing import Tuple, List, Sized, Iterable, Optional

import random
import numpy as np
import GPUtil
import torch
import torch.nn as nn

from ..torch.dist import cdf_logistic, cdf_normal, cdf_uniform

import pdb
# import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

class ListModule(nn.Module, Sized, Iterable):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __next__(self):
        for i in range(len(self)):
            return self[i]
        raise StopIteration

    def __len__(self) -> int:
        return len(self._modules)

class Ensemble(nn.Module):

    def __init__(self,
                 models: List[nn.Module],
                 ndim: int,
                 base_fn: str,
                 delta: float,
                 beta: float,
                 seed: int = 10,
                 ):
        # randomize models in ensemble using different seeds
        super(Ensemble, self).__init__()

        self.models: ListModule = ListModule(*models)

        num_modules = len(models)
        GPUtil.showUtilization()
        availablity = GPUtil.getAvailability(GPUtil.getGPUs())
        available_devices = [i for i in range(len(availablity)) if availablity[i] == 1]
        self.cuda = (len(available_devices) > 0) # whether cuda is being used
        self.devices = []
        if not self.cuda:
            print('no cuda device was found, running on CPU')
            self.devices = [torch.device('cpu')] * num_modules
        else:
            for i in range(num_modules):
                device = torch.device(f'cuda:{available_devices[0]}')
                # device = torch.device(f'cuda:{available_devices[i % len(available_devices)]}')
                self.devices.append(device)

        # initialize models, even if they are initialized already
        for i, (model, device) in enumerate(zip(iter(self.models), self.devices)):
            model.to(device)
            print(f'model {i} was move to device: {device}')
            self.set_torch_seed(seed + 10 * i)
            model.apply(self.init_weights)
            print(f'model {i} was initialized')

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
        pass

    def get_probs(self, xin: torch.Tensor, model_index=None):
        """Given an input tensor (N, dim) computes the probabilities across the cardinality space
        of each dimension. prob.shape = (N, dim, K) where K is number of possible values for each
        dimension. Assume that xin is normalized to [-1,1], and delta is given."""
        if model_index is None:
            raise ValueError('model index not specified')
        model = self.models[model_index]
        device = self.devices[model_index]
        xin_device = xin.to(device, copy=True)
        xhat = model(xin_device)

        # xhat=  self(xin)
        # device = self.devices[0]

        delta = self.delta
        dim = self.ndim

        coeffs = torch.stack([xhat[..., i::dim*3] for i in range(dim)], dim=-2)
        # Note: coeffs was previously interpreted as log_coeffs
        # interpreting outputs if NN as log is dangerous, can result in Nan's.
        # solution: here they should be positive and should add up to 1, sounds familiar? softmax!
        coeffs_norm = coeffs.softmax(dim=-1)

        eps = 1e-15
        xb = xin_device[..., None] + torch.zeros(coeffs.shape, device=device)

        if self.base_fn in ['logistic', 'normal']:
            means = torch.stack([xhat[..., i+dim::dim*3] for i in range(dim)], dim=-2)
            log_sigma = torch.stack([xhat[..., i+2*dim::dim*3] for i in range(dim)], dim=-2)
            # put a cap on the value of output so that it does not blow up
            log_sigma = torch.min(log_sigma, torch.ones(log_sigma.shape).to(device) * 50)
            # put a bottom on the value of output so that it does not diminish and becomes zero
            log_sigma = torch.max(log_sigma, torch.ones(log_sigma.shape).to(device) * (-40))
            sigma = log_sigma.exp()

            if self.base_fn == 'logistic':
                cdf_func = cdf_logistic
            else:
                cdf_func = cdf_normal

            plus_cdf = cdf_func(xb + delta / 2, means, sigma)
            minus_cdf = cdf_func(xb - delta / 2, means, sigma)
            right = cdf_func(torch.ones_like(xb), means, sigma)
            left = cdf_func(-torch.ones_like(xb), means, sigma)
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
            right = cdf_uniform(torch.ones_like(xb), a, b)
            left = cdf_uniform(-torch.ones_like(xb), a, b)
        else:
            raise ValueError(f'unsupported base_fn = {self.base_fn}')

        # -1 is mapped to (-inf, -1+d/2], 1 is mapped to [1-d/2, inf), and other 'i's are mapped to
        # [i-d/2, i+d/2)n
        probs_nonedge = plus_cdf - minus_cdf
        probs_right_edge = right - minus_cdf
        probs_left_edge = plus_cdf - left

        l_cond = xb <= (-1 + delta / 2)
        r_cond = xb >= (1 - delta / 2)
        n_cond = ~(l_cond | r_cond)
        cdfs = probs_left_edge * l_cond + probs_right_edge * r_cond + probs_nonedge * n_cond

        cdfs = cdfs / (right - left + eps)
        if torch.any(torch.isnan(cdfs)):
            pdb.set_trace()
        probs = (coeffs_norm * cdfs).sum(-1)

        return probs

    def get_nll(self, xin: torch.Tensor, weights: Optional[torch.Tensor] = None,
                ll_type = 'logp'):
        """Given an input tensor computes the average negative likelihood of observing the inputs"""
        min_objs = []
        n_modules = len(self.models)
        for i in range(n_modules):
            # if self.cuda:
            #     torch.cuda.set_device(self.devices[i])
            probs = self.get_probs(xin, model_index=i)
            eps_tens = 1e-15
            logp_vec = (probs + eps_tens).log10().sum(-1)
            log_pp_vec = (1-probs+eps_tens).log10().sum(-1)

            if ll_type is not 'logp':
                min_obj = -log_pp_vec.mean(-1)
                min_objs.append(min_obj)
                continue

            if weights is None:
                min_obj = -logp_vec.mean(-1)
                min_objs.append(min_obj)
            else:
                w_tens = weights.to(logp_vec, copy=True)
                pos_ind = (w_tens > 0).float()

                obj_term  = - w_tens.data
                ent_term = (self.beta * (1 + logp_vec)).data

                main_obj = obj_term * logp_vec
                ent_obj = ent_term * logp_vec

                npos = pos_ind.sum(-1)
                npos = 1 if npos == 0 else npos
                pos_main_obj = (main_obj * pos_ind).sum(-1) / npos
                pos_ent_obj = (ent_obj * pos_ind).sum(-1) / npos

                min_sub_obj = (pos_main_obj + pos_ent_obj) / self.ndim
                # use one of the devices as the central host for common computations
                min_objs.append(min_sub_obj.to(self.devices[0]))

        min_obj = torch.stack(min_objs, -1).mean(-1)
        return min_obj

    @classmethod
    def gen_n_numbers_with_sum(cls, n=5, sum=1):
        ans = np.random.multinomial(sum, np.ones(n)/n, size=1)[0].tolist()
        return ans

    def sample_model(self, nsamples: int, bsize: int,
                     input_vectors_norm: List[np.ndarray],
                     model_index=None) -> Tuple[torch.Tensor, torch.Tensor]:
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

        n_modules = len(self.models)
        device_shares = self.gen_n_numbers_with_sum(n_modules, nsamples)
        index = random.randrange(n_modules) if model_index is None else model_index

        dim = self.ndim
        # GPU friendly sampling
        samples, sample_ids = [], []
        for i, (device, device_nsamp) in enumerate(zip(self.devices, device_shares)):
            if device_nsamp == 0:
                continue
            # print(f'sampling {device_nsamp} samples from model {i}')
            total_niter = -(-device_nsamp // bsize)
            xsample_list, xsample_ind_list = [], []
            for iter_cnt in range(total_niter):
                if iter_cnt == total_niter - 1:
                    bsize = device_nsamp - iter_cnt * bsize
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
                    probs_reshaped = self.get_probs(xin_reshaped, model_index=index)
                    probs = probs_reshaped.view((bsize, n, dim))
                    xi_ind = self.sample_probs(probs, i)  # ith x index
                    xsample[:, i] = xin[..., i][range(bsize), xi_ind]
                    xsample_ind[:, i] = xi_ind
                xsample_ind_list.append(xsample_ind)
                xsample_list.append(xsample)
            samples.append(torch.cat(xsample_list, dim=0).to(self.devices[0]))
            sample_ids.append(torch.cat(xsample_ind_list, dim=0).to(self.devices[0]))
        samples = torch.cat(samples, dim=0)
        sample_ids = torch.cat(sample_ids, dim=0)
        return samples, sample_ids

    @classmethod
    def sample_probs(cls, probs: torch.Tensor, index: int):
        """Given a probability distribution tensor (shape = (N, D, K)) returns 1 sample from the
        distribution probs[:, index, :], the output is indices"""
        p = probs[..., index]
        sample = p.multinomial(num_samples=1).squeeze(-1)
        return sample