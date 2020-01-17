"""
This module implements an algorithm for and optimizer that uses continuous auto-regressive
probability density estimator models to estimate the probability of solution regions for a
multi-constrained black-box optimization
"""
from typing import Optional, Mapping, Any, Tuple, Union

import math
import pdb
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.file import read_yaml, write_yaml, get_full_name, write_pickle, read_pickle
from utils.loggingBase import LoggingBase

from ...benchmarks.functions import registered_functions
from ...benchmarks.fom import (
    compute_emprical_variation, get_diversity_fom
)
from ...torch.dist import cdf_logistic, cdf_normal, cdf_uniform
from ...data.buffer import CacheBuffer
from ...models.made import MADE
from ..random.random import Random
from ...data.vector import index_to_xval, xval_to_index
from ..utils.weight_compute import weight2

from ...viz.plot import (
    plot_pca_2d, plt_hist2D, plot_cost, plot_learning_with_epochs, plot_x_y
)

from sortedcontainers import SortedList

class AutoRegSearch(LoggingBase):

    # noinspection PyUnusedLocal
    def __init__(self, spec_file: str = '', spec_dict: Optional[Mapping[str, Any]] = None,
                 load: bool = False, use_time_stamp: bool = True,
                 init_buffer_path = None, **kwargs) -> None:
        LoggingBase.__init__(self)

        if spec_file:
            specs = read_yaml(spec_file)
        else:
            specs = spec_dict

        self.specs = specs
        params = specs['params']

        if load:
            self.work_dir = Path(spec_file).parent
        else:
            suffix = params.get('suffix', '')
            prefix = params.get('prefix', '')
            if use_time_stamp:
                unique_name = time.strftime('%Y%m%d%H%M%S')
                unique_name = get_full_name(unique_name, prefix, suffix)
            else:
                unique_name = f'{prefix}' if prefix else ''
                if suffix:
                    unique_name = f'{unique_name}_{suffix}' if unique_name else f'{suffix}'

            self.work_dir = Path(specs['root_dir']) / f'{unique_name}'
            write_yaml(self.work_dir / 'params.yaml', specs, mkdir=True)

        self.load = load
        self.seed = params.get('seed', 10)
        self.ndim = params['ndim']
        self.bsize = params['batch_size']
        self.hiddens = params['hidden_list']
        self.niter = params['niter']
        self.goal = params['goal_value']
        self.mode = params['mode']
        self.viz_rate = self.niter // 10
        self.lr = params['lr']
        self.nepochs = params['nepochs']
        self.nsamples = params['nsamples']
        self.n_init_samples = params['n_init_samples']
        self.init_nepochs = params['init_nepochs']
        self.cut_off = params['cut_off']
        self.beta = params['beta']
        self.nr_mix = params['nr_mix']
        self.base_fn = params['base_fn']
        self.only_pos = params['only_positive']
        # whether to run 1000 epochs of training for the later round of iteration
        self.full_training = params['full_training_last']
        self.input_scale = params['input_scale']
        self.fixed_sigma = params.get('fixed_sigma', None)
        self.on_policy = params.get('on_policy', False)
        self.problem_type = params.get('problem_type', 'csp')

        self.allow_repeated = params.get('allow_repeated', False)
        self.allow_repeated = self.on_policy or self.allow_repeated

        self.important_sampling = params.get('important_sampling', False)
        self.visited_dist: Optional[nn.Module] = None
        self.visited_fixed_sigma = params.get('visited_fixed_sigma', None)
        self.visited_nr_mix = params.get('visited_nr_mix', None)

        self.explore_coeff = params.get('explore_coeff', None)
        self.nepoch_visited = params.get('nepoch_visited', -1)

        self.normalize_weight = params.get('normalize_weight', True)
        self.add_ent_before_norm = params.get('add_entropy_before_normalization', False)

        self.model_visited = self.explore_coeff is not None or self.important_sampling

        if self.model_visited and self.nepoch_visited == -1:
            raise ValueError('nepoch_visited should be specified when a model is '
                             'learning visited states')


        self.init_buffer_paths = init_buffer_path

        eval_fn = params['eval_fn']
        try:
            self.fn = registered_functions[eval_fn]
        except KeyError:
            raise ValueError(f'{eval_fn} is not a valid benchmark function')

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'device: {self.device}')
        self.cpu = torch.device('cpu')
        self.model: Optional[nn.Module] = None
        self.buffer = None
        self.opt = None

        # hacky version of passing input vectors around
        self.input_vectors_norm = [np.linspace(start=-1.0, stop=1.0, dtype='float32',
                                               num=100) for _ in range(self.ndim)]
        self.input_vectors = [self.input_scale * vec for vec in self.input_vectors_norm]
        # TODO: remove this hacky way of keeping track of delta
        self.delta = self.input_vectors_norm[0][-1] - self.input_vectors_norm[0][-2]

        # keep track of lo and hi for indicies
        self.params_min = np.array([0] * self.ndim)
        self.params_max = np.array([len(x) - 1 for x in self.input_vectors])

        self.fvals = SortedList()

    @classmethod
    def set_seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(seed)

    def get_probs(self, xin: torch.Tensor, model: nn.Module, debug=False):
        """Given an input tensor (N, dim) computes the probabilities across the cardinality space
        of each dimension. prob.shape = (N, dim, K) where K is number of possible values for each
        dimension. Assume that xin is normalized to [-1,1], and delta is given."""
        delta = self.delta
        xin = xin.to(self.device)
        xhat = model(xin)
        dim = self.ndim
        if model is self.model:
            nparams_per_dim_mix = 2 if self.fixed_sigma else 3
            sigma_fixed = self.fixed_sigma is not None
        else:
            nparams_per_dim_mix = 2 if self.visited_fixed_sigma else 3
            sigma_fixed = self.visited_fixed_sigma is not None

        coeffs = torch.stack([xhat[..., i::dim*nparams_per_dim_mix] for i in range(dim)], dim=-2)
        # Note: coeffs was previously interpreted as log_coeffs
        # interpreting outputs if NN as log is dangerous, can result in Nan's.
        # solution: here they should be positive and should add up to 1, sounds familiar? softmax!
        coeffs_norm = coeffs.softmax(dim=-1)

        eps = 1e-15
        xb = xin[..., None] + torch.zeros(coeffs.shape, device=self.device)

        if self.base_fn in ['logistic', 'normal']:
            means = torch.stack([xhat[..., i+dim::dim*nparams_per_dim_mix] for i in range(dim)],
                                dim=-2)
            if sigma_fixed:
                sigma = self.fixed_sigma if model is self.model else self.visited_fixed_sigma
            else:
                log_sigma = torch.stack([xhat[..., i+2*dim::dim*3] for i in range(dim)], dim=-2)
                # put a cap on the value of output so that it does not blow up
                log_sigma = torch.min(log_sigma, torch.ones(log_sigma.shape).to(self.device) * 50)
                # put a bottom on the value of output so that it does not diminish and becomes zero
                log_sigma = torch.max(log_sigma, torch.ones(log_sigma.shape).to(self.device) * (-40))
                sigma = log_sigma.exp()

            if self.base_fn == 'logistic':
                plus_cdf = cdf_logistic(xb + delta / 2, means, sigma)
                minus_cdf = cdf_logistic(xb - delta / 2, means, sigma)
            else:
                plus_cdf = cdf_normal(xb + delta / 2, means, sigma)
                minus_cdf = cdf_normal(xb - delta / 2, means, sigma)
        elif self.base_fn == 'uniform':
            # does not work with self.fixed_sigma
            if self.fixed_sigma:
                raise ValueError('base_fn cannot be uniform when fixed_sigma is given!')
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
        probs_right_edge = 1 - minus_cdf
        probs_left_edge = plus_cdf

        l_cond = xb <= (-1 + delta / 2)
        r_cond = xb >= (1 - delta / 2)
        n_cond = ~(l_cond | r_cond)
        cdfs = probs_left_edge * l_cond + probs_right_edge * r_cond + probs_nonedge * n_cond

        probs = (coeffs_norm * cdfs).sum(-1)

        if debug:
            pdb.set_trace()

        return probs

    def get_nll(self, xin: torch.Tensor, model: nn.Module, weights=None, debug=False):
        """Given an input tensor computes the average negative likelihood of observing the inputs"""
        probs = self.get_probs(xin, model=model)
        eps_tens = 1e-15
        logp_vec = (probs + eps_tens).log().sum(-1)

        if weights is None:
            min_obj = -logp_vec.mean(-1)
        else:
            pos_ind = (weights > 0).float()
            neg_ind = 1 - pos_ind

            # obj_term  = - self.buffer.size * (weights * prob_x).data
            # ent_term = self.buffer.size * (self.beta * (torch.tensor(1) + logp_vec)).data
            obj_term  = - weights.data

            # TODO: Fix this bad code
            if self.add_ent_before_norm:
                ent_term = 0
            else:
                ent_term = (self.beta * (1 + logp_vec)).data

            # important sampling coefficient (with frozen gradient)
            if self.important_sampling:
                probs_visited = self.get_probs(xin, model=self.visited_dist)
                logp_visited = (probs_visited + eps_tens).log().sum(-1).to(logp_vec)
                # is_coeff = (torch.tensor(10**(-self.ndim * 2)).log() - logp_visited).exp()
                is_coeff = torch.clamp((logp_vec - logp_visited).exp(), 1e-15, 1e15).data
                is_coeff = is_coeff / is_coeff.max()
            else:
                is_coeff = 1

            main_obj = obj_term * logp_vec * is_coeff
            ent_obj = 1 / self.ndim * ent_term * logp_vec * is_coeff

            npos = pos_ind.sum(-1)
            npos = 1 if npos == 0 else npos
            pos_main_obj = (main_obj * pos_ind).sum(-1) / npos
            pos_ent_obj = (ent_obj * pos_ind).sum(-1) / npos

            nneg = neg_ind.sum(-1)
            nneg = 1 if nneg == 0 else nneg
            neg_main_obj = (main_obj * neg_ind).sum(-1) / nneg
            neg_ent_obj = (ent_obj * neg_ind).sum(-1) / nneg

            if self.only_pos:
                min_obj = (pos_main_obj + pos_ent_obj) / self.ndim
            else:
                min_obj = (pos_main_obj + neg_main_obj + pos_ent_obj + neg_ent_obj) / self.ndim

            if debug:
                for w, lp in zip(weights, logp_vec):
                    print(f'w = {w:10.4}, prob = {torch.exp(lp):10.4}')
                # probs = self.get_probs(xin, debug=True)
                foo = torch.autograd.grad(min_obj, model.net[0].weight, retain_graph=True)
                print(foo)
                pdb.set_trace()

            if torch.isnan(min_obj):
                print(min_obj)
                pdb.set_trace()

        return min_obj

    @classmethod
    def sample_probs(cls, probs: torch.Tensor, index: int):
        """Given a probability distribution tensor (shape = (N, D, K)) returns 1 sample from the
        distribution probs[:, index, :], the output is indices"""
        p = probs[..., index]
        sample = p.multinomial(num_samples=1).squeeze(-1)
        return sample

    def sample_model(self,nsamples: int,  model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """samples the current model nsamples times and returns both normalized samples i.e
        between [-1, 1] and sample indices

        Parameters
        ----------
        nsamples: int
            number of samples

        Returns
        -------
        samples: Tuple[torch.Tensor, torch.Tensor]
            normalized samples / sample indices
        """
        model.eval()

        dim = self.ndim

        # GPU friendly sampling
        if self.device != torch.device('cpu'):
            total_niter = -(-nsamples // self.bsize)
            xsample_list, xsample_ind_list = [], []
            for iter_cnt in range(total_niter):
                if iter_cnt == total_niter - 1:
                    bsize = nsamples - iter_cnt * self.bsize
                else:
                    bsize = self.bsize
                xsample = torch.zeros(bsize, dim, device=self.device)
                xsample_ind = torch.zeros(bsize, dim, device=self.device)
                for i in range(dim):
                    n = len(self.input_vectors_norm[i])
                    xin = torch.zeros(bsize, n, dim, device=self.device)
                    if i >= 1:
                        xin = torch.stack([xsample] * n, dim=-2)
                    in_torch = torch.from_numpy(self.input_vectors_norm[i]).to(self.device)
                    xin[..., i] = torch.stack([in_torch] * bsize)
                    xin_reshaped = xin.view((bsize * n, dim))
                    probs_reshaped = self.get_probs(xin_reshaped, model=model)
                    probs = probs_reshaped.view((bsize, n, dim))
                    xi_ind = self.sample_probs(probs, i)  # ith x index
                    xsample[:, i] = xin[..., i][range(bsize), xi_ind]
                    xsample_ind[:, i] = xi_ind
                xsample_ind_list.append(xsample_ind)
                xsample_list.append(xsample)

            xsample = torch.cat(xsample_list, dim=0)
            xsample_ind = torch.cat(xsample_ind_list, dim=0)
            return xsample, xsample_ind
        else:
            samples = []
            samples_ind = []
            for k in range(nsamples):
                xsample = torch.zeros(1, dim)
                xsample_ind = torch.zeros(1, dim)
                for i in range(dim):
                    n = len(self.input_vectors_norm[i])
                    xin = torch.zeros(n, dim)
                    if i >= 1:
                        xin = torch.stack([xsample.squeeze()] * n)
                    xin[:, i] = torch.from_numpy(self.input_vectors_norm[i])
                    # TODO: For normal dist this probs gets a lot of mass on the edges
                    probs = self.get_probs(xin, model=model)
                    xi_ind = self.sample_probs(probs, i)  # ith x index
                    xsample[0, i] = torch.tensor(self.input_vectors_norm[i][xi_ind])
                    xsample_ind[0, i] = xi_ind
                samples.append(xsample.squeeze())
                samples_ind.append(xsample_ind.squeeze())

            samples = torch.stack(samples, dim=0)
            samples_ind = torch.stack(samples_ind, dim=0)
            return samples, samples_ind

    def run_epoch(self, data: np.ndarray, weights: np.ndarray, model: nn.Module, mode='train',
                  debug=False):

        # for model in [self.visited_dist, self.model]:
        model.train() if mode == 'train' else model.eval()

        n, dim,  _ = data.shape

        assert n != 0, 'no data found'

        bsize = max(self.bsize, 2 ** math.floor(math.log2(n / 4))) if mode == 'train' else n
        nstep = n // bsize if mode == 'train' else 1

        nll_per_b = 0

        for step in range(nstep):
            xb = data[step * bsize: step * bsize + bsize]
            wb = weights[step * bsize: step * bsize + bsize]
            xb_tens = torch.from_numpy(xb).to(self.device)
            wb_tens = torch.from_numpy(wb).to(self.device)

            xin = xb_tens[:, 0, :]
            if model is self.model:
                loss = self.get_nll(xin, weights=wb_tens, model=self.model, debug=debug)
            else:
                loss = self.get_nll(xin, model=self.visited_dist)

            if mode == 'train':
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1e3)
                self.opt.step()
            nll_per_b += loss.to(self.cpu).item() / nstep

        return nll_per_b

    def collect_samples(self, n_samples, uniform=False):
        n_collected = 0
        new_samples = []
        vecs = self.input_vectors
        norm_vecs = self.input_vectors_norm
        # a counter for corner cases to tell the algorithm to explore more
        n_repetitive_samples = 0
        while n_collected < n_samples:
            if uniform or n_repetitive_samples > 1e2 * n_samples:
                _, xnew_id_np = Random.sample_data(self.ndim, self.input_vectors_norm, 1)
                xnew_id_np = xnew_id_np.astype('int')
            else:
                _, xnew_ind = self.sample_model(1, model=self.model)
                xnew_id_np = xnew_ind.to(self.cpu).data.numpy().astype('int')

            # simulate and compute the adjustment weights
            org_sample_list = [vecs[index][pos] for index, pos in enumerate(xnew_id_np[0, :])]
            xsample = np.array(org_sample_list, dtype='float32')
            fval = self.fn(xsample[None, :])
            self.fvals.add(fval)

            norm_sample_list = [norm_vecs[index][pos] for index, pos in enumerate(xnew_id_np[0, :])]
            norm_sample = np.array(norm_sample_list, dtype='float32')

            if self.allow_repeated or norm_sample not in self.buffer:
                self.buffer.add_samples(norm_sample[None, :], xnew_id_np, fval)
                new_samples.append(xsample)
                n_collected += 1
            else:
                n_repetitive_samples += 1
                print(f'item {norm_sample} already exists!')

        return new_samples

    def _clip_and_round(self, samples):
        lo = np.zeros(samples.shape) + self.params_min
        hi = np.zeros(samples.shape) + self.params_max
        out_samples = np.clip(samples, lo, hi)
        out_samples = np.floor(out_samples).astype('int')
        return out_samples

    def _sample_model_with_weights(self, nsample):
        xsample, xsample_ids, fvals = self._sample_model_for_eval(nsample)
        xsample_norm = index_to_xval(self.input_vectors_norm, xsample_ids)
        zavg = sorted(fvals, reverse=(self.mode == 'ge'))[self.cut_off]
        print(f'fref: {zavg}')
        weights = weight2(fvals, self.goal, zavg, self.mode, self.problem_type)
        # weights = self.update_weight(xsample, weights)
        return np.stack([xsample_norm, xsample_ids], axis=1), weights

    def update_weight(self, xin, wtr):
        eps_tens = 1e-15
        xin_tens = torch.from_numpy(xin).to(self.device)

        normalized = False
        if self.add_ent_before_norm:
            probs = self.get_probs(xin_tens, model=self.model)
            logp_vec = (probs + eps_tens).log().sum(-1).cpu().data.numpy()

            ent_term = (1 + logp_vec) / self.ndim
            ent_term = (ent_term -ent_term.mean()) / (ent_term.std() + eps_tens)
            # this normalization happens regardless of self.nomralize_weight flag
            wtr = (wtr - wtr.mean()) / (wtr.std() + eps_tens)
            wtr = wtr - self.beta * ent_term
            if self.beta != 0 and self.normalize_weight:
                wtr = (wtr - wtr.mean()) / (wtr.std() + eps_tens)
                normalized = True

        # this is just an experimentation
        # normalize before adding exploration penalty
        if self.explore_coeff is not None:
            probs_visited = self.get_probs(xin_tens, model=self.visited_dist)
            logp_visited = (probs_visited + eps_tens).log().sum(-1) / self.ndim / 2

            probs = self.get_probs(xin_tens, model=self.model)
            logp_vec = (probs + eps_tens).log().sum(-1)
            is_coeff = (logp_vec - logp_visited).exp()
            is_coeff = is_coeff / is_coeff.max()
            print('std: ',is_coeff.std())

            logp_visited = logp_visited.data.numpy()
            wtr = wtr - self.explore_coeff * logp_visited / (is_coeff.std() + eps_tens).item()
            if self.normalize_weight:
                wtr = (wtr - wtr.mean()) / (wtr.std() + eps_tens)
                normalized = True

        if not normalized and self.normalize_weight:
            wtr = (wtr - wtr.mean()) / (wtr.std() + eps_tens)

        return wtr

    def train(self, iter_cnt: int, nepochs: int, split=1.0):
        # treat the sampled data as a static data set and take some gradient steps on it
        print('-'*50)
        if self.on_policy and iter_cnt != 0:
            # TODO: this is a stupid implementation, but ok for now
            xtr, wtr = self._sample_model_with_weights(self.nsamples)
        else:
            xtr, xte, wtr, wte = self.buffer.draw_tr_te_ds(split=split,
                                                           normalize_weight=False)

        if self.model_visited:
            print('Training buffer model:')
            nepochs = self.init_nepochs if iter_cnt == 0 else self.nepoch_visited
            for epoch_id in range(nepochs):
                tr_nll = self.run_epoch(xtr, wtr, self.visited_dist, mode='train', debug=False)
                print(f'[visit_{iter_cnt}] epoch {epoch_id} loss = {tr_nll}')
            print('Finshed training buffer model')

            if (iter_cnt) % 10 == 0 and self.ndim == 2:
                _, xvisited_ind = self.sample_model(1000, model=self.visited_dist)
                self._plot_dist(xvisited_ind, 'dist', 'visited', f'{iter_cnt+1}')

        update_w = self.update_weight(xtr[:, 0, :], wtr)
        # debug
        if iter_cnt < -1:
            values = index_to_xval(self.input_vectors, xtr[:, 1, :].astype('int'))
            fvals = self.fn(values)
            wtr_norm = (wtr - wtr.mean()) / (wtr.std() + 1e-15)
            fref = sorted(fvals)[self.cut_off-1]
            print(f'fred = {fref}')
            cond = np.logical_and(fvals >= 20, fvals <= fref)
            for index, wp, wn, wnorm in zip(xtr[:, 1, :][cond], wtr[cond], update_w[cond], wtr_norm[cond]):
                print(f'index = {index}, weight_before_update = {wp:.4f}, '
                      f'weights_norm = {wnorm:.4f}, '
                      f'weight_after_update = {wn:.4f}')
            pdb.set_trace()

        wtr = update_w

        if self.ndim == 2:
            fpath = self.work_dir / get_full_name(name='dist', prefix='training',
                                                  suffix=f'{iter_cnt}_before')
            samples = index_to_xval(self.input_vectors, xtr[:, 1, :].astype('int'))
            s = self.input_scale
            plt_hist2D(samples, fpath=fpath, range=np.array([[-s, s], [-s, s]]), cmap='binary')

        # per epoch
        tr_loss = 0
        te_loss = 0
        tr_loss_list = []
        print(f'Training model: fref = {self.buffer.zavg}')
        for epoch_id in range(nepochs):
            tr_nll = self.run_epoch(xtr, wtr, self.model, mode='train', debug=False)
            tr_loss_list.append(tr_nll)
            tr_loss += tr_nll / self.nepochs

            # self.writer.add_scalar('loss', tr_nll, epoch_id)
            print(f'[train_{iter_cnt}] epoch {epoch_id} loss = {tr_nll}')

            if split < 1:
                te_nll = self.run_epoch(xte, wte, self.model, mode='test')
                te_loss += te_nll / self.nepochs
                print(f'[test_{iter_cnt}] epoch {epoch_id} loss = {te_nll}')

        print('Finished training model.')
        if split < 1:
            return tr_loss, te_loss

        return tr_loss, tr_loss_list

    def save_checkpoint(self, saved_dict):
        saved_dict.update(dict(buffer=self.buffer,
                               model_state=self.model.state_dict(),
                               opt_state=self.opt.state_dict()))
        if self.model_visited:
            saved_dict.update(dict(visited=self.visited_dist.state_dict()))
        torch.save(saved_dict, self.work_dir / 'checkpoint.tar')

    def load_checkpoint(self, ckpt_path: Union[str, Path]):
        s = time.time()
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint.pop('model_state'))
        if self.model_visited:
            self.visited_dist.load_state_dict(checkpoint.pop('visited'))
            params = list(self.model.parameters()) + list(self.visited_dist.parameters())
        else:
            params = self.model.parameters()

        self.opt.load_state_dict(checkpoint.pop('opt_state'))
        # override optimizer with input parameters
        self.opt = optim.Adam(params, self.lr)
        self.buffer = checkpoint.pop('buffer')
        print(f'Model checkpoint loaded in {time.time() - s:.4f} seconds')
        return checkpoint

    def setup_model(self):
        dim = self.ndim
        nparams_per_dim_mix = 2 if self.fixed_sigma else 3
        self.model: nn.Module = MADE(dim, self.hiddens,
                                     dim * nparams_per_dim_mix * self.nr_mix, seed=self.seed,
                                     natural_ordering=True)
        self.model.to(self.device)
        if self.model_visited:
            nparams = 2 if self.visited_fixed_sigma else 3
            self.visited_dist: nn.Module = MADE(dim, self.hiddens,
                                                dim * nparams * self.visited_nr_mix,
                                                seed=self.seed, natural_ordering=True)
            self.visited_dist.to(self.device)
            params = list(self.model.parameters()) + list(self.visited_dist.parameters())
        else:
            params = list(self.model.parameters())

        self.opt = optim.Adam(params, lr=self.lr, weight_decay=0)
        self.buffer = CacheBuffer(self.mode, self.goal, self.cut_off, self.allow_repeated,
                                  self.problem_type)

        if self.init_buffer_paths:
            for path in self.init_buffer_paths:
                # init_buffer can be either the final buffer of another algorithm (checkpoint.tar)
                # or the init_buffer of another one (init_buffer.pickle)
                if path.endswith('checkpoint.tar'):
                    ref_buffer: CacheBuffer = torch.load(path, map_location=self.device)['buffer']
                else:
                    ref_buffer: CacheBuffer = read_pickle(self.init_buffer_paths)['init_buffer']

                for ind in ref_buffer.db_set:
                    self.buffer.add_samples(ind.item[None, :], ind.item_ind[None, :],
                                            np.array([ind.val]))

            print('Buffer initialized with the provided initialization.')

    def setup_model_state(self):
        # load the model or proceed without loading checkpoints
        if self.load:
            ckpt_dict = self.load_checkpoint(self.work_dir / 'checkpoint.tar')
            tr_losses = ckpt_dict['tr_losses']
            iter_cnt = ckpt_dict['iter_cnt']
            avg_cost = ckpt_dict['avg_cost']
            sim_cnt_list = ckpt_dict['sim_cnt']
            n_sols_in_buffer = ckpt_dict['n_sols_in_buffer']
            sample_cnt_list = ckpt_dict['sample_cnt']
            top_means=dict(top_20=ckpt_dict['top_20'],
                           top_40=ckpt_dict['top_40'],
                           top_60=ckpt_dict['top_60'])
        else:
            # collect samples using the random initial model (probably a bad initialization)
            iter_cnt = 0
            tr_losses, avg_cost, \
            sim_cnt_list, sample_cnt_list, n_sols_in_buffer = [], [], [], [], []
            top_means=dict(top_20=[], top_40=[], top_60=[])
            self.model.eval()
            self.collect_samples(self.n_init_samples, uniform=True)
            write_pickle(self.work_dir / 'init_buffer.pickle', dict(init_buffer=self.buffer))
            # train the init model
            self.model.train()
            self.train(0, self.init_nepochs)

            if self.ndim == 2:
                _, xdata_ind = self.sample_model(1000, model=self.model)
                fpath = self.work_dir / get_full_name(name='dist', prefix='training',
                                                      suffix=f'0_after')
                data_ind = xdata_ind.to(self.cpu).data.numpy().astype('int')
                data = index_to_xval(self.input_vectors, data_ind)
                s = self.input_scale
                _range = np.array([[-s, s], [-s, s]])
                plt_hist2D(data, fpath=fpath, range=_range, cmap='binary')

            saved_data = dict(
                iter_cnt=iter_cnt,
                tr_losses=tr_losses,
                avg_cost=avg_cost,
                sim_cnt=sim_cnt_list,
                n_sols_in_buffer=n_sols_in_buffer,
                sample_cnt=sample_cnt_list,
                **top_means,
            )
            self.save_checkpoint(saved_data)

        return iter_cnt, tr_losses, avg_cost, sim_cnt_list, sample_cnt_list, n_sols_in_buffer, \
               top_means

    def _plot_dist(self, data_indices: torch.Tensor, name, prefix, suffix):
        fpath = self.work_dir / get_full_name(name, prefix, suffix)
        data_ind = data_indices.to(self.cpu).data.numpy().astype('int')
        data = index_to_xval(self.input_vectors, data_ind)
        s = self.input_scale
        _range = np.array([[-s, s], [-s, s]])
        plt_hist2D(data, fpath=fpath, range=_range, cmap='binary')

    def _run_alg(self):

        self.setup_model()
        ret = self.setup_model_state()
        iter_cnt, tr_losses, avg_cost, \
        sim_cnt_list, sample_cnt_list, n_sols_in_buffer, top_means = ret

        while iter_cnt < self.niter:
            print(f'iter {iter_cnt}')
            # ---- update plotting variables
            sim_cnt_list.append(self.buffer.size)
            n_sols_in_buffer.append(self.buffer.n_sols)
            sample_cnt_list.append(self.buffer.tot_freq)
            top_means['top_20'].append(np.mean(self.fvals[:20]))
            top_means['top_40'].append(np.mean(self.fvals[:40]))
            top_means['top_60'].append(np.mean(self.fvals[:60]))
            # top_means['top_20'].append(self.buffer.topn_mean(20))
            # top_means['top_40'].append(self.buffer.topn_mean(40))
            # top_means['top_60'].append(self.buffer.topn_mean(60))

            self.collect_samples(self.nsamples)
            avg_cost.append(self.buffer.mean)

            if iter_cnt == self.niter - 1 and self.full_training:
                tr_loss, tr_loss_list = self.train(iter_cnt + 1, self.nepochs * 40)
            else:
                tr_loss, tr_loss_list = self.train(iter_cnt + 1, self.nepochs)

            tr_losses.append(tr_loss_list)
            if (iter_cnt + 1) % 10 == 0 and self.ndim == 2:
                _, xdata_ind = self.sample_model(1000, model=self.model)
                self._plot_dist(xdata_ind, 'dist', 'training', f'{iter_cnt+1}_after')


            iter_cnt += 1
            saved_data = dict(
                iter_cnt=iter_cnt,
                tr_losses=tr_losses,
                avg_cost=avg_cost,
                sim_cnt=sim_cnt_list,
                n_sols_in_buffer=n_sols_in_buffer,
                sample_cnt=sample_cnt_list,
                **top_means,
            )
            self.save_checkpoint(saved_data)

        plot_learning_with_epochs(fpath=self.work_dir / 'learning_curve.png', training=tr_losses)
        plot_cost(avg_cost, fpath=self.work_dir / 'cost.png')
        plot_x_y(sample_cnt_list, n_sols_in_buffer,
                 #annotate=sim_cnt_list,marker='s', fillstyle='none'
                 fpath=self.work_dir / 'n_sols.png',
                 xlabel='n_freq', ylabel=f'n_sols')

    def _sample_model_for_eval(self, nsamples) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, sample_ids = self.sample_model(nsamples, model=self.model)
        sample_ids_arr = sample_ids.long().to(torch.device('cpu')).numpy()
        xsample_arr = index_to_xval(self.input_vectors, sample_ids_arr)
        fval = self.fn(xsample_arr)
        return xsample_arr, sample_ids_arr, fval

    def report_variation(self, nsamples):
        xsample, _, fval = self._sample_model_for_eval(nsamples)

        total_var = compute_emprical_variation(xsample)
        if self.mode == 'le':
            pos_samples = xsample[fval <= self.goal]
            pos_var = compute_emprical_variation(pos_samples)
        else:
            pos_samples = xsample[fval >= self.goal]
            pos_var = compute_emprical_variation(pos_samples)

        print(f'total solution variation / dim = {total_var:.6f}')
        if np.isnan(pos_var):
            raise ValueError('did not find any satisfying solutions!')
        print(f'pos solution variation / dim = {pos_var:.6f}')

    def report_accuracy(self, ntimes, nsamples):
        accuracy_list, times, div_list = [], [], []

        if self.ndim == 2:
            xsamples, _, _ = self._sample_model_for_eval(nsamples)
            s = self.input_scale
            _range = np.array([[-s, s], [-s, s]])
            plt_hist2D(xsamples, range=_range,
                       fpath=self.work_dir / get_full_name('trained_policy'), cmap='binary')

        for iter_id in range(ntimes):
            s = time.time()
            xsample, sample_ids, fval = self._sample_model_for_eval(nsamples)
            if self.mode == 'le':
                acc = (fval <= self.goal).sum(-1) / nsamples
                pos_samples = xsample[fval <= self.goal]
            else:
                acc = (fval >= self.goal).sum(-1) / nsamples
                pos_samples = xsample[fval >= self.goal]

            if len(pos_samples) >= self.ndim:
                div = get_diversity_fom(self.ndim, pos_samples)
                div_list.append(div)

            times.append(time.time() - s)
            accuracy_list.append(acc)

        acc_mean = 100 * float(np.mean(accuracy_list))
        acc_std = 100 * float(np.std(accuracy_list))
        acc_div = float(np.mean(div_list)) if div_list else 0
        print(f'gen_time / sample = {1e3 * np.mean(times).astype("float") / nsamples:.3f} ms')
        print(f'accuracy_avg = {acc_mean:.6f}, accuracy_std = {acc_std:.6f}, '
              f'solution diversity = {acc_div:.6f}')

        return acc_mean, acc_std, acc_div

    def load_and_sample(self, nsamples, only_positive=False) -> np.ndarray:
        """sets up the model (i.e. initializes the weights .etc) and generates samples"""
        self.setup_model()
        self.setup_model_state()
        xsample, _, fval = self._sample_model_for_eval(nsamples)
        if not only_positive:
            return xsample

        n_remaining = nsamples
        ans_list = []
        while n_remaining > 0:
            if self.mode == 'le':
                pos_samples = xsample[fval <= self.goal]
            else:
                pos_samples = xsample[fval >= self.goal]
            ans_list.append(pos_samples)
            n_remaining -= len(pos_samples)
            print(f"sampled {len(pos_samples)} pos_solutions, n_remaining: {n_remaining}")
            if n_remaining > 0:
                xsample, _, fval = self._sample_model_for_eval(n_remaining)
        ans = np.concatenate(ans_list, axis=0)

        return ans

    def plot_model_sol_pca(self, nsamples=100):
        xsample, sample_ids, fval = self._sample_model_for_eval(nsamples)
        if self.mode == 'le':
            pos_samples = xsample[fval <= self.goal]
        else:
            pos_samples = xsample[fval >= self.goal]
        plot_pca_2d(pos_samples, fpath=self.work_dir / f'pca_sol.png')

    def report_entropy(self, ntimes, nsamples):
        ent_list = []
        for iter_cnt in range(ntimes):
            samples, _ = self.sample_model(nsamples, self.model)
            probs = self.get_probs(samples, self.model)
            ent_list.append(-probs.prod(-1).log().mean().item())

        ent = float(np.mean(ent_list) / self.ndim)
        print(f'entropy/dim: {ent}')
        return ent

    def check_solutions(self, ntimes=1, nsamples=1000):
        print('-------- REPORT --------')
        # self.check_random_solutions(ntimes, nsamples)
        acc, std, divesity = self.report_accuracy(ntimes, nsamples)
        ent = self.report_entropy(ntimes, nsamples)

        saved_data = dict(acc=acc, std=std, divesity=divesity, ent=ent)
        write_yaml(self.work_dir / 'performance.yaml', saved_data)
        # self.report_variation(nsamples)
        # self.plot_model_sol_pca()

    def check_random_solutions(self, ntimes, nsamples):
        rnd_specs = deepcopy(self.specs)
        rnd_params = rnd_specs['params']
        rnd_params['work_dir'] = self.work_dir
        random_policy = Random(spec_dict=rnd_specs)
        random_policy.check_solutions(ntimes, nsamples)

    def main(self) -> None:
        # self.check_random_solutions(ntimes=10, nsamples=10)
        # input('Press Enter To continue:')
        self.set_seed(self.seed)
        self._run_alg()
        self.check_solutions(ntimes=10, nsamples=100)
