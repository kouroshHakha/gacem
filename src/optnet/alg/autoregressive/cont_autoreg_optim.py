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

from utils.file import read_yaml, write_yaml, get_full_name
from utils.loggingBase import LoggingBase

from ...benchmarks.functions import registered_functions
from ...benchmarks.fom import (
    compute_emprical_variation, get_diversity_fom
)
from ...torch.dist import cdf_logistic, cdf_normal, cdf_uniform
from ...data.buffer import CacheBuffer
from ...models.made import MADE
from ..random.random import Random
from ...data.vector import index_to_xval

from ...viz.plot import (
    plot_pca_2d, plt_hist2D, plot_cost, plot_learning_with_epochs
)


class AutoRegSearch(LoggingBase):

    # noinspection PyUnusedLocal
    def __init__(self, spec_file: str = '', spec_dict: Optional[Mapping[str, Any]] = None,
                 load: bool = False, **kwargs) -> None:
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
            unique_name = time.strftime('%Y%m%d%H%M%S')
            suffix = params.get('suffix', '')
            prefix = params.get('prefix', '')
            unique_name = get_full_name(unique_name, prefix, suffix)
            self.work_dir = Path(specs['root_dir']) / f'{unique_name}'
            write_yaml(self.work_dir / 'params.yaml', specs, mkdir=True)

        self.load = load
        self.seed = params.get('seed', 10)
        self.set_seed(self.seed)
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

    @classmethod
    def set_seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(seed)

    def get_probs(self, xin: torch.Tensor, debug=False):
        """Given an input tensor (N, dim) computes the probabilities across the cardinality space
        of each dimension. prob.shape = (N, dim, K) where K is number of possible values for each
        dimension. Assume that xin is normalized to [-1,1], and delta is given."""
        delta = self.delta
        xin = xin.to(self.device)
        xhat = self.model(xin)
        dim = self.ndim
        coeffs = torch.stack([xhat[..., i::dim*3] for i in range(dim)], dim=-2)
        # Note: coeffs was previously interpreted as log_coeffs
        # interpreting outputs if NN as log is dangerous, can result in Nan's.
        # solution: here they should be positive and should add up to 1, sounds familiar? softmax!
        coeffs_norm = coeffs.softmax(dim=-1)

        eps = torch.tensor(1e-15)
        xb = xin[..., None] + torch.zeros(coeffs.shape, device=self.device)

        if self.base_fn in ['logistic', 'normal']:
            means = torch.stack([xhat[..., i+dim::dim*3] for i in range(dim)], dim=-2)
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

        # bar_grad = torch.autograd.grad(probs.flatten()[0], self.model.net[0].bias,
        #                                retain_graph=True)[0]
        # if torch.isnan(bar_grad[0]):
        #     print(bar_grad)
        #     print(cdfs)
        #     pdb.set_trace()

        if debug:
            pdb.set_trace()

        return probs

    def get_nll(self, xin: torch.Tensor, weights=None, debug=False):
        """Given an input tensor computes the average negative likelihood of observing the inputs"""
        probs = self.get_probs(xin)
        eps_tens = torch.tensor(1e-15)
        logp_vec = (probs + eps_tens).log10().sum(-1)

        if weights is None:
            min_obj = -logp_vec.mean(-1)
        else:
            pos_ind = (weights > 0).float()
            neg_ind = torch.tensor(1) - pos_ind

            # obj_term  = - self.buffer.size * (weights * prob_x).data
            # ent_term = self.buffer.size * (self.beta * (torch.tensor(1) + logp_vec)).data
            obj_term  = - weights.data
            ent_term = (self.beta * (torch.tensor(1) + logp_vec)).data

            # coeff = (ent_term + obj_term) * pos_ind
            main_obj = obj_term * logp_vec
            ent_obj = ent_term * logp_vec

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
                foo = torch.autograd.grad(min_obj, self.model.net[0].weight, retain_graph=True)
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

    def sample_model(self, nsamples: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.model.eval()

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
                    probs = self.get_probs(xin)
                    xi_ind = self.sample_probs(probs, i)  # ith x index
                    xsample[0, i] = torch.tensor(self.input_vectors_norm[i][xi_ind])
                    xsample_ind[0, i] = xi_ind
                samples.append(xsample.squeeze())
                samples_ind.append(xsample_ind.squeeze())

            samples = torch.stack(samples, dim=0)
            samples_ind = torch.stack(samples_ind, dim=0)
            return samples, samples_ind

    def run_epoch(self, data: np.ndarray, weights: np.ndarray, mode='train',
                  debug=False):
        self.model.train() if mode == 'train' else self.model.eval()

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
            loss = self.get_nll(xin, weights=wb_tens, debug=debug)
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
        while n_collected < n_samples:
            if uniform:
                _, xnew_id_np = Random.sample_data(self.ndim, self.input_vectors_norm, 1)
                xnew_id_np = xnew_id_np.astype('int')
            else:
                _, xnew_ind = self.sample_model(1)
                xnew_id_np = xnew_ind.to(self.cpu).data.numpy().astype('int')

            # simulate and compute the adjustment weights
            org_sample_list = [vecs[index][pos] for index, pos in enumerate(xnew_id_np[0, :])]
            xsample = np.array(org_sample_list, dtype='float32')
            fval = self.fn(xsample[None, :])

            norm_sample_list = [norm_vecs[index][pos] for index, pos in enumerate(xnew_id_np[0, :])]
            norm_sample = np.array(norm_sample_list, dtype='float32')

            if norm_sample not in self.buffer:
                self.buffer.add_samples(norm_sample[None, :], xnew_id_np, fval)
                new_samples.append(xsample)
                n_collected += 1
            else:
                print(f'item {norm_sample} already exists!')

        return new_samples

    def train(self, iter_cnt: int, nepochs: int, split=1.0):
        # treat the sampled data as a static data set and take some gradient steps on it
        xtr, xte, wtr, wte = self.buffer.draw_tr_te_ds(split=split)
        if self.ndim == 2:
            fpath = self.work_dir / get_full_name(name='dist', prefix='training',
                                                  suffix=f'{iter_cnt}_before')
            samples = index_to_xval(self.input_vectors, xtr[:, 1, :].astype('int'))
            s = self.input_scale
            _range = np.array([[-s, s], [-s, s]])
            plt_hist2D(samples, fpath=fpath, range=_range, cmap='binary')

        # per epoch
        print('-'*50)
        tr_loss = 0
        te_loss = 0
        tr_loss_list = []
        for epoch_id in range(nepochs):
            tr_nll = self.run_epoch(xtr, wtr, mode='train')
            tr_loss_list.append(tr_nll)
            tr_loss += tr_nll / self.nepochs

            # self.writer.add_scalar('loss', tr_nll, epoch_id)
            print(f'[train_{iter_cnt}] epoch {epoch_id} loss = {tr_nll}')

            if split < 1:
                te_nll = self.run_epoch(xte, wte, mode='test')
                te_loss += te_nll / self.nepochs
                print(f'[test_{iter_cnt}] epoch {epoch_id} loss = {te_nll}')

        if split < 1:
            return tr_loss, te_loss

        return tr_loss, tr_loss_list

    def save_checkpoint(self, iter_cnt, tr_losses, avg_cost):
        dict_to_save = dict(
            iter_cnt=iter_cnt,
            tr_losses=tr_losses,
            avg_cost=avg_cost,
            buffer=self.buffer,
            model_state=self.model.state_dict(),
            opt_state=self.opt.state_dict(),
        )
        torch.save(dict_to_save, self.work_dir / 'checkpoint.tar')

    def load_checkpoint(self, ckpt_path: Union[str, Path]):
        s = time.time()
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.opt.load_state_dict(checkpoint['opt_state'])
        # override optimizer with input parameters
        self.opt = optim.Adam(self.model.parameters(), self.lr)
        self.buffer = checkpoint['buffer']
        items = (checkpoint['iter_cnt'], checkpoint['tr_losses'], checkpoint['avg_cost'])
        print(f'Model checkpoint loaded in {time.time() - s:.4f} seconds')
        return items

    def setup_model(self):
        dim = self.ndim
        self.model: nn.Module = MADE(dim, self.hiddens, dim * 3 * self.nr_mix, seed=self.seed,
                                     natural_ordering=True)
        self.model.to(self.device)

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

        self.buffer = CacheBuffer(self.mode, self.goal, self.cut_off)

    def setup_model_state(self):
        # load the model or proceed without loading checkpoints
        if self.load:
            items = self.load_checkpoint(self.work_dir / 'checkpoint.tar')
        else:
            # collect samples using the random initial model (probably a bad initialization)
            self.model.eval()
            self.collect_samples(self.n_init_samples, uniform=True)
            # train the init model
            self.model.train()
            self.train(0, self.n_init_samples)

            if self.ndim == 2:
                _, xdata_ind = self.sample_model(1000)
                fpath = self.work_dir / get_full_name(name='dist', prefix='training',
                                                      suffix=f'0_after')
                data_ind = xdata_ind.to(self.cpu).data.numpy().astype('int')
                data = index_to_xval(self.input_vectors, data_ind)
                s = self.input_scale
                _range = np.array([[-s, s], [-s, s]])
                plt_hist2D(data, fpath=fpath, range=_range, cmap='binary')

            items = (0, [], [])
            self.save_checkpoint(*items)
        return items

    def _run_alg(self):

        self.setup_model()
        iter_cnt, tr_losses, avg_cost = self.setup_model_state()

        while iter_cnt < self.niter:
            self.collect_samples(self.nsamples)
            avg_cost.append(self.buffer.mean)

            if iter_cnt == self.niter - 1 and self.full_training:
                tr_loss, tr_loss_list = self.train(iter_cnt + 1, self.nepochs * 10)
            else:
                tr_loss, tr_loss_list = self.train(iter_cnt + 1, self.nepochs)

            tr_losses.append(tr_loss_list)
            if (iter_cnt + 1) % 10 == 0 and self.ndim == 2:
                _, xdata_ind = self.sample_model(1000)
                fpath = self.work_dir / get_full_name(name='dist', prefix='training',
                                                      suffix=f'{iter_cnt+1}_after')
                data_ind = xdata_ind.to(self.cpu).data.numpy().astype('int')
                data = index_to_xval(self.input_vectors, data_ind)
                s = self.input_scale
                _range = np.array([[-s, s], [-s, s]])
                plt_hist2D(data, fpath=fpath, range=_range, cmap='binary')

            iter_cnt += 1
            self.save_checkpoint(iter_cnt, tr_losses, avg_cost)

        plot_learning_with_epochs(fpath=self.work_dir / 'learning_curve.png', training=tr_losses)
        plot_cost(avg_cost, fpath=self.work_dir / 'cost.png')

    def _sample_model_for_eval(self, nsamples) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vector_mat = np.stack(self.input_vectors, axis=0)
        _, sample_ids = self.sample_model(nsamples)
        sample_ids_arr = sample_ids.long().to(torch.device('cpu')).numpy()
        xsample_arr = vector_mat[np.arange(self.ndim), sample_ids_arr]
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

        print(f'gen_time / sample = {1e3 * np.mean(times).astype("float") / nsamples:.3f} ms')
        print(f'accuracy_avg = {100 * np.mean(accuracy_list).astype("float"):.6f}, '
              f'accuracy_std = {100 * np.std(accuracy_list).astype("float"):.6f}, '
              f'solution diversity = {np.mean(div_list).astype("float"):.6f}')

    def load_and_sample_ids(self, nsamples) -> np.ndarray:
        """sets up the model (i.e. initializes the weights .etc) and generates samples"""
        self.setup_model()
        self.setup_model_state()
        _, sample_ids = self.sample_model(nsamples)
        return sample_ids.to(self.cpu).data.numpy().astype('int')

    def plot_model_sol_pca(self, nsamples=100):
        xsample, sample_ids, fval = self._sample_model_for_eval(nsamples)
        if self.mode == 'le':
            pos_samples = xsample[fval <= self.goal]
        else:
            pos_samples = xsample[fval >= self.goal]
        plot_pca_2d(pos_samples, fpath=self.work_dir / f'pca_sol.png')

    def check_solutions(self, ntimes=1, nsamples=1000):
        print('-------- REPORT --------')
        self.check_random_solutions(ntimes, nsamples)
        self.report_accuracy(ntimes, nsamples)
        self.report_variation(nsamples)
        self.plot_model_sol_pca()

    def check_random_solutions(self, ntimes, nsamples):
        rnd_specs = deepcopy(self.specs)
        rnd_params = rnd_specs['params']
        rnd_params['work_dir'] = self.work_dir
        random_policy = Random(spec_dict=rnd_specs)
        random_policy.check_solutions(ntimes, nsamples)

    def main(self) -> None:
        self.check_random_solutions(ntimes=10, nsamples=10)
        input('Press Enter To continue:')
        self._run_alg()
        self.check_solutions(ntimes=3, nsamples=100)
