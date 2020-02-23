from typing import Optional, Mapping, Any, Tuple

import pickle
import random
import time
from pathlib import Path
import numpy as np
import math
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal

from utils.file import read_yaml, write_yaml, get_full_name, write_pickle

from ...benchmarks.functions import registered_functions
from ...benchmarks.fom import get_diversity_fom, compute_emprical_variation
from ...data.buffer import CacheBuffer
from ...viz.plot import plot_pca_2d, plt_hist2D, plot_cost, plot_fn2d, show_solution_region, plot_x_y
from ...data.vector import index_to_xval
from ..base import AlgBase
from utils.data.database import Database

from sortedcontainers import SortedList

class CEM:
    """
    The vanilla implementation of Cross Entropy method with gaussian distributions
    """
    def __init__(self, input_vectors, average_coeff, dist_type='gauss', **kwargs):
        """

        Parameters
        ----------
        input_vectors: List[np.ndarray]
            The possible values for each input dimension
        dist_type: str
            The Kernel type:
                'guass' for single gaussian
                'KDE' for kernel density estimation
        average_coeff: float(not supported in KDE)
            a number between 0,1: params = (1-alpha) * old_params + alpha * new_params
        **kwargs: Dict[str, Any]
            gauss_sigma: if dist_type == 'gauss' will be the constant sigma used without updating
            sigma
        """
        self.input_indices = [range(len(x)) for x in input_vectors]
        dim = len(input_vectors)
        self.params_min = np.array([0] * dim)
        self.params_max = np.array([len(x) - 1 for x in self.input_indices])

        if dist_type not in ['gauss', 'kde']:
            raise ValueError(f'{dist_type} is not a valid kernel type for CEM++: guass | kde')

        self.type = dist_type
        self.gauss_sigma = kwargs.get('gauss_sigma', None)

        self.average_coeff = average_coeff
        self.params = {}

    def fit(self, data):
        # data has to be in units of indices
        if len(data.shape) > 2:
            raise ValueError('Data should be in shape of Nxd (N samples with d dimensions)')
        ndata, ndim = data.shape
        alpha = self.average_coeff
        if self.type == 'gauss':
            # TODO: investigate why this is more stable than using new_mu in computing new_var
            old_mu = self.params.get('mu', 0)
            if self.gauss_sigma is None:
                old_var = self.params.get('var', 0)
                new_var = 1 / ndata * (data - old_mu).T @ (data - old_mu)
                self.params['var'] = old_var  * (1 - alpha) + new_var * alpha
            else:
                self.params['var'] = self.gauss_sigma * np.eye(ndim)
            new_mu = np.mean(data, axis=0)
            self.params['mu'] = old_mu * (1 - alpha) + new_mu * alpha
        elif self.type == 'kde':
            self.params['kde'] = gaussian_kde(np.transpose(data))


    def _draw_uniform_samples(self, n):
        dim = len(self.params_min)
        cols = []
        for i in range(dim):
            cols.append(np.random.randint(0, self.params_max[i], n, dtype='int'))
        samples = np.stack(cols, axis=-1)
        return samples

    def sample(self, n, uniform: bool = False):
        if not uniform:
            if not self.params:
                raise ValueError('params are not set yet, did you forget to call fit?')

            if self.type == 'gauss':
                samples = multivariate_normal.rvs(self.params['mu'],
                                                  self.params['var'], n)
            elif self.type == 'kde':
                samples = self.params['kde'].resample(n)
                samples = samples.transpose()

            lo = np.zeros(samples.shape) + self.params_min
            hi = np.zeros(samples.shape) + self.params_max
            samples = np.clip(samples, lo, hi)
            samples = np.floor(samples).astype('int')
        else:
            # uniform sampling
            samples = self._draw_uniform_samples(n)

        if len(samples.shape) == 1:
            samples = samples[None, ...]

        return samples

    def evaluate_pdf(self, samples):
        if self.type == 'gauss':
            return multivariate_normal.pdf(samples, self.params['mu'], self.params['var'],
                                           allow_singular=True)
        else:
            return self.params['kde'].pdf(samples.T).T

    def entropy(self, nsamples=None):
        samples = self.sample(nsamples)
        pdf = self.evaluate_pdf(samples)
        return (-np.log(pdf)).mean(-1)

class CEMSearch(AlgBase):

    # noinspection PyUnusedLocal
    def __init__(self, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        spec_file: str
        spec_dict: Dict[str, Any]
        some non-obvious fields
            elite_criteria: str
                'optim': from sorted x1, ..., xn choose p-quantile
                'csp': constraint satisfaction is enough, from x1, ..., xn
                choose p-quantile if it is worst than the constraint else choose all which are
                better than the constraint
            allow_repeated: bool
                True to allow repeated samples to be added to the buffer, else all samples in buffer
                will have equal likelihood when drawn from it.
            on_policy: bool
                True to allow on_policy sample usage, meaning that we won't use samples from
                previous policies to train the current policy (samples are not drawn from
                CacheBuffer)
        load: bool
        kwargs: Dict[str, Any]
        """
        super().__init__(*args, **kwargs)

        specs = self.specs
        params = specs['params']

        self.seed = params['seed']
        self.nsamples = params['nsamples']
        self.n_init_samples = params['n_init_samples']
        self.niter = params['niter']
        self.cut_off = params['cut_off']
        # self.ndim = params['ndim']
        # self.input_scale = params['input_scale']

        self.allow_repeated = params.get('allow_repeated', False)
        self.elite_criteria = params.get('elite_criteria', 'optim')
        self.on_policy = params.get('on_policy', False)

        if self.elite_criteria not in ['csp', 'optim']:
            raise ValueError('invalid elite criteria: optim | csp')

        # allow repeated does not make sense when sampling is on-policy (on-policy: T -> repeat: T)
        self.allow_repeated = self.on_policy or self.allow_repeated

        # hacky version of passing input vectors around
        self.input_vectors_norm = [np.linspace(start=-1.0, stop=1.0, dtype='float32',
                                               num=100) for _ in range(self.ndim)]
        self.input_vectors = [self.input_scale * vec for vec in self.input_vectors_norm]

        self.cem = CEM(self.input_vectors, dist_type=params['base_fn'],
                       average_coeff=params.get('average_coeff', 1),
                       gauss_sigma=params.get('gauss_sigma', None))

        self.db =
        self.buffer = CacheBuffer(self.mode, self.goal, self.cut_off,
                                  with_frequencies=self.allow_repeated)

        self.buffer_temp = {}
        self.fvals = SortedList()

    @classmethod
    def set_seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)

    def save_checkpoint(self, saved_dict):
        saved_dict.update(dict(buffer=self.buffer, cem=self.cem))
        write_pickle(self.work_dir / 'checkpoint.pickle', saved_dict)

    def load_checkpoint(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.cem = checkpoint.pop('cem')
        self.buffer = checkpoint.pop('buffer')
        return checkpoint

    def collect_samples(self, n, uniform: bool = False):
        n_collected = 0
        new_samples = []
        new_sample_fvals = []
        while n_collected < n:
            samples = self.sample_model(n - n_collected, uniform=uniform)
            xsamples = index_to_xval(self.input_vectors, samples)
            for xsample, sample in zip(xsamples, samples):
                in_buffer = xsample in self.buffer
                # look up buffer, see if already evaluated
                fval = self.buffer[xsample] if in_buffer else self.fn(xsample)
                self.fvals.add(fval)
                if fval < self.goal:
                    self.buffer_temp[xsample.tostring()] = None
                if not self.on_policy:
                    self.buffer.add_samples(xsample[None, ...], sample[None, ...], fval[None, ...])

                if self.allow_repeated or not in_buffer:
                    new_samples.append(sample)
                    new_sample_fvals.append(fval)
                    n_collected += 1

        return np.array(new_samples), np.array(new_sample_fvals)

    def get_top_samples(self, iter_cnt, samples, sample_fvals):

        if self.on_policy:
            nsamples = len(samples)
            sample_ids = range(nsamples)
            sorted_sample_ids = sorted(sample_ids, key=lambda i: sample_fvals[i],
                                       reverse=self.mode == 'ge')
            sorted_samples = samples[sorted_sample_ids]

            # find the last index which satisfies the constraint
            cond = sample_fvals <= self.goal if self.mode == 'le' else sample_fvals >= self.goal
            top_index = cond.sum(-1).astype('int')

        else:
            data, _, weights, _ =  self.buffer.draw_tr_te_ds(split=1, normalize_weight=False)
            samples = data[:, 1].astype('int')
            nsamples = len(samples)
            weights_iter = iter(weights)
            sorted_samples = np.stack(sorted(samples, key=lambda x: next(weights_iter),
                                             reverse=True), axis=0,)
            top_index = (weights == 1).sum(-1).astype('int')

        if self.elite_criteria == 'optim':
            top_index = self.cut_off
        elif self.elite_criteria == 'csp':
            top_index = max(top_index, min(self.cut_off, nsamples))

        top_samples = sorted_samples[:top_index]

        return top_samples

    def setup_state(self):
        if self.load:
            ckpt_dict = self.load_checkpoint(self.work_dir / 'checkpoint.pickle')
            iter_cnt = ckpt_dict['iter_cnt']
            avg_cost = ckpt_dict['avg_cost']
            sim_cnt_list = ckpt_dict['sim_cnt']
            n_sols_in_buffer = ckpt_dict['n_sols_in_buffer']
            sample_cnt_list = ckpt_dict['sample_cnt']
            top_means=dict(top_20=ckpt_dict['top_20'],
                           top_40=ckpt_dict['top_40'],
                           top_60=ckpt_dict['top_60'])
        else:
            iter_cnt = 0
            avg_cost,  sim_cnt_list, sample_cnt_list, n_sols_in_buffer = [], [], [], []
            top_means=dict(top_20=[], top_40=[], top_60=[])
            samples, sample_fvals = self.collect_samples(self.n_init_samples, uniform=True)
            top_samples = self.get_top_samples(0, samples, sample_fvals)
            self.cem.fit(top_samples)

        return iter_cnt, avg_cost,  sim_cnt_list, sample_cnt_list, n_sols_in_buffer, top_means

    def sample_model(self, nsamples: int, uniform=False):
        sample_ids = self.cem.sample(nsamples, uniform=uniform)
        return sample_ids

    def _sample_model_for_eval(self, nsamples) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_ids = self.sample_model(nsamples)
        samples = index_to_xval(self.input_vectors, sample_ids)
        fval = self.fn(samples)
        return samples, sample_ids, fval

    def load_and_sample(self, nsamples, only_positive=False) -> np.ndarray:
        """sets up the model and generates samples"""
        self.setup_state()
        samples, _, fvals = self._sample_model_for_eval(nsamples)

        if not only_positive:
            return samples

        n_remaining = nsamples
        ans_list = []
        while n_remaining > 0:
            if self.mode == 'le':
                pos_samples = samples[fvals <= self.goal]
            else:
                pos_samples = samples[fvals >= self.goal]
            ans_list.append(pos_samples)
            n_remaining -= len(pos_samples)
            if n_remaining > 0:
                xsample, _, fval = self._sample_model_for_eval(n_remaining)
        ans = np.concatenate(ans_list, axis=0)

        return ans

    def report_accuracy(self, ntimes, nsamples):
        accuracy_list, times, div_list = [], [], []

        for iter_id in range(ntimes):
            s = time.time()
            sample_ids = self.sample_model(nsamples)
            xsample = index_to_xval(self.input_vectors, sample_ids)
            fval: np.ndarray = self.fn(xsample)
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

    def report_variation(self, nsamples):
        sample_ids = self.sample_model(nsamples)
        xsample = index_to_xval(self.input_vectors, sample_ids)
        fval = self.fn(xsample)

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

    def report_entropy(self, ntimes, nsamples):
        ent_list = []
        for iter_cnt in range(ntimes):
            # _, sample_ids, _ = self._sample_model_for_eval(nsamples)
            ent = self.cem.entropy(nsamples)
            ent_list.append(ent)

        ent = float(np.mean(ent_list) / self.ndim)
        print(f'entropy/dim: {ent}')
        return ent

    def check_solutions(self, ntimes=1, nsamples=1000):
        print('-------- REPORT --------')
        acc, std, divesity = self.report_accuracy(ntimes, nsamples)
        ent = self.report_entropy(ntimes, nsamples)

        saved_data = dict(acc=acc, std=std, divesity=divesity, ent=ent)
        write_yaml(self.work_dir / 'performance.yaml', saved_data)
        # self.report_variation(nsamples)
        # try:
        #     self.plot_solution_space(nsamples=1000)
        # except ValueError:
        #     print('Accuracy is not enough to plot solutions space (number of satisfying solutions '
        #           'is small)')

    def plot_solution_space(self, nsamples=100):
        sample_ids = self.sample_model(nsamples)
        xsample = index_to_xval(self.input_vectors, sample_ids)
        fval = self.fn(xsample)
        if self.mode == 'le':
            pos_samples = xsample[fval <= self.goal]
        else:
            pos_samples = xsample[fval >= self.goal]
        plot_pca_2d(pos_samples, fpath=self.work_dir / f'pca_sol.png')

    def _run_alg(self):
        ret = self.setup_state()
        iter_cnt, avg_cost,  sim_cnt_list, sample_cnt_list, n_sols_in_buffer, top_means = ret
        while iter_cnt < self.niter:
            print(f'iter {iter_cnt}')
            # ---- update plotting variables
            # sim_cnt_list.append(self.buffer.size)
            # n_sols_in_buffer.append(self.buffer.n_sols)
            # sample_cnt_list.append(self.buffer.tot_freq)
            # top_means['top_20'].append(self.buffer.topn_mean(20))
            # top_means['top_40'].append(self.buffer.topn_mean(40))
            # top_means['top_60'].append(self.buffer.topn_mean(60))
            sim_cnt_list.append((iter_cnt + 1) * self.nsamples + self.n_init_samples)
            n_sols_in_buffer.append(len(self.buffer_temp))
            sample_cnt_list.append((iter_cnt + 1) * self.nsamples + self.n_init_samples)
            top_means['top_20'].append(np.mean(self.fvals[:20]))
            top_means['top_40'].append(np.mean(self.fvals[:40]))
            top_means['top_60'].append(np.mean(self.fvals[:60]))

            samples, sample_fvals = self.collect_samples(self.nsamples)
            avg_cost.append(sample_fvals.mean() if self.on_policy else self.buffer.mean)
            top_samples = self.get_top_samples(iter_cnt+1, samples, sample_fvals)
            self.cem.fit(top_samples)

            iter_cnt += 1

            saved_data = dict(
                iter_cnt=iter_cnt,
                avg_cost=avg_cost,
                sim_cnt=sim_cnt_list,
                n_sols_in_buffer=n_sols_in_buffer,
                sample_cnt=sample_cnt_list,
                **top_means,
            )
            self.save_checkpoint(saved_data)

        plot_cost(avg_cost, fpath=self.work_dir / 'cost.png')
        plot_x_y(sample_cnt_list, n_sols_in_buffer,
                 #annotate=sim_cnt_list,marker='s', fillstyle='none'
                 fpath=self.work_dir / 'n_sols.png',
                 xlabel='n_freq', ylabel=f'n_sols')

    def main(self):
        self.set_seed(self.seed)
        self._run_alg()
        self.check_solutions(ntimes=10, nsamples=100)
