from typing import Optional, Mapping, Any

import pickle
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal

from utils.file import read_yaml, write_yaml, get_full_name
from utils.loggingBase import LoggingBase

from ...benchmarks.functions import registered_functions
from ...benchmarks.fom import get_diversity_fom, compute_emprical_variation
from ...data.buffer import CacheBuffer
from ...viz.plot import plot_pca_2d, plt_hist2D, plot_cost
from ...data.vector import index_to_xval

class CEM:
    """
    The vanilla implementation of Cross Entropy method with gaussian distribution
    """
    def __init__(self, input_vectors, dist_type='gaussian'):
        self.input_indices = [range(len(x)) for x in input_vectors]
        dim = len(input_vectors)
        self.params_min = np.array([0] * dim)
        self.params_max = np.array([len(x) - 1 for x in self.input_indices])

        self.type = dist_type
        self.params = {}

    def fit(self, data):
        # data has to be in units of indices
        ndata = data.shape[0]
        if self.type == 'gaussian':
            mu = np.mean(data, axis=0)
            self.params['var'] = 1 / ndata * (data - mu).T @ (data - mu)
            # self.params['var'] =  np.var(data, axis=0)
            self.params['mu'] = mu
        elif self.type == 'kde':
            self.params['kde'] = gaussian_kde(np.transpose(data))

    def _draw_uniform_samples(self, n):
        dim = len(self.params_min)
        cols = []
        for i in range(dim):
            cols.append(np.random.randint(0, self.params_max[i], n, dtype='int'))
        samples = np.stack(cols, axis=-1)
        return samples

    def sample(self, n):
        if self.params:
            if self.type == 'gaussian':
                samples = multivariate_normal.rvs(self.params['mu'],
                                                  self.params['var'], n)
            else:
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


class CEMSearch(LoggingBase):

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
        self.set_seed(params['seed'])

        self.ndim = params['ndim']
        self.nsamples = params['nsamples']
        self.n_init_samples = params['n_init_samples']
        self.niter = params['niter']
        self.cut_off = params['cut_off']
        self.input_scale = params['input_scale']
        # goal has to always be positive if not we'll change mode and negate self.goal
        self.goal = params['goal_value']
        self.mode = params['mode']

        eval_fn = params['fn']
        try:
            fn = registered_functions[eval_fn]
            self.fn = fn
        except KeyError:
            raise ValueError(f'{eval_fn} is not a valid benchmark function')

        if self.goal < 0:
            self.mode = 'le' if self.mode == 'ge' else 'ge'
            self.fn = lambda x: -fn(x)

        # hacky version of passing input vectors around
        self.input_vectors_norm = [np.linspace(start=-1.0, stop=1.0, dtype='float32',
                                               num=100) for _ in range(self.ndim)]
        self.input_vectors = [self.input_scale * vec for vec in self.input_vectors_norm]

        self.cem = CEM(self.input_vectors, dist_type=params['base_fn'])
        self.buffer = CacheBuffer(self.mode, self.goal, self.cut_off)

    @classmethod
    def set_seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)

    def score(self, fvals):
        if self.mode == 'le':
            err = (fvals - self.goal) / self.goal
        else:
            err = (self.goal - fvals) / self.goal
        scores = np.maximum(err, 0)
        return scores

    def save_checkpoint(self, iter_cnt, avg_cost):
        dict_to_save = dict(
            iter_cnt=iter_cnt,
            avg_cost=avg_cost,
            buffer=self.buffer,
            cem=self.cem,
        )

        with open(self.work_dir / 'checkpoint.pickle', 'wb') as f:
            pickle.dump(dict_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.cem = checkpoint['cem']
        self.buffer = checkpoint['buffer']
        items = (checkpoint['iter_cnt'], checkpoint['avg_cost'])
        return items

    def collect_samples(self, n):
        n_collected = 0
        new_samples = []
        while n_collected < n:
            samples = self.sample_model(n - n_collected)
            xsamples = index_to_xval(self.input_vectors, samples)
            for xsample, sample in zip(xsamples, samples):
                if xsample not in self.buffer:
                    fval = self.fn(xsample)
                    self.buffer.add_samples(xsample[None, ...], sample[None, ...], fval[None, ...])
                    new_samples.append(xsample)
                    n_collected += 1
                else:
                    print(f'item {xsample} already exists!')

        return np.array(new_samples)

    def get_top_samples(self, iter_cnt):
        samples = np.array([x.item_ind for x in self.buffer.db_set])
        fvals = np.array([x.val for x in self.buffer.db_set])

        scores_arr = self.score(fvals)

        sample_ids = range(len(scores_arr))
        sorted_sample_ids = sorted(sample_ids, key=lambda i: scores_arr[i])

        sorted_samples = samples[sorted_sample_ids]

        top_index = int(self.cut_off * len(sorted_samples))
        top_samples = sorted_samples[:top_index]

        # plot exploration
        if self.ndim == 2:
            fpath = self.work_dir / get_full_name(name='dist', prefix='training',
                                                  suffix=f'{iter_cnt}_before')
            s = self.input_scale
            _range = np.array([[-s, s], [-s, s]])
            plt_hist2D(index_to_xval(self.input_vectors, samples), fpath=fpath, range=_range,
                       cmap='binary')

        return top_samples

    def setup_state(self):
        if self.load:
            iter_cnt, avg_cost = self.load_checkpoint(self.work_dir / 'checkpoint.pickle')
        else:
            iter_cnt = 0
            avg_cost = []
            self.collect_samples(self.n_init_samples)
            top_samples = self.get_top_samples(0)
            self.cem.fit(top_samples)
            if self.ndim == 2:
                xdata_ind = self.cem.sample(1000)
                fpath = self.work_dir / get_full_name(name='dist', prefix='training',
                                                      suffix=f'0_after')
                s = self.input_scale
                _range = np.array([[-s, s], [-s, s]])
                plt_hist2D(index_to_xval(self.input_vectors, xdata_ind), fpath=fpath,
                           range=_range, cmap='binary')

        return iter_cnt, avg_cost

    def sample_model(self, nsamples: int):
        sample_ids = self.cem.sample(nsamples)
        return sample_ids

    def load_and_sample_ids(self, nsamples) -> np.ndarray:
        """sets up the model and generates samples"""
        self.setup_state()
        sample_ids = self.sample_model(nsamples)
        return sample_ids

    def report_accuracy(self, ntimes, nsamples):
        accuracy_list, times, div_list = [], [], []

        if self.ndim == 2:
            sample_ids = self.sample_model(nsamples)
            s = self.input_scale
            _range = np.array([[-s, s], [-s, s]])
            plt_hist2D(index_to_xval(self.input_vectors, sample_ids),
                       fpath=self.work_dir / 'trained_policy',
                       range=_range, cmap='binary')

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

        print(f'gen_time / sample = {1e3 * np.mean(times).astype("float") / nsamples:.3f} ms')
        print(f'accuracy_avg = {100 * np.mean(accuracy_list).astype("float"):.6f}, '
              f'accuracy_std = {100 * np.std(accuracy_list).astype("float"):.6f}, '
              f'solution diversity = {np.mean(div_list).astype("float"):.6f}')

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

    def check_solutions(self, ntimes=1, nsamples=1000):
        print('-------- REPORT --------')
        self.report_accuracy(ntimes, nsamples)
        self.report_variation(nsamples)
        self.plot_solution_space(nsamples=1000)

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
        iter_cnt, avg_cost = self.setup_state()

        while iter_cnt < self.niter:
            self.collect_samples(self.nsamples)
            avg_cost.append(self.buffer.mean)
            top_samples = self.get_top_samples(iter_cnt+1)

            self.cem.fit(top_samples)

            if (iter_cnt + 1) % 10 == 0 and self.ndim == 2:
                xdata_ind = self.sample_model(1000)
                fpath = self.work_dir / get_full_name(name='dist', prefix='training',
                                                      suffix=f'{iter_cnt+1}_after')
                s = self.input_scale
                _range = np.array([[-s, s], [-s, s]])
                plt_hist2D(index_to_xval(self.input_vectors, xdata_ind),
                           fpath=fpath, range=_range, cmap='binary')
            iter_cnt += 1
            self.save_checkpoint(iter_cnt, avg_cost)

        plot_cost(avg_cost, fpath=self.work_dir / 'cost.png')

    def main(self):
        self._run_alg()
        self.check_solutions(ntimes=10, nsamples=1000)
