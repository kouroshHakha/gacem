from typing import Optional, Mapping, Any, Tuple

import pickle
import random
import time
from pathlib import Path
import numpy as np
import math
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal

from utils.file import read_yaml, write_yaml, get_full_name
from utils.loggingBase import LoggingBase

from ...benchmarks.functions import registered_functions
from ...benchmarks.fom import get_diversity_fom, compute_emprical_variation
from ...data.buffer import CacheBuffer
from ...viz.plot import plot_pca_2d, plt_hist2D, plot_cost, plot_fn2d, show_solution_region
from ...data.vector import index_to_xval

class CEM:
    """
    The vanilla implementation of Cross Entropy method with gaussian distributions
    """
    def __init__(self, input_vectors, average_coeff, dist_type='gauss'):
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
        """
        self.input_indices = [range(len(x)) for x in input_vectors]
        dim = len(input_vectors)
        self.params_min = np.array([0] * dim)
        self.params_max = np.array([len(x) - 1 for x in self.input_indices])

        self.type = dist_type
        self.average_coeff = average_coeff
        self.params = {}

    def fit(self, data):
        # data has to be in units of indices
        ndata = data.shape[0]
        alpha = self.average_coeff
        if self.type == 'gauss':
            new_mu = np.mean(data, axis=0)
            new_var = 1 / ndata * (data - new_mu).T @ (data - new_mu)
            self.params['mu'] = self.params.get('mu', 0) * (1 - alpha) + new_mu * alpha
            self.params['var'] = self.params.get('var', 0) * (1 - alpha) + new_var * alpha
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
            if self.type == 'gauss':
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

        self.allow_repeated = params.get('allow_repeated', False)
        self.elite_criteria = params.get('elite_criteria', 'optim')
        self.on_policy = params.get('on_policy', False)

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

        self.cem = CEM(self.input_vectors, dist_type=params['base_fn'],
                       average_coeff=params.get('average_coeff', 1))
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
        new_sample_fvals = []
        while n_collected < n:
            samples = self.sample_model(n - n_collected)
            xsamples = index_to_xval(self.input_vectors, samples)
            for xsample, sample in zip(xsamples, samples):
                if xsample not in self.buffer or self.allow_repeated:
                    fval = self.fn(xsample)
                    self.buffer.add_samples(xsample[None, ...], sample[None, ...], fval[None, ...],
                                            allow_repeated=self.allow_repeated)
                    new_samples.append(sample)
                    new_sample_fvals.append(fval)
                    n_collected += 1
                else:
                    print(f'item {xsample} already exists!')

        return np.array(new_samples), np.array(new_sample_fvals)

    def get_top_samples(self, iter_cnt, samples, sample_fvals):
        samples = samples if self.on_policy else np.array([x.item_ind for x in self.buffer.db_set])
        fvals = sample_fvals if self.on_policy else np.array([x.val for x in self.buffer.db_set])
        nsamples = len(samples)

        scores_arr = self.score(fvals)
        sample_ids = range(nsamples)
        sorted_sample_ids = sorted(sample_ids, key=lambda i: scores_arr[i])
        sorted_samples = samples[sorted_sample_ids]

        # find the last index which satisfies the constraint
        top_index = (scores_arr == 0).sum(-1).astype('int')
        if self.elite_criteria == 'optim':
            top_index = math.ceil(self.cut_off * nsamples)
        elif self.elite_criteria == 'csp':
            top_index = max(top_index, math.ceil(self.cut_off * nsamples))
        else:
            raise ValueError('invalid elite criteria: optim | csp')

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
            samples, sample_fvals = self.collect_samples(self.n_init_samples)
            top_samples = self.get_top_samples(0, samples, sample_fvals)
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
        try:
            self.plot_solution_space(nsamples=1000)
        except ValueError:
            print('Accuracy is not enough to plot solutions space (number of satisfying solutions '
                  'is small)')

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
            print(f'iter {iter_cnt}')
            samples, sample_fvals = self.collect_samples(self.nsamples)
            avg_cost.append(samples.mean() if self.on_policy else self.buffer.mean)
            top_samples = self.get_top_samples(iter_cnt+1, samples, sample_fvals)

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
        if self.ndim == 2:
            x, y = self.input_vectors
            plot_fn2d(x, y, self.fn, fpath=str(self.work_dir / 'fn2D.png'), cmap='viridis')
            show_solution_region(x, y, self.fn, self.goal, mode=self.mode,
                                 fpath=str(self.work_dir / 'dist2D.png'), cmap='binary')
        self._run_alg()
        self.check_solutions(ntimes=10, nsamples=1000)
