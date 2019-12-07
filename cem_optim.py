import numpy as np
import random
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from pathlib import Path
from utils.pdb import register_pdb_hook
import ruamel_yaml as yaml
import inspect
import time
import pickle
import pdb
from ackley import ackley_func, mixture_ackley, styblinski
from buffer import CacheBuffer
import argparse
from viz import Viz


register_pdb_hook()

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
                # samples = multivariate_normal.rvs(self.params['mu'],
                #                                   np.diag(self.params['var']), n)
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


class CEMSearch:

    def __init__(self, ndim, nsamples, n_init_samples,
                 niter, fn, goal_value, cut_off, input_scale,
                 num_input_points, mode='le',
                 load_ckpt_path='',
                 base_fn='guassian'):

        self.load_ckpt_path = load_ckpt_path
        if load_ckpt_path:
            self.dir = Path(load_ckpt_path).parent
        else:
            l_args, _, _, values = inspect.getargvalues(inspect.currentframe())
            params = dict(zip(l_args, [values[i] for i in l_args]))
            self.unique_name = time.strftime('%Y%m%d%H%M%S')
            self.dir = Path(f'data/cem_{self.unique_name}')
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / 'params.yaml', 'w') as f:
                yaml.dump(params, f)

        self.ndim = ndim
        self.nsamples = nsamples
        self.n_init_samples = n_init_samples
        self.niter = niter
        self.cut_off = cut_off
        self.input_scale = input_scale
        # goal has to always be positive if not we'll change mode and negate self.goal
        self.goal = goal_value
        self.mode = mode
        self.fn = fn
        if self.goal < 0:
            self.mode = 'le' if self.mode == 'ge' else 'ge'
            self.fn = lambda x: -fn(x)

        input_vectors_norm = [np.linspace(start=-1.0, stop=1.0, dtype='float32',
                                               num=num_input_points) for _ in range(ndim)]
        self.input_vectors = [input_scale * vec for vec in input_vectors_norm]

        self.cem = CEM(self.input_vectors, dist_type=base_fn)

        self.buffer = CacheBuffer(self.mode, self.goal, self.cut_off)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def score(self, fvals):
        if self.mode == 'le':
            err = (fvals - self.goal) / self.goal
        else:
            err = (self.goal - fvals) / self.goal
        scores = np.maximum(err , 0)
        return scores

    def plot_cost(self, avg_cost):
        plt.close()
        plt.plot(avg_cost)
        plt.title('samples mean')
        plt.xlabel('iter')
        plt.ylabel('avg_cost')
        plt.savefig(self.dir / 'cost.png')
        plt.close()

    def save_checkpoint(self, iter_cnt, avg_cost):
        dict_to_save = dict(
            iter_cnt=iter_cnt,
            avg_cost=avg_cost,
            buffer=self.buffer,
            cem=self.cem,
        )

        if not self.load_ckpt_path:
            self.load_ckpt_path = self.dir / 'checkpoint.pickle'

        with open(self.load_ckpt_path, 'wb') as f:
            pickle.dump(dict_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self):
        with open(self.load_ckpt_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.cem = checkpoint['cem']
        self.buffer = checkpoint['buffer']
        items = (checkpoint['iter_cnt'], checkpoint['avg_cost'])
        return items

    def index_to_xval(self, samples):
        cols = []
        dim = len(self.input_vectors)
        for i in range(dim):
            mat = np.zeros(shape=(samples.shape[0], 1)) + self.input_vectors[i][None, :]
            cols.append(mat[range(samples.shape[0]), samples[:, i]])
        xsamples = np.stack(cols, axis=-1)
        return xsamples

    def collect_samples(self, n):
        n_collected = 0
        new_samples = []
        while n_collected < n:
            samples = self.cem.sample(n - n_collected)
            xsamples = self.index_to_xval(samples)
            for xsample, sample in zip(xsamples, samples):
                if xsample not in self.buffer:
                    fval = self.fn(xsample)
                    self.buffer.add_samples(xsample[None, ...], sample[None, ...], fval[None, ...])
                    new_samples.append(xsample)
                    n_collected += 1
                else:
                    print(f'item {xsample} already exists!')

        return np.array(new_samples)


    def get_full_name(self, name, prefix='', suffix=''):
        if prefix:
            name = f'{prefix}_{name}'
        if suffix:
            name = f'{name}_{suffix}'
        return name

    def plt_hist2D(self, data: np.ndarray, ax=None, name='hist2D', **kwargs):
        fname = self.get_full_name(name,
                                   suffix=kwargs.pop('suffix', ''),
                                   prefix=kwargs.pop('prefix', ''))
        xvec, yvec = self.input_vectors
        if ax is None:
            ax = plt.gca()
        s = self.input_scale
        im = ax.hist2d(xvec[data[:, 0]], yvec[data[:, 1]], bins=100,
                       range=np.array([(-s, s), (-s, s)]), **kwargs)
        plt.colorbar(im[-1], ax=ax)
        plt.savefig(self.dir / f'{fname}.png')
        plt.close()

    def get_top_samples(self, iter_cnt):
        samples = np.array([x.item_ind for x in self.buffer.db_set])
        fvals = np.array([x.val for x in self.buffer.db_set])

        scores_arr = self.score(fvals)

        sample_ids = range(len(scores_arr))
        sorted_sample_ids = sorted(sample_ids, key=lambda i: scores_arr[i])

        sorted_samples = samples[sorted_sample_ids]
        sorted_scores = scores_arr[sorted_sample_ids]

        top_index = int(self.cut_off * len(sorted_samples))
        top_samples = sorted_samples[:top_index]

        # plot exploration
        if self.ndim == 2:
            self.plt_hist2D(samples, name='dist',
                            prefix='training', suffix=f'{iter_cnt}_before', cmap='binary')

        return top_samples

    def setup_state(self):
        if self.load_ckpt_path:
            iter_cnt, avg_cost = self.load_checkpoint()
        else:
            iter_cnt = 0
            avg_cost = []
            self.collect_samples(self.n_init_samples)
            top_samples = self.get_top_samples(0)
            self.cem.fit(top_samples)
            if self.ndim == 2:
                xdata_ind = self.cem.sample(1000)
                self.plt_hist2D(xdata_ind, name='dist', prefix='training',
                                suffix=f'0_after', cmap='binary')

        return iter_cnt, avg_cost

    def report_accuracy(self, ntimes, nsamples):
        accuracy_list, times, div_list = [], [], []

        if self.ndim == 2:
            sample_ids = self.cem.sample(nsamples)
            self.plt_hist2D(sample_ids, name='trained_policy', cmap='binary')

        for iter_id in range(ntimes):
            s = time.time()
            sample_ids = self.cem.sample(nsamples)
            xsample = self.index_to_xval(sample_ids)
            fval = self.fn(xsample)
            if self.mode == 'le':
                acc = (fval <= self.goal).sum(-1) / nsamples
                pos_samples = xsample[fval <= self.goal]
            else:
                acc = (fval >= self.goal).sum(-1) / nsamples
                pos_samples = xsample[fval >= self.goal]

            if len(pos_samples) >= self.ndim:
                div = Viz.get_diversity_fom(self.ndim, pos_samples)
                div_list.append(div)

            times.append(time.time() - s)
            accuracy_list.append(acc)

        print(f'gen_time / sample = {1e3 * np.mean(times).astype("float") / nsamples:.3f} ms')
        print(f'accuracy_avg = {100 * np.mean(accuracy_list).astype("float"):.6f}, '
              f'accuracy_std = {100 * np.std(accuracy_list).astype("float"):.6f}, '
              f'solution diversity = {np.mean(div_list).astype("float"):.6f}')

    def _compute_emprical_variation(self, samples):
        mean = samples.mean(0)
        cov_mat = ((samples - mean).T @ (samples - mean)) / samples.shape[0]
        var_sum = cov_mat.trace() / samples.shape[-1]
        return var_sum

    def report_variation(self, nsamples):
        sample_ids = self.cem.sample(nsamples)
        xsample = self.index_to_xval(sample_ids)
        fval = self.fn(xsample)

        total_var = self._compute_emprical_variation(xsample)
        if self.mode == 'le':
            pos_samples = xsample[fval <= self.goal]
            pos_var = self._compute_emprical_variation(pos_samples)
        else:
            pos_samples = xsample[fval >= self.goal]
            pos_var = self._compute_emprical_variation(pos_samples)

        print(f'total solution variation / dim = {total_var:.6f}')
        if np.isnan(pos_var):
            raise ValueError('did not find any satisfying solutions!')
        print(f'pos solution variation / dim = {pos_var:.6f}')

    def check_solutions(self, ntimes=1, nsamples=1000, init_seed=10):
        self.set_seed(init_seed)

        self.setup_state()

        print('-------- REPORT --------')
        self.report_accuracy(ntimes, nsamples)
        self.report_variation(nsamples)
        self.plot_solution_space(nsamples=1000)


    def plot_solution_space(self, nsamples=100):
        ax = plt.gca()
        sample_ids = self.cem.sample(nsamples)
        xsample = self.index_to_xval(sample_ids)
        fval = self.fn(xsample)
        if self.mode == 'le':
            pos_samples = xsample[fval <= self.goal]
        else:
            pos_samples = xsample[fval >= self.goal]

        Viz.plot_pca_2d(pos_samples, ax)
        plt.savefig(self.dir / f'pca_sol.png')
        plt.close()

    def main(self, seed):
        self.set_seed(seed)

        iter_cnt, avg_cost = self.setup_state()


        while iter_cnt < self.niter:
            self.collect_samples(self.nsamples)
            avg_cost.append(self.buffer.mean)
            top_samples = self.get_top_samples(iter_cnt+1)

            self.cem.fit(top_samples)

            if (iter_cnt + 1) % 10 == 0 and self.ndim == 2:
                xdata_ind = self.cem.sample(1000)
                self.plt_hist2D(xdata_ind, name='dist', prefix='training',
                                suffix=f'{iter_cnt+1}_after', cmap='binary')
            iter_cnt += 1
            self.save_checkpoint(iter_cnt, avg_cost)

        self.plot_cost(avg_cost)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', type=int, default=10, help='random seed')
    parser.add_argument('-ckpt', '--ckpt', type=str, default='', help='checkpoint path')
    args = parser.parse_args()

    searcher = CEMSearch(ndim=2,
                         goal_value=20,
                         n_init_samples=20,
                         nsamples=5,
                         niter=100,
                         fn=styblinski,
                         cut_off=0.4,
                         input_scale=5.0,
                         num_input_points=100,
                         mode='le',
                         load_ckpt_path=args.ckpt,
                         base_fn='kde',
                         )
    searcher.main(args.seed)
    searcher.check_solutions(ntimes=10, nsamples=1000, init_seed=args.seed)
