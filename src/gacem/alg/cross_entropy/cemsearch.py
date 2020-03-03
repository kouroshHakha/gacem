from typing import Sequence, cast

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedList

from bb_eval_engine.data.design import Design

from ...viz.plot import plot_x_y, plot_cost, plt_hist2D, scatter2d
from ..base import AlgBase
from .cem import CEM
from .cma_es import CMAES


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

        self.n_init_samples = params['n_init_samples']
        self.niter = params['niter']
        self.cut_off = params['cut_off']

        self.on_policy = params.get('on_policy', False)

        self.input_vectors = list(self.bb_env.params_vec.values())

        model = params.get('model', 'cem')
        if model == 'cem' :
            self.model = CEM(self.input_vectors, dist_type=params['base_fn'],
                           average_coeff=params.get('average_coeff', 1),
                           gauss_sigma=params.get('gauss_sigma', None))
        elif model == 'cma_es':
            self.model = CMAES(self.input_vectors, self.seed, lambda_=self.nsamples,
                               mu=self.cut_off)
        else:
            raise ValueError(f'model {model} is not supported! (cem | cma_es)')

        # attribute to keep track of top designs
        self.sorted_db = SortedList(key=lambda x: x['obj'])
        self.nsol = 0

    @classmethod
    def _set_seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)

    def report_accuracy(self, ntimes, nsamples):
        accuracy_list, times = [], []

        for iter_id in range(ntimes):
            s = time.time()
            samples = self.sample_model(nsamples)
            samples = self.bb_env.evaluate(samples)
            samples = self.update_obj(samples)
            objs = np.array([x['obj'] for x in samples])
            acc = float((objs <= 0).sum(-1) / nsamples)

            times.append(time.time() - s)
            accuracy_list.append(acc)

        acc_mean = 100 * float(np.mean(accuracy_list))
        acc_std = 100 * float(np.std(accuracy_list))
        self.info(f'gen_time / sample = {1e3 * float(np.mean(times)) / nsamples:.3f} ms')
        self.info(f'accuracy_avg = {acc_mean:.6f}, accuracy_std = {acc_std:.6f}')

        return acc_mean, acc_std

    def stop(self) -> bool:
        return self.iter_cnt >= self.niter

    def get_top_samples(self, samples: Sequence[Design]) -> Sequence[Design]:

        if self.on_policy:
            sorted_samples = sorted(samples, key=lambda x: x['obj'])
        else:
            sorted_samples = sorted(self.db, key=lambda x: x['obj'])

        metric = [sample['obj'] for sample in sorted_samples]
        # obj is in [-1, 1], the more negative obj is the better the design is.
        top_index = int((np.array(metric) <= 0).sum(-1))
        top_index = max(top_index, min(self.cut_off, len(samples)))
        top_samples = sorted_samples[:top_index]

        return top_samples

    def train(self, samples: Sequence[Design], **kwargs):
        top_samples = self.get_top_samples(samples)
        top_samples_np = np.array(top_samples)
        if self.bb_env.ndim == 2:
            ranges = np.array(list(zip(self.bb_env.params_min, self.bb_env.params_max)))
            fpath = self.work_dir / f'samples_{self.iter_cnt}.png'
            plt.close()
            ax = plt.gca()
            self.bb_env.plot_contours(ranges=ranges, ax=ax)
            plt_hist2D(np.array(samples), cmap='binary', range=ranges, ax=ax)
            ax.scatter(top_samples_np[:, 0], top_samples_np[:, 1], marker='x', color='red')
            mean = top_samples_np.mean(0)
            ax.scatter(mean[0], mean[1], marker='x', color='blue')
            plt.savefig(fpath, dpi=200)

        self.model.fit(top_samples_np)

    def setup_state(self):
        if self.load:
            state = self.load_checkpoint()
            self.model = state['model']
            self.state = state
        else:
            self.state = dict(
                iter_cnt=0,
                avg_cost=[],
                sim_cnt_list=[],
                sample_cnt_list=[],
                n_sols_in_db=[],
                top_20=[],
                model=self.model,
                db=self.db,
            )
            if self.use_model:
                state = self.load_checkpoint()
                self.state['model'] = state['model']
                self.model = state['model']

    def _update_state(self):
        state = self.state
        state['iter_cnt'] = self.iter_cnt
        state['sim_cnt_list'].append(len(self.db))
        state['sample_cnt_list'].append(self.db.tot_freq)
        state['n_sols_in_db'].append(self.nsol)
        if len(self.sorted_db) > 0:
            mean = m20 = np.mean([x['obj'] for x in self.sorted_db[:20]])
        else:
            mean = m20 = float('inf')
        state['avg_cost'].append(mean)
        state['top_20'].append(m20)
        state['model'] = self.model
        state['db'] = self.db

    def sample_model(self, n: int) -> Sequence[Design]:
        samples = self.model.sample(n)
        samples = [Design(sample) for sample in samples]
        return samples

    def collect_samples(self, n: int, is_init: bool = False, **kwargs) -> Sequence[Design]:
        # if is_init:
        #     samples = self.bb_env.generate_rand_designs(n, True, self.seed)
        # else:
        n_remaining = n
        samples = []
        tried = {}
        while n_remaining > 0:
            new_samples = self.sample_model(n_remaining)

            for sample in new_samples:
                if sample not in self.db and sample not in tried:
                    sample, = self.bb_env.evaluate([sample])
                    sample, = self.update_obj([sample])
                elif sample in self.db:
                    dt = self.db[sample]
                    sample = self.db[dt]
                else:
                    sample = tried[sample]

                tried[sample] = sample
                if sample['valid']:
                    if sample['obj'] <= 0:
                        self.nsol += 1
                    self.sorted_db.add(sample)
                    samples.append(sample)
                    n_remaining -= 1

        self.update_obj(samples)
        self.db.extend(samples)

        return samples

    def _end(self) -> None:
        plot_cost(self.state['avg_cost'], fpath=self.work_dir / 'cost.png')
        plot_x_y(self.state['sample_cnt_list'], self.state['n_sols_in_db'],
                 fpath=self.work_dir / 'n_sols.png',
                 xlabel='n_freq', ylabel=f'n_sols')
