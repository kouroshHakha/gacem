from typing import Optional, Mapping, Any, List, Tuple
import random
import numpy as np
from pathlib import Path
import time

from utils.file import read_yaml, write_yaml, get_full_name
from utils.loggingBase import LoggingBase

from gacem.benchmarks.functions import registered_functions
from gacem.benchmarks.fom import (
    compute_emprical_variation, get_diversity_fom
)
from gacem.viz.plot import (
    plot_fn2d, plt_hist2D, show_solution_region
)

class Random(LoggingBase):

    # noinspection PyUnusedLocal
    def __init__(self, spec_file: str = '', spec_dict: Optional[Mapping[str, Any]] = None,
                 **kwargs) -> None:
        LoggingBase.__init__(self)

        if spec_file:
            specs = read_yaml(spec_file)
        else:
            specs = spec_dict

        self.specs = specs
        params = specs['params']

        try:
            self.work_dir = params['work_dir']
        except KeyError:
            unique_name = time.strftime('%Y%m%d%H%M%S')
            self.work_dir = Path(specs['root_dir']) / f'random_{unique_name}'
            write_yaml(self.work_dir / 'params.yaml', specs, mkdir=True)


        self.ndim = params['ndim']
        self.goal = params['goal_value']
        self.mode = params['mode']
        self.input_scale = params['input_scale']

        eval_fn = params['eval_fn']
        try:
            self.fn = registered_functions[eval_fn]
        except KeyError:
            raise ValueError(f'{eval_fn} is not a valid benchmark function')

        # hacky version of passing input vectors around
        self.input_vectors_norm = [np.linspace(start=-1.0, stop=1.0, dtype='float32',
                                               num=100) for _ in range(self.ndim)]
        self.input_vectors = [self.input_scale * vec for vec in self.input_vectors_norm]
        # TODO: remove this hacky way of keeping track of delta
        self.delta = self.input_vectors_norm[0][-1] - self.input_vectors_norm[0][-2]

    @classmethod
    # TODO: make sure input_vecs_norm is passed in
    def sample_data(cls, ndim: int, input_vecs: List[np.ndarray],
                    nsample: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """ sample randomly (i.e. not on policy)
        Returns both the actual sample numbers and their corresponding indices
        """
        samples_ind = []
        while len(samples_ind) < nsample:
            temp_l = []
            for i in range(ndim):
                len_dim = len(input_vecs[i])
                temp_l.append(random.randrange(0, len_dim))
            samples_ind.append(np.array(temp_l))

        data_ind = np.stack(samples_ind)
        samples = np.stack([input_vecs[i][data_ind[:, i]] for i in range(ndim)], axis=-1)

        return samples, data_ind

    def check_solutions(self, ntimes: int, nsamples: int) -> None:
        accuracy_rnd_list = []
        total_var_list, pos_var_list = [], []
        diversity_fom_list = []

        if self.ndim == 2:
            rnd_samples, _ = self.sample_data(self.ndim, self.input_vectors, nsamples)
            s = self.input_scale
            _range = np.array([[-s, s], [s, s]])
            plt_hist2D(rnd_samples, fpath=self.work_dir / get_full_name('random_policy'),
                       range=_range, cmap='binary')
            x, y = self.input_vectors
            plot_fn2d(x, y, self.fn, fpath=str(self.work_dir / 'fn2D.png'), cmap='viridis')
            show_solution_region(x, y, self.fn, self.goal, mode=self.mode,
                                 fpath=str(self.work_dir / 'dist2D.png'), cmap='binary')

        vector_mat = np.stack(self.input_vectors, axis=0)
        for iter_id in range(ntimes):
            _, rnd_ids = self.sample_data(self.ndim, self.input_vectors_norm, nsamples)
            rnd_samples = vector_mat[np.arange(self.ndim), rnd_ids]
            total_var = compute_emprical_variation(rnd_samples)
            rnd_fval: np.ndarray = self.fn(rnd_samples)
            if self.mode == 'le':
                pos_samples = rnd_samples[rnd_fval <= self.goal]
                if len(pos_samples) != 0:
                    pos_var = compute_emprical_variation(pos_samples)
                else:
                    pos_var = np.NAN
                accuracy_rnd_list.append((rnd_fval <= self.goal).sum(-1) / nsamples)
            else:
                pos_samples = rnd_samples[rnd_fval >= self.goal]
                if len(pos_samples) != 0:
                    pos_var = compute_emprical_variation(pos_samples)
                else:
                    pos_var = np.NAN
                accuracy_rnd_list.append((rnd_fval >= self.goal).sum(-1) / nsamples)

            pos_var_list.append(pos_var)
            total_var_list.append(total_var)

            if len(pos_samples) >= self.ndim:
                div = get_diversity_fom(self.ndim, pos_samples)
                diversity_fom_list.append(div)

        accuracy_rnd = np.array(accuracy_rnd_list, dtype='float32')
        print(f'accuracy_rnd_avg = {100 * np.mean(accuracy_rnd).astype("float"):.6f}, '
              f'accuracy_rnd_std = {100 * np.std(accuracy_rnd).astype("float"):.6f}')
        print(f'random policy total variation / dim = '
              f'{np.mean(total_var_list).astype("float"):.6f}')

        pos_var_arr = np.array(pos_var_list)
        if len(pos_var_arr[~np.isnan(pos_var_arr)]) == 0:
            print('No positive solution was found with random policy')
        else:
            print(f'pos solution variation / dim ='
                  f' {np.mean(pos_var_arr[~np.isnan(pos_var_arr)]):.6f}')
            print(f'random policy solution diversity FOM: '
                  f'{np.mean(diversity_fom_list).astype("float"):.6f}')

    def main(self) -> None:
        self.check_solutions(10, 1000)
