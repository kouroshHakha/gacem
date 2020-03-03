"""
This module implements a base structure for the algorithm with specific methods to be overridden
"""
from typing import Optional, Mapping, Any, Sequence, Dict, Tuple

import time
import abc
from pathlib import Path
import numpy as np
import atexit


from utils.file import read_yaml, write_yaml, get_full_name, read_pickle, write_pickle
from utils.loggingBase import LoggingBase
from utils.data.database import Database

from bb_eval_engine.util.importlib import import_bb_env
from bb_eval_engine.data.design import Design
from bb_eval_engine.base import EvaluationEngineBase


class AlgBase(LoggingBase, abc.ABC):
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

        if spec_file:
            specs = read_yaml(spec_file)
        else:
            specs = spec_dict

        self.use_model = kwargs.get('use_model', False)
        self.specs = specs
        self.load = load

        if load or self.use_model:
            self.work_dir = Path(spec_file).parent
        else:
            unique_name = time.strftime('%Y%m%d%H%M%S')
            suffix = specs['params'].get('suffix', '')
            prefix = specs['params'].get('prefix', '')
            unique_name = get_full_name(unique_name, prefix, suffix)
            self.work_dir = Path(specs['root_dir']) / f'{unique_name}'
            write_yaml(self.work_dir / 'params.yaml', specs, mkdir=True)

        params = specs['params']
        # create evaluation core instance
        self.bb_env: EvaluationEngineBase = import_bb_env(params['bb_env'])
        check_params = params['check_params']
        self.check_ntimes = check_params['ntimes']
        self.check_nsamples = check_params['nsamples']

        self.seed = params['seed']
        self.nsamples = params['nsamples']

        self.db = Database(Design)

        self.state = None
        self.iter_cnt = 0

        LoggingBase.__init__(self, self.work_dir)
        atexit.register(self.end)

    @abc.abstractmethod
    def collect_samples(self, n: int, is_init: bool = False, **kwargs) -> Sequence[Design]:
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, samples: Sequence[Design], **kwargs):
        raise NotImplementedError

    def update_state(self):
        self._update_state()

    @abc.abstractmethod
    def _update_state(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _set_seed(cls, seed):
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def setup_state(self):
        """Sets the initial self.state by loading the checkpoint if self.load is True.
        The algorithm should be able to resume by invoking this function. After calling this
        function self.state should contain iter_cnt field"""
        raise NotImplementedError

    @abc.abstractmethod
    def report_accuracy(self, ntimes: int, nsamples: int) -> Tuple[float, float]:
        """ Reports the accuracy and std of it.
        Parameters
        ----------
        ntimes: int
            Number of times to do the experiment.
        nsamples: int
            Number of samples to draw in each experiment.

        Returns
        -------
        acc, std: Tuple[float, float]
            The average and standard deviation of the accuracy across experiments.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_model(self, n: int) -> Sequence[Design]:
        raise NotImplementedError


    def end(self) -> None:
        self.save_checkpoint()
        self._end()

    @abc.abstractmethod
    def _end(self) -> None:
        """The end goes here! things like plotting .etc"""
        raise NotImplementedError

    def _run_alg(self):
        """override this if you want a more custom implementation"""
        self.setup_state()
        self.iter_cnt = self.state['iter_cnt']
        # self.db = self.state['db']
        while not self.stop():
            is_init = (self.iter_cnt == 0) and not self.use_model
            samples: Sequence[Design] = self.collect_samples(self.nsamples, is_init=is_init)
            self.train(samples)
            self.iter_cnt += 1
            avg_obj = np.mean([sample['obj'] for sample in samples])
            self.info(f'Iter {self.iter_cnt}: avg_obj = {avg_obj}')
            self.update_state()

    def update_obj(self, designs: Sequence[Design]) -> Sequence[Design]:
        """ Takes in a list of designs and updates their obj attribute to reflect the objective of
        cem based algorithms"""
        def err(val, target, plus=True):
            if plus:
                return (val - target) / (val + target + 1e-15)
            return (target - val) / (val + target + 1e-15)
        for dsn in designs:
            objectives = []
            for key, spec in self.bb_env.spec_range.items():
                if (spec.ub is None) == (spec.lb is None):
                    raise ValueError(f'spec {key} has an error in the environment!')
                elif spec.ub:
                    # spec has an upper bound
                    spec_obj = - spec.weight * err(dsn[key], spec.ub)
                else:
                    # spec has a lower bound
                    spec_obj = - spec.weight * err(dsn[key], spec.lb, False)
                objectives.append(spec_obj)
            dsn['obj'] = max(objectives)

        return designs

    def save_checkpoint(self) -> None:
        self.info('Saving Checkpoint ...')
        write_pickle(self.work_dir / 'checkpoint.pickle', self.state)

    def load_checkpoint(self) -> Dict[str, Any]:
        return read_pickle(self.work_dir / 'checkpoint.pickle')

    def check_solutions(self, ntimes=1, nsamples=1000):
        self.log_text('-------- REPORT --------')
        acc, std = self.report_accuracy(ntimes, nsamples)

        saved_data = dict(acc=acc, std=std)
        write_yaml(self.work_dir / 'performance.yaml', saved_data)

    def load_and_sample(self, nsamples: int, only_positive: bool = False) -> Sequence[Design]:
        """sets up the model and generates samples"""
        self.setup_state()
        samples = self.sample_model(nsamples)

        if not only_positive:
            return samples

        n_remaining = nsamples
        samples = []
        tried = {}
        while n_remaining > 0:
            new_samples = self.sample_model(n_remaining)
            for sample in new_samples:
                if sample not in self.db and sample not in tried:
                    sample,  = self.bb_env.evaluate([sample])
                    sample, = self.update_obj([sample])
                elif sample in self.db:
                    dt = self.db[sample]
                    sample = self.db[dt]
                else:
                    sample = tried[sample]

                if sample['valid'] and sample['obj'] < 0:
                    tried[sample] = sample
                    samples.append(sample)
                    n_remaining -= 1

        return samples

    def main(self) -> None:
        self._set_seed(self.seed)
        self._run_alg()
        self.check_solutions(ntimes=self.check_ntimes, nsamples=self.check_nsamples)