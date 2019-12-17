"""
This module implements an ensemble version of cont_autoreg model
"""
from typing import Optional, Any, Tuple, Union, List, Dict

import math
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.file import get_full_name

from ...benchmarks.functions import registered_functions
from ...benchmarks.fom import (
    compute_emprical_variation, get_diversity_fom
)
from ...data.buffer import CacheBuffer
from ...models.made import MADE
from ...models.ensemble import Ensemble
from ..random.random import Random
from ...data.vector import index_to_xval
from ...alg.base import AlgBase

from ...viz.plot import (
    plot_pca_2d, plt_hist2D, plot_cost, plot_learning_with_epochs
)

import pdb

class AutoRegSearch(AlgBase):

    def __init__(self, *args, **kwargs):
        AlgBase.__init__(self, *args, **kwargs)

        params = self.specs['params']

        self.seed = params.get('seed', 10)
        self.set_seed(self.seed)
        self.ndim = params['ndim']
        self.bsize = params['batch_size']
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
        # whether to run 1000 epochs of training for the later round of iteration
        self.full_training = params['full_training_last']
        self.input_scale = params['input_scale']

        eval_fn = params['eval_fn']
        try:
            self.fn = registered_functions[eval_fn]
        except KeyError:
            raise ValueError(f'{eval_fn} is not a valid benchmark function')

        self.model: Optional[nn.Module] = None
        self.buffer = None
        self.opt = None

        # hacky version of passing input vectors around
        self.input_vectors_norm = [np.linspace(start=-1.0, stop=1.0, dtype='float32',
                                               num=100) for _ in range(self.ndim)]
        self.input_vectors = [self.input_scale * vec for vec in self.input_vectors_norm]
        # TODO: remove this hacky way of keeping track of delta
        self.delta = self.input_vectors_norm[0][-1] - self.input_vectors_norm[0][-2]

        self.setup_model(params['models'])

    def setup_model(self, model_params: List[Dict[str, Any]]):
        dim = self.ndim
        module_list = []
        for i, model in enumerate(model_params):
            module_list.append(MADE(dim, model['hiddens'], dim * 3 * model['nr_mix'],
                                    seed=self.seed + 10 * i, natural_ordering=True))
        params = self.specs['params']
        self.model: Ensemble = Ensemble(module_list, self.ndim, params['base_fn'], self.delta,
                                        params['beta'], seed=self.seed)

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)
        self.buffer = CacheBuffer(self.mode, self.goal, self.cut_off)

    @classmethod
    def set_seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(seed)

    def run_epoch(self, data: np.ndarray, weights: np.ndarray, mode='train'):
        self.model.train() if mode == 'train' else self.model.eval()

        n, dim,  _ = data.shape

        assert n != 0, 'no data found'

        bsize = max(self.bsize, 2 ** math.floor(math.log2(n / 4))) if mode == 'train' else n
        nstep = n // bsize if mode == 'train' else 1

        nll_per_b = 0

        for step in range(nstep):
            xb = data[step * bsize: step * bsize + bsize]
            wb = weights[step * bsize: step * bsize + bsize]
            xb_tens = torch.from_numpy(xb)
            wb_tens = torch.from_numpy(wb)

            xin = xb_tens[:, 0, :]
            loss = self.model.get_nll(xin, weights=wb_tens)
            if mode == 'train':
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1e3)
                self.opt.step()
            nll_per_b += loss.to(torch.device('cpu')).item() / nstep

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
                _, xnew_ind = self.model.sample_model(1, self.bsize, self.input_vectors_norm)
                xnew_id_np = xnew_ind.to(torch.device('cpu')).data.numpy().astype('int')

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
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.opt.load_state_dict(checkpoint['opt_state'])
        # override optimizer with input parameters
        self.opt = optim.Adam(self.model.parameters(), self.lr)
        self.buffer = checkpoint['buffer']
        items = (checkpoint['iter_cnt'], checkpoint['tr_losses'], checkpoint['avg_cost'])
        print(f'Model checkpoint loaded in {time.time() - s:.4f} seconds')
        return items

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
                _, xdata_ind = self.model.sample_model(1000, self.bsize, self.input_vectors_norm)
                fpath = self.work_dir / get_full_name(name='dist', prefix='training',
                                                      suffix=f'0_after')
                data_ind = xdata_ind.to(torch.device('cpu')).data.numpy().astype('int')
                data = index_to_xval(self.input_vectors, data_ind)
                s = self.input_scale
                _range = np.array([[-s, s], [-s, s]])
                plt_hist2D(data, fpath=fpath, range=_range, cmap='binary')

            items = (0, [], [])
            self.save_checkpoint(*items)
        return items

    def _run_alg(self):
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
                _, xdata_ind = self.model.sample_model(1000, self.bsize, self.input_vectors_norm)
                fpath = self.work_dir / get_full_name(name='dist', prefix='training',
                                                      suffix=f'{iter_cnt+1}_after')
                data_ind = xdata_ind.to(torch.device('cpu')).data.numpy().astype('int')
                data = index_to_xval(self.input_vectors, data_ind)
                s = self.input_scale
                _range = np.array([[-s, s], [-s, s]])
                plt_hist2D(data, fpath=fpath, range=_range, cmap='binary')

            iter_cnt += 1
            self.save_checkpoint(iter_cnt, tr_losses, avg_cost)

        plot_learning_with_epochs(fpath=self.work_dir / 'learning_curve.png', training=tr_losses)
        plot_cost(avg_cost, fpath=self.work_dir / 'cost.png')

    def _sample_model_for_eval(self, nsamples) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, sample_ids = self.model.sample_model(nsamples, self.bsize, self.input_vectors_norm)
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

        print(f'gen_time / sample = {1e3 * np.mean(times).astype("float") / nsamples:.3f} ms')
        print(f'accuracy_avg = {100 * np.mean(accuracy_list).astype("float"):.6f}, '
              f'accuracy_std = {100 * np.std(accuracy_list).astype("float"):.6f}, '
              f'solution diversity = {np.mean(div_list).astype("float"):.6f}')

    def load_and_sample(self, nsamples, only_positive=False) -> np.ndarray:
        """sets up the model (i.e. initializes the weights .etc) and generates samples"""
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
        # self.check_random_solutions(ntimes=10, nsamples=10)
        input('Press Enter To continue:')
        self._run_alg()
        self.check_solutions(ntimes=3, nsamples=100)
