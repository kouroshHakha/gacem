"""
This module extends ackley_auto_reg_search3_cont_model.py to n dimensional optimization with the
same optimal point.
"""
from typing import Optional
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import ruamel_yaml as yaml
import inspect
import pdb
from pathlib import Path
import random
import time

from ackley import ackley_func
from made import MADE
from buffer import CacheBuffer
from utils.pdb import register_pdb_hook

import argparse


register_pdb_hook()

class AutoReg2DSearch:

    def __init__(self, ndim, goal_value, hidden_list, mode='le', batch_size=16, cut_off=20,
                 nepochs=1, nsamples=1, n_init_samples=100, niter=1000, lr=1e-3, beta: float = 0,
                 init_nepochs=1, base_fn='logistic', nr_mix=1, only_positive=False,
                 full_training_last=False, input_scale=1.0, num_input_points=100,
                 load_ckpt_path=''):
        """

        :param ndim:
        :param goal_value:
        :param hidden_list:
        :param mode:
        :param batch_size:
        :param cut_off:
        :param nepochs:
        :param nsamples:
        :param n_init_samples:
        :param niter:
        :param lr:
        :param beta:
        :param init_nepochs:
        :param base_fn:
        :param nr_mix:
        :param only_positive:
        :param full_training_last:
        :param input_scale: input_range will be on the the scale of [-1, 1] * input_scale with
        num_input_points possible values.
        :param num_input_points:
        """

        if load_ckpt_path:
            self.dir = Path(load_ckpt_path).parent
        else:
            l_args, _, _, values = inspect.getargvalues(inspect.currentframe())
            params = dict(zip(l_args, [values[i] for i in l_args]))
            self.unique_name = time.strftime('%Y%m%d%H%M%S')
            self.dir = Path(f'data/search_fig_{self.unique_name}')
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / 'params.yaml', 'w') as f:
                yaml.dump(params, f)

        self.ndim = ndim
        self.bsize = batch_size
        self.hiddens = hidden_list
        self.niter = niter
        self.goal = goal_value
        self.mode = mode
        self.viz_rate = niter // 10
        self.lr = lr
        self.nepochs = nepochs
        self.nsamples = nsamples
        self.n_init_samples = n_init_samples
        self.init_nepochs = init_nepochs
        self.cut_off = cut_off
        self.beta = beta
        self.nr_mix = nr_mix
        self.base_fn = base_fn
        self.only_pos = only_positive
        # whether to run 1000 epochs of training for the later round of iteration
        self.full_training = full_training_last
        self.load_ckpt_path = load_ckpt_path
        self.input_scale = input_scale

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cpu = torch.device('cpu')
        self.model: Optional[nn.Module] = None
        self.opt = None

        # self.writer = SummaryWriter()

        # hacky version of passing input vectors around
        self.input_vectors_norm = [np.linspace(start=-1.0, stop=1.0, dtype='float32',
                                               num=num_input_points) for _ in range(ndim)]
        self.input_vectors = [input_scale * vec for vec in self.input_vectors_norm]
        # TODO: remove this hacky way of keeping track of delta
        self.delta = self.input_vectors_norm[0][-1] - self.input_vectors_norm[0][-2]

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def sample_data(self, nsample=100):
        """ sample randomly (i.e. not on policy)
        Returns both the actual sample numbers and their corresponding indices
        """
        samples_ind = []
        while len(samples_ind) < nsample:
            temp_l = []
            for i in range(self.ndim):
                len_dim = len(self.input_vectors_norm[i])
                temp_l.append(random.randrange(0, len_dim))
            samples_ind.append(np.array(temp_l))

        data_ind = np.stack(samples_ind)
        samples = np.stack([self.input_vectors_norm[i][data_ind[:, i]] for i in range(self.ndim)],
                           axis=-1)

        return samples, data_ind

    def cdf_normal(self, x: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor):
        x_centered = x - mean
        erf_x = (x_centered / torch.tensor(2**0.5) / sigma).erf()
        return 0.5 * (1 + erf_x)

    def cdf_logistic(self, x: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor):
        x_centered = x - mean
        sigmoid_x = (x_centered / torch.tensor(2**0.5) / sigma).sigmoid()
        return sigmoid_x

    def cdf_uniform(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        eps = torch.tensor(1e-15)
        l = (x - a) / (b - a + eps)
        bot = l.relu()
        cdf = - (torch.tensor(1)-bot).relu() + 1
        return cdf

    def get_probs(self, xin: torch.Tensor, debug=False):
        """Given an input tensor (N, D) computes the probabilities across the cardinality space
        of each dimension. prob.shape = (N, D, K) where K is number of possible values for each
        dimension. Assume that xin is normalized to [-1,1], and delta is given."""
        delta = self.delta
        xin = xin.to(self.device)
        xhat = self.model(xin)
        D = self.ndim
        coeffs = torch.stack([xhat[:, i::D*3] for i in range(D)], dim=1)
        # Note: coeffs was previously interpreted as log_coeffs
        # interpreting outputs if NN as log is dangerous, can result in Nan's.
        # solution: here they should be positive and should add up to 1, sounds familiar? softmax!
        coeffs_norm = coeffs.softmax(dim=-1)

        eps = torch.tensor(1e-15)
        xb = xin[..., None] + torch.zeros(coeffs.shape)

        if self.base_fn in ['logistic', 'normal']:
            means = torch.stack([xhat[:, i+D::D*3] for i in range(D)], dim=1)
            log_sigma = torch.stack([xhat[:, i+2*D::D*3] for i in range(D)], dim=1)
            # put a cap on the value of output so that it does not blow up
            log_sigma = torch.min(log_sigma, torch.ones(log_sigma.shape) * 50)
            # put a bottom on the value of output so that it does not diminish and becomes zero
            log_sigma = torch.max(log_sigma, torch.ones(log_sigma.shape) * (-40))
            sigma = log_sigma.exp()

            if self.base_fn == 'logistic':
                plus_cdf = self.cdf_logistic(xb + delta / 2, means, sigma)
                minus_cdf = self.cdf_logistic(xb - delta / 2, means, sigma)
            else:
                plus_cdf = self.cdf_normal(xb + delta / 2, means, sigma)
                minus_cdf = self.cdf_normal(xb - delta / 2, means, sigma)
        elif self.base_fn == 'uniform':
            center = torch.stack([xhat[:, i+D::D*3] for i in range(D)], dim=1)
            # normalize center between [-1,1] to cover all the space
            center = 2 * (center - center.min()) / (center.max() - center.min() + eps) - 1
            log_delta = torch.stack([xhat[:, i+2*D::D*3] for i in range(D)], dim=1)
            # put a cap on the value of output so that it does not blow up
            log_delta = torch.min(log_delta, torch.ones(log_delta.shape) * 50)
            bdelta = log_delta.exp()
            a = center - bdelta / 2
            b = center + bdelta / 2
            plus_cdf = self.cdf_uniform(xb + delta / 2, a, b)
            minus_cdf = self.cdf_uniform(xb - delta / 2, a, b)
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

        bar_grad = torch.autograd.grad(probs[0,0], self.model.net[0].bias, retain_graph=True)[0]
        if torch.isnan(bar_grad[0]):
            print(bar_grad)
            print(cdfs)
            pdb.set_trace()

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
            obj_term  = - (weights).data
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


    def sample_probs(self, probs: torch.Tensor, index: int):
        """Given a probability distribution tensor (shape = (N, D, K)) returns 1 sample from the
        distribution probs[:, index, :], the output is indices"""
        p = probs[:, index]
        sample = p.multinomial(num_samples=1).squeeze()
        return sample

    def sample_model(self, nsamples: int):
        self.model.eval()

        D = self.ndim

        samples = []
        samples_ind = []
        for k in range(nsamples):
            xsample = torch.zeros(1, D)
            xsample_ind = torch.zeros(1, D)
            for i in range(D):
                N = len(self.input_vectors_norm[i])
                xin = torch.zeros(N, D)
                if i >= 1:
                    xin = torch.stack([xsample.squeeze()] * N)
                xin[:, i] = torch.from_numpy(self.input_vectors_norm[i])
                #TODO: For normal dist this probs gets a lot of mass on the edges
                probs = self.get_probs(xin)
                xi_ind = self.sample_probs(probs, i)  # ith x index
                xsample[0, i] = torch.tensor(self.input_vectors_norm[i][xi_ind])
                xsample_ind[0, i] = xi_ind
            samples.append(xsample.squeeze())
            samples_ind.append(xsample_ind.squeeze())

        samples = torch.stack(samples, dim=0)
        samples_ind = torch.stack(samples_ind, dim=0)
        return samples, samples_ind

    def get_full_name(self, name, prefix='', suffix=''):
        if prefix:
            name = f'{prefix}_{name}'
        if suffix:
            name = f'{name}_{suffix}'
        return name

    def run_epoch(self, data: np.ndarray, weights: np.ndarray, mode='train',
                  debug=False):
        self.model.train() if mode == 'train' else self.model.eval()

        N, D,  _ = data.shape

        assert N != 0, 'no data found'

        B = max(self.bsize, 2 ** math.floor(math.log2(N / 4))) if mode == 'train' else N
        nstep = N // B if mode == 'train' else 1

        nll_per_b = 0

        for step in range(nstep):
            xb = data[step * B: step * B + B]
            wb = weights[step * B: step * B + B]
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
                _, xnew_id_np = self.sample_data(1)
                xnew_id_np = xnew_id_np.astype('int')
            else:
                _, xnew_ind = self.sample_model(1)
                xnew_id_np = xnew_ind.to(self.cpu).data.numpy().astype('int')

            # simulate and compute the adjustment weights
            org_sample_list = [vecs[index][id] for index, id in enumerate(xnew_id_np[0, :])]
            xsample = np.array(org_sample_list, dtype='float32')
            fval = ackley_func(xsample[None, :])

            norm_sample_list = [norm_vecs[index][id] for index, id in enumerate(xnew_id_np[0, :])]
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
        self.plt_hist2D(xtr[:, 1, :].astype('int'),
                        name='dist', prefix='training', suffix=f'{iter_cnt}_before')
        # per epoch
        print('-'*50)
        tr_loss = 0
        te_loss = 0
        tr_loss_list = []
        for epoch_id in range(nepochs):
            tr_nll = self.run_epoch(xtr, wtr, mode='train')
            tr_loss_list.append(tr_nll)
            tr_loss += tr_nll / self.nepochs

            print(f'[train_{iter_cnt}] epoch {epoch_id} loss = {tr_nll}')

            if split < 1:
                te_nll = self.run_epoch(xte, wte, mode='test')
                te_loss += te_nll / self.nepochs
                print(f'[test_{iter_cnt}] epoch {epoch_id} loss = {te_nll}')

        if split < 1:
            return tr_loss , te_loss

        return tr_loss, tr_loss_list

    def plot_learning_with_epochs(self, **kwrd_losses):
        plt.close()
        for key, loss in kwrd_losses.items():
            loss_list = []
            for i, l in enumerate(loss):
                loss_list += l
            plt.plot(loss_list, label=f'{key}_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(self.dir / 'learning_curve.png')
        plt.close()

    def plot_cost(self, avg_cost):
        plt.close()
        plt.plot(avg_cost)
        plt.title('buffer top samples mean')
        plt.xlabel('iter')
        plt.ylabel('avg_cost')
        plt.savefig(self.dir / 'cost.png')
        plt.close()

    def save_checkpoint(self, iter_cnt, tr_losses, avg_cost):
        dict_to_save = dict(
            iter_cnt=iter_cnt,
            tr_losses=tr_losses,
            avg_cost=avg_cost,
            buffer=self.buffer,
            model_state=self.model.state_dict(),
            opt_state=self.opt.state_dict(),
        )
        self.load_ckpt_path = self.dir / 'checkpoint.tar'
        torch.save(dict_to_save, self.load_ckpt_path)

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.opt.load_state_dict(checkpoint['opt_state'])
        self.buffer = checkpoint['buffer']
        items = (checkpoint['iter_cnt'], checkpoint['tr_losses'], checkpoint['avg_cost'])
        return items

    def setup_model(self, seed):
        D = self.ndim
        self.model = MADE(D, self.hiddens, D * 3 * self.nr_mix, seed=seed,
                          natural_ordering=True)
        self.model.to(self.device)

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

        self.buffer = CacheBuffer(self.mode, self.goal, self.cut_off)

    def setup_model_state(self):
        # load the model or proceed without loading checkpoints
        if self.load_ckpt_path:
            items = self.load_checkpoint(self.load_ckpt_path)
        else:
            # collect samples using the random initial model (probably a bad initialization)
            self.model.eval()
            self.collect_samples(self.n_init_samples, uniform=True)
            # train the init model
            self.model.train()
            self.train(0, self.n_init_samples)

            _, xdata_ind = self.sample_model(4000)
            self.plt_hist2D(xdata_ind.to(self.cpu).data.numpy().astype('int'),
                            name='dist', prefix='training', suffix=f'0_after')
            items = (0, [], [])
            self.save_checkpoint(*items)
        return items


    def main(self, seed=10):

        self.set_seed(seed)

        self.setup_model(seed)
        iter_cnt, tr_losses, avg_cost = self.setup_model_state()

        while iter_cnt < self.niter:
            self.collect_samples(self.nsamples)
            avg_cost.append(self.buffer.mean)

            if iter_cnt == self.niter - 1 and self.full_training:
                tr_loss, tr_loss_list = self.train(iter_cnt + 1, 1000)
            else:
                tr_loss, tr_loss_list = self.train(iter_cnt + 1, self.nepochs)

            tr_losses.append(tr_loss_list)
            self.save_checkpoint(iter_cnt, tr_losses, avg_cost)
            if (iter_cnt + 1) % 10 == 0:
                _, xdata_ind = self.sample_model(4000)
                self.plt_hist2D(xdata_ind.to(self.cpu).data.numpy().astype('int'),
                                name='dist', prefix='training', suffix=f'{iter_cnt+1}_after')
            iter_cnt += 1

        self.plot_learning_with_epochs(training=tr_losses)
        self.plot_cost(avg_cost)

    def _compute_emprical_variation(self, samples):
        mean = samples.mean(0)
        cov_mat = ((samples - mean).T @ (samples - mean)) / samples.shape[0]
        var_sum = cov_mat.trace() / samples.shape[-1]
        return var_sum

    def _sample_model_for_eval(self, nsamples):
        vecs = self.input_vectors
        _, sample_ids = self.sample_model(nsamples)
        org_sample_list = []
        for i in range(self.ndim):
            vecs_arrs = np.stack([vecs[i]] * nsamples, axis=0)
            xi_values = vecs_arrs[np.arange(nsamples), sample_ids[:, i].int()]
            org_sample_list.append(xi_values)

        xsample = np.stack(org_sample_list, axis=-1)
        fval: np.ndarray = ackley_func(xsample)

        return xsample, sample_ids, fval

    def report_variation(self, nsamples):
        xsample, _, fval = self._sample_model_for_eval(nsamples)

        if self.mode == 'le':
            pos_samples = xsample[fval <= self.goal]
            pos_var = self._compute_emprical_variation(pos_samples)
        else:
            pos_samples = xsample[fval >= self.goal]
            pos_var = self._compute_emprical_variation(pos_samples)

        print(f'solution variation / dim = {pos_var:.6f}')

    def plt_hist2D(self, data: np.ndarray, ax=None, name='hist2D', **kwargs):
        fname = self.get_full_name(name,
                                   suffix=kwargs.pop('suffix', ''),
                                   prefix=kwargs.pop('prefix', ''))
        xvec, yvec = self.input_vectors
        if ax is None:
            ax = plt.gca()
        s = self.input_scale
        im = ax.hist2d(xvec[data[:, 0]], yvec[data[:, 1]], bins=100, cmap='binary',
                       range=np.array([(-s, s), (-s, s)]), **kwargs)
        plt.colorbar(im[-1], ax=ax)
        plt.savefig(self.dir / f'{fname}.png')
        plt.close()

    def report_accuracy(self, ntimes, nsamples):
        accuracy_list, times = [], []

        _, sample_ids, _ = self._sample_model_for_eval(nsamples)
        self.plt_hist2D(sample_ids.to(self.cpu).data.numpy().astype('int'))

        for iter_id in range(ntimes):
            s = time.time()
            xsample, sample_ids, fval = self._sample_model_for_eval(nsamples)
            if self.mode == 'le':
                accuracy_list.append((fval <= self.goal).sum(-1) / nsamples)
            else:
                accuracy_list.append((fval >= self.goal).sum(-1) / nsamples)
            times.append(time.time() - s)


        accuracy = np.array(accuracy_list, dtype='float32')
        times = np.array(times, dtype='float64')

        print(f'gen_time / sample = {1e3 * np.mean(times).astype("float") / nsamples:.3f} ms')
        print(f'accuracy_avg = {100 * np.mean(accuracy).astype("float"):.2f}, '
              f'accuracy_std = {100 * np.std(accuracy).astype("float"):.2f}')


    def check_solutions(self, ntimes=1, nsamples=1000, init_seed=10):
        self.set_seed(init_seed)

        self.setup_model(init_seed)
        self.setup_model_state()

        print('-------- REPORT --------')
        self.report_accuracy(ntimes, nsamples)
        self.report_variation(nsamples)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=10, help='random seed')
    parser.add_argument('-ckpt', '--ckpt', type=str, default='', help='checkpoint path')
    args = parser.parse_args()

    searcher = AutoReg2DSearch(
        ndim=2,
        goal_value=4,
        hidden_list=[20, 20, 20],
        mode='le',
        batch_size=16,
        nepochs=100,
        nsamples=5,
        n_init_samples=20,
        init_nepochs=50,
        cut_off=20,
        niter=100,
        lr=0.0005,
        beta=0,
        base_fn='normal',
        nr_mix=100,
        only_positive=False,
        full_training_last=True,
        load_ckpt_path=args.ckpt,
        input_scale=5.0,
    )
    searcher.main(args.seed)
    searcher.check_solutions(ntimes=1, nsamples=1000, init_seed=args.seed)