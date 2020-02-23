"""
Second implementation.

This take into account the important sampling, and dependency of objective on theta.
like how p_B(x) depends on \theta.
We make buffer not include current distribution.
"""
import inspect
import math
import pdb
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ruamel_yaml as yaml
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import transforms
from torch.utils.tensorboard import SummaryWriter

from gacem.benchmarks.functions import ackley, show_weight_on_all
from gacem.data.buffer import CacheBuffer
from gacem.models.made import MADE
from utils.pdb import register_pdb_hook

register_pdb_hook()

class AutoReg2DSearch:

    def __init__(self, goal_value, hidden_list, mode='le', batch_size=16, cut_off=20,
                 nepochs=1, nsamples=1, n_init_samples=100, niter=1000, lr=1e-3, beta: float = 0,
                 init_nepochs=1, base_fn='logistic', nr_mix=1, only_positive=False,
                 full_training_last=False):

        l_args, _, _, values = inspect.getargvalues(inspect.currentframe())
        params = dict(zip(l_args, [values[i] for i in l_args]))
        self.unique_name = time.strftime('%Y%m%d%H%M%S')
        self.dir = Path(f'data/search_fig_{self.unique_name}')
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / 'params.yaml', 'w') as f:
            yaml.dump(params, f)

        self.dim = 2
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


        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cpu = torch.device('cpu')
        self.model = None
        self.opt = None

        self.writer = SummaryWriter()

        # hacky version of passing input vectors around
        x1 = np.linspace(start=-1, stop=1, num=100)
        x2 = np.linspace(start=-1, stop=1, num=100)
        self.input_vectors = [x1, x2]


        # TODO: remove this hacky way of keeping track of delta
        # self.norm_delta = None
        self.norm_delta = x1[-1] - x1[-2]

    def sample_data(self, nsample=100, goal=4):
        """ sample randomly (i.e. not on policy)"""
        x1, x2 = self.input_vectors
        samples_ind = []
        while len(samples_ind) < nsample:
            x1_rand = np.random.choice(range(len(x1)))
            x2_rand = np.random.choice(range(len(x2)))
            samples_ind.append(np.array([x1_rand, x2_rand]))

        data_ind = np.stack(samples_ind)
        samples = np.stack([x1[data_ind[:, 0]], x2[data_ind[:, 1]]], axis=-1)
        # samples_norm, _ = self.normalize(samples)
        # return samples_norm, data_ind
        return samples, data_ind

    def normalize(self, xdata):
        """hard-coded max right now"""
        # TODO: removed hard-coded max-min later
        data_norm = 2 * (xdata + 5) / 10 - 1
        delta_x = self.input_vectors[0][1] - self.input_vectors[0][0]
        delta = 2 * delta_x / 10
        self.norm_delta = delta

        return data_norm, delta

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
        delta = self.norm_delta
        if delta is None:
            raise ValueError('self.norm_delta is not set yet!')

        xin = xin.to(self.device)
        xhat = self.model(xin)
        D = self.dim
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
            # put a bottom on the value of output so that it does not blow up
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

        prob_x = probs.prod(-1)
        eps_tens = torch.tensor(1e-15)

        if weights is None:
            min_obj = -(prob_x + eps_tens).log10().mean(-1)
        else:
            pos_ind = (weights > 0).float()
            neg_ind = torch.tensor(1) - pos_ind

            logp_vec = (prob_x + eps_tens).log10()

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
                min_obj = pos_main_obj + pos_ent_obj
            else:
                min_obj = pos_main_obj + neg_main_obj + pos_ent_obj + neg_ent_obj

            if debug:
                for w, p in zip(weights, prob_x):
                    print(f'w = {w:10.4}, prob = {p:10.4}')
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

        D = self.dim
        # x1, x2 = self.input_vectors
        # input_vec_norm = []
        # for x in [x1, x2]:
        #     norm, _ = self.normalize(x)
        #     input_vec_norm.append(norm)

        samples = []
        samples_ind = []
        for k in range(nsamples):
            xsample = torch.zeros(1, D)
            xsample_ind = torch.zeros(1, D)
            for i in range(D):
                # N = len(input_vec_norm[i])
                N = len(self.input_vectors[i])
                xin = torch.zeros(N, D)
                if i > 1:
                    xin[:, :i] = torch.stack([xsample.squeeze()] * N)
                # xin[:, i] = torch.from_numpy(input_vec_norm[i])
                xin[:, i] = torch.from_numpy(self.input_vectors[i])
                probs = self.get_probs(xin)
                xi_ind = self.sample_probs(probs, i)  # ith x index
                # xsample[0, i] = torch.tensor(input_vec_norm[i][xi_ind])
                xsample[0, i] = torch.tensor(self.input_vectors[i][xi_ind])
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

    def plt_hist2D(self, data: np.ndarray, ax=None, **kwargs):
        xvec, yvec = self.input_vectors
        x, y = xvec[data[:, 0]] * 5, xvec[data[:, 1]] * 5
        if ax is None:
            ax = plt.gca()
        im = ax.hist2d(x, y, bins=100, cmap='binary', range=np.array([(-5, 5), (-5, 5)]), **kwargs)
        plt.colorbar(im[-1], ax=ax)

    def plt_weight_heatmap(self, ax=None, **kwargs):
        x1, x2 = self.input_vectors
        show_weight_on_all(x1 * 5, x2 * 5, self.goal, self.buffer.mean, self.mode, ax=ax, **kwargs)

    def viz(self, data, prefix='', suffix=''):
        """This draws both the histogram and weight heatmap."""
        ax = plt.gca()
        self.plt_hist2D(data, ax=ax)
        self.plt_weight_heatmap(ax=ax, alpha=0.5, cmap='viridis')
        name = self.get_full_name('samples', prefix, suffix)
        plt.savefig(self.dir / f'{name}.png')
        plt.close()


    def run_epoch(self, data: np.ndarray, weights: np.ndarray, mode='train',
                  debug=False):
        self.model.train() if mode == 'train' else self.model.eval()

        N, D,  _ = data.shape

        assert N != 0, 'no data found'

        B = max(self.bsize, 2 ** math.floor(math.log2(N / 4))) if mode == 'train' else N
        nstep = N // B if mode == 'train' else 1

        nll_per_b = 0

        # if mode == 'train':
        #     xdb = torch.from_numpy(data)
        #     wdb = torch.from_numpy(weights)
        #     loss = self.get_nll(xdb[:, 0, :], weights=wdb, debug=True)

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

        # if mode == 'train':
        #     loss = self.get_nll(xdb[:, 0, :], weights=wdb, debug=True)

        return nll_per_b

    def collect_samples(self, n_samples, uniform=False):

        x1, x2 = self.input_vectors
        n_collected = 0
        new_samples = []
        while n_collected < n_samples:
            if uniform:
                _, xnew_id_np = self.sample_data(1, self.goal)
                xnew_id_np = xnew_id_np.astype('int')
            else:
                _, xnew_ind = self.sample_model(1)
                xnew_id_np = xnew_ind.to(self.cpu).data.numpy().astype('int')

            xsample = np.array([[x1[xnew_id_np[0, 0]], x2[xnew_id_np[0, 1]]]]).astype('float32')
            # simulate and compute the adjustment weights
            fval = ackley(xsample * 5)

            if xsample not in self.buffer:
                self.buffer.add_samples(xsample, xnew_id_np, fval)
                new_samples.append(xsample)
                n_collected += 1
            else:
                print(f'item {xsample} already exists!')

        return new_samples


    def train(self, iter_cnt: int, nepochs: int, split=1.0):
        # treat the sampled data as a static data set and take some gradient steps on it
        xtr, xte, wtr, wte = self.buffer.draw_tr_te_ds(split=split)

        # plotting
        self.viz(xtr[:,1, :].astype('int'), 'training', f'{iter_cnt}_before')
        # self.viz(xte[:,1, :].astype('int'), 'test', f'{iter_cnt}_before')
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

    def plot_learning(self, **kwrd_losses):
        plt.close()
        for key, loss in kwrd_losses.items():
            plt.plot(loss, label=f'{key}_loss')
        plt.legend()
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.savefig(self.dir / 'learning_curve.png')
        plt.close()

    def plot_learning_with_epochs(self, **kwrd_losses):
        plt.close()
        ax = plt.gca()
        fig = plt.gcf()
        data_to_axis = ax.transData
        shift_trans = transforms.ScaledTranslation(-5/72, 0, fig.dpi_scale_trans)
        trans = data_to_axis + shift_trans

        for key, loss in kwrd_losses.items():
            init_index = 0
            loss_list = []
            for i, l in enumerate(loss):
                # plt.axvline(x=init_index, ls='--', color='k')
                # plt.annotate(f'iter = {i+1}', (init_index, 0.1), rotation='vertical',
                #              xycoords=(trans, 'axes fraction'), size='xx-small')
                # init_index += len(l)
                loss_list += l
            plt.plot(loss_list, label=f'{key}_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(self.dir / 'learning_curve.png')
        plt.close()

    def main(self, seed=10):

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        D = self.dim
        self.model: nn.Module = MADE(D, self.hiddens, D * 3 * self.nr_mix, seed=seed,
                                     natural_ordering=True)
        self.model.to(self.device)

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

        self.buffer = CacheBuffer(self.mode, self.goal, self.cut_off)

        # collect samples using the random initial model (probably a bad initialization)
        self.model.eval()
        _, xdata_ind = self.sample_model(200)
        self.plt_hist2D(xdata_ind.to(self.cpu).data.numpy().astype('int'))
        name = self.get_full_name('dist', suffix='init')
        plt.savefig(self.dir / f'{name}.png')
        plt.close()

        self.collect_samples(self.n_init_samples, uniform=True)
        # train the init model
        self.model.train()
        self.train(0, self.init_nepochs)
        # visualize empirical distribution after training
        self.model.eval()
        _, xdata_ind = self.sample_model(200)

        self.plt_hist2D(xdata_ind.to(self.cpu).data.numpy().astype('int'))
        name = self.get_full_name('dist', prefix='training', suffix='0_after')
        plt.savefig(self.dir / f'{name}.png')
        plt.close()

        tr_losses = []
        for iter_cnt in range(self.niter):
            self.collect_samples(self.nsamples)
            if iter_cnt == self.niter - 1 and self.full_training:
                tr_loss, tr_loss_list = self.train(iter_cnt + 1, 1000)
            else:
                tr_loss, tr_loss_list = self.train(iter_cnt + 1, self.nepochs)

            tr_losses.append(tr_loss_list)


            if (iter_cnt + 1) % 10 == 0:
                _, xdata_ind = self.sample_model(200)
                self.plt_hist2D(xdata_ind.to(self.cpu).data.numpy().astype('int'))
                name = self.get_full_name('dist', prefix='training', suffix=f'{iter_cnt+1}_after')
                plt.savefig(self.dir / f'{name}.png')
                plt.close()


        self.plot_learning_with_epochs(training=tr_losses)


if __name__ == '__main__':
    searcher = AutoReg2DSearch(
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
        beta=0.5,
        base_fn='normal',
        nr_mix=100,
        only_positive=False,
        full_training_last=True
    )
    searcher.main(10)