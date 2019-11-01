"""
Second implementation.

This take into account the important sampling, and dependency of objective on theta.
like how p_B(x) depends on \theta.
We make buffer not include current distribution.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import pdb

from ackley import ackley_func, show_weight_on_all
from made import MADE
from buffer import CacheBuffer
from utils.pdb import register_pdb_hook

register_pdb_hook()

class AutoReg2DSearch:

    def __init__(self, goal_value, hidden_list, mode='le', batch_size=16, cut_off=20,
                 nepochs=1, nsamples=1, n_init_samples=100, niter=1000, lr=1e-3, beta=0,
                 base_fn='logistic', nr_mix=1):

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
        self.cut_off = cut_off
        self.beta = beta
        self.nr_mix = nr_mix
        self.base_fn = base_fn

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cpu = torch.device('cpu')
        self.model = None
        self.opt = None

        # hacky version of passing input vectors around
        x1 = np.linspace(start=-5, stop=5, num=100)
        x2 = np.linspace(start=-5, stop=5, num=100)
        self.input_vectors = [x1, x2]


        # TODO: remove this hacky way of keeping track of delta
        self.norm_delta = None

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
        l = (x-a)/(b-a)
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
        log_coeffs = torch.stack([xhat[:, i::D*3] for i in range(D)], dim=1)
        coeffs = log_coeffs.exp()
        xb = xin[..., None] + torch.zeros(coeffs.shape)

        if self.base_fn in ['logistic', 'normal']:
            means = torch.stack([xhat[:, i+D::D*3] for i in range(D)], dim=1)
            log_sigma = torch.stack([xhat[:, i+2*D::D*3] for i in range(D)], dim=1)
            sigma = log_sigma.exp()

            if self.base_fn == 'logistic':
                plus_cdf = self.cdf_logistic(xb + delta / 2, means, sigma)
                minus_cdf = self.cdf_logistic(xb - delta / 2, means, sigma)
            else:
                plus_cdf = self.cdf_normal(xb + delta / 2, means, sigma)
                minus_cdf = self.cdf_normal(xb - delta / 2, means, sigma)
        elif self.base_fn == 'uniform':
            a = torch.stack([xhat[:, i+D::D*3] for i in range(D)], dim=1)
            log_delta = torch.stack([xhat[:, i+2*D::D*3] for i in range(D)], dim=1)
            bdelta = log_delta.exp()
            b = a + bdelta
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
        cdfs = probs_left_edge * l_cond +  probs_right_edge * r_cond + probs_nonedge * n_cond

        probs = (coeffs * cdfs).sum(-1) / coeffs.sum(-1)

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
            logp_vec = (prob_x + eps_tens).log10()

            ent_term = self.buffer.size * (self.beta * (torch.tensor(1) + logp_vec) * prob_x).data
            obj_term  = - self.buffer.size * (weights * prob_x).data

            coeff = (ent_term + obj_term) * pos_ind

            min_obj_vec = coeff * logp_vec

            npos = pos_ind.sum(-1)
            npos = 1 if npos == 0 else npos
            min_obj = min_obj_vec.sum(-1) / npos

            if debug:
                for w, p in zip(weights, prob_x):
                    print(f'w = {w:10.4}, prob = {p:10.4}')
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
        x1, x2 = self.input_vectors
        input_vec_norm = []
        for x in [x1, x2]:
            norm, _ = self.normalize(x)
            input_vec_norm.append(norm)

        samples = []
        samples_ind = []
        for k in range(nsamples):
            xsample = torch.zeros(1, D)
            xsample_ind = torch.zeros(1, D)
            for i in range(D):
                N = len(input_vec_norm[i])
                xin = torch.zeros(N, D)
                if i > 1:
                    xin[:, :i] = torch.stack([xsample.squeeze()] * N)
                xin[:, i] = torch.from_numpy(input_vec_norm[i])
                probs = self.get_probs(xin)
                xi_ind = self.sample_probs(probs, i)  # ith x index
                xsample[0, i] = torch.tensor(input_vec_norm[i][xi_ind])
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
        x, y = xvec[data[:, 0]], xvec[data[:, 1]]
        if ax is None:
            ax = plt.gca()
        im = ax.hist2d(x, y, bins=100, cmap='binary', range=np.array([(-5, 5), (-5, 5)]), **kwargs)
        plt.colorbar(im[-1], ax=ax)

    def plt_weight_heatmap(self, ax=None, **kwargs):
        x1, x2 = self.input_vectors
        show_weight_on_all(x1, x2, self.goal, self.buffer.mean, self.mode, ax=ax, **kwargs)

    def viz(self, data, prefix='', suffix=''):
        """This draws both the histogram and weight heatmap."""
        ax = plt.gca()
        self.plt_hist2D(data, ax=ax)
        self.plt_weight_heatmap(ax=ax, alpha=0.5, cmap='viridis')
        name = self.get_full_name('samples', prefix, suffix)
        plt.savefig(f'search_figs3/{name}.png')
        plt.close()


    def run_epoch(self, data: np.ndarray, weights: np.ndarray, mode='train',
                  debug=False):
        self.model.train() if mode == 'train' else self.model.eval()

        N, D,  _ = data.shape
        B = self.bsize if mode == 'train' else N
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
                self.opt.step()
            nll_per_b += loss.to(self.cpu).item() / nstep

        # if mode == 'train':
        #     loss = self.get_nll(xdb[:, 0, :], weights=wdb, debug=True)

        return nll_per_b

    def collect_samples(self, n_samples):

        x1, x2 = self.input_vectors
        n_collected = 0
        while n_collected < n_samples:
            xnew, xnew_ind = self.sample_model(1)
            xnew_np = xnew.to(self.cpu).data.numpy()
            xnew_id_np = xnew_ind.to(self.cpu).data.numpy().astype('int')

            xsample = np.array([[x1[xnew_id_np[0, 0]], x2[xnew_id_np[0, 1]]]])
            # simulate and compute the adjustment weights
            fval = ackley_func(xsample)

            if xnew_np not in self.buffer:
                self.buffer.add_samples(xnew_np, xnew_id_np, fval)
                n_collected += 1
            else:
                print(f'item {xnew_np} already exists!')


    def train(self, iter_cnt: int):
        # treat the sampled data as a static data set and take some gradient steps on it
        xtr, xte, wtr, wte = self.buffer.draw_tr_te_ds()
        # plotting
        self.viz(xtr[:,1, :].astype('int'), 'training', f'{iter_cnt}_before')
        self.viz(xte[:,1, :].astype('int'), 'test', f'{iter_cnt}_before')
        # per epoch
        print('-'*50)
        tr_loss, te_loss = 0, 0
        for epoch_id in range(self.nepochs):
            tr_nll = self.run_epoch(xtr, wtr, mode='train')
            te_nll = self.run_epoch(xte, wte, mode='test')
            tr_loss += tr_nll / self.nepochs
            te_loss += te_nll / self.nepochs

            print(f'[train_{iter_cnt}] epoch {epoch_id} loss = {tr_nll}')
            print(f'[test_{iter_cnt}] epoch {epoch_id} loss = {te_nll}')

        return tr_loss, te_loss

    def plot_learning(self, training_loss, test_loss):
        plt.close()
        plt.plot(training_loss, label='training_loss', color='g')
        plt.plot(test_loss, label='test_loss', color='b')
        plt.legend()
        plt.xlabel('nepoch')
        plt.ylabel('negative likelihood')
        plt.savefig('search_figs3/learning_curve.png')
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
        self.collect_samples(self.n_init_samples)
        # train the init model
        self.model.train()
        self.train(0)
        # visualize empirical distribution after training
        self.model.eval()
        # _, xdata_ind = self.sample_model(200)

        # self.plt_hist2D(xdata_ind.to(self.cpu).data.numpy().astype('int'))
        # name = self.get_full_name('dist', prefix='training', suffix='0_after')
        # plt.savefig(f'search_figs3/{name}.png')
        # plt.close()
        pdb.set_trace()

        tr_losses, te_losses = [], []
        for iter_cnt in range(self.niter):
            self.collect_samples(self.nsamples)
            tr_loss, te_loss = self.train(iter_cnt + 1)
            # _, xdata_ind = self.sample_model(200)
            # self.plt_hist2D(xdata_ind.to(self.cpu).data.numpy().astype('int'))
            # name = self.get_full_name('dist', prefix='training', suffix=f'{iter_cnt+1}_after')
            # plt.savefig(f'search_figs3/{name}.png')
            # plt.close()

            tr_losses.append(tr_loss)
            te_losses.append(tr_loss)
            # pdb.set_trace()

            # if iter_cnt % (self.viz_rate - 1) == 0:
            #     # print evaluation
            #     self.model.eval()
            #     print(self.get_nll(xp_ref, xind_ref).item())
            #     xdata, _ = self.sample_model(2000, x1_sp, x2_sp)
            #     self.viz(xdata.to(self.cpu).data.numpy(), iter_cnt)

        self.plot_learning(tr_losses, te_losses)


if __name__ == '__main__':
    searcher = AutoReg2DSearch(
        goal_value=4,
        hidden_list=[20, 20, 20],
        mode='le',
        batch_size=8,
        nepochs=1,
        nsamples=5,
        n_init_samples=20,
        cut_off=10,
        niter=50,
        lr=0.001,
        beta=2,
        base_fn='normal',
        nr_mix=10,
    )
    searcher.main(10)