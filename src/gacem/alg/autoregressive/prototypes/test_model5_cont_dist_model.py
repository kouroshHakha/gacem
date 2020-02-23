"""
This script tests whether the model can learn the distribution of a given data,
both in and out of distribution samples are used for training

In this experiment changes the output distribution model interpretation to be continuous
just like pixelcnn++ we have a continuous latent variable v whose dist is modeled as
sum of k normal distributions.

data is always carried around as indices, and their actuall value is only used for visuallization
"""

import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from optnet.benchmarks.functions import ackley
from optnet.models.made import MADE
from utils.data import split_data
from utils.pdb import register_pdb_hook

register_pdb_hook()


class Learner:

    def __init__(self, hidden_list, nsample, *, batch_size=16, nepoch=1000, lr=1e-3,
                 base_fn='logistic', nr_mix=1):

        """
        base_fn: 'logistic', 'normal', 'uniform
        nr_mix: number of mixtures
        '"""
        self.nsample = nsample
        self.dim = 2
        self.bsize = batch_size
        self.hiddens = hidden_list
        self.nepoch = nepoch
        self.lr = lr
        self.beta = 0
        self.nr_mix = nr_mix
        self.base_fn = base_fn


        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cpu = torch.device('cpu')
        self.model = None
        self.opt = None

        x1 = np.linspace(start=-5, stop=5, num=100)
        x2 = np.linspace(start=-5, stop=5, num=100)
        self.input_vectors = [x1, x2]

    def plot_data(self, xdata, label=None, scatter_loc='', hist_loc='', ax=None):
        xvec, yvec = self.input_vectors
        x, y = xvec[xdata[:, 0]], xvec[xdata[:, 1]]

        if ax:
            ax.hist2d(x, y, bins=100, cmap='binary', range=np.array([(-5, 5), (-5, 5)]))
        else:
            plt.hist2d(x, y, bins=100, cmap='binary', range=np.array([(-5, 5), (-5, 5)]))

            plt.colorbar()
            plt.savefig(hist_loc if hist_loc else f'ref_figs/test_model_5_hist2D_samples.png')
            plt.close()

    def sample_data(self, nsample=100, goal=4):
        x1, x2 = self.input_vectors
        samples_ind = []
        while len(samples_ind) < nsample:
            x1_rand = np.random.choice(range(len(x1)))
            x2_rand = np.random.choice(range(len(x2)))

            sample = np.array([x1[x1_rand], x2[x2_rand]])
            fval = ackley(sample)

            if fval < goal:
                samples_ind.append(np.array([x1_rand, x2_rand]))

        data_ind = np.stack(samples_ind)
        samples = np.stack([x1[data_ind[:, 0]], x2[data_ind[:, 1]]], axis=-1)
        samples_norm, delta = self.normalize(samples)
        data = np.stack([samples_norm, data_ind], axis=1)
        return data, delta

    def normalize(self, xdata):
        """hard-coded max right now"""
        # TODO: removed hard-coded max-min later
        data_norm = 2 * (xdata + 5) / 10 - 1
        delta_x = self.input_vectors[0][1] - self.input_vectors[0][0]
        delta = 2 * delta_x / 10
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

    def get_probs(self, xin: torch.Tensor, delta: float, debug=False):
        """Given an input tensor (N, D) computes the probabilities across the cardinality space
        of each dimension. prob.shape = (N, D, K) where K is number of possible values for each
        dimension. Assume that xin is normalized to [-1,1], and delta is given."""
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
            center = torch.stack([xhat[:, i+D::D*3] for i in range(D)], dim=1)
            # normalize center between [-1,1] to cover all the space
            center = 2 * (center - center.min()) / (center.max() - center.min()) - 1
            log_delta = torch.stack([xhat[:, i+2*D::D*3] for i in range(D)], dim=1)
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
        cdfs = probs_left_edge * l_cond +  probs_right_edge * r_cond + probs_nonedge * n_cond

        probs = (coeffs * cdfs).sum(-1) / coeffs.sum(-1)

        if debug or torch.isnan(probs.min()):
            pdb.set_trace()

        return probs

    def get_nll(self, xin: torch.Tensor, delta: float, debug=False):
        """Given an input tensor computes the average negative likelihood of observing the inputs"""
        probs = self.get_probs(xin, delta)

        prob_x = probs.prod(-1)

        eps_tens = torch.tensor(1e-15)
        min_obj = -(prob_x + eps_tens).log10().mean(-1)

        if debug:
            pdb.set_trace()

        if torch.isnan(min_obj):
            print(min_obj)
            pdb.set_trace()
            self.get_probs(xin, delta, debug=True)

        return min_obj

    def sample_probs(self, probs: torch.Tensor, index: int):
        """Given a probability distribution tensor (shape = (N, D, K)) returns 1 sample from the
        distribution probs[:, index, :], the output is indices"""
        p = probs[:, index]
        sample = p.multinomial(num_samples=1).squeeze()
        return sample

    def sample_model(self, nsamples: int, delta: float):
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
                probs = self.get_probs(xin, delta)
                xi_ind = self.sample_probs(probs, i)  # ith x index
                xsample[0, i] = torch.tensor(input_vec_norm[i][xi_ind])
                xsample_ind[0, i] = xi_ind
            samples.append(xsample.squeeze())
            samples_ind.append(xsample_ind.squeeze())

        samples = torch.stack(samples, dim=0)
        samples_ind = torch.stack(samples_ind, dim=0)
        return samples, samples_ind

    def plot_learning(self, training_loss, test_loss):
        plt.close()
        plt.plot(training_loss, label='training_loss', color='g')
        plt.plot(test_loss, label='test_loss', color='b')
        plt.legend()
        plt.xlabel('nepoch')
        plt.ylabel('negative likelihood')
        plt.savefig('figs/test_model_5_learning_curve.png')
        plt.close()

    def main(self, seed=10):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        data, delta = self.sample_data(self.nsample)
        self.plot_data(data[:, 1, :].astype('int'))
        xtr, xte = split_data(data)

        D = self.dim
        self.model: nn.Module = MADE(D, self.hiddens, D * 3 * self.nr_mix, seed=seed)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=0)
        self.lr_sch = StepLR(self.opt, step_size=50, gamma=0.9)
        B = self.bsize
        N, D, _ = xtr.shape
        # per epoch
        tr_nll, te_nll = [], []

        # _, samples_ind = self.sample_model(4000, delta)
        # samples_ind = samples_ind.to(self.cpu).data.numpy().astype('int')
        # self.plot_data(samples_ind, scatter_loc='figs/test_model_5_scatter.png',
        #                hist_loc='figs/test_model_5_init_hist2D.png')
        # return
        plt.figure(figsize=(15, 8))
        i = 0
        for epoch_id in range(self.nepoch):
            nstep = N // B
            # per batch
            tr_nll_per_b, te_nll_per_b = 0, 0
            for step in range(nstep):
                self.model.train()
                xb = xtr[step * B: step * B + B]
                xb_tens = torch.from_numpy(xb).to(self.device)

                xin = xb_tens[:, 0, :]
                loss = self.get_nll(xin, delta, debug=False)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.lr_sch.step(epoch_id)

                tr_nll_per_b += loss.to(self.cpu).item() / nstep

            self.model.eval()
            xte_tens = torch.from_numpy(xte).to(self.device)
            xin_te = xte_tens[:, 0, :]
            te_loss = self.get_nll(xin_te, delta)
            te_nll.append(te_loss)

            print(f'epoch = {epoch_id}, tr_nll = {tr_nll_per_b}')
            print(f'epoch = {epoch_id}, te_nll = {te_loss}')
            tr_nll.append(tr_nll_per_b)

            if (epoch_id + 1) % 20 == 0 and 200 <=epoch_id <= 300:
                _, samples_ind = self.sample_model(1000, delta)
                samples_ind = samples_ind.to(self.cpu).data.numpy().astype('int')

                ax = plt.subplot(1, 5, i + 1, adjustable='box', aspect=1)
                self.plot_data(samples_ind, scatter_loc='figs/test_model_5_sub_scatter.png',
                               hist_loc='figs/test_model_5_sub_hist2D.png', ax=ax)
                plt.tight_layout()
                i += 1
        plt.savefig('figs/test_model_5_sub_hist2D.png')

        self.plot_learning(tr_nll, te_nll)
        # pdb.set_trace()

        _, samples_ind = self.sample_model(1000, delta)
        samples_ind = samples_ind.to(self.cpu).data.numpy().astype('int')
        self.plot_data(samples_ind, scatter_loc='figs/test_model_5_scatter.png',
                  hist_loc='figs/test_model_5_hist2D.png')


if __name__ == '__main__':
    # arr, weights = sample_data(10000)
    # plot_data(arr[:, 0, :], label=weights)
    learner = Learner(
        hidden_list=[20, 20, 20],
        nsample=1000,
        batch_size=128,
        nepoch=100,
        lr=0.001,
        base_fn='uniform',
        nr_mix=100,
    )
    learner.main(20)