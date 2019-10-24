"""
This script tests whether the model can learn the distribution of a given data,
Only samples from the distribution are used for leanring. (i.e. no out of distribution learning)

I used this script to debug the horizontal line issue. Basically the only degree of freedom that x1
(aka independent variable) had was through the bias values of the last layer.
the function is fi = exp(bi) / sum_i(exp(bi)), grad_fi_b = k1 for i position and k2 for the rest.
This gradient is will not allow other bias values to adapt quickly to adjust distribution of x1.
That's why we got that uniform behavior on x1 and the learning rate became so slow after the first
couple of iterations.
"""



import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb

from ackley import ackley_func
from made import MADE
from utils.data import split_data
from utils.pdb import register_pdb_hook

register_pdb_hook()


def sample_data(nsample=100, goal=4):
    x1 = np.linspace(start=-5, stop=5, num=100)
    x2 = np.linspace(start=-5, stop=5, num=100)

    samples, samples_ind = [], []
    while len(samples) < nsample:
        x1_rand = np.random.choice(range(len(x1)))
        x2_rand = np.random.choice(range(len(x2)))

        sample = np.array([x1[x1_rand], x2[x2_rand]])
        fval = ackley_func(sample)

        if fval < goal:
            samples_ind.append(np.array([x1_rand, x2_rand]))
            samples.append(sample)

    samples_arr = np.stack(samples)
    samples_ind_arr = np.stack(samples_ind)
    data = np.stack([samples_arr, samples_ind_arr], axis=1)
    return data

def plot_data(xdata, scatter_loc='', hist_loc='', ax=None):
    x, y = xdata[:, 0], xdata[:, 1]
    # plt.scatter(x, y)
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    # plt.savefig(scatter_loc if scatter_loc else f'figs/test_model_ref_samples.png')
    # plt.close()

    if ax:
        ax.hist2d(x, y, bins=100, cmap='binary', range=np.array([(-5, 5), (-5, 5)]))
    else:
        plt.hist2d(x, y, bins=100, cmap='binary', range=np.array([(-5, 5), (-5, 5)]))
        plt.colorbar()
        plt.savefig(hist_loc if hist_loc else f'figs/test_model_ref_hist.png')
        plt.close()


class Learner:

    def __init__(self, hidden_list, nsample, batch_size=16, nepoch=1000, lr=1e-3):
        self.nsample = nsample
        self.dim = 2
        self.bsize = batch_size
        self.hiddens = hidden_list
        self.nepoch = nepoch
        self.lr = lr


        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cpu = torch.device('cpu')
        self.model = None
        self.opt = None

    def get_probs(self, xin: torch.Tensor):
        """Given an input tensor (N, D) computes the probabilities across the cardinality space
        of each dimension. prob.shape = (N, D, K) where K is number of possible values for each
        dimension"""
        xin = xin.to(self.device)

        xhat = self.model(xin)
        D = self.dim
        logits = torch.stack([xhat[:, i::D] for i in range(D)], dim=1)

        probs = logits.softmax(dim=-1)

        return probs

    def get_nll(self, xin: torch.Tensor, xin_ind: torch.Tensor, weights: torch.Tensor = None):
        """Given an input tensor and the corresponding index tensor (both shapes = (N,D)) computes
        the average negative likelihood of observing the inputs"""
        xin_ind = xin_ind.to(self.device)
        probs = self.get_probs(xin)

        D = self.dim
        N = xin.shape[0]
        batch_ind = np.stack([range(N)] * D, axis=-1)
        var_ind = np.stack([range(D)] * N, axis=0)
        prob_in = probs[batch_ind, var_ind, xin_ind]

        nll_samples = - prob_in.log2().sum(-1)

        # multiply each nll by the corresponding weight and take the mean
        if weights is None:
            weighted_nll = nll_samples.mean(-1)
        else:
            weighted_nll = (nll_samples * weights).sum(-1)

        return weighted_nll

    def sample_probs(self, probs: torch.Tensor, index: int):
        """Given a probability distribution tensor (shape = (N, D, K)) returns 1 sample from the
        distribution probs[:, index, :], the output is indices"""
        p = probs[:, index, :]
        sample = p.multinomial(num_samples=1).squeeze()
        return sample

    def sample_model(self, nsamples: int, *input_vectors):

        D = self.dim
        xin = torch.zeros(nsamples, D)
        xin_ind = torch.zeros(nsamples, D, dtype=torch.int64)
        for i in range(D):
            probs = self.get_probs(xin)
            xi_ind = self.sample_probs(probs, i)  # ith x index
            xin_ind[:, i] = xi_ind
            xin[:, i] = torch.from_numpy(input_vectors[i][xi_ind])
        return xin, xin_ind

    def plot_learning(self, training_loss, test_loss):
        plt.close()
        plt.plot(training_loss, label='training_loss', color='g')
        plt.plot(test_loss, label='test_loss', color='b')
        plt.legend()
        plt.xlabel('nepoch')
        plt.ylabel('negative likelihood')
        plt.savefig('figs/test_model_learning_curve.png')
        plt.close()

    def main(self, seed=10):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        data = sample_data(self.nsample)
        plot_data(data[:, 0, :])
        xtr, xte = split_data(data)

        D = self.dim
        self.model: nn.Module = MADE(D, self.hiddens, D * 100, seed=seed, natural_ordering=True)
        self.model.to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), self.lr)
        B = self.bsize
        N, D,  _ = xtr.shape
        # per epoch
        tr_nll, te_nll = [], []
        for epoch_id in range(self.nepoch):
            nstep = N // B
            # per batch
            tr_nll_per_b, te_nll_per_b = 0, 0
            for step in range(nstep):
                self.model.train()
                xb = xtr[step * B: step * B + B]
                xb_tens = torch.from_numpy(xb).to(self.device)

                xin = xb_tens[:, 0, :]
                xin_ind = xb_tens[:, 1, :].long()
                loss = self.get_nll(xin, xin_ind)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                tr_nll_per_b += loss.to(self.cpu).item() / nstep

            self.model.eval()
            xte_tens = torch.from_numpy(xte).to(self.device)
            xin_te = xte_tens[:, 0, :]
            xin_ind_te = xte_tens[:, 1, :].long()
            te_loss = self.get_nll(xin_te, xin_ind_te)
            te_nll.append(te_loss)

            print(f'epoch = {epoch_id}, tr_nll = {tr_nll_per_b}')
            print(f'epoch = {epoch_id}, te_nll = {te_loss}')
            tr_nll.append(tr_nll_per_b)

        #     x1 = np.linspace(start=-5, stop=5, num=100)
        #     x2 = np.linspace(start=-5, stop=5, num=100)
        #     samples, _ = self.sample_model(self.nsample, x1, x2)
        #     samples = samples.to(self.cpu).data.numpy()
        #
        #     ax = plt.subplot(5, 5, epoch_id + 1)
        #     plot_data(samples, scatter_loc='figs/test_model_scatter.png',
        #               hist_loc='figs/test_model_hist2D.png', ax=ax)
        #     plt.tight_layout()
        # plt.savefig('figs/test_model_hist2D.png')
        # pdb.set_trace()

        self.plot_learning(tr_nll, te_nll)

        x1 = np.linspace(start=-5, stop=5, num=100)
        x2 = np.linspace(start=-5, stop=5, num=100)
        samples, _ = self.sample_model(10000, x1, x2)
        samples = samples.to(self.cpu).data.numpy()
        plot_data(samples, scatter_loc='figs/test_model_scatter.png',
                  hist_loc='figs/test_model_hist2D.png')


if __name__ == '__main__':
    # arr = sample_data(1000)
    # plot_data(arr[:, 0, :])
    learner = Learner(
        hidden_list=[20, 20, 20],
        nsample=50,
        batch_size=8,
        nepoch=150,
        lr=0.0001,
    )
    learner.main()