"""
This script tests whether the model can learn the distribution of a given data,
both in and out of distribution samples are used for training

With this experiment I understood that only minimizing neg_ll in the objective function doesn't
perform as expectation, because nn will just assign low probability to everything and only
handful of points (after some time it's gonna be only 1 point) will have high probabilty.
So it won't work if you don't use pos_nll. This means that using continuous weights, even if we
don't have a weight of 1 we should assign positive weight to the samples that perform above the
average.
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
    weights = []
    while len(samples) < nsample:
        x1_rand = np.random.choice(range(len(x1)))
        x2_rand = np.random.choice(range(len(x2)))

        sample = np.array([x1[x1_rand], x2[x2_rand]])
        fval = ackley_func(sample)

        if fval < goal:
            weights.append(1)
        else:
            weights.append(-1)

        samples_ind.append(np.array([x1_rand, x2_rand]))
        samples.append(sample)

    samples_arr = np.stack(samples)
    samples_ind_arr = np.stack(samples_ind)
    data = np.stack([samples_arr, samples_ind_arr], axis=1)
    weights = np.array(weights)
    return data, weights

def plot_data(xdata, label=None, scatter_loc='', hist_loc='', ax=None):
    x, y = xdata[:, 0], xdata[:, 1]
    # if label is not None:
    #     plt.scatter(x[label == 1], y[label == 1], color='r', label='in dist')
    #     plt.scatter(x[label == -1], y[label == -1], color='b', label='out dist')
    #     plt.legend()
    # else:
    #     plt.scatter(x, y, color='r', label='in dist')
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    # plt.savefig(scatter_loc if scatter_loc else f'ref_figs/test_model2_scatter_samples.png')
    # plt.close()

    if ax:
        ax.hist2d(x, y, bins=100, cmap='binary', range=np.array([(-5, 5), (-5, 5)]))
    else:
        if label is not None:
            plt.hist2d(x[label == 1], y[label == 1], bins=100, cmap='binary', range=np.array([(-5, 5),
                                                                                           (-5, 5)]))
        else:
            plt.hist2d(x, y, bins=100, cmap='binary', range=np.array([(-5, 5), (-5, 5)]))

        plt.colorbar()
        plt.savefig(hist_loc if hist_loc else f'ref_figs/test_model2_hist2D_samples.png')
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
        self.model.eval()
        xin = xin.to(self.device)

        xhat = self.model(xin)
        D = self.dim
        logits = torch.stack([xhat[:, i::D] for i in range(D)], dim=1)

        probs = logits.softmax(dim=-1)

        return probs

    def get_nll(self, xin: torch.Tensor, xin_ind: torch.Tensor, weights: torch.Tensor = None,
                debug=False):
        """Given an input tensor and the corresponding index tensor (both shapes = (N,D)) computes
        the average negative likelihood of observing the inputs"""
        xin_ind = xin_ind.to(self.device)
        probs = self.get_probs(xin)

        D = self.dim
        N = xin.shape[0]
        batch_ind = np.stack([range(N)] * D, axis=-1)
        var_ind = np.stack([range(D)] * N, axis=0)
        prob_in = probs[batch_ind, var_ind, xin_ind]

        if weights is None:
            nll_samples = - prob_in.log2().sum(dim=-1)
            weighted_nll = nll_samples.mean(dim=-1)
            return weighted_nll
        else:
            # multiply each nll by the corresponding weight and take the mean
            # prob_x = prob_in.prod(-1)
            # pos_ind = (weights > 0)
            # neg_ind = ~pos_ind
            # prob_obs = prob_x ** pos_ind.float()
            # prob_obs_b = (torch.tensor(1) - prob_x) ** neg_ind.float()
            # pos_obj = (prob_obs.log2() * weights.abs()).sum(-1) / pos_ind.sum(-1)
            # neg_obj = (prob_obs_b.log2() * weights.abs()).sum(-1) / neg_ind.sum(-1)
            # ll = pos_obj + neg_obj
            # nll = -ll
            #
            # return nll

            eps_tens = torch.tensor(1e-15)
            prob_x = prob_in.prod(-1)
            pos_ind = (weights > 0).float()
            neg_ind = torch.tensor(1) - pos_ind
            logp_vec = (prob_x + eps_tens).log10()

            npos = pos_ind.sum(-1)

            if npos > 0:
                pos_ll = (logp_vec * weights.abs() * pos_ind).sum(-1) / npos
            else:
                pos_ll = (logp_vec * weights.abs() * pos_ind).sum(-1)

            nneg = neg_ind.sum(-1)

            if nneg > 0:
                neg_ll = (logp_vec * weights.abs() * neg_ind).sum(-1) / nneg
            else:
                neg_ll = (logp_vec * weights.abs() * neg_ind).sum(-1)

            # min_obj =  -pos_ll  + neg_ll
            min_obj =  -pos_ll
            # min_obj =  neg_ll

            if debug:
                pdb.set_trace()

            if torch.isnan(min_obj):
                print(min_obj)
                pdb.set_trace()
            return min_obj

    def sample_probs(self, probs: torch.Tensor, index: int):
        """Given a probability distribution tensor (shape = (N, D, K)) returns 1 sample from the
        distribution probs[:, index, :], the output is indices"""
        p = probs[:, index, :]
        sample = p.multinomial(num_samples=1).squeeze()
        return sample

    def sample_model(self, nsamples: int, *input_vectors):
        self.model.eval()

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
        plt.savefig('figs/test_model2_learning_curve.png')
        plt.close()

    def main(self, seed=10):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        data, weights = sample_data(self.nsample)
        plot_data(data[:, 0, :], label=weights)
        xtr, xte, wtr, wte = split_data(data, label=weights)

        D = self.dim
        self.model: nn.Module = MADE(D, self.hiddens, D * 100, seed=seed)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=0)
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
                wb = wtr[step * B: step * B + B]
                xb_tens = torch.from_numpy(xb).to(self.device)
                wb_tens = torch.from_numpy(wb).to(self.device)

                xin = xb_tens[:, 0, :]
                xin_ind = xb_tens[:, 1, :].long()
                loss = self.get_nll(xin, xin_ind, weights=wb_tens, debug=False)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                tr_nll_per_b += loss.to(self.cpu).item() / nstep

            self.model.eval()
            xte_tens = torch.from_numpy(xte).to(self.device)
            wte_tens = torch.from_numpy(wte).to(self.device)
            xin_te = xte_tens[:, 0, :]
            xin_ind_te = xte_tens[:, 1, :].long()
            te_loss = self.get_nll(xin_te, xin_ind_te, weights=wte_tens)
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
        #     plot_data(samples, scatter_loc='figs/test_model2_scatter.png',
        #               hist_loc='figs/test_model2_hist2D.png', ax=ax)
        #     plt.tight_layout()
        # plt.savefig('figs/test_model2_hist2D.png')

        self.plot_learning(tr_nll, te_nll)
        # pdb.set_trace()

        x1 = np.linspace(start=-5, stop=5, num=100)
        x2 = np.linspace(start=-5, stop=5, num=100)
        samples, _ = self.sample_model(10000, x1, x2)
        samples = samples.to(self.cpu).data.numpy()
        plot_data(samples, scatter_loc='figs/test_model2_scatter.png',
                  hist_loc='figs/test_model2_hist2D.png')


if __name__ == '__main__':
    # arr, weights = sample_data(10000)
    # plot_data(arr[:, 0, :], label=weights)
    learner = Learner(
        hidden_list=[20, 20, 20],
        nsample=1000,
        batch_size=16,
        nepoch=40,
        lr=0.0001,
    )
    learner.main()