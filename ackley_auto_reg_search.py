
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import pdb

from ackley import ackley_func
from made import MADE
from buffer import BufferNumpy
from utils.pdb import register_pdb_hook

register_pdb_hook()

class AutoReg2DSearch:

    def __init__(self, goal_value, hidden_list, mode='le', batch_size=16,
                 nepochs=1, nsamples=1, n_init_samples=100, niter=1000, lr=1e-3):

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

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cpu = torch.device('cpu')
        self.model = None
        self.opt = None

        # hacky version of passing input vectors around
        self.input_vectors = []


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

        # sum of entropies across x1 and across x2|x1
        ent = - (probs * probs.log10()).sum(-1).sum(-1)

        D = self.dim
        N = xin.shape[0]
        batch_ind = np.stack([range(N)] * D, axis=-1)
        var_ind = np.stack([range(D)] * N, axis=0)
        prob_in = probs[batch_ind, var_ind, xin_ind]

        if weights is None:
            nll_samples = - prob_in.log2().sum(-1)
            weighted_nll = nll_samples.mean(-1)
            return weighted_nll
        else:

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


            min_obj =  -pos_ll + neg_ll # - 0.2 * ent.mean(-1)

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
        xin_prob = torch.ones(nsamples)
        for i in range(D):
            probs = self.get_probs(xin)
            xi_ind = self.sample_probs(probs, i)  # ith x index
            xin_prob *= probs[:, i][range(nsamples), xi_ind]
            xin_ind[:, i] = xi_ind
            xin[:, i] = torch.from_numpy(input_vectors[i][xi_ind])
        return xin, xin_ind, xin_prob

    def viz(self, data: np.ndarray, prefix='', suffix=''):
        x, y = data[:, 0], data[:, 1]
        plt.scatter(x, y, marker='x')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        name = 'samples'
        if prefix:
            name = f'{prefix}_{name}'
        if suffix:
            name = f'{name}_{suffix}'

        plt.savefig(f'search_figs/{name}.png')
        plt.close()

        plt.hist2d(x, y, bins=100, cmap='binary', range=np.array([(-5, 5), (-5, 5)]))
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        # k = kde.gaussian_kde(data.T)
        # nbins = 100
        # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # plt.pcolormesh(xi, yi , zi.reshape(xi.shape), cmap='viridis')
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)
        plt.colorbar()
        name = 'hist'
        if prefix:
            name = f'{prefix}_{name}'
        if suffix:
            name = f'{name}_{suffix}'
        plt.savefig(f'search_figs/{name}.png')
        plt.close()

    def run_epoch(self, data: np.ndarray, weights: np.ndarray, mode='train',
                  debug=False):
        self.model.train() if mode == 'train' else self.model.eval()

        N, D,  _ = data.shape
        B = self.bsize if mode == 'train' else N
        nstep = N // B if mode == 'train' else 1

        nll_per_b = 0
        for step in range(nstep):
            xb = data[step * B: step * B + B]
            wb = weights[step * B: step * B + B]
            xb_tens = torch.from_numpy(xb).to(self.device)
            wb_tens = torch.from_numpy(wb).to(self.device)

            xin = xb_tens[:, 0, :]
            xin_ind = xb_tens[:, 1, :].long()
            loss = self.get_nll(xin, xin_ind, weights=wb_tens, debug=debug)
            if mode == 'train':
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            nll_per_b += loss.to(self.cpu).item() / nstep

        return nll_per_b

    def train(self, n_samples, *input_vecs, iter_cnt: int):
        xnew, xnew_ind, xnew_probs = self.sample_model(n_samples, *input_vecs)
        xnew_np = xnew.to(self.cpu).data.numpy()
        xnew_id_np = xnew_ind.to(self.cpu).data.numpy()
        xnew_probs_np = xnew_probs.to(self.cpu).data.numpy()
        self.viz(xnew_np, prefix='training', suffix=f'{iter_cnt}_before')

        # simulate and compute the adjustment weights
        fval = ackley_func(xnew_np)
        self.buffer.add_samples(xnew_np, xnew_id_np, fval, xnew_probs_np)
        # treat the sampled data as a static data set and take some gradient steps on it
        xtr, xte, wtr, wte = self.buffer.draw_tr_te_ds()
        # per epoch
        print('-'*50)
        for epoch_id in range(self.nepochs):
            tr_nll = self.run_epoch(xtr, wtr, mode='train')
            te_nll = self.run_epoch(xte, wte, mode='test')

            print(f'[train_{iter_cnt}] epoch {epoch_id} loss = {tr_nll}')
            print(f'[test_{iter_cnt}] epoch {epoch_id} loss = {te_nll}')



    def main(self, seed=10):

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        x1_sp = np.arange(-5, 5, 0.1)
        x2_sp = np.arange(-5, 5, 0.1)

        self.input_vectors = [x1_sp, x2_sp]

        assert x1_sp.shape == x2_sp.shape, ValueError('the cardinality of dimensions cannot be '
                                                      'different')

        D = self.dim
        self.model: nn.Module = MADE(D, self.hiddens, D * x1_sp.shape[-1], natural_ordering=True,
                                     seed=seed)
        self.model.to(self.device)

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

        xp_ref = torch.from_numpy(np.array([[0, 0]]))
        xind_ref = torch.from_numpy(np.array([[50, 50]]))

        self.buffer = BufferNumpy(mode=self.mode, goal=self.goal, cut_off=100)

        self.train(self.n_init_samples, x1_sp, x2_sp, iter_cnt=0)
        xdata, _, probs = self.sample_model(2000, x1_sp, x2_sp)

        self.viz(xdata.to(self.cpu).data.numpy(), prefix='training', suffix='0_after')
        # pdb.set_trace()
        for iter_cnt in range(self.niter):
            self.train(self.nsamples, x1_sp, x2_sp, iter_cnt=iter_cnt + 1)
            self.model.eval()
            xdata, _, _ = self.sample_model(2000, x1_sp, x2_sp)
            self.viz(xdata.to(self.cpu).data.numpy(), prefix='training',
                     suffix=f'{iter_cnt+1}_after')
            pdb.set_trace()

            # if iter_cnt % (self.viz_rate - 1) == 0:
            #     # print evaluation
            #     self.model.eval()
            #     print(self.get_nll(xp_ref, xind_ref).item())
            #     xdata, _ = self.sample_model(2000, x1_sp, x2_sp)
            #     self.viz(xdata.to(self.cpu).data.numpy(), iter_cnt)


if __name__ == '__main__':
    searcher = AutoReg2DSearch(
        goal_value=4,
        hidden_list=[20, 20, 20],
        mode='le',
        batch_size=8,
        nepochs=10,
        nsamples=5,
        n_init_samples=20,
        niter=30,
        lr=0.0003,
    )
    searcher.main()