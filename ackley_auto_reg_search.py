
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import kde

from ackley import ackley_func
from made import MADE
from weight_compute import weight

class AutoReg2DSearch:

    def __init__(self, goal_value, hidden_list, mode='le', batch_size=16, niter=1000, lr=1e-3):

        self.dim = 2
        self.bsize = batch_size
        self.hiddens = hidden_list
        self.niter = niter
        self.goal = goal_value
        self.mode = mode
        self.viz_rate = niter // 10
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

        nll_samples = - prob_in.log2().sum(dim=-1)

        # multiply each nll by the corresponding weight and take the mean
        if weights is None:
            weighted_nll = nll_samples.mean(dim=-1)
        else:
            weighted_nll = (nll_samples * weights).mean(dim=-1)

        return weighted_nll


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

    def viz(self, data: np.ndarray, niter: int):
        x, y = data[:, 0], data[:, 1]
        plt.scatter(x, y, marker='x')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.savefig(f'figs/samples_{niter}.png')
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
        plt.savefig(f'figs/hist_{niter}.png')
        plt.close()

    def main(self, seed=10):

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        x1_sp = np.arange(-5, 5, 0.1)
        x2_sp = np.arange(-5, 5, 0.1)

        assert x1_sp.shape == x2_sp.shape, ValueError('the cardinality of dimensions cannot be '
                                                      'different')

        D = self.dim
        self.model: nn.Module = MADE(D, self.hiddens, D * x1_sp.shape[-1], natural_ordering=True)
        self.model.to(self.device)

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

        xp_ref = torch.from_numpy(np.array([[0, 0]]))
        xind_ref = torch.from_numpy(np.array([[50, 50]]))

        # HACK/DEBUG
        # xnew_ind = np.array([[50, 50],
        #                      [50, 58],
        #                      [50, -58],
        #                      [58, 50],
        #                      [-58, 50]])
        # xnew = np.stack([x1_sp[xnew_ind[:, 0]], x2_sp[xnew_ind[:, 1]]], axis=-1)
        # xnew = torch.from_numpy(xnew)
        # xnew_ind = torch.from_numpy(xnew_ind)

        for iter_cnt in range(self.niter):
            xnew, xnew_ind = self.sample_model(self.bsize, x1_sp, x2_sp)
            # simulate and compute the adjustment weights
            fval = ackley_func(xnew.to(self.cpu).data.numpy())
            weights = weight(fval, self.goal, np.mean(fval), mode=self.mode)

            self.model.train()
            loss = self.get_nll(xnew, xnew_ind, weights=torch.from_numpy(weights))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # print evaluation
            self.model.eval()
            print(self.get_nll(xp_ref, xind_ref).item())
            xdata, _ = self.sample_model(2000, x1_sp, x2_sp)
            self.viz(xdata.to(self.cpu).data.numpy(), iter_cnt)
            seq_module = list(self.model.children())[0]
            layers = list(seq_module.children())
            print(layers[0].weight)
            print(layers[2].weight)
            print(layers[4].weight)
            import pdb
            pdb.set_trace()

            # if iter_cnt % (self.viz_rate - 1) == 0:
            #     # print evaluation
            #     self.model.eval()
            #     print(self.get_nll(zero_tensor, zero_idx).item())
            #     xdata, _ = self.sample_model(2000, x1_sp, x2_sp)
            #     self.viz(xdata.to(self.cpu).data.numpy(), iter_cnt)
            #     seq_module = list(self.model.children())[0]
            #     layers = list(seq_module.children())
            #     print(layers[0].weight)

        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    searcher = AutoReg2DSearch(
        goal_value=4,
        hidden_list=[4, 4, 4],
        mode='le',
        batch_size=1000,
        niter=100,
        lr=0.03,
    )
    searcher.main()