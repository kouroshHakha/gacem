"""
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
"""
from typing import List, cast
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pdb import register_pdb_hook

register_pdb_hook()

# ------------------------------------------------------------------------------
class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        mask_t = torch.from_numpy(mask.astype(np.int).T)
        self.mask.data.copy_(mask_t)

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False, seed=0,
                 bias_init = 0.1):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.bias_init = bias_init
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]

        # dropout_layer = nn.Dropout(0, inplace=True)

        for h0,h1 in zip(hs, hs[1:]):
            self.net += [
                # nn.BatchNorm1d(h0),
                MaskedLinear(h0, h1),
                nn.LeakyReLU(),
                # dropout_layer,
            ]
        # self.net.pop() # pop the last dropout
        self.net.pop() # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        D = nout // nin
        self.comp_net = nn.Sequential(
            nn.Linear(D, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, nout),
        )

        self.net.apply(self.init_weights)
        self.comp_net.apply(self.init_weights)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = seed # for cycling through num_masks orderings

        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

        self.indp_var = 0

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.uniform_(m.bias, a=-self.bias_init, b=self.bias_init)
            # m.bias.data.fill_(self.bias_init)
            # nn.init.normal(m.bias, 0, 1/np.sqrt(m.bias.shape[-1]))

    def update_masks(self, force_natural_ordering=False):
        if self.m and self.num_masks == 1 and self.natural_ordering: return
        # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        if self.natural_ordering or force_natural_ordering:
            self.m[-1] = np.arange(self.nin)
        else:
            self.m[-1] = rng.permutation(self.nin)
            self.indp_var = np.argwhere(self.m[-1] == 0)[0][0]

        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks: List[np.ndarray] = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        y = self.net(x.float())
        # pass the independent variable through more non-linearity to get more expressiveness
        mask = torch.ones(y.shape, dtype=torch.bool)
        mask[:, self.indp_var::self.nin] = False
        out = y * mask + self.comp_net(y[:, self.indp_var::self.nin]) * (~mask)
        return out


if __name__ == '__main__':
    # run a quick and dirty test for the autoregressive property
    D = 2
    rng = np.random.RandomState(14)
    x = cast(np.ndarray, (rng.rand(1, D) > 0.5)).astype(np.float)

    # nin, hidden_list, nout, natural_ordering
    configs = [
        # (D, [], D, False),                 # test various hidden sizes
        # (D, [200], D, False),
        # (D, [200, 220], D, False),
        # (D, [200, 220, 230], D, False),
        # (D, [200, 220], D, True),          # natural ordering test
        # (D, [200, 220], 2*D, True),       # test nout > nin
        # (D, [20, 20], 200*D, True),       # test nout > nin
        (D, [20, 20, 20], 300*D, True)
    ]

    for nin, hiddens, nout, natural_ordering in configs:

        print("checking nin %d, hiddens %s, nout %d, natural %s" %
              (nin, hiddens, nout, natural_ordering))
        model = MADE(nin, hiddens, nout, natural_ordering=natural_ordering, num_masks=3, seed=2)
        # run backpropagation for each dimension to compute what other
        # dimensions it depends on.
        res = []
        for k in range(nout):
            xtr = torch.from_numpy(x)
            xtr.requires_grad_(True)
            model.train()
            xtrhat = model(xtr)
            loss = xtrhat[0,k]
            loss.backward()
            depends = (xtr.grad[0].numpy() != 0)
            depends_ix = list(np.where(depends)[0])
            isok = k % nin not in depends_ix

            res.append((len(depends_ix), k, depends_ix, isok))

        # pretty print the dependencies
        res.sort()
        for nl, k, ix, isok in res:
            print("output %2d depends on inputs: %30s : %s" % (k, ix, "OK" if isok else "NOTOK"))