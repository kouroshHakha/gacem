import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal

import cma

class CMAES:
    """
    Covariance Matrix Adaptation: https://arxiv.org/abs/1604.00772
    This is a wrapper around cma module (https://github.com/CMA-ES/pycma) to match the interface.
    """
    def __init__(self, input_vectors, seed, lambda_, mu):
        """
        Parameters
        ----------
        input_vectors: List[np.ndarray]
            The possible values for each input dimension
        dist_type: str
            The Kernel type:
                'guass' for single gaussian
                'KDE' for kernel density estimation
        average_coeff: float(not supported in KDE)
            a number between 0,1: params = (1-alpha) * old_params + alpha * new_params
        lambda_:
            population size
        mu:
            selection size (i.e. parent size)
        """

        self.input_indices = [range(len(x)) for x in input_vectors]
        dim = len(input_vectors)
        self.params_min = np.array([0] * dim)
        self.params_max = np.array([len(x) - 1 for x in self.input_indices])

        start = (self.params_min + self.params_max) / 2
        # 3 * sigma should cover everything
        std_init = np.max((self.params_max - self.params_min)) / 2

        opts = dict(seed=seed,
                    termination_callback=lambda *args: False,
                    bounds=[self.params_min, self.params_max],
                    CMA_active=False,
                    popsize=lambda_,
                    CMA_mu=mu,
                    )
        self.es = cma.CMAEvolutionStrategy(start, std_init, opts)


    def fit(self, data):
        # data has too be selected already and sorted from the best to the worst
        # data has to be in units of indices
        ndata = len(data)
        if ndata > self.es.popsize:
            raise ValueError(f'Number of data points {ndata} cannot be more than '
                             f'popsize {self.es.popsize}')
        if len(data.shape) > 2:
            raise ValueError('Data should be in shape of Nxd (N samples with d dimensions)')

        # We are fitting a continuous distribution to discrete data, that won't just work quite
        # right, so add noise [0,1)^D to data and then fit it
        nvec = np.random.rand(*data.shape)
        data = data + nvec

        # inject fake data if data is pre-selected
        if ndata < self.es.popsize:
            nextra_rows = self.es.popsize - ndata
            extra_rows = np.zeros((nextra_rows, data.shape[-1]))
            data = np.concatenate([data, extra_rows], axis=0)

        self.es.tell(data, np.arange(len(data)), copy=True)

    def _draw_uniform_samples(self, n):
        dim = len(self.params_min)
        cols = []
        for i in range(dim):
            cols.append(np.random.randint(0, self.params_max[i], n, dtype='int'))
        samples = np.stack(cols, axis=-1)
        return samples

    def sample(self, n=None, uniform: bool = False):
        if not uniform:
            samples = np.array(self.es.ask(n))
            lo = np.zeros(samples.shape) + self.params_min
            hi = np.zeros(samples.shape) + self.params_max
            samples = np.clip(samples, lo, hi)
            samples = np.floor(samples).astype('int')
        else:
            # uniform sampling
            samples = self._draw_uniform_samples(n)

        if len(samples.shape) == 1:
            samples = samples[None, ...]

        return samples

    def evaluate_pdf(self, samples):
        return multivariate_normal.pdf(samples, self.es.mean, self.es.C, allow_singular=True)

    def entropy(self, nsamples=None):
        samples = self.sample(nsamples)
        pdf = self.evaluate_pdf(samples)
        return (-np.log(pdf)).mean(-1)
