import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal


class CEM:
    """
    The vanilla implementation of Cross Entropy method with gaussian distributions
    """
    def __init__(self, input_vectors, average_coeff, dist_type='gauss', **kwargs):
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
        **kwargs: Dict[str, Any]
            gauss_sigma: if dist_type == 'gauss' will be the constant sigma used without updating
            sigma
        """
        self.input_indices = [range(len(x)) for x in input_vectors]
        dim = len(input_vectors)
        self.params_min = np.array([0] * dim)
        self.params_max = np.array([len(x) - 1 for x in self.input_indices])

        if dist_type not in ['gauss', 'kde']:
            raise ValueError(f'{dist_type} is not a valid kernel type for CEM++: guass | kde')

        self.type = dist_type
        self.gauss_sigma = kwargs.get('gauss_sigma', None)

        self.average_coeff = average_coeff
        self.params = {}

    def fit(self, data):
        # data has to be in units of indices
        if len(data.shape) > 2:
            raise ValueError('Data should be in shape of Nxd (N samples with d dimensions)')
        ndata, ndim = data.shape
        alpha = self.average_coeff
        if self.type == 'gauss':
            # TODO: investigate why this is more stable than using new_mu in computing new_var
            old_mu = self.params.get('mu', 0)
            if self.gauss_sigma is None:
                old_var = self.params.get('var', 0)
                new_var = 1 / ndata * (data - old_mu).T @ (data - old_mu)
                self.params['var'] = old_var * (1 - alpha) + new_var * alpha
            else:
                self.params['var'] = self.gauss_sigma * np.eye(ndim)
            new_mu = np.mean(data, axis=0)
            self.params['mu'] = old_mu * (1 - alpha) + new_mu * alpha
        elif self.type == 'kde':
            self.params['kde'] = gaussian_kde(np.transpose(data))

    def _draw_uniform_samples(self, n):
        dim = len(self.params_min)
        cols = []
        for i in range(dim):
            cols.append(np.random.randint(0, self.params_max[i], n, dtype='int'))
        samples = np.stack(cols, axis=-1)
        return samples

    def sample(self, n, uniform: bool = False):
        if not uniform:
            if not self.params:
                raise ValueError('params are not set yet, did you forget to call fit?')

            if self.type == 'gauss':
                samples = multivariate_normal.rvs(self.params['mu'],
                                                  self.params['var'], n)
            elif self.type == 'kde':
                samples = self.params['kde'].resample(n)
                samples = samples.transpose()
            else:
                raise ValueError(f'oops! not supported {self.type}')

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
        if self.type == 'gauss':
            return multivariate_normal.pdf(samples, self.params['mu'], self.params['var'],
                                           allow_singular=True)
        else:
            return self.params['kde'].pdf(samples.T).T

    def entropy(self, nsamples=None):
        samples = self.sample(nsamples)
        pdf = self.evaluate_pdf(samples)
        return (-np.log(pdf)).mean(-1)
