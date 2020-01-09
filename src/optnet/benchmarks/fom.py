"""This module implements different figure of merits"""


import numpy as np
from sklearn.decomposition.pca import PCA

def compute_emprical_variation(data):
    mean = data.mean(0)
    cov_mat = ((data - mean).T @ (data - mean)) / data.shape[0]
    var_sum = cov_mat.trace() / data.shape[-1]
    return var_sum


def get_diversity_fom(ndim, data, return_pca=False):

    pca = PCA(n_components=ndim)
    pca.fit(data)

    if return_pca:
        return pca

    vec = pca.explained_variance_ratio_ + 1e-15
    div = (-vec * np.log(vec)).sum(-1) * pca.explained_variance_.sum(-1)
    div /= ndim
    return div
