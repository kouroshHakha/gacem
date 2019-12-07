
from sklearn.decomposition.pca import PCA
import matplotlib.pyplot as plt
import numpy as np

class Viz:


    @classmethod
    def plot_pca_2d(cls, data, ax=None):
        if ax is None:
            ax = plt.gca()

        pca_2d = PCA(n_components=2)
        data_hat = pca_2d.fit_transform(data)
        ax.scatter(data_hat[:,0], data_hat[:,1])
        return pca_2d

    @classmethod
    def get_diversity_fom(cls, ndim, data, return_pca=False):

        pca = PCA(n_components=ndim)
        pca.fit(data)

        if return_pca:
            return pca

        vec = pca.explained_variance_ratio_
        div = (-vec * np.log(vec)).sum(-1) * pca.explained_variance_.sum(-1)

        return div






