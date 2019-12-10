from typing import List
from sklearn.decomposition.pca import PCA
import matplotlib.pyplot as plt
import numpy as np
import warnings
from mpl_toolkits import mplot3d

from ..alg.utils.weight_compute import weight

def plt_hist2D(input_vecs: List[np.ndarray], data: np.ndarray,  ax=None, fpath='',
               show_fig=False, show_colorbar=False, **kwargs):
    if ax is None:
        plt.close()
        ax = plt.axes()

    xvec, yvec = input_vecs

    im = ax.hist2d(xvec[data[:, 0]], yvec[data[:, 1]], bins=100, **kwargs)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    if show_colorbar:
        plt.colorbar(im[-1], ax=ax)

    if fpath:
        plt.savefig(fpath)
    elif show_fig:
        plt.show()

def show_weight_on_all(x1, x2, fn, value=4, value_avg=4, mode='le', ax=None, **kwargs):
    warnings.warn('show_weight_on_all is deprecated', DeprecationWarning)

    if ax is None:
        plt.close()
        ax = plt.axes()

    x_mesh, y_mesh = np.meshgrid(x1, x2)
    xflat = x_mesh.flatten()
    yflat = y_mesh.flatten()

    xin = np.stack([xflat, yflat], axis=-1)

    yout = fn(xin)
    weights = weight(yout, value, value_avg, mode)
    z_mesh = weights.reshape(x_mesh.shape)
    im = ax.pcolormesh(x_mesh, y_mesh, z_mesh, **kwargs)
    plt.colorbar(im, ax=ax)
    return ax


def show_solution_region(x1, x2, fn, value, mode='le', ax=None, fpath='', show_fig=False,
                         show_colorbar=False, **kwargs):

    if ax is None:
        plt.close()
        ax = plt.axes()

    x_mesh, y_mesh = np.meshgrid(x1, x2)
    xflat = x_mesh.flatten()
    yflat = y_mesh.flatten()

    xin = np.stack([xflat, yflat], axis=-1)

    yout = fn(xin)
    z_mesh = yout.reshape(x_mesh.shape)
    zregion = np.zeros(z_mesh.shape)

    if mode == 'le':
        zregion[z_mesh > value] = 0
        zregion[z_mesh <= value] = 1
    elif mode == 'ge':
        zregion[z_mesh < value] = 0
        zregion[z_mesh >= value] = 1
    else:
        raise ValueError(f'unvalid optimization mode {mode}, valid options: "le" | "ge"')

    im = ax.pcolormesh(x_mesh, y_mesh, zregion, **kwargs)
    if show_colorbar:
        fig = plt.gcf()
        fig.colorbar(im)

    if fpath:
        plt.savefig(fpath)
    elif show_fig:
        plt.show()

def plot_fn2d(x1, x2, fn, ax=None, fpath='', show_fig=False, **kwargs):

    if ax is None:
        plt.close()
        ax = plt.axes(projection='3d')

    x_mesh, y_mesh = np.meshgrid(x1, x2)
    xflat = x_mesh.flatten()
    yflat = y_mesh.flatten()

    xin = np.stack([xflat, yflat], axis=-1)

    yout = fn(xin)
    z_mesh = yout.reshape(x_mesh.shape)

    ax.plot_surface(x_mesh, y_mesh, z_mesh, **kwargs)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('value')

    if fpath:
        plt.savefig(fpath)
    elif show_fig:
        plt.show()


def plot_pca_2d(data, ax=None):
    if ax is None:
        plt.close()
        ax = plt.axes()

    pca_2d = PCA(n_components=2)
    data_hat = pca_2d.fit_transform(data)
    ax.scatter(data_hat[:,0], data_hat[:,1])
    return pca_2d

