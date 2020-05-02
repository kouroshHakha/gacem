from typing import Optional, Mapping, Union
from sklearn.decomposition.pca import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pathlib import Path
import random
from mpl_toolkits import mplot3d
import itertools

from ..alg.utils.weight_compute import weight

def _get_ax(ax):
    if ax is None:
        plt.close()
        ax = plt.axes()
    return ax

def _save_show_fig(fpath: Union[str, Path], show_fig: bool, **kwargs):
    if fpath is not None:
        fpath: Path = Path(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fpath, **kwargs)
    elif show_fig:
        plt.show()

def plt_hist2D(data: np.ndarray, *, ax=None, fpath=None, show_fig=False, show_colorbar=False,
               bins=100, range: Optional[np.ndarray] = None, **kwargs):

    if range is None:
        xmin = np.min(data[:, 0])
        xmax = np.max(data[:, 0])
        ymin = np.min(data[:, 1])
        ymax = np.max(data[:, 1])
        range = np.array([[xmin, xmax], [ymin, ymax]])

    ax = _get_ax(ax)
    im = ax.hist2d(data[:, 0], data[:, 1], bins=bins, range=range, **kwargs)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    if show_colorbar:
        plt.colorbar(im[-1], ax=ax)
    _save_show_fig(fpath, show_fig)

def show_weight_on_all(x1, x2, fn, value=4, value_avg=4, mode='le', ax=None, **kwargs):
    warnings.warn('show_weight_on_all is deprecated', DeprecationWarning)
    ax = _get_ax(ax)

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

    ax = _get_ax(ax)

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

    _save_show_fig(fpath, show_fig)


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

    _save_show_fig(fpath, show_fig)

def plot_pca_2d(data, ax=None, fpath='', show_fig=False):
    ax = _get_ax(ax)

    pca_2d = PCA(n_components=2)
    data_hat = pca_2d.fit_transform(data)
    ax.scatter(data_hat[:, 0], data_hat[:, 1])
    _save_show_fig(fpath, show_fig)

    return pca_2d


def plot_learning_with_epochs(ax=None, fpath='', show_fig=False, **kwrd_losses):
    ax = _get_ax(ax)
    for key, loss in kwrd_losses.items():
        loss_list = []
        for i, l in enumerate(loss):
            loss_list += l
        ax.plot(loss_list, label=f'{key}_loss')
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    _save_show_fig(fpath, show_fig)

def plot_cost(avg_cost, ax=None, fpath='', show_fig=False):
    ax = _get_ax(ax)
    ax.plot(avg_cost)
    ax.set_title('buffer top samples mean')
    ax.set_xlabel('iter')
    ax.set_ylabel('avg_cost')
    _save_show_fig(fpath, show_fig)

def plot_x_y(x, y=None, annotate=None, ax=None, fpath='', show_fig=False, xlabel='', ylabel='',
             **kwargs):
    ax = _get_ax(ax)
    if y is None:
        ax.plot(x, **kwargs)
    else:
        ax.plot(x, y, **kwargs)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if annotate:
        diff_annotate = np.diff([0] + annotate)
        for i, txt in enumerate(annotate):
            if diff_annotate[i] > 0:
                ax.annotate(txt, (x[i], y[i]))
    _save_show_fig(fpath, show_fig)


def scatter2d(data: np.ndarray, labels: Optional[np.ndarray] = None,
              label_mapping: Optional[Mapping[int, str]] = None,
              ax=None, fpath='', show_fig=False, fig_title=None, **kwargs):
    random.seed(20)
    ax = _get_ax(ax)
    markers = (',', '+', 'o', '*')
    colors = ('r', 'g', 'b', 'c', 'm', 'y', 'k')
    marker_color = list(itertools.product(markers, colors))
    random.shuffle(marker_color)
    marker_color = iter(marker_color)
    im = None
    if labels is not None:
        for label in np.unique(labels):
            pos = (labels == label)
            if label_mapping is None:
                _label = label
            else:
                _label = label_mapping[label]
            marker, color = next(marker_color)
            im = ax.scatter(data[pos, 0], data[pos, 1], marker=marker, color=color,
                            label=_label, **kwargs)
    else:
        im = ax.scatter(data[:, 0], data[:, 1], **kwargs)

    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    if fig_title is None:
        ax.set_title(fig_title)
    _save_show_fig(fpath, show_fig, dpi=400, bbox_inches="tight")
    return im

def pca_scatter2d(data: np.ndarray, labels: Optional[np.ndarray] = None,
                  label_mapping: Optional[Mapping[int, str]] = None,
                  ax=None, fpath='', show_fig=False, title=None, **kwargs):

    pca_2d = PCA(n_components=2)
    data_hat = pca_2d.fit_transform(data)
    return scatter2d(data_hat, labels, label_mapping, ax, fpath, show_fig, title, **kwargs)

def tsne_scatter2d(data: np.ndarray, labels: Optional[np.ndarray] = None,
                   label_mapping: Optional[Mapping[int, str]] = None,
                   seed: Optional[int] = None,
                   ax=None, fpath='', show_fig=False, title=None, **kwargs):

    data_hat = TSNE(n_components=2, random_state=seed).fit_transform(data)
    return scatter2d(data_hat, labels, label_mapping, ax, fpath, show_fig, title, **kwargs)
