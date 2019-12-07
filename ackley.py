from typing import Union
import numbers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from utils.pdb import register_pdb_hook

register_pdb_hook()

from weight_compute import weight

def mixture_ackley(x: Union[np.ndarray, numbers.Number]) -> np.ndarray:
    # visualize it for x.dim = 2
    a = 20
    b = 0.2
    c = 2 * np.pi

    if isinstance(x, numbers.Number):
        x = np.array([x])

    D = x.shape[-1]
    n_org_per_dim = 2
    y_tot = np.zeros(x.shape[0])
    for dim in range(D):
        for i in range(n_org_per_dim):
            org = np.zeros(D)
            org[dim] = 5 if i % 2 == 0 else -2
            y11 = - a * np.exp(-b * np.sqrt(np.mean((x-org) ** 2, axis=-1)))
            y12 = - np.exp(np.mean(np.cos(c * (x-org)), axis=-1))
            bias = a + np.exp(1)
            y_tot += (y11 + y12 + bias) / D / n_org_per_dim

    return y_tot

def styblinski(x):
    return 0.5 * ((x ** 4).sum(-1) - 16 * (x ** 2).sum(-1) + 5 * x.sum(-1)) / x.shape[-1] + 50


def ackley_func(x: Union[np.ndarray, numbers.Number]) -> np.ndarray:
    # visualize it for x.dim = 2
    a = 20
    b = 0.2
    c = 2 * np.pi

    if isinstance(x, numbers.Number):
        x = np.array([x])

    y1 = - a * np.exp(-b * np.sqrt(np.mean(x ** 2, axis=-1)))
    y2 = - np.exp(np.mean(np.cos(c * x), axis=-1))
    y3 = a + np.exp(1)

    return y1 + y2 + y3

def plot_fn(x1, x2, fn, fpath=''):

    x_mesh, y_mesh = np.meshgrid(x1, x2)
    xflat = x_mesh.flatten()
    yflat = y_mesh.flatten()

    xin = np.stack([xflat, yflat], axis=-1)

    yout = fn(xin)
    z_mesh = yout.reshape(x_mesh.shape)

    plt.close()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('value')

    if fpath:
        plt.savefig(fpath)
    else:
        plt.show()

def show_cutted_fn(x1, x2, fn, value, fpath=''):

    x_mesh, y_mesh = np.meshgrid(x1, x2)
    xflat = x_mesh.flatten()
    yflat = y_mesh.flatten()

    xin = np.stack([xflat, yflat], axis=-1)

    yout = fn(xin)
    z_mesh = yout.reshape(x_mesh.shape)
    zregion = np.zeros(z_mesh.shape)
    zregion[z_mesh > value] = 0
    zregion[z_mesh <= value] = 1

    plt.close()
    fig = plt.figure()
    im = plt.pcolormesh(x_mesh, y_mesh, zregion, cmap='binary')
    fig.colorbar(im)

    if fpath:
        plt.savefig(fpath)
    else:
        plt.show()

def show_weight_on_all(x1, x2, fn, value=4, value_avg=4, mode='le', ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

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

if __name__ == '__main__':
    scale = 5
    x1 = np.linspace(start=-scale, stop=scale, num=100)
    x2 = np.linspace(start=-scale, stop=scale, num=100)

    goal = 4
    avg = 10
    fn = ackley_func
    plot_fn(x1, x2, fn)
    plt.savefig(f'ref_figs/fn.png')
    plt.close()
    show_cutted_fn(x1, x2, fn, value=goal)
    plt.savefig('ref_figs/goal.png')
    plt.close()
    show_weight_on_all(x1, x2, fn, value=goal, value_avg=avg, mode='le')
    plt.savefig('ref_figs/weights.png')
    plt.close()