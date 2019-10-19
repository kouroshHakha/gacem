from typing import Union
import numbers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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

def plot_ackley():

    x1 = np.linspace(start=-5, stop=5, num=100)
    x2 = np.linspace(start=-5, stop=5, num=100)

    x_mesh, y_mesh = np.meshgrid(x1, x2)
    xflat = x_mesh.flatten()
    yflat = y_mesh.flatten()

    xin = np.stack([xflat, yflat], axis=-1)

    yout = ackley_func(xin)
    z_mesh = yout.reshape(x_mesh.shape)

    ax = plt.axes(projection='3d')
    ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('value')

def show_cutted_ackley(value):

    x1 = np.linspace(start=-5, stop=5, num=100)
    x2 = np.linspace(start=-5, stop=5, num=100)

    x_mesh, y_mesh = np.meshgrid(x1, x2)
    xflat = x_mesh.flatten()
    yflat = y_mesh.flatten()

    xin = np.stack([xflat, yflat], axis=-1)

    yout = ackley_func(xin)
    z_mesh = yout.reshape(x_mesh.shape)
    z_mesh[z_mesh >= value] = 2 * value

    fig = plt.figure()
    im = plt.pcolormesh(x_mesh, y_mesh, z_mesh)
    fig.colorbar(im)


if __name__ == '__main__':
    plot_ackley()
    plt.savefig(f'ref_figs/fn.png')
    plt.close()
    show_cutted_ackley(value=4)
    plt.savefig('ref_figs/goal.png')
    plt.close()