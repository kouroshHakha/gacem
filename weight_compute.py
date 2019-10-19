"""
This module is a test for computing weights which either encourage or discourage certain points
for MLE, base on whether they satisfy the objective or not.

In summary we realized that there are only two types of objectives:
ge (greater than equal): where z_obj > z0 > 0 (z_obj can be negative)
le (less than equal): where 0 < z_obj < z0 (z_obj cannot go below zero)

For negative z0, we can negate the inequality and convert it to one of the aforementioned types.
"""
from typing import Union, cast

import numpy as np
import matplotlib.pyplot as plt


def dist(zi: np.ndarray, z0: Union[np.ndarray, float], mode='ge'):
    """
    Distance of zi and z0, should be 0 when they are close, should be large when they are far apart
    z0 is always positive, if this is not true variables should be manipulated to satisfy that.
    """
    status = np.zeros(shape=zi.shape, dtype=np.bool)
    distance = np.zeros(shape=zi.shape)
    if mode == 'ge':
        # the input space is -inf < zi < inf, linear in z0 <= zi < inf, exp in -inf < zi < z0
        distance[zi >= z0] = cast(np.ndarray, (zi / z0 - 1))[zi >= z0]
        distance[zi < z0] = np.exp(np.abs(zi - z0) / z0)[zi < z0]
        status[zi >= z0] = True
    elif mode == 'le':
        distance = np.maximum(zi, z0) / np.minimum(zi, z0) - 1
        distance[zi < 0] = np.NAN
        status[zi <= z0] = True
    else:
        raise ValueError('mode not supported')
    return distance, status

def plot_distance(mode):
    z0 = 10
    zi = np.linspace(start=-100, stop=100, num=1000)
    distance, stat = dist(zi, z0, mode=mode)
    plt.axvline(x=z0, color='r')
    plt.semilogy(zi[stat], distance[stat], color='b')
    plt.semilogy(zi[~stat], distance[~stat], color='g')

    plt.show()

def weight(zi: np.ndarray, z0: Union[np.ndarray, float], z_avg: float, mode='ge'):
    is_ok = np.greater_equal if mode == 'ge' else np.less_equal

    weights = np.zeros(shape=zi.shape)
    all_ind = np.arange(zi.shape[0])

    # weight is 1 for all those which satisfy the constraint
    ok_ind = all_ind[is_ok(zi, z0)]
    weights[ok_ind] = 1

    # weight is in [0, 1) if it doesn't satisfy the main goal but is better than average
    semi_ok_ind = all_ind[np.bitwise_and(~is_ok(zi, z0), is_ok(zi, z_avg))]
    dist_in = zi[semi_ok_ind]
    main_err, _ = dist(dist_in, z0, mode)
    minor_err, _ = dist(dist_in, z_avg, mode)
    weights[semi_ok_ind] = np.exp(-main_err / minor_err)

    # weight is in (-1, 0) if it doesn't satisfy the main goal but is also worst than average
    not_ok_ind = all_ind[np.bitwise_and(~is_ok(zi, z0), ~is_ok(zi, z_avg))]
    dist_in = zi[not_ok_ind]
    minor_err, _ = dist(dist_in, z_avg, mode)
    weights[not_ok_ind] = -np.exp(-1 / minor_err)

    return weights

def test_weight(zi, z0, z_avg, mode):
    weights = weight(zi, z0, z_avg, mode)
    print(f'zi={zi}, z0={z0}, z_avg={z_avg}, mode={mode}')
    print(f'weights={weights}')

if __name__ == '__main__':
    # plot_distance('le')
    test_weight(np.array([1, 2, 2.5, 3, 4, 10]), z0=4.0, z_avg=2.5, mode='ge')