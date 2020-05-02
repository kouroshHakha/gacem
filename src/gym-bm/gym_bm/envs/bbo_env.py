from typing import Tuple

import numpy as np
import gym
from gym import spaces

from utils.data.database import Database
gym.logger.set_level(40)

def rastrigin(x):
    """Rastrigin test objective function, shifted by 10. units away from origin"""
    x = np.copy(x)
    x -= 10
    if not np.isscalar(x[0]):
        N = len(x[0])
        return -np.array([10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])
    N = len(x)
    return -(10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x)))

def synt(x):
    return -((0.25*(x**4).sum(-1) - 2 * (x**2).sum(-1)) / x.shape[-1] + 4)


registered_fn = {'rastrigin': (rastrigin, -20, 20),
                 'synt': (synt, -5, 5),}

class BlackBoxOptEnv(gym.Env):

    def __init__(self, dim=1, fn=''):
        self.dim = dim
        self.function, low, high = registered_fn[fn]
        self.action_space = spaces.Box(low=low, high=high, shape=(dim, ))
        self.observation_space = spaces.Box(np.float('-inf'), np.float('inf'), shape=(1,))


    def reset(self, ntimes=1):
        obss = np.array([self.observation_space.sample() for _ in range(ntimes)])
        return obss

    def step(self, actions):
        rewards = self.function(actions)
        return rewards, rewards, True, {}

    def seed(self, seed=10):
        np.random.seed(seed)
