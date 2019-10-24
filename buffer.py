from typing import Sequence, Optional

import numpy as np
from weight_compute import weight

from utils.data import split_data


class BufferNumpy:
    """Naive implementation of Buffer"""

    def __init__(self, mode, goal, cut_off=20):
        """

        :param mode:
            'ge' or 'le' depending on the optimization problem
        :param goal:
            the threshold value for which you are trying to shoot for
        :param cut_off:
            the individuals above this rank will be a reference for computing the average.
        """
        self.mode = mode
        self.goal = goal
        self.data_arr: Optional[np.ndarray] = None
        self.data_ind_arr: Optional[np.ndarray] = None
        self.fval_arr: Optional[np.ndarray] = None
        self.prob_arr: Optional[np.ndarray] = None

        self.cut_off = cut_off

    @property
    def size(self):
        return self.fval_arr.shape[0]

    def add_samples(self, new_samples: np.ndarray, new_ind: np.ndarray, fvals: np.ndarray,
                    probs: np.ndarray):
        """
        new_samples = Nx...
        fvals = Nx...
        """
        if self.data_arr is None:
            self.data_arr = new_samples
            self.fval_arr = fvals
            self.data_ind_arr = new_ind
            self.prob_arr = probs
        else:
            self.data_arr = np.append(self.data_arr, new_samples, axis=0)
            self.data_ind_arr = np.append(self.data_ind_arr, new_ind, axis=0)
            self.fval_arr = np.append(self.fval_arr, fvals, axis=0)
            self.prob_arr = np.append(self.prob_arr, probs, axis=0)

    def draw_tr_te_ds(self, split=0.8):

        if self.size > self.cut_off:
            fval_rep = np.sort(self.fval_arr, kind='mergesort')
            if self.mode == 'ge':
                fval_rep = fval_rep[::-1]
            fval_rep = fval_rep[:self.cut_off]
        else:
            fval_rep = self.fval_arr

        weights = weight(self.fval_arr, self.goal, fval_rep.mean(), mode=self.mode)
        # important sampling trick
        weights /= self.prob_arr
        data = np.stack([self.data_arr, self.data_ind_arr], axis=1)
        ret = split_data(data, label=weights, train_per=split)
        return ret