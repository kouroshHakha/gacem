from typing import Sequence, Optional, Iterable

import heapq
import numpy as np

from utils.data import split_data

from weight_compute import weight

import pdb
import time

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
        self.cut_off = cut_off
        self.data_arr: Optional[np.ndarray] = None
        self.data_ind_arr: Optional[np.ndarray] = None
        self.fval_arr: Optional[np.ndarray] = None
        self.prob_arr: Optional[np.ndarray] = None


    @property
    def size(self):
        return self.fval_arr.shape[0]

    @property
    def mean(self):
        if self.size > self.cut_off:
            fval_rep = np.sort(self.fval_arr, kind='mergesort')
            if self.mode == 'ge':
                fval_rep = fval_rep[::-1]
            fval_rep = fval_rep[:self.cut_off]
        else:
            fval_rep = self.fval_arr

        mean = fval_rep.mean()

        return mean

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
        weights = weight(self.fval_arr, self.goal, self.mean, mode=self.mode)
        # important sampling trick
        # weights /= self.prob_arr
        data = np.stack([self.data_arr, self.data_ind_arr], axis=1)
        ret = split_data(data, label=weights, train_per=split)
        return ret

class CacheElement:

    def __init__(self, item: np.ndarray, item_ind: np.ndarray, val: float):
        self._item = item.squeeze()
        self._item_ind = item_ind.squeeze()
        self._val = val

    @property
    def item(self):
        return self._item

    @property
    def item_ind(self):
        return self._item_ind

    @property
    def val(self):
        return self._val

    @property
    def prob(self):
        return

    def __hash__(self):
        return hash((self._item.tostring()))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str(self._item)

    def __repr__(self):
        return repr(self._item)


class CacheBuffer:

    def __init__(self, mode, goal, cut_off=0.2):
        self.mode = mode
        self.goal = goal
        self.cut_off = cut_off
        self._running_mean = float('inf')


        self.db_set = {} # using a dictionary as ordered set (discard values)
        self.db_pq = []

    @property
    def size(self):
        return len(self.db_set)

    @property
    def mean(self):
        n = int(self.size * self.cut_off)
        ret = heapq.nsmallest(n, self.db_pq)
        vals = [x[0] for x in ret]
        new_mean = float(np.mean(vals))
        if new_mean < self._running_mean:
            self._running_mean = new_mean
        return self._running_mean

    def add_samples(self, new_samples: np.ndarray, new_ind: np.ndarray, fvals: np.ndarray):
        for item, item_ind, val in zip(new_samples, new_ind, fvals):
            element = CacheElement(item, item_ind, val)
            if element not in self.db_set:
                priority = val if self.mode == 'le' else -val
                heapq.heappush(self.db_pq, (priority, hash(element)))
                self.db_set[element] = element
            else:
                print(f'item {item} already exists!')


    def _weights(self):
        values_list = [x.val for x in self.db_set]
        values_np = np.array(values_list)
        weights = weight(values_np, self.goal, self.mean, self.mode)
        # normalize weights to have a max of 1
        weights_norm = weights / weights.max()
        return weights_norm

    def _get_all_data(self):
        data_list = []
        for el in self.db_set:
            data = np.stack([el.item, el.item_ind], axis=0)
            data_list.append(data)
        return np.stack(data_list, axis=0).astype('float32')

    def draw_tr_te_ds(self, split=0.8, only_positive = False):
        data = self._get_all_data()
        weights = self._weights()
        if only_positive:
            data = data[weights > 0]
            weights = weights[weights > 0]
        ret = split_data(data, weights, split)
        return ret

    def __contains__(self, item: np.ndarray):
        return item.squeeze().tostring() in self.db_set
