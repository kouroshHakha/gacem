import heapq
from typing import Optional

import numpy as np

from optnet.alg.utils.weight_compute import weight, weight2
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
    """Buffer with caching capabilities"""

    def __init__(self, mode, goal, cut_off=10, with_frequencies = False, problem_type = 'csp'):
        self.mode = mode
        self.goal = goal
        self.cut_off = cut_off
        self.problem_type = 'csp'
        self._running_mean = float('inf')
        self.with_freq = with_frequencies
        self.zavg = float('inf') if self.mode == 'le' else float('-inf')

        # use a dictionary for fast lookup of existance (values are frequency of repetition)
        self.db_set = {}
        # use priority queue for getting the top individuals mean quickly
        # mean should not care about frequencies, so does db_pq
        self.db_pq = []

    @property
    def size(self):
        return len(self.db_pq)

    @property
    def mean(self):
        return self.topn_mean(self.cut_off)
        # outdated mean
        # n = max(int(self.size * self.cut_off), 10) # max is for handling corner cases
        # ret = heapq.nsmallest(n, self.db_pq)
        # vals = [x[0] for x in ret]
        # new_mean = float(np.mean(vals))
        # if new_mean < self._running_mean:
        #     self._running_mean = new_mean
        # return self._running_mean

    def topn_mean(self, n):
        ret = heapq.nsmallest(n, self.db_pq)
        vals = [x[0] for x in ret]
        return float(np.mean(vals))

    @property
    def tot_freq(self):
        return sum([x[0] for x in self.db_set.values()])

    @property
    def n_sols(self):
        ret = heapq.nsmallest(self.size, self.db_pq)
        vals = [x[0] for x in ret]
        if self.mode == 'le':
            return (np.array(vals) <= self.goal).sum(-1)
        else:
            return (np.array(vals) >= self.goal).sum(-1)


    def add_samples(self, new_samples: np.ndarray, new_ind: np.ndarray, fvals: np.ndarray):
        for item, item_ind, val in zip(new_samples, new_ind, fvals):
            element = CacheElement(item, item_ind, val)

            already_exists = element in self.db_set
            if not already_exists:
                priority = val if self.mode == 'le' else -val
                heapq.heappush(self.db_pq, (priority, hash(element)))
                self.db_set[element] = [1, element]
            else:
                self.db_set[element][0] += 1

    def _weights(self, normalize_weight: bool):

        values_list = []
        for el in self.db_set:
            if self.with_freq:
                values_list += [el.val for _ in range(self.db_set[el][0])]
            else:
                values_list.append(el.val)
        values_np = np.array(values_list)

        if self.mean == float('inf'):
            raise ValueError('mean is infinite')

        # weight() is smooth weight2() is and indicator function similar to CEM
        self.zavg = heapq.nsmallest(self.cut_off, self.db_pq)[-1][0]
        # weights = weight2(values_np, self.goal, self.zavg, self.mode, self.problem_type)
        if self.problem_type == 'optim':
            raise ValueError('make problem csp')
        else:
            weights = weight(values_np, self.goal, self.zavg, self.mode)
        # normalize weights to have a max of 1
        if normalize_weight:
            weights_norm = weights / weights.max()
            return weights_norm
        return weights

    def _get_all_data(self):
        data_list = []
        for el in self.db_set:
            data = np.stack([el.item, el.item_ind], axis=0)
            if self.with_freq:
                # repeat according to frequencies
                data_list += [data for _ in range(self.db_set[el][0])]
            else:
                data_list.append(data)

        return np.stack(data_list, axis=0).astype('float32')

    def draw_tr_te_ds(self, split=0.8, only_positive = False, normalize_weight=True):
        data = self._get_all_data()
        weights = self._weights(normalize_weight)
        if only_positive:
            data = data[weights > 0]
            weights = weights[weights > 0]
        train_x, test_x, train_w, test_w = split_data(data, weights, split)
        return train_x, test_x, train_w, test_w

    def __contains__(self, item: np.ndarray):
        return item.squeeze().tostring() in self.db_set

    def __getitem__(self, item: np.ndarray):
        return self.db_set[item.tostring()][1].val