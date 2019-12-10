from typing import List
import numpy as np


def index_to_xval(input_vectors: List[np.ndarray], index: np.ndarray):
    cols = []
    dim = len(input_vectors)
    for i in range(dim):
        mat = np.zeros(shape=(index.shape[0], 1)) + input_vectors[i][None, :]
        cols.append(mat[range(index.shape[0]), index[:, i]])
    xval = np.stack(cols, axis=-1)
    return xval

def xval_to_index(input_vectors: List[np.ndarray], xval: np.ndarray):
    cols = []
    dim = len(input_vectors)
    for i in range(dim):
        step_i = input_vectors[i][-1] - input_vectors[i][-2]
        mat_indx = np.zeros(shape=(xval.shape[0], 1)) + \
                   np.arange(len(input_vectors[i]))[None, :]
        mat_ref = np.zeros(shape=(xval.shape[0], 1)) + input_vectors[i][None, :]
        xsample_brdcst = xval[:, i][:, None] + np.zeros(shape=(1, len(input_vectors[i])))
        # using np.isclose is because of weird floating point arithmetic 0.6 = 0.60000000001
        col_indices = ((np.isclose(mat_ref, xsample_brdcst, atol=step_i * 1e-2)) * mat_indx).sum(-1)
        cols.append(col_indices)
    index = np.stack(cols, axis=-1)
    return index