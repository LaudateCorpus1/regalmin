import numpy as np
from scipy.sparse.linalg import svds

import vector_squared_k_support
import l1

TOLERANCE = 1e-8

class DeflatedPCA:

    def __init__(self, matrix, parameter):
        self.matrix = matrix
        self.parameter = parameter

    def compute(self):
        residual = np.copy(self.matrix)
        r = self.parameter['r']
        n, m = self.matrix.shape
        self.left_factor = np.zeros([n, r])
        self.right_factor = np.zeros([m, r])
        for col in xrange(r):
            curr_u, sigma, curr_v = power_iteration\
                (residual,\
                    self.parameter['q'], \
                    self.parameter['right_regularizer'],\
                    self.parameter['regularization'],\
                    self.parameter['max_iter'])
            self.left_factor[:, col] = curr_u * np.sqrt(sigma)
            self.right_factor[:, col] = curr_v * np.sqrt(sigma)
            residual -= sigma * np.outer(curr_u, curr_v)


def truncate_normalize(v, k=np.inf, right_regularizer='truncate', regularization_parameter=1.0):

    if k < len(v):
        if right_regularizer == 'k_support':

            v = vector_squared_k_support.prox(v, regularization_parameter, k)
            norm_squared = np.sum(v ** 2)
        elif right_regularizer == 'l1':
            v = l1.prox(v, regularization_parameter)
            norm_squared = np.sum(v ** 2)
        else:
            threshold = sorted(np.abs(v))[-k]
            norm_squared = 0.0
            for i, x in enumerate(v):
                if np.abs(x) < threshold:
                    v[i] = 0.0
                else:
                    norm_squared += x**2
    else:
        norm_squared = np.sum(v**2)
    return v / np.sqrt(TOLERANCE + norm_squared)


def power_iteration(input_matrix, q, right_regularizer='', regularization_parameter=1.0, n_iter=10):
    u, _, v = svds(input_matrix, k=1)
    v = np.array(v.T).flatten()
    u = np.array(u).flatten()
    for _ in xrange(n_iter):
        u = input_matrix.dot(v)
        u /= np.linalg.norm(u)
        v = input_matrix.T.dot(u)
        v = truncate_normalize(v, q, right_regularizer, regularization_parameter)
    return u, u.T.dot(input_matrix.dot(v)), v