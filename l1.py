import numpy as np

# Note: v is numpy array, either a matrix or a vector


def prox(v, regularization_parameter):
    abs_part = np.abs(v) - regularization_parameter
    return v * (abs_part * (abs_part > 0.))


def value(v):
    return np.sum(np.abs(v))


def dual(v):
    np.max(np.abs(v))