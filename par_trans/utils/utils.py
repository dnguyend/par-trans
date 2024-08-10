"""Common util functions
"""
import numpy as np


def asym(a):
    """asymmetrize
    """
    return 0.5*(a-a.T)


def sym(a):
    """symmetrize
    """
    return 0.5*(a+a.T)


def lie(a, b):
    """Lie bracket
    """
    return a@b - b@a


def vcat(x, y):
    """vertical concatenate
    """
    return np.concatenate([x, y], axis=0)


def hcat(x, y):
    """horizontal concatenate
    """
    return np.concatenate([x, y], axis=1)


def cz(a):
    """ check if zero
    """
    return np.max(np.abs(a))
