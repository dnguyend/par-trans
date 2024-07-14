"""Common util functions
"""
import jax.numpy as jnp
from jax import random


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
    return jnp.concatenate([x, y], axis=0)


def hcat(x, y):
    """horizontal concatenate
    """
    return jnp.concatenate([x, y], axis=1)


def  grand(key, shape):
    """ random with key
    """
    key, sk = random.split(key)
    return random.normal(sk, shape), key


def cz(a):
    """ check if zero
    """
    return jnp.max(jnp.abs(a))
