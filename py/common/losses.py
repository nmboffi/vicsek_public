"""
Nicholas M. Boffi
4/1/24

Loss functions for learning.
"""

import jax
from jax.flatten_util import ravel_pytree
import flax.linen as nn
import jax.numpy as np
from typing import Tuple, Callable, Dict
import functools
import ml_collections.config_dict as config_dict
import common.systems as systems

Parameters = Dict[str, Dict]


@jax.jit
def compute_grad_norm(grads: Dict) -> float:
    """Computes the norm of the gradient, where the gradient is input
    as an hk.Params object (treated as a PyTree)."""
    flat_params = ravel_pytree(grads)[0]
    return np.linalg.norm(flat_params) / np.sqrt(flat_params.size)


def mean_reduce(func):
    """
    A decorator that computes the mean of the output of the decorated function.
    Designed to be used on functions that are already batch-processed (e.g., with jax.vmap).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batched_outputs = func(*args, **kwargs)

        # Check if the output is a tuple of batched arrays
        if isinstance(batched_outputs, tuple):
            # Compute the mean for each item in the tuple and return a tuple of means
            return tuple(np.mean(batch_item) for batch_item in batched_outputs)
        else:
            # If it's a single batched array, just return the mean
            return np.mean(batched_outputs)

    return wrapper


def stratonovich_loss(
    params: Parameters,
    xs: np.ndarray,  # [N, d]
    gs: np.ndarray,  # [N, d]
    noises: np.ndarray,  # [2N, d]
    *,
    cfg: config_dict.FrozenConfigDict = None,
    net: nn.Module = None,
) -> jax.Array:
    """Stratonovich loss for the velocity."""
    xgs = np.concatenate((xs, gs))
    loss = 0

    for key in params.keys():
        vel = net.apply(params[key], xs, gs)  # [N, d]

        for flip in range(2):
            noises = noises if flip == 0 else -noises
            xgs_next = cfg.system.step(xgs, noises)
            xs_next, gs_next = np.split(xgs_next, 2)
            vel_next = net.apply(params[key], xs_next, gs_next)  # [N, d]

            if key == "x":
                target = systems.wrapped_diff(cfg.width, xs_next, xs)
            else:
                target = gs_next - gs

            loss += cfg.dt_online * np.sum(vel_next**2) - np.sum(
                (vel + vel_next) * target
            )

    return loss / (2 * xs.size)
