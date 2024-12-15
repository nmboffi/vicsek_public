"""
Nicholas M. Boffi
4/1/24

Helper routines for neural network definitions.
"""

import jax
import jax.numpy as np
from typing import Callable
import flax.linen as nn

from . import systems


class MLP(nn.Module):
    """Simple MLP network with square weight pattern."""

    n_hidden: int
    n_neurons: int
    output_dim: int
    act: Callable
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: np.ndarray):
        for _ in range(self.n_hidden):
            x = nn.Dense(self.n_neurons, use_bias=self.use_bias)(x)
            x = self.act(x)

        x = nn.Dense(self.output_dim)(x)
        return x


def get_neighbors(
    xi: np.ndarray, xs: np.ndarray, width: float, n_neighbors: int
) -> np.ndarray:
    """Find the nearest neighbors of a particle in a set of particles.

    Args:
        xi: Particle to find neighbors for.
        xs: Set of particles.
        width: Width of the domain.
        n_neighbors: Number of neighbors to find.
    """
    xdiffs = systems.map_wrapped_diff(width, xi, xs)
    norms = np.linalg.norm(xdiffs, axis=1)
    inds = jax.lax.top_k(-norms, n_neighbors + 1)[1]
    return inds[1:], xdiffs[inds[1:]]
