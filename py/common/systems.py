"""
Nicholas M. Boffi
4/1/24

Systems for learning.
"""

import jax
import jax.numpy as np
from typing import Callable, Tuple
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass


def torus_project(xs: np.ndarray, width: float):
    return ((xs + width) % (2 * width)) - width


def softplus(x: np.ndarray, beta: float) -> np.ndarray:
    return jax.nn.softplus(beta * x) / beta


def d_softplus(x: float, beta: float):
    return 1.0 / (1.0 + np.exp(-x * beta))


class System(ABC):

    @abstractmethod
    def rhs(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def div_rhs(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, state: np.ndarray, noise: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def target(self, xs: np.ndarray, gs: np.ndarray, key: str) -> np.ndarray:
        pass


@dataclass
class Vicsek(System):
    """Vicsek model with local alignment and (possibly) positional noise."""

    dt: float
    r: float
    v0: float
    gamma: float
    width: float
    eps: float
    d: int
    N: int
    beta: float
    k: float
    A: float
    gstar_mag: float
    rescale_type: str

    def __post_init__(self) -> None:
        if self.rescale_type == "time":
            self.speed = self.v0 / self.gamma
            self.decay_rate = 1.0
            self.interaction_strength = self.k / self.gamma
            self.D = np.concatenate(
                (
                    np.ones((self.N, self.d)) * self.eps,
                    np.ones((self.N, self.d)),
                )
            )
        elif self.rescale_type == "g":
            self.speed = self.v0 * np.sqrt(self.gamma)
            self.decay_rate = self.gamma
            self.interaction_strength = self.k
            self.D = np.concatenate(
                (
                    np.ones((self.N, self.d)) * self.eps,
                    np.ones((self.N, self.d)),
                )
            )
        elif self.rescale_type == "none":
            self.speed = self.v0
            self.decay_rate = self.gamma
            self.interaction_strength = self.k
            self.D = np.concatenate(
                (
                    np.ones((self.N, self.d)) * self.eps,
                    np.ones((self.N, self.d)) * self.gamma,
                )
            )
        else:
            raise ValueError(f"Unknown rescale type {self.rescale_type}")

        if self.beta == 0:
            self.kernel = jax.jit(
                lambda x: self.interaction_strength * (np.sum(x**2) < (2 * self.r) ** 2)
            )
        else:
            self.kernel = jax.jit(
                lambda x: self.interaction_strength
                * jax.nn.sigmoid(-self.beta * (np.sum(x**2) - 4 * self.r**2))
            )

        # TODO: I think that this only makes sense without rescaling.
        if self.d == 2:
            self.gstar = self.gstar_mag * np.array([1.0, 0])
        else:
            self.gstar = self.gstar_mag * np.array([1.0])

    # note that jitting with self is fine, because the class is immutable
    @functools.partial(jax.jit, static_argnums=0)
    def rhs(
        self,
        xgs: np.ndarray,  # [2*N, d]
    ) -> np.ndarray:
        xs, gs = np.split(xgs, 2)
        xdots = self.speed * gs
        interactions = self.interaction(gs, gs, xs, xs)
        gdots = (
            -self.decay_rate * gs
            + np.sum(interactions, axis=1)
            + self.A * (self.gstar[None, :] - gs)
        )
        return np.concatenate((xdots, gdots))

    @functools.partial(jax.jit, static_argnums=0)
    def div_rhs(
        self,
        xgs: np.ndarray,  # [2*N, d]
    ) -> np.ndarray:
        xs, _ = np.split(xgs, 2)  # [N, d]
        div_interactions = self.div_interaction(xs, xs)  # [N, N]

        return np.concatenate(
            (
                np.zeros(self.N),
                -self.decay_rate * self.d
                + np.sum(div_interactions, axis=1)
                + self.d * self.kernel(np.zeros(self.d))
                - self.A * self.d,
            )
        )

    @functools.partial(jax.jit, static_argnums=0)
    def step(
        self,
        xgs: np.ndarray,  # [2*N, d]
        noise: np.ndarray,  # [2*N, d]
    ) -> np.ndarray:
        """Euler-Maruyama step on the system."""
        xgs_next = xgs + self.rhs(xgs) * self.dt + np.sqrt(2 * self.D * self.dt) * noise
        xs_next, gs_next = np.split(xgs_next, 2)
        xs_next = torus_project(xs_next, self.width)
        return np.concatenate((xs_next, gs_next))

    @functools.partial(jax.vmap, in_axes=(None, 0, None, 0, None))
    @functools.partial(jax.vmap, in_axes=(None, None, 0, None, 0))
    def interaction(
        self,
        gi: np.ndarray,  # [d]
        gj: np.ndarray,  # [d]
        xi: np.ndarray,  # [d]
        xj: np.ndarray,  # [d]
    ) -> np.ndarray:
        """Local Vicsek interaction."""
        return (gj - gi) * self.kernel(xi - xj)

    @functools.partial(jax.vmap, in_axes=(None, 0, None))
    @functools.partial(jax.vmap, in_axes=(None, None, 0))
    def div_interaction(
        self,
        xi: np.ndarray,  # [d]
        xj: np.ndarray,  # [d]
    ) -> np.ndarray:
        """Divergence of local Vicsek interaction w.r.t gi."""
        return -self.d * self.kernel(xi - xj)

    def target(self, xs: np.ndarray, gs: np.ndarray, key: str) -> np.ndarray:
        """Target for supervised debugging."""
        raise NotImplementedError()

    def __hash__(self):
        return hash(
            (
                self.dt,
                self.r,
                self.v0,
                self.gamma,
                self.width,
                self.eps,
                self.d,
                self.N,
                self.beta,
                self.k,
                self.A,
                self.gstar_mag,
                self.rescale_type,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, Vicsek):
            return False
        return (
            self.dt == other.dt
            and self.r == other.r
            and self.v0 == other.v0
            and self.gamma == other.gamma
            and self.width == other.width
            and self.eps == other.eps
            and self.d == other.d
            and self.N == other.N
            and self.beta == other.beta
            and self.A == other.A
            and self.gstar_mag == other.gstar_mag
            and self.rescale_type == other.rescale_type
        )


@dataclass
class MIPS(System):
    """MIPS model with (possibly) positional noise."""

    dt: float
    v0: float
    gamma: float
    width: float
    eps: float
    d: int
    N: int
    A: float
    k: float
    r: float
    beta: float

    def __post_init__(self) -> None:
        self.D = np.concatenate(
            (
                np.ones((self.N, self.d)) * self.eps,
                np.ones((self.N, self.d)) * self.gamma,
            )
        )

        self.sigmoid_prime = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2

    @functools.partial(jax.jit, static_argnums=0)
    def rhs(
        self,
        xgs: np.ndarray,  # [2*N, d]
    ) -> np.ndarray:
        xs, gs = np.split(xgs, 2)
        interactions = self.interaction(xs, xs)
        xdots = self.v0 * gs + np.sum(interactions, axis=1) - self.A * xs
        gdots = -self.gamma * gs

        return np.concatenate((xdots, gdots))

    @functools.partial(jax.jit, static_argnums=0)
    def div_rhs(
        self,
        xgs: np.ndarray,  # [2*N, d]
    ) -> np.ndarray:
        xs, _ = np.split(xgs, 2)
        div_interactions = self.div_interaction(xs, xs)

        return np.concatenate(
            (
                np.sum(div_interactions, axis=1) - self.A * self.d,
                -self.gamma * self.d * np.ones(self.N),
            )
        )

    @functools.partial(jax.jit, static_argnums=0)
    def step(
        self,
        xgs: np.ndarray,  # [2*N, d]
        noise: np.ndarray,  # [2*N, d]
    ) -> np.ndarray:
        """Euler-Maruyama step on the system."""
        xgs_next = xgs + self.rhs(xgs) * self.dt + np.sqrt(2 * self.D * self.dt) * noise
        xs_next, gs_next = np.split(xgs_next, 2)
        xs_next = torus_project(xs_next, self.width)
        return np.concatenate((xs_next, gs_next))

    @functools.partial(jax.jit, static_argnums=0)
    def kernel(self, xdiff: np.ndarray) -> np.ndarray:
        norm_sq = np.sum(xdiff**2)
        return jax.lax.cond(
            norm_sq == 0,
            lambda _: np.zeros_like(xdiff),
            lambda _: self.k
            * jax.nn.sigmoid(-self.beta * (norm_sq - 4 * self.r**2))
            * (xdiff / np.sqrt(norm_sq)),
            operand=None,
        )

    @functools.partial(jax.jit, static_argnums=0)
    def div_kernel(self, xdiff: np.ndarray) -> float:
        norm_sq = np.sum(xdiff**2)
        norm = np.sqrt(norm_sq)

        def div_nonzero(_) -> float:
            """split divergence for legibility"""
            sigmoid_argument = -self.beta * (norm_sq - 4 * self.r**2)

            sigmoid_div = (
                -2 * self.k * self.beta * norm * self.sigmoid_prime(sigmoid_argument)
            )

            xhat_div = self.k * (self.d - 1) / norm * jax.nn.sigmoid(sigmoid_argument)

            return sigmoid_div + xhat_div

        return jax.lax.cond(
            norm_sq == 0,
            lambda _: 0.0,
            div_nonzero,
            operand=None,
        )

    @functools.partial(jax.jit, static_argnums=0)
    @functools.partial(jax.vmap, in_axes=(None, 0, None))
    @functools.partial(jax.vmap, in_axes=(None, None, 0))
    def interaction(
        self,
        xi: np.ndarray,  # [d]
        xj: np.ndarray,  # [d]
    ) -> np.ndarray:
        """Local MIPS interaction."""
        return self.kernel(xi - xj)

    @functools.partial(jax.jit, static_argnums=0)
    @functools.partial(jax.vmap, in_axes=(None, 0, None))
    @functools.partial(jax.vmap, in_axes=(None, None, 0))
    def div_interaction(
        self,
        xi: np.ndarray,  # [d]
        xj: np.ndarray,  # [d]
    ) -> np.ndarray:
        """Divergence of local MIPS interaction w.r.t xi."""
        return self.div_kernel(xi - xj)

    def target(self, xs: np.ndarray, gs: np.ndarray, key: str) -> np.ndarray:
        """Target for supervised debugging."""
        raise NotImplementedError()

    # make possible to jit with self
    def __hash__(self):
        return hash(
            (
                self.dt,
                self.v0,
                self.gamma,
                self.width,
                self.eps,
                self.d,
                self.N,
                self.A,
                self.k,
                self.r,
                self.beta,
            )
        )

    # make possible to jit with self
    def __eq__(self, other):
        if not isinstance(other, MIPS):
            return False

        return (
            self.dt == other.dt
            and self.v0 == other.v0
            and self.gamma == other.gamma
            and self.width == other.width
            and self.eps == other.eps
            and self.d == other.d
            and self.N == other.N
            and self.A == other.A
            and self.k == other.k
            and self.r == other.r
            and self.beta == other.beta
        )


@functools.partial(jax.jit, static_argnums=2)
def rollout(
    init_xg: np.ndarray,  # [2N, d]
    noises: np.ndarray,  # [nsteps, 2*N, d]
    step_system: Callable,
) -> Tuple[jax.Array, jax.Array]:

    def scan_fn(xg: np.ndarray, noise: np.ndarray):
        xgnext = step_system(xg, noise)
        return xgnext, xgnext

    xg_final, xg_traj = jax.lax.scan(scan_fn, init_xg, noises)
    return xg_final, xg_traj


@functools.partial(jax.jit, static_argnums=2)
@functools.partial(jax.vmap, in_axes=(0, 0, None))
def rollout_trajs(
    init_xgs: np.ndarray,  # [ntrajs, 2*N, d]
    noises: np.ndarray,  # [ntrajs, nsteps, 2*N, d]
    step_system: Callable,
) -> np.ndarray:
    return rollout(init_xgs, noises, step_system)


@jax.jit
def wrapped_diff(width: float, x: np.ndarray, y: np.ndarray) -> jax.Array:
    d = x - y
    return d - 2 * width * np.rint(d / (2 * width))


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, None, 0))
def map_wrapped_diff(width: float, x: np.ndarray, y: np.ndarray) -> jax.Array:
    return wrapped_diff(width, x, y)


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0, None))
def compute_wrapped_diffs(width: float, x: np.ndarray, ys: np.ndarray) -> jax.Array:
    return map_wrapped_diff(width, x, ys)
