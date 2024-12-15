"""
Nicholas M. Boffi
4/1/24

Neural networks for learning probability flows.
"""

import flax.linen as nn
import jax.numpy as np
import jax
import functools
from typing import Tuple
import jraph
from . import systems
from . import network_utils
from . import transformers


class TwoParticleMLP(nn.Module):
    """Define separate scores in x and g for a very simplified two-particle system"""

    d: int
    n_hidden: int
    n_neurons: int
    width: float

    def setup(self):
        self.mlp = network_utils.MLP(
            self.n_hidden, self.n_neurons, 2 * self.d, nn.activation.gelu
        )
        self.jac_x = jax.jacfwd(self.__call__, argnums=0)
        self.jac_g = jax.jacfwd(self.__call__, argnums=1)

    def __call__(self, xs: np.ndarray, gs: np.ndarray) -> np.ndarray:
        dx = systems.wrapped_diff(self.width, xs[0], xs[1])
        dg = gs[0] - gs[1]
        inp = np.concatenate((dx, dg))
        return (self.mlp(inp) - self.mlp(-inp)).reshape((2, self.d))

    def particle_div(
        self, xs: np.ndarray, gs: np.ndarray, key: str  # [N, d, N, d]  # [N, d, N, d]
    ) -> np.ndarray:
        if key == "x":
            jac = self.jac_x(xs, gs)  # [N, d, N, d]
        elif key == "g":
            jac = self.jac_g(xs, gs)  # [N, d, N, d]
        else:
            raise ValueError("key needs to be x or g.")

        return np.diag(np.trace(jac, axis1=1, axis2=3))

    def total_div(
        self, xs: np.ndarray, gs: np.ndarray, key: str
    ) -> float:  # [N, d]  # [N, d]
        return np.sum(self.particle_div(xs, gs, key))


class SimpleGNN(nn.Module):
    """Simple fully-connected deepset-like neural network ansatz."""

    d: int
    N: int
    n_hidden: int
    n_neurons: int
    width: float

    def setup(self):
        self.encoder = network_utils.MLP(
            n_hidden=self.n_hidden,
            n_neurons=self.n_neurons,
            output_dim=self.n_neurons,
            act=nn.activation.gelu,
            use_bias=True,
        )

        self.decoder = network_utils.MLP(
            n_hidden=self.n_hidden,
            n_neurons=self.n_neurons,
            output_dim=self.d,
            act=nn.activation.gelu,
            use_bias=True,
        )

        # define jacobians
        self._jac_x = jax.jacfwd(self._network, argnums=0)
        self._jac_g = jax.jacfwd(self._network, argnums=1)

        # define divergences
        self._div_x = lambda xi, gi, xs, gs, ii: np.trace(
            self._jac_x(xi, gi, xs, gs, ii)
        )
        self._div_g = lambda xi, gi, xs, gs, ii: np.trace(
            self._jac_g(xi, gi, xs, gs, ii)
        )
        self._div_xi = lambda xs, gs, ii: self._div_x(xs[ii], gs[ii], xs, gs, ii)
        self._div_gi = lambda xs, gs, ii: self._div_g(xs[ii], gs[ii], xs, gs, ii)
        self._map_div_xi = jax.vmap(self._div_xi, in_axes=(None, None, 0))
        self._map_div_gi = jax.vmap(self._div_gi, in_axes=(None, None, 0))

    def _network(
        self, xi: np.ndarray, gi: np.ndarray, xs: np.ndarray, gs: np.ndarray, ii: int
    ) -> np.ndarray:
        """Main network computation.
        Signature written in this way to allow for easy divergence computation."""
        xdiffs = systems.map_wrapped_diff(self.width, xi, xs)  # [N, d]
        modified_gs = gs.at[ii].set(gi)  # ensure dependence on gi explicitly
        # stack gi horizontally so we obtain (gi, gj) in each row
        g_inp = np.hstack((np.tile(gi, (self.N, 1)), modified_gs))  # [N, 2 * d]
        encoder_in = np.hstack((xdiffs, g_inp))  # [N, 3 * d]
        encoder_out = self.encoder(encoder_in)  # [N, n_neurons]
        pool_out = np.sum(encoder_out, axis=0)  # [n_neurons]
        return self.decoder(pool_out)

    def _particle_i(self, xs: np.ndarray, gs: np.ndarray, ii: int) -> np.ndarray:
        """Evaluate the drift (score or velocity) for a single particle."""
        return self._network(xs[ii], gs[ii], xs, gs, ii)

    @functools.partial(jax.vmap, in_axes=(None, None, None, 0))
    def _map_particle(self, xs: np.ndarray, gs: np.ndarray, ii: int) -> np.ndarray:
        return self._particle_i(xs, gs, ii)

    def __call__(self, xs: np.ndarray, gs: np.ndarray) -> np.ndarray:
        return self._map_particle(xs, gs, np.arange(self.N))

    def particle_div(self, xs: np.ndarray, gs: np.ndarray, key: str) -> np.ndarray:
        if key == "x":
            return self._map_div_xi(xs, gs, np.arange(self.N))
        elif key == "g":
            return self._map_div_gi(xs, gs, np.arange(self.N))
        else:
            raise ValueError("key needs to be x or g.")

    def total_div(self, xs: np.ndarray, gs: np.ndarray, key: str) -> np.ndarray:
        return np.sum(self.particle_div(xs, gs, key))


class DeepsetGNN(nn.Module):
    """Simple deepset-like neural network ansatz."""

    d: int
    N: int
    n_neighbors: int
    n_hidden: int
    n_neurons: int
    width: float
    share_encoder: bool
    sum_pool: bool
    x_translation_invariant: bool
    g_translation_invariant: bool
    use_residual: bool
    use_layernorm: bool

    def setup(self):
        if self.share_encoder:
            self.encoder = network_utils.MLP(
                n_hidden=self.n_hidden,
                n_neurons=self.n_neurons,
                output_dim=self.n_neurons,
                act=nn.activation.gelu,
                use_bias=True,
            )
        else:
            self.x_encoder = network_utils.MLP(
                n_hidden=self.n_hidden,
                n_neurons=self.n_neurons,
                output_dim=self.n_neurons // 2,
                act=nn.activation.gelu,
                use_bias=True,
            )

            self.g_encoder = network_utils.MLP(
                n_hidden=self.n_hidden,
                n_neurons=self.n_neurons,
                output_dim=self.n_neurons // 2,
                act=nn.activation.gelu,
                use_bias=True,
            )

        self.decoder = network_utils.MLP(
            n_hidden=self.n_hidden,
            n_neurons=self.n_neurons,
            output_dim=self.d,
            act=nn.activation.gelu,
            use_bias=False,
        )

        # define jacobians
        self._jac_x = jax.jacfwd(self._network, argnums=0)
        self._jac_g = jax.jacfwd(self._network, argnums=1)

        # define divergences
        self._div_x = lambda xi, gi, xs, gs: np.trace(self._jac_x(xi, gi, xs, gs))
        self._div_g = lambda xi, gi, xs, gs: np.trace(self._jac_g(xi, gi, xs, gs))
        self._div_xi = lambda xs, gs, ii: self._div_x(xs[ii], gs[ii], xs, gs)
        self._div_gi = lambda xs, gs, ii: self._div_g(xs[ii], gs[ii], xs, gs)
        self._map_div_xi = jax.vmap(self._div_xi, in_axes=(None, None, 0))
        self._map_div_gi = jax.vmap(self._div_gi, in_axes=(None, None, 0))

        # define residual connections
        if self.use_residual:
            self.dense = nn.Dense(self.n_neurons)

        if self.use_layernorm:
            self.norm_x = transformers.ParticleNorm()
            self.norm_g = transformers.ParticleNorm()

            if self.share_encoder:
                self.norm_encoder = transformers.ParticleNorm()
            else:
                self.norm_x_encoder = transformers.ParticleNorm()
                self.norm_g_encoder = transformers.ParticleNorm()

    def _network(
        self, xi: np.ndarray, gi: np.ndarray, xs: np.ndarray, gs: np.ndarray
    ) -> np.ndarray:
        """Main network computation.
        Signature written in this way to allow for easy divergence computation."""
        inds, _ = jax.lax.stop_gradient(
            network_utils.get_neighbors(xi, xs, self.width, self.n_neighbors)
        )
        neighbor_xs, neighbor_gs = xs[inds], gs[inds]

        # note we need to recompute xdiffs here to properly take \nabla_x v_x
        if self.x_translation_invariant:
            x_inp = systems.map_wrapped_diff(
                self.width, xi, neighbor_xs
            )  # [n_neighbors, d]

            # for self-interaction term
            if not self.g_translation_invariant:
                x_inp = np.vstack((np.zeros(self.d), x_inp))
        else:
            x_inp = np.vstack((xi, neighbor_xs))  # [n_neighbors + 1, d]

        if self.use_layernorm:
            x_inp = self.norm_x(x_inp)

        if self.g_translation_invariant:
            g_inp = gi[None, :] - neighbor_gs  # [n_neighbors, d]

            if not self.x_translation_invariant:
                g_inp = np.vstack((np.zeros(self.d), g_inp))
        else:
            g_inp = np.vstack((gi, neighbor_gs))

        if self.use_layernorm:
            g_inp = self.norm_g(g_inp)

        # share the encoder between x and g to allow for more interaction between the two
        if self.share_encoder:
            inp = np.hstack((x_inp, g_inp))
            encoder_out = self.encoder(inp)  # [n_neighbors, n_neurons]

            if self.use_layernorm:
                encoder_out = self.norm_encoder(encoder_out)

        # otherwise, encode x and g separately
        else:
            x_enc = self.x_encoder(x_inp)  # [n_neighbors, n_neurons // 2]
            g_enc = self.g_encoder(g_inp)  # [n_neighbors, n_neurons // 2]

            if self.use_layernorm:
                x_enc = self.norm_x_encoder(x_enc)
                g_enc = self.norm_g_encoder(g_enc)

            encoder_out = np.hstack((x_enc, g_enc))  # [n_neighbors, n_neurons]

        if self.use_residual:
            encoder_out = self.dense(inp) + encoder_out  # residual connection

        if self.sum_pool:
            pool_out = np.sum(encoder_out, axis=0)  # [n_neurons]
        else:
            pool_out = np.mean(encoder_out, axis=0)  # [n_neurons]

        return self.decoder(pool_out)

    def _particle_i(self, xs: np.ndarray, gs: np.ndarray, ii: int) -> np.ndarray:
        """Evaluate the drift (score or velocity) for a single particle."""
        return self._network(xs[ii], gs[ii], xs, gs)

    @functools.partial(jax.vmap, in_axes=(None, None, None, 0))
    def _map_particle(self, xs: np.ndarray, gs: np.ndarray, ii: int) -> np.ndarray:
        return self._particle_i(xs, gs, ii)

    def __call__(self, xs: np.ndarray, gs: np.ndarray) -> np.ndarray:
        return self._map_particle(xs, gs, np.arange(self.N))

    def particle_div(self, xs: np.ndarray, gs: np.ndarray, key: str) -> np.ndarray:
        if key == "x":
            return self._map_div_xi(xs, gs, np.arange(self.N))
        elif key == "g":
            return self._map_div_gi(xs, gs, np.arange(self.N))
        else:
            raise ValueError("key needs to be x or g.")

    def total_div(self, xs: np.ndarray, gs: np.ndarray, key: str) -> float:
        return np.sum(self.particle_div(xs, gs, key))
