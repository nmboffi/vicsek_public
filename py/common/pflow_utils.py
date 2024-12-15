"""
Nicholas M. Boffi

Code for simulation of the probabilty flow, given a learned network.
"""

import jax
import jax.numpy as np
from typing import Dict, Callable
from . import systems as systems
import functools


def pflow_rhs(
    xgs: np.ndarray,
    params: Dict,
    system: systems.System,
    loss_type: str,
    eps: float,
    gamma: float,
    net: Callable,
) -> np.ndarray:  # [2N, d]
    xs, gs = np.split(xgs, 2)
    bxs, bgs = np.split(system.rhs(xgs), 2)

    if loss_type == "score_matching":
        if eps > 0:
            xdots = bxs - eps * net.apply(params["x"], xs, gs)
        else:
            xdots = bxs

        if system.rescale_type == "none":
            gdots = bgs - gamma * net.apply(params["g"], xs, gs)
        else:
            gdots = bgs - net.apply(params["g"], xs, gs)

    elif loss_type == "stratonovich":
        if eps > 0:
            xdots = net.apply(params["x"], xs, gs)
        else:
            xdots = bxs

        gdots = net.apply(params["g"], xs, gs)

    return xdots, gdots


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 6))
@functools.partial(jax.vmap, in_axes=(0, None, None, None, None, None, None))
def map_pflow_rhs(
    xgs: np.ndarray,
    params: Dict,
    system: systems.System,
    loss_type: str,
    eps: float,
    gamma: float,
    net: Callable,
) -> np.ndarray:  # [2N, d]
    return pflow_rhs(xgs, params, system, loss_type, eps, gamma, net)


def step_pflow(
    xgs: np.ndarray,
    params: Dict,
    system: systems.System,
    loss_type: str,
    eps: float,
    gamma: float,
    net: Callable,
    dt: float,
    width: float,  # [2N, d]
) -> np.ndarray:
    xs, gs = np.split(xgs, 2)
    xdots, gdots = pflow_rhs(xgs, params, system, loss_type, eps, gamma, net)
    xnexts = systems.torus_project(xs + dt * xdots, width)
    gnexts = gs + dt * gdots
    return np.concatenate((xnexts, gnexts))


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 6))
def rollout_traj_pflow(
    init_xgs: np.ndarray,  # [2N, d]
    params: Dict,
    system: systems.System,
    loss_type: str,
    eps: float,
    gamma: float,
    net: Callable,
    dt: float,
    width: float,
    steps: np.ndarray,  # [nsteps]
) -> np.ndarray:

    def scan_fn(xg: np.ndarray, step: np.ndarray):
        del step
        xgnext = step_pflow(xg, params, system, loss_type, eps, gamma, net, dt, width)
        return xgnext, xgnext

    xg_final, xg_traj = jax.lax.scan(scan_fn, init_xgs, steps)
    del xg_final

    return xg_traj


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 6))
@functools.partial(jax.vmap, in_axes=(0, None, None, None, None, None, None))
def rollout_trajs_pflow(
    init_xgs: np.ndarray,  # [2N, d]
    params: Dict,
    system: systems.System,
    loss_type: str,
    eps: float,
    gamma: float,
    net: Callable,
    dt: float,
    width: float,
    steps: np.ndarray,  # [nsteps]
) -> np.ndarray:
    return rollout_traj_pflow(
        init_xgs, params, system, loss_type, eps, gamma, net, dt, width, steps
    )
