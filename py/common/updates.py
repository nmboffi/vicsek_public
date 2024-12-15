"""
Nicholas M. Boffi
4/1/24

Update functions for learning.
"""

import jax
from typing import Callable, Tuple, Dict
from jax import value_and_grad
import functools
import optax
import ml_collections.config_dict as config_dict

Parameters = Dict[str, Dict]


@functools.partial(jax.jit, static_argnums=(2, 3))
def update(
    params: Parameters,
    opt_state: optax.OptState,
    opt: optax.GradientTransformation,
    loss_func: Callable[[Parameters], float],
    loss_func_args: Tuple = tuple(),
) -> Tuple[Parameters, optax.OptState, float, Parameters]:
    """Update the neural network.

    Args:
        params: Parameters to optimize over.
        opt_state: State of the optimizer.
        opt: Optimizer itself.
        loss_func: Loss function for the parameters.
    """
    loss_value, grads = value_and_grad(loss_func)(params, *loss_func_args)
    updates, opt_state = opt.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, grads


@functools.partial(
    jax.pmap,
    in_axes=(0, 0, None, None, 0),
    static_broadcasted_argnums=(2, 3),
    axis_name="data",
)
def pupdate(
    params: Parameters,
    opt_state: optax.OptState,
    opt: optax.GradientTransformation,
    loss_func: Callable[[Parameters], float],
    loss_func_args: Tuple = tuple(),
) -> Tuple[Parameters, optax.OptState, float, Parameters]:
    """Update the neural network using data paralellism.

    Args:
        params: Parameters to optimize over.
        opt_state: State of the optimizer.
        opt: Optimizer itself.
        loss_func: Loss function for the parameters.
    """
    loss_value, grads = jax.value_and_grad(loss_func)(params, *loss_func_args)
    loss_value = jax.lax.pmean(loss_value, axis_name="data")
    grads = jax.lax.pmean(grads, axis_name="data")
    updates, opt_state = opt.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, grads


@functools.partial(jax.jit, static_argnums=2)
def update_ema_params(
    curr_params: Parameters,
    ema_params: Dict[float, Parameters],
    cfg: config_dict.FrozenConfigDict,
) -> Dict[float, Parameters]:
    """Update EMA parameters for elliptic (two network) learning."""
    new_ema_params = {}
    for ema_fac in cfg.ema_facs:
        curr_ema_params = {}
        for key, params in curr_params.items():
            curr_ema_params[key] = jax.tree_util.tree_map(
                lambda param, ema_param: ema_fac * ema_param + (1 - ema_fac) * param,
                params,
                ema_params[ema_fac][key],
            )
        new_ema_params[ema_fac] = curr_ema_params

    return new_ema_params
