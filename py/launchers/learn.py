"""
Nicholas M. Boffi
3/12/24

Code for systematic learning of general systems
"""

import sys

sys.path.append("../../py")

import jax
import jax.numpy as np
import numpy as onp
import dill as pickle
from typing import Tuple, Callable, Dict
from ml_collections import config_dict
from copy import deepcopy
import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import functools
from tqdm.auto import tqdm as tqdm
import wandb
from flax.jax_utils import replicate, unreplicate
import optax
import common.networks as networks
import common.systems as systems
import common.losses as losses
import common.updates as updates
from typing import Callable, Tuple


####### sensible matplotlib defaults #######
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.grid.which"] = "both"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["grid.color"] = "0.8"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["figure.figsize"] = (8, 4)
mpl.rcParams["figure.titlesize"] = 7.5
mpl.rcParams["font.size"] = 10
mpl.rcParams["legend.fontsize"] = 7.5
mpl.rcParams["figure.dpi"] = 300
############################################


Parameters = Dict[str, Dict]


def train_loop(
    prng_key: np.ndarray,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    data: dict,
) -> None:
    """Carry out the training loop."""
    ## set up data and output
    loss = setup_loss()
    params = replicate(data["params"])
    ema_params = {ema_fac: deepcopy(data["params"]) for ema_fac in cfg.ema_facs}
    xgs = data["xgs"]

    ## perform training
    for curr_epoch in tqdm(range(cfg.n_epochs), desc="epochs"):
        # take some steps of the dynamics to avoid overfitting
        xgs = onp.array(xgs)
        xgs, prng_key = step_data(xgs, prng_key)
        onp.random.shuffle(xgs)
        xgs = jax.device_put(xgs, jax.devices("gpu")[0])

        # split into xs and gs
        xs, gs = np.split(xgs, 2, axis=1)  # [ntrajs, N, d], [ntrajs, N, d]

        batch_iterator = tqdm(
            range(cfg.nbatches),
            desc=f"epoch {curr_epoch+1}/{cfg.n_epochs}",
            leave=False,
        )
        for curr_batch in batch_iterator:
            ## take a step on the loss
            loss_fn_args, prng_key = setup_loss_fn_args(xs, gs, prng_key, curr_batch)
            xbatch, gbatch = loss_fn_args[0], loss_fn_args[1]
            params, opt_state, loss_value, grads = updates.pupdate(
                params, opt_state, opt, loss, loss_fn_args
            )

            ## compute EMA params
            curr_params = unreplicate(params)
            ema_params = updates.update_ema_params(curr_params, ema_params, cfg)

            ## log loss, statistics, and score norm
            data = log_metrics(
                data,
                curr_params,
                ema_params,
                xbatch,
                gbatch,
                grads,
                loss_value[0],
                curr_batch,
                curr_epoch,
            )

            batch_iterator.set_postfix({"loss": f"{loss_value[0]:.4f}"})

    # dump one final time
    pickle.dump(data, open(f"{cfg.output_folder}/{cfg.output_name}.npy", "wb"))


@jax.jit
def calc_divs(
    params: Parameters,
    xs: np.ndarray,  # [N, d]
    gs: np.ndarray,  # [N, d]
) -> Tuple:  # [N]
    """Compute score and velocity divergences."""

    xgs = np.concatenate((xs, gs))  # [2*N, d]
    if cfg.loss_type == "stratonovich":
        div_bxs, div_bgs = np.split(cfg.system.div_rhs(xgs), 2)  # [N]

        if cfg.eps > 0:
            div_vxs = net.apply(params["x"], xs, gs, "x", method="particle_div")  # [N]
        else:
            div_vxs = div_bxs

        div_vgs = net.apply(params["g"], xs, gs, "g", method="particle_div")  # [N]

        if cfg.eps > 0:
            div_sxs = (div_bxs - div_vxs) / cfg.eps  # [N]
        else:
            div_sxs = np.zeros_like(div_vxs)

        if cfg.rescale_type == "none":
            div_sgs = (div_bgs - div_vgs) / cfg.gamma  # [N]
        else:
            div_sgs = div_bgs - div_vgs

    else:
        raise ValueError(f"Loss type {cfg.loss_type} not implemented for calc_divs.")

    div_vs = div_vxs + div_vgs  # [N]

    return div_sxs, div_sgs, div_vxs, div_vgs, div_vs


@jax.jit
def calc_vs(
    params: Parameters,
    xgs: np.ndarray,  # [2*N, d]
) -> Tuple:
    xs, gs = np.split(xgs, 2)  # ([N, d], [N, d])
    if cfg.loss_type == "stratonovich":
        bxs, bgs = np.split(cfg.system.rhs(xgs), 2)  # ([N, d], [N, d])
        if cfg.eps > 0:
            vxs = net.apply(params["x"], xs, gs)  # [N, d]
        else:
            vxs = bxs
        vgs = net.apply(params["g"], xs, gs)  # [N, d]
        if cfg.eps > 0:
            sxs = (bxs - vxs) / cfg.eps  # [N, d]
        else:
            sxs = np.zeros_like(vxs)

        if cfg.rescale_type == "none":
            sgs = (bgs - vgs) / cfg.gamma  # [N, d]
        else:
            sgs = bgs - vgs

    else:
        raise ValueError(f"Loss type {cfg.loss_type} not implemented for calc_vs.")

    vs = np.hstack((vxs, vgs))  # [N, 2*d]
    return sxs, sgs, vxs, vgs, vs


@jax.jit
def compute_output_info(
    xgs: np.ndarray,
    params: Parameters,
) -> Tuple:
    """Compute the entropy, activity, etc."""
    xs, gs = np.split(xgs, 2)  # ([N, d], [N, d])
    sxs, sgs, vxs, vgs, vs = calc_vs(params, xgs)
    div_sxs, div_sgs, div_vxs, div_vgs, div_vs = calc_divs(params, xs, gs)
    gdot_mags = np.linalg.norm(vgs, axis=1)  # [N]
    xdot_mags = np.linalg.norm(vxs, axis=1)  # [N]
    scores = np.hstack((sxs, sgs))  # [N, 2*d]
    particle_score_mags = np.linalg.norm(scores, axis=1)  # [N]
    x_score_mags = np.linalg.norm(sxs, axis=1)  # [N]
    g_score_mags = np.linalg.norm(sgs, axis=1)  # [N]
    v_times_s = np.zeros(cfg.N)

    return (
        xs,
        gs,
        gdot_mags,
        xdot_mags,
        particle_score_mags,
        x_score_mags,
        g_score_mags,
        vs,
        v_times_s,
        div_vs,
        div_vxs,
        div_vgs,
        div_sxs,
        div_sgs,
    )


@jax.jit
def compute_oned_output_info(
    xgs: np.ndarray,
    params: Parameters,
) -> Tuple:
    """Compute the entropy, activity, etc."""
    xs, gs = np.split(xgs, 2)  # ([N, d], [N, d])
    sxs, sgs, vxs, vgs, _ = calc_vs(params, xgs)
    div_sxs, div_sgs, div_vxs, div_vgs, _ = calc_divs(params, xs, gs)

    return (
        xs,
        gs,
        vxs,
        vgs,
        sxs,
        sgs,
        div_vxs,
        div_vgs,
        div_sxs,
        div_sgs,
    )


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, None))
def map_oned_output_info(
    xgs: np.ndarray,
    params: Parameters,
) -> Tuple:
    return compute_oned_output_info(xgs, params)


def make_entropy_plot(
    params: Parameters,
    data: dict,
) -> None:
    # compute quantities needed for plotting
    ind = onp.random.randint(cfg.ntrajs)
    (
        xs,
        gs,
        gdot_mags,
        xdot_mags,
        particle_score_mags,
        x_score_mags,
        g_score_mags,
        vs,
        v_times_s,
        div_vs,
        div_vxs,
        div_vgs,
        div_sxs,
        div_sgs,
    ) = compute_output_info(data["xgs"][ind], params)

    seifert_entropy = np.zeros(cfg.N)

    # common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fraction = 0.15
    shrink = 0.5
    fontsize = 12.5

    ###### main entropy figure
    titles = [
        [
            r"$\Vert\dot{g}\Vert$",
            r"$\Vert\dot{x}\Vert$",
            r"$\Vert v \Vert_{D^{-1}}^2$",
        ],
        [r"$\Vert s_g\Vert$", r"$\Vert s_x\Vert$", r"$\Vert s \Vert$"],
        [r"$\nabla_g\cdot v_g$", r"$\nabla_x\cdot v_x$", r"$\nabla\cdot v$"],
        [r"$\nabla_g\cdot s_g$", r"$\nabla_x\cdot s_x$", r"$v \cdot s$"],
    ]

    cs = [
        [gdot_mags, xdot_mags, seifert_entropy],
        [g_score_mags, x_score_mags, particle_score_mags],
        [div_vgs, div_vxs, div_vs],
        [div_sgs, div_sxs, v_times_s],
    ]

    cmaps = [
        sns.color_palette("mako", as_cmap=True),
        sns.color_palette("mako", as_cmap=True),
        sns.color_palette("mako", as_cmap=True),
        sns.color_palette("mako", as_cmap=True),
    ]

    ### x entropy figure
    nrows = len(titles)
    ncols = len(titles[0])
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    axs = axs.reshape((nrows, ncols))

    for ax in axs.ravel():
        ax.set_xlim([-cfg.width, cfg.width])
        ax.set_ylim([-cfg.width, cfg.width])
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.axes.set_aspect(1.0)
        ax.set_facecolor("black")
        scale = ax.transData.get_matrix()[0, 0]
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for ii in range(nrows):
        for jj in range(ncols):
            title = titles[ii][jj]
            c = cs[ii][jj]
            ax = axs[ii, jj]
            ax.set_title(title, fontsize=fontsize)
            vmin = np.quantile(c, 0.05)
            vmax = np.quantile(c, 0.95)

            # for div_g v_g
            if ii == 2 and jj == 0:
                vmin = min(vmin, -vmax)
                vmax = max(vmax, -vmin)
                cmap = sns.color_palette("icefire", as_cmap=True)
            else:
                cmap = cmaps[ii]

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

            if cfg.system_type == "mips":
                scale = ax.transData.get_matrix()[0, 0]
                scat = ax.scatter(
                    xs[:, 0],
                    xs[:, 1],
                    s=np.pi * (scale * cfg.r) ** 2,
                    marker="o",
                    c=c,
                    cmap=cmap,
                    norm=norm,
                )
            else:
                gs = jax.vmap(lambda g: g / np.linalg.norm(g))(gs)
                try:
                    ax.quiver(
                        xs[:, 0],
                        xs[:, 1],
                        gs[:, 0],
                        gs[:, 1],
                        angles="xy",
                        scale=15.0,
                        zorder=1,
                        color=cmap(norm(c)),
                    )
                except:
                    print(
                        "Error in quiver plot, likely related to vmin, vmax business."
                    )

            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                fraction=fraction,
                shrink=shrink,
            )
            cbar.ax.tick_params(labelsize=fontsize)

    wandb.log({"entropy_figure": wandb.Image(fig)})


def make_entropy_plot_oned(
    params: Parameters,
    data: dict,
) -> None:
    # compute quantities needed for plotting
    inds = onp.random.choice(onp.arange(cfg.ntrajs), size=int(1e4), replace=True)
    (
        xs,
        gs,
        vxs,
        vgs,
        sxs,
        sgs,
        div_vxs,
        div_vgs,
        div_sxs,
        div_sgs,
    ) = map_oned_output_info(data["xgs"][inds], params)

    # compute displacement coordinates (all of shape [bs])
    xs = np.squeeze(systems.wrapped_diff(cfg.width, xs[:, 0], xs[:, 1]))
    gs = np.squeeze(gs[:, 0] - gs[:, 1])
    vxs = np.squeeze(vxs[:, 0] - vxs[:, 1])
    vgs = np.squeeze(vgs[:, 0] - vgs[:, 1])
    sxs = np.squeeze(sxs[:, 0] - sxs[:, 1])
    sgs = np.squeeze(sgs[:, 0] - sgs[:, 1])

    # note the plus with divergences -- this follows because
    # \nabla_{g^2} = -\nabla_{g}
    div_vxs = np.squeeze(div_vxs[:, 0] + div_vxs[:, 1])
    div_vgs = np.squeeze(div_vgs[:, 0] + div_vgs[:, 1])
    div_sxs = np.squeeze(div_sxs[:, 0] + div_sxs[:, 1])
    div_sgs = np.squeeze(div_sgs[:, 0] + div_sgs[:, 1])

    # compute full divergences
    div_vs = div_vxs + div_vgs

    # stack to construct vectors
    vs = np.stack((vxs, vgs), axis=1)  # [bs, 2]
    ss = np.stack((sxs, sgs), axis=1)  # [bs, 2]

    if cfg.eps > 0:
        v_times_s = np.sum(vs * ss, axis=1)  # [bs]
    else:
        v_times_s = np.zeros_like(div_vs)

    # compute norms
    gdot_mags = np.abs(vgs)
    xdot_mags = np.abs(vxs)
    x_score_mags = np.abs(sxs)
    g_score_mags = np.abs(sgs)
    v_mags = np.linalg.norm(vs, axis=1)
    particle_score_mags = np.linalg.norm(ss, axis=1)

    # compute bounds
    min_g, max_g = np.min(gs), np.max(gs)
    min_g = min(min_g, -max_g)
    max_g = max(max_g, -min_g)

    # common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fraction = 0.15
    shrink = 0.5
    fontsize = 12.5

    ###### main entropy figure
    titles = [
        [
            r"$\Vert\dot{g}\Vert$",
            r"$\Vert\dot{x}\Vert$",
            r"$\Vert v \Vert$",
        ],
        [r"$\Vert s_g\Vert$", r"$\Vert s_x\Vert$", r"$\Vert s \Vert$"],
        [r"$\nabla_g\cdot v_g$", r"$\nabla_x\cdot v_x$", r"$\nabla\cdot v$"],
        [r"$\nabla_g\cdot s_g$", r"$\nabla_x\cdot s_x$", r"$-v \cdot s$"],
    ]

    cs = [
        [gdot_mags, xdot_mags, v_mags],
        [g_score_mags, x_score_mags, particle_score_mags],
        [div_vgs, div_vxs, div_vs],
        [div_sgs, div_sxs, -v_times_s],
    ]

    cmap_sequential = sns.color_palette("mako", as_cmap=True)
    cmap_diverging = sns.color_palette("icefire", as_cmap=True)

    nrows = len(titles)
    ncols = len(titles[0])
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    axs = axs.reshape((nrows, ncols))

    for ax in axs.ravel():
        ax.set_xlim([-cfg.width, cfg.width])
        ax.set_ylim([min_g, max_g])
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)
        ax.set_facecolor("black")

    # do the plotting
    for ii in range(nrows):
        for jj in range(ncols):
            title = titles[ii][jj]
            c = cs[ii][jj]
            ax = axs[ii, jj]
            ax.set_title(title, fontsize=fontsize)

            vmin = np.quantile(c, 0.05)
            vmax = np.quantile(c, 0.95)

            # for div(v) and v^Ts, symmetrize (since E = 0 ideally)
            if title == r"$\nabla\cdot v$" or title == r"$-v \cdot s$":
                vmin = -max(abs(vmin), abs(vmax))
                vmax = max(abs(vmin), abs(vmax))
                cmap = cmap_diverging
            else:
                cmap = cmap_sequential

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            scat = ax.scatter(
                xs,
                gs,
                s=1.0,
                alpha=0.5,
                marker="o",
                c=c,
                cmap=cmap,
                norm=norm,
            )
            cbar = fig.colorbar(scat, ax=ax, fraction=fraction, shrink=shrink)
            cbar.ax.tick_params(labelsize=fontsize)
            scat.set_clim(vmin, vmax)

    wandb.log({"entropy_figure": wandb.Image(fig)})


@jax.jit
@losses.mean_reduce
@functools.partial(jax.vmap, in_axes=(None, 0))
def compute_convergence_statistics(
    params: Parameters,
    xgs: np.ndarray,  # [2*N, d]
) -> None:
    xs, gs = np.split(xgs, 2)

    _, _, _, _, div_vs = calc_divs(params, xs, gs)  # [N]
    div_v = np.sum(div_vs)

    sxs, sgs, vxs, vgs, _ = calc_vs(params, xgs)  # [N, d] for all
    if cfg.learn_sx:
        v_times_s = np.sum(sxs * vxs) + np.sum(sgs * vgs)
    else:
        v_times_s = np.zeros(cfg.N)

    return div_v, v_times_s, (div_v + v_times_s) ** 2


def step_data(
    xgs: onp.ndarray,  # [ntrajs, 2N, d]
    prng_key: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    for curr_batch in range(cfg.nbatches_online):
        lb = cfg.bs_online * curr_batch
        ub = lb + cfg.bs_online
        batch_xgs = xgs[lb:ub]
        noises = jax.random.normal(
            prng_key, shape=(batch_xgs.shape[0], cfg.nsteps_online, 2 * cfg.N, cfg.d)
        )
        xgs[lb:ub], _ = systems.rollout_trajs(batch_xgs, noises, cfg.system.step)
        prng_key = jax.random.split(prng_key)[0]

    return xgs, prng_key


def setup_loss_fn_args(
    xs: np.ndarray,  # [ntrajs, N, d]
    gs: np.ndarray,  # [ntrajs, N, d]
    prng_key: np.ndarray,
    curr_batch: int,
) -> Tuple:
    lb = cfg.bs * curr_batch
    ub = lb + cfg.bs
    xbatch, gbatch = xs[lb:ub], gs[lb:ub]
    xbatch = xbatch.reshape((cfg.ndevices, -1, cfg.N, cfg.d))
    gbatch = gbatch.reshape((cfg.ndevices, -1, cfg.N, cfg.d))
    loss_fn_args = (xbatch, gbatch)

    if cfg.loss_type == "stratonovich":
        noise_batch = jax.random.normal(
            prng_key, shape=(xbatch.shape[0], xbatch.shape[1], 2 * cfg.N, cfg.d)
        )
        loss_fn_args += (noise_batch,)

    key = jax.random.split(prng_key)[0]
    return loss_fn_args, key


def log_metrics(
    data: dict,
    curr_params: Parameters,
    ema_params: Dict[float, Parameters],
    xbatch: onp.ndarray,
    gbatch: onp.ndarray,
    grads: Parameters,
    loss_value: float,
    curr_batch: int,
    curr_epoch: int,
) -> None:
    score_norm = 0
    xbatch = xbatch.reshape((-1, cfg.N, cfg.d))  # [bs, N, d]
    gbatch = gbatch.reshape((-1, cfg.N, cfg.d))  # [bs, N, d]

    for key in curr_params.keys():
        scores = map_net(curr_params[key], xbatch, gbatch)  # [bs, N, d]

    score_norm += np.mean(np.sum(scores**2, axis=(1, 2))) / (2 * cfg.N * cfg.d)
    iteration = curr_batch + curr_epoch * cfg.nbatches
    wandb.log(
        {
            f"loss": loss_value,
            f"score_norm": -score_norm,
            f"grad": losses.compute_grad_norm(unreplicate(grads)),
            f"learning_rate": schedule(iteration),
        }
    )

    if (iteration % cfg.stat_freq) == 0:
        div_v, v_times_s, pinn = 0, 0, 0
        for curr_batch in range(cfg.nbatches_stats):
            lb = curr_batch * cfg.bs_stats
            ub = lb + cfg.bs_stats
            stat_batch = np.concatenate(
                (xbatch[lb:ub], gbatch[lb:ub]), axis=1
            )  # [bs_stats, 2N, d]
            curr_div_v, curr_v_times_s, curr_pinn = compute_convergence_statistics(
                curr_params, stat_batch
            )

            div_v += curr_div_v / cfg.nbatches_stats
            v_times_s += curr_v_times_s / cfg.nbatches_stats
            pinn += curr_pinn / cfg.nbatches_stats

        # note: cannot compute v^Ts (and therefore cannot compute pinn) for stratonovich
        wandb.log({"div_v": div_v})
        if cfg.eps > 0 or cfg.learn_sx:
            wandb.log({"v_times_s": v_times_s, "pinn": pinn})

    if (iteration % cfg.visual_freq) == 0:
        if cfg.d == 1:
            make_entropy_plot_oned(curr_params, data)
        else:
            make_entropy_plot(curr_params, data)

    if (iteration % cfg.save_freq) == 0:
        data["params"] = jax.device_put(curr_params, jax.devices("cpu")[0])
        data["ema_params"] = jax.device_put(ema_params, jax.devices("cpu")[0])
        pickle.dump(
            data,
            open(
                f"{cfg.output_folder}/{cfg.output_name}_{iteration//cfg.save_freq}.npy",
                "wb",
            ),
        )

    return data


def setup_loss() -> Callable:
    if cfg.loss_type == "stratonovich":

        @losses.mean_reduce
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def loss(params, xs, gs, noises):
            return losses.stratonovich_loss(params, xs, gs, noises, cfg=cfg, net=net)

    else:
        raise ValueError("Specified loss is not implemented.")

    return loss


def setup_system(cfg: config_dict.ConfigDict) -> systems.System:
    if cfg.system_type == "vicsek":
        cfg.system = systems.Vicsek(
            cfg.dt_online,
            cfg.r,
            cfg.v0,
            cfg.gamma,
            cfg.width,
            cfg.eps,
            cfg.d,
            cfg.N,
            cfg.beta,
            cfg.k,
            cfg.A,
            cfg.gstar_mag,
            cfg.rescale_type,
        )
    elif cfg.system_type == "mips":
        # ensure hashable
        cfg.system = systems.MIPS(
            float(cfg.dt_online),
            float(cfg.v0),
            float(cfg.gamma),
            float(cfg.width),
            float(cfg.eps),
            int(cfg.d),
            int(cfg.N),
            float(cfg.A),
            float(cfg.k),
            float(cfg.r),
            float(cfg.beta),
        )
    else:
        raise ValueError("Specified system is not implemented.")

    return cfg


def setup_config_dict():
    cfg = config_dict.ConfigDict()
    cfg.clip = 1.0

    cfg.max_n_steps = 50
    cfg.n_epochs = int(1e6)

    ## input parameters
    args = get_simulation_parameters()
    cfg.nsteps_online = args.nsteps_online
    cfg.loss_type = args.loss_type
    cfg.system_type = args.system_type
    cfg.v0 = args.v0
    cfg.phi = args.phi
    cfg.A = args.A
    cfg.gstar_mag = args.gstar_mag
    cfg.k = args.k
    cfg.beta = args.beta
    cfg.dt = args.dt
    cfg.dt_online = args.dt_online
    cfg.N = args.N
    cfg.gamma = args.gamma
    cfg.eps = args.eps
    cfg.rescale_type = args.rescale_type
    cfg.learn_sx = args.learn_sx
    cfg.ema_facs = [0.999, 0.9999]
    cfg.network_type = args.network_type
    cfg.share_encoder = args.share_encoder
    cfg.x_translation_invariant = args.x_translation_invariant
    cfg.g_translation_invariant = args.g_translation_invariant
    cfg.rotation_invariant = args.rotation_invariant
    cfg.use_residual = args.use_residual
    cfg.use_layernorm = args.use_layernorm
    cfg.layernorm_type = args.layernorm_type
    cfg.sum_pool = args.sum_pool
    cfg.n_hidden = args.n_hidden
    cfg.n_neurons = args.n_neurons
    cfg.num_gnn_layers = args.num_gnn_layers
    cfg.num_transformer_layers = args.num_transformer_layers
    cfg.num_heads = args.num_heads
    cfg.n_neighbors = args.n_neighbors
    cfg.learning_rate = args.learning_rate
    cfg.decay_steps = args.decay_steps
    cfg.warmup_steps = args.warmup_steps
    cfg.wandb_name = f"{args.wandb_name}_{args.slurm_id}"
    cfg.wandb_project = args.wandb_project
    cfg.output_folder = args.output_folder
    cfg.output_name = f"{args.output_name}_{args.slurm_id}"
    cfg.ntrajs_max = args.ntrajs_max
    cfg.bs = args.bs
    cfg.bs_stats = args.bs_stats
    cfg.bs_online = args.bs_online
    cfg.visual_freq = args.visual_freq
    cfg.stat_freq = args.stat_freq
    cfg.save_freq = args.save_freq
    cfg.ndevices = jax.local_device_count()

    # check for incompatibilities
    if cfg.eps > 0 and (not cfg.learn_sx):
        raise ValueError("Cannot have eps > 0 and learn_sx = False.")

    return cfg, args


def construct_network(
    cfg: config_dict.ConfigDict,
) -> Tuple[Callable, Callable, Callable, Callable]:
    if cfg.network_type == "simple_gnn":
        net = networks.SimpleGNN(cfg.d, cfg.N, cfg.n_hidden, cfg.n_neurons, cfg.width)

    elif cfg.network_type == "deepset_gnn":
        net = networks.DeepsetGNN(
            cfg.d,
            cfg.N,
            cfg.n_neighbors,
            cfg.n_hidden,
            cfg.n_neurons,
            cfg.width,
            cfg.share_encoder,
            cfg.sum_pool,
            cfg.x_translation_invariant,
            cfg.g_translation_invariant,
            cfg.use_residual,
            cfg.use_layernorm,
        )

    elif cfg.network_type == "two_particle_mlp":
        net = networks.TwoParticleMLP(cfg.d, cfg.n_hidden, cfg.n_neurons, cfg.width)

    else:
        raise ValueError("Network type not defined!")

    ## define map_net function
    @jax.jit
    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def map_net(params, xs, gs):
        return net.apply(params, xs, gs)

    return net, map_net


def load_data(
    cfg: config_dict.ConfigDict, key: np.ndarray
) -> Tuple[onp.ndarray, np.ndarray, config_dict.ConfigDict]:
    """Load in a dataset and update the configuration accordingly."""

    cfg.data_folder = args.data_folder

    try:
        data_name = (
            f"v0={cfg.v0}_gamma={cfg.gamma}_eps={cfg.eps}_phi={cfg.phi}"
            + f"_dt={cfg.dt}_beta={cfg.beta}_A={cfg.A}_k={cfg.k}_gstarmag={cfg.gstar_mag}"
            + f"_N={cfg.N}_rescale_type={cfg.rescale_type}"
        )
        cfg.data_path = f"{cfg.data_folder}/{data_name}.npy"
        print("loading data from", cfg.data_path)
        data_dict = pickle.load(open(cfg.data_path, "rb"))
    except:
        print("Failed! Removing gstar mag and rescale_type")
        data_name = (
            f"v0={cfg.v0}_gamma={cfg.gamma}_eps={cfg.eps}_phi={cfg.phi}"
            + f"_dt={cfg.dt}_beta={cfg.beta}_A={cfg.A}_k={cfg.k}_N={cfg.N}"
        )
        cfg.data_path = f"{cfg.data_folder}/{data_name}.npy"
        print("loading data from", cfg.data_path)
        data_dict = pickle.load(open(cfg.data_path, "rb"))
        assert cfg.rescale_type == "none", "Rescale type must be none if not saved!"

    xgs = pickle.load(open(cfg.data_path, "rb"))["xgs"]
    print(f"Dataset shape: {xgs.shape}")

    if cfg.ntrajs_max > 0:
        xgs = xgs[: cfg.ntrajs_max]
        print(f"Trimmed dataset to shape: {xgs.shape}")

    cfg.ntrajs = xgs.shape[0]
    cfg.nbatches = int(cfg.ntrajs / cfg.bs)
    cfg.nbatches += 1 if cfg.nbatches * cfg.bs < cfg.ntrajs else 0
    cfg.width = data_dict["width"]

    # update batch parameters that have a dependence on ntrajs
    cfg.nbatches = int(cfg.ntrajs / cfg.bs)
    cfg.nbatches += 1 if cfg.nbatches * cfg.bs < cfg.ntrajs else 0
    cfg.nbatches_online = int(cfg.ntrajs / cfg.bs_online)
    cfg.nbatches_online += 1 if cfg.nbatches_online * cfg.bs_online < cfg.ntrajs else 0
    cfg.nbatches_stats = max(int(cfg.bs / cfg.bs_stats), 1)

    # update the system in the config
    cfg.r = data_dict["r"]
    cfg.d = data_dict["d"]

    print(f"Particle radius r={cfg.r}.")
    return onp.array(xgs), key, cfg


def get_simulation_parameters():
    """Process command line arguments and set up associated simulation parameters."""
    parser = argparse.ArgumentParser(description="Elliptic learning.")

    parser.add_argument("--nsteps_online", type=int)
    parser.add_argument("--network_path", type=str)
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--ntrajs_max", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--bs_stats", type=int)
    parser.add_argument("--bs_online", type=int)
    parser.add_argument("--visual_freq", type=int)
    parser.add_argument("--stat_freq", type=int)
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--dt_online", type=float)
    parser.add_argument("--N", type=int)
    parser.add_argument("--v0", type=float)
    parser.add_argument("--phi", type=float)
    parser.add_argument("--A", type=float)
    parser.add_argument("--gstar_mag", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--rescale_type", type=str)
    parser.add_argument("--learn_sx", type=int)
    parser.add_argument("--network_type", type=str)
    parser.add_argument("--share_encoder", type=int)
    parser.add_argument("--x_translation_invariant", type=int)
    parser.add_argument("--g_translation_invariant", type=int)
    parser.add_argument("--rotation_invariant", type=int)
    parser.add_argument("--use_residual", type=int)
    parser.add_argument("--use_layernorm", type=int)
    parser.add_argument("--layernorm_type", type=str)
    parser.add_argument("--sum_pool", type=int)
    parser.add_argument("--n_neurons", type=int)
    parser.add_argument("--n_hidden", type=int)
    parser.add_argument("--num_gnn_layers", type=int)
    parser.add_argument("--num_transformer_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--n_neighbors", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--decay_steps", type=int)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--system_type", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--slurm_id", type=int)

    return parser.parse_args()


def initialize_network(prng_key: np.ndarray):
    if args.network_path != "":
        loaded_dict = pickle.load(open(args.network_path, "rb"))
        params = deepcopy(jax.device_put(loaded_dict["params"], jax.devices("cpu")[0]))

        # ensure we clear the memory
        del loaded_dict
    else:
        ex_xs, ex_gs = np.split(xgs[0], 2)
        key1, key2 = jax.random.split(prng_key)
        prng_key = jax.random.split(key1)[0]

        params = {}
        params["g"] = {"params": net.init(key2, ex_xs, ex_gs)["params"]}

        if cfg.learn_sx:
            params["x"] = {"params": net.init(key1, ex_xs, ex_gs)["params"]}

        print(
            f"Parameters per network: {jax.flatten_util.ravel_pytree(params['g'])[0].size}"
        )

    return params, prng_key


if __name__ == "__main__":
    cfg, args = setup_config_dict()
    prng_key = jax.random.PRNGKey(onp.random.randint(1000))
    xgs, prng_key, cfg = load_data(cfg, prng_key)
    cfg = setup_system(cfg)
    cfg = config_dict.FrozenConfigDict(cfg)  # freeze the config

    ## define and initialize the neural network
    net, map_net = construct_network(cfg)
    params, prng_key = initialize_network(prng_key)

    ## define optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=int(cfg.warmup_steps),
        decay_steps=int(cfg.decay_steps),
    )

    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip), optax.radam(learning_rate=schedule)
    )

    # for parallel training
    opt_state = replicate(opt.init(params))

    ## set up weights and biases tracking
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        config=cfg.to_dict(),
    )

    ## train the model
    data = {
        "params": jax.device_put(params, jax.devices("cpu")[0]),
        "xgs": xgs,
        "cfg": cfg,
    }

    train_loop(prng_key, opt, opt_state, data)
