"""
Nicholas M. Boffi
3/13/24

General code for dataset generation.
"""

import jax
import numpy as onp
from jax import numpy as np
import dill as pickle
import sys

sys.path.append("../")
import common.systems as systems
import argparse
import time
from functools import partial
from tqdm.auto import tqdm as tqdm
from pathlib import Path


def generate_data() -> np.ndarray:
    xs = systems.torus_project(sig0x * onp.random.randn(ntrajs, N, d), width)
    gs = sig0g * onp.random.randn(ntrajs, N, d)
    xgs = onp.concatenate((xs, gs), axis=1)
    key = jax.random.PRNGKey(onp.random.randint(100000))

    start_time = time.time()
    print(f"Starting data generation.")
    for _ in tqdm(range(nbatches)):
        noises = jax.random.normal(key, shape=(ntrajs, nsteps_batch, 2 * N, d))
        xgs, _ = systems.rollout_trajs(xgs, noises, system.step)
        key = jax.random.split(key)[0]
    end_time = time.time()
    print(f"Finished data generation. Total time={(end_time-start_time)/60.}m")

    return onp.array(xgs)


def get_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float)
    parser.add_argument("--phi", type=float)
    parser.add_argument("--v0", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--d", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--A", type=float)
    parser.add_argument("--gstar_mag", type=float)
    parser.add_argument("--rescale_type", type=str)
    parser.add_argument("--ntrajs", type=int)
    parser.add_argument("--nbatches", type=int)
    parser.add_argument("--thermalize_fac", type=float)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--slurm_id", type=int)
    parser.add_argument("--system_type", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    ## standard system parameters
    print("Entering main. Loading command line arguments.")
    n_prints = 100

    ## command line arguments
    args = get_cmd_arguments()
    dt = args.dt
    phi = args.phi
    v0 = args.v0
    gamma = args.gamma
    eps = args.eps
    d = args.d
    N = args.N
    beta = args.beta
    k = args.k
    A = args.A
    gstar_mag = args.gstar_mag
    rescale_type = args.rescale_type
    ntrajs = args.ntrajs
    dim = 2 * N * d

    ## define system geometry
    width = 1.0
    if d == 2:
        r = onp.sqrt(4 * phi * width**2 / (N * onp.pi))
    else:
        r = width * phi / N

    ## setup system
    if args.system_type == "vicsek":
        system = systems.Vicsek(
            dt, r, v0, gamma, width, eps, d, N, beta, k, A, gstar_mag, rescale_type
        )

    elif args.system_type == "mips":
        system = systems.MIPS(dt, v0, gamma, width, eps, d, N, A, k, r, beta)
    else:
        raise ValueError("Invalid system type!")

    sig0x = width / 2
    sig0g = 1 / np.sqrt(gamma) if rescale_type == "g" else 1.0

    ## set up output
    name = (
        f"N{N}_v0={v0}_gam={gamma}_eps={eps}_phi={phi}_dt={dt}"
        + f"_beta={beta}_A={A}_k={k}_gstarmag={gstar_mag}_rescale={rescale_type}"
    )
    output_folder = f"{args.output_folder}/{name}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    ## thermalization parameters
    nbatches = args.nbatches
    thermalize_fac = args.thermalize_fac
    tf = thermalize_fac / gamma
    nsteps_total = int(tf / dt) + 1
    nsteps_batch = nsteps_total // nbatches

    # generate data and set up storage
    data_dict = {
        "gamma": gamma,
        "eps": eps,
        "v0": v0,
        "phi": phi,
        "dt": dt,
        "beta": beta,
        "A": A,
        "k": k,
        "N": N,
        "tf": tf,
        "width": width,
        "r": r,
        "d": d,
        "gstar_mag": gstar_mag,
        "rescale_type": rescale_type,
        "ntrajs": ntrajs,
        "nbatches": nbatches,
        "thermalize_fac": thermalize_fac,
        "xgs": generate_data(),
    }

    print(f"Dumping data to {output_folder}/{args.slurm_id}.npy")
    pickle.dump(data_dict, open(f"{output_folder}/{args.slurm_id}.npy", "wb"))
    print(f"Successfully dumped the data!")
