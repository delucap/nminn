"""Train one physics-informed baseline (problem, variant, seed) and dump its prediction.

Kept as a single-run script so that the runs can be executed concurrently on one GPU: each of
them is far too small to saturate the device on its own.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from data1d import collocation_points
from pinn1d import PINN1D

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["A", "B"], required=True)
    ap.add_argument("--variant", choices=["vanilla", "penalty", "hardpos", "project"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adam", type=int, default=10000)
    ap.add_argument("--lbfgs", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "pinn_runs"))
    ap.add_argument("--noise_seed", type=int, default=None, help="noise realization (default: paper)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d, xd, td, ud = collocation_points(a.problem, a.noise_seed)
    base = "hardpos" if a.variant == "project" else a.variant
    mass0 = float(d["traj"][0].sum()) if a.problem == "B" else None
    p = PINN1D(a.problem, d["T"], d["cfg"]["L"], variant=base, seed=a.seed, device=a.device)
    info = p.fit(xd, td, ud, adam_iters=a.adam, lbfgs_iters=a.lbfgs, mass0=mass0)
    pred = p.predict(d["x"], d["tgrid"]).cpu().numpy()
    if a.problem == "A":
        pred = pred[:, 0, :]
    if a.variant == "project":
        m = pred.reshape(len(d["tgrid"]), -1).sum(axis=1)
        pred = pred * (mass0 / m).reshape(-1, *([1] * (pred.ndim - 1)))
    tag = f"{a.problem}_{a.variant}_{a.seed}"
    np.savez(os.path.join(a.out, f"pinn_{tag}.npz"), pred=pred)
    with open(os.path.join(a.out, f"pinn_{tag}.json"), "w") as f:
        json.dump(dict(problem=a.problem, variant=a.variant, seed=a.seed, noise_seed=a.noise_seed,
                       adam=a.adam, lbfgs=a.lbfgs, **info), f, indent=2)
    print(f"done {tag}: loss={info['loss']:.3e} coef={info['coef']} {info['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
