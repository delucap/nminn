"""Shared data definition for the 1D problems, so every method sees exactly the same inputs."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from experiment import groundtruth_A, groundtruth_B   # noqa: E402

CFG = dict(
    A=dict(M=64, L=1.0, dt=0.004, ntrain=40, ntest=120, noise=0.02, seed=7,
           true=dict(d=0.05, r=6.0)),
    B=dict(M=64, L=1.0, dt=0.004, ntrain=40, ntest=200, noise=0.02, seed=11,
           true=dict(d1=0.03, d2=0.08, k=6.0)),
)


def load(problem, noise_seed=None):
    """noise_seed overrides the seed of the noise realization (the clean trajectory never changes)."""
    c = dict(CFG[problem])
    if noise_seed is not None:
        c["seed"] = int(noise_seed)
    M, L, dt = c["M"], c["L"], c["dt"]
    h = L / M
    n = c["ntrain"] + c["ntest"]
    x, traj = (groundtruth_A if problem == "A" else groundtruth_B)(M, h, dt, n)
    rng = np.random.default_rng(c["seed"])
    tr = traj[:c["ntrain"] + 1]
    train = tr + c["noise"] * traj.max() * rng.standard_normal(tr.shape)
    tgrid = np.arange(n + 1) * dt
    return dict(cfg=c, x=x, h=h, traj=traj, train=train, tgrid=tgrid, T=n * dt)


def collocation_points(problem, noise_seed=None):
    """Data points (x, t, values) built from the noisy training snapshots."""
    d = load(problem, noise_seed)
    x, train, dt = d["x"], d["train"], d["cfg"]["dt"]
    nt = train.shape[0]
    tt = np.arange(nt) * dt
    X, T = np.meshgrid(x, tt, indexing="ij")
    if problem == "A":
        ud = train.T.reshape(-1, 1)
    else:
        ud = np.transpose(train, (2, 0, 1)).reshape(-1, 2)
    return d, X.ravel(), T.ravel(), ud
