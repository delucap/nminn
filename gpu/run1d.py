"""1D study: method-informed stepper vs unstructured stepper vs four physics-informed baselines.

All methods see the same noisy snapshots on the training window and are scored on the same grid
over the full horizon, training window plus extrapolation. Usage:

    python run1d.py [A|B|both] [--seeds 3] [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from experiment import (groundtruth_A, groundtruth_B, rollout, rollout_B,   # noqa: E402
                        train_stepper, train_stepper_B)
from nminn import NFVConserv, NFVScalar, NMINNConserv, NMINNScalar          # noqa: E402
from pinn1d import PINN1D                                                    # noqa: E402

relL2 = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def scores(pred, truth, ntrain, conservative):
    """pred, truth: (nt, ...) on the same grid. Returns the common metric set."""
    n = min(len(pred), len(truth))
    p, t = np.asarray(pred[:n]), np.asarray(truth[:n])
    out = dict(relL2=relL2(p, t) if np.all(np.isfinite(p)) else float("inf"),
               relL2_extrap=relL2(p[ntrain:], t[ntrain:]) if np.all(np.isfinite(p[ntrain:])) else float("inf"),
               min=float(p.min()) if np.all(np.isfinite(p)) else float("-inf"),
               steps=n)
    if conservative:
        m = p.reshape(n, -1).sum(axis=1)
        out["mass_drift"] = float(np.max(np.abs(m - m[0]) / abs(m[0]))) if np.all(np.isfinite(m)) else float("inf")
    return out


# ============================================================ problem A
def run_A(seeds=3, quick=False, device="cuda"):
    M, L = 64, 1.0
    h, dt = L / M, 0.004
    ntrain, ntest = 40, 120
    x, traj = groundtruth_A(M, h, dt, ntrain + ntest)
    T = (ntrain + ntest) * dt
    rng = np.random.default_rng(7)
    noise = 0.02
    train = traj[:ntrain + 1] + noise * traj.max() * rng.standard_normal(traj[:ntrain + 1].shape)
    res = {}

    # ---- method-informed and unstructured steppers (unchanged numpy reference implementation)
    nm = NMINNScalar(M, h, dt)
    t0 = time.time()
    th_nm, l_nm = train_stepper(nm.step, np.array([np.log(0.05), 1.8, -2.0]), train)
    sec_nm = time.time() - t0
    r_nm, _, _ = rollout(nm.step, th_nm, traj[0], ntrain + ntest)
    logd, a, b = nm.unpack(th_nm)
    res["NMINN"] = scores(r_nm, traj, ntrain, False) | dict(
        train_mse=float(l_nm), seconds=sec_nm, coef=dict(d=float(np.exp(logd)), r=float(a), b=float(b)))

    nf = NFVScalar(M, h, dt, H=8, seed=1)
    t0 = time.time()
    th_nf, l_nf = train_stepper(nf.step, nf.theta0, train)
    sec_nf = time.time() - t0
    r_nf, _, _ = rollout(nf.step, th_nf, traj[0], ntrain + ntest)
    res["neural stepper"] = scores(r_nf, traj, ntrain, False) | dict(
        train_mse=float(l_nf), seconds=sec_nf, coef=None)

    # ---- physics-informed baselines
    xd, td, ud = data_points_A(x, train, dt)
    tgrid = np.arange(ntrain + ntest + 1) * dt
    ai, li = (2000, 500) if quick else (15000, 3000)
    for variant in ("vanilla", "penalty", "hardpos"):
        runs = []
        for s in range(seeds):
            p = PINN1D("A", T, L, variant=variant, seed=s, device=device)
            info = p.fit(xd, td, ud, adam_iters=ai, lbfgs_iters=li)
            pred = p.predict(x, tgrid).cpu().numpy()[:, 0, :]
            sc = scores(pred, traj, ntrain, False)
            sc |= dict(seconds=info["seconds"], coef=dict(d=info["coef"][0], r=info["coef"][1]),
                       data_mse=info["data"], res_mse=info["res"])
            runs.append((sc, pred))
            print(f"  PINN-{variant} seed{s}: relL2={sc['relL2']:.3e} min={sc['min']:.3e} "
                  f"d={info['coef'][0]:.4f} r={info['coef'][1]:.3f}", flush=True)
        res[f"PINN {variant}"] = aggregate(runs)
        res[f"PINN {variant}"]["_pred"] = best_pred(runs)
    np.savez(os.path.join(HERE, "resA_gpu.npz"), x=x, traj=traj, dt=dt, ntrain=ntrain,
             r_nm=r_nm, r_nf=r_nf,
             **{f"pinn_{v}": res[f"PINN {v}"].pop("_pred") for v in ("vanilla", "penalty", "hardpos")})
    return res


def data_points_A(x, train, dt):
    nt = train.shape[0]
    tt = np.arange(nt) * dt
    X, T = np.meshgrid(x, tt, indexing="ij")
    return X.ravel(), T.ravel(), train.T.reshape(-1, 1)


# ============================================================ problem B
def run_B(seeds=3, quick=False, device="cuda"):
    M, L = 64, 1.0
    h, dt = L / M, 0.004
    ntrain, ntest = 40, 200
    x, traj = groundtruth_B(M, h, dt, ntrain + ntest)          # (nt, 2, M)
    T = (ntrain + ntest) * dt
    rng = np.random.default_rng(11)
    train = traj[:ntrain + 1] + 0.02 * traj.max() * rng.standard_normal(traj[:ntrain + 1].shape)
    res = {}

    nm = NMINNConserv(M, h, dt)
    t0 = time.time()
    th_nm, l_nm = train_stepper_B(nm.step, np.array([np.log(0.02), np.log(0.05), 0.0]), train)
    sec_nm = time.time() - t0
    r_nm, _, _, _ = rollout_B(nm.step, th_nm, traj[0], ntrain + ntest)
    res["NMINN"] = scores(r_nm, traj, ntrain, True) | dict(
        train_mse=float(l_nm), seconds=sec_nm,
        coef=dict(d1=float(np.exp(th_nm[0])), d2=float(np.exp(th_nm[1])),
                  k=float(np.log1p(np.exp(-abs(th_nm[2]))) + max(th_nm[2], 0.0))))

    nf = NFVConserv(M, h, dt, H=10, seed=2)
    t0 = time.time()
    th_nf, l_nf = train_stepper_B(nf.step, nf.theta0, train)
    sec_nf = time.time() - t0
    r_nf, _, _, _ = rollout_B(nf.step, th_nf, traj[0], ntrain + ntest)
    res["neural stepper"] = scores(r_nf, traj, ntrain, True) | dict(
        train_mse=float(l_nf), seconds=sec_nf, coef=None)

    nt = train.shape[0]
    tt = np.arange(nt) * dt
    X, Tm = np.meshgrid(x, tt, indexing="ij")
    xd, td = X.ravel(), Tm.ravel()
    ud = np.transpose(train, (2, 0, 1)).reshape(-1, 2)          # (M*nt, 2)
    mass0 = float(traj[0].sum())
    tgrid = np.arange(ntrain + ntest + 1) * dt
    ai, li = (2000, 500) if quick else (15000, 3000)
    preds = {}
    for variant in ("vanilla", "penalty", "hardpos", "project"):
        runs = []
        base = "hardpos" if variant == "project" else variant
        for s in range(seeds):
            p = PINN1D("B", T, L, variant=base, seed=s, device=device)
            info = p.fit(xd, td, ud, adam_iters=ai, lbfgs_iters=li, mass0=mass0)
            pred = p.predict(x, tgrid).cpu().numpy()                       # (nt, 2, M)
            if variant == "project":
                m = pred.reshape(len(tgrid), -1).sum(axis=1)
                pred = pred * (mass0 / m)[:, None, None]
            sc = scores(pred, traj, ntrain, True)
            sc |= dict(seconds=info["seconds"],
                       coef=dict(d1=info["coef"][0], d2=info["coef"][1], k=info["coef"][2]),
                       data_mse=info["data"], res_mse=info["res"])
            runs.append((sc, pred))
            print(f"  PINN-{variant} seed{s}: relL2={sc['relL2']:.3e} min={sc['min']:.3e} "
                  f"drift={sc['mass_drift']:.3e}", flush=True)
        res[f"PINN {variant}"] = aggregate(runs)
        preds[variant] = best_pred(runs)
        res[f"PINN {variant}"].pop("_pred", None)
    np.savez(os.path.join(HERE, "resB_gpu.npz"), x=x, traj=traj, dt=dt, ntrain=ntrain,
             r_nm=r_nm, r_nf=r_nf, **{f"pinn_{k}": v for k, v in preds.items()})
    return res


# ============================================================ helpers
def aggregate(runs):
    keys = [k for k in runs[0][0] if isinstance(runs[0][0][k], float)]
    out = {}
    for k in keys:
        vals = np.array([r[0][k] for r in runs], dtype=float)
        out[k] = float(np.median(vals))
        out[k + "_min"], out[k + "_max"] = float(vals.min()), float(vals.max())
    out["coef"] = runs[int(np.argmin([r[0]["relL2"] for r in runs]))][0]["coef"]
    out["seeds"] = len(runs)
    return out


def best_pred(runs):
    return runs[int(np.argmin([r[0]["relL2"] for r in runs]))][1]


def show(title, res, conservative):
    print(f"\n=== {title} ===")
    hdr = f"{'method':<18}{'relL2':>11}{'relL2 extrap':>14}{'min u':>13}"
    if conservative:
        hdr += f"{'mass drift':>13}"
    hdr += f"{'train s':>10}"
    print(hdr)
    for k, v in res.items():
        line = f"{k:<18}{v['relL2']:>11.3e}{v['relL2_extrap']:>14.3e}{v['min']:>13.3e}"
        if conservative:
            line += f"{v['mass_drift']:>13.3e}"
        line += f"{v['seconds']:>10.1f}"
        print(line)
    for k, v in res.items():
        if v.get("coef"):
            print(f"  {k:<16} recovered {v['coef']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="both", choices=["A", "B", "both"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    np.random.seed(0)
    out = {}
    if args.which in ("A", "both"):
        out["A"] = run_A(args.seeds, args.quick, args.device)
        show("Problem A: Fisher-KPP, true d=0.05 r=6", out["A"], False)
    if args.which in ("B", "both"):
        out["B"] = run_B(args.seeds, args.quick, args.device)
        show("Problem B: two-species conservative exchange, true d1=0.03 d2=0.08 k=6", out["B"], True)
    with open(os.path.join(HERE, "results_1d.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nwrote results_1d.json")
