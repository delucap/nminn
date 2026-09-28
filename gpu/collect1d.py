"""Collect the 1D study: structure-preserving stepper, unstructured stepper, PINN variants.

The projected variant is obtained post hoc from the hard-positive runs by rescaling every time
slice to the exact initial mass, which is the cheapest way to give a physics-informed surrogate
the conservation property. It costs no extra training and shows what such a projection does to
the accuracy.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from data1d import load                                                     # noqa: E402
from experiment import rollout, rollout_B, train_stepper, train_stepper_B   # noqa: E402
from nminn import NFVConserv, NFVScalar, NMINNConserv, NMINNScalar          # noqa: E402

relL2 = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def scores(pred, truth, ntrain, conservative):
    n = min(len(pred), len(truth))
    p, t = np.asarray(pred[:n], dtype=float), np.asarray(truth[:n], dtype=float)
    fin = bool(np.all(np.isfinite(p)))
    out = dict(relL2=relL2(p, t) if fin else float("inf"),
               relL2_extrap=relL2(p[ntrain:], t[ntrain:]) if fin else float("inf"),
               min=float(p.min()) if fin else float("-inf"))
    if conservative:
        m = p.reshape(n, -1).sum(axis=1)
        out["mass_drift"] = float(np.max(np.abs(m - m[0]) / abs(m[0]))) if fin else float("inf")
    return out


def agg(list_of_scores):
    keys = [k for k in list_of_scores[0] if isinstance(list_of_scores[0][k], float)]
    out = {}
    for k in keys:
        v = np.array([s[k] for s in list_of_scores], float)
        out[k] = float(np.median(v))
        out[k + "_min"], out[k + "_max"] = float(v.min()), float(v.max())
    out["seeds"] = len(list_of_scores)
    return out


def steppers(problem):
    d = load(problem)
    c, M, h, dt = d["cfg"], d["cfg"]["M"], d["h"], d["cfg"]["dt"]
    n = c["ntrain"] + c["ntest"]
    traj, train, ntrain = d["traj"], d["train"], c["ntrain"]
    out = {}
    if problem == "A":
        nm, nf = NMINNScalar(M, h, dt), NFVScalar(M, h, dt, H=8, seed=1)
        t0 = time.time(); th, l = train_stepper(nm.step, np.array([np.log(0.05), 1.8, -2.0]), train); s1 = time.time() - t0
        r_nm, _, _ = rollout(nm.step, th, traj[0], n)
        logd, a, b = nm.unpack(th)
        out["NMINN"] = scores(r_nm, traj, ntrain, False) | dict(
            seconds=s1, train_mse=float(l), params=3,
            coef=dict(d=float(np.exp(logd)), r=float(a), b=float(b)))
        t0 = time.time(); thf, lf = train_stepper(nf.step, nf.theta0, train); s2 = time.time() - t0
        r_nf, _, _ = rollout(nf.step, thf, traj[0], n)
        out["neural stepper"] = scores(r_nf, traj, ntrain, False) | dict(
            seconds=s2, train_mse=float(lf), params=int(nf.nparam), coef=None)
    else:
        nm, nf = NMINNConserv(M, h, dt), NFVConserv(M, h, dt, H=10, seed=2)
        t0 = time.time(); th, l = train_stepper_B(nm.step, np.array([np.log(0.02), np.log(0.05), 0.0]), train); s1 = time.time() - t0
        r_nm, _, _, _ = rollout_B(nm.step, th, traj[0], n)
        sp = float(np.log1p(np.exp(-abs(th[2]))) + max(th[2], 0.0))
        out["NMINN"] = scores(r_nm, traj, ntrain, True) | dict(
            seconds=s1, train_mse=float(l), params=3,
            coef=dict(d1=float(np.exp(th[0])), d2=float(np.exp(th[1])), k=sp))
        t0 = time.time(); thf, lf = train_stepper_B(nf.step, nf.theta0, train); s2 = time.time() - t0
        r_nf, _, _, _ = rollout_B(nf.step, thf, traj[0], n)
        out["neural stepper"] = scores(r_nf, traj, ntrain, True) | dict(
            seconds=s2, train_mse=float(lf), params=int(nf.nparam), coef=None)
    return d, out, r_nm, r_nf


def collect(problem, rundir):
    """Merge the selected learned steppers with the physics-informed runs.

    Every method is selected the same way, by its own TRAINING objective, never by a test quantity.
    For the PINNs the seed spread is reported alongside, so the selection is auditable.
    """
    d = load(problem)
    c, ntrain, traj = d["cfg"], d["cfg"]["ntrain"], d["traj"]
    conservative = problem == "B"
    mass0 = float(traj[0].sum())
    out = {}
    with open(os.path.join(HERE, "select_1d.json")) as f:
        sel = json.load(f)[problem]
    r_nm = np.load(os.path.join(HERE, f"sel_{problem}_NMINN.npy"))
    r_nf = np.load(os.path.join(HERE, f"sel_{problem}_neural.npy"))
    for name in ("NMINN", "neural stepper"):
        out[name] = dict(sel[name])
    preds = {}
    for variant in ("vanilla", "penalty", "hardpos", "project"):
        src = "hardpos" if variant == "project" else variant
        files = sorted(glob.glob(os.path.join(rundir, f"pinn_{problem}_{src}_*.npz")))
        if not files:
            continue
        sc, losses, secs, coefs, preds_list = [], [], [], [], []
        for f in files:
            p = np.load(f)["pred"]
            if variant == "project":
                m = p.reshape(len(p), -1).sum(axis=1)
                p = p * (mass0 / m).reshape(-1, *([1] * (p.ndim - 1)))
            sc.append(scores(p, traj, ntrain, conservative))
            with open(f.replace(".npz", ".json")) as jf:
                info = json.load(jf)
            losses.append(info["loss"]); secs.append(info["seconds"]); coefs.append(info["coef"])
            preds_list.append(p)
        pick = int(np.argmin(losses))          # selection on the training objective
        a = dict(sc[pick])
        a["train_loss"] = float(losses[pick])
        a["seconds"] = float(np.median(secs))
        a["seeds"] = len(files)
        for k in ("relL2", "relL2_extrap", "min"):
            vals = np.array([s[k] for s in sc], float)
            a[k + "_min"], a[k + "_max"] = float(vals.min()), float(vals.max())
        nout, ncoef = (1, 2) if problem == "A" else (2, 3)
        a["params"] = (2 * 64 + 64) + 3 * (64 * 64 + 64) + (64 * nout + nout) + ncoef
        keys = ["d", "r"] if problem == "A" else ["d1", "d2", "k"]
        a["coef"] = dict(zip(keys, coefs[pick]))
        out[f"PINN {variant}"] = a
        preds[variant] = preds_list[pick]
    return d, out, r_nm, r_nf, preds


def table(title, res, conservative):
    print(f"\n=== {title} ===")
    hdr = f"{'method':<16}{'relL2':>11}{'extrap':>11}{'min':>12}"
    if conservative:
        hdr += f"{'mass drift':>12}"
    hdr += f"{'train s':>10}{'seeds':>7}"
    print(hdr)
    for k, v in res.items():
        line = f"{k:<16}{v['relL2']:>11.3e}{v['relL2_extrap']:>11.3e}{v['min']:>12.3e}"
        if conservative:
            line += f"{v['mass_drift']:>12.3e}"
        line += f"{v['seconds']:>10.1f}{v.get('seeds', 1):>7}"
        print(line)
    for k, v in res.items():
        if v.get("coef"):
            print(f"  {k:<14} recovered " + "  ".join(f"{a}={b:.4f}" for a, b in v["coef"].items()))


if __name__ == "__main__":
    rundir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pinn_runs")
    allres = {}
    for problem in ("A", "B"):
        d, res, r_nm, r_nf, preds = collect(problem, rundir)
        allres[problem] = res
        table("Problem A: Fisher-KPP, true d=0.05 r=6" if problem == "A" else
              "Problem B: two-species conservative exchange, true d1=0.03 d2=0.08 k=6",
              res, problem == "B")
        np.savez(os.path.join(HERE, f"res{problem}_gpu.npz"),
                 x=d["x"], traj=d["traj"], dt=d["cfg"]["dt"], ntrain=d["cfg"]["ntrain"],
                 r_nm=r_nm, r_nf=r_nf, **{f"pinn_{k}": v for k, v in preds.items()})
    with open(os.path.join(HERE, "results_1d.json"), "w") as f:
        json.dump(allres, f, indent=2, default=float)
    print("\nwrote results_1d.json")
