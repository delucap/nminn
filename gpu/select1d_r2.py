"""Model selection for the 1D learned steppers, second revision: fits run to convergence.

Same data (data1d.load), same initializations (select1d.inits, seed 2026), same rollout horizons, same
iteration cap and same selection on the training loss as select1d.py. The only change is the stopping
rule of L-BFGS-B. With the default tolerances the quasi-Newton iteration stops when the objective
decreases by less than about 2e-9 per iteration in absolute terms, because the training loss is much
smaller than one, and on Problem A it stopped inside the shallow valley along which the logistic rate
and the quadratic sink of the closure trade off. Here both tolerances are set to 1e-30, so a fit stops
only at the iteration cap or when the line search can no longer decrease the objective.

Each problem is fitted on the noisy snapshots of every noise realization and on the clean snapshots,
and the scheme with the true coefficients is scored in the same way.

    python select1d_r2.py --jobs 7 --baseline          all realizations, both steppers on the paper's
    python select1d_r2.py --jobs 1 --baseline --paper_only --problems A --out select_1d_r2_timing.json
                                                      sequential sweep of the paper realization of Problem A,
                                                      whose fit times Table 2 reports
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import select1d as S                                    # noqa: E402
from data1d import load                                 # noqa: E402
from experiment import rollout, rollout_B               # noqa: E402
from nminn import NFVConserv, NFVScalar, NMINNConserv, NMINNScalar   # noqa: E402

TOL = 1e-30
ITERS = 400
PAPER_SEED = {"A": 7, "B": 11}

relL2 = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
softplus = lambda x: float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))
spinv = lambda y: float(np.log(np.expm1(y)))


def model(problem, name, M, h, dt):
    if problem == "A":
        return NMINNScalar(M, h, dt) if name == "NMINN" else NFVScalar(M, h, dt, H=8)
    return NMINNConserv(M, h, dt) if name == "NMINN" else NFVConserv(M, h, dt, H=10)


def data(problem, seed):
    d = load(problem, None if seed == "clean" else seed)
    snaps = d["traj"][:d["cfg"]["ntrain"] + 1] if seed == "clean" else d["train"]
    return d, snaps


def coef(problem, th):
    if problem == "A":
        return dict(d=float(np.exp(th[0])), r=softplus(th[1]), b=softplus(th[2]))
    return dict(d1=float(np.exp(th[0])), d2=float(np.exp(th[1])), k=softplus(th[2]))


def one_fit(task):
    problem, name, seed, K, th0 = task
    d, snaps = data(problem, seed)
    c = d["cfg"]
    m = model(problem, name, c["M"], d["h"], c["dt"])
    loss = S.make_loss(m.step, snaps, K)
    t0 = time.time()
    res = minimize(loss, np.asarray(th0, float), method="L-BFGS-B",
                   options=dict(maxiter=ITERS, maxfun=ITERS * 60, eps=1e-5, ftol=TOL, gtol=TOL))
    return dict(problem=problem, name=name, seed=seed, K=K, theta=res.x.tolist(), loss=float(res.fun),
                nit=int(res.nit), message=str(res.message), seconds=time.time() - t0)


def score(problem, name, th, save=None):
    d = load(problem)
    c = d["cfg"]
    traj, ntrain = d["traj"], c["ntrain"]
    n = ntrain + c["ntest"]
    m = model(problem, name, c["M"], d["h"], c["dt"])
    if problem == "A":
        r, _, _ = rollout(m.step, np.asarray(th), traj[0], n)
        drift = None
    else:
        r, _, drift, _ = rollout_B(m.step, np.asarray(th), traj[0], n)
    if save:
        np.save(save, r)
    nn = min(len(r), len(traj))
    out = dict(relL2=relL2(r[:nn], traj[:nn]), relL2_extrap=relL2(r[ntrain:nn], traj[ntrain:nn]),
               min=float(r[:nn].min()))
    if drift is not None:
        out["mass_drift"] = float(drift)
    return out


def tasks_for(problem, name, seed):
    S.RESTARTS = 8 if (name == "NMINN" and seed == PAPER_SEED[problem]) else 5
    rng = np.random.default_rng(2026)
    out = []
    for K in S.KS:
        for th0 in S.inits(problem, name, rng):
            out.append((problem, name, seed, K, th0.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--seeds", type=int, nargs="+", default=[21, 22, 23, 24])
    ap.add_argument("--baseline", action="store_true", help="also refit the unstructured stepper (paper seed)")
    ap.add_argument("--paper_only", action="store_true", help="only the realization of the paper")
    ap.add_argument("--problems", nargs="+", default=["A", "B"])
    ap.add_argument("--out", default=os.path.join(HERE, "select_1d_r2.json"))
    a = ap.parse_args()
    runs = []
    for p in a.problems:
        for s in [PAPER_SEED[p]] + ([] if a.paper_only else list(a.seeds) + ["clean"]):
            runs.append((p, "NMINN", s))
        if a.baseline:
            runs.append((p, "neural stepper", PAPER_SEED[p]))
    tasks = [t for r in runs for t in tasks_for(*r)]
    print(f"{len(tasks)} fits on {a.jobs} processes", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        fits = list(ex.map(one_fit, tasks, chunksize=1))
    print(f"fits done in {time.time() - t0:.0f} s", flush=True)
    out = dict(protocol=dict(optimizer="L-BFGS-B", ftol=TOL, gtol=TOL, maxiter=ITERS, maxfun=ITERS * 60,
                             fd_eps=1e-5, horizons=list(S.KS), init_seed=2026,
                             selection="lowest training loss over horizons and restarts",
                             machine=platform.node(), processes=a.jobs))
    for p, name, s in runs:
        mine = [f for f in fits if f["problem"] == p and f["name"] == name and f["seed"] == s]
        best = min(mine, key=lambda f: f["loss"])
        save = None
        if s == PAPER_SEED[p]:
            tag = os.path.splitext(os.path.basename(a.out))[0].replace("select_1d", "")
            save = os.path.join(HERE, f"sel_{p}_{name.split()[0]}{tag}.npy")
        e = dict(train_loss=best["loss"], K=best["K"], nit=best["nit"], message=best["message"],
                 seconds=sum(f["seconds"] for f in mine), trials=len(mine), theta=best["theta"],
                 trial_losses=[f["loss"] for f in mine], trial_nit=[f["nit"] for f in mine],
                 **score(p, name, best["theta"], save))
        if name == "NMINN":
            e["coef"] = coef(p, best["theta"])
            e["trial_coefs"] = [coef(p, f["theta"]) for f in mine]
        out.setdefault(p, {}).setdefault(str(s), {})[name] = e
        line = f"{p} seed {s!s:<6} {name:<15} K*={best['K']:<3} nit={best['nit']:<4} loss={best['loss']:.6e} " \
               f"relL2={e['relL2']:.4e} extrap={e['relL2_extrap']:.4e} min={e['min']:.2e}"
        if "mass_drift" in e:
            line += f" drift={e['mass_drift']:.1e}"
        if name == "NMINN":
            line += "  " + " ".join(f"{k}={v:.6g}" for k, v in e["coef"].items())
        print(line, flush=True)
    for p in a.problems:
        d, clean = data(p, "clean")
        th = [np.log(0.05), spinv(6.0), -60.0] if p == "A" else [np.log(0.03), np.log(0.08), spinv(6.0)]
        m = model(p, "NMINN", d["cfg"]["M"], d["h"], d["cfg"]["dt"])
        e = dict(coef=coef(p, th), **score(p, "NMINN", th),
                 train_loss_clean={str(K): float(S.make_loss(m.step, clean, K)(np.asarray(th))) for K in S.KS})
        out[p]["true_coefficients"] = e
        print(f"{p} true coefficients relL2={e['relL2']:.4e} extrap={e['relL2_extrap']:.4e} "
              f"clean losses {e['train_loss_clean']}", flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.basename(a.out)}")


if __name__ == "__main__":
    main()
