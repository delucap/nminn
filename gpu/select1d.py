"""Model selection for the 1D learned steppers, on training data only.

The first protocol used a single L-BFGS fit from a fixed initialization at a fixed rollout horizon. On
Problem B that left the exchange rate at its initial value: the objective does prefer the true rate,
but by a margin of a few percent of a noise-dominated value, which a finite-difference quasi-Newton step
cannot resolve from a poor starting point. The fix is a protocol, not a tweak, and it is applied
identically to both steppers: sweep the horizon, restart from several initializations, and select the
run with the lowest TRAINING loss. No test quantity enters the selection.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from data1d import load           # noqa: E402
from experiment import rollout, rollout_B   # noqa: E402
from nminn import NFVConserv, NFVScalar, NMINNConserv, NMINNScalar   # noqa: E402

relL2 = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
KS = (5, 10, 20, 30)
RESTARTS = int(os.environ.get("RESTARTS", "5"))


def make_loss(step_fn, data, K):
    starts = list(range(0, len(data) - K))

    def loss(theta):
        tot = 0.0
        for s in starts:
            U = data[s].copy()
            e = 0.0
            for j in range(1, K + 1):
                U = step_fn(U, theta)
                if not np.all(np.isfinite(U)):
                    return 1e6
                e += np.mean((U - data[s + j]) ** 2)
            tot += e / K
        return tot / len(starts)
    return loss


def fit(step_fn, theta0, data, K, eps=1e-5, iters=400):
    loss = make_loss(step_fn, data, K)
    res = minimize(loss, theta0, method="L-BFGS-B",
                   options={"maxiter": iters, "maxfun": iters * 60, "eps": eps})
    return res.x, float(res.fun)


def inits(problem, name, rng):
    """Restart initializations. Physical parameters are drawn on a log scale over a wide range, so
    the sweep is not centred on the answer."""
    out = []
    for i in range(RESTARTS):
        if problem == "A":
            if name == "NMINN":
                out.append(np.array([np.log(10 ** rng.uniform(-2.3, -0.7)),
                                     rng.uniform(-1.0, 12.0), rng.uniform(-3.0, 2.0)]))
            else:
                out.append(0.1 * rng.standard_normal(NFVScalar(64, 1 / 64, 0.004, H=8).nparam))
        else:
            if name == "NMINN":
                out.append(np.array([np.log(10 ** rng.uniform(-2.3, -0.7)),
                                     np.log(10 ** rng.uniform(-2.3, -0.7)),
                                     rng.uniform(-2.0, 12.0)]))
            else:
                out.append(0.1 * rng.standard_normal(NFVConserv(64, 1 / 64, 0.004, H=10).nparam))
    return out


def run(problem, names=("NMINN", "neural stepper"), noise_seed=None, suffix=""):
    d = load(problem, noise_seed)
    c = d["cfg"]
    M, h, dt = c["M"], d["h"], c["dt"]
    n = c["ntrain"] + c["ntest"]
    traj, train, ntrain = d["traj"], d["train"], c["ntrain"]
    softplus = lambda x: float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))
    out = {}
    for name in names:
        rng = np.random.default_rng(2026)
        best = None
        t0 = time.time()
        trials = []
        for K in KS:
            for th0 in inits(problem, name, rng):
                if problem == "A":
                    m = NMINNScalar(M, h, dt) if name == "NMINN" else NFVScalar(M, h, dt, H=8)
                else:
                    m = NMINNConserv(M, h, dt) if name == "NMINN" else NFVConserv(M, h, dt, H=10)
                th, l = fit(m.step, th0, train, K)
                trials.append(dict(K=K, train_loss=l))
                if best is None or l < best[1]:
                    best = (th, l, K, m)
        sec = time.time() - t0
        th, l, K, m = best
        if problem == "A":
            r, _, _ = rollout(m.step, th, traj[0], n)
            drift = None
            coef = None
            if name == "NMINN":
                logd, a, b = m.unpack(th)
                coef = dict(d=float(np.exp(logd)), r=float(a), b=float(b))
        else:
            r, _, drift, _ = rollout_B(m.step, th, traj[0], n)
            coef = dict(d1=float(np.exp(th[0])), d2=float(np.exp(th[1])), k=softplus(th[2])) \
                if name == "NMINN" else None
        nn = min(len(r), len(traj))
        out[name] = dict(relL2=relL2(r[:nn], traj[:nn]),
                         relL2_extrap=relL2(r[ntrain:nn], traj[ntrain:nn]),
                         min=float(r[:nn].min()), train_loss=l, K=K, seconds=sec,
                         params=3 if name == "NMINN" else int(m.nparam),
                         coef=coef, trials=len(trials))
        if drift is not None:
            out[name]["mass_drift"] = float(drift)
        np.save(os.path.join(HERE, f"sel_{problem}_{name.split()[0]}{suffix}.npy"), r)
        print(f"  {name:<16} K*={K:<3} train loss={l:.4e}  relL2={out[name]['relL2']:.3e} "
              f"extrap={out[name]['relL2_extrap']:.3e} min={out[name]['min']:.2e}"
              + (f" drift={drift:.2e}" if drift is not None else "")
              + f"  ({sec:.0f}s)")
        if coef:
            print(f"    recovered {coef}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["NMINN", "neural stepper"])
    ap.add_argument("--noise_seed", type=int, default=None,
                    help="noise realization; default = the seeds of the paper (7 for A, 11 for B)")
    ap.add_argument("--problems", nargs="+", default=["A", "B"])
    a = ap.parse_args()
    suffix = "" if a.noise_seed is None else f"_noise{a.noise_seed}"
    path = os.path.join(HERE, f"select_1d{suffix}.json")
    allout = json.load(open(path)) if os.path.exists(path) else {}
    for p in a.problems:
        print(f"=== problem {p}: {len(KS)*RESTARTS} fits per stepper, selected on training loss"
              + (f", noise seed {a.noise_seed}" if a.noise_seed is not None else "") + " ===")
        got = run(p, tuple(a.models), a.noise_seed, suffix)
        allout.setdefault(p, {}).update(got)
    with open(path, "w") as f:
        json.dump(allout, f, indent=2, default=float)
    print(f"wrote {os.path.basename(path)}")
