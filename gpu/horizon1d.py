"""Sensitivity of the 1D learned steppers to the rollout horizon used in the training loss.

The horizon is a protocol choice, so it is swept for both steppers together and reported, rather than
tuned for one of them. Run: python horizon1d.py
"""
from __future__ import annotations

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
out = {}

for problem in ("A", "B"):
    d = load(problem)
    c = d["cfg"]
    M, h, dt = c["M"], d["h"], c["dt"]
    n = c["ntrain"] + c["ntest"]
    traj, train, ntrain = d["traj"], d["train"], c["ntrain"]
    print(f"\n=== problem {problem} ===")
    print(f"{'K':>4}{'stepper':>18}{'relL2':>12}{'extrap':>12}{'min':>12}{'drift':>12}{'s':>8}  recovered")
    out[problem] = {}
    for K in (5, 10, 20, 30):
        for name in ("NMINN", "neural stepper"):
            t0 = time.time()
            if problem == "A":
                if name == "NMINN":
                    m = NMINNScalar(M, h, dt)
                    th, _ = train_stepper(m.step, np.array([np.log(0.05), 1.8, -2.0]), train, K=K)
                    logd, a, b = m.unpack(th)
                    rec = f"d={np.exp(logd):.4f} r={a:.3f}"
                else:
                    m = NFVScalar(M, h, dt, H=8, seed=1)
                    th, _ = train_stepper(m.step, m.theta0, train, K=K)
                    rec = ""
                r, _, _ = rollout(m.step, th, traj[0], n)
                drift = float("nan")
            else:
                if name == "NMINN":
                    m = NMINNConserv(M, h, dt)
                    th, _ = train_stepper_B(m.step, np.array([np.log(0.02), np.log(0.05), 0.0]), train, K=K)
                    sp = float(np.log1p(np.exp(-abs(th[2]))) + max(th[2], 0.0))
                    rec = f"d1={np.exp(th[0]):.4f} d2={np.exp(th[1]):.4f} k={sp:.3f}"
                else:
                    m = NFVConserv(M, h, dt, H=10, seed=2)
                    th, _ = train_stepper_B(m.step, m.theta0, train, K=K)
                    rec = ""
                r, _, drift, _ = rollout_B(m.step, th, traj[0], n)
            sec = time.time() - t0
            nn = min(len(r), len(traj))
            e = relL2(r[:nn], traj[:nn])
            ex = relL2(r[ntrain:nn], traj[ntrain:nn])
            mn = float(r[:nn].min())
            print(f"{K:>4}{name:>18}{e:>12.3e}{ex:>12.3e}{mn:>12.3e}{drift:>12.3e}{sec:>8.1f}  {rec}")
            out[problem][f"K{K}_{name}"] = dict(relL2=e, extrap=ex, min=mn, drift=float(drift),
                                                seconds=sec, recovered=rec)

with open(os.path.join(HERE, "horizon_1d.json"), "w") as f:
    json.dump(out, f, indent=2, default=float)
print("\nwrote horizon_1d.json")
