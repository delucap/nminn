"""Problem B, one L-BFGS-B fit of the NMINN from the fixed start of the first protocol, with SciPy's default
stopping tolerances and with the tolerances of select1d_r2.py, on the noisy snapshots of the paper.

The defaults are absolute, ftol about 2.2e-9 on the decrease of the objective and gtol 1e-5 on the projected
gradient, and the training loss of this problem is of order 3e-4, so the default rule stops the fit after a
few iterations with the exchange rate still at its initial value.

    python fixedstart_b_r2.py        -> fixedstart_B_r2.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import select1d as S                         # noqa: E402
from data1d import load                      # noqa: E402
from experiment import rollout_B             # noqa: E402
from nminn import NMINNConserv               # noqa: E402

softplus = lambda x: float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))


def main():
    d = load("B")
    c = d["cfg"]
    traj, n = d["traj"], c["ntrain"] + c["ntest"]
    m = NMINNConserv(c["M"], d["h"], c["dt"])
    th0 = np.array([np.log(0.02), np.log(0.05), 0.0])
    out = dict(start=dict(d1=0.02, d2=0.05, k=softplus(0.0)))
    for K in (5, 30):
        loss = S.make_loss(m.step, d["train"], K)
        for label, extra in (("default", {}), ("converged", dict(ftol=1e-30, gtol=1e-30))):
            r = minimize(loss, th0, method="L-BFGS-B", options=dict(maxiter=400, maxfun=24000, eps=1e-5, **extra))
            R, _, drift, _ = rollout_B(m.step, r.x, traj[0], n)
            nn = min(len(R), len(traj))
            e = dict(nit=int(r.nit), message=str(r.message), loss=float(r.fun),
                     d1=float(np.exp(r.x[0])), d2=float(np.exp(r.x[1])), k=softplus(r.x[2]),
                     relL2=float(np.linalg.norm(R[:nn] - traj[:nn]) / np.linalg.norm(traj[:nn])))
            out[f"K{K}_{label}"] = e
            print(f"K={K:<3} {label:<9} nit={e['nit']:<4} loss={e['loss']:.6e} k={e['k']:.4f} relL2={e['relL2']:.3e} "
                  f"{e['message']}")
    with open(os.path.join(HERE, "fixedstart_B_r2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote fixedstart_B_r2.json")


if __name__ == "__main__":
    main()
