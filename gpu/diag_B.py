"""Is the exchange rate of Problem B unidentified by the data, or missed by the optimizer?

Evaluates the training objective along the exchange-rate direction with the two diffusivities held at
their fitted values, and compares the value at the fitted rate with the value at the true rate.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from data1d import load          # noqa: E402
from experiment import rollout_B  # noqa: E402
from nminn import NMINNConserv    # noqa: E402

d = load("B")
c = d["cfg"]
M, h, dt = c["M"], d["h"], c["dt"]
traj, train, ntrain = d["traj"], d["train"], c["ntrain"]
n = c["ntrain"] + c["ntest"]
m = NMINNConserv(M, h, dt)
inv_sp = lambda y: np.log(np.expm1(y))          # inverse softplus
relL2 = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))


def loss(theta, K, data):
    starts = range(0, len(data) - 1 - K + 1)
    tot = 0.0
    for s in starts:
        U = data[s].copy()
        e = 0.0
        for j in range(1, K + 1):
            U = m.step(U, theta)
            e += np.mean((U - data[s + j]) ** 2)
        tot += e / K
    return tot / len(list(starts))


base = [np.log(0.0298), np.log(0.0806)]
print(f"{'k':>8}{'loss K=5 (noisy)':>20}{'loss K=30 (noisy)':>20}{'loss K=30 (clean)':>20}{'rollout relL2':>16}")
for k in (0.695, 1.0, 2.0, 4.0, 6.0, 8.0):
    th = np.array(base + [inv_sp(k)])
    r, _, _, _ = rollout_B(m.step, th, traj[0], n)
    nn = min(len(r), len(traj))
    print(f"{k:>8.3f}{loss(th, 5, train):>20.6e}{loss(th, 30, train):>20.6e}"
          f"{loss(th, 30, traj[:ntrain + 1]):>20.6e}{relL2(r[:nn], traj[:nn]):>16.3e}")
