"""The guarantees are claimed for every parameter value, so they are tested that way.

Untrained networks, deliberately badly scaled weights, and long rollouts: the method-informed stepper
must stay nonnegative and mass exact in every case, and the unstructured steppers are recorded for
contrast. Run: python test_models2d.py [cuda|cpu]
"""
import sys

import torch

from ks2d import KS2D, KSParams
from models2d import CNNStepper, FNOStepper, NMINN2D

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda"
DT = torch.float64
M, Dt = 96, 0.02
solver = KS2D(M, M, params=KSParams(chi=2e-2), device=DEV, dtype=DT)
u0, v0 = solver.initial()
h = solver.hx
fails = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


def roll(model, n=40):
    u, v = u0.clone(), v0.clone()
    m0 = float(u.sum())
    mn, drift = float(u.min()), 0.0
    with torch.no_grad():
        for _ in range(n):
            u, v = model(u, v)
            if not (torch.isfinite(u).all() and torch.isfinite(v).all()):
                return float("-inf"), float("inf"), float(v.min()), False
            mn = min(mn, float(u.min()))
            drift = max(drift, abs(float(u.sum()) - m0) / m0)
    return mn, drift, float(v.min()), True


for scale, tag in ((1.0, "at initialization"), (20.0, "with weights scaled by 20"),
                   (-50.0, "with weights scaled by -50")):
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        nm = NMINN2D(M, h, Dt, n_sub=4, u_scale=float(u0.max()), v_scale=1.0, device=DEV, dtype=DT)
        with torch.no_grad():
            for p in nm.sens.parameters():
                p.mul_(scale)
            nm.log_chi.mul_(1.0).add_(torch.log(torch.tensor(abs(scale), device=DEV, dtype=DT)))
        mn, drift, vmn, fin = roll(nm)
        check(f"NMINN2D nonnegative {tag} (seed {seed})", fin and mn >= 0.0, f"min u={mn:.2e}")
        check(f"NMINN2D mass exact {tag} (seed {seed})", fin and drift < 1e-12, f"drift={drift:.2e}")
        check(f"NMINN2D signal nonnegative {tag} (seed {seed})", fin and vmn >= 0.0, f"min v={vmn:.2e}")

# contrast: the unstructured steppers carry no such guarantee
for name, ctor in (("CNN stepper", CNNStepper), ("FNO stepper", FNOStepper)):
    torch.manual_seed(0)
    m = ctor(Dt, u_scale=float(u0.max()), v_scale=1.0, device=DEV, dtype=DT)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(20.0)
    mn, drift, vmn, fin = roll(m)
    print(f"[--- ] {name} at the same scaling: min u={mn:.2e} mass drift={drift:.2e} finite={fin}")

print("\n", "ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
