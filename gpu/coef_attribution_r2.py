"""Problem C: which identified coefficient costs the accuracy.

The classical splitting scheme is run with the true coefficients, with the coefficients identified by the NMINN
(constant sensitivity equal to the mean learned one, as in Table 6), with each identified coefficient alone in
place of the true one, and with the identified coefficients except the true cell diffusivity, at the
surrogate's internal step and at the reference step, and scored against the reference exactly as in
classical_baseline.py.

    python coef_attribution_r2.py --device cuda --tag _r1        -> coef_attribution_r2.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from ks2d import KS2D, KSParams
from run2d import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
TRUE = dict(Du=1.0e-4, Dv=1.0e-3, chi=2.0e-2, alpha=1.0, beta=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--Dt", type=float, default=2e-2)
    ap.add_argument("--n_macro", type=int, default=80)
    ap.add_argument("--ntrain", type=int, default=25)
    ap.add_argument("--dt_ref", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="_r1")
    a = ap.parse_args()
    dtype = torch.float64
    M = a.M
    solver = KS2D(M, M, params=KSParams(**TRUE), device=a.device, dtype=dtype)
    u0, v0 = solver.initial(kind="bumps")
    hx, hy = solver.hx, solver.hy
    sub = int(round(a.Dt / a.dt_ref))
    ref_u, ref_v = solver.run(u0, v0, a.dt_ref, a.n_macro * sub, save_every=sub)

    c = json.load(open(os.path.join(HERE, f"results_2d_M{M}{a.tag}.json")))["NMINN"]["coef"]
    ident = dict(Du=c["Du"], Dv=c["Dv"], chi=c["mean_sensitivity"], alpha=c["alpha"], beta=c["beta"])
    cases = {"true": dict(TRUE), "identified": dict(ident)}
    for k in ident:
        cases[f"true except {k}"] = dict(TRUE, **{k: ident[k]})
    cases["identified except Du"] = dict(ident, Du=TRUE["Du"])

    out = dict(identified=ident, true=TRUE, tag=a.tag, cases={})
    for name, p in cases.items():
        s = KS2D(M, M, params=KSParams(**p), device=a.device, dtype=dtype)
        row = {}
        for dt in (a.Dt / 4, a.dt_ref):
            k = int(round(a.Dt / dt))
            us, vs = s.run(u0, v0, dt, a.n_macro * k, save_every=k)
            m = metrics(us, vs, ref_u, ref_v, a.ntrain, hx, hy)
            row[str(dt)] = dict(relL2_u=m["relL2_u"], relL2_u_extrap=m["relL2_u_extrap"], peak_ratio=m.get("peak_ratio"))
        out["cases"][name] = dict(params=p, **row)
        print(f"{name:<24} internal step {row[str(a.Dt / 4)]['relL2_u']:.3e}   reference step "
              f"{row[str(a.dt_ref)]['relL2_u']:.3e}", flush=True)
    with open(os.path.join(HERE, "coef_attribution_r2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote coef_attribution_r2.json")


if __name__ == "__main__":
    main()
