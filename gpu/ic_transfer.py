"""Transfer of the trained surrogates to initial data they never saw (Reviewer 1, JOCS first round).

Every trained model of a training run (identified by its tag) is rolled out on the training grid from new
initial data and scored against a reference computed for that datum with the same protocol as Problem C.
The classical splitting scheme with the true coefficients at the surrogate's internal step is scored
alongside as the discretization floor for that datum. The projected variant is given the mass of the new
datum, which is what a modeller who knows the invariant would do.

Kinds: bumps (the training datum), bumps_rand with seeds 1 to 3 (same regime, new geometry), strong (the
training bumps at 1.5 times the amplitude), subcritical (two small bumps below the aggregation threshold).

    python ic_transfer.py --device cuda --tag _r1
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from ks2d import KS2D, KSParams
from models2d import MODEL_NAMES, build_model
from run2d import metrics, rollout

HERE = os.path.dirname(os.path.abspath(__file__))
TRUE = KSParams(Du=1.0e-4, Dv=1.0e-3, chi=2.0e-2, alpha=1.0, beta=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--Dt", type=float, default=2e-2)
    ap.add_argument("--dt_ref", type=float, default=5e-4)
    ap.add_argument("--n_macro", type=int, default=80)
    ap.add_argument("--ntrain", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--kinds", nargs="+", default=["bumps", "bumps_rand:1", "bumps_rand:2", "bumps_rand:3",
                                                    "strong", "subcritical"])
    a = ap.parse_args()
    dtype = torch.float64
    M = a.M
    setup = json.load(open(os.path.join(HERE, f"results_2d_M{M}{a.tag}.json")))["_setup"]
    solver = KS2D(M, M, params=TRUE, device=a.device, dtype=dtype)
    hx, hy = solver.hx, solver.hy
    sub = int(round(a.Dt / a.dt_ref))
    models = {}
    for short, name in MODEL_NAMES.items():
        wpath = os.path.join(HERE, f"model_{short}_M{M}{a.tag}.pt")
        if os.path.exists(wpath):
            m = build_model(short, M, hx, a.Dt, setup["n_sub"], setup["u_scale"], setup["v_scale"],
                            setup.get("mass0", 1.0), a.device, dtype)
            m.load_state_dict(torch.load(wpath, map_location=a.device))
            models[name] = m
    out = {"_setup": dict(setup, kinds=a.kinds)}
    for spec in a.kinds:
        kind, _, seed = spec.partition(":")
        seed = int(seed) if seed else 0
        u0, v0 = solver.initial(kind=kind, seed=seed)
        us, vs = solver.run(u0, v0, a.dt_ref, a.n_macro * sub, save_every=sub)
        row = dict(kind=kind, seed=seed, mass=float(u0.sum() * hx * hy), u0_max=float(u0.max()),
                   peak_growth=float(us[-1].max() / us[0].max()), u_max_final=float(us[-1].max()))
        # classical scheme at the surrogate's internal step, true coefficients
        cu, cv = solver.run(u0, v0, a.Dt / 4, a.n_macro * 4, save_every=4)
        row["classical_internal_step"] = metrics(cu, cv, us, vs, a.ntrain, hx, hy)
        for name, m in models.items():
            if hasattr(m, "project") and m.project:
                m.mass0 = float(u0.sum())
            pu, pv = rollout(m, u0, v0, a.n_macro)
            row[name] = metrics(pu, pv, us, vs, a.ntrain, hx, hy)
        out[spec] = row
        print(f"{spec:<14} mass={row['mass']:.3f} peak growth={row['peak_growth']:7.1f}x  "
              f"classical@Dt/4 {row['classical_internal_step']['relL2_u']:.3e}  " +
              "  ".join(f"{[s for s, n in MODEL_NAMES.items() if n == k][0]} "
                        f"{row[k]['relL2_u']:.3e}/{row[k]['min_u']:+.1e}/{row[k]['mass_drift']:.0e}"
                        for k in models), flush=True)
    with open(os.path.join(HERE, f"ic_transfer{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote ic_transfer{a.tag}.json")


if __name__ == "__main__":
    main()
