"""Matched classical baseline for Problem C (Reviewer 1 and Reviewer 2, JOCS first round).

The structure-preserving splitting solver with the TRUE coefficients is run at a sweep of time steps, from
the reference step 5e-4 up to the surrogate's macro step 2e-2, on an idle device, and scored exactly as the
surrogates are: relative L2 error of u over the 81 macro snapshots against the reference at 5e-4, error on
the extrapolation window, minimum of u, mass drift, and wall-clock time of the whole horizon (best of two
timed runs after a warm-up). A run at 2.5e-4 estimates the error of the reference itself.

The same solver is also run with the coefficients identified by the NMINN (constant sensitivity equal to
the mean learned sensitivity), at the fine step and at the surrogate's internal step, which separates the
part of the surrogate's error due to coefficient identification from the part due to the discretization.

The trained surrogates are timed in the same run, so that the cost comparison is like for like, and the
uniformization rate theta is recorded along the reference trajectory: 1/theta_max is the largest step at
which an explicit upwind step of the same flux would be positive.

    python classical_baseline.py --device cuda --tag _r1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ks2d import KS2D, KSParams
from models2d import MODEL_NAMES, build_model, count_params
from ops import poisson_terms
from run2d import metrics, rollout
from transport import UpwindTaxis2D

HERE = os.path.dirname(os.path.abspath(__file__))
TRUE = KSParams(Du=1.0e-4, Dv=1.0e-3, chi=2.0e-2, alpha=1.0, beta=1.0)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, repeat=2):
    best = float("inf")
    out = None
    for _ in range(repeat):
        sync()
        t0 = time.time()
        out = fn()
        sync()
        best = min(best, time.time() - t0)
    return out, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--Dt", type=float, default=2e-2)
    ap.add_argument("--n_macro", type=int, default=80)
    ap.add_argument("--ntrain", type=int, default=25)
    ap.add_argument("--dts", type=float, nargs="+", default=[2e-2, 1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4])
    ap.add_argument("--dt_ref", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ic", default="bumps")
    a = ap.parse_args()
    dtype = torch.float64
    M = a.M
    solver = KS2D(M, M, params=TRUE, device=a.device, dtype=dtype)
    u0, v0 = solver.initial(kind=a.ic)
    hx, hy = solver.hx, solver.hy

    # warm-up (kernels, allocator)
    solver.run(u0, v0, a.Dt, 5, save_every=5)

    trajs, rows = {}, {}
    for dt in a.dts:
        sub = int(round(a.Dt / dt))
        assert abs(sub * dt - a.Dt) < 1e-12, (dt, sub)
        (us, vs), sec = timed(lambda: solver.run(u0, v0, dt, a.n_macro * sub, save_every=sub))
        trajs[dt] = (us, vs)
        rows[dt] = dict(dt=dt, nsteps=a.n_macro * sub, seconds=sec)
        print(f"dt={dt:.1e} steps={a.n_macro*sub:5d}  {sec:6.2f} s", flush=True)

    ref_u, ref_v = trajs[a.dt_ref]
    for dt, (us, vs) in trajs.items():
        m = metrics(us, vs, ref_u, ref_v, a.ntrain, hx, hy)
        m.pop('steps', None)
        rows[dt].update(m)
        rows[dt]["speedup_vs_reference"] = rows[a.dt_ref]["seconds"] / rows[dt]["seconds"]
    # error of the reference itself, against the finest run available
    finest = min(a.dts)
    self_err = None
    if finest < a.dt_ref:
        self_err = metrics(ref_u, ref_v, trajs[finest][0], trajs[finest][1], a.ntrain, hx, hy)
        self_err["finest_dt"] = finest

    # uniformization rate along the reference trajectory
    tx = UpwindTaxis2D(hx, hy)
    thetas = []
    for i in range(len(ref_u)):
        ax, ay = tx.velocities(ref_v[i], TRUE.chi, TRUE.chi)
        thetas.append(float(tx.theta(ax, ay)))
    th_max, th_train = max(thetas), max(thetas[:a.ntrain + 1])
    theta_info = dict(theta_max=th_max, theta_max_training_window=th_train, cfl_dt=1.0 / th_max,
                      explicit_diffusion_dt_u=hx * hx / (4 * TRUE.Du), explicit_diffusion_dt_v=hx * hx / (4 * TRUE.Dv),
                      internal_step=a.Dt / 4, mu_internal=th_max * a.Dt / 4,
                      K_internal_tol1e14=poisson_terms(min(th_max * a.Dt / 4, 6.0), 1e-14),
                      nsub_internal=max(1, int(-(-th_max * a.Dt / 4 // 6.0))),
                      thetas=thetas)
    print(f"theta_max={th_max:.1f} (training window {th_train:.1f}), explicit upwind positive only for "
          f"dt <= {1/th_max:.2e}; surrogate internal step {a.Dt/4:.1e}", flush=True)

    # the classical scheme with the coefficients identified by the NMINN
    identified = {}
    rpath = os.path.join(HERE, f"results_2d_M{M}{a.tag}.json")
    if os.path.exists(rpath):
        res = json.load(open(rpath))
        c = res["NMINN"]["coef"]
        pid = KSParams(Du=c["Du"], Dv=c["Dv"], chi=c["mean_sensitivity"], alpha=c["alpha"], beta=c["beta"])
        sid = KS2D(M, M, params=pid, device=a.device, dtype=dtype)
        for dt in (a.Dt / 4, a.dt_ref):
            sub = int(round(a.Dt / dt))
            (us, vs), sec = timed(lambda: sid.run(u0, v0, dt, a.n_macro * sub, save_every=sub))
            m = metrics(us, vs, ref_u, ref_v, a.ntrain, hx, hy)
            m["seconds"] = sec
            identified[str(dt)] = m
            print(f"identified coefficients, dt={dt:.1e}: relL2 {m['relL2_u']:.3e} extrap {m['relL2_u_extrap']:.3e}",
                  flush=True)
        identified["coefficients"] = pid.__dict__
        setup = res["_setup"]
    else:
        setup = None

    # trained surrogates, timed in the same run
    surrogates = {}
    if setup is not None:
        for short, name in MODEL_NAMES.items():
            wpath = os.path.join(HERE, f"model_{short}_M{M}{a.tag}.pt")
            if not os.path.exists(wpath):
                continue
            model = build_model(short, M, hx, a.Dt, setup["n_sub"], setup["u_scale"], setup["v_scale"],
                                setup.get("mass0", float(u0.sum())), a.device, dtype)
            model.load_state_dict(torch.load(wpath, map_location=a.device))
            rollout(model, u0, v0, 3)
            (pu, pv), sec = timed(lambda: rollout(model, u0, v0, a.n_macro), repeat=3)
            m = metrics(pu, pv, ref_u, ref_v, a.ntrain, hx, hy)
            m["seconds"] = sec
            m["params"] = count_params(model)
            m["speedup_vs_reference"] = rows[a.dt_ref]["seconds"] / sec
            surrogates[name] = m
            print(f"{name:<24} relL2 {m['relL2_u']:.3e}  {sec:.2f} s", flush=True)

    out = dict(setup=dict(M=M, Dt=a.Dt, n_macro=a.n_macro, ntrain=a.ntrain, dt_ref=a.dt_ref, ic=a.ic,
                          torch=torch.__version__,
                          device_name=torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu"),
               classical={str(k): v for k, v in rows.items()}, reference_self_error=self_err,
               theta=theta_info, identified=identified, surrogates=surrogates)
    with open(os.path.join(HERE, f"classical_baseline{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("\n=== classical splitting scheme, true coefficients ===")
    print(f"{'dt':>8}{'steps':>7}{'relL2 u':>11}{'extrap':>11}{'min u':>11}{'mass drift':>12}{'time s':>9}{'speedup':>9}")
    for dt in sorted(rows, reverse=True):
        r = rows[dt]
        print(f"{dt:>8.1e}{r['nsteps']:>7d}{r['relL2_u']:>11.3e}{r['relL2_u_extrap']:>11.3e}{r['min_u']:>11.2e}"
              f"{r['mass_drift']:>12.1e}{r['seconds']:>9.2f}{r['speedup_vs_reference']:>9.1f}")
    if self_err:
        print(f"reference (dt={a.dt_ref:.1e}) against dt={finest:.1e}: relL2 {self_err['relL2_u']:.3e}")
    print(f"wrote classical_baseline{a.tag}.json")


if __name__ == "__main__":
    main()
