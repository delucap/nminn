"""Cost measurements for the 2D study, run on an otherwise idle device.

Revision note. The submitted version timed freshly constructed, untrained surrogates. For the
method-informed stepper that understates the cost, because the number of uniformization terms grows
with the learned sensitivity, so with --tag the trained weights of that run (closure trained at 128^2,
transferred to the other grids as in the resolution study) are loaded and the classical splitting
scheme with the true coefficients at the surrogate's internal step is timed alongside.

Reports, for each resolution, the wall-clock time of the reference solver over the horizon and of one
rollout of each surrogate, together with the number of macro steps per second. Training times are
measured separately by run2d.py and are not comparable across methods run concurrently.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ks2d import KS2D, KSParams
from models2d import MODEL_NAMES, build_model, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
TRUE = KSParams(Du=1.0e-4, Dv=1.0e-3, chi=2.0e-2, alpha=1.0, beta=1.0)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, repeat=3):
    fn()          # warm up
    sync()
    best = float("inf")
    for _ in range(repeat):
        sync()
        t0 = time.time()
        fn()
        sync()
        best = min(best, time.time() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--dt_ref", type=float, default=5e-4)
    ap.add_argument("--n_macro", type=int, default=80)
    ap.add_argument("--sub", type=int, default=40)
    ap.add_argument("--n_sub", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="", help="load the trained weights of this run (empty: untrained)")
    a = ap.parse_args()
    Dt = a.dt_ref * a.sub
    out = {}
    setup = None
    if a.tag:
        setup = json.load(open(os.path.join(HERE, f"results_2d_M128{a.tag}.json")))["_setup"]
    for M in a.grids:
        s = KS2D(M, M, params=TRUE, device=a.device, dtype=torch.float64)
        u0, v0 = s.initial()
        ref = bench(lambda: s.run(u0, v0, a.dt_ref, a.n_macro * a.sub,
                                  save_every=a.n_macro * a.sub), repeat=1)
        row = dict(reference=ref, reference_steps=a.n_macro * a.sub)
        # the classical scheme at the surrogate's internal step, true coefficients
        row["classical_internal_step"] = dict(
            seconds=bench(lambda: s.run(u0, v0, Dt / a.n_sub, a.n_macro * a.n_sub,
                                        save_every=a.n_macro * a.n_sub), repeat=2),
            steps=a.n_macro * a.n_sub)
        row["classical_internal_step"]["speedup"] = ref / row["classical_internal_step"]["seconds"]
        u_scale = setup["u_scale"] if setup else float(u0.max())
        v_scale = setup["v_scale"] if setup else 1.0
        models = {short: build_model(short, M, s.hx, Dt, a.n_sub, u_scale, v_scale, float(u0.sum()),
                                     a.device, torch.float64)
                  for short in ("NMINN", "CNN", "FNO", "CNNFLUX", "CNNPOS", "CNNPROJ")}
        if setup:
            for short, m in list(models.items()):
                wpath = os.path.join(HERE, f"model_{short}_M128{a.tag}.pt")
                if os.path.exists(wpath):
                    m.load_state_dict(torch.load(wpath, map_location=a.device))
                else:
                    del models[short]
        row["weights"] = "trained at 128^2" if setup else "untrained"
        for name, m in models.items():
            def run():
                u, v = u0.clone(), v0.clone()
                with torch.no_grad():
                    for _ in range(a.n_macro):
                        u, v = m(u, v)
            t = bench(run)
            row[name] = dict(seconds=t, macro_steps_per_second=a.n_macro / t,
                             params=count_params(m), speedup=ref / t)
        out[M] = row
        print(f"M={M:4d}  reference {ref:7.2f}s   " +
              "   ".join(f"{k} {row[k]['seconds']:6.2f}s ({row[k]['speedup']:5.1f}x)"
                         for k in models))
    with open(os.path.join(HERE, f"timings_2d{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote timings_2d{a.tag}.json")


if __name__ == "__main__":
    main()
