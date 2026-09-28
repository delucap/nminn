"""Physics-informed baselines on the 2D Keller-Segel problem, scored with the same metrics."""
from __future__ import annotations

import argparse
import json
import os

import torch

from pinn2d import PINN2D
from run2d import make_data, metrics

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--dt_ref", type=float, default=5e-4)
    ap.add_argument("--n_macro", type=int, default=80)
    ap.add_argument("--sub", type=int, default=40)
    ap.add_argument("--ntrain", type=int, default=25)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--variant", default="vanilla", choices=["vanilla", "penalty", "hardpos"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adam", type=int, default=15000)
    ap.add_argument("--n_coll", type=int, default=20000)
    ap.add_argument("--batch_data", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    solver, us, vs, ref_sec = make_data(a.M, a.dt_ref, a.n_macro, a.sub, a.device, torch.float64)
    Dt = a.dt_ref * a.sub
    T = Dt * a.n_macro
    u_scale = float(us[:a.ntrain + 1].max())
    v_scale = float(vs[:a.ntrain + 1].max()) + 1e-12

    g = torch.Generator(device="cpu").manual_seed(3)
    nu = us.clone(); nv = vs.clone()
    nu[:a.ntrain + 1] += a.noise * u_scale * torch.randn(nu[:a.ntrain + 1].shape, generator=g).to(nu.device).double()
    nv[:a.ntrain + 1] += a.noise * v_scale * torch.randn(nv[:a.ntrain + 1].shape, generator=g).to(nv.device).double()

    # training points: the noisy snapshots of the training window
    x, y = solver.x, solver.y
    tt = torch.arange(a.ntrain + 1, device=us.device, dtype=us.dtype) * Dt
    X, Y, T3 = torch.meshgrid(x, y, tt, indexing="ij")
    ud = nu[:a.ntrain + 1].permute(1, 2, 0)
    vd = nv[:a.ntrain + 1].permute(1, 2, 0)
    xd, yd, td = X.reshape(-1), Y.reshape(-1), T3.reshape(-1)

    p = PINN2D(T, 1.0, u_scale, v_scale, variant=a.variant, seed=a.seed, device=a.device)
    info = p.fit(xd, yd, td, ud.reshape(-1), vd.reshape(-1), adam_iters=a.adam,
                 n_coll=a.n_coll, batch_data=a.batch_data)
    tg = torch.arange(a.n_macro + 1, device=us.device, dtype=us.dtype) * Dt
    pu, pv = p.predict(x, y, tg)
    m = metrics(pu.double(), pv.double(), us, vs, a.ntrain, solver.hx, solver.hy)
    m |= dict(seconds=info["seconds"], coef=info["coef"], variant=a.variant, seed=a.seed,
              params=sum(q.numel() for q in p.params if q.requires_grad),
              data_mse=info["data"], res_mse=info["res"], precision="float32")
    print(f"PINN2D {a.variant} seed{a.seed}: relL2 u {m['relL2_u']:.3e} (extrap {m['relL2_u_extrap']:.3e}) "
          f"min u {m['min_u']:.3e} mass drift {m['mass_drift']:.3e} {info['seconds']:.0f}s")
    print(f"  recovered {info['coef']}")
    out = os.path.join(HERE, "pinn2d_runs")
    os.makedirs(out, exist_ok=True)
    torch.save(dict(pu=pu.cpu(), pv=pv.cpu()), os.path.join(out, f"pred_{a.variant}_{a.seed}.pt"))
    with open(os.path.join(out, f"pinn2d_{a.variant}_{a.seed}.json"), "w") as f:
        json.dump(m, f, indent=2, default=float)


if __name__ == "__main__":
    main()
