"""2D Keller-Segel surrogate study on the GPU.

Reference trajectory from the symmetric splitting integrator at a fine timestep; three learned
macro-steppers trained on the same noisy snapshots of the first part of the trajectory and rolled
out over the whole horizon. Usage:

    python run2d.py --M 128 --iters 1500
    python run2d.py --gridstudy      (zero-shot transfer of the trained closure to other grids)
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


def make_data(M, dt_ref, n_macro, sub, device, dtype, chi=None, ic="bumps", ic_seed=0):
    p = KSParams(**{**TRUE.__dict__, **({"chi": chi} if chi else {})})
    solver = KS2D(M, M, params=p, device=device, dtype=dtype)
    u0, v0 = solver.initial(kind=ic, seed=ic_seed)
    t0 = time.time()
    us, vs = solver.run(u0, v0, dt_ref, n_macro * sub, save_every=sub)
    if device == "cuda":
        torch.cuda.synchronize()
    return solver, us, vs, time.time() - t0


def relL2(a, b):
    return float((a - b).norm() / b.norm())


def metrics(pu, pv, tu, tv, ntrain, hx, hy):
    n = min(len(pu), len(tu))
    pu, pv, tu, tv = pu[:n], pv[:n], tu[:n], tv[:n]
    ok = bool(torch.isfinite(pu).all() and torch.isfinite(pv).all())
    if not ok:
        return dict(relL2_u=float("inf"), relL2_v=float("inf"), relL2_u_extrap=float("inf"),
                    min_u=float("-inf"), min_v=float("-inf"), mass_drift=float("inf"),
                    peak_ratio=float("nan"), steps=n, finite=False)
    m = pu.reshape(n, -1).sum(dim=1) * hx * hy
    return dict(relL2_u=relL2(pu, tu), relL2_v=relL2(pv, tv),
                relL2_u_extrap=relL2(pu[ntrain:], tu[ntrain:]),
                min_u=float(pu.min()), min_v=float(pv.min()),
                mass_drift=float(((m - m[0]).abs() / m[0]).max()),
                peak_ratio=float(pu[-1].max() / tu[-1].max()), steps=n, finite=True)


def rollout(model, u0, v0, nsteps):
    u, v = u0.clone(), v0.clone()
    us, vs = [u], [v]
    with torch.no_grad():
        for _ in range(nsteps):
            u, v = model(u, v)
            if not (torch.isfinite(u).all() and torch.isfinite(v).all()):
                break
            us.append(u)
            vs.append(v)
    return torch.stack(us), torch.stack(vs)


def train(model, us, vs, ntrain, K=6, iters=900, batch=4, lr=3e-3, coef_lr=3e-2, log=100):
    """Relative multi-step rollout loss.

    Two choices matter here and were fixed after a first attempt failed. The horizon K must be long
    enough that the systematic error of a wrong coefficient exceeds the noise floor over the window,
    otherwise the loss is flat in the coefficients and the minimizer of the noisy objective is an
    over-diffusive model, which is what a short horizon produced. And the per-snapshot relative
    normalization keeps the early low-amplitude part of the trajectory from being ignored in favour of
    the late aggregation peaks, which would otherwise dominate an absolute mean-squared error.
    """
    groups = []
    named = dict(model.named_parameters())
    coef_keys = [k for k in named if k.startswith("log")]
    if coef_keys:
        groups.append(dict(params=[named[k] for k in coef_keys], lr=coef_lr))
    groups.append(dict(params=[v for k, v in named.items() if k not in coef_keys], lr=lr))
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-5)
    starts = torch.arange(0, ntrain - K + 1)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    hist = []
    nrm = lambda a: a.pow(2).mean().clamp(min=1e-30)
    for it in range(iters):
        idx = starts[torch.randperm(len(starts), generator=g)[:batch]]
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for s in idx.tolist():
            u, v = us[s], vs[s]
            for j in range(1, K + 1):
                u, v = model(u, v)
                loss = loss + (u - us[s + j]).pow(2).mean() / nrm(us[s + j]) \
                    + (v - vs[s + j]).pow(2).mean() / nrm(vs[s + j])
        loss = loss / (len(idx) * K)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for gr in groups for p in gr["params"]], 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
        if log and it % log == 0:
            print(f"    it {it:5d}  loss {hist[-1]:.4e}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return dict(train_loss=hist[-1], seconds=time.time() - t0, iters=iters)


def timed_rollout(model, u0, v0, nsteps, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        rollout(model, u0, v0, nsteps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--dt_ref", type=float, default=5e-4)
    ap.add_argument("--n_macro", type=int, default=80)
    ap.add_argument("--sub", type=int, default=40, help="fine steps per macro step")
    ap.add_argument("--n_sub", type=int, default=4, help="substeps inside the method-informed stepper")
    ap.add_argument("--ntrain", type=int, default=25)
    ap.add_argument("--iters", type=int, default=900)
    ap.add_argument("--K", type=int, default=6, help="macro steps in the training rollout loss")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--models", nargs="+", default=["NMINN", "CNN", "FNO"],
                    help="any of " + " ".join(MODEL_NAMES))
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--noise_seed", type=int, default=3, help="seed of the noise realization (3 = paper)")
    ap.add_argument("--ic", default="bumps", help="initial datum kind, see KS2D.initial")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--gridstudy", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dtype = dict(float64=torch.float64, float32=torch.float32)[a.dtype]
    torch.manual_seed(0)

    solver, us, vs, ref_sec = make_data(a.M, a.dt_ref, a.n_macro, a.sub, a.device, dtype, ic=a.ic)
    Dt = a.dt_ref * a.sub
    hx, hy = solver.hx, solver.hy
    u_scale, v_scale = float(us[:a.ntrain + 1].max()), float(vs[:a.ntrain + 1].max()) + 1e-12
    print(f"reference: M={a.M} dt={a.dt_ref} macro Dt={Dt} steps={a.n_macro} "
          f"T={Dt*a.n_macro:.2f}  {ref_sec:.2f} s")
    print(f"  u in [{float(us.min()):.3e}, {float(us.max()):.3e}]  peak growth "
          f"{float(us[-1].max()/us[0].max()):.1f}x  mass drift "
          f"{float(((us.reshape(len(us),-1).sum(1)-us[0].sum()).abs()/us[0].sum()).max()):.2e}")

    # noise is scaled per snapshot: the amplitude of this problem grows by orders of magnitude over
    # the horizon, so a single global scale would mean a relative perturbation of the early data
    # tens of times larger than of the late data
    g = torch.Generator(device="cpu").manual_seed(a.noise_seed)
    nu, nv = us.clone(), vs.clone()
    for i in range(a.ntrain + 1):
        nu[i] += a.noise * float(us[i].max()) * torch.randn(us[i].shape, generator=g).to(nu.device).to(dtype)
        nv[i] += a.noise * max(float(vs[i].max()), 1e-12) * torch.randn(vs[i].shape, generator=g).to(nv.device).to(dtype)

    mass0 = float(us[0].sum())
    shorts = [s for s in MODEL_NAMES if s in a.models]
    models = {MODEL_NAMES[s]: build_model(s, a.M, hx, Dt, a.n_sub, u_scale, v_scale, mass0, a.device, dtype)
              for s in shorts}
    res = {"_setup": dict(M=a.M, dt_ref=a.dt_ref, Dt=Dt, n_macro=a.n_macro, ntrain=a.ntrain,
                          noise=a.noise, noise_scaling="per snapshot maximum", noise_seed=a.noise_seed,
                          ic=a.ic, K=a.K, batch=a.batch, iters=a.iters, n_sub=a.n_sub,
                          dtype=a.dtype, reference_seconds=ref_sec,
                          true=TRUE.__dict__, u_scale=u_scale, v_scale=v_scale, mass0=mass0,
                          torch=torch.__version__, device_name=(torch.cuda.get_device_name(0)
                                                                if a.device == "cuda" else "cpu"))}
    preds = {}
    for name, model in models.items():
        print(f"\n-- training {name} ({count_params(model)} parameters)")
        info = train(model, nu, nv, a.ntrain, K=a.K, batch=a.batch, iters=a.iters)
        pu, pv = rollout(model, us[0], vs[0], a.n_macro)
        m = metrics(pu, pv, us, vs, a.ntrain, hx, hy)
        m |= info | dict(params=count_params(model))
        m["inference_seconds"] = timed_rollout(model, us[0], vs[0], a.n_macro)
        m["speedup_vs_reference"] = ref_sec / m["inference_seconds"]
        if hasattr(model, "coefficients"):
            c = model.coefficients()
            # chi multiplies a learned sensitivity network, and the taxis velocity scales with the
            # quasi-steady signal alpha/beta, so the identifiable quantity is the product below,
            # not the three coefficients separately
            with torch.no_grad():
                s_mean = float(model.sensitivity(us[0], vs[0]).mean())
            c["mean_sensitivity"] = s_mean
            c["identifiable_chi_alpha_over_beta"] = s_mean * c["alpha"] / c["beta"]
            c["true_chi_alpha_over_beta"] = TRUE.chi * TRUE.alpha / TRUE.beta
            m["coef"] = c
        res[name] = m
        preds[name] = (pu.cpu(), pv.cpu())
        print(f"   relL2 u {m['relL2_u']:.3e} (extrap {m['relL2_u_extrap']:.3e})  min u {m['min_u']:.3e}"
              f"  mass drift {m['mass_drift']:.3e}  infer {m['inference_seconds']:.2f}s")
        if "coef" in m:
            print(f"   recovered {m['coef']}")
        short = [s for s in MODEL_NAMES if MODEL_NAMES[s] == name][0]
        torch.save(model.state_dict(), os.path.join(HERE, f"model_{short}_M{a.M}{a.tag}.pt"))

    torch.save(dict(us=us.cpu(), vs=vs.cpu(), preds=preds, setup=res["_setup"]),
               os.path.join(HERE, f"res2d_M{a.M}{a.tag}.pt"))
    with open(os.path.join(HERE, f"results_2d_M{a.M}{a.tag}.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    print("\n=== Problem C: 2D Keller-Segel aggregation ===")
    print(f"{'method':<14}{'params':>8}{'relL2 u':>11}{'extrap':>11}{'relL2 v':>11}"
          f"{'min u':>12}{'mass drift':>12}{'train s':>9}{'infer s':>9}")
    for k, v in res.items():
        if k.startswith("_"):
            continue
        print(f"{k:<14}{v['params']:>8}{v['relL2_u']:>11.3e}{v['relL2_u_extrap']:>11.3e}"
              f"{v['relL2_v']:>11.3e}{v['min_u']:>12.3e}{v['mass_drift']:>12.3e}"
              f"{v['seconds']:>9.1f}{v['inference_seconds']:>9.2f}")
    print(f"reference solver: {ref_sec:.2f} s for the same horizon")

    if a.gridstudy:
        print("\n=== zero-shot transfer of the trained closure to other resolutions ===")
        nm = models["NMINN"]
        out = {}
        for M2 in (64, 128, 256):
            s2, u2, v2, sec2 = make_data(M2, a.dt_ref, a.n_macro, a.sub, a.device, dtype, ic=a.ic)
            m2 = build_model("NMINN", M2, s2.hx, Dt, a.n_sub, u_scale, v_scale, mass0, a.device, dtype)
            m2.load_state_dict(nm.state_dict())
            pu, pv = rollout(m2, u2[0], v2[0], a.n_macro)
            mm = metrics(pu, pv, u2, v2, a.ntrain, s2.hx, s2.hy)
            mm["inference_seconds"] = timed_rollout(m2, u2[0], v2[0], a.n_macro)
            mm["reference_seconds"] = sec2
            out[M2] = mm
            print(f"  M={M2:<5} relL2 u {mm['relL2_u']:.3e}  min u {mm['min_u']:.3e}  "
                  f"mass drift {mm['mass_drift']:.3e}  infer {mm['inference_seconds']:.2f}s "
                  f"(reference {sec2:.2f}s)")
        with open(os.path.join(HERE, f"results_2d_grid{a.tag}.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
