"""Gradient check of the uniformization transport layer and of the full method-informed stepper.

Reviewer 1 (JOCS, first round): the implemented finite-K renormalized operator depends on theta, on the
number of substeps and on the truncation order, so state what function is differentiated in training and
check its gradient.

Part A, the layer alone. F(S) = <w, E_K(tau; A(a(S))) u> with the face sensitivities S as the variable and
theta, K and nsub recomputed from S inside F, i.e. F is the implemented map. Three derivatives along a
random direction d are compared:
    frozen   autograd with theta frozen (the training default),
    total    autograd with theta differentiated (theta_grad=True), K and nsub fixed,
    fd       fourth-order central differences of F itself over a sweep of step sizes.
The discrepancy |frozen - total| is compared with the bound 2 tau pi_K(mu)/(1-rho_K)^2 |w|_inf |u|_1 |d theta . d|
of the theta-sensitivity proposition, for tolerances from 1e-14 down to 1e-2.

Part B, the whole stepper. The training loss over one window, differentiated with respect to all parameters
with theta frozen as in training, against fourth-order central differences along three random directions
(coefficients only, closure network only, all parameters), at the trained weights on the 128^2 grid and at
random weights on a 64^2 grid.

    python gradcheck_taxis.py --device cuda --tag _r1
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from ks2d import KS2D, KSParams
from models2d import NMINN2D
from ops import poisson_pmf, poisson_tail, uniformization
from transport import UpwindTaxis2D

HERE = os.path.dirname(os.path.abspath(__file__))
TRUE = KSParams(Du=1.0e-4, Dv=1.0e-3, chi=2.0e-2, alpha=1.0, beta=1.0)


def fd4(f, p, d, eps):
    """Fourth-order central difference of the scalar map f along d at p."""
    return float((-f(p + 2 * eps * d) + 8 * f(p + eps * d) - 8 * f(p - eps * d) + f(p - 2 * eps * d)) / (12 * eps))


def relerr(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# ------------------------------------------------------------------------------------------ part A
def part_A(device, dtype, M=64, seed=0, taus=(5e-3, 2e-2)):
    torch.manual_seed(seed)
    solver = KS2D(M, M, params=TRUE, device=device, dtype=dtype)
    u0, v0 = solver.initial()
    us, vs = solver.run(u0, v0, 5e-4, 2000, save_every=2000)        # t = 1.0, aggregation under way
    u, v = us[-1], vs[-1]
    tx = UpwindTaxis2D(solver.hx, solver.hy)
    g = torch.Generator(device="cpu").manual_seed(seed)
    Sx0 = TRUE.chi * (1.0 + 0.5 * torch.rand(M - 1, M, generator=g)).to(device).to(dtype)
    Sy0 = TRUE.chi * (1.0 + 0.5 * torch.rand(M, M - 1, generator=g)).to(device).to(dtype)
    w = torch.rand(M, M, generator=g).to(device).to(dtype)
    p0 = torch.cat([Sx0.reshape(-1), Sy0.reshape(-1)])
    d = torch.randn(p0.numel(), generator=g).to(device).to(dtype)
    d = d / d.norm()
    nx = (M - 1) * M

    def split(p):
        return p[:nx].reshape(M - 1, M), p[nx:].reshape(M, M - 1)

    def F(p, tau, tol, mode):
        Sx, Sy = split(p)
        ax, ay = tx.velocities(v, Sx, Sy)
        th = tx.theta(ax, ay)
        if mode == "frozen":
            th = th.detach()
        out = uniformization(lambda z: tx.apply(z, ax, ay), u, tau, th, tol=tol,
                             theta_grad=(mode == "total"))
        return (w * out).sum()

    def theta_of(p):
        Sx, Sy = split(p)
        ax, ay = tx.velocities(v, Sx, Sy)
        return tx.theta(ax, ay)

    rows = []
    for tau in taus:
        for tol in (1e-14, 1e-10, 1e-6, 1e-2):
            # what the implemented map uses at p0
            Sx, Sy = split(p0)
            ax, ay = tx.velocities(v, Sx, Sy)
            th0 = float(tx.theta(ax, ay))
            _, info = uniformization(lambda z: tx.apply(z, ax, ay), u, tau, th0, tol=tol, return_info=True)
            mu, K, nsub = info["mu"], info["K"], info["nsub"]
            piK, rhoK = poisson_pmf(mu, K), poisson_tail(mu, K)
            grads = {}
            for mode in ("frozen", "total"):
                p = p0.clone().requires_grad_(True)
                val = F(p, tau, tol, mode)
                if K == 0:      # the layer is the identity at this tolerance, no dependence on p
                    grads[mode] = 0.0
                    continue
                (gp,) = torch.autograd.grad(val, p)
                grads[mode] = float((gp * d).sum())
            p = p0.clone().requires_grad_(True)
            (gth,) = torch.autograd.grad(theta_of(p), p)
            dtheta_d = float((gth * d).sum())
            bound = 2 * tau * piK / (1 - rhoK) ** 2 * float(w.abs().max()) * float(u.abs().sum()) * abs(dtheta_d)
            with torch.no_grad():
                fds = {}
                for e in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
                    eps = TRUE.chi * e
                    fds[e] = fd4(lambda q: F(q, tau, tol, "none"), p0, d, eps)
            best_e = min(fds, key=lambda e: relerr(fds[e], grads["total"]))
            row = dict(tau=tau, tol=tol, theta=th0, mu=mu, K=K, nsub=nsub, piK=piK, rhoK=rhoK,
                       g_frozen=grads["frozen"], g_total=grads["total"], fd_best=fds[best_e], fd_eps=best_e,
                       fd_all={str(e): v_ for e, v_ in fds.items()},
                       rel_frozen_vs_fd=relerr(grads["frozen"], fds[best_e]),
                       rel_total_vs_fd=relerr(grads["total"], fds[best_e]),
                       abs_frozen_minus_total=abs(grads["frozen"] - grads["total"]),
                       bound_frozen_minus_total=bound, dtheta_d=dtheta_d)
            rows.append(row)
            print(f"tau={tau:.0e} tol={tol:.0e} theta={th0:8.2f} mu={mu:5.2f} nsub={nsub} K={K:2d} "
                  f"piK={piK:.1e}  frozen={grads['frozen']:+.10e} total={grads['total']:+.10e} "
                  f"fd={fds[best_e]:+.10e}  |frozen-fd|/|fd|={row['rel_frozen_vs_fd']:.1e} "
                  f"|total-fd|/|fd|={row['rel_total_vs_fd']:.1e}  |frozen-total|={row['abs_frozen_minus_total']:.1e} "
                  f"bound={bound:.1e}", flush=True)
    return rows


# ------------------------------------------------------------------------------------------ part B
def window_loss(model, us, vs, s, K):
    nrm = lambda a: a.pow(2).mean().clamp(min=1e-30)
    u, v = us[s], vs[s]
    loss = 0.0
    for j in range(1, K + 1):
        u, v = model(u, v)
        loss = loss + (u - us[s + j]).pow(2).mean() / nrm(us[s + j]) + (v - vs[s + j]).pow(2).mean() / nrm(vs[s + j])
    return loss / K


def part_B(device, dtype, M, tag, K=3, s=10, seed=0, trained=True):
    solver = KS2D(M, M, params=TRUE, device=device, dtype=dtype)
    u0, v0 = solver.initial()
    us, vs = solver.run(u0, v0, 5e-4, (s + K) * 40, save_every=40)
    setup_path = os.path.join(HERE, f"results_2d_M{M}{tag}.json")
    if trained and os.path.exists(setup_path):
        setup = json.load(open(setup_path))["_setup"]
        u_scale, v_scale = setup["u_scale"], setup["v_scale"]
    else:
        u_scale, v_scale = float(us.max()), float(vs.max()) + 1e-12
    model = NMINN2D(M, solver.hx, 2e-2, n_sub=4, u_scale=u_scale, v_scale=v_scale, device=device, dtype=dtype)
    label = "random weights"
    if trained:
        wpath = os.path.join(HERE, f"model_NMINN_M{M}{tag}.pt")
        if not os.path.exists(wpath):
            wpath = os.path.join(HERE, f"model_NMINN_M{M}.pt")
        model.load_state_dict(torch.load(wpath, map_location=device))
        label = f"trained weights ({os.path.basename(wpath)})"
    params = [p for p in model.parameters()]
    p0 = torch.nn.utils.parameters_to_vector(params).detach().clone()
    ncoef = 5   # logc (4) + log_chi (1) come first in the parameter order
    names = [n for n, _ in model.named_parameters()]
    assert names[0] == "logc" and names[1] == "log_chi", names[:2]

    def f(p):
        torch.nn.utils.vector_to_parameters(p, params)
        with torch.no_grad():
            return window_loss(model, us, vs, s, K)

    torch.nn.utils.vector_to_parameters(p0, params)
    for q in params:
        q.grad = None
    loss = window_loss(model, us, vs, s, K)
    loss.backward()
    g = torch.cat([q.grad.reshape(-1) for q in params]).detach().clone()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dirs = {}
    dc = torch.zeros_like(p0); dc[:ncoef] = torch.randn(ncoef, generator=gen).to(device).to(dtype)
    dn = torch.zeros_like(p0); dn[ncoef:] = torch.randn(p0.numel() - ncoef, generator=gen).to(device).to(dtype)
    da = torch.randn(p0.numel(), generator=gen).to(device).to(dtype)
    for k_, d_ in (("coefficients", dc), ("closure network", dn), ("all parameters", da)):
        dirs[k_] = d_ / d_.norm()
    rows = []
    for k_, d_ in dirs.items():
        gd = float((g * d_).sum())
        fds = {e: fd4(f, p0, d_, e) for e in (1e-2, 1e-3, 1e-4, 1e-5)}
        best_e = min(fds, key=lambda e: relerr(fds[e], gd))
        rows.append(dict(M=M, weights=label, direction=k_, autograd=gd, fd_best=fds[best_e], fd_eps=best_e,
                         fd_all={str(e): v_ for e, v_ in fds.items()}, rel=relerr(gd, fds[best_e]),
                         loss=float(loss), K=K, start=s))
        print(f"M={M} {label:<45} {k_:<16} autograd={gd:+.10e} fd={fds[best_e]:+.10e} "
              f"rel={rows[-1]['rel']:.1e} (eps={best_e:.0e})", flush=True)
    torch.nn.utils.vector_to_parameters(p0, params)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dtype = torch.float64
    print("=== Part A: uniformization layer, frozen theta vs total derivative vs finite differences ===")
    A = part_A(a.device, dtype)
    print("\n=== Part B: full stepper, autograd (theta frozen) vs finite differences ===")
    B = part_B(a.device, dtype, 128, a.tag, trained=True) + part_B(a.device, dtype, 64, a.tag, trained=False)
    out = dict(part_A=A, part_B=B, torch=torch.__version__, device=a.device)
    with open(os.path.join(HERE, f"gradcheck_taxis{a.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote gradcheck_taxis{a.tag}.json")


if __name__ == "__main__":
    main()
