"""Physics-informed baselines for the 1D problems, in the same inverse setting as the NMINN.

The comparison is deliberately favourable to the PINN. Both families see the same noisy snapshots
on the training window; the PINN is additionally told the exact analytic form of the equation and
is allowed to enforce its residual on the whole space-time domain, including the extrapolation
window that the learned steppers never see. Only the coefficients are unknown to both.

Four variants, in increasing order of how hard they try to fix the structure by hand:
  vanilla   data + residual
  penalty   data + residual + a penalty on negative values (and on mass drift, problem B)
  hardpos   output passed through a softplus, so u >= 0 pointwise by construction
  project   hardpos, then rescaled at evaluation time to the exact initial mass (problem B)
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, din, dout, width=64, depth=4, act=torch.tanh):
        super().__init__()
        dims = [din] + [width] * depth + [dout]
        self.lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.act = act
        for m in self.lin:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z):
        for m in self.lin[:-1]:
            z = self.act(m(z))
        return self.lin[-1](z)


def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


class PINN1D:
    """Inverse PINN for either the Fisher-KPP scalar problem or the two-species exchange."""

    def __init__(self, problem, T, L=1.0, variant="vanilla", width=64, depth=4,
                 w_res=1.0, w_bc=1.0, w_pen=10.0, seed=0, device="cuda", dtype=torch.float64):
        assert problem in ("A", "B")
        assert variant in ("vanilla", "penalty", "hardpos", "project")
        torch.manual_seed(seed)
        self.problem, self.variant, self.T, self.L = problem, variant, T, L
        self.dev, self.dt_type = torch.device(device), dtype
        self.nout = 1 if problem == "A" else 2
        self.net = MLP(2, self.nout, width, depth).to(self.dev).to(dtype)
        self.w_res, self.w_bc, self.w_pen = w_res, w_bc, w_pen
        # unknown coefficients, learned from the data exactly like the NMINN learns its own
        if problem == "A":
            init = [math.log(0.02), math.log(3.0)]          # log d, log r
        else:
            init = [math.log(0.02), math.log(0.05), math.log(1.0)]   # log d1, log d2, log k
        self.coef = nn.Parameter(torch.tensor(init, device=self.dev, dtype=dtype))
        self.params = list(self.net.parameters()) + [self.coef]

    # ---------------------------------------------------------------- forward
    def raw(self, x, t):
        z = torch.stack([x / self.L, t / self.T], dim=-1)
        return self.net(z)

    def u(self, x, t):
        out = self.raw(x, t)
        if self.variant in ("hardpos", "project"):
            out = nn.functional.softplus(out)
        return out

    # ---------------------------------------------------------------- residual
    def residual(self, x, t):
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        U = self.u(x, t)
        if self.problem == "A":
            d, r = torch.exp(self.coef[0]), torch.exp(self.coef[1])
            u = U[..., 0]
            ut = _grad(u, t)
            ux = _grad(u, x)
            uxx = _grad(ux, x)
            return (ut - d * uxx - r * u * (1.0 - u)).unsqueeze(-1)
        d1, d2, k = torch.exp(self.coef[0]), torch.exp(self.coef[1]), torch.exp(self.coef[2])
        u1, u2 = U[..., 0], U[..., 1]
        u1t, u2t = _grad(u1, t), _grad(u2, t)
        u1xx = _grad(_grad(u1, x), x)
        u2xx = _grad(_grad(u2, x), x)
        q = k * u1 * u2 / (1.0 + u1)
        return torch.stack([u1t - d1 * u1xx + q, u2t - d2 * u2xx - q], dim=-1)

    def neumann(self, t):
        out = 0.0
        for xb in (0.0, self.L):
            x = torch.full_like(t, xb).requires_grad_(True)
            U = self.u(x, t)
            for c in range(self.nout):
                out = out + _grad(U[..., c], x).pow(2).mean()
        return out

    # ---------------------------------------------------------------- training
    def fit(self, xd, td, ud, n_coll=20000, adam_iters=15000, lbfgs_iters=3000, mass0=None, log=None):
        dev, dt = self.dev, self.dt_type
        xd, td, ud = [torch.as_tensor(a, device=dev, dtype=dt) for a in (xd, td, ud)]
        g = torch.Generator(device="cpu").manual_seed(1234)
        xf = (torch.rand(n_coll, generator=g).to(dev).to(dt) * self.L)
        tf = (torch.rand(n_coll, generator=g).to(dev).to(dt) * self.T)
        tb = (torch.rand(2000, generator=g).to(dev).to(dt) * self.T)
        h = self.L / 64.0

        def closure_loss():
            pred = self.u(xd, td)
            l_data = (pred - ud).pow(2).mean()
            l_res = self.residual(xf, tf).pow(2).mean()
            l_bc = self.neumann(tb)
            loss = l_data + self.w_res * l_res + self.w_bc * l_bc
            if self.variant == "penalty":
                raw = self.raw(xf, tf)
                loss = loss + self.w_pen * raw.clamp(max=0.0).pow(2).mean()
                if self.problem == "B" and mass0 is not None:
                    # penalty on the relative drift of the total mass across time slices
                    xs = torch.linspace(0.5 * h, self.L - 0.5 * h, 64, device=dev, dtype=dt)
                    ts = torch.linspace(0.0, self.T, 12, device=dev, dtype=dt)
                    X, Tt = torch.meshgrid(xs, ts, indexing="ij")
                    U = self.u(X.reshape(-1), Tt.reshape(-1)).reshape(64, 12, 2)
                    m = U.sum(dim=(0, 2))
                    loss = loss + self.w_pen * ((m - mass0) / mass0).pow(2).mean()
            return loss, l_data.item(), l_res.item()

        t0 = time.time()
        opt = torch.optim.Adam(self.params, lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_iters, eta_min=1e-5)
        for it in range(adam_iters):
            opt.zero_grad(set_to_none=True)
            loss, ld, lr_ = closure_loss()
            loss.backward()
            opt.step()
            sched.step()
            if log and it % max(1, adam_iters // 5) == 0:
                print(f"    adam {it:6d} loss={loss.item():.3e} data={ld:.3e} res={lr_:.3e}", flush=True)
        opt2 = torch.optim.LBFGS(self.params, max_iter=lbfgs_iters, history_size=50,
                                 tolerance_grad=1e-12, tolerance_change=1e-14,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad(set_to_none=True)
            loss, _, _ = closure_loss()
            loss.backward()
            return loss

        opt2.step(closure)
        final, ld, lr_ = closure_loss()
        return dict(loss=float(final.detach()), data=ld, res=lr_, seconds=time.time() - t0,
                    coef=[float(c) for c in torch.exp(self.coef.detach())])

    # ---------------------------------------------------------------- evaluation
    @torch.no_grad()
    def predict(self, x, tgrid):
        dev, dt = self.dev, self.dt_type
        x = torch.as_tensor(x, device=dev, dtype=dt)
        tgrid = torch.as_tensor(tgrid, device=dev, dtype=dt)
        X, T = torch.meshgrid(x, tgrid, indexing="ij")
        U = self.u(X.reshape(-1), T.reshape(-1)).reshape(len(x), len(tgrid), self.nout)
        return U.permute(1, 2, 0)              # (nt, nfields, nx)
