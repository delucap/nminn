"""Physics-informed baseline for the 2D Keller-Segel problem, in the same inverse setting.

Same concession as in one dimension: the network is told the analytic form of the system and may
enforce its residual on the whole space-time cylinder, including the extrapolation window, while
the learned steppers only ever see the training snapshots. Random Fourier features are used
because a plain tanh network cannot resolve the aggregation peaks at all.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn


def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


class FourierMLP(nn.Module):
    def __init__(self, din=3, dout=2, width=128, depth=5, n_feat=64, sigma=6.0, dtype=torch.float32):
        super().__init__()
        self.B = nn.Parameter(sigma * torch.randn(din, n_feat, dtype=dtype), requires_grad=False)
        dims = [2 * n_feat] + [width] * depth + [dout]
        self.lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], dtype=dtype) for i in range(len(dims) - 1)])
        for m in self.lin:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z):
        p = 2 * math.pi * (z @ self.B)
        h = torch.cat([torch.sin(p), torch.cos(p)], dim=-1)
        for m in self.lin[:-1]:
            h = torch.tanh(m(h))
        return self.lin[-1](h)


class PINN2D:
    def __init__(self, T, L=1.0, u_scale=1.0, v_scale=1.0, variant="vanilla", seed=0,
                 width=128, depth=5, w_res=1.0, w_bc=1.0, w_pen=10.0,
                 device="cuda", dtype=torch.float32):
        torch.manual_seed(seed)
        self.T, self.L, self.variant = T, L, variant
        self.us, self.vs = u_scale, v_scale
        self.dev, self.dtype = torch.device(device), dtype
        self.net = FourierMLP(3, 2, width, depth, dtype=dtype).to(self.dev)
        self.w_res, self.w_bc, self.w_pen = w_res, w_bc, w_pen
        self.coef = nn.Parameter(torch.tensor(
            [math.log(3e-4), math.log(3e-3), math.log(5e-3), math.log(0.5), math.log(0.5)],
            device=self.dev, dtype=dtype))       # log Du, Dv, chi, alpha, beta
        self.params = list(self.net.parameters()) + [self.coef]

    def fields(self, x, y, t):
        z = torch.stack([x / self.L, y / self.L, t / self.T], dim=-1)
        out = self.net(z)
        if self.variant == "hardpos":
            out = nn.functional.softplus(out)
        return out[..., 0] * self.us, out[..., 1] * self.vs

    def residual(self, x, y, t):
        x, y, t = (a.clone().requires_grad_(True) for a in (x, y, t))
        u, v = self.fields(x, y, t)
        Du, Dv, chi, al, be = torch.exp(self.coef)
        ux, uy, ut = _grad(u, x), _grad(u, y), _grad(u, t)
        vx, vy, vt = _grad(v, x), _grad(v, y), _grad(v, t)
        uxx, uyy = _grad(ux, x), _grad(uy, y)
        vxx, vyy = _grad(vx, x), _grad(vy, y)
        ru = ut - Du * (uxx + uyy) + chi * (ux * vx + uy * vy + u * (vxx + vyy))
        rv = vt - Dv * (vxx + vyy) - al * u + be * v
        return ru / self.us, rv / self.vs

    def bc(self, t):
        loss = 0.0
        n = t.shape[0]
        for axis in (0, 1):
            for b in (0.0, self.L):
                r = torch.rand(n, device=self.dev, dtype=self.dtype) * self.L
                x = torch.full_like(r, b) if axis == 0 else r
                y = r if axis == 0 else torch.full_like(r, b)
                x, y = x.requires_grad_(True), y.requires_grad_(True)
                u, v = self.fields(x, y, t)
                d = x if axis == 0 else y
                loss = loss + (_grad(u, d) / self.us).pow(2).mean() + (_grad(v, d) / self.vs).pow(2).mean()
        return loss

    def fit(self, xd, yd, td, ud, vd, n_coll=30000, adam_iters=15000, batch_data=20000, log=2000):
        dev, dt = self.dev, self.dtype
        xd, yd, td, ud, vd = [torch.as_tensor(a, device=dev, dtype=dt) for a in (xd, yd, td, ud, vd)]
        g = torch.Generator(device="cpu").manual_seed(5)
        xf = torch.rand(n_coll, generator=g).to(dev).to(dt) * self.L
        yf = torch.rand(n_coll, generator=g).to(dev).to(dt) * self.L
        tf = torch.rand(n_coll, generator=g).to(dev).to(dt) * self.T
        tb = torch.rand(2000, generator=g).to(dev).to(dt) * self.T
        opt = torch.optim.Adam(self.params, lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_iters, eta_min=1e-5)
        t0 = time.time()
        nd = xd.shape[0]
        for it in range(adam_iters):
            idx = torch.randint(0, nd, (min(batch_data, nd),), device=dev)
            opt.zero_grad(set_to_none=True)
            u, v = self.fields(xd[idx], yd[idx], td[idx])
            l_data = ((u - ud[idx]) / self.us).pow(2).mean() + ((v - vd[idx]) / self.vs).pow(2).mean()
            ru, rv = self.residual(xf, yf, tf)
            l_res = ru.pow(2).mean() + rv.pow(2).mean()
            loss = l_data + self.w_res * l_res + self.w_bc * self.bc(tb)
            if self.variant == "penalty":
                uu, vv = self.fields(xf, yf, tf)
                loss = loss + self.w_pen * ((uu / self.us).clamp(max=0.0).pow(2).mean()
                                            + (vv / self.vs).clamp(max=0.0).pow(2).mean())
            loss.backward()
            opt.step()
            sched.step()
            if log and it % log == 0:
                print(f"    adam {it:6d} loss={loss.item():.3e} data={l_data.item():.3e} "
                      f"res={l_res.item():.3e}", flush=True)
        return dict(loss=float(loss.detach()), data=float(l_data.detach()), res=float(l_res.detach()),
                    seconds=time.time() - t0,
                    coef=dict(zip(["Du", "Dv", "chi", "alpha", "beta"],
                                  [float(c) for c in torch.exp(self.coef.detach())])))

    @torch.no_grad()
    def predict(self, xg, yg, tg, chunk=200000):
        dev, dt = self.dev, self.dtype
        X, Y, T = torch.meshgrid(torch.as_tensor(xg, device=dev, dtype=dt),
                                 torch.as_tensor(yg, device=dev, dtype=dt),
                                 torch.as_tensor(tg, device=dev, dtype=dt), indexing="ij")
        sh = X.shape
        xs, ys, ts = X.reshape(-1), Y.reshape(-1), T.reshape(-1)
        us, vs = [], []
        for i in range(0, xs.shape[0], chunk):
            a, b = self.fields(xs[i:i + chunk], ys[i:i + chunk], ts[i:i + chunk])
            us.append(a); vs.append(b)
        u = torch.cat(us).reshape(sh).permute(2, 0, 1)
        v = torch.cat(vs).reshape(sh).permute(2, 0, 1)
        return u, v
