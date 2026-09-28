"""Reference solver and data generation for the 2D Keller-Segel aggregation problem.

    u_t = Du div(grad u) - div(chi u grad v)          cell density, u >= 0, total mass conserved
    v_t = Dv div(grad v) + alpha u - beta v           chemoattractant, v >= 0

on a rectangle with no-flux boundaries. The reference integrator is a Strang splitting of three
exactly solvable pieces: the exponential of the Neumann Laplacian for u, the uniformization
exponential of the upwind taxis generator, and an exponential time differencing step for v. Each
piece is unconditionally positive and the first two preserve the mass of u exactly, so the
reference trajectory carries the invariants to round off and can be used to score a surrogate on
them without the reference itself being the limiting factor.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ops import ExpDiffusion, uniformization
from transport import UpwindTaxis2D


@dataclass
class KSParams:
    Du: float = 1.0e-4
    Dv: float = 1.0e-3
    chi: float = 8.0e-3
    alpha: float = 1.0
    beta: float = 1.0


class KS2D:
    def __init__(self, Mx, My, Lx=1.0, Ly=1.0, params=KSParams(), device="cuda", dtype=torch.float64):
        self.Mx, self.My = Mx, My
        self.hx, self.hy = Lx / Mx, Ly / My
        self.p = params
        self.dev, self.dtype = torch.device(device), dtype
        self.ed = ExpDiffusion((Mx, My), (self.hx, self.hy), self.dev, dtype)
        self.tx = UpwindTaxis2D(self.hx, self.hy)
        self.x = (torch.arange(Mx, device=self.dev, dtype=dtype) + 0.5) * self.hx
        self.y = (torch.arange(My, device=self.dev, dtype=dtype) + 0.5) * self.hy

    def _t(self, val):
        return torch.as_tensor(val, device=self.dev, dtype=self.dtype)

    def initial(self, kind="bumps", seed=0):
        """Initial data. "bumps" is the training datum of the paper. The other kinds are used only to
        test a trained surrogate on initial conditions it never saw:
          bumps_rand   three to six Gaussian bumps with random centres and widths (seeded),
          subcritical  two small bumps, total mass below the aggregation threshold, so the density
                       spreads instead of aggregating,
          strong       the four bumps of the training datum with 1.5 times their amplitude,
          random       uniform density with a small random perturbation (linear instability).
        """
        X, Y = torch.meshgrid(self.x, self.y, indexing="ij")
        bump = lambda cx, cy, s: torch.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2))
        train_centres = [(0.30, 0.32, 0.075), (0.68, 0.30, 0.065),
                         (0.35, 0.70, 0.070), (0.72, 0.72, 0.060)]
        if kind == "bumps":
            u = 0.05 * torch.ones_like(X)
            for cx, cy, s in train_centres:
                u = u + bump(cx, cy, s)
        elif kind == "strong":
            u = 0.05 * torch.ones_like(X)
            for cx, cy, s in train_centres:
                u = u + 1.5 * bump(cx, cy, s)
        elif kind == "subcritical":
            u = 0.02 * torch.ones_like(X)
            for cx, cy, s in [(0.35, 0.40, 0.06), (0.65, 0.62, 0.05)]:
                u = u + 0.5 * bump(cx, cy, s)
        elif kind == "bumps_rand":
            g = torch.Generator(device="cpu").manual_seed(seed)
            nb = int(torch.randint(3, 7, (1,), generator=g))
            u = 0.05 * torch.ones_like(X)
            for _ in range(nb):
                c = 0.2 + 0.6 * torch.rand(2, generator=g)
                s = float(0.05 + 0.04 * torch.rand(1, generator=g))
                u = u + bump(float(c[0]), float(c[1]), s)
        else:
            g = torch.Generator(device="cpu").manual_seed(seed)
            u = 1.0 + 0.1 * torch.rand(self.Mx, self.My, generator=g).to(self.dev).to(self.dtype)
        v = torch.zeros_like(u)
        return u, v

    # ------------------------------------------------------------------ one Strang step
    def step(self, u, v, dt, p=None, scheme="dvtvd"):
        """Symmetric splitting D(dt/2) V(dt/2) T(dt) V(dt/2) D(dt/2).

        D  exponential diffusion of u          nonnegative, mass neutral
        T  uniformization of the taxis flux    nonnegative, mass exact
        V  exponential step of the v equation  nonnegative
        The palindromic arrangement makes the composition second order in dt; the older
        asymmetric arrangement (scheme="dtvd") is kept for the convergence comparison.
        """
        p = p or self.p
        Du, Dv, al, be = self._t(p.Du), self._t(p.Dv), self._t(p.alpha), self._t(p.beta)
        chi = self._t(p.chi)
        if scheme == "dtvd":
            u = self.ed.propagate(u, Du, dt / 2)
            ax, ay = self.tx.velocities(v, chi, chi)
            u = uniformization(lambda w: self.tx.apply(w, ax, ay), u, dt, self.tx.theta(ax, ay))
            v = self.ed.etd1(v, Dv, dt, al * u, decay=be)
            return self.ed.propagate(u, Du, dt / 2), v
        u = self.ed.propagate(u, Du, dt / 2)
        v = self.ed.etd1(v, Dv, dt / 2, al * u, decay=be)
        ax, ay = self.tx.velocities(v, chi, chi)
        u = uniformization(lambda w: self.tx.apply(w, ax, ay), u, dt, self.tx.theta(ax, ay))
        v = self.ed.etd1(v, Dv, dt / 2, al * u, decay=be)
        u = self.ed.propagate(u, Du, dt / 2)
        return u, v

    def run(self, u, v, dt, nsteps, save_every=1, scheme="dvtvd"):
        us, vs = [u.clone()], [v.clone()]
        for n in range(nsteps):
            u, v = self.step(u, v, dt, scheme=scheme)
            if (n + 1) % save_every == 0:
                us.append(u.clone())
                vs.append(v.clone())
        return torch.stack(us), torch.stack(vs)


# ---------------------------------------------------------------------- diagnostics
def invariants(us, hx, hy):
    m = us.reshape(us.shape[0], -1).sum(dim=1) * hx * hy
    return dict(min=float(us.min()),
                mass_drift=float(((m - m[0]).abs() / m[0]).max()),
                peak_growth=float(us[-1].max() / us[0].max()))


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--dt", type=float, default=2.0e-3)
    ap.add_argument("--nsteps", type=int, default=400)
    ap.add_argument("--chi", type=float, default=8.0e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conv", action="store_true", help="temporal self-convergence study")
    a = ap.parse_args()

    p = KSParams(chi=a.chi)
    s = KS2D(a.M, a.M, params=p, device=a.device)
    u0, v0 = s.initial()
    t0 = time.time()
    us, vs = s.run(u0, v0, a.dt, a.nsteps, save_every=max(1, a.nsteps // 20))
    torch.cuda.synchronize() if a.device == "cuda" else None
    inv = invariants(us, s.hx, s.hy)
    print(f"M={a.M} dt={a.dt} steps={a.nsteps} T={a.dt*a.nsteps:.3f} chi={a.chi}  "
          f"{time.time()-t0:.2f} s")
    print(f"  u range [{float(us.min()):.3e}, {float(us.max()):.3e}]  peak growth "
          f"{inv['peak_growth']:.2f}x  mass drift {inv['mass_drift']:.2e}")
    print(f"  v range [{float(vs.min()):.3e}, {float(vs.max()):.3e}]")

    if a.conv:
        import math
        for scheme in ("dtvd", "dvtvd"):
            ref, _ = s.run(u0, v0, a.dt / 16, a.nsteps * 16, save_every=a.nsteps * 16, scheme=scheme)
            errs = []
            print(f"  scheme {scheme}")
            for f in (1, 2, 4):
                uu, _ = s.run(u0, v0, a.dt / f, a.nsteps * f, save_every=a.nsteps * f, scheme=scheme)
                e = float((uu[-1] - ref[-1]).norm() / ref[-1].norm())
                errs.append(e)
                print(f"    dt={a.dt/f:.2e}  relerr={e:.3e}")
            for i in range(len(errs) - 1):
                print(f"    observed order {math.log2(errs[i]/errs[i+1]):.2f}")
