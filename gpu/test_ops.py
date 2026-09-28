"""Verification of the structure-preserving layers. Run: python test_ops.py [cpu|cuda]"""
import sys

import numpy as np
import torch
from scipy.fft import dct as sdct
from scipy.linalg import expm

from ops import ExpDiffusion, dct, eig_neumann, idct, mp22, uniformization
from transport import UpwindTaxis1D, UpwindTaxis2D

DEV = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
DT = torch.float64
torch.manual_seed(0)
np.random.seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


# ---------------------------------------------------------------- DCT
x = torch.randn(4, 16, device=DEV, dtype=DT)
check("dct matches scipy", rel(dct(x, -1).cpu().numpy(), sdct(x.cpu().numpy(), type=2, norm="ortho")) < 1e-12,
      f"rel={rel(dct(x,-1).cpu().numpy(), sdct(x.cpu().numpy(), type=2, norm='ortho')):.2e}")
check("idct inverts dct", rel(idct(dct(x, -1), -1).cpu().numpy(), x.cpu().numpy()) < 1e-12)

# ---------------------------------------------------------------- exponential diffusion, 1D
M, h, d, dt = 12, 1.0 / 12, 0.37, 0.11
L = np.zeros((M, M))
for i in range(M):
    for j, w in ((i - 1, 1.0), (i, -2.0), (i + 1, 1.0)):
        if 0 <= j < M:
            L[i, j] += w / h ** 2
        else:                      # Neumann mirror
            L[i, i] += w / h ** 2
E = expm(dt * d * L)
u0 = np.abs(np.random.rand(M)) + 0.1
ed = ExpDiffusion(M, h, DEV, DT)
got = ed.propagate(torch.tensor(u0, device=DEV, dtype=DT), torch.tensor(d, device=DEV, dtype=DT), dt).cpu().numpy()
check("expdiff 1D == dense expm", rel(got, E @ u0) < 1e-11, f"rel={rel(got, E @ u0):.2e}")
check("expdiff 1D nonnegative propagator", E.min() > -1e-14, f"min={E.min():.2e}")
check("expdiff 1D mass exact", abs(got.sum() - u0.sum()) / u0.sum() < 1e-13,
      f"drift={abs(got.sum()-u0.sum())/u0.sum():.2e}")

# ---------------------------------------------------------------- exponential diffusion, 2D
Mx, My = 8, 6
hx, hy = 1.0 / Mx, 1.0 / My
ed2 = ExpDiffusion((Mx, My), (hx, hy), DEV, DT)
U0 = torch.rand(Mx, My, device=DEV, dtype=DT)
U1 = ed2.propagate(U0, torch.tensor(0.5, device=DEV, dtype=DT), 0.03)
check("expdiff 2D mass exact", abs(float(U1.sum() - U0.sum())) / float(U0.sum()) < 1e-13)
check("expdiff 2D positivity", float(U1.min()) >= -1e-15, f"min={float(U1.min()):.2e}")

# huge step: exponential must relax to the mean, not blow up
Ubig = ed2.propagate(U0, torch.tensor(0.5, device=DEV, dtype=DT), 1e4)
check("expdiff 2D unconditional stability", float((Ubig - U0.mean()).abs().max()) < 1e-10)

# ---------------------------------------------------------------- ETD1 positivity + reference
src = torch.rand(Mx, My, device=DEV, dtype=DT)
V0 = torch.rand(Mx, My, device=DEV, dtype=DT)
V1 = ed2.etd1(V0, torch.tensor(0.2, device=DEV, dtype=DT), 0.05, src, decay=1.3)
Lx = np.zeros((Mx, Mx)); Ly = np.zeros((My, My))
for A_, n_, hh in ((Lx, Mx, hx), (Ly, My, hy)):
    for i in range(n_):
        for j, w in ((i - 1, 1.0), (i, -2.0), (i + 1, 1.0)):
            if 0 <= j < n_:
                A_[i, j] += w / hh ** 2
            else:
                A_[i, i] += w / hh ** 2
L2 = np.kron(Lx, np.eye(My)) + np.kron(np.eye(Mx), Ly)
A2 = 0.2 * L2 - 1.3 * np.eye(Mx * My)
ref = expm(0.05 * A2) @ V0.cpu().numpy().ravel() + \
      (np.linalg.solve(A2, (expm(0.05 * A2) - np.eye(Mx * My)))) @ src.cpu().numpy().ravel()
check("etd1 2D == dense reference", rel(V1.cpu().numpy().ravel(), ref) < 1e-10,
      f"rel={rel(V1.cpu().numpy().ravel(), ref):.2e}")
check("etd1 positivity", float(V1.min()) >= 0.0)

# ---------------------------------------------------------------- Patankar
u = torch.rand(50, device=DEV, dtype=DT)
out = mp22(u, lambda w: 3.0 * w * (1 - w).clamp(min=0), lambda w: 8.0 * w.clamp(min=0), 1e4)
check("mp22 positive at dt=1e4", float(out.min()) >= 0.0, f"min={float(out.min()):.2e}")

# ---------------------------------------------------------------- uniformization transport
tx = UpwindTaxis2D(hx, hy)
v = torch.rand(Mx, My, device=DEV, dtype=DT)
Sx = 2.0 * torch.rand(Mx - 1, My, device=DEV, dtype=DT) - 1.0   # arbitrary sign on purpose
Sy = 2.0 * torch.rand(Mx, My - 1, device=DEV, dtype=DT) - 1.0
ax, ay = tx.velocities(v, Sx, Sy)
N = Mx * My
Adense = np.zeros((N, N))
for j in range(N):
    e = torch.zeros(N, device=DEV, dtype=DT)
    e[j] = 1.0
    Adense[:, j] = tx.apply(e.reshape(Mx, My), ax, ay).cpu().numpy().ravel()
off = Adense - np.diag(np.diag(Adense))
check("taxis generator is Metzler", off.min() >= -1e-14, f"min offdiag={off.min():.2e}")
check("taxis generator has zero column sums", np.abs(Adense.sum(0)).max() < 1e-12,
      f"max|colsum|={np.abs(Adense.sum(0)).max():.2e}")
th = tx.theta(ax, ay)
check("theta bounds the diagonal", float(th) >= np.abs(np.diag(Adense)).max() - 1e-12,
      f"theta={float(th):.4f} maxdiag={np.abs(np.diag(Adense)).max():.4f}")

u0 = torch.rand(Mx, My, device=DEV, dtype=DT)
dt_t = 0.02
got = uniformization(lambda w: tx.apply(w, ax, ay), u0, dt_t, th)
ref = expm(dt_t * Adense) @ u0.cpu().numpy().ravel()
check("uniformization == dense expm", rel(got.cpu().numpy().ravel(), ref) < 1e-10,
      f"rel={rel(got.cpu().numpy().ravel(), ref):.2e}")

# the point of the layer: guarantees survive brutal truncation
crude = uniformization(lambda w: tx.apply(w, ax, ay), u0, dt_t, th, tol=1e-1, mu_max=1e9)
check("uniformization positive under crude truncation", float(crude.min()) >= 0.0,
      f"min={float(crude.min()):.2e}")
check("uniformization mass exact under crude truncation",
      abs(float(crude.sum() - u0.sum())) / float(u0.sum()) < 1e-13,
      f"drift={abs(float(crude.sum()-u0.sum()))/float(u0.sum()):.2e}")

# a velocity 1000x larger must not break either guarantee (no CFL condition anywhere)
axb, ayb = tx.velocities(v, 1000.0 * Sx, 1000.0 * Sy)
thb = tx.theta(axb, ayb)
fast = uniformization(lambda w: tx.apply(w, axb, ayb), u0, dt_t, thb)
check("uniformization positive at 1000x velocity", float(fast.min()) >= 0.0, f"min={float(fast.min()):.2e}")
check("uniformization mass exact at 1000x velocity",
      abs(float(fast.sum() - u0.sum())) / float(u0.sum()) < 1e-12)

# ---------------------------------------------------------------- 1D transport
t1 = UpwindTaxis1D(h)
v1 = torch.rand(M, device=DEV, dtype=DT)
S1 = torch.rand(M - 1, device=DEV, dtype=DT)
a1 = t1.velocities(v1, S1)
u1 = torch.rand(M, device=DEV, dtype=DT)
g1 = uniformization(lambda w: t1.apply(w, a1), u1, 0.05, t1.theta(a1))
check("1D taxis mass exact", abs(float(g1.sum() - u1.sum())) / float(u1.sum()) < 1e-13)
check("1D taxis positive", float(g1.min()) >= 0.0)

# ---------------------------------------------------------------- autograd through the layers
p = torch.tensor(0.3, device=DEV, dtype=DT, requires_grad=True)
loss = ed2.propagate(U0, p, 0.02).pow(2).sum()
loss.backward()
check("autograd through expdiff", p.grad is not None and torch.isfinite(p.grad))

Sp = torch.rand(Mx - 1, My, device=DEV, dtype=DT, requires_grad=True)
axg, ayg = tx.velocities(v, Sp, Sy)
lo = uniformization(lambda w: tx.apply(w, axg, ayg), u0, dt_t, tx.theta(axg, ayg).detach()).pow(2).sum()
lo.backward()
check("autograd through uniformization", Sp.grad is not None and bool(torch.isfinite(Sp.grad).all()))

print("\n", "ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
