"""Learned macro-steppers for the 2D Keller-Segel problem.

All three map (u^n, v^n) at time t to (u^{n+1}, v^{n+1}) at time t + Dt with Dt much larger than
the timestep of the reference integrator. They differ only in how the discretization enters:

  NMINN2D   the forward pass factors through the same three structure-preserving operators used by
            the reference solver, with learned coefficients and a learned nonnegative chemotactic
            sensitivity network. Positivity of u and v and exact conservation of the mass of u hold
            for every value of the weights.
  CNNStepper / FNOStepper   unstructured residual steppers with more parameters and no guarantee.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ops import ExpDiffusion, uniformization
from transport import UpwindTaxis2D


class PointwiseMLP(nn.Module):
    """Applied at every cell to the local state, so it is resolution independent."""

    def __init__(self, din=2, width=32, depth=2, dout=1):
        super().__init__()
        dims = [din] + [width] * depth + [dout]
        self.lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        for m in self.lin:
            nn.init.xavier_normal_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)

    def forward(self, z):
        for m in self.lin[:-1]:
            z = torch.tanh(m(z))
        return self.lin[-1](z)


class NMINN2D(nn.Module):
    """Method-informed macro-stepper: n_sub symmetric splitting substeps per macro step."""

    def __init__(self, M, h, Dt, n_sub=4, u_scale=1.0, v_scale=1.0, width=32,
                 device="cuda", dtype=torch.float64):
        super().__init__()
        self.M, self.h, self.Dt, self.n_sub = M, h, Dt, n_sub
        self.dev, self.dtype = torch.device(device), dtype
        self.ed = ExpDiffusion((M, M), (h, h), self.dev, dtype)
        self.tx = UpwindTaxis2D(h, h)
        self.u_scale, self.v_scale = u_scale, v_scale
        # coefficients, parametrized through logs so they stay positive for any weight value
        self.logc = nn.Parameter(torch.tensor(
            [math.log(3e-4), math.log(3e-3), math.log(0.5), math.log(0.5)],
            device=self.dev, dtype=dtype))          # log Du, log Dv, log alpha, log beta
        self.log_chi = nn.Parameter(torch.tensor(math.log(5e-3), device=self.dev, dtype=dtype))
        self.sens = PointwiseMLP(2, width, 2, 1).to(self.dev).to(dtype)

    # -------------------------------------------------------------- learned closures
    def sensitivity(self, u, v):
        """Nonnegative chemotactic sensitivity at cell centres, chi * softplus(net(u,v)).

        The structural guarantees do not rely on this being nonnegative; it is kept nonnegative
        only so that the learned closure remains physically readable.
        """
        z = torch.stack([u / self.u_scale, v / self.v_scale], dim=-1)
        s = nn.functional.softplus(self.sens(z)[..., 0] + 0.5413)   # softplus(0.5413) = 1
        return torch.exp(self.log_chi) * s

    def _faces(self, S):
        return 0.5 * (S[1:, :] + S[:-1, :]), 0.5 * (S[:, 1:] + S[:, :-1])

    # -------------------------------------------------------------- one macro step
    def forward(self, u, v):
        Du, Dv, al, be = torch.exp(self.logc[0]), torch.exp(self.logc[1]), \
            torch.exp(self.logc[2]), torch.exp(self.logc[3])
        dt = self.Dt / self.n_sub
        for _ in range(self.n_sub):
            u = self.ed.propagate(u, Du, dt / 2)
            v = self.ed.etd1(v, Dv, dt / 2, al * u, decay=be)
            S = self.sensitivity(u, v)
            Sx, Sy = self._faces(S)
            ax, ay = self.tx.velocities(v, Sx, Sy)
            th = self.tx.theta(ax, ay).detach()
            u = uniformization(lambda w: self.tx.apply(w, ax, ay), u, dt, th)
            v = self.ed.etd1(v, Dv, dt / 2, al * u, decay=be)
            u = self.ed.propagate(u, Du, dt / 2)
        return u, v

    def coefficients(self):
        c = torch.exp(self.logc.detach()).tolist()
        return dict(Du=c[0], Dv=c[1], alpha=c[2], beta=c[3], chi=float(torch.exp(self.log_chi.detach())))


# ---------------------------------------------------------------------- unstructured baselines
class CNNStepper(nn.Module):
    """u^{n+1} = u^n + Dt * CNN(u^n, v^n), replicate padding to mimic the no-flux boundary."""

    def __init__(self, Dt, ch=32, depth=4, u_scale=1.0, v_scale=1.0, device="cuda", dtype=torch.float64):
        super().__init__()
        self.Dt, self.u_scale, self.v_scale = Dt, u_scale, v_scale
        layers = []
        cin = 2
        for i in range(depth):
            layers.append(nn.Conv2d(cin, ch, 3, padding=1, padding_mode="replicate"))
            layers.append(nn.Tanh())
            cin = ch
        layers.append(nn.Conv2d(ch, 2, 3, padding=1, padding_mode="replicate"))
        self.net = nn.Sequential(*layers).to(device).to(dtype)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, u, v):
        z = torch.stack([u / self.u_scale, v / self.v_scale], dim=0).unsqueeze(0)
        f = self.net(z)[0]
        return u + self.Dt * f[0] * self.u_scale, v + self.Dt * f[1] * self.v_scale


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, modes, dtype):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (cin * cout)
        self.w = nn.Parameter(scale * torch.randn(cin, cout, modes, modes, 2, dtype=dtype))

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        m = self.modes
        w = torch.view_as_complex(self.w)
        out = torch.zeros(B, w.shape[1], H, W // 2 + 1, dtype=xf.dtype, device=x.device)
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], w)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], w)
        return torch.fft.irfft2(out, s=(H, W), norm="ortho")


class FNOStepper(nn.Module):
    """Two spectral layers, the standard neural operator surrogate, again with no guarantee."""

    def __init__(self, Dt, width=24, modes=12, u_scale=1.0, v_scale=1.0,
                 device="cuda", dtype=torch.float64):
        super().__init__()
        self.Dt, self.u_scale, self.v_scale = Dt, u_scale, v_scale
        self.lift = nn.Conv2d(2, width, 1)
        self.s1, self.s2 = SpectralConv2d(width, width, modes, dtype), SpectralConv2d(width, width, modes, dtype)
        self.w1, self.w2 = nn.Conv2d(width, width, 1), nn.Conv2d(width, width, 1)
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.Tanh(), nn.Conv2d(width, 2, 1))
        self.to(device).to(dtype)
        with torch.no_grad():
            self.proj[-1].weight.mul_(0.01)
            self.proj[-1].bias.zero_()

    def forward(self, u, v):
        z = torch.stack([u / self.u_scale, v / self.v_scale], dim=0).unsqueeze(0)
        z = self.lift(z)
        z = torch.tanh(self.s1(z) + self.w1(z))
        z = torch.tanh(self.s2(z) + self.w2(z))
        f = self.proj(z)[0]
        return u + self.Dt * f[0] * self.u_scale, v + self.Dt * f[1] * self.v_scale


# ---------------------------------------------------------- hard-constrained neural baselines (revision)
def _conv_stack(cin, cout, ch, depth, device, dtype):
    layers = []
    c = cin
    for _ in range(depth):
        layers.append(nn.Conv2d(c, ch, 3, padding=1, padding_mode="replicate"))
        layers.append(nn.Tanh())
        c = ch
    layers.append(nn.Conv2d(ch, cout, 3, padding=1, padding_mode="replicate"))
    net = nn.Sequential(*layers).to(device).to(dtype)
    with torch.no_grad():
        net[-1].weight.mul_(0.01)
        net[-1].bias.zero_()
    return net


class CNNFluxStepper(nn.Module):
    """Conservative flux-form stepper, the direct neural analogue of a conservation layer.

    The same convolutional backbone as CNNStepper proposes a cell-centred flux potential pair
    (Gx, Gy) for u and a residual for v. Face fluxes are the averages of the two neighbouring cells
    on interior faces and zero on boundary faces, and u^{n+1} = u^n - Dt div_h F. The discrete
    divergence telescopes, so the mass of u is exact for every value of the weights, at every step.
    Nothing constrains the sign of u. This is the hard-constraint route of Beucler et al. and of the
    neural conservation laws literature, applied to the stepper.
    """

    def __init__(self, Dt, h, ch=32, depth=4, u_scale=1.0, v_scale=1.0, device="cuda", dtype=torch.float64):
        super().__init__()
        self.Dt, self.h, self.u_scale, self.v_scale = Dt, h, u_scale, v_scale
        self.net = _conv_stack(2, 3, ch, depth, device, dtype)

    def forward(self, u, v):
        z = torch.stack([u / self.u_scale, v / self.v_scale], dim=0).unsqueeze(0)
        f = self.net(z)[0]
        Gx, Gy = f[0] * self.u_scale * self.h, f[1] * self.u_scale * self.h
        Fx = 0.5 * (Gx[1:, :] + Gx[:-1, :])
        Fy = 0.5 * (Gy[:, 1:] + Gy[:, :-1])
        pad = torch.nn.functional.pad
        div = (pad(Fx, (0, 0, 0, 1)) - pad(Fx, (0, 0, 1, 0))) / self.h \
            + (pad(Fy, (0, 1)) - pad(Fy, (1, 0))) / self.h
        return u - self.Dt * div, v + self.Dt * f[2] * self.v_scale


class CNNPosStepper(nn.Module):
    """Positive-output stepper, u^{n+1} = u^n exp(Dt f_u), the stepper analogue of a nonnegative output
    activation: u stays positive for every weight, and nothing constrains its total. With project=True
    every output slice is rescaled to the initial mass, which secures conservation at the output level
    but requires the modeller to know that the mass is conserved and to know its value.
    """

    def __init__(self, Dt, ch=32, depth=4, u_scale=1.0, v_scale=1.0, project=False, mass0=None,
                 device="cuda", dtype=torch.float64):
        super().__init__()
        self.Dt, self.u_scale, self.v_scale = Dt, u_scale, v_scale
        self.project, self.mass0 = project, mass0
        if project and mass0 is None:
            raise ValueError("the projected variant needs the conserved mass")
        self.net = _conv_stack(2, 2, ch, depth, device, dtype)

    def forward(self, u, v):
        z = torch.stack([u / self.u_scale, v / self.v_scale], dim=0).unsqueeze(0)
        f = self.net(z)[0]
        un = u * torch.exp(self.Dt * f[0])
        if self.project:
            un = un * (self.mass0 / un.sum())
        return un, v + self.Dt * f[1] * self.v_scale


# short names used on the command line and in file names, long names used in tables
MODEL_NAMES = {"NMINN": "NMINN", "CNN": "CNN stepper", "FNO": "FNO stepper",
               "CNNFLUX": "CNN flux-form", "CNNPOS": "CNN positive", "CNNPROJ": "CNN positive projected"}


def build_model(short, M, h, Dt, n_sub, u_scale, v_scale, mass0, device, dtype):
    if short == "NMINN":
        return NMINN2D(M, h, Dt, n_sub=n_sub, u_scale=u_scale, v_scale=v_scale, device=device, dtype=dtype)
    if short == "CNN":
        return CNNStepper(Dt, u_scale=u_scale, v_scale=v_scale, device=device, dtype=dtype)
    if short == "FNO":
        return FNOStepper(Dt, u_scale=u_scale, v_scale=v_scale, device=device, dtype=dtype)
    if short == "CNNFLUX":
        return CNNFluxStepper(Dt, h, u_scale=u_scale, v_scale=v_scale, device=device, dtype=dtype)
    if short == "CNNPOS":
        return CNNPosStepper(Dt, u_scale=u_scale, v_scale=v_scale, device=device, dtype=dtype)
    if short == "CNNPROJ":
        return CNNPosStepper(Dt, u_scale=u_scale, v_scale=v_scale, project=True, mass0=mass0,
                             device=device, dtype=dtype)
    raise KeyError(short)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
