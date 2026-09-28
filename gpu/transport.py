"""Conservative upwind transport generator for a taxis flux, and its structural properties.

For a velocity field a = S grad v (chemotaxis: cells climb the gradient of the signal v) the
transport part of the model is

    u_t + div(a u) = 0,   a . n = 0 on the boundary.

The cell centred upwind discretization in flux form,

    (A u)_i = -(F_{i+1/2} - F_{i-1/2}) / h,   F_{i+1/2} = a^+_{i+1/2} u_i + a^-_{i+1/2} u_{i+1},

has nonnegative off diagonal entries (Metzler) and exactly zero column sums (what leaves a cell
enters a neighbour, and boundary faces carry no flux). Those two properties are what the
uniformization layer converts into positivity and exact mass conservation, for ANY velocity
field, hence for any network that produces S.

Only the action u -> A u is needed, so no matrix is ever assembled.
"""
from __future__ import annotations

import torch


def _faces_x(v, h):
    """Centred difference of v across interior x faces: shape (Mx-1, My)."""
    return (v[1:, :] - v[:-1, :]) / h


def _faces_y(v, h):
    return (v[:, 1:] - v[:, :-1]) / h


class UpwindTaxis2D:
    """Generator of  -div(S grad(v) u)  with no flux boundaries, cell centred upwind.

    S is given at faces (broadcastable to the face shapes) and may be any real field: the
    Metzler and zero column sum properties do not depend on its sign or magnitude.
    """

    def __init__(self, hx, hy):
        self.hx, self.hy = hx, hy

    def velocities(self, v, Sx, Sy):
        ax = Sx * _faces_x(v, self.hx)          # (Mx-1, My)
        ay = Sy * _faces_y(v, self.hy)          # (Mx, My-1)
        return ax, ay

    def theta(self, ax, ay):
        """Uniformization rate: max over cells of the total outflow rate = max_j |A_jj|."""
        axp, axm = ax.clamp(min=0.0), (-ax).clamp(min=0.0)
        ayp, aym = ay.clamp(min=0.0), (-ay).clamp(min=0.0)
        Mx, My = ax.shape[0] + 1, ax.shape[1]
        out = torch.zeros((Mx, My), device=ax.device, dtype=ax.dtype)
        out[:-1, :] += axp / self.hx            # outflow of cell i through face i+1/2
        out[1:, :] += axm / self.hx             # outflow of cell i+1 through face i+1/2
        out[:, :-1] += ayp / self.hy
        out[:, 1:] += aym / self.hy
        return out.max()

    def apply(self, u, ax, ay):
        axp, axm = ax.clamp(min=0.0), ax.clamp(max=0.0)
        ayp, aym = ay.clamp(min=0.0), ay.clamp(max=0.0)
        Fx = axp * u[:-1, :] + axm * u[1:, :]   # flux on interior x faces
        Fy = ayp * u[:, :-1] + aym * u[:, 1:]
        out = torch.zeros_like(u)
        # -(F_{i+1/2} - F_{i-1/2})/hx  with zero flux on the boundary faces
        out = out - torch.nn.functional.pad(Fx, (0, 0, 0, 1)) / self.hx
        out = out + torch.nn.functional.pad(Fx, (0, 0, 1, 0)) / self.hx
        out = out - torch.nn.functional.pad(Fy, (0, 1)) / self.hy
        out = out + torch.nn.functional.pad(Fy, (1, 0)) / self.hy
        return out


class UpwindTaxis1D:
    def __init__(self, h):
        self.h = h

    def velocities(self, v, S):
        return S * (v[1:] - v[:-1]) / self.h

    def theta(self, a):
        ap, am = a.clamp(min=0.0), (-a).clamp(min=0.0)
        out = torch.zeros(a.shape[0] + 1, device=a.device, dtype=a.dtype)
        out[:-1] += ap / self.h
        out[1:] += am / self.h
        return out.max()

    def apply(self, u, a):
        ap, am = a.clamp(min=0.0), a.clamp(max=0.0)
        F = ap * u[:-1] + am * u[1:]
        out = torch.zeros_like(u)
        out = out - torch.nn.functional.pad(F, (0, 1)) / self.h
        out = out + torch.nn.functional.pad(F, (1, 0)) / self.h
        return out
