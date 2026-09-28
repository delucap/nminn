"""Structure-preserving layers for method-informed networks (torch, CPU/CUDA).

Every layer here is a *parametrized* map whose structural guarantee holds for ANY value of the
parameters, so a network that factors through it inherits the guarantee by construction.

Layers
------
expdiff    exact exponential of a Neumann finite-difference Laplacian, diagonalized by the
           orthonormal DCT-II. Metzler generator with zero column sums => entrywise nonnegative
           propagator, exact mass neutrality. O(M^d log M) matrix free.
etd1       exponential time differencing for  w' = L w + s  with L Metzler, s >= 0. Uses the
           phi_1 function of the same DCT-diagonal operator; both e^{tL} and phi_1(tL)t are
           entrywise nonnegative, hence w >= 0 for any nonnegative state and source.
uniform    uniformization (Jensen) exponential of a Metzler generator A with zero column sums,
           given only its action u -> A u. Writing A = -theta I + B with B >= 0 entrywise and
           column sums theta, e^{tA} = e^{-theta t} sum_k (theta t)^k / k! P^k with P = B/theta
           column stochastic. Every term is nonnegative and column stochastic, therefore the
           TRUNCATED series is nonnegative, and after division by the scalar partial Poisson
           weight it is exactly mass conserving. Guarantees survive truncation; this is what
           makes the transport layer usable inside a network.
mp22       second order modified-Patankar solve of a production-destruction reaction, positive
           for any timestep.
"""
from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------------------------
# orthonormal DCT-II / DCT-III along one dimension, via FFT, autograd friendly
# --------------------------------------------------------------------------------------
def dct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x.transpose(dim, -1)
    N = x.shape[-1]
    v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    w = torch.exp(torch.complex(torch.zeros_like(k), -math.pi * k / (2 * N)))
    out = 2.0 * (V * w).real
    scale = torch.full((N,), math.sqrt(1.0 / (2 * N)), device=x.device, dtype=x.dtype)
    scale = torch.cat([torch.full((1,), math.sqrt(1.0 / (4 * N)), device=x.device, dtype=x.dtype),
                       scale[1:]])
    return (out * scale).transpose(dim, -1)


def idct(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    X = X.transpose(dim, -1)
    N = X.shape[-1]
    k = torch.arange(N, device=X.device, dtype=X.dtype)
    scale = torch.full((N,), math.sqrt(2.0 / N), device=X.device, dtype=X.dtype)
    scale = torch.cat([torch.full((1,), math.sqrt(1.0 / N), device=X.device, dtype=X.dtype),
                       scale[1:]])
    g = X * scale
    G = torch.complex(g, torch.zeros_like(g)) * torch.exp(
        torch.complex(torch.zeros_like(k), math.pi * k / (2 * N)))
    Gh = torch.zeros(X.shape[:-1] + (2 * N,), dtype=G.dtype, device=X.device)
    Gh[..., :N] = G
    x = (torch.fft.ifft(Gh, dim=-1) * (2 * N))[..., :N].real
    return x.transpose(dim, -1)


def eig_neumann(M: int, h: float, device=None, dtype=torch.float64) -> torch.Tensor:
    """Eigenvalues of the cell-centred Neumann FD Laplacian: -(4/h^2) sin^2(k pi / 2M) <= 0."""
    k = torch.arange(M, device=device, dtype=dtype)
    return -(4.0 / h ** 2) * torch.sin(k * math.pi / (2 * M)) ** 2


# --------------------------------------------------------------------------------------
# exponential diffusion layer
# --------------------------------------------------------------------------------------
class ExpDiffusion:
    """exp(dt * d * Laplacian) applied to fields of shape (..., Mx) or (..., Mx, My).

    Nonnegative propagator (Metzler generator) with lam_0 = 0, so the discrete mass
    sum(U) * cell volume is preserved to round off for any dt and any d >= 0.
    """

    def __init__(self, shape, h, device=None, dtype=torch.float64):
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(h, float):
            h = (h,) * len(shape)
        self.ndim = len(shape)
        self.lam = [eig_neumann(M, hh, device, dtype) for M, hh in zip(shape, h)]
        if self.ndim == 1:
            self.L = self.lam[0]
        elif self.ndim == 2:
            self.L = self.lam[0][:, None] + self.lam[1][None, :]
        else:
            raise ValueError("only 1D and 2D")

    def _fwd(self, U):
        if self.ndim == 1:
            return dct(U, -1)
        return dct(dct(U, -1), -2)

    def _inv(self, F):
        if self.ndim == 1:
            return idct(F, -1)
        return idct(idct(F, -2), -1)

    def propagate(self, U, d, dt):
        """d may be a scalar tensor (learnable). Returns exp(dt d L) U."""
        return self._inv(torch.exp(dt * d * self.L) * self._fwd(U))

    def etd1(self, U, d, dt, source, decay=0.0):
        """One exponential step of  U' = (d L - decay) U + source, source frozen and >= 0.

        U <- e^{dt A} U + dt phi_1(dt A) source,  A = d L - decay.
        Both operators are entrywise nonnegative (A is Metzler), so U stays nonnegative.
        """
        A = d * self.L - decay
        z = dt * A
        e = torch.exp(z)
        # phi_1(z) = (e^z - 1)/z, stable near z = 0
        small = z.abs() < 1e-8
        phi = torch.where(small, 1.0 + z / 2.0 + z * z / 6.0, (e - 1.0) / torch.where(small, torch.ones_like(z), z))
        return self._inv(e * self._fwd(U) + dt * phi * self._fwd(source))


# --------------------------------------------------------------------------------------
# modified-Patankar reaction layer
# --------------------------------------------------------------------------------------
def mp22(u, prod, dest_rate, dt, eps=1e-13):
    """Second order modified-Patankar step for u' = p(u) - l(u) u with p, l >= 0.

    prod and dest_rate are callables u -> tensor >= 0. Positive for any dt > 0 because each
    stage is a diagonal M-matrix solve with a nonnegative right hand side.
    """
    p0, l0 = prod(u), dest_rate(u)
    u1 = (u + dt * p0) / (1.0 + dt * l0)
    p1, l1 = prod(u1), dest_rate(u1)
    pA = 0.5 * (p0 + p1)
    lA = 0.5 * (l0 * u + l1 * u1) / (u1 + eps)
    return (u + dt * pA) / (1.0 + dt * lA)


# --------------------------------------------------------------------------------------
# uniformization transport layer
# --------------------------------------------------------------------------------------
def poisson_terms(mu: float, tol: float = 1e-14, kmax: int = 512) -> int:
    """Smallest K with  sum_{k>K} e^{-mu} mu^k / k! < tol."""
    if mu <= 0:
        return 0
    logw = -mu
    cum = math.exp(logw)
    k = 0
    while 1.0 - cum > tol and k < kmax:
        k += 1
        logw += math.log(mu) - math.log(k)
        cum += math.exp(logw)
    return k


def uniformization(apply_A, u, dt, theta, tol=1e-14, mu_max=6.0, sum_dims=None,
                   theta_grad=False, return_info=False):
    """exp(dt A) u for a Metzler generator A with zero column sums, given u -> A u.

    theta must satisfy theta >= max_j |A_jj|. The series is truncated but the result stays
    entrywise nonnegative (all terms are nonnegative) and, after the scalar renormalization by
    the partial Poisson weight, conserves sum(u) exactly.

    Substepping keeps the Poisson parameter theta*dt_sub <= mu_max so the series is short.

    What is differentiated. The truncation order K and the number of substeps nsub are integers
    fixed from the numerical value of theta, so the implemented map is piecewise analytic in the
    velocity field, and on each piece it is the finite renormalized series at fixed (K, nsub).
    With theta_grad=False (the training default) theta enters P = I + A/theta and the Poisson
    weights as a frozen constant, so autograd returns the partial derivative at fixed theta; the
    omitted term is bounded by 2 dt pi_K(theta dt) / (1 - rho_K)^2 in the induced 1-norm
    (Proposition on the theta-sensitivity of the truncated layer), which is below round-off at
    the tolerance used. With theta_grad=True and a tensor theta, autograd returns the total
    derivative of the implemented map on the current piece.
    """
    theta_f = float(theta.detach()) if torch.is_tensor(theta) else float(theta)
    if theta_f <= 0.0 or dt == 0.0:
        return (u, dict(nsub=0, K=0, mu=0.0)) if return_info else u
    total = theta_f * dt
    nsub = max(1, int(math.ceil(total / mu_max)))
    dts = dt / nsub
    mu_f = theta_f * dts
    K = poisson_terms(mu_f, tol)
    th = theta if (theta_grad and torch.is_tensor(theta)) else theta_f
    mu = th * dts
    for _ in range(nsub):
        term = u
        acc = u  # k = 0 term, weight 1
        w = 1.0
        wsum = 1.0
        for k in range(1, K + 1):
            # P u = (A u)/theta + u
            term = apply_A(term) / th + term
            w = w * mu / k
            acc = acc + w * term
            wsum = wsum + w
        u = acc / wsum  # exact renormalization: preserves nonnegativity and total mass
    if return_info:
        return u, dict(nsub=nsub, K=K, mu=mu_f)
    return u


def poisson_pmf(mu: float, k: int) -> float:
    """e^{-mu} mu^k / k!, computed in logs."""
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))


def poisson_tail(mu: float, K: int) -> float:
    """sum_{k>K} e^{-mu} mu^k / k!  (rho_K in the manuscript)."""
    if mu <= 0:
        return 0.0
    cum = sum(poisson_pmf(mu, k) for k in range(K + 1))
    return max(0.0, 1.0 - cum)
