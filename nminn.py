"""#12 NMINN building blocks (pure numpy/scipy).
   NMINN = Numerical-Method-Informed Neural Network: a learnable time-stepper whose forward pass
   FACTORS THROUGH a structure-preserving discrete operator, so its outputs inherit the operator's
   invariants (positivity, conservation) BY CONSTRUCTION, for ANY learned parameters.

   Two steppers implemented on 1D reaction-diffusion, Neumann BC:
   - NMINN stepper: exact Metzler-exp diffusion (DCT-II) o Modified-Patankar reaction with a small
     learnable, nonnegative reaction closure. Positive-by-construction; mass-conserving in the
     conservative-reaction case.
   - Baseline "neural finite-volume" (NFV) stepper: u^{n+1}_i = u^n_i + dt * f_theta(u_{i-1},u_i,u_{i+1}),
     a learned local 3-point stencil (shared small MLP). No positivity/conservation structure.
"""
import numpy as np
from scipy.fft import dct, idct

# ---------------- structure-preserving building blocks ----------------
def diffusion_eigs(M, h):
    k = np.arange(M)
    return -(2.0/h**2)*(1.0 - np.cos(np.pi*k/M))          # DCT-II Neumann Laplacian eigenvalues (<=0)

def E_flow(u, dcoef, dt, mu):
    """exact Metzler-exp diffusion flow: u <- idct(exp(dt*dcoef*mu) dct(u)). Entrywise >=0 for dt>=0."""
    return idct(np.exp(dt*dcoef*mu)*dct(u, type=2, norm='ortho'), type=2, norm='ortho')

def P22_scalar(u, g, lam, dt, eps=1e-13):
    """order-2 Modified-Patankar reaction step for u' = g(u) - lam(u)*u (g,lam >= 0).
       M-matrix (positive) for dt>0. g,lam are callables u-> array>=0."""
    g0, l0 = g(u), lam(u)
    u1 = (u + dt*g0)/(1.0 + dt*l0)
    g1, l1 = g(u1), lam(u1)
    gA = 0.5*(g0+g1); lA = 0.5*(l0*u + l1*u1)/(u1+eps)
    return (u + dt*gA)/(1.0 + dt*lA)

# ---------------- NMINN scalar stepper (Fisher-KPP-like) ----------------
class NMINNScalar:
    """Learnable positive-by-construction stepper: E(diffusion) o Patankar(reaction).
       Learnable params: log d (diffusion) and a nonnegative reaction closure
         production g(u)  = softplus(a) * u * relu(1-u)     (logistic-type growth)
         per-capita destruction lam(u) = softplus(b) * u    (quadratic sink r u^2 = (r u) u)
       Both >=0 for ANY a,b, so E>=0 and Patankar M-matrix => output >=0 by construction, any params.
       Covers logistic (a>0,b~0) and annihilation/decay (a~0,b>0) regimes."""
    def __init__(self, M, h, dt):
        self.M, self.h, self.dt, self.mu = M, h, dt, diffusion_eigs(M, h)
    @staticmethod
    def _sp(x): return np.log1p(np.exp(-abs(x))) + max(x, 0.0)   # softplus (stable)
    def unpack(self, theta): return theta[0], self._sp(theta[1]), self._sp(theta[2])  # logd, a, b
    def step(self, u, theta):
        logd, a, b = self.unpack(theta)
        d = np.exp(logd)
        u = E_flow(u, d, self.dt/2, self.mu)
        g   = lambda w: a*np.maximum(w, 0.0)*np.maximum(1.0-w, 0.0)
        lam = lambda w: b*np.maximum(w, 0.0)
        u = P22_scalar(u, g, lam, self.dt)
        u = E_flow(u, d, self.dt/2, self.mu)
        return u

# ---------------- NMINN 2-species conservative stepper ----------------
class NMINNConserv:
    """Two-species conservative exchange 1<->2 + diffusion. Mass u1+u2 conserved by construction:
       exact diffusion (mass-neutral) + Patankar reaction with d_12=p_21 (conservative transfer)."""
    def __init__(self, M, h, dt):
        self.M, self.h, self.dt, self.mu = M, h, dt, diffusion_eigs(M, h)
    @staticmethod
    def _sp(x): return np.log1p(np.exp(-abs(x))) + max(x, 0.0)
    def step(self, U, theta, eps=1e-13):
        logd1, logd2, k = theta[0], theta[1], self._sp(theta[2])
        d1, d2 = np.exp(logd1), np.exp(logd2)
        u1, u2 = U
        # half diffusion
        u1 = E_flow(u1, d1, self.dt/2, self.mu); u2 = E_flow(u2, d2, self.dt/2, self.mu)
        # conservative Patankar exchange: rate q = k*u1*u2/(1+u1), 1->2. destruction of u1, production of u2.
        q = k*u1*u2/(1.0+u1)
        dcap = q/(u1+eps)                                   # per-capita destruction of species 1
        u1n = u1/(1.0 + self.dt*dcap)                       # M-matrix solve (positive)
        u2n = u2 + self.dt*q*(u1n/(u1+eps))                 # gains exactly what u1 lost (conservative)
        u1, u2 = u1n, u2n
        # half diffusion
        u1 = E_flow(u1, d1, self.dt/2, self.mu); u2 = E_flow(u2, d2, self.dt/2, self.mu)
        return np.array([u1, u2])

# ---------------- baseline: neural finite-volume stepper (no structure) ----------------
class NFVScalar:
    """u^{n+1}_i = u^n_i + dt * f_theta(u_{i-1}, u_i, u_{i+1}); f_theta a shared 1-hidden-layer MLP
       (input 3, hidden H, tanh, output 1). Neumann padding. No positivity/conservation guarantee."""
    def __init__(self, M, h, dt, H=8, seed=0):
        self.M, self.h, self.dt, self.H = M, h, dt, H
        self.nparam = 3*H + H + H + 1                       # W1(H x3), b1(H), W2(H), b2(1)
        rng = np.random.default_rng(seed)
        self.theta0 = 0.1*rng.standard_normal(self.nparam)
    def unpack(self, theta):
        H = self.H; i = 0
        W1 = theta[i:i+3*H].reshape(H, 3); i += 3*H
        b1 = theta[i:i+H]; i += H
        W2 = theta[i:i+H]; i += H
        b2 = theta[i]
        return W1, b1, W2, b2
    def step(self, u, theta):
        W1, b1, W2, b2 = self.unpack(theta)
        um = np.concatenate([u[:1], u[:-1]])               # u_{i-1} (Neumann)
        up = np.concatenate([u[1:], u[-1:]])               # u_{i+1}
        X = np.stack([um, u, up], axis=1)                  # (M,3)
        Hh = np.tanh(X @ W1.T + b1)                        # (M,H)
        f = Hh @ W2 + b2                                   # (M,)
        return u + self.dt*f

class NFVConserv:
    """Two-channel neural stepper: U^{n+1}_c,i = U^n_c,i + dt*f_theta(stencil of both channels).
       Shared MLP input = [u1_{i-1..i+1}, u2_{i-1..i+1}] (6), output 2. No mass/positivity structure."""
    def __init__(self, M, h, dt, H=10, seed=0):
        self.M, self.h, self.dt, self.H = M, h, dt, H
        self.nparam = 6*H + H + 2*H + 2
        rng = np.random.default_rng(seed)
        self.theta0 = 0.1*rng.standard_normal(self.nparam)
    def unpack(self, theta):
        H = self.H; i = 0
        W1 = theta[i:i+6*H].reshape(H, 6); i += 6*H
        b1 = theta[i:i+H]; i += H
        W2 = theta[i:i+2*H].reshape(2, H); i += 2*H
        b2 = theta[i:i+2]
        return W1, b1, W2, b2
    def step(self, U, theta):
        W1, b1, W2, b2 = self.unpack(theta)
        feats = []
        for c in range(2):
            u = U[c]
            feats += [np.concatenate([u[:1], u[:-1]]), u, np.concatenate([u[1:], u[-1:]])]
        X = np.stack(feats, axis=1)                        # (M,6)
        Hh = np.tanh(X @ W1.T + b1)                        # (M,H)
        f = Hh @ W2.T + b2                                 # (M,2)
        return U + self.dt*f.T
