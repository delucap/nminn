"""#12 NMINN experiments: learned time-steppers on reaction-diffusion.
   A) scalar Fisher-KPP -> positivity + long-rollout stability (NMINN vs baseline neural stepper).
   B) 2-species conservative exchange -> mass conservation (NMINN vs baseline).
   Ground truth from a fine reference integrator; both steppers trained by one-step MSE (L-BFGS).
   HONEST: we build both, train identically, roll out, and report what actually happens."""
import os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.dirname(__file__))
from nminn import (NMINNScalar, NFVScalar, NMINNConserv, NFVConserv,
                   diffusion_eigs, E_flow)

np.random.seed(0)
relL2 = lambda a, b: np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-30)

# ============================ Problem A: Fisher-KPP traveling front (positivity killer) ============================
def groundtruth_A(M, h, dt, nsteps, d_true=0.05, r_true=6.0):
    """u_t = d u_xx + r u(1-u)  (Fisher-KPP): a front travels into the u=0 region. Ahead of the front
       u->0 with a steep gradient -- exactly where an unconstrained learned stepper undershoots negative.
       Ground truth by a fine positive integrator (exact-exp diffusion + positive logistic step)."""
    x = (np.arange(M)+0.5)*h
    u = 0.5*(1.0 - np.tanh((x-0.25)/0.04))                    # smoothed step: ~1 left, ~0 right
    mu = diffusion_eigs(M, h)
    traj = [u.copy()]; sub = 20; ddt = dt/sub
    for _ in range(nsteps):
        for _ in range(sub):
            u = E_flow(u, d_true, ddt/2, mu)
            u = u + ddt*r_true*u*(1.0-u)                      # fine explicit logistic (accurate, positive)
            u = E_flow(u, d_true, ddt/2, mu)
        traj.append(u.copy())
    return x, np.array(traj)

def train_stepper(step_fn, theta0, traj, K=5, iters=400):
    """K-step rollout MSE over the trajectory (identifies dynamics far better than one-step); L-BFGS."""
    n = len(traj)-1
    starts = list(range(0, n-K+1))
    def loss(theta):
        err = 0.0
        for s in starts:
            u = traj[s].copy(); e = 0.0
            for j in range(1, K+1):
                u = step_fn(u, theta)
                if not np.all(np.isfinite(u)): return 1e6
                e += np.mean((u-traj[s+j])**2)
            err += e/K
        return err/len(starts)
    res = minimize(loss, theta0, method='L-BFGS-B',
                   options={'maxiter': iters, 'maxfun': iters*60})
    return res.x, res.fun

def rollout(step_fn, theta, u0, nsteps):
    u = u0.copy(); traj = [u.copy()]; mn = u.min()
    for _ in range(nsteps):
        u = step_fn(u, theta)
        if not np.all(np.isfinite(u)): return np.array(traj), -np.inf, True
        mn = min(mn, u.min()); traj.append(u.copy())
    return np.array(traj), mn, False

def run_A(noise=0.02):
    M, L = 64, 1.0; h = L/M; dt = 0.004; ntrain = 40; ntest = 120
    x, traj = groundtruth_A(M, h, dt, ntrain+ntest)
    rng = np.random.default_rng(7)
    train = traj[:ntrain+1] + noise*traj.max()*rng.standard_normal(traj[:ntrain+1].shape)  # noisy data
    nm = NMINNScalar(M, h, dt); nf = NFVScalar(M, h, dt, H=8, seed=1)
    th_nm, l_nm = train_stepper(nm.step, np.array([np.log(0.05), 1.8, -2.0]), train)
    th_nf, l_nf = train_stepper(nf.step, nf.theta0, train)
    # long rollout from the true IC over the FULL horizon (extrapolation beyond training)
    r_nm, mn_nm, blew_nm = rollout(nm.step, th_nm, traj[0], ntrain+ntest)
    r_nf, mn_nf, blew_nf = rollout(nf.step, th_nf, traj[0], ntrain+ntest)
    # rollout errors vs ground truth (only over the length actually produced)
    def rerr(r):
        n = min(len(r), len(traj))
        return relL2(r[:n], traj[:n]) if np.all(np.isfinite(r[:n])) else np.inf
    print("=== Problem A: Fisher-KPP traveling front (u_t = d u_xx + r u(1-u)), learned steppers ===")
    print(f"  grid M={M}, dt={dt}, train steps={ntrain}, test/rollout steps={ntrain+ntest}")
    print(f"  NMINN     : train MSE={l_nm:.2e}  rollout relL2={rerr(r_nm):.2e}  min u over rollout={mn_nm:.3e}  blew_up={blew_nm}")
    print(f"  baseline  : train MSE={l_nf:.2e}  rollout relL2={rerr(r_nf):.2e}  min u over rollout={mn_nf:.3e}  blew_up={blew_nf}")
    print(f"  learned NMINN diffusion d={np.exp(nm.unpack(th_nm)[0]):.4f} (true 0.05), logistic rate a={nm.unpack(th_nm)[1]:.3f} (true r=6), b={nm.unpack(th_nm)[2]:.3f}")
    np.savez(os.path.join(os.path.dirname(__file__), "resA.npz"),
             x=x, traj=traj, r_nm=r_nm, r_nf=r_nf, dt=dt, ntrain=ntrain)
    return dict(mn_nm=mn_nm, mn_nf=mn_nf, err_nm=rerr(r_nm), err_nf=rerr(r_nf),
                blew_nf=blew_nf, blew_nm=blew_nm, l_nm=l_nm, l_nf=l_nf)

# ============================ Problem B: 2-species conservative ============================
def groundtruth_B(M, h, dt, nsteps, d1=0.03, d2=0.08, k=6.0):
    x = (np.arange(M)+0.5)*h
    u1 = 0.8*np.exp(-((x-0.35)/0.09)**2)                      # bumps on ~zero background
    u2 = 0.8*np.exp(-((x-0.65)/0.09)**2)
    U = np.array([u1, u2]); mu = diffusion_eigs(M, h)
    traj = [U.copy()]; sub = 20; ddt = dt/sub
    for _ in range(nsteps):
        for _ in range(sub):
            U0 = U.copy()
            U[0] = E_flow(U[0], d1, ddt/2, mu); U[1] = E_flow(U[1], d2, ddt/2, mu)
            q = k*U[0]*U[1]/(1+U[0])                          # conservative 1->2 exchange
            U[0] = U[0] - ddt*q; U[1] = U[1] + ddt*q
            U[0] = E_flow(U[0], d1, ddt/2, mu); U[1] = E_flow(U[1], d2, ddt/2, mu)
        traj.append(U.copy())
    return x, np.array(traj)

def train_stepper_B(step_fn, theta0, traj, K=5, iters=400):
    n = len(traj)-1; starts = list(range(0, n-K+1))
    def loss(theta):
        err = 0.0
        for s in starts:
            U = traj[s].copy(); e = 0.0
            for j in range(1, K+1):
                U = step_fn(U, theta)
                if not np.all(np.isfinite(U)): return 1e6
                e += np.mean((U-traj[s+j])**2)
            err += e/K
        return err/len(starts)
    res = minimize(loss, theta0, method='L-BFGS-B', options={'maxiter': iters, 'maxfun': iters*60})
    return res.x, res.fun

def rollout_B(step_fn, theta, U0, nsteps):
    U = U0.copy(); mass0 = U.sum(); mn = U.min(); drift = 0.0; blew = False
    Us = [U.copy()]
    for _ in range(nsteps):
        U = step_fn(U, theta)
        if not np.all(np.isfinite(U)): blew = True; break
        mn = min(mn, U.min()); drift = max(drift, abs(U.sum()-mass0)/mass0); Us.append(U.copy())
    return np.array(Us), mn, drift, blew

def run_B(noise=0.02):
    M, L = 64, 1.0; h = L/M; dt = 0.004; ntrain = 40; ntest = 200
    x, traj = groundtruth_B(M, h, dt, ntrain+ntest)
    rng = np.random.default_rng(11)
    train = traj[:ntrain+1] + noise*traj.max()*rng.standard_normal(traj[:ntrain+1].shape)  # noisy data
    nm = NMINNConserv(M, h, dt); nf = NFVConserv(M, h, dt, H=10, seed=2)
    th_nm, l_nm = train_stepper_B(nm.step, np.array([np.log(0.02), np.log(0.05), 0.0]), train)
    th_nf, l_nf = train_stepper_B(nf.step, nf.theta0, train)
    r_nm, mn_nm, dr_nm, bl_nm = rollout_B(nm.step, th_nm, traj[0], ntrain+ntest)
    r_nf, mn_nf, dr_nf, bl_nf = rollout_B(nf.step, th_nf, traj[0], ntrain+ntest)
    def rerr(r):
        n = min(len(r), len(traj))
        return relL2(r[:n], traj[:n]) if np.all(np.isfinite(r[:n])) else np.inf
    print("\n=== Problem B: 2-species conservative exchange, learned steppers ===")
    print(f"  grid M={M}, dt={dt}, train steps={ntrain}, rollout steps={ntrain+ntest}")
    print(f"  NMINN     : train MSE={l_nm:.2e}  rollout relL2={rerr(r_nm):.2e}  min={mn_nm:.3e}  mass drift={dr_nm:.2e}  blew_up={bl_nm}")
    print(f"  baseline  : train MSE={l_nf:.2e}  rollout relL2={rerr(r_nf):.2e}  min={mn_nf:.3e}  mass drift={dr_nf:.2e}  blew_up={bl_nf}")
    np.savez(os.path.join(os.path.dirname(__file__), "resB.npz"),
             x=x, traj=traj, r_nm=r_nm, r_nf=r_nf, dt=dt, ntrain=ntrain)
    return dict(mn_nm=mn_nm, mn_nf=mn_nf, dr_nm=dr_nm, dr_nf=dr_nf, err_nm=rerr(r_nm), err_nf=rerr(r_nf),
                bl_nf=bl_nf, bl_nm=bl_nm)

if __name__ == "__main__":
    A = run_A(); B = run_B()
    print("\n=== SUMMARY ===")
    print(f"A positivity: NMINN min={A['mn_nm']:.2e} (>=0 by construction) | baseline min={A['mn_nf']:.2e}")
    print(f"B mass drift: NMINN={B['dr_nm']:.2e} (0 by construction) | baseline={B['dr_nf']:.2e}")
    print("=== done ===")
