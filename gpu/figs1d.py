"""Figures for the 1D study, including the physics-informed baselines. Run collect1d.py first."""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "figs"))  # figures of the paper, regenerated in place
os.makedirs(FIG, exist_ok=True)

STYLE = {"NMINN": dict(color="C0", ls="-", lw=1.9),
         "neural stepper": dict(color="C3", ls=":", lw=1.6),
         "PINN": dict(color="C2", ls="--", lw=1.6),
         "PINN (positive output)": dict(color="C4", ls="-.", lw=1.4)}


def rel_series(R, truth, n):
    return np.array([np.linalg.norm(R[i] - truth[i]) / (np.linalg.norm(truth[i]) + 1e-30) for i in range(n)])


# ------------------------------------------------------------------ problem A
A = np.load(os.path.join(HERE, "resA_gpu.npz"))
x, traj, dt, ntrain = A["x"], A["traj"], float(A["dt"]), int(A["ntrain"])
curves = {"NMINN": A["r_nm"], "neural stepper": A["r_nf"],
          "PINN": A["pinn_vanilla"], "PINN (positive output)": A["pinn_hardpos"]}
n = min([len(traj)] + [len(c) for c in curves.values()])
tt = np.arange(n) * dt

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(x, traj[n - 1], "k-", lw=2, label="reference")
for k, c in curves.items():
    ax[0].plot(x, c[n - 1], label=k, **STYLE[k])
ax[0].axhline(0, color="0.6", lw=0.6)
ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$u$")
ax[0].set_title(f"(a) front profile at $t={(n-1)*dt:.2f}$")
ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

for k, c in curves.items():
    ax[1].semilogy(tt, rel_series(c, traj, n) + 1e-16, label=k, **STYLE[k])
ax[1].axvline(ntrain * dt, color="0.5", ls="--", lw=1)
ax[1].text(ntrain * dt * 1.03, ax[1].get_ylim()[0] * 3, "training horizon", fontsize=7,
           color="0.4", rotation=90, va="bottom")
ax[1].set_xlabel("$t$"); ax[1].set_ylabel("relative error")
ax[1].set_title("(b) rollout error and extrapolation")
ax[1].grid(alpha=0.3, which="both")

for k, c in curves.items():
    ax[2].plot(tt, [c[i].min() for i in range(n)], label=k, **STYLE[k])
ax[2].axhline(0, color="k", lw=0.8)
ax[2].set_xlabel("$t$"); ax[2].set_ylabel(r"$\min_i u_i$")
ax[2].set_title("(c) positivity")
ax[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_nminn_A.pdf"), bbox_inches="tight")
print("wrote fig_nminn_A.pdf")

# ------------------------------------------------------------------ problem B
B = np.load(os.path.join(HERE, "resB_gpu.npz"))
trajB, dtB, ntrainB = B["traj"], float(B["dt"]), int(B["ntrain"])
curvesB = {"NMINN": B["r_nm"], "neural stepper": B["r_nf"],
           "PINN": B["pinn_vanilla"], "PINN (positive output)": B["pinn_hardpos"]}
nB = min([len(trajB)] + [len(c) for c in curvesB.values()])
ttB = np.arange(nB) * dtB
m0 = trajB[0].sum()

fig2, ax2 = plt.subplots(1, 3, figsize=(13, 3.6))
for k, c in curvesB.items():
    ax2[0].semilogy(ttB, rel_series(c, trajB, nB) + 1e-16, label=k, **STYLE[k])
ax2[0].axvline(ntrainB * dtB, color="0.5", ls="--", lw=1)
ax2[0].set_xlabel("$t$"); ax2[0].set_ylabel("relative error")
ax2[0].set_title("(a) rollout error")
ax2[0].legend(fontsize=7); ax2[0].grid(alpha=0.3, which="both")

for k, c in curvesB.items():
    ax2[1].plot(ttB, [c[i].min() for i in range(nB)], label=k, **STYLE[k])
ax2[1].axhline(0, color="k", lw=0.8)
ax2[1].set_xlabel("$t$"); ax2[1].set_ylabel(r"$\min_{c,i} U_{c,i}$")
ax2[1].set_title("(b) positivity")
ax2[1].grid(alpha=0.3)

for k, c in curvesB.items():
    mass = np.array([c[i].sum() for i in range(nB)])
    ax2[2].semilogy(ttB, np.abs(mass - m0) / m0 + 1e-17, label=k, **STYLE[k])
ax2[2].set_xlabel("$t$"); ax2[2].set_ylabel("relative mass drift")
ax2[2].set_title("(c) conservation")
ax2[2].grid(alpha=0.3, which="both")
fig2.tight_layout()
fig2.savefig(os.path.join(FIG, "fig_nminn_B.pdf"), bbox_inches="tight")
print("wrote fig_nminn_B.pdf")
