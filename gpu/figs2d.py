"""Figures for the 2D Keller-Segel study. Run run2d.py (and optionally run2d_pinn.py) first."""
import glob
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "manuscript", "figs"))
os.makedirs(FIG, exist_ok=True)
M = int(os.environ.get("M2D", "128"))

d = torch.load(os.path.join(HERE, f"res2d_M{M}.pt"), weights_only=False)
us = d["us"].numpy()
setup = d["setup"]
Dt, ntrain = setup["Dt"], setup["ntrain"]
preds = {k: (a.numpy(), b.numpy()) for k, (a, b) in d["preds"].items()}

pinn = sorted(glob.glob(os.path.join(HERE, "pinn2d_runs", "pred_*.pt")))
if pinn:
    best, bestval = None, np.inf
    for f in pinn:
        j = json.load(open(f.replace("pred_", "pinn2d_").replace(".pt", ".json")))
        if j["relL2_u"] < bestval:
            bestval, best = j["relL2_u"], f
    pd = torch.load(best, weights_only=False)
    preds["PINN"] = (pd["pu"].double().numpy(), pd["pv"].double().numpy())

order = [k for k in ("NMINN", "CNN stepper", "FNO stepper", "PINN") if k in preds]
n = min([len(us)] + [len(preds[k][0]) for k in order])
tt = np.arange(n) * Dt

# ------------------------------------------------------------------ fields at the final time
vmax = float(us[n - 1].max())
fig, ax = plt.subplots(1, len(order) + 1, figsize=(3.1 * (len(order) + 1), 3.0))
norm = LogNorm(vmin=max(1e-3, float(us[n - 1].min())), vmax=vmax)
im = ax[0].imshow(us[n - 1].T, origin="lower", extent=[0, 1, 0, 1], norm=norm, cmap="magma")
ax[0].set_title(f"reference, $t={(n-1)*Dt:.2f}$", fontsize=9)
for a, k in zip(ax[1:], order):
    p = preds[k][0][n - 1]
    a.imshow(np.clip(p, norm.vmin, None).T, origin="lower", extent=[0, 1, 0, 1], norm=norm, cmap="magma")
    neg = (p < 0).sum()
    a.set_title(f"{k}" + (f"\n{neg} negative cells" if neg else "\nno negative cell"), fontsize=9)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.colorbar(im, ax=ax, shrink=0.85, label="$u$")
fig.savefig(os.path.join(FIG, "fig_nminn_C_fields.pdf"), bbox_inches="tight")
print("wrote fig_nminn_C_fields.pdf")

# ------------------------------------------------------------------ error, positivity, mass
rel = lambda P: np.array([np.linalg.norm(P[i] - us[i]) / np.linalg.norm(us[i]) for i in range(n)])
fig2, ax2 = plt.subplots(1, 3, figsize=(13, 3.6))
sty = {"NMINN": dict(color="C0", ls="-", lw=1.9), "CNN stepper": dict(color="C3", ls=":", lw=1.6),
       "FNO stepper": dict(color="C1", ls="-.", lw=1.6), "PINN": dict(color="C2", ls="--", lw=1.6)}
for k in order:
    ax2[0].semilogy(tt, rel(preds[k][0]) + 1e-16, label=k, **sty[k])
ax2[0].axvline(ntrain * Dt, color="0.5", ls="--", lw=1)
ax2[0].text(ntrain * Dt * 1.03, ax2[0].get_ylim()[0] * 3, "training horizon", fontsize=7,
            color="0.4", rotation=90, va="bottom")
ax2[0].set_xlabel("$t$"); ax2[0].set_ylabel("relative error in $u$")
ax2[0].set_title("(a) rollout error"); ax2[0].legend(fontsize=7); ax2[0].grid(alpha=0.3, which="both")

for k in order:
    ax2[1].plot(tt, [preds[k][0][i].min() for i in range(n)], label=k, **sty[k])
ax2[1].axhline(0, color="k", lw=0.8)
ax2[1].set_xlabel("$t$"); ax2[1].set_ylabel(r"$\min_{ij} u_{ij}$")
ax2[1].set_title("(b) positivity"); ax2[1].grid(alpha=0.3)

m0 = us[0].sum()
for k in order:
    m = np.array([preds[k][0][i].sum() for i in range(n)])
    ax2[2].semilogy(tt, np.abs(m - m0) / m0 + 1e-17, label=k, **sty[k])
ax2[2].set_xlabel("$t$"); ax2[2].set_ylabel("relative mass drift")
ax2[2].set_title("(c) conservation"); ax2[2].grid(alpha=0.3, which="both")
fig2.tight_layout()
fig2.savefig(os.path.join(FIG, "fig_nminn_C_invariants.pdf"), bbox_inches="tight")
print("wrote fig_nminn_C_invariants.pdf")

# ------------------------------------------------------------------ cost against accuracy
res = json.load(open(os.path.join(HERE, f"results_2d_M{M}.json")))
# cost numbers come from the dedicated timing run on an idle device, not from the training run,
# which shared the device with other jobs
tpath = os.path.join(HERE, "timings_2d.json")
if os.path.exists(tpath):
    t = json.load(open(tpath))[str(M)]
    setup["reference_seconds"] = t["reference"]
    for k, key in (("NMINN", "NMINN"), ("CNN stepper", "CNN"), ("FNO stepper", "FNO")):
        if k in res:
            res[k]["inference_seconds"] = t[key]["seconds"]
fig3, ax3 = plt.subplots(figsize=(5.0, 3.6))
for k in order:
    if k not in res:
        continue
    ax3.scatter(res[k]["inference_seconds"], res[k]["relL2_u"], s=70,
                color=sty[k]["color"], label=f"{k} ({res[k]['params']} par.)")
ax3.axvline(setup["reference_seconds"], color="k", ls="--", lw=1)
ax3.text(setup["reference_seconds"], ax3.get_ylim()[1], " reference solver", fontsize=7,
         rotation=90, va="top")
ax3.set_xscale("log"); ax3.set_yscale("log")
ax3.set_xlabel("rollout wall-clock time (s)"); ax3.set_ylabel("relative error in $u$")
ax3.set_title("cost against accuracy over the full horizon", fontsize=9)
ax3.legend(fontsize=7); ax3.grid(alpha=0.3, which="both")
fig3.tight_layout()
fig3.savefig(os.path.join(FIG, "fig_nminn_C_cost.pdf"), bbox_inches="tight")
print("wrote fig_nminn_C_cost.pdf")
