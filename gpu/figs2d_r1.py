"""Figures of the two-dimensional study for the revised manuscript. Run on the machine holding the results.

    python figs2d_r1.py --tag _r1

Reads res2d_M128<tag>.pt (reference and predictions of every trained stepper), classical_baseline<tag>.json
(work-precision curve of the classical scheme and trained-weight timings) and the physics-informed run with
the lowest training loss in pinn2d_runs/. Writes the three figures of Problem C into ../../manuscript/figs
and into ../../revision_1/figs when that folder exists.
"""
import argparse
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
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_r1")
ap.add_argument("--M", type=int, default=128)
a = ap.parse_args()
OUTS = [os.path.abspath(os.path.join(HERE, "..", "figs"))]  # figures of the paper, regenerated in place
for o in OUTS:
    os.makedirs(o, exist_ok=True)


def save(fig, name):
    for o in OUTS:
        fig.savefig(os.path.join(o, name), bbox_inches="tight")
    print("wrote", name, "->", ", ".join(OUTS))


d = torch.load(os.path.join(HERE, f"res2d_M{a.M}{a.tag}.pt"), weights_only=False, map_location="cpu")
us = d["us"].numpy()
setup = d["setup"]
Dt, ntrain = setup["Dt"], setup["ntrain"]
preds = {k: (p.numpy(), q.numpy()) for k, (p, q) in d["preds"].items()}

pinn = sorted(glob.glob(os.path.join(HERE, "pinn2d_runs", "pred_*.pt")))
if pinn:
    # selection on the training objective, whatever the run stored it as; never on a test quantity
    def train_crit(f):
        j = json.load(open(f.replace("pred_", "pinn2d_").replace(".pt", ".json")))
        for k in ("loss", "train_loss", "final_loss", "total_loss"):
            if k in j:
                return float(j[k])
        if "data_mse" in j and "res_mse" in j:
            return float(j["data_mse"]) + float(j["res_mse"])
        return 0.0
    best = min(pinn, key=train_crit)
    pd = torch.load(best, weights_only=False, map_location="cpu")
    preds["PINN"] = (pd["pu"].double().numpy(), pd["pv"].double().numpy())

short = {"NMINN": "NMINN", "CNN stepper": "convolutional", "FNO stepper": "Fourier operator",
         "CNN flux-form": "conv., flux form", "CNN positive": "conv., positive", "CNN positive projected": "conv., projected",
         "PINN": "PINN"}
order = [k for k in ("NMINN", "CNN stepper", "FNO stepper", "CNN flux-form", "CNN positive",
                     "CNN positive projected", "PINN") if k in preds]
n = min([len(us)] + [len(preds[k][0]) for k in order])
tt = np.arange(n) * Dt
sty = {"NMINN": dict(color="C0", ls="-", lw=2.0), "CNN stepper": dict(color="C3", ls=":", lw=1.5),
       "FNO stepper": dict(color="C1", ls="-.", lw=1.5), "CNN flux-form": dict(color="C4", ls="--", lw=1.3),
       "CNN positive": dict(color="C5", ls=":", lw=1.3), "CNN positive projected": dict(color="C6", ls="-.", lw=1.3),
       "PINN": dict(color="C2", ls="--", lw=1.5)}

# ------------------------------------------------------------------ fields at the final time, two rows
panels = ["reference"] + order
ncol = 4
nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
axes = np.atleast_2d(axes).ravel()
vmax = float(us[n - 1].max())
norm = LogNorm(vmin=max(1e-3, float(us[n - 1].min())), vmax=vmax)
im = axes[0].imshow(us[n - 1].T, origin="lower", extent=[0, 1, 0, 1], norm=norm, cmap="magma")
axes[0].set_title(f"reference, $t={(n-1)*Dt:.2f}$", fontsize=9)
for ax, k in zip(axes[1:], order):
    p = preds[k][0][n - 1]
    ax.imshow(np.clip(p, norm.vmin, None).T, origin="lower", extent=[0, 1, 0, 1], norm=norm, cmap="magma")
    neg = int((p < 0).sum())
    ax.set_title(short[k] + (f"\n{neg} negative cells" if neg else "\nno negative cell"), fontsize=9)
for ax in axes[len(panels):]:
    ax.axis("off")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="$u$")
save(fig, "fig_nminn_C_fields.pdf")

# ------------------------------------------------------------------ error, positivity, mass
rel = lambda P: np.array([np.linalg.norm(P[i] - us[i]) / np.linalg.norm(us[i]) for i in range(n)])
fig2, ax2 = plt.subplots(1, 3, figsize=(13, 3.8))
for k in order:
    ax2[0].semilogy(tt, rel(preds[k][0]) + 1e-16, label=short[k], **sty[k])
ax2[0].axvline(ntrain * Dt, color="0.5", ls="--", lw=1)
ax2[0].text(ntrain * Dt * 1.03, ax2[0].get_ylim()[0] * 3, "training horizon", fontsize=7, color="0.4",
            rotation=90, va="bottom")
ax2[0].set_xlabel("$t$"); ax2[0].set_ylabel("relative error in $u$")
ax2[0].set_title("(a) rollout error"); ax2[0].legend(fontsize=6.5, ncol=2); ax2[0].grid(alpha=0.3, which="both")
for k in order:
    ax2[1].plot(tt, [preds[k][0][i].min() for i in range(n)], label=short[k], **sty[k])
ax2[1].axhline(0, color="k", lw=0.8)
ax2[1].set_yscale("symlog", linthresh=1e-2)
ax2[1].set_xlabel("$t$"); ax2[1].set_ylabel(r"$\min_{ij} u_{ij}$")
ax2[1].set_title("(b) positivity"); ax2[1].grid(alpha=0.3)
m0 = us[0].sum()
for k in order:
    m = np.array([preds[k][0][i].sum() for i in range(n)])
    ax2[2].semilogy(tt, np.abs(m - m0) / m0 + 1e-17, label=short[k], **sty[k])
ax2[2].set_xlabel("$t$"); ax2[2].set_ylabel("relative mass drift")
ax2[2].set_title("(c) conservation"); ax2[2].grid(alpha=0.3, which="both")
fig2.tight_layout()
save(fig2, "fig_nminn_C_invariants.pdf")

# ------------------------------------------------------------------ cost against accuracy, with the classical curve
cb_path = os.path.join(HERE, f"classical_baseline{a.tag}.json")
res = json.load(open(os.path.join(HERE, f"results_2d_M{a.M}{a.tag}.json")))
fig3, ax3 = plt.subplots(figsize=(5.4, 3.9))
if os.path.exists(cb_path):
    cb = json.load(open(cb_path))
    cl = sorted(cb["classical"].values(), key=lambda r: r["seconds"])
    xs = [r["seconds"] for r in cl if r["relL2_u"] > 0]
    ys = [r["relL2_u"] for r in cl if r["relL2_u"] > 0]
    ax3.plot(xs, ys, "k-o", ms=4, lw=1.2, label="classical scheme, true coefficients")
    for r in cl:
        if r["relL2_u"] > 0:
            ax3.annotate(f"$\\tau={r['dt']:.0e}$".replace("e-0", "\\times10^{-") + "}$" if False else f"{r['dt']:.1e}",
                         (r["seconds"], r["relL2_u"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ref_sec = cb["classical"][str(cb["setup"]["dt_ref"])]["seconds"] if str(cb["setup"]["dt_ref"]) in cb["classical"] \
        else cb["classical"]["0.0005"]["seconds"]
    ident = cb.get("identified", {})
    for k, v in ident.items():
        if k != "coefficients" and abs(float(k) - cb["setup"]["Dt"] / 4) < 1e-12:
            ax3.scatter(v["seconds"], v["relL2_u"], marker="s", s=60, color="0.4",
                        label="classical, identified coefficients", zorder=5)
    times = {k: v["seconds"] for k, v in cb.get("surrogates", {}).items()}
else:
    ref_sec = setup["reference_seconds"]
    times = {k: res[k]["inference_seconds"] for k in res if not k.startswith("_")}
for k in order:
    if k in res and k in times:
        ax3.scatter(times[k], res[k]["relL2_u"], s=70, color=sty[k]["color"], zorder=6,
                    label=f"{short[k]} ({res[k]['params']} par.)")
ax3.axvline(ref_sec, color="k", ls="--", lw=1)
ax3.text(ref_sec, ax3.get_ylim()[1], " reference", fontsize=7, rotation=90, va="top")
ax3.set_xscale("log"); ax3.set_yscale("log")
ax3.set_xlabel("rollout wall-clock time (s), trained weights, idle device"); ax3.set_ylabel("relative error in $u$")
ax3.set_title("cost against accuracy over the full horizon", fontsize=9)
ax3.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False); ax3.grid(alpha=0.3, which="both")
fig3.tight_layout()
save(fig3, "fig_nminn_C_cost.pdf")
