"""Variability across noise realizations for the 1D problems (Reviewer 1, JOCS first round).

For each noise seed the NMINN selection sweep (select1d.py --noise_seed s) and the positive-output PINN
(pinn_worker.py --variant hardpos --noise_seed s, its projected variant derived post hoc) are scored
against the clean trajectory, and mean, standard deviation, minimum and maximum over the realizations are
reported for the rollout error, the extrapolation error and the recovered coefficients. The realization of
the paper (seed 7 for A, 11 for B) is included from select_1d.json and pinn_runs/, where the PINN entry is
the run with the lowest training loss among its three initializations, as in the paper.

    python noise_variability_1d.py --seeds 21 22 23 24
    python noise_variability_1d.py --seeds 21 22 23 24 --tag _r2     NMINN fits run to convergence
                                                                      (select1d_r2.py), PINN runs unchanged
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from data1d import load  # noqa: E402

relL2 = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def scores(pred, traj, ntrain, conservative):
    n = min(len(pred), len(traj))
    p, t = np.asarray(pred[:n], float), np.asarray(traj[:n], float)
    out = dict(relL2=relL2(p, t), relL2_extrap=relL2(p[ntrain:], t[ntrain:]), min=float(p.min()))
    if conservative:
        m = p.reshape(n, -1).sum(axis=1)
        out["mass_drift"] = float(np.max(np.abs(m - m[0]) / abs(m[0])))
    return out


def pinn_entry(problem, rundir, traj, ntrain, conservative, mass0, variant):
    files = sorted(glob.glob(os.path.join(rundir, f"pinn_{problem}_hardpos_*.npz")))
    if not files:
        return None
    losses = [json.load(open(f.replace(".npz", ".json")))["loss"] for f in files]
    pick = files[int(np.argmin(losses))]
    p = np.load(pick)["pred"]
    if variant == "project":
        m = p.reshape(len(p), -1).sum(axis=1)
        p = p * (mass0 / m).reshape(-1, *([1] * (p.ndim - 1)))
    info = json.load(open(pick.replace(".npz", ".json")))
    keys = ["d", "r"] if problem == "A" else ["d1", "d2", "k"]
    return scores(p, traj, ntrain, conservative) | dict(coef=dict(zip(keys, info["coef"])), run=os.path.basename(pick))


def stats(vals):
    v = np.array(vals, float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                min=float(v.min()), max=float(v.max()), n=int(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[21, 22, 23, 24])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sel_r2 = json.load(open(os.path.join(HERE, f"select_1d{a.tag}.json"))) if a.tag else None
    out = {}
    for problem in ("A", "B"):
        d = load(problem)
        traj, ntrain = d["traj"], d["cfg"]["ntrain"]
        conservative = problem == "B"
        mass0 = float(traj[0].sum())
        per_seed = {}
        # the realization of the paper
        base = sel_r2[problem][str(d["cfg"]["seed"])]["NMINN"] if sel_r2 else \
            json.load(open(os.path.join(HERE, "select_1d.json")))[problem]["NMINN"]
        entry = {"NMINN": base}
        for variant in (("hardpos", "project") if conservative else ("hardpos",)):
            e = pinn_entry(problem, os.path.join(HERE, "pinn_runs"), traj, ntrain, conservative, mass0, variant)
            if e:
                entry[f"PINN {variant}"] = e
        per_seed[str(d["cfg"]["seed"])] = entry
        for s in a.seeds:
            if sel_r2:
                if str(s) not in sel_r2.get(problem, {}):
                    continue
                entry = {"NMINN": sel_r2[problem][str(s)]["NMINN"]}
            else:
                p = os.path.join(HERE, f"select_1d_noise{s}.json")
                if not os.path.exists(p) or problem not in json.load(open(p)):
                    continue
                entry = {"NMINN": json.load(open(p))[problem]["NMINN"]}
            for variant in (("hardpos", "project") if conservative else ("hardpos",)):
                e = pinn_entry(problem, os.path.join(HERE, f"pinn_runs_noise{s}"), traj, ntrain, conservative,
                               mass0, variant)
                if e:
                    entry[f"PINN {variant}"] = e
            per_seed[str(s)] = entry
        if sel_r2:
            # the physics-informed runs were not repeated, so their entries are those of the first revision
            r1 = json.load(open(os.path.join(HERE, "noise_variability_1d.json")))[problem]["per_seed"]
            for s_, e_ in per_seed.items():
                for m_, v_ in r1.get(s_, {}).items():
                    if m_.startswith("PINN"):
                        e_[m_] = v_
        methods = sorted({m for e in per_seed.values() for m in e})
        summary = {}
        for m in methods:
            rows = [e[m] for e in per_seed.values() if m in e]
            summary[m] = {k: stats([r[k] for r in rows]) for k in ("relL2", "relL2_extrap", "min")}
            if conservative:
                summary[m]["mass_drift"] = stats([r["mass_drift"] for r in rows])
            if rows[0].get("coef"):
                summary[m]["coef"] = {ck: stats([r["coef"][ck] for r in rows]) for ck in rows[0]["coef"]}
        out[problem] = dict(per_seed=per_seed, summary=summary)
        print(f"\n=== Problem {problem}: {len(per_seed)} noise realizations ===")
        for m in methods:
            s_ = summary[m]
            line = f"{m:<16} relL2 {s_['relL2']['mean']:.3e} +- {s_['relL2']['std']:.1e} " \
                   f"[{s_['relL2']['min']:.3e}, {s_['relL2']['max']:.3e}]  extrap {s_['relL2_extrap']['mean']:.3e} " \
                   f"+- {s_['relL2_extrap']['std']:.1e}  n={s_['relL2']['n']}"
            if "coef" in s_:
                line += "  " + " ".join(f"{k}={v['mean']:.4f}+-{v['std']:.4f}" for k, v in s_["coef"].items())
            print(line)
    with open(os.path.join(HERE, f"noise_variability_1d{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote noise_variability_1d{a.tag}.json")


if __name__ == "__main__":
    main()
