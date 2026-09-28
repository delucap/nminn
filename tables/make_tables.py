"""Generate every table of the revised manuscript from the stored JSON results in ../gpu.

No number in a table is typed by hand. By default the six tables are written to tables_r1.tex in this
folder; with --tex <manuscript.tex> they are spliced into the manuscript between its
'%% BEGIN AUTO <name>' and '%% END AUTO <name>' markers instead. Missing result files leave a comment.

    python make_tables.py [--tag _r1] [--out tables_r1.tex | --tex path/to/manuscript.tex]

The tables use booktabs and the macro \\eu of the manuscript, \\newcommand{\\eu}{\\mathrm{e}}.
"""
import argparse
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.path.normpath(os.path.join(HERE, "..", "gpu"))


def load(name, tag, fallback=True):
    p = os.path.join(GPU, f"{name}{tag}.json")
    if not os.path.exists(p) and fallback and tag:
        p = os.path.join(GPU, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def sci(x, d=2):
    """1.37e-02 -> $1.37\\eu{-}2$ in the notation of the paper, plain for 0.1 <= |x| < 100."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    if x == 0:
        return "$0$"
    if 10 <= abs(x) < 100 and float(x).is_integer():
        return f"${int(x)}$"
    if 10 <= abs(x) < 100 and d <= 1:
        return f"${x:.0f}$"
    if 0.1 <= abs(x) < 100:
        return f"${x:.{d}f}$"
    m, e = f"{x:.{d}e}".split("e")
    e = int(e)
    return f"${m}\\eu{{{'-' if e < 0 else '+'}}}{abs(e)}$"


def sec(x, d=2):
    return "--" if x is None else f"${x:.{d}f}$"


def table(caption, label, colspec, header, rows, small=True, resize=True, notes=None):
    out = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if small:
        out.append("\\small")
    if resize:
        out.append("\\resizebox{\\textwidth}{!}{%")
    out += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + "\\\\", "\\midrule"]
    for r in rows:
        out.append(r if r in ("\\midrule",) else r + "\\\\")
    if notes:
        out += ["\\midrule"] + [f"\\multicolumn{{{len(colspec.replace('@', ''))}}}{{l}}{{\\footnotesize {n}}}\\\\" for n in notes]
    out += ["\\bottomrule", "\\end{tabular}" + ("}" if resize else ""), "\\end{table}"]
    return "\n".join(out)


def splice(tex, name, body):
    pat = re.compile(rf"(%% BEGIN AUTO {re.escape(name)}\n)(.*?)(%% END AUTO {re.escape(name)})", re.S)
    if not pat.search(tex):
        print(f"  marker {name} not found in tex")
        return tex
    return pat.sub(lambda m: m.group(1) + body + "\n" + m.group(3), tex)


def tab_grad(tag):
    g = load("gradcheck_taxis", tag)
    if not g:
        return None
    rows = []
    for r in g["part_A"]:
        if abs(r["tau"] - 5e-3) > 1e-12:
            continue
        rows.append(" & ".join([sci(r["tol"], 0), str(r["K"]), sci(r["piK"], 1), sci(r["rel_frozen_vs_fd"], 1),
                                sci(r["rel_total_vs_fd"], 1), sci(r["abs_frozen_minus_total"], 1),
                                sci(r["bound_frozen_minus_total"], 1)]))
    hdr = "tolerance & $K$ & $\\pi_K$ & frozen $\\theta$ vs FD & total vs FD & $|$frozen $-$ total$|$ & bound \\eqref{eq:dtheta}"
    return table("Gradient check of the truncated uniformization layer at the internal step $\\tau=5\\times10^{-3}$.",
                 "tab:grad", "lrccccc", hdr, rows)


def tab_W(tag):
    c = load("classical_baseline", tag)
    if not c:
        return None
    rows = []
    for k in sorted(c["classical"], key=lambda z: -float(z)):
        r = c["classical"][k]
        rows.append(" & ".join([f"classical, $\\tau={sci(r['dt'],1)[1:-1]}$", str(r["nsteps"]), sci(r["relL2_u"]),
                                sci(r["relL2_u_extrap"]), sci(r["min_u"], 1), sci(r["mass_drift"], 1), sec(r["seconds"])]))
    rows.append("\\midrule")
    ident = c.get("identified", {})
    for k, v in ident.items():
        if k == "coefficients":
            continue
        rows.append(" & ".join([f"classical, identified coefficients, $\\tau={sci(float(k),1)[1:-1]}$",
                                str(int(round(c['setup']['Dt'] * c['setup']['n_macro'] / float(k)))),
                                sci(v["relL2_u"]), sci(v["relL2_u_extrap"]), sci(v["min_u"], 1), sci(v["mass_drift"], 1),
                                sec(v["seconds"])]))
    sur = c.get("surrogates", {})
    if "NMINN" in sur:
        v = sur["NMINN"]
        rows.append(" & ".join(["NMINN, trained, $\\Delta t/4$ internal step", str(4 * c["setup"]["n_macro"]),
                                sci(v["relL2_u"]), sci(v["relL2_u_extrap"]), sci(v["min_u"], 1), sci(v["mass_drift"], 1),
                                sec(v["seconds"])]))
    hdr = "solver & steps & rel.\\ $L^2$ ($u$) & extrapolation & $\\min u$ & mass drift & time (s)"
    notes = None
    if c.get("reference_self_error"):
        notes = [f"error of the reference itself against the run at $\\tau=2.5\\times10^{{-4}}$, {sci(c['reference_self_error']['relL2_u'])}. "
                 f"Largest uniformization rate along the reference, $\\theta_{{\\max}}={c['theta']['theta_max']:.0f}$, explicit upwind positive only for $\\tau\\le{sci(c['theta']['cfl_dt'],1)[1:-1]}$."]
    return table("Problem C, the classical splitting scheme with the true coefficients at decreasing steps, against the trained surrogate and the scheme with the identified coefficients.",
                 "tab:W", "lrccccc", hdr, rows, notes=notes)


PINN_STATIC = None


def pinn_rows_2d():
    """Physics-informed baselines of Problem C from the stored runs (unchanged in the revision)."""
    d = os.path.join(GPU, "pinn2d_runs")
    out = []
    names = {"vanilla": "PINN", "penalty": "PINN, positivity penalty", "hardpos": "PINN, positive output"}
    for v, name in names.items():
        files = sorted(f for f in os.listdir(d) if f.startswith(f"pinn2d_{v}") and f.endswith(".json")) if os.path.isdir(d) else []
        if not files:
            continue
        best = min((json.load(open(os.path.join(d, f))) for f in files), key=lambda j: j.get("loss", j.get("train_loss", 1e9)))
        out.append((name, best))
    return out


def tab_C(tag):
    res = load("results_2d_M128", tag, fallback=False)
    if not res:
        return None
    cb = load("classical_baseline", tag, fallback=False) or {}
    sur_t = cb.get("surrogates", {})
    order = ["NMINN", "CNN stepper", "FNO stepper", "CNN flux-form", "CNN positive", "CNN positive projected"]
    label = {"NMINN": "NMINN", "CNN stepper": "convolutional stepper", "FNO stepper": "Fourier neural operator",
             "CNN flux-form": "convolutional, flux form", "CNN positive": "convolutional, positive output",
             "CNN positive projected": "convolutional, positive and projected"}
    rows = []
    for i, k in enumerate(order):
        if k not in res:
            continue
        r = res[k]
        t_roll = sur_t.get(k, {}).get("seconds", r.get("inference_seconds"))
        cells = [label[k], f"${r['params']:,}$".replace(",", "\\,"), sci(r["relL2_u"]), sci(r["relL2_u_extrap"]),
                 sci(r["min_u"], 1), sci(r["mass_drift"], 1), sec(r["seconds"], 0), sec(t_roll)]
        if k == "NMINN":
            cells = [cells[0], cells[1]] + [f"\\mathbf{{{c[1:-1]}}}".join(["$", "$"]) if c.startswith("$") else c for c in cells[2:6]] + cells[6:]
        rows.append(" & ".join(cells))
        if k == "FNO stepper":
            rows.append("\\midrule")
    pinn = pinn_rows_2d()
    if pinn:
        rows.append("\\midrule")
        for name, j in pinn:
            rows.append(" & ".join([name, "$82\\,823$", f"${j['relL2_u']:.4f}$", f"${j['relL2_u_extrap']:.4f}$",
                                    sci(j["min_u"], 1), sci(j["mass_drift"], 1), sec(j.get("seconds"), 0), "n/a"]))
    hdr = "surrogate & params & rel.\\ $L^2$ ($u$) & extrapolation & $\\min u$ & mass drift & training (s) & rollout (s)"
    return table("Problem C, two-dimensional Keller--Segel aggregation on a $128\\times128$ grid, macro step $\\Delta t=2\\times10^{-2}$, eighty steps.",
                 "tab:C", "lrcccccc", hdr, rows)


def tab_scale(tag):
    t = load("timings_2d", tag, fallback=False)
    if not t:
        return None
    rows = []
    for M in sorted(t, key=int):
        r = t[M]
        cl = r["classical_internal_step"]
        n = r["NMINN"]
        rows.append(" & ".join([f"${M}^2$", sec(r["reference"]), sec(cl["seconds"]), sec(n["seconds"]),
                                f"${r['reference'] / n['seconds']:.1f}$", f"${n['seconds'] / cl['seconds']:.1f}$",
                                sec(r["CNN"]["seconds"]), sec(r["FNO"]["seconds"])]))
    hdr = ("grid & reference & classical, $\\Delta t/4$ & NMINN & reference / NMINN & NMINN / classical & "
           "convolutional & Fourier")
    return table("Problem C, wall-clock time in seconds of the eighty-step horizon at four resolutions, trained weights, idle device.",
                 "tab:scale", "lccccccc", hdr, rows)


def tab_IC(tag):
    d = load("ic_transfer", tag, fallback=False)
    if not d:
        return None
    label = {"bumps": "training datum", "bumps_rand": "random bumps", "strong": "strong bumps", "subcritical": "two small bumps"}
    order = ["NMINN", "CNN stepper", "FNO stepper", "CNN flux-form", "CNN positive", "CNN positive projected"]
    rows = []
    for spec, r in d.items():
        if spec.startswith("_"):
            continue
        name = label[r["kind"]] + (f" {r['seed']}" if r["kind"] == "bumps_rand" else "")
        cells = [name, f"${r['peak_growth']:.0f}$", sci(r["classical_internal_step"]["relL2_u"])]
        for k in order:
            cells.append(sci(r[k]["relL2_u"]) if k in r else "--")
        rows.append(" & ".join(cells))
    hdr = "initial datum & peak growth & classical & NMINN & conv. & Fourier & flux form & positive & projected"
    return table("Problem C, relative error of the density over the full horizon for trained surrogates rolled out from initial data they never saw.",
                 "tab:IC", "lcccccccc", hdr, rows)


def tab_noise(tag):
    d = load("noise_variability_1d", "", fallback=False)
    if not d:
        return None
    names = {"NMINN": "NMINN", "PINN hardpos": "PINN, positive output", "PINN project": "PINN, positive and projected"}
    sym = {"d": "d", "r": "r", "d1": "d_1", "d2": "d_2", "k": "k"}

    def fs(x, d=2):
        """always in exponent form, so that a column never mixes 0.10 with 8.51e-2"""
        m, e = f"{x:.{d}e}".split("e")
        return f"{m}\\eu{{{'-' if int(e) < 0 else '+'}}}{abs(int(e))}"

    rows = []
    for problem in ("A", "B"):
        s = d[problem]["summary"]
        for m in ("NMINN", "PINN hardpos", "PINN project"):
            if m not in s:
                continue
            st = s[m]
            rng = f"$[{fs(st['relL2']['min'])},\\ {fs(st['relL2']['max'])}]$"
            coef = "$" + ",\\ ".join(
                f"{sym[k]}={v['mean']:.4f}\\pm{v['std']:.4f}" if k.startswith("d") else f"{sym[k]}={v['mean']:.2f}\\pm{v['std']:.2f}"
                for k, v in st["coef"].items() if k != "b") + "$"
            rows.append(" & ".join([problem, names[m],
                                    f"${fs(st['relL2']['mean'])}\\pm{fs(st['relL2']['std'], 1)}$", rng,
                                    f"${fs(st['relL2_extrap']['mean'])}\\pm{fs(st['relL2_extrap']['std'], 1)}$",
                                    coef]))
        if problem == "A":
            rows.append("\\midrule")
    hdr = ("problem & method & rollout rel.\\ $L^2$, mean $\\pm$ std & range & extrapolation, mean $\\pm$ std & "
           "recovered coefficients, mean $\\pm$ std")
    notes = ["five realizations of the two-percent noise per problem, the one of Tables~\\ref{tab:A} and~\\ref{tab:B} and four further ones. "
             "True coefficients, A: $d=0.05$, $r=6$. B: $d_1=0.03$, $d_2=0.08$, $k=6$."]
    return table("Problems A and B, rollout error, extrapolation error and recovered coefficients over five noise realizations.",
                 "tab:noise", "llcccl", hdr, rows, notes=notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_r1", help="suffix of the result files of the revised paper")
    ap.add_argument("--out", default=os.path.join(HERE, "tables_r1.tex"), help="file receiving the six tables")
    ap.add_argument("--tex", default=None, help="manuscript with AUTO markers to splice the tables into")
    a = ap.parse_args()
    tables = []
    for name, fn in (("tab:grad", tab_grad), ("tab:W", tab_W), ("tab:C", tab_C), ("tab:scale", tab_scale),
                     ("tab:IC", tab_IC), ("tab:noise", tab_noise)):
        body = fn(a.tag)
        if body is None:
            print(f"  {name}: results not available, left as a comment")
            body = f"%% {name}: results not available"
        else:
            print(f"  {name}: generated")
        tables.append((name, body))
    if a.tex:
        tex = open(a.tex).read()
        for name, body in tables:
            tex = splice(tex, name, body)
        open(a.tex, "w").write(tex)
        print("updated", a.tex)
    else:
        out = ["%% Tables of the revised manuscript, generated by tables/make_tables.py from the JSON results in gpu/.",
               "%% Requires booktabs, graphicx (resizebox) and \\newcommand{\\eu}{\\mathrm{e}}.", ""]
        for name, body in tables:
            out += [f"%% BEGIN AUTO {name}", body, f"%% END AUTO {name}", ""]
        open(a.out, "w").write("\n".join(out))
        print("wrote", os.path.relpath(a.out))


if __name__ == "__main__":
    main()
