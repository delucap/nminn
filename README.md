# Numerical-method-informed neural networks (NMINN)

Reference implementation and reproduction package for

> P. De Luca, *Numerical-method-informed neural networks, structure-preserving discretizations as
> architectural inductive bias*, Journal of Computational Science, manuscript JOCSCI-D-26-02458
> (second revision).

A method-informed network is a learnable map whose forward pass factors through a structure-preserving
discrete operator, so its outputs satisfy the operator's invariants for every value of the weights.
This repository contains the structural layers (exact exponential diffusion, Modified-Patankar reaction,
truncated uniformization transport), the learned steppers and every baseline of the paper, the scripts that
produced every number, the stored results, and the scripts that regenerate every table and figure from them.

## Layout

```
nminn.py, experiment.py      1D steppers and reference solvers, pure numpy/scipy (Problems A and B)
gpu/ops.py                   structural layers in torch (DCT exponential, ETD, MP22, uniformization)
gpu/transport.py             upwind taxis generator, Metzler with zero column sums
gpu/models2d.py              NMINN2D and the baselines (CNN, FNO, flux-form, positive, projected)
gpu/ks2d.py                  2D Keller-Segel reference solver (palindromic splitting, order 2)
gpu/run2d.py                 Problem C: data, training, rollout, resolution transfer
gpu/classical_baseline.py    matched classical scheme at decreasing steps, identified coefficients
gpu/gradcheck_taxis.py       gradient check of the uniformization layer and of the full stepper
gpu/ic_transfer.py           trained steppers on initial data they never saw
gpu/timings2d.py             wall-clock cost at four resolutions (--tag loads trained weights)
gpu/data1d.py, run1d.py      1D data (noisy snapshots) and the first single-fit protocol
gpu/select1d.py              1D model selection on the training loss, multi-start (--noise_seed)
gpu/select1d_r2.py           the same selection with the fits run to convergence (second revision), every
                             noise realization and the noise-free snapshots
gpu/fixedstart_b_r2.py       Problem B, one fit from a fixed start under the default and the tight stopping rule
gpu/coef_attribution_r2.py   Problem C, classical scheme with the identified coefficients one at a time
gpu/horizon1d.py, diag_B.py  horizon sweep and the landscape diagnostic of Problem B
gpu/pinn1d.py, pinn_worker.py, launch.sh, pinn2d.py, run2d_pinn.py   physics-informed baselines
gpu/collect1d.py, noise_variability_1d.py                            1D tables and noise-realization study
gpu/test_ops.py, test_models2d.py                                    verification suites (23 + 27 checks)
gpu/figs1d.py, figs2d_r1.py  figures, written to figs/
tables/make_tables.py        the six tables of the revised paper from the JSON results, to tables/tables_r1.tex
figs/                        the five figures of the paper as submitted
```

## Stored results

Every number in the paper is read from a file in `gpu/`, no number in a table is typed by hand.

- `gpu/*.json`: results of every experiment. The files with the suffix `_r1` are the runs of the revised
  paper (retrained surrogates, matched classical baseline, gradient check, initial-data transfer, timings at
  trained weights); the files without it are the runs of the first submitted version, kept for provenance.
  The 1D results (`results_1d.json`, `select_1d*.json`, `noise_variability_1d.json`, `pinn_runs*/*.json`,
  `pinn2d_runs/*.json`) are shared by both versions.
- `gpu/model_*_M128_r1*.pt`: trained weights of the six 2D steppers on the paper's noise realization and of
  the NMINN on two further realizations (`torch.load(..., weights_only=False)`).
- `gpu/sel_*_NMINN_noise*.npy`, `gpu/resA_gpu.npz`, `gpu/resB_gpu.npz`: selected 1D rollouts, the last two
  are what `figs1d.py` plots.
- `gpu/pinn_runs_noise*/*.npz`: predictions of the positive-output PINN on the four further noise
  realizations of the 1D noise study.

The 2D rollout fields `gpu/res2d_M128_r1*.pt` (149 MB and 2 x 42 MB) and the 2D PINN predictions
`gpu/pinn2d_runs/pred_*.pt` (3 x 11 MB) are not stored, because of their size. They are only needed by
`figs2d_r1.py`, and `run2d.py` and `run2d_pinn.py` regenerate them deterministically with the commands below
(about eight minutes for the NMINN alone, fifteen for the six steppers, eight per PINN variant on an RTX 4080
SUPER).

- Second revision (suffix `_r2`): `select_1d_r2.json` (every 1D fit of the learned steppers, rerun to
  convergence, on the five noise realizations and on the noise-free snapshots), `select_1d_r2_timing.json`
  (the sequential sweep of Problem A that gives the fit times of Table 2), `results_1d_r2.json` and
  `resA_gpu_r2.npz`, `resB_gpu_r2.npz` (Tables 2 and 3, Figures 1 and 2), `noise_variability_1d_r2.json`
  (Table 4), `fixedstart_B_r2.json`, `results_2d_M128_r2_clean.json` and `model_NMINN_M128_r2_clean.pt`
  (Problem C trained on noise-free snapshots), `coef_attribution_r2.json` (identified coefficients one at a
  time). The physics-informed runs were not repeated, and their entries are taken from the files above.

## Environment

Python 3.12, torch 2.13.0+cu130, numpy 2.5, scipy 1.18, matplotlib. Double precision throughout except the
2D physics-informed networks (single). All GPU numbers in the paper were measured on one NVIDIA RTX 4080
SUPER (driver 595.58) with no other process on the device. The 1D steppers of Problems A and B are pure
numpy/scipy and run on a CPU, torch is needed for the physics-informed baselines and for Problem C.

```
pip install -r requirements.txt
```

## Reproduce

Verification of the structural guarantees (positivity, conservation, agreement with dense exponentials,
autograd), 23 + 27 checks:

```
cd gpu
python test_ops.py cuda && python test_models2d.py cuda
```

Problem C, all surrogates, training on the paper's noise realization (seed 3), plus the resolution study:

```
python run2d.py --M 128 --n_macro 80 --ntrain 25 --K 6 --batch 4 --iters 700 --noise_seed 3 \
                --models NMINN CNN FNO CNNFLUX CNNPOS CNNPROJ --tag _r1 --gridstudy
python run2d.py --M 128 --n_macro 80 --ntrain 25 --K 6 --batch 4 --iters 700 --noise_seed 1 --models NMINN --tag _r1_noise1
python run2d.py --M 128 --n_macro 80 --ntrain 25 --K 6 --batch 4 --iters 700 --noise_seed 2 --models NMINN --tag _r1_noise2
python classical_baseline.py --tag _r1        # matched classical scheme, theta along the trajectory
python gradcheck_taxis.py --tag _r1           # gradient check
python ic_transfer.py --tag _r1               # initial data never seen in training
python timings2d.py --tag _r1                 # cost at four resolutions (idle device only)
python run2d_pinn.py --variant vanilla --seed 0   # then penalty, hardpos (sequentially, ~4 GB each)
python figs2d_r1.py --tag _r1
```

Problems A and B:

```
python select1d.py                                    # steppers, 32 fits each, selected on training loss
bash launch.sh                                        # 18 physics-informed runs (3 variants x 3 seeds x 2 problems)
python collect1d.py                                   # Tables A and B -> results_1d.json, resA_gpu.npz, resB_gpu.npz
for s in 21 22 23 24; do python select1d.py --models NMINN --noise_seed $s; done
for s in 21 22 23 24; do for p in A B; do
  python pinn_worker.py --problem $p --variant hardpos --seed 0 --noise_seed $s --out pinn_runs_noise$s; done; done
python noise_variability_1d.py --seeds 21 22 23 24    # noise-realization study
python figs1d.py
```

The whole sequence of the revision, in the order it was run, is `gpu/revision_runs.sh` (the paths in it are
those of the machine it ran on). Torch and numpy seeds are fixed in every script
(`torch.manual_seed(0)` for initializations, `manual_seed(3)` for the 2D noise, 7 and 11 for the 1D noise of
Problems A and B, restarts drawn from `default_rng(2026)`), and the NMINN results reproduce to every printed
digit on the same hardware. The convolutional baselines are reproducible to three digits in the error
because of cuDNN nondeterminism.

## Second revision (v1.1.0)

The first revision ran the 1D fits with SciPy's L-BFGS-B at its default stopping tolerances. These are
absolute, `ftol` about 2.2e-9 on the decrease of the objective and `gtol` 1e-5 on the projected gradient, and
the training losses of Problems A and B are of order 1e-4, so the default rule ended fits before convergence.
The second revision reruns every 1D fit of the learned steppers with both tolerances at 1e-30 and nothing else
changed (data, initializations from `default_rng(2026)`, horizons, cap of 400 iterations, selection on the
training loss), adds fits to the noise-free snapshots, which measure the part of a recovered coefficient that
the step causes, and trains the Problem C surrogate on noise-free snapshots.

```
cd gpu
python select1d_r2.py --jobs 7 --baseline        # -> select_1d_r2.json, sel_*_r2.npy (304 fits)
python select1d_r2.py --jobs 1 --baseline --paper_only --problems A --out select_1d_r2_timing.json
python select1d_r2.py --jobs 6 --baseline --out select_1d_r2_rainbow.json   # cross-check on a second machine
python fixedstart_b_r2.py                        # -> fixedstart_B_r2.json
python collect1d.py --tag _r2                    # Tables 2 and 3 -> results_1d_r2.json, res*_gpu_r2.npz
python noise_variability_1d.py --seeds 21 22 23 24 --tag _r2    # Table 4 -> noise_variability_1d_r2.json
python figs1d.py --tag _r2                       # -> figs/fig_nminn_A_r2.pdf, fig_nminn_B_r2.pdf
python run2d.py --M 128 --n_macro 80 --ntrain 25 --K 6 --batch 4 --iters 700 --noise 0.0 --noise_seed 3 \
                --models NMINN --tag _r2_clean   # -> results_2d_M128_r2_clean.json
python coef_attribution_r2.py --device cuda --tag _r1          # -> coef_attribution_r2.json
```

Environments of the second revision. The 1D fits of `select_1d_r2.json`, from which Tables 2-4 and
Figures 1-2 are generated, ran on an Apple M3 (Python 3.9.6, numpy 2.0.2, SciPy 1.13.1, 7 processes). The fit
times of Table 2 come from `select_1d_r2_timing.json`, the sweep of Problem A run sequentially in a single
process on an AMD Ryzen 9 9900X (Python 3.12.3, numpy 2.5.2, SciPy 1.18.0); the Problem C runs are on an
NVIDIA RTX 4080 SUPER (torch 2.13.0+cu130). All 304 fits were also rerun on the Ryzen machine, 6 processes,
in `select_1d_r2_rainbow.json`. Every NMINN fit agrees across the two machines within 0.2% in the rollout
error and 0.03% in the training loss. The unstructured stepper, nonconvex in its weights, ends on the Ryzen
machine in a different local minimum with a higher training loss: by 0.8% in Problem A, rollout error 1.21e-1
instead of 3.88e-2, and by 0.05% in Problem B, 3.01e-1 instead of 2.82e-1. The tables keep the fits with the
lower training loss, which is the selection rule of the paper and the choice less favorable to the NMINN.

## Tables and figures of the paper

```
python tables/make_tables.py            # -> tables/tables_r1.tex, the six tables of the revised paper
python gpu/figs1d.py                    # -> figs/fig_nminn_A.pdf, fig_nminn_B.pdf
python gpu/figs1d.py --tag _r2          # -> figs/fig_nminn_A_r2.pdf, fig_nminn_B_r2.pdf, second revision
python gpu/figs2d_r1.py --tag _r1       # -> figs/fig_nminn_C_*.pdf (needs the regenerated rollout fields)
```

## License

MIT, see `LICENSE`. Please cite the paper and the archived release (`CITATION.cff`).
