# Numerical-method-informed neural networks (NMINN)

Reference implementation and reproduction package for

> P. De Luca, *Numerical-method-informed neural networks, structure-preserving discretizations as
> architectural inductive bias*, Journal of Computational Science, manuscript JOCSCI-D-26-02458
> (revised version).

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

## Tables and figures of the paper

```
python tables/make_tables.py            # -> tables/tables_r1.tex, the six tables of the revised paper
python gpu/figs1d.py                    # -> figs/fig_nminn_A.pdf, fig_nminn_B.pdf
python gpu/figs2d_r1.py --tag _r1       # -> figs/fig_nminn_C_*.pdf (needs the regenerated rollout fields)
```

## License

MIT, see `LICENSE`. Please cite the paper and the archived release (`CITATION.cff`).
