#!/bin/bash
# JOCS revision, round 1: every new measurement, run sequentially on an otherwise idle device.
# Usage on rainbow:  nohup bash revision_runs.sh > /dev/null 2>&1 &     (log: revision_runs.log)
set -u
cd /storage/internal_02/pdeluca-data/nminn12/code/gpu
PY=/home/pdeluca/venvs/deb/bin/python
LOG=revision_runs.log
{
echo "=== START $(date)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
echo "--- [1] verification suites"
$PY test_ops.py cuda | tail -3
$PY test_models2d.py cuda | tail -3
echo "--- [2] gradient check (trained weights of the submitted run)"
$PY gradcheck_taxis.py --device cuda --tag ""
echo "--- [3] classical baseline, true and identified coefficients, surrogates of the submitted run"
$PY classical_baseline.py --device cuda --tag ""
echo "--- [4] 2D retraining on the idle device: paper noise seed, all surrogates incl. hard-constrained"
$PY run2d.py --M 128 --n_macro 80 --ntrain 25 --K 6 --batch 4 --iters 700 --noise_seed 3 \
    --models NMINN CNN FNO CNNFLUX CNNPOS CNNPROJ --tag _r1 --gridstudy
echo "--- [5] 2D NMINN on two further noise realizations"
for s in 1 2; do
  $PY run2d.py --M 128 --n_macro 80 --ntrain 25 --K 6 --batch 4 --iters 700 --noise_seed $s \
      --models NMINN --tag _r1_noise$s
done
echo "--- [6] classical baseline against the retrained surrogates + gradient check at the retrained weights"
$PY classical_baseline.py --device cuda --tag _r1
$PY gradcheck_taxis.py --device cuda --tag _r1
echo "--- [7] transfer to new initial data"
$PY ic_transfer.py --device cuda --tag _r1
echo "--- [8] idle timings incl. the hard-constrained baselines"
$PY timings2d.py
echo "--- [9] 1D NMINN selection on four further noise realizations"
for s in 21 22 23 24; do $PY select1d.py --models NMINN --noise_seed $s; done
echo "--- [10] 1D positive-output PINN on the same realizations (4 concurrent, each is small)"
for s in 21 22 23 24; do for p in A B; do echo "${p}_${s}"; done; done | \
  xargs -P 4 -I{} bash -c 'IFS=_ read -r p s <<< "{}"; /home/pdeluca/venvs/deb/bin/python pinn_worker.py --problem $p --variant hardpos --seed 0 --noise_seed $s --adam 10000 --lbfgs 2000 --out pinn_runs_noise$s >> worker_noise.log 2>&1'
$PY noise_variability_1d.py --seeds 21 22 23 24
echo "=== END $(date)"
} >> $LOG 2>&1
