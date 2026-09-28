#!/usr/bin/env bash
# The eighteen physics-informed runs of Problems A and B: three variants (vanilla, penalty, hardpos) x
# three seeds x two problems, several at a time on one GPU, since none of them saturates the device.
# The projected variant of the tables is derived from the hardpos runs by collect1d.py.
# Each run takes 25-60 minutes on an RTX 4080 SUPER. Fit records and predictions go to pinn_runs/.
#
#   bash launch.sh [jobs]        default: 4 concurrent jobs; PYTHON=<interpreter> to choose one
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${1:-4}
export PY=${PYTHON:-python}
for p in A B; do for v in vanilla penalty hardpos; do for s in 0 1 2; do echo "${p}_${v}_${s}"; done; done; done |
  xargs -P "$JOBS" -I{} bash -c 'IFS=_ read -r p v s <<< "{}"; "$PY" pinn_worker.py --problem "$p" --variant "$v" --seed "$s" --out pinn_runs >> launch.log 2>&1'
