#!/usr/bin/env bash
# Accuracy re-check for CAJQ-Full (see scripts/exp_cajq_full_accuracy.py).
#   bash scripts/run_cajq_full_accuracy.sh smoke   # ~3 min, tiny settings, checks for errors
#   bash scripts/run_cajq_full_accuracy.sh full    # ~1-1.5 h on Apple MPS, runs in background
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mode="${1:-smoke}"
if [ "$mode" = "smoke" ]; then
  python scripts/exp_cajq_full_accuracy.py --seeds 42 --n-samples 4 --ft-steps 3 \
    --context-lengths 1024 --output results/scaled/smoke_cajq_full_acc.json 2>&1 | tail -45
elif [ "$mode" = "full" ]; then
  # caffeinate -i keeps the Mac from idle-sleeping while the job runs
  nohup caffeinate -i python scripts/exp_cajq_full_accuracy.py --context-lengths 1024 2048 4096 \
    > results/scaled/exp_cajq_full_accuracy.log 2>&1 &
  echo "started full run, pid $!  (log: results/scaled/exp_cajq_full_accuracy.log)"
else
  echo "usage: bash scripts/run_cajq_full_accuracy.sh [smoke|full]"; exit 1
fi
