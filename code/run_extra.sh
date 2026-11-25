#!/usr/bin/env bash
# The P1 / P2 experiments on top of the finished main runs (run_all.sh); no main run is retrained.
# Runs run_extra.py: ~170 work units longest first on a pool of single-threaded workers (all cores busy, memory-aware),
# then all reports on the same pool, then the latency measurement alone on the idle machine, MCU files and figures.
# Resumable: finished parts are skipped, so after an interruption just start it again.
# Usage (inside tmux):  bash run_extra.sh        options: --workers N --reserve_gb G --stages 1 2 3
# Logs: results/logs/extra/<unit>.log, progress in results/logs/extra_scheduler.log
set -o pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=-1
mkdir -p results/logs
python -u run_extra.py "$@" 2>&1 | tee -a results/logs/extra_scheduler.log
