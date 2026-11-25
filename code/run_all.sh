#!/usr/bin/env bash
# Runs every experiment for every model, then all final statistics. Resumable: finished runs are skipped,
# so after an interruption just start it again.
# Usage (inside tmux):  bash run_all.sh        Logs: results/logs/<job>.log
# Windows: run_all.ps1 does the same.
#
# Hardware split (set for an Intel i9-14900: 24 cores / 32 threads, 64 GB, machine used only for this):
# 9 parallel jobs (fixed split, early stopping, one cross-validation job per model) x CPU_THREADS=4.
# These models are tiny, so many processes with few threads each use the CPU better than few wide processes.
# Each job needs roughly 2-4 GB RAM (SisFall is the larger dataset), about 30 GB in total.
# Other machines: CPU_THREADS ~ logical threads / 9 (e.g. CPU_THREADS=2 bash run_all.sh on a 20-thread CPU).
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=-1               # CPU only (also set by overall.py when config.USE_GPU is False)
export CPU_THREADS=${CPU_THREADS:-4}
mkdir -p results/logs
ALL="tinycnn tinytcn tinyssm cnnlstm tinyeegnet tinytransformer musci2021"
PROTOCOL=$(python -c "import config; print(config.THRESHOLD_PROTOCOL)")
echo "Threshold protocol: $PROTOCOL    CPU threads per job: $CPU_THREADS"
python -c "import tensorflow as tf;g=tf.config.list_physical_devices('GPU'); print('GPUs visible to TensorFlow:', g if g else 'none (CPU only)')"

# 1. data self-check + data manifest (before the parallel jobs)
python overall.py check || exit 1
python overall.py check --dataset sisfall || exit 1

# 2. all training experiments in parallel
pids=(); names=()
job() {
    local name=$1; shift
    bash -c "$*" > "results/logs/$name.log" 2>&1 &
    pids+=($!); names+=("$name")
    echo "$(date '+%F %T')  started $name  (log: results/logs/$name.log)"
}
# slowest first (LSTMs), so they get the CPU from the start
for model in cnnlstm musci2021 tinytransformer tinyssm tinytcn tinycnn tinyeegnet; do
    job "cv_$model" "python overall.py cv --models $model && python overall.py cv --dataset sisfall --models $model"
done
if [ "$PROTOCOL" = "validation" ]; then
    job train "python overall.py train --models $ALL --out_dir results/fixed_valthr"
else
    job train "python overall.py train --models $ALL && python overall.py deploy --models $ALL"
fi
job earlystop "python overall.py earlystop"

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "$(date '+%F %T')  finished ${names[$i]}"
    else echo "$(date '+%F %T')  FAILED ${names[$i]}, see results/logs/${names[$i]}.log"; failed=1; fi
done
if [ $failed -ne 0 ]; then
    echo "Some jobs failed. Fix the problem and run 'bash run_all.sh' again (finished runs are skipped)."
    exit 1
fi

# 3. final statistics over all saved runs
python overall.py report                  2>&1 | tee results/logs/report_kfall.log
python overall.py report --dataset sisfall 2>&1 | tee results/logs/report_sisfall.log
python overall.py transfer                2>&1 | tee results/logs/transfer.log
python overall.py earlystop --report      2>&1 | tee results/logs/earlystop_report.log
python overall.py analyze                 2>&1 | tee results/logs/analyze.log
python plot_sa06.py                       2>&1 | tee results/logs/plot_sa06.log
echo "$(date '+%F %T')  ALL DONE. Copy results/ back."
