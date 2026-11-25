"""
The P1 / P2 experiments on top of the finished main runs, scheduled to keep every core busy (made for an i9-14900:
24 cores / 32 threads, 64 GB). No main run is retrained, and every part is resumable (finished runs are skipped), so
after an interruption just start it again.

  stage 1  about 170 work units (one dataset / model / fold each), longest first, on a pool of single-threaded
           workers: one finished unit is replaced by the next at once. A unit only starts when the memory estimates
           of the running units plus its own fit the budget (total memory - reserve) and, on Linux / Windows, the
           free memory reported by the OS (minus the units started in the last 2 minutes) leaves the reserve free.
           A queued unit that does not fit lets a smaller one go first, so small KFall units fill the cores while
           the large SisFall units wait for memory.
  stage 2  all reports and analyses, on the same pool
  stage 3  alone on the idle machine: latency measurement (pinned to one P-core), MCU files, figures

Usage:  python run_extra.py [--workers N] [--reserve_gb G] [--stages 1 2 3]      (run_extra.sh calls this)
Logs:   results/logs/extra/<unit>.log, progress on the console
Memory estimates (peak GB per unit) were measured on single-threaded units: SisFall training 3.2, SisFall
quantization 2.2; KFall units need about 60 % of that.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import config as cfg
import models

PY = [sys.executable, "-u", "overall.py"]
LOGS = cfg.RESULTS_DIR / "logs" / "extra"
DATASETS = ("sisfall", "kfall")
GB = {("sisfall", "train"): 3.4, ("kfall", "train"): 2.0, ("sisfall", "eval"): 2.4, ("kfall", "eval"): 1.4}
# SisFall training units fill the memory before the cores (about 17 fit in 64 GB), so they get 2 threads each; every
# other unit runs single-threaded, which gives these tiny models the most work per core
TRAIN_THREADS = {"sisfall": 2, "kfall": 1}
SLOWER = {1: 1.5, 2: 1.15}   # unit vs the main runs (4 threads) at 1 / 2 threads; only used to order the units
QUANT_MIN = {"musci2021": 10.0, "cnnlstm": 8.0, "tinytransformer": 2.0}   # minutes per SisFall run (KFall: half)
# one thread for each numeric library in every unit: the pool, not the libraries, uses the cores
SINGLE = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


@dataclass
class Unit:
    name: str
    args: list
    hours: float          # estimated duration, for the longest-first order
    gb: float             # estimated peak memory
    threads: int = 1


def stamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def memory_gb():
    """(total, available) physical memory in GB, (None, None) if unknown."""
    try:
        info = {}
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                key, value = line.split(":", 1)
                info[key] = int(value.split()[0]) / 2 ** 20
        return info["MemTotal"], info["MemAvailable"]
    except (OSError, KeyError, ValueError):
        pass
    if os.name == "nt":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                        ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                        ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
        s = Status()
        s.length = ctypes.sizeof(s)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return s.total / 2 ** 30, s.available / 2 ** 30
    return None, None


def run_minutes():
    """Mean training minutes of a main run per (dataset, model), read from the saved runs."""
    out = {}
    for ds in DATASETS:
        for p in (cfg.RESULTS_DIR / f"cv_{ds}" / "runs").glob("*_seed*.json"):
            if not p.name.endswith(".config.json"):
                m = json.loads(p.read_text())
                out.setdefault((ds, m["model"]), []).append(m.get("training_minutes") or 15.0)
    return {k: sum(v) / len(v) for k, v in out.items()}


def stage1():
    mins = run_minutes()
    run_h = lambda ds, m: mins.get((ds, m), 15.0) * SLOWER[TRAIN_THREADS[ds]] / 60     # hours of one 200-epoch run
    seeds, folds = len(cfg.CV_SEEDS), range(cfg.CV_FOLDS)
    units = []
    for ds in DATASETS:
        train = dict(gb=GB[ds, "train"], threads=TRAIN_THREADS[ds])
        for m in cfg.ES_MODELS:              # item 8: up to 200 epochs on 85 % of the training subjects, 2 evaluations
            for f in folds:
                units.append(Unit(f"es_{ds}_{m}_fold{f}", ["es", "--dataset", ds, "--models", m, "--folds", str(f),
                                                           "--no_report"],
                                  seeds * (0.75 * run_h(ds, m) + 0.05), **train))
        for m in models.MAIN:                # items 1, 2, 15: quantization variants, no training
            for f in folds:
                per_run = QUANT_MIN.get(m, 2.5) * (1 if ds == "sisfall" else 0.5) / 60
                units.append(Unit(f"quant_{ds}_{m}_fold{f}", ["quant", "--dataset", ds, "--models", m, "--folds", str(f),
                                                              "--no_report"], seeds * per_run, GB[ds, "eval"]))
        for m in cfg.QAT_MODELS:             # item 15: a QAT epoch costs about 3 normal ones, plus the control
            for f in folds:
                units.append(Unit(f"qat_{ds}_{m}_fold{f}", ["qat", "--dataset", ds, "--models", m, "--folds", str(f),
                                                            "--no_report"],
                                  4 * cfg.QAT_EPOCHS / cfg.EPOCHS * run_h(ds, m) + 0.05, **train))
        for f in folds:                      # item 14
            units.append(Unit(f"prune_{ds}_fold{f}", ["prune", "--dataset", ds, "--folds", str(f), "--no_report"],
                              len(cfg.PRUNE_SPARSITIES) * (cfg.PRUNE_FINETUNE_EPOCHS / cfg.EPOCHS * run_h(ds, "tinytcn")
                                                           + 0.03), **train))
        units.append(Unit(f"int8diag_{ds}", ["int8diag", "--dataset", ds], 1.0 if ds == "sisfall" else 0.5,
                          GB[ds, "eval"]))   # item 1
    for v in models.SWEEP_TINYTCN:           # item 5: KFall, seed 0
        for f in folds:
            units.append(Unit(f"sweep_{v}_fold{f}", ["cv", "--models", v, "--folds", str(f), "--seeds",
                                                     str(cfg.CV_SEEDS[0]), "--out_dir",
                                                     str(cfg.RESULTS_DIR / "sweep_tinytcn_kfall"), "--no_report"],
                              run_h("kfall", "tinytcn"), GB["kfall", "train"], TRAIN_THREADS["kfall"]))
    units.append(Unit("transfer", ["transfer", "--models", *models.MAIN], 2.5, 3.6))      # item 12
    return units


def stage2():
    units = []
    for ds in DATASETS:
        units += [Unit(f"report_{ds}", ["report", "--dataset", ds], 0.3, GB[ds, "eval"]),           # item 4
                  Unit(f"quant_report_{ds}", ["quant", "--dataset", ds, "--report"], 0.3, GB[ds, "eval"]),
                  Unit(f"qat_report_{ds}", ["qat", "--dataset", ds, "--report"], 0.1, GB[ds, "eval"]),
                  Unit(f"es_report_{ds}", ["es", "--dataset", ds, "--report"], 0.4, GB[ds, "eval"]),
                  Unit(f"prune_report_{ds}", ["prune", "--dataset", ds, "--report"], 0.1, GB[ds, "eval"])]
    units += [Unit("sweep_report", ["report", "--out_dir", str(cfg.RESULTS_DIR / "sweep_tinytcn_kfall")], 0.1, 1.5),
              Unit("analyze", ["analyze"], 0.8, 5.0),                       # items 3, 6, 7, 9, 13, 16
              Unit("earlystop_report", ["earlystop", "--report"], 0.01, 1.0)]
    return units


def run_pool(units, workers, threads, reserve_gb, label):
    """Runs the units longest first on up to `workers` processes and `threads` threads within the memory budget.
    Returns the names of the failed units."""
    LOGS.mkdir(parents=True, exist_ok=True)
    total, _ = memory_gb()
    budget = (total or 64.0) - reserve_gb
    queue = sorted(units, key=lambda u: -u.hours)
    running, failed, done, t_start = {}, [], 0, time.time()
    print(f"{stamp()}  {label}: {len(queue)} units, up to {workers} processes / {threads} threads at once, memory "
          f"budget {budget:.0f} GB (estimated total {sum(u.hours for u in units):.0f} unit-hours)", flush=True)
    while queue or running:
        while queue and len(running) < workers:
            now = time.time()
            used = sum(u.gb for u, _, _ in running.values())
            busy = sum(u.threads for u, _, _ in running.values())
            recent = sum(u.gb for u, t0, _ in running.values() if now - t0 < 120)
            _, available = memory_gb()

            def fits(u):
                if not running:
                    return True
                return (used + u.gb <= budget and busy + u.threads <= threads
                        and (available is None or available - recent - u.gb >= reserve_gb))

            i = next((i for i, u in enumerate(queue) if fits(u)), None)
            if i is None:
                break
            u = queue.pop(i)
            env = {**os.environ, **SINGLE, "CPU_THREADS": str(u.threads), "CUDA_VISIBLE_DEVICES": "-1",
                   "PYTHONUNBUFFERED": "1"}
            log = open(LOGS / f"{u.name}.log", "w", encoding="utf-8")
            p = subprocess.Popen(PY + u.args, cwd=cfg.CODE_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[p] = (u, now, log)
            print(f"{stamp()}  started  {u.name}  [{len(running)} running, {len(queue)} queued, "
                  f"{used + u.gb:.0f} GB estimated]", flush=True)
        time.sleep(3)
        for p in [p for p in running if p.poll() is not None]:
            u, t0, log = running.pop(p)
            log.close()
            done += 1
            if p.returncode != 0:
                failed.append(u.name)
            print(f"{stamp()}  {'finished' if p.returncode == 0 else 'FAILED  '} {u.name} ({(time.time() - t0) / 60:.0f} "
                  f"min)  [{done} done, {len(running)} running, {len(queue)} queued]", flush=True)
    print(f"{stamp()}  {label} done in {(time.time() - t_start) / 3600:.1f} h"
          + (f"; FAILED: {' '.join(failed)} (logs in {LOGS})" if failed else ""), flush=True)
    return failed


def simulate(units, workers, threads, budget_gb):
    """Duration of the pool from the unit estimates (same order, thread and memory rules, no OS memory check)."""
    queue = sorted(units, key=lambda u: -u.hours)
    running, clock = [], 0.0                                   # (end time, unit)
    while queue or running:
        while queue and len(running) < workers:
            used = sum(u.gb for _, u in running)
            busy = sum(u.threads for _, u in running)
            i = next((i for i, u in enumerate(queue)
                      if not running or (used + u.gb <= budget_gb and busy + u.threads <= threads)), None)
            if i is None:
                break
            u = queue.pop(i)
            running.append((clock + u.hours, u))
        running.sort(key=lambda r: r[0])
        clock, _ = running.pop(0)
    return clock


def run_alone(name, args, threads=None):
    """One command with the whole machine (stage 3), logged like the units."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONUNBUFFERED": "1"}
    if threads is None:
        env.pop("CPU_THREADS", None)
    else:
        env["CPU_THREADS"] = str(threads)
    print(f"{stamp()}  {name}", flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        code = subprocess.run(PY + args, cwd=cfg.CODE_DIR, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    print(f"{stamp()}  {'finished' if code == 0 else 'FAILED  '} {name}", flush=True)
    return code == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                        help="processes at once (default: logical CPUs - 2)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4,
                        help="threads of all running units together (default: logical CPUs)")
    parser.add_argument("--reserve_gb", type=float, default=4.0, help="memory kept free for the system")
    parser.add_argument("--stages", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--only", nargs="+", default=None, help="run only units whose name contains one of these")
    parser.add_argument("--dry_run", action="store_true", help="list the units and estimate the duration, run nothing")
    args = parser.parse_args()
    total, available = memory_gb()
    print(f"{stamp()}  {os.cpu_count()} logical CPUs, memory {total or 0:.0f} GB ({available or 0:.0f} GB free), "
          f"results in {cfg.RESULTS_DIR}", flush=True)
    pick = lambda units: [u for u in units if not args.only or any(s in u.name for s in args.only)]
    stages = {1: pick(stage1()), 2: pick(stage2())}
    if args.dry_run:
        budget = (total or 64.0) - args.reserve_gb
        for k, units in stages.items():
            if k in args.stages:
                for u in sorted(units, key=lambda u: -u.hours):
                    print(f"  stage {k}  {u.name:42s} {u.hours:5.2f} h  {u.gb:.1f} GB")
                print(f"stage {k}: {len(units)} units, {sum(u.hours for u in units):.0f} unit-hours, estimated "
                      f"{simulate(units, args.workers, args.threads, budget):.1f} h with {args.workers} processes, "
                      f"{args.threads} threads and {budget:.0f} GB")
        return
    if not args.only:
        for ds in DATASETS:
            n = sum(1 for p in (cfg.RESULTS_DIR / f"cv_{ds}" / "runs").glob("*_seed*.json")
                    if not p.name.endswith(".config.json"))
            if n < len(models.MAIN) * cfg.CV_FOLDS * len(cfg.CV_SEEDS):
                sys.exit(f"results/cv_{ds} has {n} runs: run the main experiments (run_all.sh) first.")
    ok = True
    if 1 in args.stages and run_pool(stages[1], args.workers, args.threads, args.reserve_gb, "stage 1 (experiments)"):
        sys.exit("Stage 1 had failures: fix them and start again (finished parts are skipped).")
    if 2 in args.stages:
        ok &= not run_pool(stages[2], args.workers, args.threads, args.reserve_gb, "stage 2 (reports and analyses)")
    if 3 in args.stages:                                       # alone: the latency measurement needs an idle machine
        ok &= run_alone("deploy", ["deploy", "--models", *models.MAIN])               # item 10
        ok &= run_alone("mcu", ["mcu", "--models", "tinytcn", "tinycnn"])             # items 11, 17
        ok &= run_alone("figures", ["figures"])
    print(f"{stamp()}  {'ALL DONE. Copy results/ back.' if ok else 'Done with failures, see the logs above.'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
