"""
Cross-dataset model transfer (review item 14): every saved cross-validation model of the source dataset is tested
on the whole other dataset. Everything is taken from the model's own source training fold: normalisation
statistics, tuned thresholds, gate thresholds and the INT8 calibration; nothing is tuned on the target dataset.
Needs the runs of the source dataset in results/cv_<source>/runs/ (.json, .npz, .keras).
Output: results/transfer/runs.csv (one row per model x source fold x seed) and summary.csv (mean ± std), with the
in-dataset results of the same runs next to them for comparison.
"""
import json

import keras
import numpy as np
import pandas as pd

import config as cfg
import cv
import deployment
import gate
import models
import preprocessing as pre
import training
from utils import print_columns, save_csv

OUT = cfg.RESULTS_DIR / "transfer"
SHOWN = ["Trial sensitivity", "Trial specificity", "Trial G-mean", "FP32 tuned Trial G-mean",
         "INT8 tuned Trial G-mean", "Gated FP32 tuned Trial G-mean", "Gated INT8 tuned Trial sensitivity",
         "Gated INT8 tuned Trial specificity", "Gated INT8 tuned Trial G-mean", "Gate fall trials kept, test"]


def run(names, int8=True):
    models.register_custom_layers()
    datasets = ("kfall", "sisfall")
    trials = {ds: pre.load_trials(ds) for ds in datasets}
    rows = []
    for src, dst in (("kfall", "sisfall"), ("sisfall", "kfall")):
        metas = [m for m in cv.saved_runs(src, cfg.EPOCHS, cv.out_dir_for(src))[0] if m["model"] in names]
        keras_files = {id(m): cv.out_dir_for(src) / "runs" / f"{m['model']}_fold{m['fold']}_seed{m['seed']}.keras"
                       for m in metas}
        metas = [m for m in metas if keras_files[id(m)].exists()]
        if not metas:
            print(f"{src} -> {dst}: no saved {src} runs with models for {names}")
            continue
        folds = cv.fold_subjects(sorted({t["subject"] for t in trials[src]}))
        target = pre.split(trials[dst], test_subjects={t["subject"] for t in trials[dst]})   # all of dst as test
        splits = {}
        for meta in sorted(metas, key=lambda m: (m["fold"], m["model"], m["seed"])):
            # the source run's own split: normalisation, threshold-tuning subjects and INT8 calibration windows
            key = (meta["fold"], json.dumps(meta.get("subjects"), sort_keys=True))
            if key not in splits:
                source = cv.run_split(trials[src], meta, folds)
                tune = pre.metric_data(source)
                splits[key] = (source, pre.normalize(target["test"], source["mean"], source["std"]),
                               gate.tune_filter(tune["gate_tr"], tune["y_tr"], tune["train"]),
                               dict(train=tune["train"], y_tr=tune["y_tr"], gate_tr=tune["gate_tr"],
                                    test=target["test"], y_te=target["y_te"], gate_te=target["gate_te"]))
            source, X_dst, gate_thr, data = splits[key]
            print(f"{src} -> {dst}: {meta['model']} fold {meta['fold']} seed {meta['seed']}")
            model = keras.models.load_model(keras_files[id(meta)])
            saved = cv.load_predictions(cv.out_dir_for(src), meta)     # predictions on the threshold-tuning windows
            p_dst = training.predict(model, X_dst)
            if int8:
                tfl, _, _ = deployment.quantize_int8(model, source["X_tr"])
                p8_dst = deployment.predict(tfl, X_dst)
            else:
                saved["p8_tr"], p8_dst = saved["p_tr"], p_dst
            m = training.all_metrics(saved["p_tr"], p_dst, saved["p8_tr"], p8_dst, data, gate_thr)
            rows.append({"Source": src, "Target": dst, "Model": models.display_name(meta["model"]),
                         "Fold": meta["fold"], "Seed": meta["seed"], **m})
    if not rows:
        return
    runs = save_csv(rows, OUT / "runs.csv")

    # in-dataset results of the same runs, for comparison
    own = {ds: cv.evaluate_runs(ds, cfg.EPOCHS, cv.out_dir_for(ds)) for ds in set(runs["Source"])}
    summary = []
    for (src, dst, model), g in runs.groupby(["Source", "Target", "Model"], sort=False):
        name = next(n for n in models.VARIANTS if models.display_name(n) == model)
        mine = own[src][own[src]["model"] == name]
        row = {"Model": f"{model}, trained on {src}", "Runs": len(g)}
        for c in SHOWN:
            row[f"{c}: tested on the other dataset"] = cv.mean_std(g[c])
            row[f"{c}: in-dataset (same runs)"] = cv.mean_std(mine[c]) if len(mine) > 1 else "-"
        summary.append(row)
    print_columns(save_csv(summary, OUT / "summary.csv"))
    print(f"Saved to {OUT}")
