"""
Review analyses computed from cached data and saved results (no retraining), written to results/analysis/.
Model predictions come from the cross-validation runs in results/cv_<dataset>/runs/; parts that need them are
skipped for a dataset without runs.
  1    gate_ops.csv                     gate cost per window, operation by operation
  2-3  ops_per_window.csv               average ops per window = gate ops + pass rate x MACs, reduction factors
  4    macs_by_layer.csv                per-layer MACs of every registered model variant
  5-6  sizes.csv                        parameter bytes, .tflite constant data / file size, peak activation memory
  7    gate_detector.csv                the gate alone as a fall detector, per fold + mean ± std
                                        (trial balanced accuracy) and the filter thresholds of the main setting
  8    gate_recall_sweep.csv            gate as a filter at several training recall levels (GATE_RECALL_SWEEP)
  9    gate_transfer.csv                filter thresholds tuned on one dataset applied to the other
  10   misses_by_fall_type.csv          missed fall trials of the main setting: by the gate or by the model
       false_alarms_by_adl_type.csv     ADL trials passed by the gate and false alarms, per ADL type (D01-D21)
       false_alarms_by_subject.csv      the same per subject
       metrics_by_subject_group.csv     sensitivity / specificity / G-mean per subject group (SisFall young / elderly)
  16   gate_ablation.csv                gate with 1, 2 or 3 of its conditions and 1 or 2 windows in a row: pass
                                        rates, fall trials kept, ops per window, and the gated INT8 results of
                                        config.GATE_ABLATION_MODELS
  figures/gate_recall_sweep.csv, figures/gate_ablation.csv   tidy data for plotting
The gate filter is tuned on the subjects given by config.THRESHOLD_PROTOCOL (as in cross-validation).
"""
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
from utils import print_table, save_csv

OUT = cfg.RESULTS_DIR / "analysis"
UNGATED_BASELINE = 153616          # TinyCNN, same padding, no gate
MAIN = "Gated INT8 tuned"          # main setting: gate + threshold tuned on the training data + full INT8


def mean_std(x):
    x = np.asarray(x, float)
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}" if len(x) > 1 else f"{x.mean():.2f}"


def fold_splits(dataset):
    """Per fold: all training subjects / test subjects (the gate as a detector is tuned on all training subjects)."""
    trials = pre.load_trials(dataset)
    folds = cv.fold_subjects(sorted({t["subject"] for t in trials}))
    return trials, [pre.split(trials, f) for f in folds]


def fold_filter_splits(dataset):
    """Per fold: the subjects the gate filter is tuned on under config.THRESHOLD_PROTOCOL (all training subjects,
    or the held-out threshold subjects chosen exactly as in cross-validation) and the test subjects."""
    trials = pre.load_trials(dataset)
    everyone = {t["subject"] for t in trials}
    out = []
    for f, test in enumerate(cv.fold_subjects(sorted(everyone))):
        train = sorted(everyone - set(test))
        tune = pre.threshold_subjects(train, f) if cfg.THRESHOLD_PROTOCOL == "validation" else train
        tune_trials = [t for t in trials if t["subject"] in set(tune)]
        ev = pre.split(trials, test)
        out.append(dict(train=tune_trials, y_tr=np.concatenate([t["y"] for t in tune_trials]),
                        gate_tr=np.concatenate([t["gate"] for t in tune_trials]),
                        test=ev["test"], y_te=ev["y_te"], gate_te=ev["gate_te"]))
    return trials, out


def variant_macs():
    return {n: deployment.count_macs(deployment.convert(models.build(n, len(cfg.CHANNELS), cfg.WINDOW)))
            for n in models.VARIANTS}


class FilterCache:
    """Gate filter thresholds per (threshold-tuning subjects, recall, conditions, windows in a row): every run of a
    fold tunes on the same subjects, so each combination is tuned once."""

    def __init__(self):
        self.cache = {}

    def get(self, data, recall=cfg.GATE_TRIAL_RECALL, used=cfg.GATE_CONDITIONS, k=cfg.GATE_CONSECUTIVE):
        subjects = tuple(sorted({t["subject"] for t in data["train"]}))
        key = (subjects, recall, tuple(used), k)
        if key not in self.cache:
            self.cache[key] = gate.tune_filter(data["gate_tr"], data["y_tr"], data["train"], recall=recall,
                                               grid=gate.subset_grid(cfg.GATE_FILTER_GRID, used), k=k)
        return self.cache[key]


def gated_int8(p, data, thresholds, k=cfg.GATE_CONSECUTIVE):
    """Main setting of one saved run with the given gate: gated INT8, threshold tuned on the tuning trials."""
    pass_tr = gate.passes(data["gate_tr"], data["train"], thresholds, k)
    pass_te = gate.passes(data["gate_te"], data["test"], thresholds, k)
    return training.tuned_metrics(p["p8_tr"] * pass_tr, p["p8_te"] * pass_te, data)


# ---- 1 ----
def gate_ops():
    rows, full, stream = gate.ops_per_window()
    save_csv(rows, OUT / "gate_ops.csv")
    print_table(list(rows[0]), [[str(v) for v in r.values()] for r in rows])
    return full, stream


# ---- 4 ----
def macs_by_layer():
    rows = []
    for name in models.VARIANTS:
        for op, out_shape, w_shape, per_out, macs in deployment.macs_by_op(
                deployment.convert(models.build(name, len(cfg.CHANNELS), cfg.WINDOW))):
            rows.append({"Model": models.display_name(name), "Op": op, "Output shape": out_shape,
                         "Weight shape": w_shape, "MACs per output": per_out, "MACs": macs})
        rows.append({"Model": models.display_name(name), "Op": "TOTAL", "MACs": sum(r["MACs"] for r in rows
                     if r["Model"] == models.display_name(name) and r["Op"] != "TOTAL")})
    df = save_csv(rows, OUT / "macs_by_layer.csv")
    print_table(list(df.columns), df.fillna("").astype(str).values.tolist())


# ---- 5-6 ----
def sizes():
    """Trained fixed-split models from results/<model>/ where available, otherwise freshly built ones (sizes do
    not depend on the weight values). INT8 calibration uses the fixed KFall split's training data."""
    models.register_custom_layers()
    X_tr = pre.split_and_normalize(pre.load_trials("kfall"))["X_tr"]
    rows = []
    for name in models.VARIANTS:
        path = cfg.RESULTS_DIR / name / f"{name}.keras"
        model = keras.models.load_model(path) if path.exists() else models.build(name, len(cfg.CHANNELS), cfg.WINDOW)
        fp32, (int8, mode, _) = deployment.convert(model), deployment.quantize_int8(model, X_tr)
        n = model.count_params()
        rows.append({"Model": models.display_name(name), "Weights": "trained" if path.exists() else "untrained",
                     "Parameters": n, "Parameter bytes FP32 (B, x4)": 4 * n, "Parameter bytes INT8 (B, x1)": n,
                     "Constant data .tflite FP32 (B)": deployment.constant_bytes(fp32),
                     "Constant data .tflite INT8 (B)": deployment.constant_bytes(int8),
                     ".tflite file FP32 (B)": len(fp32), ".tflite file INT8 (B)": len(int8), "INT8 mode": mode,
                     "Peak activation FP32 (B)": deployment.peak_activation_bytes(fp32),
                     "Peak activation INT8 (B)": deployment.peak_activation_bytes(int8),
                     "Operators in graph": len(deployment.graph_info(fp32)[0])})
    df = save_csv(rows, OUT / "sizes.csv")
    print_table(["Metric", *df["Model"]], [[c, *df[c].astype(str)] for c in df.columns if c != "Model"])


# ---- 2-3 ----
def ops_per_window(datasets, gate_ops_full):
    macs = variant_macs()
    rows = []
    for ds in datasets:
        _, splits = fold_filter_splits(ds)
        stats = [gate.filter_stats(d["gate_te"], d["y_te"], d["test"],
                                   gate.tune_filter(d["gate_tr"], d["y_tr"], d["train"])) for d in splits]
        for scope, key in (("all windows", "Pass rate, all windows"), ("ADL windows", "Pass rate, ADL windows")):
            p = np.mean([s[key] for s in stats]) / 100
            for name in models.DEFAULT + [n for n in models.VARIANTS if n not in models.DEFAULT]:
                m = macs[name]
                avg = gate_ops_full + p * m
                rows.append({"Dataset": ds, "Windows": scope, "Model": models.display_name(name), "MACs": m,
                             "Gate ops": gate_ops_full, "Pass rate (%)": round(100 * p, 2),
                             "Avg ops / window": round(avg, 1),
                             "Reduction vs ungated TinyCNN (x)": round(UNGATED_BASELINE / avg, 2),
                             "Model factor (x) = 153616 / MACs": round(UNGATED_BASELINE / m, 2),
                             "Gate factor (x) = MACs / avg ops": round(m / avg, 2)})
    df = save_csv(rows, OUT / "ops_per_window.csv")
    print_table(list(df.columns), df.astype(str).values.tolist())


# ---- 7 ----
def gate_detector(datasets):
    """The gate alone decides: with detector tuning, and with the filter thresholds of the main
    setting (>= GATE_TRIAL_RECALL of the training fall trials kept, fewest training windows passing)."""
    rows = []
    for ds in datasets:
        trials, splits = fold_splits(ds)
        setups = [(f"fold {i}", d) for i, d in enumerate(splits)]
        if ds == "kfall":
            setups.append(("fixed split", pre.split(trials)))
        for mode, tune in (("detector (trial balanced accuracy)", gate.tune_detector),
                           (f"filter thresholds (>= {100 * cfg.GATE_TRIAL_RECALL:g} % training fall trials kept)",
                            gate.tune_filter)):
            for label, d in setups:
                thr = tune(d["gate_tr"], d["y_tr"], d["train"])
                m = training.evaluate(gate.passes(d["gate_te"], d["test"], thr).astype(float), d["y_te"], d["test"])[1]
                rows.append({"Dataset": ds, "Thresholds": mode, "Split": label, "acc (g)": thr[0],
                             "gyr (deg/s)": thr[1], "tilt (deg)": thr[2], **{k: round(v, 2) for k, v in m.items()}})
            f = [r for r in rows if r["Dataset"] == ds and r["Thresholds"] == mode and r["Split"].startswith("fold")]
            rows.append({"Dataset": ds, "Thresholds": mode, "Split": "5-fold mean ± std",
                         **{k: mean_std([r[k] for r in f]) for k in f[0] if k not in ("Dataset", "Thresholds", "Split")}})
    df = save_csv(rows, OUT / "gate_detector.csv")
    print_table(list(df.columns), df.astype(str).values.tolist())


# ---- 8 ----
def recall_sweep(datasets):
    """Gate filter at each training recall target: thresholds, pass rates, fall trials kept, and the gated INT8
    tuned results of every model with saved runs (gate tuned on each run's threshold-tuning subjects)."""
    rows, tidy = [], []
    for ds in datasets:
        _, splits = fold_filter_splits(ds)
        metas, ctx = cv.saved_runs(ds, cfg.EPOCHS, cv.out_dir_for(ds))
        filters = FilterCache()
        preds = {id(m): cv.load_predictions(cv.out_dir_for(ds), m) for m in metas}
        for r in cfg.GATE_RECALL_SWEEP:
            thr = [filters.get(d, recall=r) for d in splits]
            train_stats = [gate.filter_stats(d["gate_tr"], d["y_tr"], d["train"], t) for d, t in zip(splits, thr)]
            test_stats = [gate.filter_stats(d["gate_te"], d["y_te"], d["test"], t) for d, t in zip(splits, thr)]
            row = {"Dataset": ds, "Training recall target (%)": 100 * r,
                   "Thresholds per fold (acc, gyr, tilt)": "; ".join(f"({a:g}, {g:g}, {t:g})" for a, g, t in thr),
                   "Train fall trials kept (%)": mean_std([s["Fall trials kept"] for s in train_stats]),
                   **{f"Test {k.lower()} (%)": mean_std([s[k] for s in test_stats]) for k in test_stats[0]}}
            for k in test_stats[0]:
                v = [s[k] for s in test_stats]
                tidy.append({"Dataset": ds, "Recall target (%)": 100 * r, "Series": "gate", "Metric": k,
                             "mean": np.mean(v), "std": np.std(v, ddof=1)})
            for name in models.VARIANTS:                 # gated model results from the saved runs
                runs = [m for m in metas if m["model"] == name]
                if not runs:
                    continue
                res = []
                for m in runs:
                    data, _ = ctx.get(m)
                    g = gated_int8(preds[id(m)], data, filters.get(data, recall=r))
                    res.append({f"{MAIN} {k}": g[k] for k in ("Trial sensitivity", "Trial specificity", "Trial G-mean")})
                row[f"{models.display_name(name)} runs"] = len(res)
                for k in ("Trial sensitivity", "Trial specificity", "Trial G-mean"):
                    v = [x[f"{MAIN} {k}"] for x in res]
                    row[f"{models.display_name(name)} {MAIN} {k}"] = mean_std(v)
                    tidy.append({"Dataset": ds, "Recall target (%)": 100 * r, "Series": models.display_name(name),
                                 "Metric": f"{MAIN} {k}", "mean": np.mean(v), "std": np.std(v, ddof=1) if len(v) > 1 else np.nan})
            rows.append(row)
    df = save_csv(rows, OUT / "gate_recall_sweep.csv")
    save_csv(tidy, cv.figures_dir() / f"gate_recall_sweep{'_valthr' if cfg.THRESHOLD_PROTOCOL == 'validation' else ''}.csv")
    for ds in datasets:
        part = df[df["Dataset"] == ds].drop(columns="Dataset")
        print(f"\n--- {ds} ---")
        print_table(["Metric", *part["Training recall target (%)"].astype(str)],
                    [[c, *part[c].astype(str)] for c in part.columns if c != "Training recall target (%)"])


# ---- 9 ----
def transfer():
    data = {ds: pre.split(pre.load_trials(ds), test_subjects=[]) for ds in ("kfall", "sisfall")}  # all subjects
    rows = []
    for src in data:
        thr = gate.tune_filter(data[src]["gate_tr"], data[src]["y_tr"], data[src]["train"])
        for dst in data:
            s = gate.filter_stats(data[dst]["gate_tr"], data[dst]["y_tr"], data[dst]["train"], thr)
            rows.append({"Thresholds tuned on": src, "Applied to": dst, "acc (g)": thr[0], "gyr (deg/s)": thr[1],
                         "tilt (deg)": thr[2], **{f"{k} (%)": round(v, 2) for k, v in s.items()}})
    df = save_csv(rows, OUT / "gate_transfer.csv")
    print_table(list(df.columns), df.astype(str).values.tolist())


def run_outcomes(ds):
    """Per saved run: (meta, per-test-trial table) with the decisions of the main setting (gated INT8, tuned) and
    of ungated FP32 tuned, and whether the gate passed any window of the trial."""
    metas, ctx = cv.saved_runs(ds, cfg.EPOCHS, cv.out_dir_for(ds))
    for meta in metas:
        data, gthr = ctx.get(meta)
        p = cv.load_predictions(cv.out_dir_for(ds), meta)
        pass_tr = gate.passes(data["gate_tr"], data["train"], gthr)
        pass_te = gate.passes(data["gate_te"], data["test"], gthr)
        main_max, is_fall = training.trial_scores(p["p8_te"] * pass_te, data["test"])
        fp32_max, _ = training.trial_scores(p["p_te"], data["test"])
        gate_any = training.trial_scores(pass_te.astype(np.float32), data["test"])[0] > 0
        yield meta, pd.DataFrame({
            "Subject": [t["subject"] for t in data["test"]], "Group": [pre.subject_group(t["subject"]) for t in data["test"]],
            "Activity": [pre.activity_code(t) for t in data["test"]], "Fall": is_fall, "Gate passed": gate_any,
            "Main": main_max >= training.select_threshold(p["p8_tr"] * pass_tr, data["train"]),
            "FP32 tuned": fp32_max >= training.select_threshold(p["p_tr"], data["train"])})


# ---- 10 and false alarms ----
def outcome_tables(datasets):
    """misses_by_fall_type.csv, false_alarms_by_adl_type.csv, false_alarms_by_subject.csv and
    metrics_by_subject_group.csv (SisFall young / elderly) for the main setting and ungated FP32 tuned."""
    misses, fa_type, fa_subject, groups = [], [], [], []
    for ds in datasets:
        tables = [(m, t.assign(model=m["model"], run=f"{m['fold']}-{m['seed']}")) for m, t in run_outcomes(ds)]
        if not tables:
            continue
        allt = pd.concat([t for _, t in tables])
        for (name, act), g in allt[allt["Fall"]].groupby(["model", "Activity"]):
            missed = ~g["Main"]
            misses.append({"Dataset": ds, "Model": models.display_name(name), "Fall type": act,
                           "Fall trials (all runs)": len(g), "Missed by gate": int((missed & ~g["Gate passed"]).sum()),
                           "Missed by model": int((missed & g["Gate passed"]).sum()),
                           "Missed by gate (%)": round(100 * (missed & ~g["Gate passed"]).mean(), 2),
                           "Missed by model (%)": round(100 * (missed & g["Gate passed"]).mean(), 2),
                           "Missed, ungated FP32 tuned (%)": round(100 * (~g["FP32 tuned"]).mean(), 2)})
        adl = allt[~allt["Fall"]]
        for keys, out, label in ((["model", "Activity"], fa_type, "ADL type"), (["model", "Subject"], fa_subject, "Subject")):
            for (name, k), g in adl.groupby(keys):
                fa_type_row = {"Dataset": ds, "Model": models.display_name(name), label: k,
                               "ADL trials (all runs)": len(g),
                               "Passed by the gate (>= 1 window)": int(g["Gate passed"].sum()),
                               "Passed by the gate (%)": round(100 * g["Gate passed"].mean(), 2),
                               "False alarms, main setting": int(g["Main"].sum()),
                               "False alarms, main setting (%)": round(100 * g["Main"].mean(), 2),
                               "False alarms, ungated FP32 tuned": int(g["FP32 tuned"].sum()),
                               "False alarms, ungated FP32 tuned (%)": round(100 * g["FP32 tuned"].mean(), 2)}
                if label == "Subject":
                    fa_type_row["Group"] = pre.subject_group(k)
                out.append(fa_type_row)
        # per run and subject group, then mean ± std over runs
        per_run = []
        for (name, run, grp), g in allt.groupby(["model", "run", "Group"]):
            row = {"model": name, "group": grp, "fall trials": int(g["Fall"].sum()), "ADL trials": int((~g["Fall"]).sum())}
            for setting in ("Main", "FP32 tuned"):
                se = 100 * g.loc[g["Fall"], setting].mean() if g["Fall"].any() else np.nan
                sp = 100 * (~g.loc[~g["Fall"], setting]).mean() if (~g["Fall"]).any() else np.nan
                row.update({f"{setting} sensitivity": se, f"{setting} specificity": sp,
                            f"{setting} G-mean": np.sqrt(se * sp) if np.isfinite(se) and np.isfinite(sp) else np.nan})
            per_run.append(row)
        per_run = pd.DataFrame(per_run)
        for (name, grp), g in per_run.groupby(["model", "group"]):
            row = {"Dataset": ds, "Model": models.display_name(name), "Group": grp, "Runs": len(g),
                   "Runs with falls in this group": int((g["fall trials"] > 0).sum()),
                   "Fall trials per run (mean)": round(g["fall trials"].mean(), 1),
                   "ADL trials per run (mean)": round(g["ADL trials"].mean(), 1)}
            for c in [c for c in g.columns if "sensitivity" in c or "specificity" in c or "G-mean" in c]:
                v = g[c].dropna()
                row[c.replace("Main", MAIN)] = mean_std(v) if len(v) else "-"
            groups.append(row)
    if not misses:
        print("No saved cross-validation runs yet.")
        return
    for rows, name in ((misses, "misses_by_fall_type"), (fa_type, "false_alarms_by_adl_type"),
                       (fa_subject, "false_alarms_by_subject"), (groups, "metrics_by_subject_group")):
        df = save_csv(rows, OUT / f"{name}.csv")
        print(f"\n--- {name} ---")
        print_table(list(df.columns), df.astype(str).values.tolist())


# ---- 16 ----
ABLATION_CONDITIONS = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]


def gate_ablation(datasets, names=None):
    """The gate with 1, 2 or all 3 of its conditions (acc, gyr, tilt) and 1 or 2 windows in a row, each tuned like
    the main gate (>= GATE_TRIAL_RECALL of the training fall trials kept, fewest training windows passing): test pass
    rates, fall trials kept, ops per window, and the gated INT8 tuned results of `names` from their saved runs."""
    names = cfg.GATE_ABLATION_MODELS if names is None else names
    rows, tidy = [], []
    for ds in datasets:
        _, splits = fold_filter_splits(ds)
        metas, ctx = cv.saved_runs(ds, cfg.EPOCHS, cv.out_dir_for(ds))
        metas = [m for m in metas if m["model"] in names]
        preds = {id(m): cv.load_predictions(cv.out_dir_for(ds), m) for m in metas}
        filters = FilterCache()
        for k in (1, 2):
            for used in ABLATION_CONDITIONS:
                thr = [filters.get(d, used=used, k=k) for d in splits]
                test = [gate.filter_stats(d["gate_te"], d["y_te"], d["test"], t, k) for d, t in zip(splits, thr)]
                train = [gate.filter_stats(d["gate_tr"], d["y_tr"], d["train"], t, k) for d, t in zip(splits, thr)]
                ops = gate.ops_per_window(k, used)[1]
                label = gate.conditions_label(used)
                main_rule = set(used) == set(cfg.GATE_CONDITIONS) and k == cfg.GATE_CONSECUTIVE
                fmt = lambda t: "(" + ", ".join("-" if not np.isfinite(v) else f"{v:g}" for v in t) + ")"
                row = {"Dataset": ds, "Conditions": label, "Windows in a row": k, "Main gate": "yes" if main_rule else "",
                       "Gate ops / window": ops, "Thresholds per fold (acc, gyr, tilt)": "; ".join(fmt(t) for t in thr),
                       "Train fall trials kept (%)": mean_std([s["Fall trials kept"] for s in train]),
                       **{f"Test {key.lower()} (%)": mean_std([s[key] for s in test]) for key in test[0]}}
                for key in test[0]:
                    tidy.append({"Dataset": ds, "Conditions": label, "Windows in a row": k, "Series": "gate",
                                 "Metric": key, "mean": np.mean([s[key] for s in test]),
                                 "std": np.std([s[key] for s in test], ddof=1)})
                for name in names:
                    runs = [m for m in metas if m["model"] == name]
                    if not runs:
                        continue
                    res = [gated_int8(preds[id(m)], ctx.get(m)[0], filters.get(ctx.get(m)[0], used=used, k=k), k)
                           for m in runs]
                    row[f"{models.display_name(name)} runs"] = len(res)
                    for key in ("Trial sensitivity", "Trial specificity", "Trial G-mean"):
                        v = [x[key] for x in res]
                        row[f"{models.display_name(name)} {MAIN} {key}"] = mean_std(v)
                        tidy.append({"Dataset": ds, "Conditions": label, "Windows in a row": k,
                                     "Series": models.display_name(name), "Metric": f"{MAIN} {key}",
                                     "mean": np.mean(v), "std": np.std(v, ddof=1) if len(v) > 1 else np.nan})
                rows.append(row)
    df = save_csv(rows, OUT / "gate_ablation.csv")
    save_csv(tidy, cv.figures_dir() / f"gate_ablation{'_valthr' if cfg.THRESHOLD_PROTOCOL == 'validation' else ''}.csv")
    print_table(list(df.columns), df.astype(str).values.tolist())


def run(datasets):
    title = lambda t: print(f"\n==================== {t} ====================")
    title("1  gate cost per window")
    gate_ops_full, _ = gate_ops()
    title("2-3  average ops per window")
    ops_per_window(datasets, gate_ops_full)
    title("4  MACs by layer")
    macs_by_layer()
    title("5-6  sizes and activation memory")
    sizes()
    title("7  gate alone as a detector")
    gate_detector(datasets)
    title("8  gate recall sweep")
    recall_sweep(datasets)
    title("9  gate transfer between datasets")
    transfer()
    title("10 missed falls by fall type, false alarms by ADL type / subject, metrics by subject group")
    outcome_tables(datasets)
    title("16 gate ablation: conditions and windows in a row")
    gate_ablation(datasets)
    print(f"\nSaved to {OUT}")
