"""
Subject-wise k-fold cross-validation x several seeds: mean ± std, paired tests and equivalence tests.
The subjects are split into folds once (identical for every model and seed, so runs are paired by (fold, seed)).
Normalisation statistics come from the training (fit) subjects; decision and gate thresholds are tuned on the
subjects given by config.THRESHOLD_PROTOCOL (the training subjects, or held-out training subjects).

Each (model, fold, seed) run saves to results/cv_<dataset>[_valthr]/runs/ as soon as it finishes:
  <run>.keras          trained model
  <run>.npz            P(fall) of the FP32 and full-INT8 model on the threshold-tuning windows (p_tr, p8_tr)
                       and on the test windows (p_te, p8_te)
  <run>.history.csv    training loss / accuracy per epoch
  <run>.config.json    full configuration snapshot (records.config_snapshot) incl. the code fingerprint
  <run>.json           written last (marks the run complete): metadata, subject lists (fit / threshold / test),
                       normalisation statistics, INT8 calibration set and I/O quantization
All metrics are computed from the saved predictions in report(), so the evaluation (gate, threshold rules, ...)
can change without retraining. Runs are resumable and models can run in parallel terminals.

Paired tests over the (fold, seed) pairs, Holm-corrected within each family of comparisons: Wilcoxon signed-rank
test, and the corrected resampled t-test (Nadeau & Bengio, 2003), which accounts for the overlap between training
sets of different folds. Runs of different seeds on the same fold are not independent, so both p-values are
approximate (the corrected t-test is the more conservative one). Equivalence: TOST with the corrected t statistic.
"""
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

import config as cfg
import deployment
import gate
import models
import preprocessing as pre
import records
import training
from utils import print_columns, print_table, save_csv

# main table: absolute values (mean ± std) per model; settings x gate
MAIN_METRICS = ["Trial sensitivity", "Trial specificity", "Trial G-mean", "Window G-mean"]
MAIN_SETTINGS = ["", "FP32 tuned", "INT8 tuned", "INT8 at FP32 threshold"]
EXTRA = ["FP32 Trial AUC", "INT8 Trial AUC", "Gated FP32 Trial AUC", "Gated INT8 Trial AUC",
         "FP32 tuned threshold", "INT8 tuned threshold", "Gated FP32 tuned threshold", "Gated INT8 tuned threshold",
         "Gate acc threshold (g)", "Gate gyr threshold (deg/s)", "Gate tilt threshold (deg)",
         "Gate pass rate, test windows", "Gate pass rate, ADL test windows", "Gate fall trials kept, test"]
MAIN = "Gated INT8 tuned Trial G-mean"           # main setting: gate + tuned threshold + full INT8
# equivalence tests (TOST) at every margin in config.TOST_MARGINS
TOST_BETWEEN = [MAIN, "FP32 tuned Trial G-mean"]
TOST_WITHIN = [("Gated FP32 tuned Trial G-mean", "Gated INT8 tuned Trial G-mean", "gated FP32", "gated INT8"),
               ("FP32 tuned Trial G-mean", "INT8 tuned Trial G-mean", "FP32", "INT8")]
# paired tests
BETWEEN_MODELS = ["Trial G-mean", "FP32 tuned Trial G-mean", "INT8 tuned Trial G-mean",
                  "Gated FP32 tuned Trial G-mean", "Gated INT8 tuned Trial G-mean"]
WITHIN_MODEL = [("FP32 tuned Trial G-mean", "INT8 tuned Trial G-mean", "FP32", "INT8 (own threshold)"),
                ("FP32 tuned Trial G-mean", "INT8 at FP32 threshold Trial G-mean", "FP32", "INT8 (FP32 threshold)"),
                ("Gated FP32 tuned Trial G-mean", "Gated INT8 tuned Trial G-mean", "gated FP32", "gated INT8"),
                ("FP32 tuned Trial G-mean", "Gated FP32 tuned Trial G-mean", "FP32", "gated FP32"),
                ("INT8 tuned Trial G-mean", "Gated INT8 tuned Trial G-mean", "INT8", "gated INT8")]
# each model's deployment configurations vs the baseline alone (config.BASELINE_MODEL, FP32, no gate)
VS_BASELINE = [(MAIN, "gate + INT8"), ("INT8 tuned Trial G-mean", "INT8"), ("Gated FP32 tuned Trial G-mean", "gate + FP32"),
               ("FP32 tuned Trial G-mean", "FP32")]
BASELINE_METRIC = "FP32 tuned Trial G-mean"
META = ("dataset", "model", "fold", "seed", "epochs", "training_minutes", "int8_mode", "threshold_protocol")


def main_columns():
    cols = [" ".join(p for p in (g, s, m) if p) for g in ("", "Gated") for s in MAIN_SETTINGS for m in MAIN_METRICS]
    return cols + EXTRA


def out_dir_for(dataset):
    return cfg.RESULTS_DIR / f"cv_{dataset}{'_valthr' if cfg.THRESHOLD_PROTOCOL == 'validation' else ''}"


def figures_dir():
    path = cfg.RESULTS_DIR / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fold_subjects(subjects, n_folds=cfg.CV_FOLDS):
    """Fixed subject folds (same for every model and seed)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.CV_FOLD_SEED)
    return [[subjects[i] for i in test] for _, test in kf.split(subjects)]


def split_table(trials):
    """Per fold and role, the subjects of the two training protocols with their trial / window counts:
      main            200 epochs on all training subjects; thresholds and gate tuned on them (THRESHOLD_PROTOCOL
                      "train"); normalisation and INT8 calibration from them
      early stopping  training on the fit subjects, early stopping on the validation subjects
                      (THRESHOLD_VAL_FRACTION of the training subjects, preprocessing.threshold_subjects);
                      thresholds and gate tuned on the validation subjects or on fit + validation (both reported);
                      normalisation and INT8 calibration from the fit subjects
    The test subjects are the same in both, so every run of one protocol is paired with one of the other."""
    subjects = sorted({t["subject"] for t in trials})
    count = {s: {"trials": 0, "falls": 0, "windows": 0, "accident windows": 0} for s in subjects}
    for t in trials:
        c = count[t["subject"]]
        c["trials"] += 1
        c["falls"] += t["type"] == "F"
        c["windows"] += len(t["y"])
        c["accident windows"] += int(t["y"].sum())
    total_w = sum(c["windows"] for c in count.values())
    rows = []
    for f, test in enumerate(fold_subjects(subjects)):
        train = sorted(set(subjects) - set(test))
        val = pre.threshold_subjects(train, f)
        fit = sorted(set(train) - set(val))
        for protocol, role, group in (
                ("main (200 epochs)", "training; thresholds and gate tuned here", train),
                ("main (200 epochs)", "test", test),
                ("early stopping", "training (fit)", fit),
                ("early stopping", "validation: early stopping; thresholds and gate (or on fit + validation)", val),
                ("early stopping", "test", test)):
            n = {k: sum(count[s][k] for s in group) for k in ("trials", "falls", "windows", "accident windows")}
            rows.append({"Fold": f, "Protocol": protocol, "Role": role, "Subjects": len(group),
                         "Subjects (%)": round(100 * len(group) / len(subjects), 1), "Trials": n["trials"],
                         "Fall trials": n["falls"], "ADL trials": n["trials"] - n["falls"], "Windows": n["windows"],
                         "Windows (%)": round(100 * n["windows"] / total_w, 1), "Accident windows": n["accident windows"],
                         "Subject IDs": " ".join(sorted(group))})
    return rows


def stem_of(out_dir, meta):
    return out_dir / "runs" / f"{meta['model']}_fold{meta['fold']}_seed{meta['seed']}"


def run(dataset, names, seeds, folds, epochs, out_dir):
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    trials = pre.load_trials(dataset)
    records.write_data_manifest(dataset, trials)
    subject_folds = fold_subjects(sorted({t["subject"] for t in trials}))
    if not (out_dir / "folds.csv").exists():             # parallel jobs on one folder: written once
        save_csv([{"Fold": f, "Test subjects": " ".join(s)} for f, s in enumerate(subject_folds)],
                 out_dir / "folds.csv")

    # a run is complete once its .json exists: it is written last, after the model and the predictions
    todo = [(f, m, s) for f in folds for m in names for s in seeds
            if not (runs_dir / f"{m}_fold{f}_seed{s}.json").exists()]
    print(f"{len(todo)} runs to do ({epochs} epochs each, threshold protocol '{cfg.THRESHOLD_PROTOCOL}'), "
          f"{len(folds) * len(names) * len(seeds) - len(todo)} already saved")
    data, data_fold = None, None
    for i, (fold, name, seed) in enumerate(todo, 1):
        if fold != data_fold:                          # split / normalise once per fold
            data, data_fold = pre.experiment_split(trials, subject_folds[fold], key=fold), fold
        print(f"\n[{i}/{len(todo)}] {name}  fold {fold}  seed {seed}")
        model, minutes, history = training.train_model(name, data["X_tr"], data["y_tr"], epochs, seed=seed,
                                                       progress=0, norm=(data["mean"], data["std"]))
        int8, int8_mode, _ = deployment.quantize_int8(model, data["X_tr"])
        X_tune, X_te = data["X_tune"].astype(np.float32), data["X_te"].astype(np.float32)
        preds = dict(p_tr=training.predict(model, X_tune), p_te=training.predict(model, X_te),
                     p8_tr=deployment.predict(int8, X_tune), p8_te=deployment.predict(int8, X_te))
        m = training.all_metrics(**preds, data=pre.metric_data(data))
        print(f"  FP32 tuned trial G-mean {m['FP32 tuned Trial G-mean']:.2f}  "
              f"INT8 tuned {m['INT8 tuned Trial G-mean']:.2f}  "
              f"gated INT8 tuned {m['Gated INT8 tuned Trial G-mean']:.2f}  ({minutes:.1f} min)")
        meta = {"dataset": dataset, "model": name, "fold": fold, "seed": seed}
        stem = stem_of(out_dir, meta)
        model.save(stem.with_suffix(".keras"))
        # INT8 outputs are multiples of 1/256, exact in float16
        np.savez_compressed(stem.with_suffix(".npz"), p_tr=preds["p_tr"], p_te=preds["p_te"],
                            p8_tr=preds["p8_tr"].astype(np.float16), p8_te=preds["p8_te"].astype(np.float16))
        save_csv(training.history_rows(history), stem.with_suffix(".history.csv"))
        stem.with_suffix(".config.json").write_text(json.dumps(records.config_snapshot(
            name, model, training.training_setup(epochs), out_dir,
            {"run": {**meta, "epochs": epochs}, "threshold protocol": cfg.THRESHOLD_PROTOCOL}), indent=1))
        stem.with_suffix(".json").write_text(json.dumps(
            {**meta, "epochs": epochs, "training_minutes": minutes, "int8_mode": int8_mode,
             "threshold_protocol": cfg.THRESHOLD_PROTOCOL, "subjects": data["subjects"],
             "normalization": records.normalization_record(data["mean"], data["std"]),
             "int8": deployment.int8_record(int8, int8_mode, data["X_tr"], data["y_tr"], data["subjects"]["fit"],
                                            data["mean"], data["std"]),
             "code sha256": records.code_fingerprint()["code sha256"]}, indent=1))


def run_split(trials, meta, subject_folds):
    """experiment_split (with model inputs) of a saved run, rebuilt from its subject lists. Runs saved before the
    lists were recorded used the fold's test subjects and tuned their thresholds on all training subjects."""
    subjects = meta.get("subjects")
    if subjects:
        return pre.experiment_split(trials, subjects["test"], fit_subjects=subjects["fit"],
                                    tune_subjects=subjects["threshold"])
    test = subject_folds[meta["fold"]]
    train = sorted({t["subject"] for t in trials} - set(test))
    return pre.experiment_split(trials, test, fit_subjects=train, tune_subjects=train)


class RunContext:
    """Evaluation data of saved runs without the model inputs, cached per subject split, with the gate filter
    thresholds tuned on each split's threshold-tuning subjects."""

    def __init__(self, dataset):
        self.trials = pre.load_trials(dataset)
        self.folds = fold_subjects(sorted({t["subject"] for t in self.trials}))
        self.cache = {}

    def get(self, meta):
        subjects = meta.get("subjects") or {}
        key = (meta["fold"], tuple(subjects.get("threshold", [])), tuple(subjects.get("test", [])))
        if key not in self.cache:
            if subjects:
                test, tune = subjects["test"], subjects["threshold"]
            else:                                   # runs saved before subject lists were recorded
                test = self.folds[meta["fold"]]
                tune = sorted({t["subject"] for t in self.trials} - set(test))
            ev = pre.split(self.trials, test)       # test part
            tune_trials = [t for t in self.trials if t["subject"] in set(tune)]
            data = dict(train=tune_trials, y_tr=np.concatenate([t["y"] for t in tune_trials]),
                        gate_tr=np.concatenate([t["gate"] for t in tune_trials]),
                        test=ev["test"], y_te=ev["y_te"], gate_te=ev["gate_te"])
            self.cache[key] = (data, gate.tune_filter(data["gate_tr"], data["y_tr"], data["train"]))
        return self.cache[key]


def run_metas(out_dir, epochs):
    """Metadata of the complete saved runs in out_dir/runs/ with the given epochs."""
    metas = [json.loads(p.read_text()) for p in sorted((out_dir / "runs").glob("*.json"))
             if not p.name.endswith(".config.json") and p.with_suffix(".npz").exists()]
    return [m for m in metas if m["epochs"] == epochs]


def saved_runs(dataset, epochs, out_dir):
    """(metadata of complete saved runs, RunContext) for the given epochs."""
    metas = run_metas(out_dir, epochs)
    return metas, (RunContext(dataset) if metas else None)


PREDICTIONS = ("p_tr", "p_te", "p8_tr", "p8_te")     # FP32 / INT8 on the threshold-tuning and test windows


def load_predictions(out_dir, meta):
    p = np.load(stem_of(out_dir, meta).with_suffix(".npz"))
    return {k: p[k].astype(np.float32) for k in PREDICTIONS}


def evaluate_runs(dataset, epochs, out_dir, with_curves=False):
    """All metrics of every saved run, computed from its predictions; optionally the threshold sweep and ROC rows."""
    metas, ctx = saved_runs(dataset, epochs, out_dir)
    rows, sweep, roc = [], [], []
    for meta in metas:
        data, gthr = ctx.get(meta)
        p = load_predictions(out_dir, meta)
        base = {k: meta.get(k, "train" if k == "threshold_protocol" else None) for k in META}
        rows.append({**base, **training.all_metrics(**p, data=data, gate_thresholds=gthr)})
        if with_curves:
            s, r = training.curves(p["p_te"], p["p8_te"], data, gthr)
            ids = {"model": meta["model"], "fold": meta["fold"], "seed": meta["seed"]}
            sweep += [{**ids, **x} for x in s]
            roc += [{**ids, **x} for x in r]
    return (pd.DataFrame(rows), pd.DataFrame(sweep), pd.DataFrame(roc)) if with_curves else pd.DataFrame(rows)


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def corrected_ttest(diff, test_train_ratio):
    """Nadeau & Bengio corrected resampled t-test, two-sided p-value."""
    n, mean, var = len(diff), diff.mean(), diff.var(ddof=1)
    if var == 0:
        return 1.0 if mean == 0 else 0.0
    t = mean / np.sqrt((1 / n + test_train_ratio) * var)
    return float(2 * stats.t.sf(abs(t), n - 1))


def mean_std(x, decimals=2):
    x = np.asarray(x, float)
    return f"{np.mean(x):.{decimals}f} ± {np.std(x, ddof=1):.{decimals}f}" if len(x) > 1 else f"{np.mean(x):.{decimals}f}"


def paired(a, b, label_a, label_b, metric):
    """One paired comparison of two aligned score vectors, with both absolute values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    return {"Metric": metric, "A": label_a, "B": label_b, "Pairs": len(diff),
            "A mean ± std": mean_std(a), "B mean ± std": mean_std(b), "A - B mean ± std": mean_std(diff),
            "Wilcoxon p": stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0,
            "Corrected t-test p": corrected_ttest(diff, 1 / (cfg.CV_FOLDS - 1))}


def with_holm(block):
    for col in ["Wilcoxon p", "Corrected t-test p"]:
        for row, adj in zip(block, holm([r[col] for r in block])):
            row[f"{col} (Holm)"] = adj
    return block


def tost(a, b, label_a, label_b, metric, margin):
    """Two one-sided tests for |mean(A - B)| < margin with the corrected resampled t statistic.
    Equivalent at the 5 % level when p < 0.05, i.e. when the 90 % confidence interval lies inside +-margin."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    n, mean = len(diff), diff.mean()
    se = np.sqrt((1 / n + 1 / (cfg.CV_FOLDS - 1)) * diff.var(ddof=1))
    if se == 0:
        p = 0.0 if abs(mean) < margin else 1.0
        half = 0.0
    else:
        p = float(max(stats.t.sf((mean + margin) / se, n - 1), stats.t.cdf((mean - margin) / se, n - 1)))
        half = stats.t.ppf(0.95, n - 1) * se
    return {"Metric": metric, "A": label_a, "B": label_b, "Pairs": n, "Margin": margin,
            "A mean ± std": mean_std(a), "B mean ± std": mean_std(b), "A - B mean": round(float(mean), 3),
            "90% CI of A - B": f"[{mean - half:.2f}, {mean + half:.2f}]",
            "TOST p": p, "Equivalent (p < 0.05)": "yes" if p < 0.05 else "no"}


def vs_baseline_pairs(runs, names):
    """{setting column: [(model values, baseline values, label, baseline label)]} on the shared (fold, seed) pairs:
    every model's setting vs config.BASELINE_MODEL with FP32 and its own tuned threshold, no gate."""
    out = {col: [] for col, _ in VS_BASELINE}
    base_name = cfg.BASELINE_MODEL
    if base_name not in names:
        return out
    base = runs[runs["model"] == base_name].set_index(["fold", "seed"])[BASELINE_METRIC]
    for col, label in VS_BASELINE:
        for n in names:
            if n == base_name and col == BASELINE_METRIC:
                continue
            mine = runs[runs["model"] == n].set_index(["fold", "seed"])[col]
            pair = pd.concat([mine.rename("a"), base.rename("b")], axis=1).dropna()
            if len(pair) >= 2:
                out[col].append((pair["a"], pair["b"], f"{models.display_name(n)} {label}",
                                 f"{models.display_name(base_name)} FP32 (no gate)"))
    return out


def print_tests(tests):
    shown = tests.copy()
    for c in [c for c in shown.columns if c.endswith(" p") or c.endswith("(Holm)")]:
        shown[c] = shown[c].map(lambda v: f"{v:.4f}")
    print_table(list(shown.columns), shown.astype(str).values.tolist())


def mean_std_columns(df, keys, value_cols):
    """Mean and std of value_cols per group of keys, as separate columns."""
    g = df.groupby(keys, sort=False)[value_cols]
    out = g.mean().add_suffix(" mean").join(g.std(ddof=1).add_suffix(" std")).join(g.size().rename("runs"))
    return out.reset_index()


def write_figures(prefix, runs, sweep, roc, names, out_dir):
    """Per-figure CSVs in results/figures/<prefix>_*.csv, read directly by plotting scripts."""
    fig = figures_dir()
    disp = lambda df: df.assign(model=df["model"].map(models.display_name))
    if not sweep.empty:
        cols = ["Trial sensitivity", "Trial specificity", "Trial G-mean"]
        save_csv(disp(mean_std_columns(sweep, ["model", "Setting", "Threshold"], cols)),
                 fig / f"{prefix}_threshold_sweep.csv")
        save_csv(disp(sweep), out_dir / "threshold_sweep_runs.csv")
    if not roc.empty:
        save_csv(disp(mean_std_columns(roc, ["model", "Setting", "FPR"], ["TPR", "AUC"])), fig / f"{prefix}_roc_trial.csv")
    # training curves of the runs that saved them
    hist = []
    for _, r in runs.iterrows():
        path = stem_of(out_dir, r).with_suffix(".history.csv")
        if path.exists():
            hist.append(pd.read_csv(path).assign(model=r["model"], fold=r["fold"], seed=r["seed"]))
    if hist:
        hist = pd.concat(hist)
        save_csv(disp(mean_std_columns(hist, ["model", "epoch"], [c for c in hist.columns if c not in
                                                                    ("epoch", "model", "fold", "seed")])),
                 fig / f"{prefix}_training_curves.csv")
    # parameter count / MACs vs G-mean
    rows = []
    for n in names:
        model = models.build(n, len(cfg.CHANNELS), cfg.WINDOW)
        r = runs[runs["model"] == n]
        row = {"model": models.display_name(n), "parameters": model.count_params(),
               "MACs / window": deployment.count_macs(deployment.convert(model)), "runs": len(r)}
        for c in [MAIN, "FP32 tuned Trial G-mean", "Trial G-mean"]:
            row[f"{c} mean"], row[f"{c} std"] = r[c].mean(), r[c].std(ddof=1)
        rows.append(row)
    save_csv(rows, fig / f"{prefix}_params_vs_gmean.csv")
    save_csv([{k: v for k, v in r.items() if k != "parameters"} | {"parameters": r["parameters"]} for r in rows],
             fig / f"{prefix}_macs_vs_gmean.csv")


def report(dataset, epochs, out_dir):
    runs, sweep, roc = evaluate_runs(dataset, epochs, out_dir, with_curves=True)
    if runs.empty:
        print("No saved runs with predictions yet.")
        return
    if runs["threshold_protocol"].nunique() > 1:
        raise ValueError(f"{out_dir} mixes threshold protocols {sorted(runs['threshold_protocol'].unique())}")
    save_csv(runs, out_dir / "all_runs.csv")
    names = [n for n in models.VARIANTS if n in set(runs["model"])] + \
            sorted(set(runs["model"]) - set(models.VARIANTS))
    by = {n: runs[runs["model"] == n] for n in names}

    # absolute values: every metric (full table) and the main paper table
    metric_cols = [c for c in runs.columns if c not in META]
    full = [{"Model": models.display_name(n), "Runs": len(by[n]),
             **{c: mean_std(by[n][c], 4 if "AUC" in c else 2) for c in metric_cols}} for n in names]
    save_csv(full, out_dir / "summary_all_metrics.csv")
    main = save_csv([{k: r[k] for k in ["Model", "Runs", *main_columns()]} for r in full], out_dir / "summary.csv")
    print(f"\n===== {dataset}, {epochs} epochs, thresholds tuned on: {runs['threshold_protocol'].iloc[0]} subjects; "
          f"mean ± std over folds x seeds (%; thresholds as P, AUC as fraction, gate in g) =====")
    print_columns(main)

    # paired tests
    tests = []
    for metric in BETWEEN_MODELS:                       # model vs model on the same (fold, seed)
        table = runs.pivot_table(index=["fold", "seed"], columns="model", values=metric).dropna()
        pairs = list(combinations([n for n in names if n in table], 2))
        if len(table) >= 2 and pairs:
            tests += with_holm([paired(table[a], table[b], models.display_name(a), models.display_name(b), metric)
                                for a, b in pairs])
    for col_a, col_b, label_a, label_b in WITHIN_MODEL:  # same model, two settings
        block = [paired(by[n][col_a], by[n][col_b], f"{models.display_name(n)} {label_a}",
                        f"{models.display_name(n)} {label_b}", "Trial G-mean") for n in names if len(by[n]) >= 2]
        if block:
            tests += with_holm(block)
    vs_base = vs_baseline_pairs(runs, names)
    for col, label in VS_BASELINE:                       # deployment configurations vs the baseline alone
        block = [paired(a, b, la, lb, f"{col} vs baseline {BASELINE_METRIC}") for a, b, la, lb in vs_base[col]]
        if block:
            tests += with_holm(block)
    if tests:
        print("\n===== Paired tests over the same folds and seeds =====")
        print_tests(save_csv(tests, out_dir / "paired_tests.csv"))

    # equivalence (TOST) at several margins: between models, and FP32 vs INT8 within each model
    eq = []
    for margin in cfg.TOST_MARGINS:
        for metric in TOST_BETWEEN:
            table = runs.pivot_table(index=["fold", "seed"], columns="model", values=metric).dropna()
            if len(table) >= 2:
                eq += [tost(table[a], table[b], models.display_name(a), models.display_name(b), metric, margin)
                       for a, b in combinations([n for n in names if n in table], 2)]
        for col_a, col_b, label_a, label_b in TOST_WITHIN:
            eq += [tost(by[n][col_a], by[n][col_b], f"{models.display_name(n)} {label_a}",
                        f"{models.display_name(n)} {label_b}", col_a, margin) for n in names if len(by[n]) >= 2]
        for col, _ in VS_BASELINE:
            eq += [tost(a, b, la, lb, f"{col} vs baseline {BASELINE_METRIC}", margin) for a, b, la, lb in vs_base[col]]
    if eq:
        print(f"\n===== Equivalence tests (TOST, margins {cfg.TOST_MARGINS} G-mean points) =====")
        print_tests(save_csv(eq, out_dir / "equivalence_tests.csv"))

    write_figures(out_dir.name, runs, sweep, roc, names, out_dir)
    print(f"\nSaved to {out_dir} and {figures_dir()}")
