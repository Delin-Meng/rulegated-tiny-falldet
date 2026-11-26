"""
Early stopping vs the fixed 200-epoch protocol, KFall, k folds x 1 seed.
In each fold, config.ES_VAL_FRACTION of the training subjects are held out as a validation set (never the test
subjects). One training run per model and fold records two weight sets on the same trajectory:
  - early stopping: the weights with the lowest validation loss up to the epoch where early stopping
    (patience config.ES_PATIENCE) would stop, i.e. Keras EarlyStopping with restore_best_weights=True
  - 200 epochs: the weights after the last epoch (the protocol used everywhere else)
Both are evaluated on the fold's test subjects with all settings of training.all_metrics (thresholds and gate tuned
on the reduced training set). Runs are resumable; output in results/early_stopping/.

Early stopping on both datasets, run_es() / report_es(): k folds x seeds, models config.ES_MODELS.
Validation = config.THRESHOLD_VAL_FRACTION of each fold's training subjects (preprocessing.threshold_subjects, the
same subjects as THRESHOLD_PROTOCOL "validation"); training on the rest. One training run per (model, fold, seed)
gives two early-stopping rules, both keeping the weights with the lowest validation loss:
  "es"    at most config.EPOCHS epochs, stopped after config.ES_PATIENCE epochs without a lower validation loss
          (the training run itself stops there)         -> results/es_<dataset>/runs/
  "plan"  the short schedule: at most config.ES_PLAN_EPOCHS epochs, patience config.ES_PLAN_PATIENCE, read off the same
          run (with the same seed its first epochs are those of a separate short run)  -> results/es_plan_<dataset>/runs/
Saved like cross-validation runs (thresholds tuned on the validation subjects) plus the predictions on the training
(fit) windows. report_es(): the usual cross-validation report of each rule, and paired comparisons (incl. TOST) of
each rule with the 200-epoch main runs (results/cv_<dataset>/) and of the two rules with each other, at two
threshold-tuning sets:
  "validation subjects"  the held-out subjects (the main runs tune on their training subjects)
  "training subjects"    fit + validation subjects, i.e. the same subjects the main runs tune on
The early-stopped models saw 85 % of the training subjects, the main runs all of them.
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
import records
import training
from utils import print_columns, save_csv

OUT = cfg.RESULTS_DIR / "early_stopping"
SHOWN = ["Trial G-mean", "FP32 tuned Trial G-mean", "Gated FP32 tuned Trial G-mean", "Gated INT8 tuned Trial G-mean",
         "Gated INT8 tuned Trial sensitivity", "Gated INT8 tuned Trial specificity", "Window G-mean"]


class EarlyStoppingMonitor(keras.callbacks.Callback):
    """Records where early stopping would stop; stop=True also stops the training there.
    max_epochs: the rule's own epoch budget (it stops there at the latest, while the training may go on)."""

    def __init__(self, patience, stop=False, max_epochs=None):
        super().__init__()
        self.patience, self.best, self.best_epoch, self.stop = patience, np.inf, 0, stop
        self.max_epochs = max_epochs
        self.stop_epoch, self.weights, self.history = None, None, []

    def on_epoch_end(self, epoch, logs=None):
        loss = float(logs["val_loss"])
        self.history.append({"epoch": epoch + 1, "loss": float(logs["loss"]), "val_loss": loss})
        if self.stop_epoch is not None:
            return
        if loss < self.best:
            self.best, self.best_epoch, self.weights = loss, epoch + 1, self.model.get_weights()
        elif epoch + 1 - self.best_epoch >= self.patience:
            self.stop_epoch = epoch + 1
        if self.stop_epoch is None and self.max_epochs is not None and epoch + 1 >= self.max_epochs:
            self.stop_epoch = epoch + 1
        if self.stop_epoch is not None and self.stop:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stop_epoch is None:            # never triggered: early stopping would also run all epochs
            self.stop_epoch = len(self.history)


def evaluate(model, data, X_test, test):
    """All metrics on the test subjects. Thresholds and gate are tuned on the reduced training set, or on the
    validation subjects under config.THRESHOLD_PROTOCOL = "validation" (they are held out from training)."""
    int8, _, _ = deployment.quantize_int8(model, data["X_tr"])
    if cfg.THRESHOLD_PROTOCOL == "validation":
        X_tune, tune = data["X_te"], dict(train=data["test"], y_tr=data["y_te"], gate_tr=data["gate_te"])
    else:
        X_tune, tune = data["X_tr"], dict(train=data["train"], y_tr=data["y_tr"], gate_tr=data["gate_tr"])
    X_tune = X_tune.astype(np.float32)
    test_data = dict(tune, test=test["test"], y_te=test["y_te"], gate_te=test["gate_te"])
    return training.all_metrics(training.predict(model, X_tune), training.predict(model, X_test),
                                deployment.predict(int8, X_tune), deployment.predict(int8, X_test), test_data)


def run(names, folds, seed, epochs, patience=cfg.ES_PATIENCE):
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    trials = pre.load_trials("kfall")
    subject_folds = cv.fold_subjects(sorted({t["subject"] for t in trials}))
    for fold in folds:
        todo = [n for n in names if not (OUT / "runs" / f"{n}_fold{fold}_seed{seed}.json").exists()]
        if not todo:
            continue
        test_subjects = subject_folds[fold]
        train_subjects = sorted({t["subject"] for t in trials} - set(test_subjects))
        rng = np.random.default_rng(seed + 1000 * fold)
        val_subjects = sorted(rng.choice(train_subjects, round(cfg.ES_VAL_FRACTION * len(train_subjects)),
                                         replace=False))
        inner = [t for t in trials if t["subject"] not in test_subjects]
        data = pre.split_and_normalize(inner, test_subjects=val_subjects)    # train = reduced, "test" = validation
        test = pre.split(trials, test_subjects)
        X_test = pre.normalize(test["test"], data["mean"], data["std"])
        for name in todo:
            print(f"\n[early stopping] {name}  fold {fold}  seed {seed}  validation subjects {' '.join(val_subjects)}")
            monitor = EarlyStoppingMonitor(patience)
            model, minutes, _ = training.train_model(name, data["X_tr"], data["y_tr"], epochs, seed=seed, progress=0,
                                                     validation=(data["X_te"], data["y_te"]), callbacks=[monitor],
                                                     norm=(data["mean"], data["std"]))
            stem = OUT / "runs" / f"{name}_fold{fold}_seed{seed}"
            stem.with_suffix(".config.json").write_text(json.dumps(records.config_snapshot(
                name, model, training.training_setup(epochs), OUT,
                {"run": {"fold": fold, "seed": seed, "patience": patience}, "threshold protocol": cfg.THRESHOLD_PROTOCOL}),
                indent=1))
            final = evaluate(model, data, X_test, test)
            model.set_weights(monitor.weights)
            stopped = evaluate(model, data, X_test, test)
            save_csv(monitor.history, stem.with_suffix(".history.csv"))
            fit_subjects = sorted(set(train_subjects) - set(val_subjects))
            stem.with_suffix(".json").write_text(json.dumps(
                {"model": name, "fold": fold, "seed": seed, "epochs": epochs, "patience": patience,
                 "threshold_protocol": cfg.THRESHOLD_PROTOCOL,
                 "subjects": {"fit": fit_subjects, "validation (early stopping)": val_subjects,
                              "threshold": val_subjects if cfg.THRESHOLD_PROTOCOL == "validation" else fit_subjects,
                              "test": sorted(test_subjects)},
                 "normalization": records.normalization_record(data["mean"], data["std"]),
                 "best_epoch": monitor.best_epoch, "stop_epoch": monitor.stop_epoch, "training_minutes": minutes,
                 "early_stopping": stopped, "200_epochs": final}, indent=1))
            print(f"  best epoch {monitor.best_epoch}, early stopping at {monitor.stop_epoch}; "
                  f"gated INT8 tuned trial G-mean: early stopping {stopped['Gated INT8 tuned Trial G-mean']:.2f}, "
                  f"200 epochs {final['Gated INT8 tuned Trial G-mean']:.2f}")
    report()


RULES = {"es": "early stopping", "plan": "plan rule"}


def rule_dir(dataset, rule):
    """results/es_<dataset>/ (early stopping within config.EPOCHS) or results/es_plan_<dataset>/ (the short schedule)."""
    return cfg.RESULTS_DIR / (f"es_{dataset}" if rule == "es" else f"es_plan_{dataset}")


def save_rule_run(out, meta, model, data, monitor, budget, patience, epochs_used, minutes, history):
    """One rule's weights of a training run, saved like a cross-validation run (+ predictions on the fit windows)."""
    int8, int8_mode, _ = deployment.quantize_int8(model, data["X_tr"])
    X_val, X_te, X_fit = (data[k].astype(np.float32) for k in ("X_tune", "X_te", "X_tr"))
    stem = cv.stem_of(out, meta)
    model.save(stem.with_suffix(".keras"))
    np.savez_compressed(stem.with_suffix(".npz"), p_tr=training.predict(model, X_val), p_te=training.predict(model, X_te),
                        p8_tr=deployment.predict(int8, X_val).astype(np.float16),
                        p8_te=deployment.predict(int8, X_te).astype(np.float16),
                        p_fit=training.predict(model, X_fit), p8_fit=deployment.predict(int8, X_fit).astype(np.float16))
    save_csv(history[:epochs_used], stem.with_suffix(".history.csv"))
    stem.with_suffix(".config.json").write_text(json.dumps(records.config_snapshot(
        meta["model"], model, training.training_setup(budget), out,
        {"run": {**meta, "max epochs": budget, "patience": patience},
         "threshold protocol": "validation (the early-stopping validation subjects)"}), indent=1))
    stem.with_suffix(".json").write_text(json.dumps(
        {**meta, "epochs": budget, "patience": patience, "best_epoch": monitor.best_epoch,
         "stop_epoch": monitor.stop_epoch, "epochs_trained": epochs_used, "training_minutes": minutes,
         "int8_mode": int8_mode, "threshold_protocol": "validation", "subjects": data["subjects"],
         "normalization": records.normalization_record(data["mean"], data["std"]),
         "int8": deployment.int8_record(int8, int8_mode, data["X_tr"], data["y_tr"], data["subjects"]["fit"],
                                        data["mean"], data["std"]),
         "code sha256": records.code_fingerprint()["code sha256"]}, indent=1))


def run_es(dataset, names, folds, seeds, max_epochs=cfg.EPOCHS, patience=cfg.ES_PATIENCE,
           plan_epochs=cfg.ES_PLAN_EPOCHS, plan_patience=cfg.ES_PLAN_PATIENCE):
    rules = {"es": (max_epochs, patience), "plan": (plan_epochs, plan_patience)}
    for rule in rules:
        (rule_dir(dataset, rule) / "runs").mkdir(parents=True, exist_ok=True)
    trials = pre.load_trials(dataset)
    everyone = sorted({t["subject"] for t in trials})
    subject_folds = cv.fold_subjects(everyone)
    done = lambda f, n, s: all((rule_dir(dataset, r) / "runs" / f"{n}_fold{f}_seed{s}.json").exists() for r in rules)
    todo = [(f, n, s) for f in folds for n in names for s in seeds if not done(f, n, s)]
    print(f"Early stopping: {dataset}, {len(todo)} runs to do (<= {max_epochs} epochs, patience {patience}; "
          f"plan rule <= {plan_epochs} epochs, patience {plan_patience})")
    data, data_fold = None, None
    for i, (fold, name, seed) in enumerate(todo, 1):
        if fold != data_fold:
            test = subject_folds[fold]
            train = sorted(set(everyone) - set(test))
            val = pre.threshold_subjects(train, fold)
            data, data_fold = None, None
            data = pre.experiment_split(trials, test, fit_subjects=sorted(set(train) - set(val)), tune_subjects=val)
            data_fold = fold
        print(f"\n[early stopping {i}/{len(todo)}] {name}  fold {fold}  seed {seed}")
        monitors = {"es": EarlyStoppingMonitor(patience, stop=True, max_epochs=max_epochs),
                    "plan": EarlyStoppingMonitor(plan_patience, max_epochs=plan_epochs)}
        model, minutes, _ = training.train_model(name, data["X_tr"], data["y_tr"], max_epochs, seed=seed, progress=0,
                                                 validation=(data["X_tune"], data["tune"]["y_tr"]),
                                                 callbacks=list(monitors.values()), norm=(data["mean"], data["std"]))
        history = monitors["es"].history
        trained = len(history)
        meta = {"dataset": dataset, "model": name, "fold": fold, "seed": seed}
        for rule, monitor in monitors.items():
            budget, pat = rules[rule]
            used = min(monitor.stop_epoch, trained)          # epochs this rule trains (the short schedule's own run)
            model.set_weights(monitor.weights)
            save_rule_run(rule_dir(dataset, rule), meta, model, data, monitor, budget, pat, used,
                          minutes * used / trained, history)
        print(f"  early stopping: best epoch {monitors['es'].best_epoch}, {trained} epochs trained; plan rule: best "
              f"epoch {monitors['plan'].best_epoch}, stops at {monitors['plan'].stop_epoch} ({minutes:.1f} min)")


def _training_subject_data(trials, meta):
    """Evaluation data with all training subjects (fit + validation) as the tuning set, in the order of the saved
    predictions (fit windows, then validation windows)."""
    subjects = meta["subjects"]
    pick = lambda names: [t for t in trials if t["subject"] in set(names)]
    tune = pick(subjects["fit"]) + pick(subjects["threshold"])
    ev = pre.split(trials, subjects["test"])
    return dict(train=tune, y_tr=np.concatenate([t["y"] for t in tune]),
                gate_tr=np.concatenate([t["gate"] for t in tune]), test=ev["test"], y_te=ev["y_te"], gate_te=ev["gate_te"])


COMPARED = ["Trial G-mean", "FP32 tuned Trial G-mean", "INT8 tuned Trial G-mean", "Gated FP32 tuned Trial G-mean",
            "Gated INT8 tuned Trial G-mean", "Gated INT8 tuned Trial sensitivity", "Gated INT8 tuned Trial specificity"]
SUMMARY_METRICS = ["FP32 tuned Trial G-mean", "Gated INT8 tuned Trial G-mean"]
TUNINGS = ("validation subjects", "training subjects")


def on_training_subjects(dataset, out, budget, trials):
    """Metrics of a rule's runs with thresholds and gate tuned on all training subjects (fit + validation)."""
    rows, cache = [], {}
    for meta in cv.run_metas(out, budget):
        key = json.dumps(meta["subjects"], sort_keys=True)
        if key not in cache:
            d = _training_subject_data(trials, meta)
            cache[key] = (d, gate.tune_filter(d["gate_tr"], d["y_tr"], d["train"]))
        d, gthr = cache[key]
        p = np.load(cv.stem_of(out, meta).with_suffix(".npz"))
        f = lambda k: p[k].astype(np.float32)
        m = training.all_metrics(np.concatenate([f("p_fit"), f("p_tr")]), f("p_te"),
                                 np.concatenate([f("p8_fit"), f("p8_tr")]), f("p8_te"), d, gthr)
        rows.append({"model": meta["model"], "fold": meta["fold"], "seed": meta["seed"], "best epoch": meta["best_epoch"],
                     "epochs trained": meta["epochs_trained"], "budget": meta["epochs"], "patience": meta["patience"], **m})
    return pd.DataFrame(rows)


def report_es(dataset, max_epochs=cfg.EPOCHS, plan_epochs=cfg.ES_PLAN_EPOCHS):
    budgets = {"es": max_epochs, "plan": plan_epochs}
    trials = pre.load_trials(dataset)
    res = {}
    for rule, budget in budgets.items():
        out = rule_dir(dataset, rule)
        val = cv.evaluate_runs(dataset, budget, out) if (out / "runs").exists() else pd.DataFrame()
        if val.empty:
            continue
        cv.report(dataset, budget, out)                   # within the rule's runs (thresholds on validation subjects)
        res[rule] = {"validation subjects": val, "training subjects": on_training_subjects(dataset, out, budget, trials)}
        save_csv(res[rule]["training subjects"], out / "all_runs_thresholds_on_training_subjects.csv")
    if not res:
        print(f"No early-stopping runs for {dataset} yet.")
        return
    main = cv.evaluate_runs(dataset, cfg.EPOCHS, cv.out_dir_for(dataset))
    label = lambda rule: (f"{RULES[rule]} <= {budgets[rule]} ep., patience "
                          f"{int(res[rule]['training subjects']['patience'].iloc[0])}")
    pair_of = lambda a, b, n, metric: pd.concat(
        [a[a["model"] == n].set_index(["fold", "seed"])[metric].rename("a"),
         b[b["model"] == n].set_index(["fold", "seed"])[metric].rename("b")], axis=1).dropna()
    tests, eq = [], []
    for tuning in TUNINGS:
        for metric in COMPARED:
            block = []
            comparisons = [(r, "main") for r in res] + ([("es", "plan")] if len(res) == 2 else [])
            for ra, rb in comparisons:
                a = res[ra][tuning]
                b = main if rb == "main" else res[rb][tuning]
                if b.empty:
                    continue
                for n in [n for n in models.VARIANTS if n in set(a["model"])]:
                    pair = pair_of(a, b, n, metric)
                    if len(pair) < 2:
                        continue
                    la = f"{models.display_name(n)} {label(ra)} (thresholds: {tuning})"
                    lb = (f"{models.display_name(n)} {cfg.EPOCHS} epochs, all training subjects (main runs)" if rb == "main"
                          else f"{models.display_name(n)} {label(rb)} (thresholds: {tuning})")
                    block.append(cv.paired(pair["a"], pair["b"], la, lb, metric))
                    if metric in SUMMARY_METRICS:
                        eq += [cv.tost(pair["a"], pair["b"], la, lb, metric, mg) for mg in cfg.TOST_MARGINS]
            tests += cv.with_holm(block) if block else []
    summary = []
    names = [n for n in models.VARIANTS if any(n in set(r["training subjects"]["model"]) for r in res.values())]
    for n in names:
        row = {"Model": models.display_name(n)}
        for rule in res:
            e = res[rule]["training subjects"]
            e = e[e["model"] == n]
            row[f"{RULES[rule].capitalize()}: runs"] = len(e)
            row[f"{RULES[rule].capitalize()}: best epoch (val loss)"] = cv.mean_std(e["best epoch"])
            row[f"{RULES[rule].capitalize()}: epochs trained"] = cv.mean_std(e["epochs trained"])
            row[f"{RULES[rule].capitalize()}: runs stopped before {budgets[rule]} epochs"] = \
                int((e["epochs trained"] < budgets[rule]).sum())
        for metric in SUMMARY_METRICS:
            for tuning in TUNINGS:
                for rule in res:
                    g = res[rule][tuning]
                    row[f"{metric}, {RULES[rule]} (thresholds: {tuning})"] = cv.mean_std(g[g["model"] == n][metric])
            mm = main[main["model"] == n] if not main.empty else pd.DataFrame()
            row[f"{metric}, {cfg.EPOCHS} epochs (main runs)"] = cv.mean_std(mm[metric]) if len(mm) > 1 else "-"
        summary.append(row)
    out = rule_dir(dataset, "es") if "es" in res else rule_dir(dataset, "plan")
    print(f"\n===== {dataset}: early stopping vs {cfg.EPOCHS} epochs =====")
    print_columns(save_csv(summary, out / "comparison_summary.csv"))
    if tests:
        cv.print_tests(save_csv(tests, out / "comparison_paired_tests.csv"))
    if eq:
        save_csv(eq, out / "comparison_equivalence_tests.csv")
    print(f"Saved to {out}")


def report():
    files = sorted(p for p in (OUT / "runs").glob("*.json") if not p.name.endswith(".config.json"))
    if not files:
        print("No early-stopping runs yet.")
        return
    recs = [json.loads(p.read_text()) for p in files]
    rows = []
    for name in [n for n in models.VARIANTS if any(r["model"] == n for r in recs)]:
        rs = [r for r in recs if r["model"] == name]
        row = {"Model": models.display_name(name), "Folds": len(rs),
               "Best epoch (val loss)": cv.mean_std([r["best_epoch"] for r in rs]),
               "Early stopping epoch": cv.mean_std([r["stop_epoch"] for r in rs])}
        for c in SHOWN:
            es = np.array([r["early_stopping"][c] for r in rs])
            fixed = np.array([r["200_epochs"][c] for r in rs])
            row[f"{c}: early stopping"] = cv.mean_std(es)
            row[f"{c}: 200 epochs"] = cv.mean_std(fixed)
            row[f"{c}: 200 epochs - early stopping"] = cv.mean_std(fixed - es)
        rows.append(row)
    print_columns(save_csv(rows, OUT / "summary.csv"))
    print(f"Saved to {OUT}")
