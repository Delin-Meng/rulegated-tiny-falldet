"""
Magnitude pruning with fine-tuning, from the saved cross-validation models (no training from scratch).
Each saved run of the given models (default TinyTCN, all folds x seed 0) is pruned to every sparsity in
config.PRUNE_SPARSITIES: layer by layer, the smallest |w| of each conv / depthwise / dense / LSTM kernel are set to
zero (biases, normalisation, SSM parameters kept), then fine-tuned config.PRUNE_FINETUNE_EPOCHS epochs at
config.PRUNE_LEARNING_RATE with the pruned weights held at zero (a variable constraint applied after every update).
Sparsity 0 is the control: the same fine-tuning without pruning.
Evaluated like the main runs: FP32 and full INT8 (zeros stay zero in symmetric int8), thresholds tuned on the run's
tuning subjects, without and with the gate.
Note: TFLite / TFLM execute dense kernels, so the zeros save neither time nor memory without sparse kernels;
reported are the nonzero weights and the MACs that multiply a nonzero weight (what sparse kernels would execute).
Saved in results/pruning/<dataset>/runs/<model>_sp<percent>_fold<f>_seed<s>.{keras,npz,json};
report(): summary.csv, paired_tests.csv (vs the control and vs the unpruned main run), figures/pruning_<dataset>.csv.
"""
import json

import keras
import numpy as np
import pandas as pd

import config as cfg
import cv
import deployment
import models
import preprocessing as pre
import training
from utils import print_columns, save_csv

OUT = cfg.RESULTS_DIR / "pruning"
PRUNABLE = ("kernel", "recurrent_kernel", "depthwise_kernel", "pointwise_kernel")
SHOWN = ["FP32 tuned Trial G-mean", "INT8 tuned Trial G-mean", "Gated INT8 tuned Trial G-mean",
         "Gated INT8 tuned Trial sensitivity", "Gated INT8 tuned Trial specificity"]


@keras.saving.register_keras_serializable(package="pruning")
class Mask(keras.constraints.Constraint):
    """Keeps pruned weights at zero: applied by the optimizer after every update."""

    def __init__(self, mask):
        self.mask = np.asarray(mask, np.float32)

    def __call__(self, w):
        return w * keras.ops.convert_to_tensor(self.mask, dtype=w.dtype)

    def get_config(self):
        return {"mask": self.mask.tolist()}


def prune(model, sparsity):
    """Layer-wise magnitude pruning in place. Returns (prunable weights, of them set to zero)."""
    total = zeros = 0
    for v in model.trainable_weights:
        if v.name not in PRUNABLE:
            continue
        w = keras.ops.convert_to_numpy(v)
        k = int(round(sparsity * w.size))
        total += w.size
        if k == 0:
            continue
        mask = np.ones(w.size, np.float32)
        mask[np.argsort(np.abs(w).ravel(), kind="stable")[:k]] = 0.0
        mask = mask.reshape(w.shape)
        v.assign(w * mask)
        v.constraint = Mask(mask)
        zeros += k
    return total, zeros


def count_zeros(model):
    w = [keras.ops.convert_to_numpy(v) for v in model.trainable_weights if v.name in PRUNABLE]
    return sum(x.size for x in w), sum(int((x == 0).sum()) for x in w)


def run_name(model, sparsity):
    return f"{model}_sp{int(round(100 * sparsity)):02d}"


def stem(dataset, meta, sparsity):
    return OUT / dataset / "runs" / f"{run_name(meta['model'], sparsity)}_fold{meta['fold']}_seed{meta['seed']}"


def run(dataset, names, folds, seeds, sparsities=None, epochs=cfg.PRUNE_FINETUNE_EPOCHS):
    sparsities = cfg.PRUNE_SPARSITIES if sparsities is None else sparsities
    models.register_custom_layers()
    src = cv.out_dir_for(dataset)
    (OUT / dataset / "runs").mkdir(parents=True, exist_ok=True)
    metas = sorted([m for m in cv.run_metas(src, cfg.EPOCHS) if m["model"] in names and m["fold"] in folds
                    and m["seed"] in seeds], key=lambda m: (m["fold"], m["model"], m["seed"]))
    todo = [(m, s) for m in metas for s in sparsities if not stem(dataset, m, s).with_suffix(".json").exists()]
    print(f"Pruning {dataset}: {len(todo)} of {len(metas) * len(sparsities)} runs to do "
          f"({epochs} fine-tuning epochs, lr {cfg.PRUNE_LEARNING_RATE})")
    if not todo:
        return
    trials = pre.load_trials(dataset)
    subject_folds = cv.fold_subjects(sorted({t["subject"] for t in trials}))
    data, data_key = None, None
    for i, (meta, sparsity) in enumerate(todo, 1):
        key = (meta["fold"], json.dumps(meta.get("subjects"), sort_keys=True))
        if key != data_key:
            data, data_key = None, None
            data, data_key = cv.run_split(trials, meta, subject_folds), key
        print(f"\n[pruning {i}/{len(todo)}] {meta['model']} fold {meta['fold']} seed {meta['seed']} "
              f"sparsity {100 * sparsity:.0f}%")
        model = keras.models.load_model(cv.stem_of(src, meta).with_suffix(".keras"))
        prunable, pruned = prune(model, sparsity)
        minutes, history = training.fine_tune(model, data["X_tr"], data["y_tr"], epochs, cfg.PRUNE_LEARNING_RATE,
                                              meta["seed"])
        total, zeros = count_zeros(model)
        for v in model.trainable_weights:
            v.constraint = None                   # the zeros are in the weights now; keeps the saved model plain
        int8, int8_mode, _ = deployment.quantize_int8(model, data["X_tr"])
        fp32 = deployment.convert(model, "fp32")
        nonzero_macs = deployment.nonzero_macs(fp32)
        X_tune, X_te = data["X_tune"].astype(np.float32), data["X_te"].astype(np.float32)
        s = stem(dataset, meta, sparsity)
        model.save(s.with_suffix(".keras"))
        np.savez_compressed(s.with_suffix(".npz"), p_tr=training.predict(model, X_tune), p_te=training.predict(model, X_te),
                            p8_tr=deployment.predict(int8, X_tune).astype(np.float16),
                            p8_te=deployment.predict(int8, X_te).astype(np.float16))
        save_csv(training.history_rows(history), s.with_suffix(".history.csv"))
        s.with_suffix(".json").write_text(json.dumps(
            {k: meta[k] for k in ("dataset", "model", "fold", "seed", "subjects")} | {
                "sparsity": sparsity, "fine-tuning epochs": epochs,
                "learning rate": cfg.PRUNE_LEARNING_RATE, "fine-tuning minutes": minutes,
                "prunable weights": int(total), "zero prunable weights": int(zeros),
                "zero prunable weights right after pruning": int(pruned), "parameters": int(model.count_params()),
                "nonzero parameters": int(model.count_params() - zeros),
                "MACs (dense kernels)": deployment.count_macs(fp32), "MACs on nonzero weights": nonzero_macs,
                ".tflite INT8 (KB)": round(len(int8) / 1024, 2), "INT8 mode": int8_mode}, indent=1))
        print(f"  zero prunable weights {zeros} of {total}; MACs on nonzero weights {nonzero_macs}")


MAIN_RUN = "main run (no fine-tuning)"


def level(sparsity):
    return f"{100 * sparsity:.0f}%"


def evaluate(dataset):
    src = cv.out_dir_for(dataset)
    metas, ctx = cv.saved_runs(dataset, cfg.EPOCHS, src)
    by_key = {(m["model"], m["fold"], m["seed"]): m for m in metas}
    rows, done = [], set()
    for path in sorted((OUT / dataset / "runs").glob("*.json")):
        rec = json.loads(path.read_text())
        key = (rec["model"], rec["fold"], rec["seed"])
        meta = by_key.get(key)
        if meta is None:
            continue
        data, gthr = ctx.get(meta)
        p = {k: v.astype(np.float32) for k, v in np.load(path.with_suffix(".npz")).items()}
        m = training.all_metrics(**p, data=data, gate_thresholds=gthr)
        rows.append({"model": rec["model"], "fold": rec["fold"], "seed": rec["seed"], "level": level(rec["sparsity"]),
                     "sparsity": rec["sparsity"],
                     **{k: rec[k] for k in ("nonzero parameters", "MACs (dense kernels)", "MACs on nonzero weights",
                                            ".tflite INT8 (KB)")}, **m})
        if key not in done:                       # the unpruned main run of the same fold and seed, for comparison
            done.add(key)
            base = training.all_metrics(**cv.load_predictions(src, meta), data=data, gate_thresholds=gthr)
            rows.append({"model": rec["model"], "fold": rec["fold"], "seed": rec["seed"], "level": MAIN_RUN,
                         "sparsity": 0.0, "nonzero parameters": rec["parameters"], **base})
    return pd.DataFrame(rows)


def report(dataset):
    runs = evaluate(dataset)
    if runs.empty:
        print(f"No pruning results for {dataset} yet.")
        return
    out = OUT / dataset
    save_csv(runs, out / "all_runs.csv")
    summary, tests, fig = [], [], []
    control = level(0.0)
    for n in [n for n in models.VARIANTS if n in set(runs["model"])]:
        r = runs[runs["model"] == n]
        levels = [MAIN_RUN] + [level(s) for s in sorted(set(r.loc[r["level"] != MAIN_RUN, "sparsity"]))]
        for lv in levels:
            g = r[r["level"] == lv]
            row = {"Model": models.display_name(n), "Sparsity": lv, "Runs": len(g)}
            for c in ["nonzero parameters", "MACs (dense kernels)", "MACs on nonzero weights", ".tflite INT8 (KB)", *SHOWN]:
                if c in g and g[c].notna().any():
                    row[c] = cv.mean_std(g[c].dropna(), 0 if "MACs" in c or "parameters" in c else 2)
            summary.append(row)
            for c in SHOWN[:3]:
                fig.append({"model": models.display_name(n), "sparsity": lv, "metric": c, "mean": g[c].mean(),
                            "std": g[c].std(ddof=1), "nonzero parameters": g["nonzero parameters"].mean(),
                            "runs": len(g)})
        for c in SHOWN[:3]:
            t = r.pivot_table(index=["fold", "seed"], columns="level", values=c)
            block = []
            for lv in [x for x in levels if x not in (MAIN_RUN, control)]:
                for ref in (control, MAIN_RUN):
                    if lv in t and ref in t and len(t[[lv, ref]].dropna()) >= 2:
                        ref_label = f"{control} (same fine-tuning, no pruning)" if ref == control else ref
                        block.append(cv.paired(t[lv], t[ref], f"{models.display_name(n)} {lv}",
                                               f"{models.display_name(n)} {ref_label}", c))
            tests += cv.with_holm(block) if block else []
    save_csv(summary, out / "summary.csv")
    print(f"\n===== Pruning, {dataset} (mean ± std over runs) =====")
    s = pd.DataFrame(summary)
    print_columns(s.assign(Model=s["Model"] + " / " + s["Sparsity"]).drop(columns="Sparsity"))
    if tests:
        cv.print_tests(save_csv(tests, out / "paired_tests.csv"))
    save_csv(fig, cv.figures_dir() / f"pruning_{dataset}.csv")
    print(f"Saved to {out}")
