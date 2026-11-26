"""
Quantization of the saved cross-validation models without retraining.

run(): every saved run of results/cv_<dataset>/runs/ is converted again in each variant of config.QUANT_VARIANTS
and evaluated on the run's own threshold-tuning and test windows:
  fp16                float16 weights (TFLite float16 quantization), float32 compute
  weight_int8_tflite  TFLite dynamic-range ("weight-only") INT8. The converter leaves weight tensors with fewer than
                      1,024 elements in float32, which is every weight tensor of most of these models; the run
                      records how many it quantized, and a model with none quantized counts as its FP32 model
  weight_int8_sim     weight-only INT8 simulated on the FP32 TFLite graph: every conv / fully-connected weight
                      tensor rounded to symmetric per-output-channel int8 and back, float compute
  int8_balanced       full INT8 as in the main runs, calibrated on windows of which config.CALIB_FALL_SHARE are
                      accident windows (the main runs draw 1,000 windows uniformly: ~2.5 % falls on SisFall)
  int16x8             int16 activations and I/O, int8 weights (TFLite 16x8 mode)
The main runs' FP32 and full-INT8 predictions are taken from the run's .npz ("fp32", "int8").
Saved per run in results/quantization/<dataset>/runs/: <run>.npz (P(fall) per variant), <run>.json (sizes,
quantization parameters, quantized tensors, conversion errors), written last.

report(): metrics of every variant from the saved predictions, without and with the gate, at three thresholds:
  "tuned"           THRESHOLD_GRID (every INT8 output level k / 256) on the run's tuning trials, as in the main runs
  "grid 0.50-0.99"  THRESHOLD_GRID_CAPPED, the grid of protocol v1: shows what its binding lower end cost
  "FP32 threshold"  the FP32 model's tuned threshold reused without re-tuning
-> all_runs.csv, summary.csv, paired_tests.csv (each variant vs FP32 and vs full INT8), equivalence_tests.csv
(TOST vs FP32), figures/quantization_<dataset>.csv.

diagnose() (where the full-INT8 accuracy loss comes from), per saved run of the given models:
  - the full-INT8 model is converted again and compared with the saved predictions. The calibration runs the float
    model with oneDNN, whose summation order depends on the machine and thread count, so ranges can differ in the
    last float digits; a model whose INT8 outputs then change (share of windows, G-mean with the same threshold)
    is numerically sensitive, a large systematic change would point to a bug
  - input quantization: scale / zero point, share of input values outside the int8 range (fall vs ADL windows)
  - output: share of test windows at the highest / lowest INT8 output level per class, mean P(fall) of FP32 and
    INT8, Spearman correlation of FP32 and INT8 window scores, trial AUC
  - thresholds: tuned on THRESHOLD_GRID (full) and on the capped grid 0.50-0.99 of protocol v1, with the resulting
    trial G-mean; how many runs the capped grid's ends were binding for
  - per-tensor saturation of every int8 activation tensor on sampled test windows (fall / ADL): share of values
    at the upper limit, and at the lower limit where that is not the real value 0 (ReLU outputs); ops with a fixed
    output range (LOGISTIC, TANH, SOFTMAX) are marked, their limits are not calibration clipping
-> results/quantization/diagnosis/<dataset>_runs.csv, <dataset>_summary.csv, <dataset>_<model>_tensors.csv,
   <dataset>_output_levels.csv (histograms of FP32 and INT8 output levels per class, for plotting)
"""
import json

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

import config as cfg
import cv
import deployment
import gate
import models
import preprocessing as pre
import training
from utils import print_columns, save_csv

OUT = cfg.RESULTS_DIR / "quantization"
LABELS = {"fp32": "FP32", "int8": "Full INT8", "fp16": "FP16 weights",
          "weight_int8_tflite": "Weight-only INT8 (TFLite)", "weight_int8_sim": "Weight-only INT8 (simulated)",
          "int8_balanced": "Full INT8, balanced calibration", "int16x8": "INT16 activations, INT8 weights"}
EXACT_FLOAT16 = {"int8_balanced"}          # outputs are multiples of 1/256, stored exactly as float16
FLOAT_KEY = "float activation tensors (0 = fully integer)"
SIZE_KEYS = ("peak activation memory (KB)", FLOAT_KEY)     # record fields read from the converted graph
FIXED_RANGE_OPS = {"LOGISTIC", "TANH", "SOFTMAX"}
CAPPED = "grid 0.50-0.99"
SHOWN = ["Trial G-mean (tuned)", f"Trial G-mean ({CAPPED})", "Trial G-mean (FP32 threshold)",
         "Gated Trial G-mean (tuned)", f"Gated Trial G-mean ({CAPPED})", "Trial AUC"]


def out_dir(dataset):
    return OUT / f"{dataset}{'_valthr' if cfg.THRESHOLD_PROTOCOL == 'validation' else ''}"


def stem(dataset, meta):
    return out_dir(dataset) / "runs" / f"{meta['model']}_fold{meta['fold']}_seed{meta['seed']}"


def _pack(p, variant):
    if variant in EXACT_FLOAT16:
        return p.astype(np.float16)
    return np.round(np.clip(p, 0, 1) * 65535).astype(np.uint16)       # 1.5e-5 resolution


def _unpack(a):
    return a.astype(np.float32) / 65535 if a.dtype == np.uint16 else a.astype(np.float32)


def make_variant(model, variant, data):
    """(tflite bytes or None, record) of one quantization variant of a trained Keras model."""
    rec = {}
    try:
        if variant == "fp16":
            tfl = deployment.convert(model, "fp16")
        elif variant == "weight_int8_tflite":
            tfl = deployment.convert(model, "weight_only")
            w = deployment.weight_tensors(tfl)
            rec["weight tensors"] = len(w)
            rec["weight tensors quantized by TFLite"] = sum(d == "int8" for _, d, _ in w.values())
        elif variant == "weight_int8_sim":
            tfl, info = deployment.simulate_weight_only_int8(deployment.convert(model, "fp32"))
            rec.update(info)
        elif variant == "int8_balanced":
            tfl = deployment.convert(model, "int8", data["X_tr"], data["y_tr"], cfg.CALIB_FALL_SHARE)
            rec["calibration"] = deployment.calibration_set(data["X_tr"], data["y_tr"], cfg.CALIB_FALL_SHARE)[1]
        elif variant == "int16x8":
            tfl = deployment.convert(model, "int16x8", data["X_tr"])
        else:
            raise ValueError(f"unknown variant {variant}")
    except Exception as e:                                       # noqa: BLE001 - recorded, the run goes on
        return None, {"error": f"{type(e).__name__}: {e}".splitlines()[0][:300]}
    rec.update({".tflite (KB)": round(len(tfl) / 1024, 2),
                "constant data (KB)": round(deployment.constant_bytes(tfl) / 1024, 2),
                "peak activation memory (KB)": round(deployment.peak_activation_bytes(tfl) / 1024, 2),
                **deployment.quantization_info(tfl)})
    if variant in ("int8_balanced", "int16x8"):              # ops the converter left in float (NEG, a SPLIT ...)
        rec[FLOAT_KEY] = int(deployment.float_activation_tensors(tfl))
    return tfl, rec


def refresh_sizes(dataset, names, folds=None, seeds=None, variants=None):
    """Converts every saved run's variants again and rewrites the record fields that are read from the converted
    graph (SIZE_KEYS); predictions, file sizes and the other fields stay. For records written before
    deployment.graph_info read the plain graph: with XNNPACK's delegate nodes in the op list the peak activation
    memory of one and the same graph differed from run to run (Musci 2021 on SisFall: 3.4 - 6.9 KB). Resumable."""
    variants = variants or cfg.QUANT_VARIANTS
    models.register_custom_layers()
    src = cv.out_dir_for(dataset)
    metas = [m for m in cv.run_metas(src, cfg.EPOCHS) if m["model"] in names
             and (folds is None or m["fold"] in folds) and (seeds is None or m["seed"] in seeds)]
    metas.sort(key=lambda m: (m["fold"], m["model"], m["seed"]))

    def pending(rec):
        return [v for v in variants if v in rec["variants"] and "error" not in rec["variants"][v]
                and not rec["variants"][v].get("sizes refreshed")]

    todo = [m for m in metas if stem(dataset, m).with_suffix(".json").exists()
            and pending(json.loads(stem(dataset, m).with_suffix(".json").read_text()))]
    print(f"Refreshing graph sizes of variants {variants}, {dataset}: {len(todo)} of {len(metas)} saved runs to do")
    if not todo:
        return
    trials = pre.load_trials(dataset)
    subject_folds = cv.fold_subjects(sorted({t["subject"] for t in trials}))
    data, data_key = None, None
    for i, meta in enumerate(todo, 1):
        key = (meta["fold"], json.dumps(meta.get("subjects"), sort_keys=True))
        if key != data_key:
            data, data_key = None, None
            data, data_key = cv.run_split(trials, meta, subject_folds), key
        path = stem(dataset, meta).with_suffix(".json")
        rec = json.loads(path.read_text())
        model = keras.models.load_model(cv.stem_of(src, meta).with_suffix(".keras"))
        print(f"[{i}/{len(todo)}] {meta['model']} fold {meta['fold']} seed {meta['seed']}")
        for v in pending(rec):
            old = rec["variants"][v]
            tfl, info = make_variant(model, v, data)
            if tfl is None:
                print(f"  {v:20s} conversion failed now: {info['error']}")
                continue
            changes = {k: (old.get(k), info[k]) for k in SIZE_KEYS if k in info and old.get(k) != info[k]}
            old.update({k: info[k] for k in SIZE_KEYS if k in info})
            old["sizes refreshed"] = True
            print(f"  {v:20s} " + (", ".join(f"{k}: {a} -> {b}" for k, (a, b) in changes.items()) or "unchanged"))
        path.write_text(json.dumps(rec, indent=1))


def run(dataset, names, folds=None, seeds=None, variants=None):
    variants = variants or cfg.QUANT_VARIANTS
    models.register_custom_layers()
    src = cv.out_dir_for(dataset)
    (out_dir(dataset) / "runs").mkdir(parents=True, exist_ok=True)
    metas = [m for m in cv.run_metas(src, cfg.EPOCHS) if m["model"] in names
             and (folds is None or m["fold"] in folds) and (seeds is None or m["seed"] in seeds)]
    metas.sort(key=lambda m: (m["fold"], m["model"], m["seed"]))
    todo = []
    for m in metas:
        path = stem(dataset, m).with_suffix(".json")
        have = json.loads(path.read_text())["variants"] if path.exists() else {}
        if [v for v in variants if v not in have]:
            todo.append(m)
    print(f"Quantization variants {variants}, {dataset}: {len(todo)} of {len(metas)} saved runs to do")
    if not todo:
        return
    trials = pre.load_trials(dataset)
    subject_folds = cv.fold_subjects(sorted({t["subject"] for t in trials}))
    data, data_key = None, None
    for i, meta in enumerate(todo, 1):
        key = (meta["fold"], json.dumps(meta.get("subjects"), sort_keys=True))
        if key != data_key:
            data, data_key = None, None                          # free the previous fold's arrays first
            data, data_key = cv.run_split(trials, meta, subject_folds), key
        s = stem(dataset, meta)
        rec = json.loads(s.with_suffix(".json").read_text()) if s.with_suffix(".json").exists() else \
            {k: meta[k] for k in ("dataset", "model", "fold", "seed")} | {"variants": {}}
        preds = dict(np.load(s.with_suffix(".npz"))) if s.with_suffix(".npz").exists() else {}
        model = keras.models.load_model(cv.stem_of(src, meta).with_suffix(".keras"))
        X_tune, X_te = data["X_tune"].astype(np.float32), data["X_te"].astype(np.float32)
        print(f"[{i}/{len(todo)}] {meta['model']} fold {meta['fold']} seed {meta['seed']}")
        for v in [v for v in variants if v not in rec["variants"]]:
            tfl, info = make_variant(model, v, data)
            if tfl is not None and info.get("weight tensors quantized by TFLite") == 0:
                info["predictions"] = "FP32 (no weight tensor quantized)"
            elif tfl is not None:
                preds[f"{v}_tr"] = _pack(deployment.predict(tfl, X_tune), v)
                preds[f"{v}_te"] = _pack(deployment.predict(tfl, X_te), v)
            rec["variants"][v] = info
            print(f"  {v:20s} {info.get('error') or info.get('predictions') or 'done'}")
        np.savez_compressed(s.with_suffix(".npz"), **preds)
        s.with_suffix(".json").write_text(json.dumps(rec, indent=1))   # written last


# ---------------------------------------------------------------- metrics

def _settings(p_tr, p_te, fp32_thr, data):
    """Trial / window metrics at the three thresholds of one prediction set (one gate setting)."""
    out = {}
    for label, grid in (("tuned", None), (CAPPED, cfg.THRESHOLD_GRID_CAPPED)):
        m = training.tuned_metrics(p_tr, p_te, data, grid)
        out[f"threshold ({label})"] = m.pop("threshold")
        out.update({f"{k} ({label})": v for k, v in m.items()})
    m = training.evaluate(p_te, data["y_te"], data["test"], fp32_thr)[1]
    out.update({f"{k} (FP32 threshold)": v for k, v in m.items()})
    return out


def variant_predictions(dataset, meta, rec):
    """{variant: (P on tuning windows, P on test windows)} of one run, incl. the main runs' FP32 and INT8."""
    base = cv.load_predictions(cv.out_dir_for(dataset), meta)
    preds = {"fp32": (base["p_tr"], base["p_te"]), "int8": (base["p8_tr"], base["p8_te"])}
    q = np.load(stem(dataset, meta).with_suffix(".npz"))
    for v, info in rec["variants"].items():
        if "error" in info:
            continue
        if info.get("predictions", "").startswith("FP32"):
            preds[v] = preds["fp32"]
        elif f"{v}_tr" in q:
            preds[v] = (_unpack(q[f"{v}_tr"]), _unpack(q[f"{v}_te"]))
    return preds


def evaluate(dataset):
    metas, ctx = cv.saved_runs(dataset, cfg.EPOCHS, cv.out_dir_for(dataset))
    rows = []
    for meta in metas:
        path = stem(dataset, meta).with_suffix(".json")
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        data, gthr = ctx.get(meta)
        pass_tr = gate.passes(data["gate_tr"], data["train"], gthr)
        pass_te = gate.passes(data["gate_te"], data["test"], gthr)
        preds = variant_predictions(dataset, meta, rec)
        fp32 = preds["fp32"]
        thr = {False: training.select_threshold(fp32[0], data["train"]),
               True: training.select_threshold(fp32[0] * pass_tr, data["train"])}
        for v, (a, b) in preds.items():
            info = rec["variants"].get(v, {})
            row = {"model": meta["model"], "fold": meta["fold"], "seed": meta["seed"], "variant": v}
            for gated in (False, True):
                pa, pb = (a * pass_tr, b * pass_te) if gated else (a, b)
                row.update({f"{'Gated ' if gated else ''}{k}": val for k, val in _settings(pa, pb, thr[gated], data).items()})
            row["Trial AUC"] = training.trial_roc(b, data["test"])[1]
            for k in (".tflite (KB)", "constant data (KB)", "peak activation memory (KB)", FLOAT_KEY,
                      "weight tensors quantized by TFLite", "weight tensors rounded to int8"):
                if k in info:
                    row[k] = info[k]
            rows.append(row)
        for v, info in rec["variants"].items():
            if "error" in info:
                rows.append({"model": meta["model"], "fold": meta["fold"], "seed": meta["seed"], "variant": v,
                             "error": info["error"]})
    return pd.DataFrame(rows)


def report(dataset):
    runs = evaluate(dataset)
    if runs.empty:
        print(f"No quantization results for {dataset} yet.")
        return
    out = out_dir(dataset)
    save_csv(runs, out / "all_runs.csv")
    ok = runs[runs["error"].isna()] if "error" in runs else runs
    names = [n for n in models.VARIANTS if n in set(ok["model"])]
    order = ["fp32", "int8", *cfg.QUANT_VARIANTS]
    metric_cols = [c for c in ok.columns if c not in ("model", "fold", "seed", "variant", "error")]
    summary = []
    for n in names:
        for v in [v for v in order if v in set(ok["variant"])]:
            g = ok[(ok["model"] == n) & (ok["variant"] == v)]
            if g.empty:
                continue
            row = {"Model": models.display_name(n), "Variant": LABELS.get(v, v), "Runs": len(g)}
            for c in metric_cols:
                vals = g[c].dropna()
                if not len(vals):
                    continue
                if c == FLOAT_KEY:                           # runs whose integer graph kept an op in float
                    row[c] = f"{int((vals > 0).sum())} of {len(vals)} runs"
                else:
                    row[c] = cv.mean_std(vals, 4 if "AUC" in c else 2)
            errors = runs[(runs["model"] == n) & (runs["variant"] == v)]
            if "error" in errors and errors["error"].notna().any():
                row["Conversion errors"] = int(errors["error"].notna().sum())
            summary.append(row)
        for v in [v for v in order if v not in set(ok[ok["model"] == n]["variant"])]:
            errs = runs[(runs["model"] == n) & (runs["variant"] == v)] if "error" in runs else pd.DataFrame()
            if not errs.empty:
                summary.append({"Model": models.display_name(n), "Variant": LABELS.get(v, v), "Runs": 0,
                                "Conversion errors": len(errs), "Error": errs["error"].iloc[0]})
    save_csv(summary, out / "summary.csv")
    print(f"\n===== Quantization variants, {dataset} (mean ± std over runs) =====")
    shown = pd.DataFrame(summary)
    print_columns(shown[[c for c in ["Model", "Variant", "Runs", *SHOWN, ".tflite (KB)", "Conversion errors"]
                         if c in shown]].assign(Model=shown["Model"] + " / " + shown["Variant"]).drop(columns="Variant"))

    # paired tests and TOST: every variant vs FP32 and vs the main runs' full INT8, same runs
    tests, eq = [], []
    for metric in ("Trial G-mean (tuned)", f"Trial G-mean ({CAPPED})", "Gated Trial G-mean (tuned)"):
        for ref in ("fp32", "int8"):
            block = []
            for n in names:
                table = ok[ok["model"] == n].pivot_table(index=["fold", "seed"], columns="variant", values=metric)
                if ref not in table:
                    continue
                for v in [v for v in order if v in table and v != ref and (ref != "int8" or v != "fp32")]:
                    pair = table[[ref, v]].dropna()
                    if len(pair) >= 2:
                        block.append(cv.paired(pair[v], pair[ref], f"{models.display_name(n)} {LABELS[v]}",
                                               f"{models.display_name(n)} {LABELS[ref]}", metric))
                        if ref == "fp32" and metric != f"Trial G-mean ({CAPPED})":
                            eq += [cv.tost(pair[v], pair[ref], f"{models.display_name(n)} {LABELS[v]}",
                                           f"{models.display_name(n)} {LABELS[ref]}", metric, mg)
                                   for mg in cfg.TOST_MARGINS]
            tests += cv.with_holm(block) if block else []
    if tests:
        cv.print_tests(save_csv(tests, out / "paired_tests.csv"))
    if eq:
        save_csv(eq, out / "equivalence_tests.csv")
    fig = []
    for (n, v), g in ok.groupby(["model", "variant"], sort=False):
        for c in ("Trial G-mean (tuned)", f"Trial G-mean ({CAPPED})", "Gated Trial G-mean (tuned)", "Trial AUC"):
            fig.append({"model": models.display_name(n), "variant": LABELS.get(v, v), "metric": c,
                        "mean": g[c].mean(), "std": g[c].std(ddof=1), "runs": len(g)})
    save_csv(fig, cv.figures_dir() / f"quantization_{dataset}.csv")
    print(f"Saved to {out}")


# ---------------------------------------------------------------- diagnosis

def _levels(p):
    """INT8 softmax output level 0..255 of P(fall) (P = level / 256), also for FP32 probabilities (binned)."""
    return np.clip(np.floor(np.asarray(p, np.float64) * 256 + 1e-9), 0, 255).astype(int)


def tensor_saturation(tflite, X_fall, X_adl):
    """Per integer activation tensor: scale, zero point and the share of values at the limits, on fall / ADL
    windows. Lower limit counted only where it is not the real value 0 (i.e. zero point above the minimum)."""
    # built-in kernels only: tensors of ops handed to the XNNPACK delegate are not kept in the interpreter
    it = tf.lite.Interpreter(model_content=tflite, num_threads=1, experimental_preserve_all_tensors=True,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ops = it._get_ops_details()
    td = {d["index"]: d for d in it.get_tensor_details()}
    producer = {t: op["op_name"] for op in ops for t in op["outputs"]}
    acts = [t for t in producer if np.dtype(td[t]["dtype"]) in (np.int8, np.int16)]
    put, _ = deployment._io(it)
    put(X_fall[:1])
    it.invoke()
    readable = []
    for t in acts:
        try:
            it.get_tensor(t)
            readable.append(t)
        except ValueError:                                       # no data kept for this tensor
            pass
    acts = readable
    stats_ = {}
    for cls, X in (("fall", X_fall), ("ADL", X_adl)):
        hi = {t: 0 for t in acts}
        lo = {t: 0 for t in acts}
        n = {t: 0 for t in acts}
        for x in X:
            put(x[None])
            it.invoke()
            for t in acts:
                v = it.get_tensor(t)
                lim = np.iinfo(v.dtype)
                hi[t] += int((v == lim.max).sum())
                if td[t]["quantization"][1] > lim.min:
                    lo[t] += int((v == lim.min).sum())
                n[t] += v.size
        stats_[cls] = {t: (100 * hi[t] / max(n[t], 1), 100 * lo[t] / max(n[t], 1)) for t in acts}
    rows = []
    for t in acts:
        rows.append({"tensor index": t, "tensor": td[t]["name"], "op": producer[t],
                     "fixed output range": producer[t] in FIXED_RANGE_OPS,
                     "dtype": np.dtype(td[t]["dtype"]).name, "scale": float(td[t]["quantization"][0]),
                     "zero point": int(td[t]["quantization"][1]),
                     **{f"at upper limit, {c} (%)": stats_[c][t][0] for c in stats_},
                     **{f"at lower limit, {c} (%)": stats_[c][t][1] for c in stats_}})
    return rows


def diagnose(dataset, names, max_runs=None, sample_windows=100):
    models.register_custom_layers()
    src = cv.out_dir_for(dataset)
    metas, ctx = cv.saved_runs(dataset, cfg.EPOCHS, src)
    metas = sorted([m for m in metas if m["model"] in names], key=lambda m: (m["fold"], m["model"], m["seed"]))
    if max_runs:
        keep = {n: [m for m in metas if m["model"] == n][:max_runs] for n in names}
        metas = [m for n in names for m in keep[n]]
    if not metas:
        print(f"No saved {dataset} runs of {names}")
        return
    out = OUT / "diagnosis"
    trials = ctx.trials
    subject_folds = ctx.folds
    rows, hist, tensors = [], [], {}
    data, data_key = None, None
    rng = np.random.default_rng(0)
    for i, meta in enumerate(metas, 1):
        key = (meta["fold"], json.dumps(meta.get("subjects"), sort_keys=True))
        if key != data_key:
            data, data_key = None, None
            data, data_key = cv.run_split(trials, meta, subject_folds), key
        print(f"[diagnosis {i}/{len(metas)}] {meta['model']} fold {meta['fold']} seed {meta['seed']}")
        mdata, gthr = ctx.get(meta)
        saved = cv.load_predictions(src, meta)
        model = keras.models.load_model(cv.stem_of(src, meta).with_suffix(".keras"))
        int8, mode, _ = deployment.quantize_int8(model, data["X_tr"])
        X_te = data["X_te"].astype(np.float32)
        p8_now = deployment.predict(int8, X_te)
        q = deployment.quantization_info(int8)
        s_in, z_in = q["input"]["scale"], q["input"]["zero point"]
        lo_in, hi_in = (-128 - z_in) * s_in, (127 - z_in) * s_in
        y = data["y_te"] == 1
        outside = (X_te < lo_in) | (X_te > hi_in)
        lv8, lv32 = _levels(saved["p8_te"]), _levels(saved["p_te"])
        diff = np.abs(p8_now - saved["p8_te"])
        row = {"model": meta["model"], "fold": meta["fold"], "seed": meta["seed"], "INT8 mode": mode,
               "INT8 reproduces saved predictions": bool(diff.max() < 1e-6),
               "max |INT8 now - saved|": float(diff.max()),
               "test windows with another INT8 output (%)": float(100 * (diff > 1e-6).mean()),
               "INT8 trial G-mean, saved model": training.tuned_metrics(saved["p8_tr"], saved["p8_te"], mdata)["Trial G-mean"],
               "INT8 trial G-mean, converted again (same threshold)":
                   training.evaluate(p8_now, mdata["y_te"], mdata["test"],
                                     training.select_threshold(saved["p8_tr"], mdata["train"]))[1]["Trial G-mean"],
               "input scale": s_in, "input zero point": z_in,
               "input values outside int8 range, fall windows (%)": 100 * outside[y].mean(),
               "input values outside int8 range, ADL windows (%)": 100 * outside[~y].mean(),
               "output scale": q["output"]["scale"], "output zero point": q["output"]["zero point"]}
        for cls, msk in (("fall", y), ("ADL", ~y)):
            row[f"INT8 at highest level, {cls} windows (%)"] = 100 * (lv8[msk] == 255).mean()
            row[f"INT8 at lowest level, {cls} windows (%)"] = 100 * (lv8[msk] == 0).mean()
            row[f"mean P FP32, {cls} windows"] = float(saved["p_te"][msk].mean())
            row[f"mean P INT8, {cls} windows"] = float(saved["p8_te"][msk].mean())
            for kind, lv in (("FP32", lv32), ("INT8", lv8)):
                counts = np.bincount(lv[msk], minlength=256)
                hist += [{"model": meta["model"], "kind": kind, "class": cls, "level": k, "count": int(c)}
                         for k, c in enumerate(counts) if c]
        row["Spearman FP32 vs INT8 (test windows)"] = float(stats.spearmanr(saved["p_te"], saved["p8_te"])[0])
        row["FP32 trial AUC"] = training.trial_roc(saved["p_te"], mdata["test"])[1]
        row["INT8 trial AUC"] = training.trial_roc(saved["p8_te"], mdata["test"])[1]
        for kind, (a, b) in (("FP32", (saved["p_tr"], saved["p_te"])), ("INT8", (saved["p8_tr"], saved["p8_te"]))):
            for label, grid in (("full grid", None), (CAPPED, cfg.THRESHOLD_GRID_CAPPED)):
                m = training.tuned_metrics(a, b, mdata, grid)
                row[f"{kind} threshold ({label})"] = m["threshold"]
                row[f"{kind} trial G-mean ({label})"] = m["Trial G-mean"]
                row[f"{kind} trial sensitivity ({label})"] = m["Trial sensitivity"]
                row[f"{kind} trial specificity ({label})"] = m["Trial specificity"]
        row["INT8 threshold at an end of the capped grid (0.50 / 0.99)"] = row[f"INT8 threshold ({CAPPED})"] in (
            cfg.THRESHOLD_GRID_CAPPED[0], cfg.THRESHOLD_GRID_CAPPED[-1])
        # per-tensor saturation on sampled test windows
        fall_idx, adl_idx = np.flatnonzero(y), np.flatnonzero(~y)
        pick = lambda idx: X_te[rng.choice(idx, min(sample_windows, len(idx)), replace=False)]
        for r in tensor_saturation(int8, pick(fall_idx), pick(adl_idx)):
            tensors.setdefault(meta["model"], {}).setdefault(r["tensor index"], []).append(r)
        rows.append(row)
    runs = save_csv(rows, out / f"{dataset}_runs.csv")
    summary = []
    for n in [n for n in names if n in set(runs["model"])]:
        g = runs[runs["model"] == n]
        row = {"Model": models.display_name(n), "Runs": len(g)}
        for c in [c for c in runs.columns if c not in ("model", "fold", "seed", "INT8 mode")]:
            if runs[c].dtype == bool:
                row[c] = f"{int(g[c].sum())} of {len(g)}"
            else:
                row[c] = cv.mean_std(g[c], 4)
        summary.append(row)
        t_rows = []
        for idx, rs in tensors.get(n, {}).items():
            first = rs[0]
            t_rows.append({k: first[k] for k in ("tensor index", "tensor", "op", "fixed output range", "dtype")}
                          | {k: float(np.mean([r[k] for r in rs])) for k in first
                             if k.startswith("at ") or k in ("scale", "zero point")} | {"runs": len(rs)})
        t = pd.DataFrame(t_rows)
        if not t.empty:
            t["max over classes, upper limit (%)"] = t[[c for c in t if c.startswith("at upper")]].max(axis=1)
            t = t.sort_values("max over classes, upper limit (%)", ascending=False)
            save_csv(t, out / f"{dataset}_{n}_tensors.csv")
            calibrated = t[~t["fixed output range"]]
            row["activation tensors"] = len(t)
            row["calibrated tensors with > 1 % of values at a limit"] = int(
                ((calibrated[[c for c in t if c.startswith("at ")]] > 1).any(axis=1)).sum())
            row["largest share at a limit, calibrated tensors (%)"] = float(
                calibrated[[c for c in t if c.startswith("at ")]].max().max()) if len(calibrated) else 0.0
    save_csv(summary, out / f"{dataset}_summary.csv")
    save_csv(pd.DataFrame(hist).groupby(["model", "kind", "class", "level"], as_index=False)["count"].sum(),
             out / f"{dataset}_output_levels.csv")
    print_columns(pd.DataFrame(summary))
    print(f"Saved to {out}")
