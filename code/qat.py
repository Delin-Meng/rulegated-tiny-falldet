"""
Quantization-aware training (QAT) of the recurrent baselines (P2 item 15): is their full-INT8 loss fixable?
Per saved cross-validation run of config.QAT_MODELS (given folds and seeds, default all folds x seed 0), three arms:
  qat       the trained FP32 model rebuilt with fake quantization at every tensor the INT8 graph stores (input,
            conv / dense outputs, both LSTM matrix products, gates, cell state, hidden state, logits,
            probabilities; weights symmetric per tensor), fine-tuned config.QAT_EPOCHS epochs at
            config.QAT_LEARNING_RATE, then converted to full INT8 with the learned ranges (no calibration set)
  finetune  control: the same fine-tuning without fake quantization, then post-training full INT8 exactly as in the
            main runs. qat vs finetune is the effect of QAT itself, finetune vs ptq that of the extra epochs
  ptq       the main run (its saved FP32 / full-INT8 predictions)
Ranges: fixed where TFLite fixes them (sigmoid [0, 255/256], tanh [-1, 127/128], softmax [0, 255/256]) and for the
input (the clip of the PTQ calibration, CALIB_PERCENTILE of |z| over the training windows); the others are moving
averages (config.QAT_EMA_MOMENTUM) of the batch min / max. Musci 2021: its BatchNorm (after the ReLU) is folded into
the LSTM input weights; its L2 regularisation is not applied during the fine-tuning.
Saved in results/qat/<dataset>/runs/: <run>.npz (P(fall) of the float model and of the INT8 file of each arm),
<run>_qat_int8.tflite, <run>.history.csv, <run>.json (written last: fully integer check, agreement of the INT8 file
with the fake-quantized Keras model, sizes, minutes).
report(): results of the three arms (THRESHOLD_GRID and the capped grid 0.50-0.99 of protocol v1, without and with
gate) and paired differences.
"""
import json

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

import config as cfg
import cv
import deployment
import gate
import models
import preprocessing as pre
import training
from utils import print_columns, save_csv

OUT = cfg.RESULTS_DIR / "qat"
ARMS = {"ptq": "PTQ (main run)", "finetune": "fine-tuned, PTQ", "qat": "QAT"}
SIGMOID, TANH, PROB = (0.0, 255 / 256), (-1.0, 127 / 128), (0.0, 255 / 256)


@keras.saving.register_keras_serializable(package="qat")
class FakeQuant(keras.layers.Layer):
    """8-bit fake quantization of an activation: fixed range, or moving average of the batch min / max while
    training (the range always contains 0)."""

    def __init__(self, fixed=None, momentum=cfg.QAT_EMA_MOMENTUM, **kwargs):
        super().__init__(**kwargs)
        self.fixed = None if fixed is None else (float(fixed[0]), float(fixed[1]))
        self.momentum = momentum

    def build(self, input_shape):
        self.lo = self.add_weight(shape=(), initializer="zeros", trainable=False, name="lo")
        self.hi = self.add_weight(shape=(), initializer="zeros", trainable=False, name="hi")
        self.seen = self.add_weight(shape=(), initializer="zeros", trainable=False, name="seen")

    def call(self, x, training=False):
        if self.fixed is not None:
            lo, hi = self.fixed
        else:
            if training:
                m = tf.where(self.seen > 0, self.momentum, 0.0)
                self.lo.assign(m * self.lo + (1 - m) * tf.minimum(tf.reduce_min(x), 0.0))
                self.hi.assign(m * self.hi + (1 - m) * tf.maximum(tf.reduce_max(x), 0.0))
                self.seen.assign(1.0)
            lo, hi = self.lo, self.hi
        return tf.quantization.fake_quant_with_min_max_vars(x, lo, hi, num_bits=8)

    def get_config(self):
        return {**super().get_config(), "fixed": self.fixed, "momentum": self.momentum}


def fq_weight(w):
    """Symmetric per-tensor int8 fake quantization of a weight tensor (TFLite stores weights this way; per-channel
    weights would need the converter to follow the transposes it inserts)."""
    m = tf.maximum(tf.reduce_max(tf.abs(w)), 1e-8)
    return tf.quantization.fake_quant_with_min_max_vars(w, -m, m, num_bits=8, narrow_range=True)


@keras.saving.register_keras_serializable(package="qat")
class QDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", name="kernel")
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", name="bias")

    def call(self, x):
        return tf.matmul(x, fq_weight(self.kernel)) + self.bias

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.saving.register_keras_serializable(package="qat")
class QConv1D(keras.layers.Layer):
    """Conv1D, stride 1, "same" padding, with fake-quantized weights."""

    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters, self.kernel_size = filters, kernel_size

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.kernel_size, input_shape[-1], self.filters),
                                      initializer="glorot_uniform", name="kernel")
        self.bias = self.add_weight(shape=(self.filters,), initializer="zeros", name="bias")

    def call(self, x):
        return tf.nn.conv1d(x, fq_weight(self.kernel), stride=1, padding="SAME") + self.bias

    def get_config(self):
        return {**super().get_config(), "filters": self.filters, "kernel_size": self.kernel_size}


@keras.saving.register_keras_serializable(package="qat")
class QLSTMCell(keras.layers.Layer):
    """LSTM cell (Keras gate order i, f, c, o; sigmoid / tanh) with fake quantization of both matrix products, the
    pre-activations, the gates, both cell-state terms, the cell state and the hidden state."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]
        self.output_size = units

    def build(self, input_shape):
        d, u = input_shape[-1], self.units
        self.kernel = self.add_weight(shape=(d, 4 * u), initializer="glorot_uniform", name="kernel")
        self.recurrent_kernel = self.add_weight(shape=(u, 4 * u), initializer="orthogonal", name="recurrent_kernel")
        self.bias = self.add_weight(shape=(4 * u,), initializer="zeros", name="bias")
        self.fq_xw, self.fq_hu, self.fq_z = FakeQuant(name="fq_xw"), FakeQuant(name="fq_hu"), FakeQuant(name="fq_z")
        self.fq_sig, self.fq_tanh = FakeQuant(fixed=SIGMOID, name="fq_sig"), FakeQuant(fixed=TANH, name="fq_tanh")
        self.fq_fc, self.fq_ig, self.fq_c = FakeQuant(name="fq_fc"), FakeQuant(name="fq_ig"), FakeQuant(name="fq_c")
        self.fq_h = FakeQuant(fixed=TANH, name="fq_h")

    def call(self, x, states, training=False):
        h, c = states
        t, u = training, self.units
        z = self.fq_xw(tf.matmul(x, fq_weight(self.kernel)) + self.bias, training=t) + \
            self.fq_hu(tf.matmul(h, fq_weight(self.recurrent_kernel)), training=t)
        z = self.fq_z(z, training=t)
        # slices, not tf.split: at the first step c = 0 makes the forget gate unused, and a SPLIT with an unused
        # output is left in float by the converter; an unused slice is simply removed
        zi, zf, zc, zo = z[:, :u], z[:, u:2 * u], z[:, 2 * u:3 * u], z[:, 3 * u:]
        i, f, o = (self.fq_sig(tf.sigmoid(v), training=t) for v in (zi, zf, zo))
        g = self.fq_tanh(tf.tanh(zc), training=t)
        c = self.fq_c(self.fq_fc(f * c, training=t) + self.fq_ig(i * g, training=t), training=t)
        h = self.fq_h(o * self.fq_tanh(tf.tanh(c), training=t), training=t)
        return h, [h, c]

    def get_config(self):
        return {**super().get_config(), "units": self.units}


def _layer(model, cls):
    return [l for l in model.layers if isinstance(l, cls)]


def _head(x):
    x = FakeQuant(name="fq_logits")(QDense(2, name="qdense_out")(x))
    return FakeQuant(fixed=PROB, name="fq_prob")(keras.layers.Softmax()(x))


def build_qat(name, base, clip):
    """QAT copy of a trained cnnlstm / musci2021 model (weights copied, BatchNorm folded)."""
    L = keras.layers
    n_axes, window = base.input_shape[1], base.input_shape[2]
    inp = keras.Input(shape=base.input_shape[1:])
    x = L.Permute((2, 1))(L.Reshape((n_axes, window))(inp))
    x = FakeQuant(fixed=(-clip, clip), name="fq_input")(x)
    lstm = _layer(base, L.LSTM)[0]
    dense_out = base.layers[-1]
    W, U, b = (keras.ops.convert_to_numpy(v) for v in (lstm.cell.kernel, lstm.cell.recurrent_kernel, lstm.cell.bias))
    if name == "cnnlstm":
        conv = _layer(base, L.Conv1D)[0]
        x = QConv1D(conv.filters, conv.kernel_size[0], name="qconv")(x)
        x = FakeQuant(name="fq_conv")(L.ReLU()(x))
        rnn = L.RNN(QLSTMCell(lstm.units), unroll=True, name="qlstm")
        x = _head(rnn(x))
        model = keras.Model(inp, x, name="CNN-LSTM-QAT")
        model.get_layer("qconv").kernel.assign(conv.kernel)
        model.get_layer("qconv").bias.assign(conv.bias)
    elif name == "musci2021":
        d1 = [l for l in _layer(base, L.Dense) if l is not dense_out][0]
        bn = _layer(base, L.BatchNormalization)[0]
        rate = _layer(base, L.Dropout)[0].rate if _layer(base, L.Dropout) else 0.0
        gamma, beta, mean, var = (keras.ops.convert_to_numpy(v) for v in
                                  (bn.gamma, bn.beta, bn.moving_mean, bn.moving_variance))
        a = gamma / np.sqrt(var + bn.epsilon)
        W, b = a[:, None] * W, b + (beta - mean * a) @ W          # BatchNorm(r) = a * r + c folded into the LSTM
        x = QDense(d1.units, name="qdense1")(x)
        x = L.Dropout(rate)(FakeQuant(name="fq_dense1")(L.ReLU()(x)))
        rnn = L.RNN(QLSTMCell(lstm.units), unroll=True, name="qlstm")
        x = _head(L.Dropout(rate)(rnn(x)))
        model = keras.Model(inp, x, name="Musci2021-QAT")
        model.get_layer("qdense1").kernel.assign(d1.kernel)
        model.get_layer("qdense1").bias.assign(d1.bias)
    else:
        raise ValueError(f"no QAT builder for {name}")
    cell = model.get_layer("qlstm").cell
    cell.kernel.assign(W)
    cell.recurrent_kernel.assign(U)
    cell.bias.assign(b)
    model.get_layer("qdense_out").kernel.assign(dense_out.kernel)
    model.get_layer("qdense_out").bias.assign(dense_out.bias)
    return model


def convert_qat(model):
    """Full INT8 TFLite file from the fake-quantization ranges (no calibration set), int8 input and output."""
    inp = keras.Input(batch_shape=(1, *model.input_shape[1:]))
    converter = tf.lite.TFLiteConverter.from_keras_model(keras.Model(inp, model(inp)))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = converter.inference_output_type = tf.int8
    return deployment.run_converter(converter)


def stem(dataset, meta):
    return OUT / dataset / "runs" / f"{meta['model']}_fold{meta['fold']}_seed{meta['seed']}"


def run(dataset, names, folds, seeds, epochs=cfg.QAT_EPOCHS):
    models.register_custom_layers()
    src = cv.out_dir_for(dataset)
    (OUT / dataset / "runs").mkdir(parents=True, exist_ok=True)
    metas = sorted([m for m in cv.run_metas(src, cfg.EPOCHS) if m["model"] in names and m["fold"] in folds
                    and m["seed"] in seeds], key=lambda m: (m["fold"], m["model"], m["seed"]))
    todo = [m for m in metas if not stem(dataset, m).with_suffix(".json").exists()]
    print(f"QAT {dataset}: {len(todo)} of {len(metas)} runs to do ({epochs} epochs, lr {cfg.QAT_LEARNING_RATE})")
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
        print(f"\n[QAT {i}/{len(todo)}] {meta['model']} fold {meta['fold']} seed {meta['seed']}")
        base = keras.models.load_model(cv.stem_of(src, meta).with_suffix(".keras"))
        X_tr, y_tr = data["X_tr"], data["y_tr"]
        X_tune, X_te = data["X_tune"].astype(np.float32), data["X_te"].astype(np.float32)
        clip = float(np.percentile(np.abs(X_tr), cfg.CALIB_PERCENTILE))
        preds = {}

        ctrl = keras.models.clone_model(base)
        ctrl.set_weights(base.get_weights())
        min_ctrl, hist_ctrl = training.fine_tune(ctrl, X_tr, y_tr, epochs, cfg.QAT_LEARNING_RATE, meta["seed"])
        int8_ctrl, mode_ctrl, _ = deployment.quantize_int8(ctrl, X_tr)
        preds.update(finetune_p_tr=training.predict(ctrl, X_tune), finetune_p_te=training.predict(ctrl, X_te),
                     finetune_p8_tr=deployment.predict(int8_ctrl, X_tune), finetune_p8_te=deployment.predict(int8_ctrl, X_te))

        q = build_qat(meta["model"], base, clip)
        min_qat, hist_qat = training.fine_tune(q, X_tr, y_tr, epochs, cfg.QAT_LEARNING_RATE, meta["seed"])
        tfl = convert_qat(q)
        preds.update(qat_p_tr=training.predict(q, X_tune), qat_p_te=training.predict(q, X_te),
                     qat_p8_tr=deployment.predict(tfl, X_tune), qat_p8_te=deployment.predict(tfl, X_te))
        s = stem(dataset, meta)
        s.with_name(s.name + "_qat_int8.tflite").write_bytes(tfl)
        np.savez_compressed(s.with_suffix(".npz"), **{k: (v.astype(np.float16) if "p8" in k else v)
                                                      for k, v in preds.items()})
        save_csv([{"epoch": e + 1, **{f"finetune {k}": v[e] for k, v in hist_ctrl.items()},
                   **{f"qat {k}": v[e] for k, v in hist_qat.items()}} for e in range(epochs)],
                 s.with_suffix(".history.csv"))
        rec = {k: meta[k] for k in ("dataset", "model", "fold", "seed")} | {
            "epochs": epochs, "learning rate": cfg.QAT_LEARNING_RATE, "input clip (z units)": clip,
            "fine-tuning minutes": {"finetune": min_ctrl, "qat": min_qat}, "finetune INT8 mode": mode_ctrl,
            "qat INT8 file": {"float activation tensors (0 = fully integer)": int(deployment.float_activation_tensors(tfl)),
                              ".tflite (KB)": round(len(tfl) / 1024, 2),
                              "peak activation memory (KB)": round(deployment.peak_activation_bytes(tfl) / 1024, 2),
                              **deployment.quantization_info(tfl)},
            "max |INT8 file - fake-quantized Keras model|, test windows":
                float(np.max(np.abs(preds["qat_p8_te"] - preds["qat_p_te"]))),
            "mean |INT8 file - fake-quantized Keras model|, test windows":
                float(np.mean(np.abs(preds["qat_p8_te"] - preds["qat_p_te"])))}
        s.with_suffix(".json").write_text(json.dumps(rec, indent=1))
        print(f"  QAT INT8 file: {rec['qat INT8 file']['float activation tensors (0 = fully integer)']} float tensors, "
              f"max |INT8 - QAT model| {rec['max |INT8 file - fake-quantized Keras model|, test windows']:.4f}")


def refresh_sizes(dataset):
    """Recompute "peak activation memory (KB)" of the saved QAT INT8 files in the run records (for records written
    before deployment.graph_info read the plain graph, without delegate nodes)."""
    for path in sorted((OUT / dataset / "runs").glob("*.json")):
        rec = json.loads(path.read_text())
        tfl_file = path.with_name(path.stem + "_qat_int8.tflite")
        if "qat INT8 file" not in rec or not tfl_file.exists():
            continue
        old = rec["qat INT8 file"].get("peak activation memory (KB)")
        new = round(deployment.peak_activation_bytes(tfl_file.read_bytes()) / 1024, 2)
        if old != new:
            rec["qat INT8 file"]["peak activation memory (KB)"] = new
            path.write_text(json.dumps(rec, indent=1))
        print(f"{path.name}: peak activation memory {old} -> {new} KB")


def report(dataset):
    src = cv.out_dir_for(dataset)
    metas, ctx = cv.saved_runs(dataset, cfg.EPOCHS, src)
    rows = []
    for meta in metas:
        s = stem(dataset, meta)
        if not s.with_suffix(".json").exists():
            continue
        rec = json.loads(s.with_suffix(".json").read_text())
        data, gthr = ctx.get(meta)
        npz = {k: v.astype(np.float32) for k, v in np.load(s.with_suffix(".npz")).items()}
        base = cv.load_predictions(src, meta)
        arms = {"ptq": (base["p_tr"], base["p_te"], base["p8_tr"], base["p8_te"])}
        for arm in ("finetune", "qat"):
            arms[arm] = tuple(npz[f"{arm}_{k}"] for k in ("p_tr", "p_te", "p8_tr", "p8_te"))
        pass_tr = gate.passes(data["gate_tr"], data["train"], gthr)
        pass_te = gate.passes(data["gate_te"], data["test"], gthr)
        for arm, (a, b, a8, b8) in arms.items():
            row = {"model": meta["model"], "fold": meta["fold"], "seed": meta["seed"], "arm": arm}
            for kind, (x, z) in (("float model", (a, b)), ("INT8", (a8, b8))):
                for gated in (False, True):
                    pa, pb = (x * pass_tr, z * pass_te) if gated else (x, z)
                    for label, grid in (("tuned", None), ("grid 0.50-0.99", cfg.THRESHOLD_GRID_CAPPED)):
                        m = training.tuned_metrics(pa, pb, data, grid)
                        pre_ = f"{'Gated ' if gated else ''}{kind}"
                        row[f"{pre_} threshold ({label})"] = m["threshold"]
                        for k in ("Trial sensitivity", "Trial specificity", "Trial G-mean"):
                            row[f"{pre_} {k} ({label})"] = m[k]
                row[f"{kind} Trial AUC"] = training.trial_roc(z, data["test"])[1]
            if arm == "qat":
                row["QAT file fully integer"] = rec["qat INT8 file"]["float activation tensors (0 = fully integer)"] == 0
                row["max |INT8 file - QAT model|"] = rec["max |INT8 file - fake-quantized Keras model|, test windows"]
            rows.append(row)
    if not rows:
        print(f"No QAT results for {dataset} yet.")
        return
    runs = save_csv(rows, OUT / dataset / "all_runs.csv")
    summary, tests = [], []
    cols = [c for c in runs.columns if c not in ("model", "fold", "seed", "arm")]
    for n in [n for n in models.VARIANTS if n in set(runs["model"])]:
        for arm in ARMS:
            g = runs[(runs["model"] == n) & (runs["arm"] == arm)]
            row = {"Model": models.display_name(n), "Arm": ARMS[arm], "Runs": len(g)}
            for c in cols:
                v = g[c].dropna()
                if len(v) and set(map(type, v)) <= {bool, np.bool_}:
                    row[c] = f"{int(sum(v))} of {len(v)}"
                elif len(v):
                    row[c] = cv.mean_std(v.astype(float), 4 if "AUC" in c or "|" in c else 2)
            summary.append(row)
        for metric in ("INT8 Trial G-mean (tuned)", "INT8 Trial G-mean (grid 0.50-0.99)", "Gated INT8 Trial G-mean (tuned)"):
            t = runs[runs["model"] == n].pivot_table(index=["fold", "seed"], columns="arm", values=metric)
            block = [cv.paired(t[a], t[b], f"{models.display_name(n)} {ARMS[a]}", f"{models.display_name(n)} {ARMS[b]}",
                               metric) for a, b in (("qat", "ptq"), ("qat", "finetune"), ("finetune", "ptq"))
                     if a in t and b in t and len(t[[a, b]].dropna()) >= 2]
            tests += cv.with_holm(block) if block else []
    save_csv(summary, OUT / dataset / "summary.csv")
    print(f"\n===== QAT, {dataset} (mean ± std over runs) =====")
    s = pd.DataFrame(summary)
    print_columns(s.assign(Model=s["Model"] + " / " + s["Arm"]).drop(columns="Arm"))
    if tests:
        cv.print_tests(save_csv(tests, OUT / dataset / "paired_tests.csv"))
    print(f"Saved to {OUT / dataset}")
