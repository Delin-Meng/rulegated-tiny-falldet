"""Training with the common settings (config.py) and the evaluation metrics shared by all experiments."""
import time

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.keras import TqdmCallback

import config as cfg
import gate
import models


def make_optimizer():
    return keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)


def make_loss():
    return keras.losses.CategoricalFocalCrossentropy()       # Keras defaults


def training_setup(epochs):
    """Optimizer, loss (with all their arguments, e.g. the focal loss alpha / gamma) and fit settings."""
    return {"optimizer": {"class": "Adam", **{k: v for k, v in make_optimizer().get_config().items()
                                              if isinstance(v, (int, float, str, bool, type(None)))}},
            "loss": {"class": "CategoricalFocalCrossentropy", **make_loss().get_config()},
            "batch_size": cfg.BATCH_SIZE, "epochs": epochs, "shuffle": True, "metrics": ["accuracy"],
            "labels": "one-hot [normal, fall]"}


def train_model(name, X_tr, y_tr, epochs=cfg.EPOCHS, seed=cfg.FIXED_SEED, progress=1, validation=None,
                callbacks=(), norm=None):
    """Build model variant `name` with the given seed and train it.
    Returns (model, training minutes, history = {metric: [value per epoch]}).
    progress: 1 = epoch and batch progress bars, 0 = epoch bar only.
    validation: optional (X_val, y_val), evaluated after every epoch (val_loss for callbacks).
    norm: (mean, std) of the z-score of X_tr, for models that need raw values (see models.build)."""
    keras.utils.set_random_seed(seed)
    model = models.build(name, n_axes=len(cfg.CHANNELS), window=cfg.WINDOW, norm=norm)
    model.compile(optimizer=make_optimizer(), loss=make_loss(), metrics=["accuracy"])
    if validation is not None:
        validation = (validation[0], keras.utils.to_categorical(validation[1], 2))
    t0 = time.time()
    history = model.fit(X_tr, keras.utils.to_categorical(y_tr, 2), batch_size=cfg.BATCH_SIZE, epochs=epochs,
                        shuffle=True, verbose=0, validation_data=validation,
                        callbacks=[TqdmCallback(verbose=progress), *callbacks])
    return model, (time.time() - t0) / 60, {k: [float(v) for v in vals] for k, vals in history.history.items()}


def fine_tune(model, X_tr, y_tr, epochs, learning_rate, seed=cfg.FIXED_SEED, progress=0, callbacks=()):
    """Continue training an existing model with the common loss and batch size, Adam at `learning_rate`.
    Returns (training minutes, history = {metric: [value per epoch]})."""
    keras.utils.set_random_seed(seed)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=make_loss(), metrics=["accuracy"])
    t0 = time.time()
    history = model.fit(X_tr, keras.utils.to_categorical(y_tr, 2), batch_size=cfg.BATCH_SIZE, epochs=epochs,
                        shuffle=True, verbose=0, callbacks=[TqdmCallback(verbose=progress), *callbacks])
    return (time.time() - t0) / 60, {k: [float(v) for v in vals] for k, vals in history.history.items()}


def history_rows(history):
    return [{"epoch": i + 1, **{k: v[i] for k, v in history.items()}} for i in range(len(next(iter(history.values()))))]


def predict(model, X):
    return model.predict(X, batch_size=4096, verbose=0)[:, 1]


def sens_spec(pred, truth):
    return 100 * (pred & truth).sum() / truth.sum(), 100 * (~pred & ~truth).sum() / (~truth).sum()


def evaluate(prob, y, trials, threshold=cfg.THRESHOLD):
    """Window and trial level sensitivity / specificity / G-mean (in %) from P(fall) per window.
    A trial is an accident if any of its windows has P(fall) >= threshold.
    Returns (per-trial table, metrics dict)."""
    pred = prob >= threshold
    rows, start = [], 0
    for t in trials:
        n = len(t["y"])
        n_pred = int(pred[start:start + n].sum())
        start += n
        rows.append({"Subject": t["subject"], "File": t["file"], "Type(D/F)": t["type"], "Windows": n,
                     "Predicted accident windows": n_pred,
                     "Trial decision": "Accident" if n_pred > 0 else "Normal"})
    per_trial = pd.DataFrame(rows)
    trial_se, trial_sp = sens_spec((per_trial["Trial decision"] == "Accident").to_numpy(),
                                   (per_trial["Type(D/F)"] == "F").to_numpy())
    win_se, win_sp = sens_spec(pred, y == 1)
    metrics = {"Trial sensitivity": trial_se, "Trial specificity": trial_sp,
               "Trial G-mean": float(np.sqrt(trial_se * trial_sp)),
               "Window sensitivity": win_se, "Window specificity": win_sp,
               "Window G-mean": float(np.sqrt(win_se * win_sp))}
    return per_trial, metrics


def trial_scores(prob, trials):
    """Per trial: largest window P(fall) (the trial is an accident at threshold t iff it is >= t) and is-fall."""
    starts = np.cumsum([0] + [len(t["y"]) for t in trials[:-1]])
    return np.maximum.reduceat(prob, starts), np.array([t["type"] == "F" for t in trials])


def select_threshold(prob, trials, grid=None):
    """Threshold in `grid` (default config.THRESHOLD_GRID) with the highest trial-level G-mean on the given
    (tuning) trials; the lowest one wins ties."""
    grid = cfg.THRESHOLD_GRID if grid is None else grid
    trial_max, is_fall = trial_scores(prob, trials)
    g = [np.sqrt(np.prod(sens_spec(trial_max >= thr, is_fall))) for thr in grid]
    return float(grid[int(np.argmax(g))])


def tuned_metrics(p_tune, p_test, data, grid=None):
    """One setting: threshold tuned on the tuning trials (data["train"]) and applied to the test trials.
    Returns {"threshold", trial / window sensitivity, specificity, G-mean}."""
    thr = select_threshold(p_tune, data["train"], grid)
    return {"threshold": thr, **evaluate(p_test, data["y_te"], data["test"], thr)[1]}


def threshold_sweep(prob, trials, thresholds=cfg.SWEEP_THRESHOLDS):
    """Trial-level sensitivity / specificity / G-mean (%) at each decision threshold."""
    trial_max, is_fall = trial_scores(prob, trials)
    rows = []
    for thr in thresholds:
        se, sp = sens_spec(trial_max >= thr, is_fall)
        rows.append({"Threshold": thr, "Trial sensitivity": se, "Trial specificity": sp,
                     "Trial G-mean": float(np.sqrt(se * sp))})
    return rows


def trial_roc(prob, trials, fpr_grid=cfg.ROC_FPR_GRID):
    """Trial-level ROC (score = largest window P(fall)): TPR interpolated at fpr_grid, and the AUC."""
    trial_max, is_fall = trial_scores(prob, trials)
    fpr, tpr, _ = roc_curve(is_fall, trial_max)
    return np.interp(fpr_grid, fpr, tpr), float(roc_auc_score(is_fall, trial_max))


# result settings: label -> (model, threshold rule); "" is (FP32, threshold 0.5)
SETTINGS = {"": ("fp32", "fixed"), "FP32 tuned": ("fp32", "own"),
            "INT8": ("int8", "fixed"), "INT8 tuned": ("int8", "own"), "INT8 at FP32 threshold": ("int8", "fp32")}


def all_metrics(p_tr, p_te, p8_tr, p8_te, data, gate_thresholds=None, gate_k=cfg.GATE_CONSECUTIVE):
    """Test metrics for every setting in SETTINGS, without and with the gate ("Gated ..." keys).
    Thresholds are tuned on the training data: "tuned" = re-selected on that model's own training
    predictions, "at FP32 threshold" = the INT8 model reuses the FP32 model's tuned threshold.
    Gate: gate.py as the first stage (filter thresholds tuned on the training data at config.GATE_TRIAL_RECALL
    unless given); gated-out windows get P(fall) = 0 and the thresholds are re-tuned on gated predictions.
    gate_k: windows in a row the gate requires."""
    if gate_thresholds is None:
        gate_thresholds = gate.tune_filter(data["gate_tr"], data["y_tr"], data["train"], k=gate_k)
    pass_tr = gate.passes(data["gate_tr"], data["train"], gate_thresholds, gate_k)
    pass_te = gate.passes(data["gate_te"], data["test"], gate_thresholds, gate_k)
    stats = gate.filter_stats(data["gate_te"], data["y_te"], data["test"], gate_thresholds, gate_k)
    out = {"Gate acc threshold (g)": gate_thresholds[0], "Gate gyr threshold (deg/s)": gate_thresholds[1],
           "Gate tilt threshold (deg)": gate_thresholds[2],
           "Gate pass rate, test windows": stats["Pass rate, all windows"],
           "Gate pass rate, ADL test windows": stats["Pass rate, ADL windows"],
           "Gate fall trials kept, test": stats["Fall trials kept"]}
    key = lambda *parts: " ".join(p for p in parts if p)
    for gated in ("", "Gated"):
        probs = {"fp32": (p_tr, p_te), "int8": (p8_tr, p8_te)}
        if gated:
            probs = {k: (a * pass_tr, b * pass_te) for k, (a, b) in probs.items()}
        own = {k: select_threshold(a, data["train"]) for k, (a, _) in probs.items()}
        out[key(gated, "FP32 tuned threshold")], out[key(gated, "INT8 tuned threshold")] = own["fp32"], own["int8"]
        for kind, (_, b) in probs.items():
            out[key(gated, kind.upper(), "Trial AUC")] = trial_roc(b, data["test"])[1]
        for label, (kind, rule) in SETTINGS.items():
            thr = {"fixed": cfg.THRESHOLD, "own": own[kind], "fp32": own["fp32"]}[rule]
            m = evaluate(probs[kind][1], data["y_te"], data["test"], thr)[1]
            out.update({key(gated, label, k): v for k, v in m.items()})
    return out


def format_metrics(metrics):
    """all_metrics for the tables: thresholds as numbers, AUC as a fraction, everything else in %."""
    return {k: f"{v:.4f}" if "AUC" in k else f"{v:.2f}" if "threshold" in k.lower() else f"{v:.1f}%"
            for k, v in metrics.items()}


def curves(p_te, p8_te, data, gate_thresholds):
    """Threshold sweep rows and ROC rows on the test trials for FP32 / INT8, without and with the gate."""
    pass_te = gate.passes(data["gate_te"], data["test"], gate_thresholds)
    sweep, roc = [], []
    for gated in (False, True):
        for kind, prob in (("FP32", p_te), ("INT8", p8_te)):
            p = prob * pass_te if gated else prob
            setting = f"{'Gated ' if gated else ''}{kind}"
            sweep += [{"Setting": setting, **r} for r in threshold_sweep(p, data["test"])]
            tpr, auc = trial_roc(p, data["test"])
            roc += [{"Setting": setting, "FPR": f, "TPR": t, "AUC": auc} for f, t in zip(cfg.ROC_FPR_GRID, tpr)]
    return sweep, roc
