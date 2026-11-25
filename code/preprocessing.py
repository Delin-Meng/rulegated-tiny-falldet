"""
Data preprocessing following the team's common rules (settings in config.py):
windowing, window labels, self-check, train/test split by subject, normalisation.
KFall (100 Hz) is the main dataset; SisFall (200 Hz -> 100 Hz, Musci et al. temporal labels) is used
for cross-dataset checks. Parsed raw signals are cached in results/cache/, so only the first run reads
the CSV / txt files; windows are always rebuilt from the cache with the current config.
"""
import pickle
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cfg
import gate


def make_trial(subject, file, is_fall, signal, onset=None, impact=None):
    """Crop a fall trial after impact + FALL_EXT rows, cut windows and label them.
    Also stores each window's gate features (gate.FEATURES, from the raw signal)."""
    in_fall = np.zeros(len(signal), dtype=bool)
    if is_fall:
        end = impact + cfg.FALL_EXT + 1
        in_fall[onset:end] = True
        signal, in_fall = signal[:end], in_fall[:end]       # drop rows after the falling period
    view = np.lib.stride_tricks.sliding_window_view
    windows = view(signal, cfg.WINDOW, axis=0)[::cfg.STEP]
    y = view(in_fall, cfg.WINDOW)[::cfg.STEP].sum(1) >= cfg.FALL_MIN
    return dict(subject=subject, file=file, type="F" if is_fall else "D", X=windows, y=y.astype(np.int8),
                gate=gate.window_features(signal), rows=len(signal), onset=onset, impact=impact)


KFALL_ADL_TASKS = list(range(1, 20)) + [35, 36]        # KFall ADL task IDs, D01-D21 in this order


def activity_code(trial):
    """F01-F15 for falls, D01-D21 (KFall) / D01-D19 (SisFall) for ADLs. KFall tasks T20-T34 are F01-F15 and its
    ADL tasks T01-T19, T35, T36 are numbered D01-D21 in task order; SisFall file names start with the code."""
    m = re.match(r"S\d+T(\d+)R\d+", trial["file"])
    if not m:
        return trial["file"][:3]
    task = int(m.group(1))
    return f"F{task - 19:02d}" if 20 <= task <= 34 else f"D{KFALL_ADL_TASKS.index(task) + 1:02d}"


fall_type = activity_code


def subject_group(subject):
    """SisFall SE* subjects are elderly, every other subject (SisFall SA*, all of KFall) is young."""
    return "elderly" if subject.startswith("SE") else "young"


# how each dataset is read (written to the data manifest, see records.write_data_manifest)
DESCRIPTION = {
    "kfall": {"files": "sensor_data_new/SA*/S*T*R*.csv; labels label_data_new/SA*_label.xlsx",
              "channels": "AccX, AccY, AccZ (g), GyrX, GyrY, GyrZ (deg/s) as recorded",
              "sampling": "100 Hz, no resampling",
              "labels": "Fall_onset_frame / Fall_impact_frame matched to the FrameCounter column; falls = tasks T20-T34"},
    "sisfall": {"files": "SisFall_dataset/S*/*.txt; labels SisFall_temporally_annotated/S*/*.txt (Musci et al. 2021)",
                "channels": "ADXL345 columns 1-3 x 32/8192 -> g, ITG3200 columns 4-6 x 4000/65536 -> deg/s",
                "sampling": "200 Hz -> 100 Hz by keeping every 2nd sample (plain decimation, no anti-alias filter)",
                "labels": "onset / impact = first / last sample of the FALL (2) segment of the per-sample annotation, "
                          "halved to 100 Hz rows; files without annotation or without a FALL segment are excluded"},
}


def excluded_trials(dataset):
    """Recordings that are not used, with the reason (KFall: none, every recording has its labels)."""
    if dataset != "sisfall":
        return []
    root, rows = cfg.DATASET_DIR / "SisFall", []
    for txt in sorted((root / "SisFall_dataset").glob("S*/*.txt")):
        label_file = root / "SisFall_temporally_annotated" / txt.parent.name / txt.name
        if not label_file.exists():
            rows.append({"Subject": txt.parent.name, "File": txt.name, "Reason": "no temporal annotation file"})
        elif txt.name.startswith("F") and not (np.loadtxt(label_file) == 2).any():
            rows.append({"Subject": txt.parent.name, "File": txt.name, "Reason": "no FALL segment in the annotation"})
    return rows


def read_kfall_labels(path):
    """{(task, trial): (onset_frame, impact_frame)} from one KFall subject's label xlsx."""
    df = pd.read_excel(path)
    task = df.iloc[:, 0].ffill().str.extract(r"\((\d+)\)")[0].astype(int)
    return {(t, int(r)): (int(o), int(i))
            for t, r, o, i in zip(task, df["Trial ID"], df.iloc[:, 3], df.iloc[:, 4])}


def read_kfall(root=cfg.DATASET_DIR / "KFall"):
    """Raw records (subject, file, is_fall, signal [T, 6], onset row, impact row)."""
    records, labels = [], {}
    for csv in tqdm(sorted((root / "sensor_data_new").glob("SA*/*.csv")), desc="Reading KFall", unit="file"):
        subject = csv.parent.name
        if subject not in labels:
            labels[subject] = read_kfall_labels(root / "label_data_new" / f"{subject}_label.xlsx")
        _, task, trial = map(int, re.match(r"S(\d+)T(\d+)R(\d+)", csv.stem).groups())
        is_fall = 20 <= task <= 34
        df = pd.read_csv(csv)
        onset = impact = None
        if is_fall:
            # label frames match FrameCounter (starts at 1)
            onset_f, impact_f = labels[subject][(task, trial)]
            frames = df["FrameCounter"].to_numpy()
            onset, impact = int(np.flatnonzero(frames == onset_f)[0]), int(np.flatnonzero(frames == impact_f)[0])
        records.append((subject, csv.name, is_fall, df[cfg.CHANNELS].to_numpy(np.float32), onset, impact))
    return records


def read_sisfall(root=cfg.DATASET_DIR / "SisFall"):
    """ADXL345 accelerometer (g) + ITG3200 gyroscope (deg/s), downsampled 200 -> 100 Hz.
    Fall onset / impact = first / last sample of the FALL (2) segment of the Musci et al. labels;
    files without temporal labels are skipped."""
    acc_scale, gyr_scale = 32 / 8192, 4000 / 65536       # 13-bit +-16 g, 16-bit +-2000 deg/s
    records = []
    for txt in tqdm(sorted((root / "SisFall_dataset").glob("S*/*.txt")), desc="Reading SisFall", unit="file"):
        label_file = root / "SisFall_temporally_annotated" / txt.parent.name / txt.name
        if not label_file.exists():
            continue
        raw = pd.read_csv(txt, header=None, usecols=range(6)).to_numpy(np.float32)
        signal = np.concatenate([raw[:, :3] * acc_scale, raw[:, 3:6] * gyr_scale], axis=1)[::2]
        is_fall = txt.name.startswith("F")
        onset = impact = None
        if is_fall:
            fall = np.flatnonzero(np.loadtxt(label_file) == 2)
            if len(fall) == 0:
                continue
            onset, impact = int(fall[0]) // 2, int(fall[-1]) // 2
        records.append((txt.parent.name, txt.name, is_fall, np.ascontiguousarray(signal), onset, impact))
    return records


def load_trials(dataset="kfall", use_cache=True):
    """One dict per trial: windows [n, 6, 50], window labels [n]."""
    cache = cfg.CACHE_DIR / f"{dataset}_raw.pkl"
    if use_cache and cache.exists():
        records = pickle.loads(cache.read_bytes())
    else:
        records = {"kfall": read_kfall, "sisfall": read_sisfall}[dataset]()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(records, protocol=pickle.HIGHEST_PROTOCOL))
    return [make_trial(*r) for r in records]


def self_check(trials):
    """Counts to compare with config.KFALL_EXPECTED."""
    return {"Trials": len(trials), "Falls": sum(t["type"] == "F" for t in trials),
            "Subjects": len({t["subject"] for t in trials}),
            "Windows": sum(len(t["y"]) for t in trials),
            "Accident windows": sum(int(t["y"].sum()) for t in trials)}


def stack(trials):
    return np.concatenate([t["X"] for t in trials]), np.concatenate([t["y"] for t in trials])


def split(trials, test_subjects=cfg.KFALL_TEST_SUBJECTS):
    """Subject-wise split without the model inputs: train / test trials (in window order), window labels
    y_tr / y_te and gate features gate_tr / gate_te [n, 3]. Enough to evaluate saved predictions."""
    train = [t for t in trials if t["subject"] not in test_subjects]
    test = [t for t in trials if t["subject"] in test_subjects]
    cat = lambda ts, key: np.concatenate([t[key] for t in ts]) if ts else np.empty((0, 3) if key == "gate" else 0)
    return dict(train=train, test=test, y_tr=cat(train, "y"), y_te=cat(test, "y"),
                gate_tr=cat(train, "gate"), gate_te=cat(test, "gate"))


def split_and_normalize(trials, test_subjects=cfg.KFALL_TEST_SUBJECTS):
    """split() plus the model inputs X_tr / X_te [n, 6, 50, 1], z-scored per axis with training statistics only
    (kept as mean / std to normalise other data the same way, see normalize())."""
    data = split(trials, test_subjects)
    X_tr, X_te = stack(data["train"])[0], stack(data["test"])[0]
    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std = X_tr.std(axis=(0, 2), keepdims=True)
    print(f"Train: {len(data['train'])} trials / {len(X_tr)} windows, "
          f"Test: {len(data['test'])} trials / {len(X_te)} windows")
    return dict(data, X_tr=((X_tr - mean) / std)[..., None], X_te=((X_te - mean) / std)[..., None],
                mean=mean, std=std)


def normalize(trials, mean, std):
    """Model inputs [n, 6, 50, 1] of the given trials with given normalisation statistics."""
    return ((stack(trials)[0] - mean) / std)[..., None].astype(np.float32)


def threshold_subjects(train_subjects, key):
    """Subjects held out for threshold tuning under THRESHOLD_PROTOCOL = "validation" (deterministic per key,
    i.e. per fold, so every model and seed of a fold uses the same ones)."""
    rng = np.random.default_rng(cfg.CV_FOLD_SEED + 1000 + key)
    n = max(1, round(cfg.THRESHOLD_VAL_FRACTION * len(train_subjects)))
    return sorted(rng.choice(sorted(train_subjects), n, replace=False).tolist())


def experiment_split(trials, test_subjects, key=0, fit_subjects=None, tune_subjects=None):
    """Training / threshold-tuning / test sets for one fold under config.THRESHOLD_PROTOCOL.
      X_tr, y_tr, train   model training set (fit subjects), z-scored with its own statistics (mean, std)
      tune                trials, labels and gate features the decision and gate thresholds are tuned on
                          ("train": the fit subjects themselves; "validation": held-out training subjects)
      X_tune              model inputs of the tuning set (the same array as X_tr under "train")
      X_te, y_te, gate_te, test   the test subjects
      subjects            {"fit", "threshold", "test"} subject lists, saved with every run
    fit_subjects / tune_subjects override the protocol (to rebuild the split of a saved run)."""
    test_subjects = sorted(set(test_subjects))
    train_subjects = sorted({t["subject"] for t in trials} - set(test_subjects))
    if fit_subjects is None:
        tune_subjects = (threshold_subjects(train_subjects, key) if cfg.THRESHOLD_PROTOCOL == "validation"
                         else train_subjects)
        fit_subjects = train_subjects if tune_subjects == train_subjects else \
            sorted(set(train_subjects) - set(tune_subjects))
    pick = lambda subjects: [t for t in trials if t["subject"] in set(subjects)]
    cat = lambda ts, k: np.concatenate([t[k] for t in ts])
    fit, tune, test = pick(fit_subjects), pick(tune_subjects), pick(test_subjects)
    X_fit = stack(fit)[0]
    mean, std = X_fit.mean(axis=(0, 2), keepdims=True), X_fit.std(axis=(0, 2), keepdims=True)
    X_tr = ((X_fit - mean) / std)[..., None]
    print(f"Train: {len(fit)} trials / {len(X_tr)} windows, threshold tuning: "
          f"{'the training subjects' if tune_subjects == fit_subjects else f'{len(tune)} trials of {len(tune_subjects)} held-out subjects'}, "
          f"test: {len(test)} trials")
    return dict(X_tr=X_tr, y_tr=cat(fit, "y"), train=fit,
                tune=dict(train=tune, y_tr=cat(tune, "y"), gate_tr=cat(tune, "gate")),
                X_tune=X_tr if tune_subjects == fit_subjects else normalize(tune, mean, std),
                X_te=normalize(test, mean, std), y_te=cat(test, "y"), gate_te=cat(test, "gate"), test=test,
                mean=mean, std=std, subjects={"fit": fit_subjects, "threshold": tune_subjects, "test": test_subjects})


def metric_data(split):
    """The part of an experiment_split() that training.all_metrics needs (tuning set as "train")."""
    return dict(split["tune"], test=split["test"], y_te=split["y_te"], gate_te=split["gate_te"])
