"""
Reproducibility records written by every experiment:
  - config snapshot per run: every setting of config.py, training setup (optimizer, focal loss alpha / gamma, ...),
    model structure (build arguments, layers, parameter counts), threshold protocol, library versions, CPU,
    and a fingerprint of the code (SHA-256 of every .py file; a zip of that code version is kept once per version)
  - data manifest per dataset (results/data/): how the data are read, SHA-256 of every sensor / label file,
    per-trial details and the excluded recordings
  - results/deployment_mcu.csv: template for on-device results (filled by hand)
"""
import datetime
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

import config as cfg
import models
from utils import cpu_name, save_csv

LIBRARIES = ["tensorflow", "keras", "numpy", "pandas", "scipy", "scikit-learn"]
_CODE = None


def _plain(value):
    """config values as JSON-friendly data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def settings():
    return {k: _plain(v) for k, v in vars(cfg).items() if k.isupper()}


def code_files():
    return sorted([*cfg.CODE_DIR.glob("*.py"), *(cfg.CODE_DIR / "models").glob("*.py")])


def code_fingerprint():
    """SHA-256 per code file and of all of them together; the git commit if the code is in a git repository."""
    global _CODE
    if _CODE is None:
        files = {str(p.relative_to(cfg.CODE_DIR)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in code_files()}
        combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in files.items()).encode()).hexdigest()
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.CODE_DIR, capture_output=True,
                                    text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            commit = None
        _CODE = {"code sha256": combined, "git commit": commit, "files": files}
    return _CODE


def snapshot_code(out_dir):
    """Zip of the current code version in out_dir/code_snapshots/ (once per version). Returns its path."""
    path = out_dir / "code_snapshots" / f"code_{code_fingerprint()['code sha256'][:12]}.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")     # parallel jobs may write the same snapshot
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in code_files():
                z.write(p, p.relative_to(cfg.CODE_DIR))
        os.replace(tmp, path)
    return path


def model_structure(name, model):
    """Registry entry, every build_model argument (defaults + overrides) and the layers of the built model."""
    module, kwargs, display = models.VARIANTS[name]
    signature = inspect.signature(importlib.import_module(f"models.{module}").build_model)
    args = {k: p.default for k, p in signature.parameters.items() if p.default is not inspect.Parameter.empty}
    args.update(kwargs)
    args.pop("norm", None)                      # normalisation statistics are stored separately
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    return {"name": name, "display name": display, "file": f"models/{module}.py", "build arguments": _plain(args),
            "parameters": int(model.count_params()), "trainable parameters": trainable,
            "non-trainable parameters": int(model.count_params()) - trainable,
            "layers": [{"name": l.name, "class": type(l).__name__, "parameters": int(l.count_params()),
                        "output shape": _plain(list(l.output.shape))} for l in model.layers]}


def environment():
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": cpu_name(),
            "libraries": versions, "CPU_THREADS env": os.environ.get("CPU_THREADS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


def config_snapshot(name, model, training_setup, out_dir, extra=None):
    """Full description of one run (written next to it as <run>.config.json)."""
    snapshot_code(out_dir)
    return {"created": datetime.datetime.now().isoformat(timespec="seconds"), "settings": settings(),
            "training": training_setup, "model": model_structure(name, model), "environment": environment(),
            "code": code_fingerprint(), **(extra or {})}


def normalization_record(mean, std):
    return {"method": "z-score per axis with the training (fit) subjects' windows", "channels": cfg.CHANNELS,
            "mean": [float(v) for v in np.ravel(mean)], "std": [float(v) for v in np.ravel(std)]}


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_data_manifest(dataset, trials, force=False):
    """results/data/<dataset>_manifest.json, _files.csv (path, kind, bytes, SHA-256), _trials.csv, _excluded.csv.
    Written once per dataset (force=True rewrites)."""
    import preprocessing as pre
    out = cfg.RESULTS_DIR / "data"
    manifest_path = out / f"{dataset}_manifest.json"
    if manifest_path.exists() and not force:
        return manifest_path
    root = cfg.DATASET_DIR / {"kfall": "KFall", "sisfall": "SisFall"}[dataset]
    kinds = {"kfall": [("sensor", "sensor_data_new/SA*/*.csv"), ("label", "label_data_new/SA*_label.xlsx")],
             "sisfall": [("sensor", "SisFall_dataset/S*/*.txt"),
                         ("label", "SisFall_temporally_annotated/S*/*.txt")]}[dataset]
    files = [{"Path": str(p.relative_to(root)).replace("\\", "/"), "Kind": kind, "Bytes": p.stat().st_size,
              "SHA-256": _sha256(p)} for kind, pattern in kinds for p in sorted(root.glob(pattern))]
    save_csv(files, out / f"{dataset}_files.csv")
    save_csv([{"Subject": t["subject"], "Group": pre.subject_group(t["subject"]), "File": t["file"],
               "Type(D/F)": t["type"], "Activity": pre.activity_code(t), "Rows used (100 Hz)": t["rows"],
               "Onset row": t["onset"], "Impact row": t["impact"], "Windows": len(t["y"]),
               "Accident windows": int(t["y"].sum())} for t in trials], out / f"{dataset}_trials.csv")
    excluded = pre.excluded_trials(dataset)
    save_csv(excluded or [{"Subject": "", "File": "", "Reason": "none"}], out / f"{dataset}_excluded.csv")
    combined = {kind: hashlib.sha256("".join(f["SHA-256"] for f in files if f["Kind"] == kind).encode()).hexdigest()
                for kind, _ in kinds}
    manifest = {"dataset": dataset, "root": str(root), "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "reading": pre.DESCRIPTION[dataset],
                "windowing": {"window rows": cfg.WINDOW, "step rows": cfg.STEP, "fall period": f"onset .. impact + {cfg.FALL_EXT} rows, later rows dropped",
                              "accident window": f">= {cfg.FALL_MIN} of {cfg.WINDOW} rows in the fall period"},
                "files": {kind: sum(f["Kind"] == kind for f in files) for kind, _ in kinds},
                "combined SHA-256 (per kind, over the sorted per-file hashes)": combined,
                "self check": pre.self_check(trials), "excluded recordings": len(excluded)}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"Data manifest written to {manifest_path}")
    return manifest_path


MCU_COLUMNS = ["Model", "Variant", "Model file", "Board", "MCU", "Clock (MHz)", "Toolchain", "Toolchain version",
               "Flash (KB)", "RAM (KB)", "Latency (ms)", "Energy per inference (mJ)", "Measured or estimated", "Notes"]


def mcu_template(names):
    """Adds empty rows (FP32 / INT8) for the given models to results/deployment_mcu.csv; filled rows are kept."""
    import pandas as pd
    path = cfg.RESULTS_DIR / "deployment_mcu.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=MCU_COLUMNS)
    have = set(zip(df["Model"], df["Variant"]))
    new = [{"Model": models.display_name(n), "Variant": v, "Model file": f"results/{n}/{n}_{v.lower()}.tflite"}
           for n in names for v in ("FP32", "INT8") if (models.display_name(n), v) not in have]
    if new:
        df = pd.concat([df, pd.DataFrame(new)], ignore_index=True).reindex(columns=MCU_COLUMNS).fillna("")
        save_csv(df, path)
