"""
Single entry point for all experiments (settings in config.py, model variants in models/__init__.py).

  python overall.py check  [--dataset kfall|sisfall]          data self-check only
  python overall.py list                                      registered model variants
  python overall.py train  [--models ...] [--epochs N]        fixed KFall split (team rule): per-model results
                                                              in results/<model>/ + results/comparison.csv
  python overall.py cv     [--dataset] [--models] [--seeds] [--folds] [--epochs]
                                                              k-fold x seeds, resumable -> results/cv_<dataset>/
  python overall.py report [--dataset] [--epochs]             statistics of the saved cv runs only
  python overall.py deploy [--models ...]                     TFLite / INT8 analysis of trained models
                                                              -> results/deployment.csv
  python overall.py transfer [--models ...]                   saved cv models tested on the other dataset
                                                              -> results/transfer/
  python overall.py earlystop [--models ...] [--folds ...]    early stopping vs 200 epochs (KFall)
                                                              -> results/early_stopping/
  python overall.py analyze [--datasets kfall sisfall]        review analyses from saved results
                                                              (gate, ops, MACs, sizes, misses) -> results/analysis/
  python overall.py quant  [--dataset] [--models] [--variants] float16 / weight-only / balanced-calibration / 16x8
                                                              variants of the saved cv models -> results/quantization/
  python overall.py int8diag [--dataset] [--models] [--runs N] why full INT8 fails (saturation vs bug)
                                                              -> results/quantization/diagnosis/
  python overall.py es     [--dataset] [--models] [--seeds]   early stopping (<= 200 epochs, patience 10) and the
                                                              plan's rule (<= 10, patience 3) from the same runs
                                                              -> results/es_<dataset>/, results/es_plan_<dataset>/
  python overall.py prune  [--dataset] [--models] [--sparsities]   magnitude pruning + fine-tuning -> results/pruning/
  python overall.py qat    [--dataset] [--models]             quantization-aware training of the LSTMs -> results/qat/
  python overall.py mcu    [--models]                         MCU files + validation data -> results/mcu/
  python overall.py figures                                   paper figures -> results/figures/paper/
Without a command, `train` runs with its defaults.
"""
import os

import config as cfg
if not cfg.USE_GPU:                       # must happen before TensorFlow is imported
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
keras.utils.set_random_seed(0)

import tensorflow as tf
_threads = os.environ.get("CPU_THREADS") or cfg.CPU_THREADS
if _threads:                              # must happen before TensorFlow runs any op
    tf.config.threading.set_intra_op_parallelism_threads(int(_threads))
    tf.config.threading.set_inter_op_parallelism_threads(2)
if cfg.DETERMINISTIC:
    tf.config.experimental.enable_op_determinism()

from pathlib import Path
import argparse
import json

import cv
import deployment
import gate
import models
import preprocessing as pre
import records
import training
from utils import print_columns, save_csv, save_pip_freeze


FIXED_SPLIT_KEY = 100        # key of the held-out threshold subjects of the fixed split (THRESHOLD_PROTOCOL "validation")


def load_fixed_split(subjects=None):
    """KFall with the team's fixed test subjects (experiment_split), after the self-check; `subjects` rebuilds the
    split of a saved model ({"fit", "threshold", "test"})."""
    trials = pre.load_trials("kfall")
    check = pre.self_check(trials)
    print(f"Self-check: {check}  ->  {'OK' if check == cfg.KFALL_EXPECTED else 'MISMATCH, expected ' + str(cfg.KFALL_EXPECTED)}")
    records.write_data_manifest("kfall", trials)
    if subjects:
        return pre.experiment_split(trials, subjects["test"], fit_subjects=subjects["fit"],
                                    tune_subjects=subjects["threshold"]), check
    return pre.experiment_split(trials, cfg.KFALL_TEST_SUBJECTS, key=FIXED_SPLIT_KEY), check


def evaluate_model(model, data, out_dir, name):
    """Deployment analysis + all metrics + threshold sweep / ROC files of a trained model on a split."""
    deploy, p8_tr, p8_te = deployment.analyze(model, data["X_tr"], data["X_te"], out_dir, name, X_tune=data["X_tune"],
                                              y_tr=data["y_tr"], subjects=data["subjects"]["fit"],
                                              mean=data["mean"], std=data["std"])
    p_tr, p_te = training.predict(model, data["X_tune"]), training.predict(model, data["X_te"])
    mdata = pre.metric_data(data)
    gate_thr = gate.tune_filter(mdata["gate_tr"], mdata["y_tr"], mdata["train"])
    metrics = training.all_metrics(p_tr, p_te, p8_tr, p8_te, mdata, gate_thr)
    sweep, roc = training.curves(p_te, p8_te, mdata, gate_thr)
    save_csv(sweep, out_dir / "threshold_sweep.csv")
    save_csv(roc, out_dir / "roc_trial.csv")
    return deploy, metrics, p_te


def run_record(data, epochs, minutes=None):
    return {"threshold_protocol": cfg.THRESHOLD_PROTOCOL, "subjects": data["subjects"], "epochs": epochs,
            "training_minutes": minutes, "normalization": records.normalization_record(data["mean"], data["std"]),
            "code sha256": records.code_fingerprint()["code sha256"]}


def train_one(name, data, check, epochs, out_dir):
    """Train one variant on the fixed split, write its files to out_dir, return its summary row."""
    model, minutes, history = training.train_model(name, data["X_tr"], data["y_tr"], epochs,
                                                   norm=(data["mean"], data["std"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv([check], out_dir / "self_check.csv")
    save_csv(training.history_rows(history), out_dir / "history.csv")
    deploy, metrics, p_te = evaluate_model(model, data, out_dir, name)
    save_csv(training.evaluate(p_te, data["y_te"], data["test"])[0], out_dir / "trial_results.csv")
    summary = {"Model": models.display_name(name), **training.format_metrics(metrics),
               "Parameters": model.count_params(), **deploy,
               "Epochs": epochs, "Training time": f"{minutes:.1f} min"}
    save_csv([summary], out_dir / "summary.csv")
    (out_dir / "run.json").write_text(json.dumps(run_record(data, epochs, minutes), indent=1))
    (out_dir / "config.json").write_text(json.dumps(records.config_snapshot(
        name, model, training.training_setup(epochs), out_dir.parent,
        {"run": {"split": "fixed KFall split", "seed": cfg.FIXED_SEED}, "threshold protocol": cfg.THRESHOLD_PROTOCOL}),
        indent=1))
    save_pip_freeze(out_dir / "pip_freeze.txt")
    model.save(out_dir / f"{name}.keras")
    records.mcu_template([name])
    print(f"Results saved to {out_dir}")
    return summary


def cmd_train(args):
    out_root = Path(args.out_dir or cfg.RESULTS_DIR)
    data, check = load_fixed_split()
    rows = []
    for name in args.models:
        print(f"\n==================== {name} ====================")
        rows.append(train_one(name, data, check, args.epochs, out_root / name))
    paper = {c: cfg.PAPER.get(c, "-") for c in rows[0]} | {"Model": "TinyCNN paper"}
    comparison = save_csv(rows + [paper], out_root / "comparison.csv")
    print("\n==================== Comparison ====================")
    print_columns(comparison)
    print(f"Comparison saved to {out_root / 'comparison.csv'}")


def cmd_cv(args):
    out_dir = Path(args.out_dir or cv.out_dir_for(args.dataset))
    cv.run(args.dataset, args.models, args.seeds, args.folds, args.epochs, out_dir)
    if not args.no_report:
        cv.report(args.dataset, args.epochs, out_dir)


def cmd_report(args):
    cv.report(args.dataset, args.epochs, Path(args.out_dir or cv.out_dir_for(args.dataset)))


def cmd_deploy(args):
    models.register_custom_layers()
    if args.refresh_sizes:
        deployment.refresh_sizes(args.models)
        return
    rows, splits = [], {}
    for name in args.models:
        path = cfg.RESULTS_DIR / name / f"{name}.keras"
        if not path.exists():
            print(f"{name}: {path} not found, train it first (python overall.py train --models {name})")
            continue
        print(f"\n===== {name} =====")
        # the split the model was trained on (models trained before run.json existed: the default fixed split)
        run_file = path.parent / "run.json"
        subjects = json.loads(run_file.read_text()).get("subjects") if run_file.exists() else None
        key = json.dumps(subjects, sort_keys=True)
        if key not in splits:
            splits[key] = load_fixed_split(subjects)[0]
        model = keras.models.load_model(path)
        info, metrics, _ = evaluate_model(model, splits[key], path.parent, name)
        rows.append({"Model": models.display_name(name), "Parameters": model.count_params(), **info,
                     **training.format_metrics(metrics)})
    if rows:
        print_columns(save_csv(rows, cfg.RESULTS_DIR / "deployment.csv"))
        records.mcu_template([n for n in args.models if (cfg.RESULTS_DIR / n / f"{n}.keras").exists()])
        print(f"Saved to {cfg.RESULTS_DIR / 'deployment.csv'} (on-device results template: deployment_mcu.csv)")


def cmd_transfer(args):
    import transfer
    transfer.run(args.models, int8=not args.no_int8)


def cmd_earlystop(args):
    import early_stopping
    if args.report:
        early_stopping.report()
    else:
        early_stopping.run(args.models, args.folds, args.seed, args.epochs, args.patience)


def cmd_analyze(args):
    import analysis
    analysis.run(args.datasets)


def cmd_quant(args):
    import quantization
    if args.refresh_sizes:
        quantization.refresh_sizes(args.dataset, args.models, args.folds, args.seeds, args.variants)
    elif not args.report:
        quantization.run(args.dataset, args.models, args.folds, args.seeds, args.variants)
    if not args.no_report:
        quantization.report(args.dataset)


def cmd_int8diag(args):
    import quantization
    quantization.diagnose(args.dataset, args.models, args.runs, args.windows)


def cmd_es(args):
    import early_stopping
    if not args.report:
        early_stopping.run_es(args.dataset, args.models, args.folds, args.seeds, args.max_epochs, args.patience,
                              args.plan_epochs, args.plan_patience)
    if not args.no_report:
        early_stopping.report_es(args.dataset, args.max_epochs, args.plan_epochs)


def cmd_prune(args):
    import pruning
    if not args.report:
        pruning.run(args.dataset, args.models, args.folds, args.seeds, args.sparsities, args.epochs)
    if not args.no_report:
        pruning.report(args.dataset)


def cmd_qat(args):
    import qat
    if args.refresh_sizes:
        qat.refresh_sizes(args.dataset)
        return
    if not args.report:
        qat.run(args.dataset, args.models, args.folds, args.seeds, args.epochs)
    if not args.no_report:
        qat.report(args.dataset)


def cmd_mcu(args):
    import mcu
    mcu.export(args.models, args.windows)


def cmd_figures(args):
    import figures
    figures.run()


def cmd_check(args):
    """Data self-check; also writes the data manifest (results/data/) once, before any parallel experiment."""
    trials = pre.load_trials(args.dataset)
    check = pre.self_check(trials)
    print(f"Self-check ({args.dataset}): {check}")
    if args.dataset == "kfall":
        print(f"  expected: {cfg.KFALL_EXPECTED}  ->  {'OK' if check == cfg.KFALL_EXPECTED else 'MISMATCH'}")
    records.write_data_manifest(args.dataset, trials)


def cmd_splits(args):
    """Subject splits of the main and the early-stopping protocol, per fold -> results/data/<dataset>_splits.csv."""
    for ds in args.datasets:
        rows = cv.split_table(pre.load_trials(ds))
        df = save_csv(rows, cfg.RESULTS_DIR / "data" / f"{ds}_splits.csv")
        print(f"\n===== {ds} =====")
        print(df.drop(columns="Subject IDs").to_string(index=False))
    print(f"\nSaved to {cfg.RESULTS_DIR / 'data'} (with the subject IDs of every group)")


def cmd_list(args):
    for name, (module, kwargs, display) in models.VARIANTS.items():
        model = models.build(name, len(cfg.CHANNELS), cfg.WINDOW)
        default = " (default)" if name in models.DEFAULT else ""
        print(f"{name:16s} {display:26s} models/{module}.py {kwargs or ''}  {model.count_params()} params{default}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    model_arg = dict(nargs="+", choices=list(models.VARIANTS), default=models.DEFAULT)
    dataset_arg = dict(choices=["kfall", "sisfall"], default="kfall")

    p = sub.add_parser("train", help="fixed KFall split (team rule)")
    p.add_argument("--models", **model_arg)
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--out_dir", default=None, help="default: results/")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("cv", help="k-fold x seeds cross-validation (resumable)")
    p.add_argument("--dataset", **dataset_arg)
    p.add_argument("--models", **model_arg)
    p.add_argument("--seeds", nargs="+", type=int, default=cfg.CV_SEEDS)
    p.add_argument("--folds", nargs="+", type=int, default=list(range(cfg.CV_FOLDS)))
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--out_dir", default=None, help="default: results/cv_<dataset>/")
    p.add_argument("--no_report", action="store_true", help="train only, no summary (parallel jobs on one folder)")
    p.set_defaults(func=cmd_cv)

    p = sub.add_parser("report", help="statistics of saved cv runs")
    p.add_argument("--dataset", **dataset_arg)
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--out_dir", default=None, help="default: results/cv_<dataset>/")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("deploy", help="TFLite / INT8 analysis of trained models in results/<model>/")
    p.add_argument("--models", **model_arg)
    p.add_argument("--refresh_sizes", action="store_true",
                   help="only recompute the graph-derived columns (op count, sizes, peak activation memory) of the "
                        "saved records from the saved .tflite files: no conversion, no timing")
    p.set_defaults(func=cmd_deploy)

    p = sub.add_parser("transfer", help="saved cv models of one dataset tested on the whole other dataset")
    p.add_argument("--models", nargs="+", choices=list(models.VARIANTS), default=["tinycnn", "tinytcn"])
    p.add_argument("--no_int8", action="store_true", help="skip the INT8 conversion (FP32 results only)")
    p.set_defaults(func=cmd_transfer)

    p = sub.add_parser("earlystop", help="early stopping vs 200 epochs on KFall (k folds x 1 seed, resumable)")
    p.add_argument("--models", nargs="+", choices=list(models.VARIANTS), default=["tinycnn", "tinytcn"])
    p.add_argument("--folds", nargs="+", type=int, default=list(range(cfg.CV_FOLDS)))
    p.add_argument("--seed", type=int, default=cfg.CV_SEEDS[0])
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--patience", type=int, default=cfg.ES_PATIENCE)
    p.add_argument("--report", action="store_true", help="only summarise the saved runs")
    p.set_defaults(func=cmd_earlystop)

    p = sub.add_parser("analyze", help="review analyses from saved results (gate, ops, MACs, sizes, misses)")
    p.add_argument("--datasets", nargs="+", choices=["kfall", "sisfall"], default=["kfall", "sisfall"])
    p.set_defaults(func=cmd_analyze)

    folds_arg = dict(nargs="+", type=int, default=list(range(cfg.CV_FOLDS)))
    report_arg = dict(action="store_true", help="only summarise the saved results")
    no_report_arg = dict(action="store_true", help="compute only, no summary (several jobs share one folder)")

    p = sub.add_parser("quant", help="quantization variants of the saved cv models (no retraining, resumable)")
    p.add_argument("--dataset", **dataset_arg)
    p.add_argument("--models", nargs="+", choices=list(models.VARIANTS), default=models.MAIN)
    p.add_argument("--variants", nargs="+", choices=cfg.QUANT_VARIANTS, default=cfg.QUANT_VARIANTS)
    p.add_argument("--folds", **folds_arg)
    p.add_argument("--seeds", nargs="+", type=int, default=cfg.CV_SEEDS)
    p.add_argument("--report", **report_arg)
    p.add_argument("--no_report", **no_report_arg)
    p.add_argument("--refresh_sizes", action="store_true",
                   help="convert the saved runs again and rewrite only the graph-derived record fields (peak "
                        "activation memory, float ops), then report")
    p.set_defaults(func=cmd_quant)

    p = sub.add_parser("int8diag", help="diagnosis of the full-INT8 failures (saturation vs bug)")
    p.add_argument("--dataset", **dataset_arg)
    p.add_argument("--models", nargs="+", choices=list(models.VARIANTS),
                   default=["musci2021", "cnnlstm", "tinytransformer", "tinytcn"])
    p.add_argument("--runs", type=int, default=None, help="at most this many saved runs per model (default: all)")
    p.add_argument("--windows", type=int, default=100, help="test windows per class for the tensor saturation")
    p.set_defaults(func=cmd_int8diag)

    p = sub.add_parser("es", help="early stopping (<= 200 epochs, patience 10) + the plan's rule, both datasets "
                                  "(resumable)")
    p.add_argument("--dataset", **dataset_arg)
    p.add_argument("--models", nargs="+", choices=list(models.VARIANTS), default=cfg.ES_MODELS)
    p.add_argument("--folds", **folds_arg)
    p.add_argument("--seeds", nargs="+", type=int, default=cfg.CV_SEEDS)
    p.add_argument("--max_epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--patience", type=int, default=cfg.ES_PATIENCE)
    p.add_argument("--plan_epochs", type=int, default=cfg.ES_PLAN_EPOCHS, help="the plan rule's maximum epochs")
    p.add_argument("--plan_patience", type=int, default=cfg.ES_PLAN_PATIENCE)
    p.add_argument("--report", **report_arg)
    p.add_argument("--no_report", **no_report_arg)
    p.set_defaults(func=cmd_es)

    p = sub.add_parser("prune", help="magnitude pruning + fine-tuning of saved cv models (resumable)")
    p.add_argument("--dataset", **dataset_arg)
    p.add_argument("--models", nargs="+", choices=list(models.VARIANTS), default=["tinytcn"])
    p.add_argument("--sparsities", nargs="+", type=float, default=cfg.PRUNE_SPARSITIES)
    p.add_argument("--folds", **folds_arg)
    p.add_argument("--seeds", nargs="+", type=int, default=[cfg.CV_SEEDS[0]])
    p.add_argument("--epochs", type=int, default=cfg.PRUNE_FINETUNE_EPOCHS, help="fine-tuning epochs")
    p.add_argument("--report", **report_arg)
    p.add_argument("--no_report", **no_report_arg)
    p.set_defaults(func=cmd_prune)

    p = sub.add_parser("qat", help="quantization-aware training of the recurrent models (resumable)")
    p.add_argument("--dataset", **dataset_arg)
    p.add_argument("--models", nargs="+", choices=cfg.QAT_MODELS, default=cfg.QAT_MODELS)
    p.add_argument("--folds", **folds_arg)
    p.add_argument("--seeds", nargs="+", type=int, default=[cfg.CV_SEEDS[0]])
    p.add_argument("--epochs", type=int, default=cfg.QAT_EPOCHS, help="fine-tuning epochs")
    p.add_argument("--report", **report_arg)
    p.add_argument("--no_report", **no_report_arg)
    p.add_argument("--refresh_sizes", action="store_true",
                   help="only recompute the peak activation memory of the saved QAT INT8 files in the run records")
    p.set_defaults(func=cmd_qat)

    p = sub.add_parser("mcu", help="MCU deployment files and validation data of fixed-split models")
    p.add_argument("--models", nargs="+", choices=list(models.VARIANTS), default=["tinytcn", "tinycnn"])
    p.add_argument("--windows", type=int, default=200, help="validation windows (half accident windows)")
    p.set_defaults(func=cmd_mcu)

    sub.add_parser("figures", help="paper figures from the saved results").set_defaults(func=cmd_figures)

    p = sub.add_parser("check", help="data self-check")
    p.add_argument("--dataset", **dataset_arg)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("splits", help="subject splits of the main and the early-stopping protocol, per fold")
    p.add_argument("--datasets", nargs="+", choices=["kfall", "sisfall"], default=["kfall", "sisfall"])
    p.set_defaults(func=cmd_splits)

    sub.add_parser("list", help="registered model variants").set_defaults(func=cmd_list)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["train"])
    args.func(args)
