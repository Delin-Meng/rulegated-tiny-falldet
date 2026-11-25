# Code reference: fall detection experiments (KFall / SisFall)

Environment: conda env `falldet` (TensorFlow 2.21, Keras 3), see the [main README](../README.md).
Run everything from this `code/` folder.

## Files

| File                                                         | Contents                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `config.py`                                                  | **All settings**: data rules, epochs / batch / learning rate, seeds, folds, thresholds, gate, INT8 calibration, timing |
| `models/__init__.py`                                         | **Model registry**: name -> (model file, build arguments, display name) |
| `models/tinytcn.py`, `tinycnn.py`, `tinyssm.py`, `tinyeegnet.py`, `tinytransformer.py`, `cnnlstm.py` | Six of the seven models (`build_model(...)`): TinyTCN (458 params), TinyCNN (546), TinySSM (378), TinyEEGNet (338), TinyTransformer (946), CNN-LSTM (1,426) |
| `models/musci2021.py`                                        | Re-implementation of the LSTM of Musci et al. (2021), 2,322 params. The source, the parameter check and the differences from the original are at the top of the file |
| `models/layers.py`                                           | Weight-free input layers for baselines that are not part of this release; not used by the seven models |
| `preprocessing.py`                                           | Reading KFall / SisFall, windowing, labels, self-check, split, normalization (raw data cached in `results/cache/`) |
| `training.py`                                                | Training with the common settings; trial / window sensitivity, specificity, G-mean; threshold tuning |
| `deployment.py`                                              | TFLite FP32 / full-INT8 conversion, MACs, sizes, peak activation memory, CPU latency |
| `cv.py`                                                      | k-fold x seeds cross-validation, mean ± std, paired tests, TOST equivalence tests |
| `gate.py`                                                    | Kinematic rule gate (peak acceleration, angular velocity and tilt): as a stand-alone detector and as a first-stage filter |
| `analysis.py`                                                | Analyses from saved results: gate cost, ops per window, MACs by layer, sizes, gate sweeps / transfer, misses by fall type |
| `transfer.py`                                                | Saved cross-validation models of one dataset tested on the whole other dataset |
| `early_stopping.py`                                          | Early stopping vs the fixed 200 epochs (same training run, two weight sets) |
| `quantization.py`                                            | FP16 / weight-only / balanced-calibration / 16x8 variants of the saved cv models; diagnosis of the full-INT8 accuracy loss |
| `qat.py`                                                     | Quantization-aware training of CNN-LSTM and the Musci LSTM with hand-written fake quantization, fully integer TFLite |
| `pruning.py`                                                 | Magnitude pruning + fine-tuning of saved cv models (not reported in the paper) |
| `mcu.py`                                                     | MCU deployment files and on-target validation data           |
| `figures.py`                                                 | Paper figures (PNG + PDF) from the saved results             |
| `records.py`                                                 | Reproducibility records: config snapshot per run, data manifests, SHA-256 fingerprint of the code |
| `run_extra.py`                                               | Scheduler of the follow-up experiments: work units on a CPU / memory-aware process pool |
| `overall.py`                                                 | **Single entry point** (commands below)                      |
| `utils.py`                                                   | Small shared helpers: console tables, CSV output, environment info |
| `plot_sa06.py`                                               | Acceleration plot of one KFall trial of participant SA06, to check data loading |

## Commands

```
python overall.py check [--dataset kfall|sisfall]        # data self-check only
python overall.py list                                   # registered model variants
python overall.py splits [--datasets ...]                # subject splits of the main and the early-stopping protocol, per fold
python overall.py train [--models ...] [--epochs N]      # fixed KFall split
python overall.py cv [--dataset] [--models] [--seeds] [--folds] [--epochs]   # resumable
python overall.py report [--dataset]                     # statistics of saved cv runs
python overall.py deploy [--models ...]                  # TFLite / INT8 analysis of trained models
python overall.py analyze [--datasets ...]               # analyses from saved results (gate, ops, MACs, sizes, misses)
python overall.py transfer [--models ...]                # cross-dataset model transfer (needs both cv folders)
python overall.py quant [--dataset] [--models] [--variants]   # quantization variants of saved cv models
python overall.py int8diag [--dataset] [--models] [--runs N]  # diagnosis of the full-INT8 accuracy loss
python overall.py es [--dataset] [--models] [--seeds]         # early stopping (<= 200 epochs, patience 10) and the short schedule (<= 10 epochs, patience 3)
python overall.py earlystop [--models ...] [--folds ...]      # earlier single-seed comparison on KFall (not reported in the paper)
python overall.py prune [--dataset] [--models] [--sparsities] # magnitude pruning + fine-tuning (not reported in the paper)
python overall.py qat [--dataset] [--models]                  # quantization-aware training of the LSTMs
python overall.py mcu [--models]                              # MCU files + validation data
python overall.py figures                                     # paper figures -> results/figures/paper/
```

`quant`, `es`, `prune`, `qat`: `--report` only summarizes, `--no_report` only computes (for parallel jobs).
`deploy`, `quant`, `qat`: `--refresh_sizes` recomputes the graph-derived record fields (operator count, peak
activation memory) of saved results without re-measuring anything.
All are resumable and reuse the saved main runs; none retrains them.

`python overall.py` alone = `train` with the default models. Quick smoke test: `--epochs 2`
(for `cv` also e.g. `--folds 0 --seeds 0`).

### Main runs of the paper

```
python overall.py cv --dataset kfall   --models tinytcn tinycnn tinyssm tinyeegnet tinytransformer cnnlstm musci2021
python overall.py cv --dataset sisfall --models tinytcn tinycnn tinyssm tinyeegnet tinytransformer cnnlstm musci2021
python overall.py report --dataset kfall                 # mean ± std, paired tests, TOST
python overall.py report --dataset sisfall
python overall.py transfer --models tinytcn tinycnn tinyssm tinyeegnet tinytransformer cnnlstm musci2021   # after both cv folders exist
```

**Everything at once:** `bash run_all.sh` (inside tmux on a Linux server): self-check, fixed split + deployment,
cross-validation of all 7 models on both datasets, early stopping, then all reports / transfer / analyses.
Resumable; logs in `results/logs/`. Windows: `powershell -ExecutionPolicy Bypass -File run_all.ps1`.
Set for an i9-14900 (32 threads, 64 GB): 9 parallel jobs x `CPU_THREADS=4`; on other CPUs use about threads / 9.

### Follow-up experiments

After `run_all.sh`, run `bash run_extra.sh` inside tmux (= `python run_extra.py`).
A scheduler splits the work into about 190 units (one dataset / model / fold each) and runs them longest first on
a pool that keeps every core busy: single-threaded units (2 threads for the memory-heavy SisFall training units),
at most `--threads` threads (default: all logical CPUs) and only as many units as the free memory allows (measured
peaks: SisFall training 3.2 GB, SisFall quantization 2.2 GB). Stage 1 (experiments, about 5 h on the i9-14900),
stage 2 (all reports and analyses, same pool), stage 3 alone on the idle machine (latency, MCU files, figures).
`python run_extra.py --dry_run` lists the units and estimates the durations from the saved main runs;
`--only <name part>` runs selected units. Logs: `results/logs/extra/<unit>.log`, progress in
`results/logs/extra_scheduler.log`.

| Experiment                                                   | Command                                                      | Output                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Diagnosis of the full-INT8 accuracy loss                     | `int8diag`                                                   | `quantization/diagnosis/`                                    |
| Weight-only INT8 (FP16 is also available, not reported in the paper) | `quant`                                                      | `quantization/<dataset>/`                                    |
| Gate operating-point sweep over the training-recall target   | `analyze` (`GATE_RECALL_SWEEP`)                              | `analysis/gate_recall_sweep.csv`, `figures/gate_recall_sweep.csv` |
| Paired tests and TOST, including the deployed model vs TinyCNN | `report`                                                     | `cv_<dataset>/equivalence_tests.csv`, `paired_tests.csv`     |
| TinyTCN size sweep                                           | `cv --models tinytcn_c4 ... --seeds 0 --out_dir results/sweep_tinytcn_kfall` | `sweep_tinytcn_kfall/`                                       |
| Gate alone as a detector                                     | `analyze`                                                    | `analysis/gate_detector.csv`                                 |
| Missed falls and false alarms by activity type               | `analyze`                                                    | `analysis/misses_by_fall_type.csv`, `false_alarms_by_adl_type.csv` |
| Early stopping (<= 200 epochs, patience 10) and the short schedule (<= 10 epochs, patience 3), one training run | `es`                                                         | `es_<dataset>/`, `es_plan_<dataset>/`                        |
| Gate cost                                                    | `analyze`                                                    | `analysis/gate_ops.csv`, `ops_per_window.csv`                |
| Latency on one CPU                                           | `deploy` (pinned to `TIMING_CPU`, CPU name recorded)         | `deployment.csv`, `<model>/*_latency_*.csv`                  |
| MCU files                                                    | `mcu`, then ST Edge AI Developer Cloud (see `mcu/<model>/README.txt`) | `mcu/`, `deployment_mcu.csv`                                 |
| Cross-dataset model transfer                                 | `transfer --models <all 7>`                                  | `transfer/`                                                  |
| Transfer of the gate thresholds                              | `analyze`                                                    | `analysis/gate_transfer.csv`                                 |
| Pruning (not reported in the paper)                          | `prune`                                                      | `pruning/<dataset>/`                                         |
| QAT and other INT8 repairs                                   | `qat`, `quant` (balanced calibration, 16x8)                  | `qat/<dataset>/`, `quantization/<dataset>/`                  |
| Gate condition ablation                                      | `analyze`                                                    | `analysis/gate_ablation.csv`                                 |

Setting `ACT_RESULTS_DIR` points the code at another results folder (e.g. a copy of the results of another machine).

## Typical tasks

- **Try a model variant**: add one line to `VARIANTS` in `models/__init__.py`, e.g.
  `"tinytcn_c12": ("tinytcn", {"channels": 12}, "TinyTCN (12 ch)"),`
  then `python overall.py train --models tinytcn tinytcn_c12`.
- **Change a training / data setting**: edit `config.py` (e.g. `EPOCHS`, `CV_SEEDS`, `WINDOW`).
  The data cache only holds raw signals, so window settings take effect without re-reading the files.
- **Run cross-validation in parallel**: one terminal per model, e.g.
  `python overall.py cv --models tinytcn`; finished runs are skipped when restarted;
  afterwards `python overall.py report`.
- **Re-read the raw data** (e.g. after the dataset files change): delete `results/cache/<dataset>_raw.pkl`.

## Outputs (`results/`)

| Path             | Contents                                                     |
| ---------------- | ------------------------------------------------------------ |
| `<model>/`       | `summary.csv`, `trial_results.csv` (per-trial table, threshold 0.5), `self_check.csv`, `pip_freeze.txt`, `<model>.keras`, `<model>_fp32.tflite`, `<model>_int8.tflite` |
| `comparison.csv` | all models of the last `train` run + the values published for TinyCNN |
| `deployment.csv` | last `deploy` run                                            |
| `cv_<dataset>/`  | `runs/*.json` (one per model / fold / seed), `folds.csv`, `all_runs.csv`, `summary.csv`, `paired_tests.csv`, `equivalence_tests.csv` |

Reproducibility records (written automatically, `records.py`):

| Path                                                         | Contents                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `cv_<dataset>/runs/<run>.config.json`, `<model>/config.json` | every config.py setting, optimizer / focal loss arguments, model build arguments and layers, library versions, CPU, code SHA-256 (code zip in `code_snapshots/`) |
| `cv_<dataset>/runs/<run>.json`, `<model>/run.json`           | subject lists (fit / threshold tuning / test), normalization mean / std, INT8 calibration set and I/O scale / zero point |
| `cv_<dataset>/runs/<run>.history.csv`, `<model>/history.csv` | training loss / accuracy per epoch                           |
| `<model>/<model>_int8_quantization.json`, `_latency_fp32.csv`, `_latency_int8.csv` | quantization record; all 1,000 timing runs (summary: median, p5, p95) |
| `<model>/threshold_sweep.csv`, `roc_trial.csv`; `cv_<dataset>/threshold_sweep_runs.csv` | trial-level sensitivity / specificity / G-mean per decision threshold; trial-level ROC and AUC |
| `data/<dataset>_manifest.json`, `_files.csv`, `_trials.csv`, `_excluded.csv` | how the data are read, SHA-256 of every file, per-trial details, excluded recordings |
| `figures/<experiment>_*.csv`                                 | one CSV per figure: threshold sweep, ROC, training curves, parameters / MACs vs G-mean, gate recall sweep |
| `deployment_mcu.csv`                                         | template for on-device results (board, toolchain, flash, RAM, latency, energy, measured / estimated) |

## Protocol settings

**Threshold protocol** (`config.THRESHOLD_PROTOCOL`): `"train"` tunes the decision and gate thresholds on the
model's training subjects (the main protocol of the paper); `"validation"` holds out `THRESHOLD_VAL_FRACTION` of
the training subjects for tuning only, and writes cross-validation results to `cv_<dataset>_valthr/`.

**Decision threshold**: tuned on the full grid k / 256 (`THRESHOLD_GRID`). The restricted grid 0.50–0.99
(`THRESHOLD_GRID_CAPPED`) is kept only for the comparison of the two grids.

**Gate**: a window passes when all three conditions exceed their thresholds in that window
(`GATE_CONSECUTIVE = 1`), and the thresholds are tuned to keep 100 % of the training fall trials
(`GATE_TRIAL_RECALL = 1.0`). `GATE_CONDITIONS` selects the gate's conditions (e.g. `(0,)` = acceleration only);
`analyze` runs the ablation over all subsets.

**Metrics**: "Trial ..." / "Window ..." use the fixed threshold 0.5; "Tuned ..." use the threshold that maximizes
the trial G-mean on the training data (no test data); "INT8 ..." are the same for the full-INT8 TFLite model.
