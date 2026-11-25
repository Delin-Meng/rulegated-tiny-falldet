# Kinematic Rule Gating with a Tiny Temporal Convolutional Network for Wearable Fall Detection on Microcontrollers

**Sub-2.5K-parameter neural networks behind a kinematic rule gate for fall detection on microcontrollers**,
evaluated on the KFall and SisFall wearable IMU datasets.

This repository contains the experiment code of the manuscript of the same title by Delin Meng, Sanghyeuk Kim,
Shilei Zong, Hamida Mahama and Jaesoo Yoo (submitted for publication): data preprocessing, seven tiny models,
subject-wise cross-validation with paired and equivalence statistics, full-INT8 TensorFlow Lite deployment
analysis, the kinematic rule gate that runs before the network, and the follow-up studies (quantization variants,
QAT, early stopping, cross-dataset transfer, MCU export, paper figures).

---

## Highlights

- **Two-stage detector**: a kinematic rule gate tests the peak acceleration, peak angular velocity and peak tilt
  angle of each 0.5 s window, and only the windows that exceed all three thresholds are sent to the network.
- **Tiny models (338–2,322 parameters)**: TinyTCN, TinySSM, TinyCNN, TinyEEGNet, TinyTransformer, CNN-LSTM and a
  re-implementation of the LSTM of Musci et al. (2021), all trained with one common protocol.
- **Deployment-first evaluation**: every model is converted to full-INT8 TFLite (int8 weights, activations and I/O).
  The pipeline reports MACs, parameter bytes, `.tflite` size, peak activation memory and single-thread CPU latency,
  and exports the files used for the STM32 measurements.
- **Statistics**: 5-fold subject-wise CV × 5 seeds, Nadeau–Bengio corrected resampled t-tests with Holm adjustment,
  and TOST equivalence tests. The paper reports the corrected t-test and the 1- and 2-point margins; the code also
  computes Wilcoxon tests and a 0.5-point margin.
- **Resumable and traceable**: every run saves its model, predictions, config snapshot, subject lists,
  normalization statistics, INT8 calibration record and a SHA-256 fingerprint of the code. All metrics are
  recomputed from saved predictions, so evaluation rules can change without retraining.

## Results in brief

Subject-wise five-fold cross-validation with five seeds (25 runs per model and dataset), as reported in the paper:

| Finding                                                      |   KFall |  SisFall |
| ------------------------------------------------------------ | ------: | -------: |
| Windows passed by the gate                                   |      9% |      14% |
| Trial-level G-mean of gated full-INT8 TinyTCN                |   99.3% |    98.3% |
| Equivalence margin to the re-implemented floating-point TinyCNN | 1 point | 2 points |

- The gate changes the G-mean of INT8 TinyTCN by less than 0.1 points.
- On a Cortex-M4 board, the estimated inference time per window is about 15 times lower than that of ungated
  TinyCNN on KFall.
- Post-training INT8 quantization costs the convolutional and state-space models at most 0.6 points and the
  attention and recurrent models up to 5 points.

## Models

All models take a `[6 axes × 50 samples]` window (0.5 s at 100 Hz) and output `P(fall)` through a 2-way softmax.

| Name (`--models`) | Model           | Parameters | Description                                                  |
| ----------------- | --------------- | ---------: | ------------------------------------------------------------ |
| `tinytcn`         | **TinyTCN**     |        458 | 1×1 conv → 4 residual blocks of causal dilated depthwise conv (d = 1, 2, 4, 8) + 1×1 conv → GAP. Receptive field 31 samples |
| `tinyssm`         | TinySSM         |        378 | 1×1 conv → 2 residual blocks of an S4D-style diagonal state-space layer + 1×1 conv → GAP |
| `tinycnn`         | TinyCNN         |        546 | Re-implementation of Yu et al. (2023): temporal conv (1×16) → spatial conv (6×1) → GAP |
| `tinyeegnet`      | TinyEEGNet      |        338 | EEGNet (Lawhern et al., 2018) scaled to 6 axes × 50 samples  |
| `tinytransformer` | TinyTransformer |        946 | Patch embedding (10 tokens) + one pre-norm Transformer block (2 heads) |
| `cnnlstm`         | CNN-LSTM        |      1,426 | Conv1D (8 filters) → LSTM(12), recurrent baseline            |
| `musci2021`       | Musci LSTM      |      2,322 | Re-implementation of the LSTM of Musci et al. (2021) with its original layer configuration. Differences from the original implementation are documented in [models/musci2021.py](code/models/musci2021.py) |

Additional registered variants: `tinycnn_valid` (EEGNet-style valid padding) and a TinyTCN size sweep
(`tinytcn_c4`, `_c6`, `_c12`, `_c16`, `_b2`, `_b3`). Run `python overall.py list` to print all of them with their
parameter counts. To add your own variant, add one line to `VARIANTS` in [models/\_\_init\_\_.py](code/models/__init__.py).

## Method overview

| Stage           | Setting (all in [config.py](code/config.py))                 |
| --------------- | ------------------------------------------------------------ |
| Input           | AccX/Y/Z (g) and GyrX/Y/Z (°/s) at 100 Hz. SisFall is decimated from 200 Hz by taking every second sample |
| Windowing       | 50 samples (0.5 s), step 10 (0.1 s). A window is a *fall window* (called an accident window in the code) if ≥ 20 of its 50 samples lie in the fall interval, which runs from the annotated fall onset to 50 samples after the impact (for SisFall, the annotated fall segment). Samples after the interval are dropped |
| Normalization   | Per-axis z-score with statistics from the training subjects only |
| Training        | Adam (lr 5e-4), batch 512, 200 epochs, focal loss (Keras `CategoricalFocalCrossentropy`, γ = 2, α = 0.25), CPU only |
| Quantization    | Full-INT8 TFLite, calibrated on 1,000 training windows clipped at the 99.9th percentile of \|z\| |
| Gate            | Per window: max \|a\|, max \|ω\| and max tilt angle from the upright sensor axis. A window passes if all three exceed their thresholds. The thresholds minimize the share of training windows that pass while keeping 100 % of the training fall trials |
| Trial decision  | A trial is a fall if any window reaches `P(fall) ≥ threshold`. The threshold is chosen on the INT8 output grid k/256 to maximize trial-level G-mean on the training subjects (no test data) |
| Evaluation      | Subject-wise 5-fold cross-validation with fixed folds and seeds 0, 7, 42, 123 and 1337 |
| Headline metric | `Gated INT8 tuned Trial G-mean`: gate + tuned threshold + full INT8 model |

## Repository structure

```
rulegated-tiny-falldet/
├── code/
│   ├── overall.py          # single entry point for every experiment (see Usage)
│   ├── config.py           # all settings: data rules, training, thresholds, gate, quantization, timing
│   ├── models/             # model registry + one file per architecture
│   ├── preprocessing.py    # reading KFall / SisFall, windowing, labels, subject-wise splits
│   ├── training.py         # training and metrics (trial / window sensitivity, specificity, G-mean, AUC)
│   ├── gate.py             # kinematic rule gate: as a stand-alone detector and as a first-stage filter
│   ├── cv.py               # k-fold × seeds cross-validation, paired tests, TOST
│   ├── deployment.py       # TFLite FP32 / INT8 conversion, MACs, sizes, peak memory, latency
│   ├── quantization.py     # FP16 / weight-only / balanced-calibration / 16x8 variants, INT8 loss diagnosis
│   ├── qat.py              # quantization-aware training of the recurrent models
│   ├── pruning.py          # magnitude pruning + fine-tuning
│   ├── early_stopping.py   # early stopping vs fixed 200 epochs
│   ├── transfer.py         # cross-dataset transfer (KFall ↔ SisFall)
│   ├── analysis.py         # gate cost, ops per window, MACs by layer, misses by fall type, gate ablation
│   ├── mcu.py              # files for STM32 / ST Edge AI deployment and on-target validation
│   ├── figures.py          # paper figures (PNG + PDF)
│   ├── records.py          # reproducibility records (config snapshots, data manifests, code hashes)
│   ├── run_all.sh / .ps1   # main experiments, all models, both datasets
│   ├── run_extra.py / .sh  # follow-up experiments on a CPU- and memory-aware process pool
│   └── README.md           # detailed reference: every command, option and output file
└── dataset/                # not included: create this folder and put the datasets inside (see Data)
    ├── KFall/
    └── SisFall/
```

## Installation

The code was developed with **TensorFlow 2.21 / Keras 3** and runs on the CPU (GPUs are hidden by default,
see `USE_GPU` in `config.py`).

```bash
conda create -n falldet python=3.11
conda activate falldet
pip install "tensorflow==2.21.*" numpy pandas scipy scikit-learn tqdm matplotlib openpyxl
```

## Data

The datasets are **not included** in this repository. KFall in particular may not be redistributed. Obtain the
datasets from their authors and place them like this:

```
dataset/
├── KFall/
│   ├── sensor_data_new/SA06/S06T01R01.csv ...        # 32 subjects, 100 Hz
│   └── label_data_new/SA06_label.xlsx ...            # fall onset / impact frames
└── SisFall/
    ├── SisFall_dataset/SA01/D01_SA01_R01.txt ...     # 200 Hz raw sensor files
    └── SisFall_temporally_annotated/SA01/...txt      # per-sample labels of Musci et al. (2021)
```

| Dataset                 | Source                                                       |
| ----------------------- | ------------------------------------------------------------ |
| KFall                   | Request access at <https://sites.google.com/view/kfalldataset> |
| SisFall                 | Sucerquia et al., *Sensors* 17(1):198, 2017. [doi:10.3390/s17010198](https://doi.org/10.3390/s17010198) |
| SisFall temporal labels | <https://bitbucket.org/unipv_cvmlab/sisfalltemporallyannotated/> |

Run `python overall.py check` to verify the data. For KFall the expected counts are 5,075 trials, 2,346 falls,
32 subjects, 310,512 windows and 24,573 fall windows. The corresponding SisFall counts in the paper are 4,500
trials, 1,798 falls, 38 subjects and 634,427 windows; the five SisFall recordings of participant SA15 in activity
D17 are excluded because they have no per-sample labels. The first run caches the parsed signals in
`results/cache/`.

## Usage

Run everything from the `code/` folder.

### Quick start

```bash
cd code
python overall.py check                                   # data self-check
python overall.py list                                    # registered models and parameter counts
python overall.py train --models tinytcn --epochs 2       # smoke test on the fixed KFall split
python overall.py cv --models tinytcn --folds 0 --seeds 0 --epochs 2
```

### Main experiments

```bash
python overall.py train  --models tinytcn tinycnn tinyssm      # fixed KFall split -> results/<model>/
python overall.py deploy --models tinytcn tinycnn tinyssm      # TFLite FP32 / INT8 analysis -> results/deployment.csv
python overall.py cv --dataset kfall   --models tinytcn         # 5 folds x 5 seeds, resumable -> results/cv_kfall/
python overall.py cv --dataset sisfall --models tinytcn         # -> results/cv_sisfall/
python overall.py report --dataset kfall                        # mean ± std, paired tests, TOST
```

Cross-validation runs can be started in parallel terminals (one per model), and finished runs are skipped on restart.

### Follow-up studies

| Command                              | Study                                                        |
| ------------------------------------ | ------------------------------------------------------------ |
| `python overall.py transfer`         | Models trained on one dataset, tested on the other           |
| `python overall.py earlystop` / `es` | Early stopping vs the fixed 200 epochs                       |
| `python overall.py quant`            | Weight-only INT8, balanced calibration and 16x8 variants of the saved CV models; FP16 is also available but not reported in the paper |
| `python overall.py int8diag`         | Diagnosis of the full-INT8 accuracy loss of some models      |
| `python overall.py qat`              | Quantization-aware training of CNN-LSTM and the Musci LSTM   |
| `python overall.py prune`            | Magnitude pruning + fine-tuning (30 / 50 / 70 % sparsity); not reported in the paper |
| `python overall.py analyze`          | Gate cost and ablation, ops per window, MACs by layer, misses by fall type |
| `python overall.py mcu`              | INT8 model, `deployment.json` and validation tensors for STM32 / ST Edge AI |
| `python overall.py figures`          | Paper figures -> `results/figures/paper/`                    |

Run `python overall.py <command> -h` to list a command's options. [code/README.md](code/README.md) documents
every option and output file.

### Full reproduction

```bash
cd code
bash run_all.sh       # self-check, fixed split + deployment, CV of all 7 models on both datasets, reports
bash run_extra.sh     # all follow-up studies on a process pool, then latency, MCU files and figures
```

On Windows, run `powershell -ExecutionPolicy Bypass -File run_all.ps1`. Both scripts are resumable, and logs go
to `results/logs/`. They are tuned for an Intel i9-14900 (32 threads, 64 GB): `run_all.sh` starts 9 parallel jobs
with `CPU_THREADS=4`, so on other machines set `CPU_THREADS` to about your thread count / 9.
`python run_extra.py --dry_run` lists the work units and their estimated durations.

## On-device measurements

The STM32 latency and memory figures in the paper were measured with the
[ST Edge AI Developer Cloud](https://stedgeai-dc.st.com), using ST Edge AI Core 4.1.0 with "balanced"
optimization and the default memory-pool templates, on four boards: NUCLEO-F401RE, STM32L4R9I-DISCO,
NUCLEO-G474RE and NUCLEO-U3C5ZI-Q.

`python overall.py mcu` writes the full-INT8 `.tflite` file of each model trained on the fixed KFall split,
together with `deployment.json` and the validation tensors, to `results/mcu/<model>/`. Upload the `.tflite` file
to the Developer Cloud to repeat the benchmark. The gate is not part of the exported graph; its cost is counted
analytically as 852 operations per window.

<!-- CONFIRM BEFORE UNCOMMENTING: replace both paths with the real locations in this repository.
The per-run results behind the tables of the paper are in `<path-to-results>`, and the ST Edge AI reports of the
benchmarked models are in `<path-to-reports>`.
-->

## Outputs

Results are written to `code/results/`. Set the `ACT_RESULTS_DIR` environment variable to use another folder.

| Path                                                         | Contents                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `<model>/`                                                   | Fixed-split model (`.keras`, `_fp32.tflite`, `_int8.tflite`), `summary.csv`, per-trial results, threshold sweep, ROC, latency |
| `cv_<dataset>/`                                              | One model, prediction file and config snapshot per (model, fold, seed); `summary.csv`, `paired_tests.csv`, `equivalence_tests.csv` |
| `deployment.csv`                                             | Parameters, MACs, sizes, peak activation memory, CPU latency, FP32 / INT8 metrics |
| `analysis/`, `transfer/`, `quantization/`, `qat/`, `pruning/`, `es_<dataset>/` | Follow-up studies                                            |
| `mcu/<model>/`                                               | INT8 `.tflite`, `deployment.json` (normalization, quantization, gate, thresholds), validation `.npy` files |
| `data/`                                                      | Data manifests: SHA-256 of every input file, per-trial details, excluded recordings |
| `figures/`                                                   | One tidy CSV per figure; rendered figures in `figures/paper/` |

## Configuration

All settings live in [code/config.py](code/config.py). Some useful ones:

- `THRESHOLD_PROTOCOL = "train" | "validation"`: tune the decision and gate thresholds on the training subjects,
  or on a held-out 15 % of them (results then go to `cv_<dataset>_valthr/`).
- `GATE_CONDITIONS`, `GATE_CONSECUTIVE`, `GATE_TRIAL_RECALL`: which gate conditions are used, how many
  consecutive windows they must hold for, and what share of training fall trials the gate must keep.
- `CV_FOLDS`, `CV_SEEDS`, `EPOCHS`, `WINDOW`, `STEP`. Window settings take effect without re-reading the raw
  files.
- `DETERMINISTIC`: set to `True` for bit-identical reruns with the same seed (slower). The runs reported in the
  paper did not use this mode, so repeating them gives results that agree closely but are not bit-identical.

## Citation

If you use this code, please cite:

```bibtex
@unpublished{meng2026rulegating,
  title  = {Kinematic Rule Gating with a Tiny Temporal Convolutional Network for Wearable Fall Detection on Microcontrollers},
  author = {Meng, Delin and Kim, Sanghyeuk and Zong, Shilei and Mahama, Hamida and Yoo, Jaesoo},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

Please also cite the datasets you use:

- X. Yu, J. Jang, S. Xiong. *A Large-Scale Open Motion Dataset (KFall) and Benchmark Algorithms for Detecting
  Pre-impact Fall of the Elderly Using Wearable Inertial Sensors.* Frontiers in Aging Neuroscience 13:692865, 2021.
- A. Sucerquia, J. D. López, J. F. Vargas-Bonilla. *SisFall: A Fall and Movement Dataset.* Sensors 17(1):198, 2017.
- M. Musci et al. *Online Fall Detection Using Recurrent Neural Networks on Smart Wearable Devices.*
  IEEE Transactions on Emerging Topics in Computing 9(3):1276–1289, 2021.
