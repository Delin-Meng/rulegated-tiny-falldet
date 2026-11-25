# Results

This folder contains the result files behind the tables and figures of

> **Kinematic Rule Gating with a Tiny Temporal Convolutional Network for Wearable Fall Detection on Microcontrollers**

Every number reported in the paper can be traced to a file here and recomputed from it. If you only want the headline
numbers, open `cv_kfall/summary.csv` and `cv_sisfall/summary.csv`.

## What is in this folder

```
cv_kfall/, cv_sisfall/     main experiment: 7 backbones x 5 subject-wise folds x 5 seeds
analysis/                  gate ablations, operation counts, memory, missed falls and false alarms
transfer/                  models trained on one dataset and tested on the other
quantization/              quantization variants of the trained models and the INT8 diagnosis
qat/                       quantization-aware training of the two recurrent backbones
es_kfall/, es_sisfall/     early stopping and validation-based thresholds
sweep_tinytcn_kfall/       TinyTCN width and depth sweep
data/                      subject splits and excluded recordings
figures/                   numbers shown in Fig. 5
tinytcn/, tinycnn/, ...    one folder per backbone: the full-INT8 model benchmarked on the boards
MCU/                       flash, RAM and latency measured on the four STM32 boards
```

The CSV files are UTF-8 with a byte-order mark, so they open correctly in Excel; in pandas, read them with
`encoding="utf-8-sig"`.

Two things are not here. The trained Keras models and the per-window predictions of the individual runs take about
3 GB; you can regenerate them with the code (see the last section). The KFall and SisFall datasets are not
redistributed and have to be obtained from their authors.

## Where each table comes from

`<ds>` stands for `kfall` or `sisfall`.

| Paper | Files | What to look at |
|---|---|---|
| Table I | `data/<ds>_splits.csv` | Sum of the five `main (200 epochs)` / `test` rows |
| Table II | `analysis/sizes.csv`, `analysis/macs_by_layer.csv`, `MCU/mcu_paper_table.csv` | Parameters; MACs in the `TOTAL` rows; toolchain MACC in `tool_macc` |
| Table III | `cv_<ds>/summary.csv` | The `FP32 tuned`, `INT8 tuned` and `Gated INT8 tuned` columns |
| Table IV | `analysis/gate_ablation.csv` | Rows with `Windows in a row` = 1 |
| Tables V, VI | `cv_<ds>/all_runs.csv` | Paired differences between two configurations (see "Statistical tests") |
| Table VII | `cv_<ds>/summary.csv`, `cv_<ds>/all_runs.csv`, `quantization/diagnosis/<ds>_summary.csv` | G-means; PTQ loss = `FP32 tuned` minus `INT8 tuned`; ρ = `Spearman FP32 vs INT8 (test windows)` |
| Table VIII | `analysis/ops_per_window.csv`, `analysis/sizes.csv` | `all windows` rows; `Peak activation INT8 (B)` divided by 1,024 |
| Table IX | `MCU/mcu_paper_table.csv`, `MCU/mcu_paper_table_wide.csv` | One row per model and board, or one row per model |
| Table X | `quantization/<ds>/all_runs.csv` | Variant `int8`: `Trial G-mean (tuned)` minus `Trial G-mean (grid 0.50-0.99)`, and the same pair with `Gated` |
| Table XI | `sweep_tinytcn_kfall/summary.csv` | `FP32 tuned` and `Gated INT8 tuned` trial G-mean |
| Table XII | `es_<ds>/comparison_summary.csv` | See the column key below |
| Table XIII | `analysis/gate_recall_sweep.csv` | `kfall` rows, TinyTCN columns |
| Table XIV | `analysis/gate_ablation.csv`, `analysis/gate_transfer.csv`, `transfer/runs.csv` | Thresholds per fold in the `Main gate` row; the "All" row; `Gate fall trials kept, test` |
| Table XV | `analysis/false_alarms_by_subject.csv`, `cv_sisfall/folds.csv` | Specificity = 1 − false alarms / ADL trials, pooled per participant group |
| Table XVI | `cv_<ds>/<ds>_f1_summary.csv` | `Gate` = no with `Setting` = FP32, and `Gate` = yes with `Setting` = INT8 |
| Table XVII | `quantization/<ds>/all_runs.csv`, `quantization/diagnosis/<ds>_summary.csv` | `fp32` minus `weight_int8_sim`; `int8` minus `int8_balanced`; input values outside the INT8 range |
| Fig. 5 | `figures/gate_distributions_kfall_summary.csv` | Share of windows that pass each condition |
| Sec. IV-B, missed falls | `analysis/misses_by_fall_type.csv` | Falls missed by the gate or by the model, per fall type |
| Sec. IV-D, repairs | `quantization/<ds>/summary.csv`, `qat/<ds>/` | Variant `INT16 activations, INT8 weights`; QAT against post-training quantization and the control |
| Sec. IV-F | `transfer/summary.csv`, `transfer/runs.csv` | Results on the other dataset next to the in-dataset results of the same runs |

## How to read the files

- **Per-run and summary files.** `all_runs.csv` has one row per run with numeric values. `summary.csv` gives
  mean ± SD over the runs of each backbone. Metrics are percentages and, unless a column says `Window`, they are
  trial-level.
- **Model names.** `tinycnn` or "TinyCNN series" is TinyCNN, and `musci2021` or "Musci 2021 (LSTM)" is the Musci LSTM.
  "TinyCNN series (valid)" is TinyCNN with its MACs counted under valid padding.
- **Folds** are numbered 0-4 in the files and 1-5 in the paper. `cv_<ds>/folds.csv` lists the test subjects of each.
- **Configurations.** `tuned` means the decision threshold was selected on the training subjects, and `Gated` means
  the model runs behind the rule gate. The three configurations of the paper are `FP32 tuned`, `INT8 tuned` and
  `Gated INT8 tuned`. Columns without `tuned` use a fixed threshold of 0.5; they are a reference setting that the
  paper does not report, labeled "FP32, threshold 0.5 (team rule)" in the F1 summaries.
- **Windows.** An "accident window" is a fall-labeled window.
- **Quantization variants.** `int8` is the full-INT8 model, `weight_int8_sim` quantizes the weights only,
  `int8_balanced` uses the class-balanced calibration set, and `int16x8` has 16-bit activations and 8-bit weights.
  Columns marked `grid 0.50-0.99` use the restricted threshold grid of Table X.
- **Table XII columns.** "Early Stop." is `early stopping (thresholds: training subjects)`, "≤10 Ep." is
  `plan rule (thresholds: training subjects)`, and "Val. Thr." is `early stopping (thresholds: validation subjects)`.
  The "plan rule" is the short schedule of at most 10 epochs with a patience of 3.
- **Operator counts.** Table II counts the operators of the INT8 models in the backbone folders.
  `analysis/sizes.csv` counts the FP32 graph instead, which gives 51 for TinyTransformer and 752 for the Musci LSTM.
  `MCU/mcu_paper_table.csv` also lists 752 for the Musci LSTM, the count of a fully integer conversion; the
  benchmarked file has 756, as noted under Table II.

## Statistical tests

The paper tests differences at the fold level (Section III-D.5), and you can recompute every test from the
`all_runs.csv` files:

1. Take the paired difference between the two columns for each fold and seed.
2. Average the differences over the five seeds of each fold.
3. Apply the Nadeau-Bengio corrected t-test to the five fold-level differences, with the variance factor
   1/5 + 1/4 and four degrees of freedom.
4. Holm-adjust the p-values over the seven backbones of a dataset.

Two configurations are equivalent at a margin if the 90 % confidence interval of the corrected mean difference lies
inside it. The quantization-aware-training comparisons use the five runs with seed 0; their tests are stored in
`qat/<ds>/paired_tests.csv`.

## Regenerating the files

```bash
cd code
bash run_all.sh             # main experiments: cv_<ds>/, analysis/, transfer/, the backbone folders
bash run_extra.sh           # follow-up studies: quantization/, qat/, es_<ds>/, sweep_tinytcn_kfall/,
                            # and transfer/ for all seven backbones
python overall.py f1        # precision and F1 summaries
python overall.py splits    # data/<ds>_splits.csv
python overall.py figures   # figures/
```

The scripts write to `code/results/`, and the files here keep the same relative paths. Training does not use
deterministic TensorFlow kernels, so a rerun gives slightly different numbers.

The on-device results in `MCU/` were measured by uploading the `<model>_int8.tflite` files to the
[ST Edge AI Developer Cloud](https://stedgeai-dc.st.com) (ST Edge AI Core 4.1.0, balanced optimization).
`mcu_all_runs_completed.csv` lists every benchmark run with the fields of the toolchain report, and the remaining
files are the per-model exports. Each `<model>_int8_quantization.json` gives the input normalization and the
input and output scales needed to run the model.
