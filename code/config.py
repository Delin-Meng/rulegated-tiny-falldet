"""All experiment settings in one place. Command-line options of overall.py override some of them."""
import os
from pathlib import Path

USE_GPU = False                # False: hide all GPUs from TensorFlow, every run uses the CPU only
# CPU threads per process (None = TensorFlow default, all cores). These models are tiny, so on many-core
# servers a few threads per process (e.g. 4-8) is usually faster, and leaves cores for parallel processes.
# Can also be set per run with the environment variable CPU_THREADS.
CPU_THREADS = None
# True: tf.config.experimental.enable_op_determinism() (as in the team's cnn_lstm.py) -> bit-identical reruns with
# the same seed, somewhat slower. Off for the experiments already running so that their settings stay the same.
DETERMINISTIC = False

CODE_DIR = Path(__file__).resolve().parent
DATASET_DIR = CODE_DIR.parent / "dataset"
# results folder; the environment variable ACT_RESULTS_DIR points the code at another one (e.g. a copied results/)
RESULTS_DIR = Path(os.environ.get("ACT_RESULTS_DIR") or CODE_DIR / "results")
CACHE_DIR = RESULTS_DIR / "cache"

# ---- data (team common rules) ----
WINDOW, STEP = 50, 10          # 0.5 s / 0.1 s at 100 Hz
FALL_EXT = 50                  # falling period = onset .. impact + 50 rows, later rows are dropped
FALL_MIN = 20                  # >= 40 % of the 50 rows inside the falling period -> accident window
CHANNELS = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
KFALL_TEST_SUBJECTS = ["SA32", "SA33", "SA35", "SA36", "SA37", "SA38"]
KFALL_EXPECTED = {"Trials": 5075, "Falls": 2346, "Subjects": 32, "Windows": 310512, "Accident windows": 24573}

# ---- training (team common rules / paper Sec. 2.3) ----
EPOCHS = 200
BATCH_SIZE = 512
LEARNING_RATE = 0.0005         # Adam; loss = Keras CategoricalFocalCrossentropy with default arguments
FIXED_SEED = 0                 # fixed-split runs
CV_SEEDS = [0, 7, 42, 123, 1337]
CV_FOLDS = 5
CV_FOLD_SEED = 0               # subject-to-fold assignment, identical for every model and seed

# ---- evaluation ----
THRESHOLD = 0.5                # team rule: a trial is an accident if any window has P(fall) >= this
# candidates for the tuned threshold: every output level of the INT8 softmax (k / 256), the same grid for FP32.
# Protocol v1 (results before 2026-10-22) used THRESHOLD_GRID_CAPPED; its lower end 0.50 was binding for the INT8
# LSTM on SisFall (quantization shifts its P(fall) down), which cost it ~17 G-mean points (see quantization.diagnose)
THRESHOLD_GRID = [k / 256 for k in range(1, 256)]
THRESHOLD_GRID_CAPPED = [round(0.50 + 0.01 * i, 2) for i in range(50)]   # 0.50 .. 0.99, kept for the diagnosis
# Which subjects the decision thresholds and the gate thresholds are tuned on:
#   "train"      the subjects the model is trained on (in-sample predictions); protocol of the runs made so far
#   "validation" THRESHOLD_VAL_FRACTION of the training subjects are held out from training and used only for
#                tuning (out-of-sample); chosen once per fold, the same for every model and seed
# Cross-validation results of the two protocols go to different folders (cv_<dataset> / cv_<dataset>_valthr).
THRESHOLD_PROTOCOL = "train"
THRESHOLD_VAL_FRACTION = 0.15
# saved curves: decision threshold sweep (trial level) and ROC (trial score = largest window P(fall))
SWEEP_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]       # 0.05 .. 0.95
ROC_FPR_GRID = [round(0.01 * i, 2) for i in range(101)]            # ROC curves interpolated at these FPRs

# ---- gate: threshold gating (kinematic rule gate), see gate.py ----
# per window: max |a| (g), max |w| (deg/s), max tilt (deg) = angle between -Y and a; the conditions in
# GATE_CONDITIONS (feature indices 0 = acc, 1 = gyr, 2 = tilt) above their thresholds for GATE_CONSECUTIVE windows
# in a row. Y is vertical (AccY ~ -1 g upright) in KFall and SisFall.
# Protocol v1: all three conditions, 2 windows in a row, 99 % training recall; the ablation (analysis item 16)
# showed the 2-window rule changes nothing and the 100 % target costs no accuracy (analysis item 3).
VERTICAL_AXIS = 1
GATE_CONDITIONS = (0, 1, 2)
GATE_CONSECUTIVE = 1
GATE_START = (1.5, 60.0, 45.0)                     # team starting values: acc (g), gyr (deg/s), tilt (deg)
# gate as a detector: grid and criterion (trial balanced accuracy on the training subjects)
GATE_DETECTOR_GRID = ([round(1.5 + 0.25 * i, 2) for i in range(11)],     # 1.5 .. 4.0
                      [60 + 20 * i for i in range(13)],                   # 60 .. 300
                      [45 + 5 * i for i in range(19)])                    # 45 .. 135
# gate as the first stage before a model: thresholds with the smallest share of training windows passing,
# subject to >= GATE_TRIAL_RECALL of the training fall trials keeping at least one accident window.
# The grid starts at 0 so that every recall level is reachable.
GATE_FILTER_GRID = ([0.25 * i for i in range(17)],                        # 0 .. 4.0
                    [20 * i for i in range(16)],                          # 0 .. 300
                    [5 * i for i in range(28)])                           # 0 .. 135
GATE_TRIAL_RECALL = 1.0                            # main setting: no training fall trial dropped by the gate
GATE_RECALL_SWEEP = [0.97, 0.98, 0.99, 0.995, 1.0]  # gate constraint sweep (P1 item 3)

# ---- model groups used by the analyses ----
BASELINE_MODEL = "tinycnn"                          # "the baseline" of the equivalence claims
EQUIVALENCE_GROUP = ["tinytcn", "tinyssm", "tinyeegnet"]   # conv models equivalent to TinyCNN (TOST, 1 point)
GATE_ABLATION_MODELS = ["tinytcn", "tinycnn"]       # gated results per gate variant (P2 item 16)

# ---- early-stopping comparison (review item 15) ----
ES_PATIENCE = 10               # epochs without a lower validation loss before early stopping would stop
ES_VAL_FRACTION = 0.2          # share of each fold's training subjects held out as the validation set
# early stopping on both datasets (P1 item 8): at most EPOCHS epochs, stopped after ES_PATIENCE epochs without a lower
# validation loss, best weights kept; validation = THRESHOLD_VAL_FRACTION of the training subjects. The plan's rule
# (at most ES_PLAN_EPOCHS epochs, patience ES_PLAN_PATIENCE) is read off the same training runs.
ES_PLAN_EPOCHS = 10
ES_PLAN_PATIENCE = 3
ES_MODELS = ["tinytcn", "tinyssm", "tinyeegnet", "tinycnn", "cnnlstm", "tinytransformer", "musci2021"]   # all 7

# ---- statistics ----
TOST_MARGINS = [0.5, 1.0, 2.0]                     # equivalence margins in G-mean percentage points

# ---- deployment (TFLite) ----
CALIB_SAMPLES = 1000           # training windows used to calibrate full-INT8 quantization
CALIB_PERCENTILE = 99.9        # calibration windows clipped at this percentile of |input|
TIMING_RUNS, TIMING_WARMUP = 1000, 100
# latency measurement pinned to this logical CPU (Linux only; None = no pinning). On Intel hybrid CPUs (e.g. the
# i9-14900) Linux numbers the performance-core threads first, so CPU 0 is a P-core.
TIMING_CPU = 0

# ---- quantization variants of the saved cross-validation models (P1 items 1-2, P2 item 15) ----
QUANT_VARIANTS = ["fp16", "weight_int8_tflite", "weight_int8_sim", "int8_balanced", "int16x8"]
CALIB_FALL_SHARE = 0.5         # "int8_balanced": share of accident windows in the calibration set

# ---- pruning (P2 item 14) ----
PRUNE_SPARSITIES = [0.0, 0.3, 0.5, 0.7]            # 0.0 = the same fine-tuning without pruning (control)
PRUNE_FINETUNE_EPOCHS = 30
PRUNE_LEARNING_RATE = 0.0005

# ---- quantization-aware training (P2 item 15) ----
QAT_MODELS = ["cnnlstm", "musci2021"]
QAT_EPOCHS = 15                # fine-tuning, 7.5 % of the 200 training epochs; a QAT epoch costs ~3 normal epochs
QAT_LEARNING_RATE = 0.0001
QAT_EMA_MOMENTUM = 0.99        # moving average of the activation ranges during QAT

# ---- TinyCNN paper reference (Tables 3 and 5, KFall, 5-fold CV mean; G-mean from the reported Se / Sp) ----
PAPER = {"Trial sensitivity": "99.59%", "Trial specificity": "99.49%", "Trial G-mean": "99.54%",
         "INT8 Trial sensitivity": "99.83%", "INT8 Trial specificity": "99.17%", "INT8 Trial G-mean": "99.50%",
         "Parameters": 546, "Epochs": 200}
