"""
Deployment analysis with TensorFlow Lite, applied identically to every model:
  - conversion of the batch-1 static graph (what a microcontroller runs): FP32 and full INT8
    (int8 weights, activations and I/O); if full INT8 conversion fails, weight-only INT8 is used instead
  - further variants (quantization.py): float16 weights, int16 activations with int8 weights (16x8),
    TFLite dynamic-range INT8, and weight-only INT8 simulated on the FP32 graph
  - MACs counted on the converted FP32 graph (CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, BATCH_MATMUL)
  - size: parameter bytes (arithmetic), constant data stored in the .tflite (weights as deployed, e.g. the
    SSM's expanded convolution kernel) and the .tflite file size (adds graph structure / quantization metadata)
  - peak activation memory from tensor lifetimes in execution order, an estimate of the tensor arena (graph
    inputs / outputs kept for the whole run, as TFLite's arena planner does)
  Graph structure (ops, MACs, sizes, lifetimes) is read from the plain converted graph, i.e. an interpreter without
  the XNNPACK delegate: with the delegate, the op list carries the delegated partitions as extra DELEGATE nodes
  (op count off by the number of partitions, lifetimes garbled, and the partitioning depends on the quantization
  parameters, so the same architecture could get different "peak activation" values from run to run)
  - CPU latency with the TFLite interpreter, 1 thread, batch 1, median over config.TIMING_RUNS runs, pinned to
    logical CPU config.TIMING_CPU where the OS allows it (Linux)
Settings in config.py; run through `python overall.py deploy`.
"""
import contextlib
import io
import json
import os
import time
import warnings

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

import config as cfg
from utils import cpu_name

warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")

TIMING_METHOD = f"TFLite, 1 thread, batch 1, median of {cfg.TIMING_RUNS} runs ({cfg.TIMING_WARMUP} warm-up)"
WEIGHT_OPS = {"CONV_2D": 0, "DEPTHWISE_CONV_2D": 3, "FULLY_CONNECTED": 0}   # weight input 1, output-channel axis


CALIB_SEED = 0


def timing_method():
    pinned = cfg.TIMING_CPU is not None and hasattr(os, "sched_setaffinity")
    return TIMING_METHOD + (f", pinned to logical CPU {cfg.TIMING_CPU}" if pinned else "")


@contextlib.contextmanager
def pinned_cpu(cpu=cfg.TIMING_CPU):
    """Run the calling thread on one logical CPU (Linux) and restore the previous affinity afterwards."""
    if cpu is None or not hasattr(os, "sched_setaffinity"):
        yield
        return
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {cpu})
    try:
        yield
    finally:
        os.sched_setaffinity(0, old)


def calibration_set(X_rep, y_rep=None, fall_share=None):
    """Representative windows for full-INT8 calibration and their description.
    z-scored inputs reach +-40 at fall impacts while 99.9 % lie within +-8.7; calibrating on min/max would spend
    the 256 int8 levels on those spikes, so the calibration windows are clipped at CALIB_PERCENTILE of |input|.
    fall_share (needs y_rep): that share of the windows is drawn from the accident windows and the rest from the
    others (0.5 = balanced); default: a uniform random draw, as in the main runs."""
    rng = np.random.default_rng(CALIB_SEED)
    n = min(cfg.CALIB_SAMPLES, len(X_rep))
    if fall_share is None or y_rep is None:
        idx = rng.choice(len(X_rep), n, replace=False)
    else:
        y = np.asarray(y_rep)
        falls, others = np.flatnonzero(y == 1), np.flatnonzero(y != 1)
        n_fall = min(len(falls), int(round(fall_share * n)))
        idx = np.concatenate([rng.choice(falls, n_fall, replace=False), rng.choice(others, n - n_fall, replace=False)])
    clip = float(np.percentile(np.abs(X_rep), cfg.CALIB_PERCENTILE))  # over all training windows
    info = {"windows": int(len(idx)), "drawn from": f"the model's {len(X_rep)} training windows",
            "random seed": CALIB_SEED, "clip percentile of |z|": cfg.CALIB_PERCENTILE,
            "clip value (z units)": clip}
    if fall_share is not None:
        info["requested fall window share"] = fall_share
    if y_rep is not None:
        info["fall windows in the calibration set"] = int(np.asarray(y_rep)[idx].sum())
        info["fall window share (%)"] = float(100 * np.asarray(y_rep)[idx].mean())
    return np.clip(X_rep[idx], -clip, clip).astype(np.float32), info


def quantization_info(tflite):
    """dtype, scale and zero point of the input and output tensors of a converted model."""
    it = _interpreter(tflite)
    io = lambda d: {"dtype": np.dtype(d["dtype"]).name, "scale": float(d["quantization"][0]),
                    "zero point": int(d["quantization"][1]), "shape": [int(s) for s in d["shape"]]}
    return {"input": io(it.get_input_details()[0]), "output": io(it.get_output_details()[0])}


def int8_record(tflite, mode, X_rep, y_rep=None, subjects=None, mean=None, std=None):
    """Everything the quantization section and an MCU port need: calibration set, I/O quantization and how a raw
    sample becomes the int8 model input."""
    rec = {"mode": mode, "calibration": calibration_set(X_rep, y_rep)[1], **quantization_info(tflite)}
    if subjects is not None:
        rec["calibration"]["subjects"] = list(subjects)
    if mean is not None:
        rec["input preprocessing"] = {
            "formula": "q = clip(round(((x - mean) / std) / input_scale + input_zero_point), -128, 127)",
            "channels": cfg.CHANNELS, "mean": [float(v) for v in np.ravel(mean)],
            "std": [float(v) for v in np.ravel(std)]}
    return rec


def run_converter(converter):
    """converter.convert() without the SavedModel export messages it prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return converter.convert()


def convert(model, mode="fp32", X_rep=None, y_rep=None, fall_share=None):
    """mode: 'fp32' | 'int8' (full integer, int8 input/output) | 'int16x8' (int16 activations and I/O, int8
    weights) | 'fp16' (float16 weights) | 'weight_only' (TFLite dynamic-range INT8).
    X_rep / y_rep / fall_share: calibration windows of the integer modes, see calibration_set()."""
    # fixed batch size 1 removes the dynamic-shape ops; from_keras_model freezes the weights into constants
    inp = keras.Input(batch_shape=(1, *model.input_shape[1:]))
    converter = tf.lite.TFLiteConverter.from_keras_model(keras.Model(inp, model(inp)))
    if mode != "fp32":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if mode == "fp16":
        converter.target_spec.supported_types = [tf.float16]
    if mode in ("int8", "int16x8"):
        rep, _ = calibration_set(X_rep, y_rep, fall_share)

        def representative_data():
            for i in range(len(rep)):
                yield [rep[i:i + 1]]

        converter.representative_dataset = representative_data
        if mode == "int8":
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = converter.inference_output_type = tf.int8
        else:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            converter.inference_input_type = converter.inference_output_type = tf.int16
    return run_converter(converter)


def quantize_int8(model, X_rep, y_rep=None, fall_share=None):
    """Full INT8 if the converter accepts the graph, otherwise weight-only INT8.
    Note: TFLite weight-only quantization skips weight tensors with < 1024 elements, which is every
    tensor of these tiny models, so the fallback file is effectively FP32 (its size shows this).
    Returns (tflite bytes, mode, error message of the failed full-INT8 attempt)."""
    try:
        return convert(model, "int8", X_rep, y_rep, fall_share), "full INT8", ""
    except Exception as e:                                      # noqa: BLE001 - report any converter failure
        return convert(model, "weight_only"), "weight-only INT8", f"{type(e).__name__}: {e}".splitlines()[0]


def weight_tensors(tflite):
    """Constant weight tensors of the conv / fully-connected ops: {tensor index: (op name, dtype name, size)}."""
    ops, tensors, _, _ = graph_info(tflite)
    produced = {t for op in ops for t in op["outputs"]}
    out = {}
    for op in ops:
        if op["op_name"] in WEIGHT_OPS and len(op["inputs"]) > 1 and op["inputs"][1] not in produced:
            t = tensors[op["inputs"][1]]
            out[op["inputs"][1]] = (op["op_name"], np.dtype(t["dtype"]).name, int(np.prod(t["shape"])))
    return out


def simulate_weight_only_int8(tflite):
    """The FP32 graph with every conv / fully-connected weight tensor rounded to symmetric per-output-channel int8
    and back: what storing the weights as int8 with float compute (weight-only INT8) does to the predictions.
    Biases and other constants stay float32. Returns (tflite bytes, info)."""
    from tensorflow.lite.python import schema_py_generated as schema_fb
    from tensorflow.lite.tools import flatbuffer_utils as fbu
    model = fbu.read_model_from_bytearray(bytearray(tflite))
    graph = model.subgraphs[0]
    done, elements, channels = set(), 0, 0
    for op in graph.operators:
        name = fbu.opcode_to_name(model, op.opcodeIndex)
        if name not in WEIGHT_OPS or op.inputs is None or len(op.inputs) < 2:
            continue
        t = graph.tensors[op.inputs[1]]
        buf = model.buffers[t.buffer]
        if t.type != schema_fb.TensorType.FLOAT32 or buf.data is None or len(buf.data) == 0 or t.buffer in done:
            continue
        w = np.frombuffer(bytes(buf.data), dtype=np.float32).reshape(tuple(int(s) for s in t.shape))
        axis = WEIGHT_OPS[name]
        scale = np.abs(w).max(axis=tuple(i for i in range(w.ndim) if i != axis), keepdims=True) / 127
        scale[scale == 0] = 1.0
        wq = (np.clip(np.round(w / scale), -127, 127) * scale).astype(np.float32)
        buf.data[:] = np.frombuffer(wq.tobytes(), dtype=np.uint8)
        done.add(t.buffer)
        elements += w.size
        channels += w.shape[axis]
    info = {"weight tensors rounded to int8": len(done), "weight elements": int(elements),
            "weight bytes as int8 + float32 scales": int(elements + 4 * channels)}
    return bytes(fbu.convert_object_to_bytearray(model)), info


def float_activation_tensors(tflite):
    """Number of op outputs with a floating-point dtype (0 for a fully integer graph)."""
    ops, tensors, _, _ = graph_info(tflite)
    return sum(np.issubdtype(np.dtype(tensors[t]["dtype"]), np.floating) for t in {t for op in ops for t in op["outputs"]})


def nonzero_macs(tflite):
    """MACs that multiply a nonzero weight (conv / fully-connected ops of an FP32 graph), i.e. the work left with
    sparse kernels after pruning. Dense TFLite kernels still execute count_macs()."""
    it = _plain_interpreter(tflite)
    total = 0
    weights = {i: it.get_tensor(i) for i in weight_tensors(tflite)}
    ops = it._get_ops_details()
    rows = [r for r in macs_by_op(tflite)]
    k = 0
    for op in ops:
        if op["op_name"] not in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "BATCH_MATMUL"):
            continue
        macs = rows[k][-1]
        k += 1
        w = weights.get(op["inputs"][1]) if len(op["inputs"]) > 1 else None
        total += macs * (np.count_nonzero(w) / w.size if w is not None and w.size else 1.0)
    return int(round(total))


def _interpreter(tflite, threads=1):
    """Interpreter for inference and timing (default delegates, i.e. XNNPACK where it applies)."""
    it = tf.lite.Interpreter(model_content=tflite, num_threads=threads)
    it.allocate_tensors()
    return it


def _plain_interpreter(tflite):
    """Interpreter without the default delegate, for reading the graph: its op list is the converted graph itself
    (with XNNPACK, _get_ops_details() appends one DELEGATE node per delegated partition)."""
    it = tf.lite.Interpreter(model_content=tflite, num_threads=1,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    return it


def _io(it):
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    integer = (np.int8, np.int16)

    def put(x):
        if inp["dtype"] in integer:
            scale, zero = inp["quantization"]
            lim = np.iinfo(inp["dtype"])
            x = np.clip(np.round(x / scale + zero), lim.min, lim.max).astype(inp["dtype"])
        it.set_tensor(inp["index"], x)

    def get():
        y = it.get_tensor(out["index"])
        if out["dtype"] in integer:
            scale, zero = out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y
    return put, get


def predict(tflite, X):
    """P(fall) for every window of X [n, 6, 50, 1], one window at a time (batch-1 graph)."""
    it = _interpreter(tflite)
    put, get = _io(it)
    prob = np.empty(len(X), np.float32)
    for i in range(len(X)):
        put(X[i:i + 1])
        it.invoke()
        prob[i] = get()[0, 1]
    return prob


def latency_samples(tflite, x1):
    """TIMING_RUNS single-window inference times (ms) after TIMING_WARMUP warm-up runs, 1 thread, on logical CPU
    config.TIMING_CPU where possible."""
    it = _interpreter(tflite, threads=1)
    put, get = _io(it)
    with pinned_cpu():
        for _ in range(cfg.TIMING_WARMUP):
            put(x1); it.invoke(); get()
        times = []
        for _ in range(cfg.TIMING_RUNS):
            t0 = time.perf_counter()
            put(x1); it.invoke(); get()
            times.append(time.perf_counter() - t0)
    return 1e3 * np.array(times)


def latency_ms(tflite, x1):
    return float(np.median(latency_samples(tflite, x1)))


def graph_info(tflite):
    """(ops in execution order, tensor details by index, graph input indices, graph output indices) of a converted
    model's plain graph (no delegate nodes)."""
    it = _plain_interpreter(tflite)
    return (it._get_ops_details(), {d["index"]: d for d in it.get_tensor_details()},
            {d["index"] for d in it.get_input_details()}, {d["index"] for d in it.get_output_details()})


def macs_by_op(tflite):
    """Per conv / fully-connected op of the converted graph, in execution order:
    (op, output shape, weight shape, MACs per output, MACs). Element-wise ops are not counted."""
    ops, tensors, _, _ = graph_info(tflite)
    rows = []
    for op in ops:
        name = op["op_name"]
        if name not in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "BATCH_MATMUL"):
            continue
        out_shape, w = tensors[op["outputs"][0]]["shape"], tensors[op["inputs"][1]]["shape"]
        if name == "CONV_2D":                                   # weights [out_c, kh, kw, in_c]
            per_output = w[1] * w[2] * w[3]
        elif name == "DEPTHWISE_CONV_2D":                       # weights [1, kh, kw, channels]
            per_output = w[1] * w[2]
        elif name == "FULLY_CONNECTED":                         # weights [out, in]
            per_output = w[1]
        else:                                                   # BATCH_MATMUL [.., M, K] x [.., K, N] (attention)
            a = tensors[op["inputs"][0]]["shape"]
            per_output = a[-1] if a[-2] == out_shape[-2] else a[-2]     # inner dim, also when lhs is transposed
        rows.append((name, tuple(int(s) for s in out_shape), tuple(int(s) for s in w), int(per_output),
                     int(np.prod(out_shape)) * int(per_output)))
    return rows


def count_macs(tflite):
    """MACs per window of the conv / fully-connected ops in the converted graph (element-wise ops excluded)."""
    return sum(r[-1] for r in macs_by_op(tflite))


def _nbytes(t):
    return int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize


def constant_bytes(tflite):
    """Bytes of constant tensors (weights, biases, shape constants): tensors no op produces and not inputs."""
    ops, tensors, graph_in, _ = graph_info(tflite)
    produced = {t for op in ops for t in op["outputs"]}
    used = {t for op in ops for t in op["inputs"] if t >= 0}
    return sum(_nbytes(tensors[t]) for t in used - produced - graph_in)


def peak_activation_bytes(tflite):
    """Largest total size of activation tensors alive at the same time while the ops of the plain graph run in
    order: an estimate of the tensor arena. Graph inputs and outputs are kept for the whole run (as TFLite's arena
    planner does); weights are excluded (they stay in flash); buffers are not shared in place, so this is an upper
    estimate of what a memory planner needs."""
    ops, tensors, graph_in, graph_out = graph_info(tflite)
    first, last = {t: 0 for t in graph_in}, {}
    for i, op in enumerate(ops):
        for t in op["inputs"]:
            if t in first:                              # produced earlier or graph input -> activation
                last[t] = i
        for t in op["outputs"]:
            first[t] = i
            last.setdefault(t, i)
    for t in graph_in | graph_out:
        last[t] = len(ops) - 1
    return max(sum(_nbytes(tensors[t]) for t in first if first[t] <= i <= last[t]) for i in range(len(ops)))


def size_info(model, fp32, int8, int8_mode):
    """The graph-derived part of the deployment record: MACs, op count, parameter / constant / file bytes and peak
    activation memory of the FP32 and INT8 .tflite files."""
    n = model.count_params()
    return {
        "MACs / window": count_macs(fp32),
        "Operators in graph": len(graph_info(fp32)[0]),
        "Parameter bytes FP32 (KB, params x 4)": f"{n * 4 / 1024:.2f}",
        "Parameter bytes INT8 (KB, params x 1)": f"{n / 1024:.2f}",
        "Constant data in .tflite FP32 (KB)": f"{constant_bytes(fp32) / 1024:.2f}",
        "Constant data in .tflite INT8 (KB)": f"{constant_bytes(int8) / 1024:.2f}",
        ".tflite file FP32 (KB)": f"{len(fp32) / 1024:.1f}",
        ".tflite file INT8 (KB)": f"{len(int8) / 1024:.1f}",
        "INT8 mode": int8_mode,
        "Peak activation memory FP32 (KB)": f"{peak_activation_bytes(fp32) / 1024:.2f}",
        "Peak activation memory INT8 (KB)": f"{peak_activation_bytes(int8) / 1024:.2f}",
    }


def refresh_sizes(names, results_dir=None):
    """Recompute the graph-derived columns (size_info) of the fixed-split records from the saved
    results/<model>/<model>_{fp32,int8}.tflite files and write them into deployment.csv, comparison.csv and
    <model>/summary.csv; the measured columns (latency, metrics) stay. For records written before graph_info read
    the plain graph: the delegate nodes inflated the op count by the number of XNNPACK partitions and distorted the
    peak activation memory of the models with more than one partition."""
    import models
    results_dir = results_dir or cfg.RESULTS_DIR
    models.register_custom_layers()
    new = {}
    for name in names:
        d = results_dir / name
        files = [d / f"{name}.keras", d / f"{name}_fp32.tflite", d / f"{name}_int8.tflite"]
        if not all(f.exists() for f in files):
            print(f"{name}: saved model / .tflite files not found in {d}, skipped")
            continue
        qfile = d / f"{name}_int8_quantization.json"
        mode = json.loads(qfile.read_text()).get("mode", "full INT8") if qfile.exists() else "full INT8"
        new[models.display_name(name)] = size_info(keras.models.load_model(files[0]), files[1].read_bytes(),
                                                   files[2].read_bytes(), mode)
    for path in [results_dir / "deployment.csv", results_dir / "comparison.csv",
                 *(results_dir / n / "summary.csv" for n in names)]:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        changes = []
        for i, m in df["Model"].items():
            for k, v in new.get(m, {}).items():
                if k in df.columns and df.at[i, k] != str(v):
                    changes.append(f"{m}: {k} {df.at[i, k]} -> {v}")
                    df.at[i, k] = str(v)
        if changes:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"{path}:\n  " + "\n  ".join(changes))
        else:
            print(f"{path}: up to date")
    return new


def analyze(model, X_tr, X_te, out_dir, key, X_tune=None, y_tr=None, subjects=None, mean=None, std=None):
    """Convert, measure and save in out_dir:
      <key>_fp32.tflite, <key>_int8.tflite (full INT8, or weight-only if that fails)
      <key>_int8_quantization.json   calibration set, I/O scale / zero point, input preprocessing for an MCU
      <key>_latency_fp32.csv, <key>_latency_int8.csv   all TIMING_RUNS single-window inference times (ms)
    X_tr: the model's training windows (INT8 calibration); X_tune: windows the thresholds are tuned on
    (default X_tr). Returns (metrics dict, INT8 P(fall) on X_tune, INT8 P(fall) on X_te)."""
    X_tune = X_tr if X_tune is None else X_tune
    fp32 = convert(model, "fp32")
    int8, int8_mode, int8_error = quantize_int8(model, X_tr)
    (out_dir / f"{key}_fp32.tflite").write_bytes(fp32)
    (out_dir / f"{key}_int8.tflite").write_bytes(int8)
    if int8_error:
        print(f"  full INT8 conversion failed, using weight-only INT8: {int8_error}")
    (out_dir / f"{key}_int8_quantization.json").write_text(json.dumps(
        int8_record(int8, int8_mode, X_tr, y_tr, subjects, mean, std), indent=1))
    x1 = X_te[:1].astype(np.float32)
    times = {}
    for kind, tfl in (("FP32", fp32), ("INT8", int8)):
        times[kind] = latency_samples(tfl, x1)
        pd.DataFrame({"run": np.arange(1, len(times[kind]) + 1), "ms": times[kind]}).to_csv(
            out_dir / f"{key}_latency_{kind.lower()}.csv", index=False)
    info = {
        **size_info(model, fp32, int8, int8_mode),
        **{f"Inference time {kind} {stat} (ms)": f"{np.percentile(t, q):.4f}"
           for kind, t in times.items() for stat, q in (("median", 50), ("p5", 5), ("p95", 95))},
        **{f"Inference time {kind} mean ± std (ms)": f"{t.mean():.4f} ± {t.std(ddof=1):.4f}" for kind, t in times.items()},
        "Timing method": timing_method(),
        "CPU": cpu_name(),
    }
    return info, predict(int8, X_tune.astype(np.float32)), predict(int8, X_te.astype(np.float32))
