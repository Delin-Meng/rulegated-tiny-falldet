"""
Files for microcontroller deployment and on-device validation, from the models trained on
the fixed KFall split (results/<model>/, `python overall.py train`). Per model, results/mcu/<model>/:
  <model>_int8.tflite          full-INT8 model (int8 input / output), the file to load in ST Edge AI / STM32Cube.AI
  deployment.json              what the firmware needs: channel order, window / step, normalisation mean / std,
                               input / output scale and zero point, gate thresholds and rule, decision thresholds
                               (INT8 tuned, without and with the gate), gate cost per window
  validation_input_float.npy   N z-scored test windows [N, 6, 50, 1] (half of them accident windows)
  validation_input_int8.npy    the same quantized with the model's input scale / zero point (what the MCU feeds)
  validation_output_int8.npy   the INT8 model's raw outputs on them [N, 2] (reference for an on-target check)
  validation_labels.npy        window labels (1 = accident window)
  README.txt                   how to get flash / RAM / cycles and an on-target check, and how to record them
"""
import json
import shutil

import keras
import numpy as np

import config as cfg
import deployment
import gate
import models
import preprocessing as pre
import training

FIXED_SPLIT_KEY = 100        # as overall.FIXED_SPLIT_KEY

README = """Model {display}: files for an MCU estimate and an on-device check (P2 item 17).

1. Estimate without a board: ST Edge AI Developer Cloud (https://stedgeai-dc.st.com) or STM32Cube.AI / the
   stedgeai CLI.
   - Load {tflite}. Choose a target, for example an STM32F4 or STM32L4 at its maximum clock.
   - "Analyze" gives flash (weights + code) and RAM (activations).
   - "Benchmark" on the Developer Cloud runs the model on a real board of ST's board farm and gives the
     inference time (cycles). That is a measurement of the model alone, not an emulator; the gate is not part of it.
   - Write the numbers into results/deployment_mcu.csv. Put "estimated (tool)" or "measured (ST board farm)" in
     "Measured or estimated", and the tool version, board and clock in their columns.
2. Check the numerics on a target: "Validate" in the Developer Cloud / STM32Cube.AI with custom data:
   input validation_input_int8.npy (or validation_input_float.npy if the tool quantizes the input itself), expected
   output validation_output_int8.npy. The outputs should match exactly or within 1 level.
3. Firmware (when a board is available): per window, z-score each axis with the mean / std of deployment.json,
   quantize with the input scale / zero point, run the gate (deployment.json "gate") on the raw signal, and only
   for windows that pass run the model; P(fall) = (output[1] - zero point) * scale; a trial is a fall if any
   window reaches the decision threshold ("decision threshold, gated INT8").
"""


def fixed_split(trials, run_file):
    subjects = json.loads(run_file.read_text()).get("subjects") if run_file.exists() else None
    if subjects:
        return pre.experiment_split(trials, subjects["test"], fit_subjects=subjects["fit"],
                                    tune_subjects=subjects["threshold"])
    return pre.experiment_split(trials, cfg.KFALL_TEST_SUBJECTS, key=FIXED_SPLIT_KEY)


def export(names, n_windows=200):
    models.register_custom_layers()
    trials = pre.load_trials("kfall")
    splits = {}
    for name in names:
        src = cfg.RESULTS_DIR / name
        if not (src / f"{name}.keras").exists():
            print(f"{name}: {src / (name + '.keras')} not found, train it first (python overall.py train --models {name})")
            continue
        run_file = src / "run.json"
        key = run_file.read_text() if run_file.exists() else "default"
        if key not in splits:
            splits = {key: fixed_split(trials, run_file)}          # one split in memory at a time
        data = splits[key]
        model = keras.models.load_model(src / f"{name}.keras")
        int8_file = src / f"{name}_int8.tflite"
        tfl = int8_file.read_bytes() if int8_file.exists() else deployment.quantize_int8(model, data["X_tr"])[0]
        mdata = pre.metric_data(data)
        gthr = gate.tune_filter(mdata["gate_tr"], mdata["y_tr"], mdata["train"])
        p8_tune = deployment.predict(tfl, data["X_tune"].astype(np.float32))
        pass_tune = gate.passes(mdata["gate_tr"], mdata["train"], gthr)
        rng = np.random.default_rng(0)
        y = data["y_te"]
        half = n_windows // 2
        idx = np.concatenate([rng.choice(np.flatnonzero(y == 1), min(half, int((y == 1).sum())), replace=False),
                              rng.choice(np.flatnonzero(y == 0), n_windows - min(half, int((y == 1).sum())), replace=False)])
        X = data["X_te"][idx].astype(np.float32)
        q = deployment.quantization_info(tfl)
        s_in, z_in = q["input"]["scale"], q["input"]["zero point"]
        X8 = np.clip(np.round(X / s_in + z_in), -128, 127).astype(np.int8)
        it = deployment._interpreter(tfl)
        inp, out = it.get_input_details()[0], it.get_output_details()[0]
        raw = []
        for x in X8:
            it.set_tensor(inp["index"], x[None])
            it.invoke()
            raw.append(it.get_tensor(out["index"])[0].copy())
        raw = np.array(raw, np.int8)
        dst = cfg.RESULTS_DIR / "mcu" / name
        dst.mkdir(parents=True, exist_ok=True)
        (dst / f"{name}_int8.tflite").write_bytes(tfl)
        np.save(dst / "validation_input_float.npy", X)
        np.save(dst / "validation_input_int8.npy", X8)
        np.save(dst / "validation_output_int8.npy", raw)
        np.save(dst / "validation_labels.npy", y[idx].astype(np.int8))
        ops_full, ops_stream = gate.ops_per_window()[1:]
        info = {"model": name, "display name": models.display_name(name), "parameters": int(model.count_params()),
                "MACs / window": deployment.count_macs(deployment.convert(model)),
                ".tflite INT8 (bytes)": len(tfl), "peak activation memory INT8 (bytes, estimate)":
                    deployment.peak_activation_bytes(tfl),
                "channels": cfg.CHANNELS, "units": "acceleration in g, angular velocity in deg/s",
                "sampling rate (Hz)": 100, "window (samples)": cfg.WINDOW, "step (samples)": cfg.STEP,
                "input layout": "[1, 6 axes, 50 samples, 1]",
                "normalisation": {"mean": [float(v) for v in np.ravel(data["mean"])],
                                  "std": [float(v) for v in np.ravel(data["std"])]},
                **q,
                "gate": {"features": "per window: max |a| (g), max |w| (deg/s), max tilt (deg) = arccos(-a_y / |a|)",
                         "conditions used": gate.conditions_label(),
                         "thresholds (acc g, gyr deg/s, tilt deg; -inf = condition not used)": [float(v) for v in gthr],
                         "rule": gate.rule_text(),
                         "ops / window": {"recomputing the window": ops_full, "streaming": ops_stream}},
                "decision threshold, INT8": training.select_threshold(p8_tune, mdata["train"]),
                "decision threshold, gated INT8": training.select_threshold(p8_tune * pass_tune, mdata["train"]),
                "trial rule": "fall if any window has P(fall) >= the decision threshold",
                "thresholds tuned on": f"the fixed KFall split's threshold-tuning subjects ({cfg.THRESHOLD_PROTOCOL} protocol)",
                "validation windows": {"count": int(len(idx)), "accident windows": int((y[idx] == 1).sum()),
                                       "drawn from": "the fixed split's test subjects, random seed 0"}}
        (dst / "deployment.json").write_text(json.dumps(info, indent=1))
        (dst / "README.txt").write_text(README.format(display=models.display_name(name), tflite=f"{name}_int8.tflite"),
                                        encoding="utf-8")
        if (src / f"{name}_int8_quantization.json").exists():
            shutil.copyfile(src / f"{name}_int8_quantization.json", dst / f"{name}_int8_quantization.json")
        print(f"{name}: MCU files in {dst}")
