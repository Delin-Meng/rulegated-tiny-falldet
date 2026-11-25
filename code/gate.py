"""
Threshold gating (the kinematic rule gate of the paper), used in two ways:
  - detector: the gate alone decides "accident"; thresholds on GATE_DETECTOR_GRID maximising
    trial balanced accuracy, ties broken by window balanced accuracy, then by grid order
  - filter:   the first stage before a model; windows that do not pass count as normal. Thresholds on
    GATE_FILTER_GRID minimising the share of training windows that pass, subject to a minimum share of training
    fall trials that keep at least one passing accident window
Features: per window max |a| (g), max |w| (deg/s) and max tilt (deg),
tilt = angle between -Y and the acceleration. A window passes when all three exceed their thresholds in it and
in the GATE_CONSECUTIVE - 1 windows before it of the same trial.
Ablations (fewer conditions, another number of windows in a row): subset_grid() gives an unused condition the
single threshold -inf (always met), and every function takes the number of windows in a row as `k`.
"""
import numpy as np

import config as cfg

FEATURES = ("acc_max", "gyr_max", "tilt_max")
FEATURE_LABELS = ("acc", "gyr", "tilt")


def window_features(signal):
    """Raw trial signal [T, 6] (g, deg/s) -> gate features [n_windows, 3] for the windows of make_trial."""
    view = np.lib.stride_tricks.sliding_window_view
    acc_svm = np.sqrt((signal[:, :3] ** 2).sum(axis=1))
    gyr_svm = np.sqrt((signal[:, 3:6] ** 2).sum(axis=1))
    cos_tilt = np.clip(-signal[:, cfg.VERTICAL_AXIS] / np.maximum(acc_svm, 1e-6), -1.0, 1.0)
    tilt = np.degrees(np.arccos(cos_tilt))
    return np.stack([view(x, cfg.WINDOW)[::cfg.STEP].max(axis=1) for x in (acc_svm, gyr_svm, tilt)],
                    axis=1).astype(np.float32)


def _starts(trials):
    return np.cumsum([0] + [len(t["y"]) for t in trials[:-1]])


def consecutive_min(features, trials, k=cfg.GATE_CONSECUTIVE):
    """Per feature, the minimum over this window and the k-1 previous windows of the same trial (-inf when the
    trial has fewer). "all features > thresholds" on this is the "k windows in a row" rule."""
    pos = np.concatenate([np.arange(len(t["y"])) for t in trials])     # window position inside its trial
    out = features.astype(np.float64)
    for j in range(1, k):
        prev = np.full_like(out, -np.inf)
        prev[j:] = features[:-j]
        prev[pos < j] = -np.inf
        out = np.minimum(out, prev)
    return out


def passes(gate_features, trials, thresholds, k=cfg.GATE_CONSECUTIVE):
    """Boolean per window: passes the gate."""
    return (consecutive_min(gate_features, trials, k) > np.asarray(thresholds)).all(axis=1)


def subset_grid(grid, used):
    """The grid with the conditions not in `used` (feature indices) reduced to the threshold -inf (always met)."""
    return tuple(list(g) if i in used else [-np.inf] for i, g in enumerate(grid))


def conditions_label(used=cfg.GATE_CONDITIONS):
    return " + ".join(FEATURE_LABELS[i] for i in sorted(used))


def rule_text(used=cfg.GATE_CONDITIONS, k=cfg.GATE_CONSECUTIVE):
    """The gate rule in words, for records and the MCU deployment file."""
    names = {0: "max |a| (g)", 1: "max |w| (deg/s)", 2: "max tilt (deg)"}
    conds = " and ".join(f"{names[i]} > threshold" for i in sorted(used))
    return f"a window passes when {conds}" + (f" in {k} windows in a row" if k > 1 else "") + \
        f"; thresholds: the smallest training pass rate keeping >= {100 * cfg.GATE_TRIAL_RECALL:g} % of the training fall trials"


def _scan(gate_features, y, trials, grid, k=cfg.GATE_CONSECUTIVE):
    """Per threshold combination (acc-major grid order): trial and window
    statistics on the given trials. Returns a dict of arrays shaped [len(acc), len(gyr), len(tilt)]."""
    fc = consecutive_min(gate_features, trials, k)
    starts = _starts(trials)
    is_fall = np.array([t["type"] == "F" for t in trials])
    A = np.asarray(grid[0], float)[:, None]
    shape = (len(grid[0]), len(grid[1]), len(grid[2]))
    out = {k: np.zeros(shape) for k in ("trial_sens", "trial_spec", "win_sens", "win_spec", "pass_rate", "recall")}
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    for ig, g in enumerate(grid[1]):
        for it, t in enumerate(grid[2]):
            a = np.where((fc[:, 1] > g) & (fc[:, 2] > t), fc[:, 0], -np.inf)    # passes iff a > acc threshold
            trial_max = np.maximum.reduceat(a, starts)
            acc_win_max = np.maximum.reduceat(np.where(y == 1, a, -np.inf), starts)[is_fall]
            s_all, s1, s0 = np.sort(a), np.sort(a[y == 1]), np.sort(a[y == 0])
            above = lambda s: len(s) - np.searchsorted(s, A[:, 0], side="right")
            out["trial_sens"][:, ig, it] = (trial_max[is_fall][None] > A).mean(axis=1)
            out["trial_spec"][:, ig, it] = (trial_max[~is_fall][None] <= A).mean(axis=1)
            out["win_sens"][:, ig, it] = above(s1) / n1
            out["win_spec"][:, ig, it] = 1 - above(s0) / n0
            out["pass_rate"][:, ig, it] = above(s_all) / len(a)
            out["recall"][:, ig, it] = (acc_win_max[None] > A).mean(axis=1)
    return out


def _pick(grid, keys):
    """Grid point that is best by `keys` (list of flat arrays, larger is better), first in grid order on ties."""
    order = np.lexsort([np.arange(keys[0].size)] + [-k for k in reversed(keys)])
    ia, ig, it = np.unravel_index(order[0], (len(grid[0]), len(grid[1]), len(grid[2])))
    return (float(grid[0][ia]), float(grid[1][ig]), float(grid[2][it]))


def tune_detector(gate_features, y, trials, grid=cfg.GATE_DETECTOR_GRID, k=cfg.GATE_CONSECUTIVE):
    """Max trial balanced accuracy, then max window balanced accuracy."""
    s = _scan(gate_features, y, trials, grid, k)
    return _pick(grid, [((s["trial_sens"] + s["trial_spec"]) / 2).ravel(),
                        ((s["win_sens"] + s["win_spec"]) / 2).ravel()])


def tune_filter(gate_features, y, trials, recall=cfg.GATE_TRIAL_RECALL, grid=None, k=cfg.GATE_CONSECUTIVE):
    """Smallest training pass rate with fall-trial recall >= `recall`; ties: higher recall, then grid order.
    Default grid: GATE_FILTER_GRID restricted to the conditions in config.GATE_CONDITIONS."""
    grid = subset_grid(cfg.GATE_FILTER_GRID, cfg.GATE_CONDITIONS) if grid is None else grid
    s = _scan(gate_features, y, trials, grid, k)
    feasible = (s["recall"] >= recall - 1e-12).ravel()
    return _pick(grid, [feasible.astype(float), -s["pass_rate"].ravel(), s["recall"].ravel()])


def filter_stats(gate_features, y, trials, thresholds, k=cfg.GATE_CONSECUTIVE):
    """Pass rates (all / ADL windows) and fall-trial recall (>= 1 passing accident window), in %."""
    p = passes(gate_features, trials, thresholds, k)
    adl = np.concatenate([np.full(len(t["y"]), t["type"] == "D") for t in trials])
    starts = _starts(trials)
    is_fall = np.array([t["type"] == "F" for t in trials])
    kept = np.maximum.reduceat((p & (y == 1)).astype(int), starts)[is_fall]
    return {"Pass rate, all windows": 100 * p.mean(), "Pass rate, ADL windows": 100 * p[adl].mean(),
            "Fall trials kept": 100 * kept.mean()}


def ops_per_window(consecutive=cfg.GATE_CONSECUTIVE, used=cfg.GATE_CONDITIONS):
    """Elementary operations of the gate for one window, as a table, for recomputing all WINDOW samples and for
    streaming (each sample's magnitudes and tilt computed once, i.e. STEP new samples per window).
    `used`: the conditions evaluated (feature indices); the tilt needs |a| even without the acc condition."""
    n, new = cfg.WINDOW, cfg.STEP
    acc_mag = [("|a|: square x, y, z", "mul", 3), ("|a|: sum", "add", 2), ("|a|: square root", "sqrt", 1)]
    per_sample = (acc_mag if (0 in used or 2 in used) else []) + \
        ([("|w|: square x, y, z", "mul", 3), ("|w|: sum", "add", 2), ("|w|: square root", "sqrt", 1)]
         if 1 in used else []) + \
        ([("tilt: -a_y / |a|", "div", 1), ("tilt: arccos (+ to degrees)", "acos", 1)] if 2 in used else [])
    names = ", ".join({0: "|a|", 1: "|w|", 2: "tilt"}[i] for i in sorted(used))
    m = len(used)
    per_window = [(f"window max of {names}", "compare", m * (n - 1)),
                  (f"{m} threshold comparison{'s' if m > 1 else ''}", "compare", m),
                  (f"AND of the {m} conditions", "and", m - 1),
                  (f"{consecutive} windows in a row", "and", consecutive - 1)]
    per_window = [r for r in per_window if r[2] > 0]
    rows = [{"Step": s, "Operation": o, "Per sample": c, "Full window (x50)": c * n, f"Streaming (x{new})": c * new}
            for s, o, c in per_sample]
    rows += [{"Step": s, "Operation": o, "Per sample": "", "Full window (x50)": c, f"Streaming (x{new})": c}
             for s, o, c in per_window]
    total_full = sum(c * n for *_, c in per_sample) + sum(c for *_, c in per_window)
    total_stream = sum(c * new for *_, c in per_sample) + sum(c for *_, c in per_window)
    rows.append({"Step": "Total", "Operation": "", "Per sample": "", "Full window (x50)": total_full,
                 f"Streaming (x{new})": total_stream})
    return rows, total_full, total_stream
