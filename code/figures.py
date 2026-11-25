"""
Paper figures from the CSVs the experiments write, saved as PNG (300 dpi) and PDF in results/figures/paper/.
A figure whose data are missing is skipped with a message, so this can run after any subset of the experiments.

Style: the documented default categorical palette in its validated order (adjacent forms use slots 1-4, scatter-type
forms at most 3 hues), solid hairline grid, no dual axes. Figures with all seven models use the emphasis form:
TinyTCN (the proposed model) in slot 1, the TinyCNN baseline in slot 2, the other models in gray with direct labels.
Static print figures: no hover layer or dark mode; each figure's data CSV is its table view.
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import config as cfg

OUT = cfg.RESULTS_DIR / "figures" / "paper"
FIG = cfg.RESULTS_DIR / "figures"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4 (validated order)
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#1c5cab"                  # sequential blue steps 250 / 550 (dumbbells)
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
CENTER, BASELINE = "TinyTCN", "TinyCNN series"
SHORT = {"TinyCNN series": "TinyCNN", "Musci 2021 (LSTM)": "Musci LSTM", "TinyCNN series (valid)": "TinyCNN (valid)"}
DATASETS = {"kfall": "KFall", "sisfall": "SisFall"}
EQUIVALENT = ["TinyTCN", "TinyCNN series", "TinySSM", "TinyEEGNet"]
WIDTH_1, WIDTH_2 = 3.5, 7.0                                    # single / double column (in)


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlesize": 8.5,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.5, "lines.markersize": 5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": INK, "axes.axisbelow": True, "figure.constrained_layout.use": True})


def short(name):
    return SHORT.get(name, name)


def plain_log_axis(ax, axis="x"):
    """Log axis with plain tick labels, compact from 1,000 up (500 / 1K / 10K)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fmt = FuncFormatter(lambda v, _: f"{v / 1000:g}K" if v >= 1000 else f"{v:g}")
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(fmt)
    a.set_minor_formatter(NullFormatter())


def color(model):
    return {CENTER: SLOT[0], BASELINE: SLOT[1]}.get(model, MUTED)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def read(path):
    if not path.exists():
        print(f"  skipped: {path.relative_to(cfg.RESULTS_DIR)} not found")
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def mean_std(text):
    """'98.71 ± 0.55' -> (98.71, 0.55)."""
    m = re.match(r"\s*(-?[\d.]+)\s*±\s*([\d.]+)", str(text))
    return (float(m.group(1)), float(m.group(2))) if m else (float(text), 0.0)


def emphasis_handles(extra=()):
    return [Line2D([], [], marker="o", ls="", color=SLOT[0], label=f"{CENTER} (proposed)"),
            Line2D([], [], marker="o", ls="", color=SLOT[1], label="TinyCNN (baseline)"),
            Line2D([], [], marker="o", ls="", color=MUTED, label="other models"), *extra]


def figure_legend(fig, handles, ncol=None):
    """One legend below all panels (also when a panel is hidden)."""
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol or len(handles),
               frameon=False)


def place_labels(ax, points, texts, fontsize=6.5):
    """Direct labels beside points, each at the first candidate offset that overlaps neither an earlier label nor a
    marker (a label that fits nowhere keeps the first offset)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    taken = []
    for x, y in points:
        px, py = ax.transData.transform((x, y))
        taken.append(Bbox([[px - 5, py - 5], [px + 5, py + 5]]))
    offsets = [(6, 0, "left"), (-6, 0, "right"), (6, 8, "left"), (6, -8, "left"), (-6, 8, "right"),
               (-6, -8, "right"), (0, 11, "center"), (0, -11, "center"), (6, 15, "left"), (6, -15, "left"),
               (-6, 15, "right"), (-6, -15, "right"), (10, 24, "left"), (10, -24, "left"), (-10, 24, "right"),
               (-10, -24, "right"), (0, 32, "center"), (0, -32, "center")]
    inside = ax.get_window_extent(renderer)
    for (x, y), s in zip(points, texts):
        first = None
        for dx, dy, ha in offsets:
            far = abs(dx) + abs(dy) > 16                   # a label pushed away gets a hairline leader to its point
            t = ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=INK2,
                            ha=ha, va="center",
                            arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6, shrinkA=0, shrinkB=3) if far else None)
            bb = t.get_window_extent(renderer)
            within = bb.x0 >= inside.x0 and bb.x1 <= inside.x1 and bb.y0 >= inside.y0 and bb.y1 <= inside.y1
            if within and not any(bb.overlaps(b) for b in taken):
                taken.append(bb)
                break
            first = first or (dx, dy, ha)
            t.remove()
        else:
            dx, dy, ha = first
            t = ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=INK2,
                            ha=ha, va="center")
            taken.append(t.get_window_extent(renderer))
    ax.figure.canvas.draw()


# ------------------------------------------------------------------ figures

def pareto():
    """Operations per window vs trial G-mean: each model alone (FP32, hollow) -> gate + INT8 (filled)."""
    ops = read(cfg.RESULTS_DIR / "analysis" / "ops_per_window.csv")
    if ops is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2, 3.2), sharey=False)
    drawn = False
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        g = read(FIG / f"cv_{ds}_macs_vs_gmean.csv")
        if g is None:
            ax.set_visible(False)
            continue
        o = ops[(ops["Dataset"] == ds) & (ops["Windows"] == "all windows")].set_index("Model")["Avg ops / window"]
        points, texts = [], []
        for _, r in g.iterrows():
            m, c = r["model"], color(r["model"])
            if m not in o:
                continue
            x0, y0 = r["MACs / window"], r["FP32 tuned Trial G-mean mean"]
            x1, y1 = o[m], r["Gated INT8 tuned Trial G-mean mean"]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=AXIS if c == MUTED else c, lw=0.8, shrinkA=4, shrinkB=4))
            ax.errorbar(x0, y0, yerr=r["FP32 tuned Trial G-mean std"], fmt="o", mfc="white", mec=c, ecolor=c,
                        elinewidth=0.8, capsize=0, ms=5, mew=1.2, zorder=3)
            ax.errorbar(x1, y1, yerr=r["Gated INT8 tuned Trial G-mean std"], fmt="o", color=c, ecolor=c,
                        elinewidth=0.8, capsize=0, ms=5, mec="white", mew=1.0, zorder=4)
            points.append((x1, y1))
            texts.append(short(m))
        ax.set_xscale("log")
        plain_log_axis(ax)
        ax.set_xlabel("Operations per window (log scale)")
        ax.set_title(title, loc="left")
        place_labels(ax, points, texts)
        drawn = True
    axes[0].set_ylabel("Trial G-mean (%)")
    figure_legend(fig, emphasis_handles([Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="model alone, FP32"),
                                         Line2D([], [], marker="o", ls="", color=INK2, label="gate + model, INT8")]))
    if drawn:
        save(fig, "fig_pareto_ops_vs_gmean")


def int8_dumbbell():
    """Per model: FP32 -> full INT8 (own tuned threshold), both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2, 2.8), sharey=True)
    order = None
    drawn = False
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        s = read(cfg.RESULTS_DIR / f"cv_{ds}" / "summary.csv")
        if s is None:
            ax.set_visible(False)
            continue
        s = s.set_index("Model")
        if order is None:
            order = sorted(s.index, key=lambda m: mean_std(s.loc[m, "FP32 tuned Trial G-mean"])[0])
        for i, m in enumerate(order):
            if m not in s.index:
                continue
            (f, fs), (q, qs) = mean_std(s.loc[m, "FP32 tuned Trial G-mean"]), mean_std(s.loc[m, "INT8 tuned Trial G-mean"])
            ax.plot([f, q], [i, i], color=GRID, lw=2, zorder=1)
            ax.errorbar(f, i, xerr=fs, fmt="o", color=BLUE_LIGHT, elinewidth=0.8, ms=5, mec="white", zorder=3)
            ax.errorbar(q, i, xerr=qs, fmt="o", color=BLUE_DARK, elinewidth=0.8, ms=5, mec="white", zorder=4)
        ax.set_yticks(range(len(order)), [short(m) for m in order])
        ax.set_xlabel("Trial G-mean (%)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        drawn = True
    figure_legend(fig, [Line2D([], [], marker="o", ls="", color=BLUE_LIGHT, label="FP32"),
                        Line2D([], [], marker="o", ls="", color=BLUE_DARK, label="full INT8")])
    if drawn:
        save(fig, "fig_int8_fp32_vs_int8")


def gate_sweep():
    """Gate constraint sweep: pass rate, fall trials kept, gated INT8 G-mean of the equivalence group."""
    d = read(FIG / "gate_recall_sweep.csv")
    if d is None:
        return
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH_2, 6.0), sharex=True)
    for col, (ds, title) in enumerate(DATASETS.items()):
        g = d[d["Dataset"] == ds]
        if g.empty:
            for r in range(3):
                axes[r, col].set_visible(False)
            continue
        panels = [("Test pass rate (%)", [("Pass rate, all windows", "all windows", SLOT[0]),
                                          ("Pass rate, ADL windows", "ADL windows", SLOT[1])], "gate"),
                  ("Test fall trials kept (%)", [("Fall trials kept", None, SLOT[0])], "gate"),
                  ("Gated INT8 trial G-mean (%)", [], None)]
        for r, (ylabel, series, src) in enumerate(panels):
            ax = axes[r, col]
            if src == "gate":
                for metric, label, c in series:
                    s = g[(g["Series"] == "gate") & (g["Metric"] == metric)].sort_values("Recall target (%)")
                    ax.errorbar(s["Recall target (%)"], s["mean"], yerr=s["std"], color=c, marker="o", ms=4,
                                elinewidth=0.8, label=label)
                if len(series) > 1:
                    ax.legend(loc="best")
            else:
                for i, m in enumerate(EQUIVALENT):
                    s = g[(g["Series"] == m) & (g["Metric"] == "Gated INT8 tuned Trial G-mean")].sort_values("Recall target (%)")
                    if s.empty:
                        continue
                    ax.errorbar(s["Recall target (%)"], s["mean"], yerr=s["std"], color=SLOT[i], marker="o", ms=4,
                                elinewidth=0.8, label=short(m))
                ax.legend(loc="best", ncol=2)
            ax.axvline(100 * cfg.GATE_TRIAL_RECALL, color=AXIS, lw=0.8, zorder=0)
            if col == 0:
                ax.set_ylabel(ylabel)
            if r == 0:
                ax.set_title(title, loc="left")
        axes[2, col].set_xlabel("Training fall-trial recall target of the gate (%)")
    save(fig, "fig_gate_constraint_sweep")


def tost_forest():
    """Deployment configurations vs the TinyCNN baseline alone (FP32): mean difference and 90 % CI."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2, 3.0), sharey=True)
    drawn, labels = False, None
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        e = read(cfg.RESULTS_DIR / f"cv_{ds}" / "equivalence_tests.csv")
        if e is None:
            ax.set_visible(False)
            continue
        e = e[e["Metric"].str.contains("vs baseline", na=False) & (e["Margin"] == 1.0)
              & e["Metric"].str.startswith("Gated INT8")]
        if e.empty:
            print(f"  {ds}: no 'vs baseline' rows yet (rerun python overall.py report --dataset {ds})")
            ax.set_visible(False)
            continue
        labels = labels or list(e["A"])
        for i, a in enumerate(labels):
            r = e[e["A"] == a]
            if r.empty:
                continue
            r = r.iloc[0]
            lo, hi = (float(v) for v in re.findall(r"-?[\d.]+", r["90% CI of A - B"])[:2])
            m = r["A - B mean"]
            name = a.replace(" gate + INT8", "")
            c = color(name)
            c = INK2 if c == MUTED else c
            ax.plot([lo, hi], [i, i], color=c, lw=1.5, solid_capstyle="round")
            ax.plot(m, i, "o", color=c, mec="white", ms=5)
        for mg, alpha in ((2.0, 0.35), (1.0, 0.6), (0.5, 1.0)):
            ax.axvspan(-mg, mg, color="#f0efec", alpha=alpha, lw=0, zorder=0)
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
        ax.set_yticks(range(len(labels)), [short(a.replace(" gate + INT8", "")) for a in labels])
        ax.set_xlabel("G-mean difference vs TinyCNN alone, FP32 (points)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        drawn = True
    if drawn:
        fig.text(0.5, -0.02, "Each model with gate + INT8 vs TinyCNN alone (FP32). Bands: equivalence margins ±0.5 / ±1 / "
                 "±2 points; bars: 90 % CI (equivalent by TOST at 5 % when inside a band).", ha="center", va="top",
                 fontsize=6.5, color=INK2)
        save(fig, "fig_tost_vs_baseline")


def threshold_sweep():
    """Trial G-mean vs decision threshold (TinyTCN and TinyCNN; FP32, INT8, gated INT8)."""
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_2, 4.6), sharex=True, sharey="row")
    drawn = False
    for col, (ds, title) in enumerate(DATASETS.items()):
        d = read(FIG / f"cv_{ds}_threshold_sweep.csv")
        for row, m in enumerate([CENTER, BASELINE]):
            ax = axes[row, col]
            if d is None or d[d["model"] == m].empty:
                ax.set_visible(False)
                continue
            for i, setting in enumerate(["FP32", "INT8", "Gated INT8"]):
                s = d[(d["model"] == m) & (d["Setting"] == setting)].sort_values("Threshold")
                ax.plot(s["Threshold"], s["Trial G-mean mean"], color=SLOT[i], label=setting)
                ax.fill_between(s["Threshold"], s["Trial G-mean mean"] - s["Trial G-mean std"],
                                s["Trial G-mean mean"] + s["Trial G-mean std"], color=SLOT[i], alpha=0.1, lw=0)
            ax.axvline(cfg.THRESHOLD, color=AXIS, lw=0.8)
            ax.set_title(f"{title}: {short(m)}", loc="left")
            if col == 0:
                ax.set_ylabel("Trial G-mean (%)")
            if row == 1:
                ax.set_xlabel("Decision threshold on P(fall)")
            drawn = True
    if drawn:
        axes[0, 0].legend(loc="lower left")
        save(fig, "fig_threshold_sweep")


def size_sweep():
    """TinyTCN width / depth sweep: parameters vs G-mean."""
    d = read(FIG / "sweep_tinytcn_kfall_params_vs_gmean.csv")
    if d is None:
        return
    def family(m):
        if "blocks" in m:
            return "blocks"
        return "channels"
    d = d.assign(family=d["model"].map(family))
    base = d[d["model"] == "TinyTCN"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2, 2.8), sharey=True)
    for ax, (metric, title) in zip(axes, (("FP32 tuned Trial G-mean", "FP32, no gate"),
                                          ("Gated INT8 tuned Trial G-mean", "gate + INT8"))):
        for i, fam in enumerate(("channels", "blocks")):
            s = pd.concat([d[d["family"] == fam], base]).drop_duplicates("model").sort_values("parameters")
            ax.errorbar(s["parameters"], s[f"{metric} mean"], yerr=s[f"{metric} std"], color=SLOT[i], marker="o",
                        ms=4, elinewidth=0.8)
        labels = {"TinyTCN": "8 ch, 4 blocks", "TinyTCN (4 ch)": "4 ch", "TinyTCN (16 ch)": "16 ch",
                  "TinyTCN (2 blocks)": "2 blocks"}
        shown = d[d["model"].isin(labels)]
        ax.set_xscale("log")
        plain_log_axis(ax)
        place_labels(ax, list(zip(shown["parameters"], shown[f"{metric} mean"])), [labels[m] for m in shown["model"]])
        ax.set_xlabel("Parameters (log scale)")
        ax.set_title(f"KFall, TinyTCN size sweep: {title}", loc="left")
    axes[0].set_ylabel("Trial G-mean (%)")
    figure_legend(fig, [Line2D([], [], marker="o", color=SLOT[0], label="channels 4-16 (4 blocks)"),
                        Line2D([], [], marker="o", color=SLOT[1], label="blocks 2-4 (8 channels)")])
    save(fig, "fig_tinytcn_size_sweep")


def pruning():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2, 2.8), sharey=True)
    drawn = False
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        d = read(FIG / f"pruning_{ds}.csv")
        if d is None:
            ax.set_visible(False)
            continue
        for m in d["model"].unique():
            for i, metric in enumerate(["FP32 tuned Trial G-mean", "INT8 tuned Trial G-mean", "Gated INT8 tuned Trial G-mean"]):
                s = d[(d["model"] == m) & (d["metric"] == metric) & d["sparsity"].str.endswith("%")]
                x = s["sparsity"].str.rstrip("%").astype(float)
                ax.errorbar(x, s["mean"], yerr=s["std"], color=SLOT[i], marker="o", ms=4, elinewidth=0.8,
                            label=metric.replace(" Trial G-mean", "").replace(" tuned", ""))
                ref = d[(d["model"] == m) & (d["metric"] == metric) & ~d["sparsity"].str.endswith("%")]
                if not ref.empty:
                    ax.axhline(ref["mean"].iloc[0], color=SLOT[i], lw=0.8, alpha=0.5)
            ax.set_title(f"{title}: {short(m)} (lines: main run)", loc="left")
        ax.set_xlabel("Sparsity of the kernels (%)")
        drawn = True
    if drawn:
        axes[0].set_ylabel("Trial G-mean (%)")
        axes[0].legend(loc="lower left")
        save(fig, "fig_pruning")


def quantization_heatmap():
    """G-mean change of each quantization variant vs FP32, per model (tuned grid and full grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_2, 5.2))
    drawn = False
    for col, (ds, title) in enumerate(DATASETS.items()):
        d = read(FIG / f"quantization_{ds}.csv")
        for row, metric in enumerate(["Trial G-mean (tuned)", "Trial G-mean (grid 0.50-0.99)"]):
            ax = axes[row, col]
            if d is None:
                ax.set_visible(False)
                continue
            g = d[d["metric"] == metric].pivot_table(index="model", columns="variant", values="mean")
            if "FP32" not in g:
                ax.set_visible(False)
                continue
            delta = g.drop(columns="FP32").sub(g["FP32"], axis=0)
            names = {"FP16 weights": "FP16", "Weight-only INT8 (TFLite)": "W8\nTFLite", "Weight-only INT8 (simulated)":
                     "W8\nsim.", "Full INT8": "INT8", "Full INT8, balanced calibration": "INT8\nbal. cal.",
                     "INT16 activations, INT8 weights": "16x8"}
            cols = [c for c in names if c in delta]
            delta = delta[cols]
            lim = max(1.0, float(np.nanmax(np.abs(delta.values))))
            im = ax.imshow(delta.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
            for (i, j), v in np.ndenumerate(delta.values):
                ax.text(j, i, f"{v:+.2f}" if np.isfinite(v) else "n/a", ha="center", va="center", fontsize=6.5,
                        color="white" if np.isfinite(v) and abs(v) > 0.6 * lim else INK)
            ax.set_xticks(range(len(cols)), [names[c] for c in cols], fontsize=6.5)
            ax.set_yticks(range(len(delta.index)), [short(m) for m in delta.index])
            ax.grid(False)
            ax.set_title(f"{title}: threshold on the {'full grid (main)' if 'tuned' in metric else 'capped grid 0.50-0.99 (protocol v1)'}",
                         loc="left")
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="G-mean change vs FP32 (points)")
            drawn = True
    if drawn:
        save(fig, "fig_quantization_variants")


def qat_bars():
    """Full-INT8 G-mean of the recurrent models: PTQ, the same fine-tuning + PTQ, and QAT (dots; line: FP32)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2, 2.6), sharex=True)
    drawn = False
    arms = [("ptq", "PTQ (main run)"), ("finetune", "same fine-tuning, then PTQ"), ("qat", "QAT")]
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        d = read(cfg.RESULTS_DIR / "qat" / ds / "all_runs.csv")
        if d is None:
            ax.set_visible(False)
            continue
        names = [m for m in ("cnnlstm", "musci2021") if m in set(d["model"])]
        for j, m in enumerate(names):
            f = d[(d["model"] == m) & (d["arm"] == "ptq")]["float model Trial G-mean (tuned)"]
            ax.vlines(f.mean(), j - 0.35, j + 0.35, color=INK2, lw=0.8)
            for i, (arm, label) in enumerate(arms):
                v = d[(d["model"] == m) & (d["arm"] == arm)]["INT8 Trial G-mean (tuned)"]
                ax.errorbar(v.mean(), j + (1 - i) * 0.2, xerr=v.std(ddof=1), fmt="o", color=SLOT[i], ecolor=SLOT[i],
                            elinewidth=0.8, ms=5, mec="white")
        ax.set_yticks(range(len(names)), [{"cnnlstm": "CNN-LSTM", "musci2021": "Musci LSTM"}[m] for m in names])
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_xlabel("Full-INT8 trial G-mean (%)")
        ax.set_title(f"{title} (vertical line: FP32)", loc="left")
        ax.grid(axis="y", visible=False)
        drawn = True
    if drawn:
        figure_legend(fig, [Line2D([], [], marker="o", ls="", color=SLOT[i], label=label)
                            for i, (_, label) in enumerate(arms)])
        save(fig, "fig_qat")


def misses_and_false_alarms(model=CENTER):
    m = read(cfg.RESULTS_DIR / "analysis" / "misses_by_fall_type.csv")
    f = read(cfg.RESULTS_DIR / "analysis" / "false_alarms_by_adl_type.csv")
    if m is None or f is None:
        return
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_2, 6.6))
    for col, (ds, title) in enumerate(DATASETS.items()):
        a = m[(m["Dataset"] == ds) & (m["Model"] == model)].sort_values("Fall type", ascending=False)
        b = f[(f["Dataset"] == ds) & (f["Model"] == model)].sort_values("ADL type", ascending=False)
        if a.empty or b.empty:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        ax = axes[0, col]
        y = np.arange(len(a))
        ax.barh(y, a["Missed by gate (%)"], height=0.6, color=SLOT[0])
        ax.barh(y, a["Missed by model (%)"], left=a["Missed by gate (%)"], height=0.6, color=SLOT[1],
                edgecolor="white", linewidth=1)
        ax.set_yticks(y, a["Fall type"])
        ax.set_xlabel("Missed fall trials (%)")
        ax.set_title(f"{title}: {short(model)}, gate + INT8, missed falls", loc="left")
        ax.grid(axis="y", visible=False)
        ax.legend(handles=[Patch(color=SLOT[0], label="missed by the gate"),
                           Patch(color=SLOT[1], label="passed the gate, missed by the model")], loc="best")
        ax = axes[1, col]
        y = np.arange(len(b))
        ax.barh(y + 0.18, b["False alarms, ungated FP32 tuned (%)"], height=0.34, color=SLOT[0])
        ax.barh(y - 0.18, b["False alarms, main setting (%)"], height=0.34, color=SLOT[1])
        ax.set_yticks(y, b["ADL type"])
        ax.set_xlabel("ADL trials with a false alarm (%)")
        ax.set_title(f"{title}: false alarms by ADL type", loc="left")
        ax.grid(axis="y", visible=False)
        ax.legend(handles=[Patch(color=SLOT[0], label="model alone, FP32"),
                           Patch(color=SLOT[1], label="gate + model, INT8")], loc="best")
    save(fig, f"fig_misses_false_alarms_{model.lower()}")


def gate_ablation():
    d = read(FIG / "gate_ablation.csv")
    if d is None:
        return
    order = ["acc", "gyr", "tilt", "acc + gyr", "acc + tilt", "gyr + tilt", "acc + gyr + tilt"]
    panels = [("Pass rate, ADL windows", "Test ADL windows passing (%)"), ("Fall trials kept", "Test fall trials kept (%)"),
              (f"Gated INT8 tuned Trial G-mean", f"{CENTER} gate + INT8 G-mean (%)")]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_2, 4.8), sharey=True)
    for row, (ds, title) in enumerate(DATASETS.items()):
        g = d[d["Dataset"] == ds]
        if g.empty:
            for ax in axes[row]:
                ax.set_visible(False)
            continue
        for col, (metric, xlabel) in enumerate(panels):
            ax = axes[row, col]
            series = CENTER if metric.startswith("Gated") else "gate"
            if g[(g["Series"] == series) & (g["Metric"] == metric)].empty:
                ax.set_visible(False)
                continue
            y = np.arange(len(order))
            for i, k in enumerate((1, 2)):
                s = g[(g["Series"] == series) & (g["Metric"] == metric) & (g["Windows in a row"] == k)].set_index("Conditions")
                s = s.reindex(order)
                ax.errorbar(s["mean"], y + (0.15 if k == 1 else -0.15), xerr=s["std"], fmt="o", color=SLOT[i],
                            ecolor=SLOT[i], elinewidth=0.8, ms=4, mec="white")
            ax.set_yticks(y, order)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_title(title, loc="left")
    figure_legend(fig, [Line2D([], [], marker="o", ls="", color=SLOT[i], label=f"{k} window{'s' if k > 1 else ''} in a row")
                        for i, k in enumerate((1, 2))])
    save(fig, "fig_gate_ablation")


def early_stopping():
    """Per model: the plan's rule, early stopping within 200 epochs, and the 200-epoch main runs (thresholds tuned on
    the training subjects in all three), gate + INT8. Ordered by training budget: one blue ramp, light to dark."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2, 2.8), sharey=True)
    drawn, order = False, None
    ramp = ["#86b6ef", "#3987e5", "#184f95"]                  # sequential blue steps 250 / 400 / 600 (ordinal)
    metric = "Gated INT8 tuned Trial G-mean"
    series = [(f"{metric}, plan rule (thresholds: training subjects)",
               f"plan rule (<= {cfg.ES_PLAN_EPOCHS} epochs, patience {cfg.ES_PLAN_PATIENCE})"),
              (f"{metric}, early stopping (thresholds: training subjects)",
               f"early stopping (<= {cfg.EPOCHS} epochs, patience {cfg.ES_PATIENCE})"),
              (f"{metric}, {cfg.EPOCHS} epochs (main runs)", f"{cfg.EPOCHS} epochs, all training subjects")]
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        s = read(cfg.RESULTS_DIR / f"es_{ds}" / "comparison_summary.csv")
        if s is None:
            ax.set_visible(False)
            continue
        s = s.set_index("Model")
        order = order or list(s.index)
        for i, m in enumerate(order):
            if m not in s.index:
                continue
            vals = [mean_std(s.loc[m, c]) if c in s and s.loc[m, c] != "-" else None for c, _ in series]
            xs = [v[0] for v in vals if v]
            ax.plot([min(xs), max(xs)], [i, i], color=GRID, lw=2, zorder=1)
            for k, v in enumerate(vals):
                if v:
                    ax.errorbar(v[0], i, xerr=v[1], fmt="o", color=ramp[k], ecolor=ramp[k], elinewidth=0.8, ms=5,
                                mec="white", zorder=3 + k)
        ax.set_yticks(range(len(order)), [short(m) for m in order])
        ax.set_xlabel("Gate + INT8 trial G-mean (%)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        drawn = True
    if drawn:
        figure_legend(fig, [Line2D([], [], marker="o", ls="", color=ramp[k], label=label)
                            for k, (_, label) in enumerate(series)], ncol=1)
        save(fig, "fig_early_stopping")


def training_curves(ds="kfall"):
    d = read(FIG / f"cv_{ds}_training_curves.csv")
    if d is None:
        return
    names = list(d["model"].unique())
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(WIDTH_2, 1.7 * rows), sharex=True, squeeze=False)
    for ax, m in zip(axes.ravel(), names):                      # one series per panel: one color, the title names it
        s = d[d["model"] == m]
        ax.plot(s["epoch"], s["loss mean"], color=SLOT[0])
        ax.fill_between(s["epoch"], s["loss mean"] - s["loss std"], s["loss mean"] + s["loss std"], color=SLOT[0],
                        alpha=0.1, lw=0)
        ax.set_title(short(m), loc="left")
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for j in range(cols):                                       # x labels on the lowest panel of each column
        shown = [axes[i, j] for i in range(rows) if axes[i, j].get_visible()]
        if shown:
            shown[-1].xaxis.set_tick_params(labelbottom=True)
            shown[-1].set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Training loss")
    save(fig, f"fig_training_curves_{ds}")


def roc(model=CENTER):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2, 2.8), sharey=True)
    drawn = False
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        d = read(FIG / f"cv_{ds}_roc_trial.csv")
        if d is None or d[d["model"] == model].empty:
            ax.set_visible(False)
            continue
        for i, setting in enumerate(["FP32", "INT8", "Gated INT8"]):
            s = d[(d["model"] == model) & (d["Setting"] == setting)].sort_values("FPR")
            ax.plot(100 * s["FPR"], 100 * s["TPR mean"], color=SLOT[i],
                    label=f"{setting} (AUC {s['AUC mean'].iloc[0]:.4f})")
        ax.set_xlim(0, 10)
        ax.set_xlabel("False alarm rate, ADL trials (%)")
        ax.set_title(f"{title}: {short(model)}, trial-level ROC", loc="left")
        ax.legend(loc="lower right")
        drawn = True
    if drawn:
        axes[0].set_ylabel("Fall trials detected (%)")
        save(fig, f"fig_roc_{model.lower()}")


def architecture():
    """System diagram of the gated detector: IMU -> sliding windows -> gate -> TinyTCN (full INT8) -> decision, with
    the TinyTCN layers and one residual block expanded. Thresholds, parameter and operation counts come from the MCU
    deployment record (fixed KFall split) and the ops table; the layer structure is that of models/tinytcn.py."""
    import json
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    dep_path = cfg.RESULTS_DIR / "mcu" / "tinytcn" / "deployment.json"
    dep = json.loads(dep_path.read_text(encoding="utf-8")) if dep_path.exists() else {}
    acc, gyr, tilt = dep.get("gate", {}).get("thresholds (acc g, gyr deg/s, tilt deg; -inf = condition not used)",
                                             (1.75, 100.0, 80.0))
    theta = float(dep.get("decision threshold, gated INT8", 214 / 256))
    params, macs = dep.get("parameters", 458), dep.get("MACs / window", 20016)
    flash = dep.get(".tflite INT8 (bytes)", 16592) / 1024
    ram = dep.get("peak activation memory INT8 (bytes, estimate)", 1756) / 1024
    scale = dep.get("input", {}).get("scale", 0.0684)
    gate_ops = dep.get("gate", {}).get("ops / window", {"recomputing the window": 852, "streaming": 292})
    ops = read(cfg.RESULTS_DIR / "analysis" / "ops_per_window.csv")
    rate = {"kfall": 9.18, "sisfall": 14.45}
    if ops is not None:
        for ds in rate:
            r = ops[(ops["Dataset"] == ds) & (ops["Windows"] == "all windows") & (ops["Model"] == CENTER)]
            if len(r):
                rate[ds] = float(r.iloc[0]["Pass rate (%)"])

    tint = {SLOT[0]: "#e6eefb", SLOT[1]: "#fdebe3", SLOT[2]: "#e1f5ed", MUTED: "#f3f2ee"}
    fig, ax = plt.subplots(figsize=(WIDTH_2, WIDTH_2 * 0.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, lines, edge=MUTED, title=None, fs=5.9, sub=None, lw=0.9, ls="-", fill=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                    fc=tint[edge] if fill else "none", ec=edge, lw=lw, ls=ls))
        if title:
            ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=fs + 0.5, fontweight="bold",
                    color=INK)
            ax.text(x + w / 2, y + h - 3.9, lines, ha="center", va="top", fontsize=fs, color=INK2, linespacing=1.25)
        elif lines:
            ax.text(x + w / 2, y + h / 2, lines, ha="center", va="center", fontsize=fs, color=INK2, linespacing=1.25)
        if sub:
            ax.text(x + w / 2, y - 0.9, sub, ha="center", va="top", fontsize=5.3, color=MUTED, linespacing=1.2)

    def arrow(p, q, text=None, above=True, color=INK2, fs=5.6):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.8, color=color,
                                     shrinkA=0, shrinkB=0))
        if text:
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            if p[1] == q[1]:                                   # horizontal arrow: label above or below
                ax.text(mx, my + (0.9 if above else -0.9), text, ha="center", va="bottom" if above else "top",
                        fontsize=fs, color=color, linespacing=1.15)
            else:                                              # vertical arrow: label to the right
                ax.text(mx + 0.9, my, text, ha="left", va="center", fontsize=fs, color=color)

    # ---- stage row: one window every 0.1 s
    y0, h0 = 44, 12
    box(0.5, y0, 11, h0, "6 axes, 100 Hz\nacc (g)\ngyr (°/s)", title="Waist IMU")
    box(13.5, y0, 12.5, h0, "0.5 s = 50 samples\nstep 0.1 s (10)\n[50 × 6] per step", title="Sliding window")
    box(28, y0, 27.5, h0,
        f"per sample: |a|, |ω|, tilt = arccos(−a$_y$ / |a|)\nwindow maximum of each, then\n"
        f"max |a| > {acc:g} g and max |ω| > {gyr:g} °/s and\nmax tilt > {tilt:g}°  →  pass\n"
        f"{gate_ops['recomputing the window']} ops / window ({gate_ops['streaming']} streaming)",
        edge=SLOT[1], title="Stage 1: gate, runs on every window")
    box(64, y0, 18, h0, f"{params} parameters, {macs:,} MACs\n{flash:.1f} KB flash, {ram:.1f} KB RAM\n"
        "runs only on passed windows\nlayers expanded below", edge=SLOT[0], title="Stage 2: TinyTCN, full INT8")
    box(86.5, y0, 13, h0, f"P(fall) ≥ {theta:.3f}\n({round(theta * 256)}/256, int8)\nthreshold tuned on\n"
        "training subjects", edge=SLOT[2], title="Decision")
    ym = y0 + h0 / 2
    arrow((11.5, ym), (13.5, ym))
    arrow((26, ym), (28, ym))
    arrow((55.5, ym), (64, ym), "pass", color=SLOT[1])
    ax.text(59.75, ym - 0.9, f"KFall {rate['kfall']:.0f} %\nSisFall {rate['sisfall']:.0f} %\nof windows",
            ha="center", va="top", fontsize=5.3, color=SLOT[1], linespacing=1.15)
    arrow((82, ym), (86.5, ym), "P(fall)", color=SLOT[0])
    # branches below the stage row
    arrow((41.75, y0), (41.75, 40.6), "fail", color=SLOT[1])
    box(28.5, 35.5, 26.5, 5.1, f"fail (KFall {100 - rate['kfall']:.0f} %, SisFall {100 - rate['sisfall']:.0f} % of "
        "windows):\nno alarm, model skipped", edge=SLOT[1], fs=5.6, fill=False)
    arrow((93, y0), (93, 40.6), color=SLOT[2])
    box(76, 35.5, 23.5, 5.1, "≥ θ: fall alarm, otherwise no alarm\ntrial = fall if any window fires",
        edge=SLOT[2], fs=5.6, fill=False)

    # ---- TinyTCN layers
    box(0.5, 18, 99, 15.2, None, edge=SLOT[0], ls=(0, (3, 2)), fill=False, lw=0.7)
    ax.text(1.8, 32.4, "Stage 2 expanded: TinyTCN layers", ha="left", va="top", fontsize=6.4, fontweight="bold",
            color=SLOT[0])
    ax.text(98.2, 32.4, "[rows × channels]; causal convolutions, kernel 3, 8 channels", ha="right", va="top",
            fontsize=5.6, color=INK2)
    layers = [("Input window\nz-score per axis\nint8, scale %.3f\n[50 × 6]" % scale, "no params"),
              ("Conv 1×1\n6 → 8, ReLU\n\n[50 × 8]", "56 params\n2,400 MACs")]
    layers += [(f"Residual block\ndilation d = {d}\n(detail below)\n[50 × 8]", "96 params\n4,400 MACs") for d in (1, 2, 4, 8)]
    layers += [("Global average\npooling over\nthe 50 rows\n[8]", "no params"),
               ("Dense 8 → 2\nsoftmax\n\n[2]: P(fall)", "18 params\n16 MACs")]
    w, gap, x = 10.4, 2.0, 1.5
    yb, hb = 22.5, 7.6
    for i, (text, sub) in enumerate(layers):
        edge = SLOT[0] if 1 <= i <= 7 else MUTED
        box(x, yb, w, hb, text, edge=edge, fs=5.5, sub=sub)
        if i < len(layers) - 1:
            arrow((x + w, yb + hb / 2), (x + w + gap, yb + hb / 2))
        x += w + gap

    # ---- one residual block
    box(0.5, 1, 67.5, 15.2, None, edge=MUTED, ls=(0, (3, 2)), fill=False, lw=0.7)
    ax.text(1.8, 15.4, "Residual block (dilation d, kernel 3): 96 parameters, 4,400 MACs", ha="left", va="top",
            fontsize=6.4, fontweight="bold", color=INK)
    yc, hc = 4.3, 5.8
    blocks = [(2, 8, "x\n[50 × 8]", MUTED), (12.5, 14, "causal zero-pad\n2d rows at the start", SLOT[0]),
              (29, 13.5, "depthwise conv\nk = 3, dilation d\nno bias", SLOT[0]), (45, 11, "conv 1×1\n8 → 8, ReLU", SLOT[0]),
              (58.5, 4.5, "+", MUTED)]
    for i, (bx, bw, text, edge) in enumerate(blocks):
        box(bx, yc, bw, hc, text, edge=edge, fs=5.5)
        if i < len(blocks) - 1:
            arrow((bx + bw, yc + hc / 2), (blocks[i + 1][0], yc + hc / 2))
    arrow((63, yc + hc / 2), (66.5, yc + hc / 2))
    ax.text(64.75, yc + hc / 2 + 0.9, "[50 × 8]", ha="center", va="bottom", fontsize=5.3, color=INK2)
    ys = yc + hc + 2.2                                          # skip connection over the top
    ax.plot([6, 6, 60.75], [yc + hc, ys, ys], color=INK2, lw=0.8)
    arrow((60.75, ys), (60.75, yc + hc))
    ax.text(33, ys + 0.5, "skip", ha="center", va="bottom", fontsize=5.3, color=INK2)
    ax.text(1.8, 2.0, "receptive field after the four blocks: 1 + 2·(1 + 2 + 4 + 8) = 31 samples = 0.31 s",
            ha="left", va="bottom", fontsize=5.6, color=INK2)

    # ---- colour key
    box(70, 1, 29.5, 15.2, None, edge=MUTED, ls=(0, (3, 2)), fill=False, lw=0.7)
    ax.text(71.3, 15.4, "Colour key", ha="left", va="top", fontsize=6.4, fontweight="bold", color=INK)
    key = [(SLOT[1], "fixed rule: thresholds tuned on the\ntraining subjects (100 % of training\nfalls kept)"),
           (SLOT[0], "learned: focal loss, 200 epochs, then\npost-training full-INT8 quantisation"),
           (SLOT[2], "decision threshold tuned on the\ntraining subjects (grid k/256)")]
    for (c, text), yk in zip(key, (12.6, 8.2, 4.2)):
        ax.add_patch(FancyBboxPatch((71.3, yk - 0.6), 2.2, 1.5, boxstyle="round,pad=0,rounding_size=0.3",
                                    fc=tint[c], ec=c, lw=0.8))
        ax.text(74.4, yk + 0.2, text, ha="left", va="top", fontsize=5.4, color=INK2, linespacing=1.2)
    save(fig, "fig_architecture")


ALL = [architecture, pareto, int8_dumbbell, gate_sweep, tost_forest, threshold_sweep, size_sweep, pruning,
       quantization_heatmap, qat_bars, misses_and_false_alarms, gate_ablation, early_stopping, training_curves, roc]


def run():
    setup()
    print(f"Figures -> {OUT}")
    for f in ALL:
        print(f"{f.__name__}:")
        try:
            f()
        except Exception as e:                                   # noqa: BLE001 - one broken figure must not stop the rest
            plt.close("all")
            print(f"  failed: {type(e).__name__}: {e}")
