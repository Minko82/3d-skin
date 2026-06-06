"""
plot_drift_raw_grid.py
----------------------
Raw-data reproducibility grid. Lays out a panel for every (node × skin)
combination and overlays that skin's individual drift tests on top of one
another — so you can see, at a glance, that:

    • the 3 tests within each panel overlap        → test-to-test repeatability
    • panels across a row look alike               → skin-to-skin consistency

Folder layout expected (one subfolder per skin, drift CSVs inside):

    drift_test/artifacts/
        skin_1/  drift_*.csv  drift_*.csv  drift_*.csv
        skin_2/  drift_*.csv  ...
        skin_3/  drift_*.csv  ...

Usage
-----
    python drift_test/plot_drift_raw_grid.py

    python drift_test/plot_drift_raw_grid.py --delta        # zero each test to its own baseline
    python drift_test/plot_drift_raw_grid.py --smooth-min 0 # truly raw, no smoothing
    python drift_test/plot_drift_raw_grid.py --output my_grid.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Style  (matches rest of project)
# ─────────────────────────────────────────────────────────────────────────────

CHANNELS    = ["sensor_1", "sensor_5", "sensor_7"]
NODE_LABELS = ["Node 1", "Node 5", "Node 7"]

# One colour per test (Wong color-blind-safe palette)
TEST_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

TRACE_LW = 1.1

FONTSIZE_AXIS_LABEL = 14
FONTSIZE_TICK_LABEL = 10
FONTSIZE_COL_TITLE  = 15
FONTSIZE_LEGEND     = 12
FONTSIZE_PANEL_TAG  = 13

RC_BASE = {
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "font.weight":          "bold",
    "font.size":            FONTSIZE_TICK_LABEL,
    "axes.labelsize":       FONTSIZE_AXIS_LABEL,
    "axes.labelweight":     "bold",
    "xtick.labelsize":      FONTSIZE_TICK_LABEL,
    "ytick.labelsize":      FONTSIZE_TICK_LABEL,
    "legend.frameon":       False,
    "axes.linewidth":       1.2,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "xtick.major.width":    1.2,
    "ytick.major.width":    1.2,
    "xtick.minor.width":    0.7,
    "ytick.minor.width":    0.7,
    "xtick.major.size":     4.0,
    "ytick.major.size":     4.0,
    "xtick.minor.size":     2.0,
    "ytick.minor.size":     2.0,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "lines.solid_capstyle": "round",
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.03,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
}

N_PLOT = 4000  # points drawn per trace (keeps the vector PDF light)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [c for c in ["elapsed_ms"] + CHANNELS if c not in df.columns]
    if missing:
        print(f"  Warning: missing columns {missing} in {path.name} — skipping.",
              file=sys.stderr)
        return pd.DataFrame()
    df = df[["elapsed_ms"] + CHANNELS].apply(pd.to_numeric, errors="coerce").dropna()
    df["elapsed_h"] = df["elapsed_ms"] / 3_600_000.0
    return df.reset_index(drop=True)


def smooth_series(values: np.ndarray, median_dt_ms: float, smooth_min: float) -> np.ndarray:
    if smooth_min <= 0:
        return values
    samples_per_min = 60_000.0 / max(median_dt_ms, 1.0)
    window = max(1, int(smooth_min * samples_per_min))
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().values


def prettify(name: str) -> str:
    """skin_1 -> Skin 1"""
    return name.replace("_", " ").title()


def drift_stats(t_h: np.ndarray, values: np.ndarray, baseline_min: float) -> tuple[float, float]:
    """
    Drift relative to an initial baseline (mean of the first `baseline_min` minutes).

    Returns:
        slope_per_h — least-squares drift rate in ADC counts / hour
        net_drift   — (end level) − (baseline), in ADC counts
    """
    t_h = np.asarray(t_h, dtype=float)
    v   = np.asarray(values, dtype=float)
    good = np.isfinite(t_h) & np.isfinite(v)
    t_h, v = t_h[good], v[good]
    if len(v) < 2:
        return 0.0, 0.0

    win_h = baseline_min / 60.0
    base_mask = t_h <= (t_h[0] + win_h)
    baseline  = float(v[base_mask].mean()) if base_mask.any() else float(v[0])

    # Slope is offset-invariant; fit the raw trace against time in hours
    slope_per_h = float(np.polyfit(t_h, v, 1)[0])

    # Net drift: end level (mean of final window) minus baseline
    end_mask = t_h >= (t_h[-1] - win_h)
    end_val  = float(v[end_mask].mean()) if end_mask.any() else float(v[-1])
    net_drift = end_val - baseline

    return slope_per_h, net_drift


def fmt_signed(x: float, decimals: int = 1) -> str:
    return f"{'+' if x >= 0 else ''}{x:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_grid(
    skin_names: list[str],
    skin_runs: list[list[pd.DataFrame]],
    smooth_min: float,
    delta: bool,
    baseline_min: float,
    output_path: Path,
    cell_w: float = 3.0,
    cell_h: float = 2.3,
) -> None:

    plt.rcParams.update(RC_BASE)

    n_rows = len(CHANNELS)       # nodes
    n_cols = len(skin_names)     # skins
    max_tests = max(len(r) for r in skin_runs)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols, cell_h * n_rows),
        sharex=True, sharey=True,          # every panel shares one identical y-axis
        squeeze=False,
    )

    max_h = max(float(df["elapsed_h"].max())
                for runs in skin_runs for df in runs)
    y_label = "Δ ADC Count" if delta else "ADC Count"

    # Pre-compute every trace + per-test drift stats, and track global y range
    grid = {}  # (ri, ci) -> {"traces":[(t,v,color)], "slopes":[], "nets":[]}
    g_lo, g_hi = np.inf, -np.inf
    for ri, ch in enumerate(CHANNELS):
        for ci, runs in enumerate(skin_runs):
            traces, slopes, nets = [], [], []
            for ti, df in enumerate(runs):
                color = TEST_COLORS[ti % len(TEST_COLORS)]
                median_dt = float(df["elapsed_ms"].diff().median())
                t_h = df["elapsed_h"].values
                raw = df[ch].values.astype(float)

                if delta:
                    mask = df["elapsed_h"] <= baseline_min / 60.0
                    base = float(df.loc[mask, ch].mean()) if mask.any() else 0.0
                    raw = raw - base

                smoothed = smooth_series(raw, median_dt, smooth_min)

                slope, net = drift_stats(t_h, smoothed, baseline_min)
                slopes.append(slope)
                nets.append(net)

                # Decimate to N_PLOT evenly spaced samples for a light PDF
                if len(t_h) > N_PLOT:
                    idx = np.linspace(0, len(t_h) - 1, N_PLOT).astype(int)
                    t_p, v_p = t_h[idx], smoothed[idx]
                else:
                    t_p, v_p = t_h, smoothed

                finite = v_p[np.isfinite(v_p)]
                if finite.size:
                    g_lo = min(g_lo, float(finite.min()))
                    g_hi = max(g_hi, float(finite.max()))
                traces.append((t_p, v_p, color))
            grid[(ri, ci)] = {"traces": traces, "slopes": slopes, "nets": nets}

    span = (g_hi - g_lo) or 1.0
    y_lo = g_lo - 0.10 * span
    y_hi = g_hi + 0.22 * span  # extra headroom for the per-cell stats label

    for ri, (ch, node_label) in enumerate(zip(CHANNELS, NODE_LABELS)):
        for ci, skin_name in enumerate(skin_names):
            ax = axes[ri][ci]
            cell = grid[(ri, ci)]

            for (t_p, v_p, color) in cell["traces"]:
                ax.plot(t_p, v_p, color=color, lw=TRACE_LW, zorder=3, alpha=0.9)

            if delta:
                ax.axhline(0, color="#AAAAAA", lw=0.9, ls="--", zorder=1)

            ax.set_xlim(0.0, max_h)
            ax.set_ylim(y_lo, y_hi)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
            ax.yaxis.grid(True, color="#ECECEC", lw=0.7, zorder=0)
            ax.set_axisbelow(True)

            # Per-cell drift summary: mean over this cell's tests (slope + net)
            mslope = float(np.mean(cell["slopes"])) if cell["slopes"] else 0.0
            mnet   = float(np.mean(cell["nets"]))   if cell["nets"]   else 0.0
            ax.text(0.04, 0.95,
                    f"{fmt_signed(mslope, 1)} ADC/h\n{fmt_signed(mnet, 0)} ADC net",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=9, color="#222222", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#BBBBBB", alpha=0.85, linewidth=0.6))

            # Column header (skin name) on top row
            if ri == 0:
                ax.set_title(prettify(skin_name),
                             fontsize=FONTSIZE_COL_TITLE, fontweight="bold", pad=8)

            # Row label (node) on left column
            if ci == 0:
                ax.set_ylabel(f"{y_label}\n{node_label}",
                              rotation=0, ha="right", va="center",
                              fontsize=FONTSIZE_AXIS_LABEL, labelpad=14)

            # X label on bottom row
            if ri == n_rows - 1:
                ax.set_xlabel("Time (hours)", labelpad=3)

    # Single shared legend (one entry per test) in the top-right panel
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=TEST_COLORS[ti % len(TEST_COLORS)],
               lw=TRACE_LW + 0.6, label=f"Test {ti + 1}")
        for ti in range(max_tests)
    ]
    axes[0][n_cols - 1].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=FONTSIZE_LEGEND,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        borderaxespad=0.3,
    )

    caption = (
        f"Per-panel drift vs. initial baseline (mean of first {baseline_min:.0f} min), "
        f"averaged over the {max_tests} tests.   "
        "ADC/h = drift rate (least-squares slope).   "
        "ADC net = end level − baseline (total drift)."
    )
    fig.text(0.5, 0.012, caption, ha="center", va="bottom",
             fontsize=8.5, color="#444444", fontstyle="italic")

    fig.tight_layout(pad=0.5, h_pad=0.6, w_pad=0.6, rect=[0.0, 0.035, 1.0, 1.0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Raw drift overlay grid: rows=nodes, cols=skins, tests overlaid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--smooth-min", type=float, default=2.0, metavar="MIN",
        help="Light rolling-mean window for readability (default: 2; use 0 for truly raw)",
    )
    p.add_argument(
        "--delta", action="store_true",
        help="Subtract each test's own initial baseline — y=0 at start of each test",
    )
    p.add_argument(
        "--baseline-min", type=float, default=5.0, metavar="MIN",
        help="Minutes used to compute each test's baseline for --delta (default: 5)",
    )
    p.add_argument(
        "--data-dir", "-d", default=None,
        help="Folder whose subfolders are skins (default: drift_test/artifacts/). "
             "e.g. --data-dir artifacts/robot",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output PDF path (default: drift_test/figures/<data-dir-name>_raw_grid[...]_figure.pdf)",
    )
    p.add_argument(
        "--cell-width", type=float, default=3.0,
        help="Width per panel in inches (default: 3.0)",
    )
    p.add_argument(
        "--cell-height", type=float, default=2.3,
        help="Height per panel in inches (default: 2.3)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).parent  # drift_test/

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = data_dir if data_dir.exists() else here / args.data_dir
    else:
        data_dir = here / "artifacts"

    if not data_dir.exists():
        print(f"Error: folder not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    skin_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and any(d.glob("drift_*.csv"))
    ])
    if not skin_dirs:
        print(f"Error: no skin subfolders with drift CSVs found in {data_dir}",
              file=sys.stderr)
        sys.exit(1)

    label = data_dir.name if data_dir.name != "artifacts" else "drift"
    print(f"Found {len(skin_dirs)} skin build(s) in {data_dir}/:")
    skin_names, skin_runs = [], []
    for skin_dir in skin_dirs:
        runs = []
        for path in sorted(skin_dir.glob("drift_*.csv")):
            df = load_csv(path)
            if not df.empty:
                runs.append(df)
        if runs:
            skin_names.append(skin_dir.name)
            skin_runs.append(runs)
            print(f"  {prettify(skin_dir.name):<12} {len(runs)} test(s)")

    if not skin_runs:
        print("Error: no valid CSVs could be loaded.", file=sys.stderr)
        sys.exit(1)

    variant = "_delta" if args.delta else ""
    out_path = (
        Path(args.output).with_suffix(".pdf")
        if args.output
        else here / "figures" / f"{label}_raw_grid{variant}_figure.pdf"
    )

    print("\nGenerating raw overlay grid...")
    make_grid(
        skin_names=skin_names,
        skin_runs=skin_runs,
        smooth_min=args.smooth_min,
        delta=args.delta,
        baseline_min=args.baseline_min,
        output_path=out_path,
        cell_w=args.cell_width,
        cell_h=args.cell_height,
    )


if __name__ == "__main__":
    main()
