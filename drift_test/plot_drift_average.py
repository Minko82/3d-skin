"""
plot_drift_average.py
---------------------
Loads every drift CSV in drift_test/artifacts/, aligns them all to t=0,
computes the mean ± 1 SD across runs at each time point, and saves a single
figure to drift_test/figures/drift_average_figure.pdf.

Runs that ended early only contribute data up to their own duration — the
average at each time point is computed from however many runs are still active
there (nanmean / nanstd).

Usage
-----
    python drift_test/plot_drift_average.py

    python drift_test/plot_drift_average.py --smooth-min 10
    python drift_test/plot_drift_average.py --delta        # Δ from each run's own baseline
    python drift_test/plot_drift_average.py --no-traces    # hide individual run lines
    python drift_test/plot_drift_average.py --output my_avg.pdf
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
# Style  (matches plot_drift.py)
# ─────────────────────────────────────────────────────────────────────────────

CHANNELS    = ["sensor_1", "sensor_5", "sensor_7"]
NODE_LABELS = ["Node 1", "Node 5", "Node 7"]
PANEL_TAGS  = ["(a)", "(b)", "(c)"]
SENSOR_COLORS = ["#0072B2", "#D55E00", "#009E73"]

MEAN_LW    = 2.2
BAND_ALPHA = 0.25
TRACE_LW   = 0.6
TRACE_ALPHA = 0.18

FONTSIZE_AXIS_LABEL = 14
FONTSIZE_TICK_LABEL = 12
FONTSIZE_PANEL_TAG  = 16
FONTSIZE_ANNOT      = 11

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
    "xtick.major.size":     5.0,
    "ytick.major.size":     5.0,
    "xtick.minor.size":     3.0,
    "ytick.minor.size":     3.0,
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

N_GRID = 1000  # interpolation grid points


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
    samples_per_min = 60_000.0 / max(median_dt_ms, 1.0)
    window = max(1, int(smooth_min * samples_per_min))
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().values


# ─────────────────────────────────────────────────────────────────────────────
# Averaging
# ─────────────────────────────────────────────────────────────────────────────

def build_average(
    dfs: list[pd.DataFrame],
    smooth_min: float,
    delta: bool,
    baseline_min: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    """
    Returns:
        t_grid      — shared time axis in hours (N_GRID points, 0 → max duration)
        means       — {channel: mean array}
        stds        — {channel: std array}
        traces      — {channel: list of per-run interpolated arrays} (for optional display)
    """
    max_h = max(float(df["elapsed_h"].max()) for df in dfs)
    t_grid = np.linspace(0.0, max_h, N_GRID)

    all_traces: dict[str, list[np.ndarray]] = {ch: [] for ch in CHANNELS}

    for df in dfs:
        median_dt = float(df["elapsed_ms"].diff().median())
        t_h = df["elapsed_h"].values

        for ch in CHANNELS:
            raw = df[ch].values.astype(float)

            if delta:
                mask = df["elapsed_h"] <= baseline_min / 60.0
                base = float(df.loc[mask, ch].mean()) if mask.any() else 0.0
                raw  = raw - base

            smoothed = smooth_series(raw, median_dt, smooth_min)

            # Interpolate onto shared grid; NaN outside this run's range
            interp = np.interp(t_grid, t_h, smoothed, left=np.nan, right=np.nan)
            all_traces[ch].append(interp)

    means = {ch: np.nanmean(np.vstack(all_traces[ch]), axis=0) for ch in CHANNELS}
    stds  = {ch: np.nanstd( np.vstack(all_traces[ch]), axis=0, ddof=1)
             if len(dfs) > 1 else {ch: np.zeros(N_GRID) for ch in CHANNELS}[ch]
             for ch in CHANNELS}

    return t_grid, means, stds, all_traces


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def auto_ticks(ax, nx: int = 6, ny: int = 4, n_minor: int = 4) -> None:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nx, steps=[1, 2, 5, 10]))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=ny, steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))


def make_figure(
    t_grid: np.ndarray,
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
    traces: dict[str, list[np.ndarray]],
    n_runs: int,
    delta: bool,
    show_traces: bool,
    output_path: Path,
    fig_width: float = 7.2,
    panel_height: float = 2.0,
) -> None:

    plt.rcParams.update(RC_BASE)

    n_panels = len(CHANNELS)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(fig_width, panel_height * n_panels),
        sharex=True,
    )

    y_label = "Δ ADC Count" if delta else "ADC Count"

    for ci, (ax, ch, node_label, tag, color) in enumerate(
        zip(axes, CHANNELS, NODE_LABELS, PANEL_TAGS, SENSOR_COLORS)
    ):
        mean_vals = means[ch]
        std_vals  = stds[ch]

        # Individual run traces (faint)
        if show_traces:
            for run_trace in traces[ch]:
                ax.plot(t_grid, run_trace, color=color,
                        lw=TRACE_LW, alpha=TRACE_ALPHA, zorder=2)

        # ±1 SD band
        ax.fill_between(t_grid,
                        mean_vals - std_vals,
                        mean_vals + std_vals,
                        color=color, alpha=BAND_ALPHA, linewidth=0, zorder=3)

        # Mean line
        ax.plot(t_grid, mean_vals, color=color, lw=MEAN_LW, zorder=4)

        # Zero reference in delta mode
        if delta:
            ax.axhline(0, color="#AAAAAA", lw=0.9, ls="--", zorder=1)

        ax.set_xlim(0.0, float(t_grid[-1]))
        auto_ticks(ax)
        ax.yaxis.grid(True, color="#E8E8E8", lw=0.8, zorder=0)
        ax.set_axisbelow(True)

        ax.set_ylabel(f"{y_label}\n{node_label}",
                      rotation=0, ha="left", va="center",
                      fontsize=FONTSIZE_AXIS_LABEL)
        ax.yaxis.set_label_coords(-0.30, 0.5)

        ax.text(-0.30, 1.04, tag,
                transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL_TAG, fontweight="bold",
                va="bottom", ha="left", clip_on=False)

        if ci == n_panels - 1:
            ax.set_xlabel("Time (hours)", labelpad=4)
        else:
            ax.tick_params(axis="x", which="both", length=0)

    # Run count annotation in top-right of first panel
    axes[0].text(0.98, 0.95,
                 f"n = {n_runs} run{'s' if n_runs != 1 else ''}  |  mean ± 1 SD",
                 transform=axes[0].transAxes,
                 fontsize=FONTSIZE_ANNOT,
                 ha="right", va="top",
                 color="#444444")

    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.3, h_pad=0.2, rect=[0.22, 0.0, 1.0, 1.0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Average all drift CSVs and plot mean ± 1 SD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--smooth-min", type=float, default=5.0, metavar="MIN",
        help="Rolling-mean window applied to each run before averaging (default: 5)",
    )
    p.add_argument(
        "--delta", action="store_true",
        help="Subtract each run's own initial baseline before averaging",
    )
    p.add_argument(
        "--baseline-min", type=float, default=5.0, metavar="MIN",
        help="Minutes used to compute the per-run baseline for --delta (default: 5)",
    )
    p.add_argument(
        "--no-traces", action="store_true",
        help="Hide the individual run traces; show mean ± SD band only",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output PDF path (default: drift_test/figures/drift_average_figure.pdf)",
    )
    p.add_argument(
        "--width", type=float, default=7.2,
        help="Figure width in inches (default: 7.2)",
    )
    p.add_argument(
        "--panel-height", type=float, default=2.0,
        help="Height per panel in inches (default: 2.0)",
    )
    return p.parse_args()


def main():
    args   = parse_args()
    here   = Path(__file__).parent  # drift_test/
    artifacts_dir = here / "artifacts"

    csvs = sorted(artifacts_dir.glob("drift_*.csv"))
    if not csvs:
        print(f"Error: no drift CSVs found in {artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(csvs)} run(s) from {artifacts_dir.name}/")
    dfs = []
    for path in csvs:
        df = load_csv(path)
        if df.empty:
            continue
        dfs.append(df)
        print(f"  {path.name}  ({len(df)} samples, {df['elapsed_h'].max():.2f} h)")

    if not dfs:
        print("Error: no valid CSVs could be loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"\nComputing average across {len(dfs)} run(s)...")
    t_grid, means, stds, traces = build_average(
        dfs,
        smooth_min=args.smooth_min,
        delta=args.delta,
        baseline_min=args.baseline_min,
    )

    out_path = (
        Path(args.output).with_suffix(".pdf")
        if args.output
        else here / "figures" / "drift_average_figure.pdf"
    )

    make_figure(
        t_grid=t_grid,
        means=means,
        stds=stds,
        traces=traces,
        n_runs=len(dfs),
        delta=args.delta,
        show_traces=not args.no_traces,
        output_path=out_path,
        fig_width=args.width,
        panel_height=args.panel_height,
    )


if __name__ == "__main__":
    main()
