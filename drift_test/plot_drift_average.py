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
import warnings
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

FONTSIZE_AXIS_LABEL = 15
FONTSIZE_TICK_LABEL = 12
FONTSIZE_PANEL_TAG  = 16
FONTSIZE_ANNOT      = 13

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stds  = {ch: np.nanstd( np.vstack(all_traces[ch]), axis=0, ddof=1)
                 if len(dfs) > 1 else {ch: np.zeros(N_GRID) for ch in CHANNELS}[ch]
                 for ch in CHANNELS}

    return t_grid, means, stds, all_traces


# One colour per run (Wong palette)
RUN_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def auto_ticks(ax, nx: int = 6, ny: int = 4, n_minor: int = 4) -> None:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nx, steps=[1, 2, 5, 10]))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=ny, steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))


def make_hourly_figure(
    run_names: list[str],
    dfs: list[pd.DataFrame],
    smooth_min: float,
    delta: bool,
    baseline_min: float,
    output_path: Path,
    fig_width: float = 9.0,
    row_height: float = 1.8,
) -> None:
    """Grid figure: rows = hours, columns = nodes, each cell = all runs overlaid."""

    plt.rcParams.update(RC_BASE)

    max_h   = max(float(df["elapsed_h"].max()) for df in dfs)
    n_hours = int(np.ceil(max_h))
    n_ch    = len(CHANNELS)

    # Fine grid — 500 pts per hour so each zoomed panel has enough resolution
    t_grid = np.linspace(0.0, max_h, 500 * n_hours)

    # Smooth + interpolate every run onto the shared grid
    run_traces: list[dict[str, np.ndarray]] = []
    for df in dfs:
        median_dt = float(df["elapsed_ms"].diff().median())
        t_h = df["elapsed_h"].values
        ch_data: dict[str, np.ndarray] = {}
        for ch in CHANNELS:
            raw = df[ch].values.astype(float)
            if delta:
                mask_b = df["elapsed_h"] <= baseline_min / 60.0
                base   = float(df.loc[mask_b, ch].mean()) if mask_b.any() else 0.0
                raw    = raw - base
            smoothed    = smooth_series(raw, median_dt, smooth_min)
            ch_data[ch] = np.interp(t_grid, t_h, smoothed)
        run_traces.append(ch_data)

    # Global y limits (shared across every panel)
    all_vals = np.concatenate([
        ch_data[ch] for ch_data in run_traces for ch in CHANNELS
    ])
    all_vals = all_vals[np.isfinite(all_vals)]
    pad  = (all_vals.max() - all_vals.min()) * 0.07 or 1.0
    y_lo = all_vals.min() - pad
    y_hi = all_vals.max() + pad

    y_label = "Δ ADC Count" if delta else "ADC Count"

    fig, axes = plt.subplots(
        n_hours, n_ch,
        figsize=(fig_width, row_height * n_hours),
        sharey=True, sharex=False,
    )

    dt = t_grid[1] - t_grid[0]

    for hi in range(n_hours):
        h_lo  = float(hi)
        h_hi  = float(min(hi + 1, max_h))
        mask  = (t_grid >= h_lo) & (t_grid <= h_hi + dt * 0.5)
        t_min = (t_grid[mask] - h_lo) * 60.0   # convert to minutes within this hour

        for ci, (ch, node_label) in enumerate(zip(CHANNELS, NODE_LABELS)):
            ax    = axes[hi, ci]
            color_node = SENSOR_COLORS[ci]

            for ri, (run_name, ch_data) in enumerate(zip(run_names, run_traces)):
                run_color = RUN_COLORS[ri % len(RUN_COLORS)]
                label = run_name if (hi == 0 and ci == n_ch - 1) else "_nolegend_"
                ax.plot(t_min, ch_data[ch][mask],
                        color=run_color, lw=1.4, label=label, zorder=3)

            ax.set_xlim(0.0, 60.0)
            ax.yaxis.grid(True, color="#E8E8E8", lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
            ax.tick_params(labelsize=9)

            # Column header: node name (top row only)
            if hi == 0:
                ax.set_title(node_label,
                             fontsize=FONTSIZE_AXIS_LABEL, fontweight="bold", pad=5)

            # Row label: hour number (left column only)
            if ci == 0:
                ax.set_ylabel(f"Hr {hi + 1}",
                              rotation=0, ha="right", va="center",
                              fontsize=11, labelpad=10)

            # X-axis label on bottom row only
            if hi == n_hours - 1:
                ax.set_xlabel("Minutes", labelpad=3, fontsize=11)
            else:
                ax.tick_params(axis="x", which="both", labelbottom=False)

    # Set shared y limits once (sharey propagates to all panels)
    axes[0, 0].set_ylim(y_lo, y_hi)

    # Legend in top-right panel
    axes[0, n_ch - 1].legend(
        loc="upper right",
        fontsize=FONTSIZE_ANNOT,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
    )

    # Global y-axis label on the left margin
    fig.text(0.01, 0.5, y_label,
             va="center", ha="center", rotation="vertical",
             fontsize=FONTSIZE_AXIS_LABEL, fontweight="bold",
             fontfamily="serif")

    fig.tight_layout(pad=0.4, h_pad=0.4, w_pad=0.4, rect=[0.04, 0.0, 1.0, 1.0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path.name}")


def make_overlay_figure(
    run_names: list[str],
    dfs: list[pd.DataFrame],
    smooth_min: float,
    delta: bool,
    baseline_min: float,
    output_path: Path,
    fig_width: float = 7.2,
    panel_height: float = 3.2,
) -> None:
    """Plot every individual run as its own coloured line, one panel per node."""

    plt.rcParams.update(RC_BASE)

    n_panels = len(CHANNELS)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(fig_width, panel_height * n_panels),
        sharex=True,
    )

    y_label = "Δ ADC Count" if delta else "ADC Count"

    # Smooth each run and interpolate onto a shared display grid
    max_h = max(float(df["elapsed_h"].max()) for df in dfs)
    t_grid = np.linspace(0.0, max_h, N_GRID)

    run_traces: list[dict[str, np.ndarray]] = []
    for df in dfs:
        median_dt = float(df["elapsed_ms"].diff().median())
        t_h = df["elapsed_h"].values
        ch_data: dict[str, np.ndarray] = {}
        for ch in CHANNELS:
            raw = df[ch].values.astype(float)
            if delta:
                mask = df["elapsed_h"] <= baseline_min / 60.0
                base = float(df.loc[mask, ch].mean()) if mask.any() else 0.0
                raw = raw - base
            smoothed = smooth_series(raw, median_dt, smooth_min)
            ch_data[ch] = np.interp(t_grid, t_h, smoothed)
        run_traces.append(ch_data)

    # Shared y range across all panels and all runs
    all_vals = np.concatenate([
        ch_data[ch] for ch_data in run_traces for ch in CHANNELS
    ])
    all_vals = all_vals[np.isfinite(all_vals)]
    pad = (all_vals.max() - all_vals.min()) * 0.07 or 1.0
    y_lo = all_vals.min() - pad
    y_hi = all_vals.max() + pad

    for ci, (ax, ch, node_label, tag) in enumerate(
        zip(axes, CHANNELS, NODE_LABELS, PANEL_TAGS)
    ):
        for ri, (run_name, ch_data) in enumerate(zip(run_names, run_traces)):
            color = RUN_COLORS[ri % len(RUN_COLORS)]
            label = run_name if ci == 0 else "_nolegend_"
            ax.plot(t_grid, ch_data[ch], color=color, lw=MEAN_LW,
                    label=label, zorder=3)

        if delta:
            ax.axhline(0, color="#AAAAAA", lw=0.9, ls="--", zorder=1)

        ax.set_xlim(0.0, max_h)
        ax.set_ylim(y_lo, y_hi)
        auto_ticks(ax)
        ax.yaxis.grid(True, color="#E8E8E8", lw=0.8, zorder=0)
        ax.set_axisbelow(True)

        ylabel_text = "\n".join(y_label.split() + [node_label])
        ax.set_ylabel(ylabel_text,
                      rotation=0, ha="left", va="center",
                      fontsize=FONTSIZE_AXIS_LABEL)
        ax.yaxis.set_label_coords(-0.20, 0.5)

        ax.text(-0.20, 1.04, tag,
                transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL_TAG, fontweight="bold",
                va="bottom", ha="left", clip_on=False)

        if ci == n_panels - 1:
            ax.set_xlabel("Time (hours)", labelpad=4)
        else:
            ax.tick_params(axis="x", which="both", length=0)

    axes[0].legend(
        loc="upper right",
        ncol=1,
        fontsize=FONTSIZE_ANNOT,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
    )

    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.5, h_pad=0.8, rect=[0.18, 0.03, 1.0, 1.0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path.name}")


def make_figure(
    t_grid: np.ndarray,
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
    traces: dict[str, list[np.ndarray]],
    n_runs: int,
    delta: bool,
    baseline_min: float,
    show_traces: bool,
    output_path: Path,
    fig_width: float = 7.2,
    panel_height: float = 2.0,
    clean: bool = False,
) -> None:

    plt.rcParams.update(RC_BASE)

    n_panels = len(CHANNELS)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(fig_width, panel_height * n_panels),
        sharex=True,
    )

    y_label = "Δ ADC Count" if delta else "ADC Count"

    if not clean:
        print(f"\nAverage deviation from baseline (first {baseline_min:.0f} min):")

    # Global y range across all panels
    all_vals = np.concatenate([
        np.concatenate([means[ch], means[ch] - stds[ch], means[ch] + stds[ch]])
        for ch in CHANNELS
    ])
    all_vals = all_vals[np.isfinite(all_vals)]
    pad = (all_vals.max() - all_vals.min()) * 0.07 or 1.0
    y_lo = all_vals.min() - pad
    y_hi = all_vals.max() + pad

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

        if not clean:
            # Baseline reference — mean of first baseline_min minutes of averaged trace
            baseline_mask = t_grid <= baseline_min / 60.0
            baseline_val  = float(np.nanmean(mean_vals[baseline_mask])) if baseline_mask.any() else float(np.nanmean(mean_vals[:10]))
            ax.axhline(baseline_val, color="#444444", lw=1.1, ls="--", zorder=5,
                       label="Baseline" if ci == 0 else "_nolegend_")

            # Average deviation from baseline
            avg_dev = float(np.nanmean(mean_vals) - baseline_val)
            mean_level = baseline_val + avg_dev          # == nanmean(mean_vals)
            ax.axhline(mean_level, color=color, lw=1.0, ls=":", alpha=0.7, zorder=5)
            sign = "+" if avg_dev >= 0 else ""
            ax.annotate(
                f"Avg Δ = {sign}{avg_dev:.1f} ADC",
                xy=(float(t_grid[-1]), mean_level),
                xytext=(-6, 5 if avg_dev >= 0 else -5),
                textcoords="offset points",
                ha="right", va="bottom" if avg_dev >= 0 else "top",
                fontsize=FONTSIZE_ANNOT,
                color=color,
                zorder=6,
            )
            print(f"  {node_label}: avg deviation from baseline = {sign}{avg_dev:.2f} ADC counts")

        # Zero reference in delta mode
        if delta:
            ax.axhline(0, color="#AAAAAA", lw=0.9, ls=":", zorder=1)

        ax.set_xlim(0.0, float(t_grid[-1]))
        ax.set_ylim(y_lo, y_hi)
        auto_ticks(ax)
        ax.yaxis.grid(True, color="#E8E8E8", lw=0.8, zorder=0)
        ax.set_axisbelow(True)

        ylabel_text = "\n".join(y_label.split() + [node_label])
        ax.set_ylabel(ylabel_text,
                      rotation=0, ha="left", va="center",
                      fontsize=FONTSIZE_AXIS_LABEL)
        ax.yaxis.set_label_coords(-0.20, 0.5)

        ax.text(-0.20, 1.04, tag,
                transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL_TAG, fontweight="bold",
                va="bottom", ha="left", clip_on=False)

        if ci == n_panels - 1:
            ax.set_xlabel("Time (hours)", labelpad=4)
        else:
            ax.tick_params(axis="x", which="both", length=0)

    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.3, h_pad=0.2, rect=[0.18, 0.03, 1.0, 1.0])

    if not clean:
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="#444444", lw=1.1, ls="--",
                   label=f"Baseline (first {baseline_min:.0f} min avg)"),
        ]
        axes[0].legend(
            handles=legend_handles,
            loc="upper right",
            ncol=1,
            fontsize=FONTSIZE_ANNOT,
            frameon=False,
            handlelength=1.8,
            handletextpad=0.5,
        )

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
        "--clean", action="store_true",
        help="Strip all overlays (baseline line, avg-deviation annotation, legend) — raw average only",
    )
    p.add_argument(
        "--overlay", action="store_true",
        help="Plot each individual run as its own coloured line instead of averaging",
    )
    p.add_argument(
        "--hourly", action="store_true",
        help="Grid figure: rows=hours, columns=nodes — each cell shows all runs for that hour",
    )
    p.add_argument(
        "--data-dir", "-d", default=None,
        help="Folder containing drift CSVs. Defaults to drift_test/artifacts/.",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output PDF path (default: drift_test/figures/<data-dir-name>_average_figure.pdf)",
    )
    p.add_argument(
        "--width", type=float, default=7.2,
        help="Figure width in inches (default: 7.2)",
    )
    p.add_argument(
        "--panel-height", type=float, default=2.5,
        help="Height per panel in inches (default: 2.5)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).parent  # drift_test/

    artifacts_dir = Path(args.data_dir) if args.data_dir else here / "artifacts"
    if not artifacts_dir.exists():
        print(f"Error: folder not found: {artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    csvs = sorted(artifacts_dir.glob("drift_*.csv"))
    if not csvs:
        print(f"Error: no drift CSVs found in {artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(csvs)} run(s) from {artifacts_dir}/")
    dfs, run_names = [], []
    for path in csvs:
        df = load_csv(path)
        if df.empty:
            continue
        dfs.append(df)
        run_names.append(f"Test {len(dfs)}")
        print(f"  {path.name}  ({len(df)} samples, {df['elapsed_h'].max():.2f} h)")

    if not dfs:
        print("Error: no valid CSVs could be loaded.", file=sys.stderr)
        sys.exit(1)

    # ── Hourly grid mode ────────────────────────────────────────────────────
    if args.hourly:
        default_name = f"{artifacts_dir.name}_hourly_figure.pdf"
        out_path = (
            Path(args.output).with_suffix(".pdf")
            if args.output
            else here / "figures" / default_name
        )
        print(f"\nGenerating hourly grid figure ({len(dfs)} runs)...")
        make_hourly_figure(
            run_names=run_names,
            dfs=dfs,
            smooth_min=args.smooth_min,
            delta=args.delta,
            baseline_min=args.baseline_min,
            output_path=out_path,
            fig_width=args.width if args.width != 7.2 else 9.0,
            row_height=args.panel_height if args.panel_height != 2.5 else 1.8,
        )
        return

    # ── Overlay mode ────────────────────────────────────────────────────────
    if args.overlay:
        default_name = f"{artifacts_dir.name}_overlay_figure.pdf"
        out_path = (
            Path(args.output).with_suffix(".pdf")
            if args.output
            else here / "figures" / default_name
        )
        print(f"\nGenerating overlay figure ({len(dfs)} runs)...")
        make_overlay_figure(
            run_names=run_names,
            dfs=dfs,
            smooth_min=args.smooth_min,
            delta=args.delta,
            baseline_min=args.baseline_min,
            output_path=out_path,
            fig_width=args.width,
            panel_height=args.panel_height,
        )
        return

    # ── Average mode ────────────────────────────────────────────────────────
    print(f"\nComputing average across {len(dfs)} run(s)...")
    t_grid, means, stds, traces = build_average(
        dfs,
        smooth_min=args.smooth_min,
        delta=args.delta,
        baseline_min=args.baseline_min,
    )

    suffix = "_raw_average_figure" if args.clean else "_average_figure"
    default_name = f"{artifacts_dir.name}{suffix}.pdf"
    out_path = (
        Path(args.output).with_suffix(".pdf")
        if args.output
        else here / "figures" / default_name
    )

    make_figure(
        t_grid=t_grid,
        means=means,
        stds=stds,
        traces=traces,
        n_runs=len(dfs),
        delta=args.delta,
        baseline_min=args.baseline_min,
        show_traces=not args.no_traces,
        output_path=out_path,
        fig_width=args.width,
        panel_height=args.panel_height,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
