"""
plot_sensors_panels.py
----------------------
Generate publication-quality multi-panel sensor figures (one panel per sensor)
formatted for Science Robotics journal submission.

Each sensor gets its own axes panel. Panels share a common x-axis so signals
are perfectly aligned. The bottom panel alone shows x-tick labels (clean look).
Panel tags (a), (b), (c) are placed in the upper-left corner of each panel.

Usage
-----
    python plot_sensors_panels.py data.csv
    python plot_sensors_panels.py data.csv --layout side          # 1 row × 3 cols
    python plot_sensors_panels.py data.csv --layout stacked       # 3 rows × 1 col (default)
    python plot_sensors_panels.py data.csv --smooth 15 --show-raw
    python plot_sensors_panels.py data.csv --labels "Sensor A" "Sensor B" "Sensor C"
    python plot_sensors_panels.py data.csv --shared-y             # same y-scale across panels

CSV format expected
-------------------
    run_ms, raw_ch4, raw_ch5, raw_ch7
    1189, 18349, 18615, 14974
    ...

Auto-detects time column (run_ms, time, t, timestamp, …) and skips flat/
disconnected channels (e.g. columns where every value is -2).
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
# Science Robotics / AAAS figure style  (shared with plot_sensors.py)
# ─────────────────────────────────────────────────────────────────────────────

WIDTHS = {
    "1col":  3.5,
    "1.5col": 5.5,
    "2col":  7.2,
}

PALETTE = [
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # green
]

LINEWIDTH = 1.2
ALPHA     = 0.92

FONT_FAMILY         = "Arial"
FONTSIZE_AXIS_LABEL = 8
FONTSIZE_TICK_LABEL = 7
FONTSIZE_LEGEND     = 7
FONTSIZE_PANEL_TAG  = 8   # (a), (b), (c) labels

RC_BASE = {
    "font.family":          "sans-serif",
    "font.sans-serif":      [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
    "font.size":            FONTSIZE_TICK_LABEL,
    "axes.labelsize":       FONTSIZE_AXIS_LABEL,
    "axes.titlesize":       FONTSIZE_AXIS_LABEL,
    "xtick.labelsize":      FONTSIZE_TICK_LABEL,
    "ytick.labelsize":      FONTSIZE_TICK_LABEL,
    "legend.fontsize":      FONTSIZE_LEGEND,
    "legend.frameon":       False,
    "legend.handlelength":  1.4,
    "legend.handletextpad": 0.4,
    "axes.linewidth":       0.6,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.minor.width":    0.4,
    "ytick.minor.width":    0.4,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "xtick.minor.size":     1.8,
    "ytick.minor.size":     1.8,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "lines.linewidth":      LINEWIDTH,
    "lines.solid_capstyle": "round",
    "figure.dpi":           300,
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.02,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
}

PANEL_TAGS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> tuple[pd.Series, pd.DataFrame]:
    """Load CSV, auto-detect time column, skip flat/disconnected channels."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Time column detection
    time_candidates = [
        c for c in df.columns
        if c.lower() in ("time", "t", "timestamp", "run_ms", "time_ms", "time_s",
                         "elapsed_ms", "elapsed_s", "ms", "seconds")
    ]
    if not time_candidates:
        first_col = df.columns[0]
        if df[first_col].is_monotonic_increasing:
            time_candidates = [first_col]
            print(f"  [auto] Using '{first_col}' as time column.")
    if not time_candidates:
        raise ValueError(
            "No time column found. Expected 'time', 'run_ms', 'timestamp', etc."
        )
    time_col = time_candidates[0]
    time = df[time_col].astype(float)

    # Sensor column detection — skip flat/disconnected channels
    sensor_candidates = [c for c in df.columns if c.lower() in
                         ("sensor1", "sensor2", "sensor3", "v1", "v2", "v3")]
    if len(sensor_candidates) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_time = [c for c in numeric_cols if c != time_col]
        active_cols = [c for c in non_time if df[c].nunique() > 1]
        if len(active_cols) < 3:
            raise ValueError(
                f"Found only {len(active_cols)} active sensor column(s): {active_cols}. Need ≥ 3."
            )
        sensor_cols = active_cols[:3]
        print(f"  [auto] Active channels: {sensor_cols}")
        skipped = [c for c in non_time if c not in sensor_cols]
        if skipped:
            print(f"  [auto] Skipped flat/disconnected: {skipped}")
    else:
        sensor_cols = sensor_candidates[:3]

    return time, df[sensor_cols].astype(float)


def smooth_signal(series: pd.Series, window: int) -> pd.Series:
    """Hann-weighted centered rolling average."""
    if window <= 1:
        return series
    weights = np.hanning(window)
    weights /= weights.sum()
    padded = np.pad(series.values, window // 2, mode="edge")
    smoothed = np.convolve(padded, weights, mode="valid")[: len(series)]
    return pd.Series(smoothed, index=series.index)


def auto_ticks(ax, axis="both", n_major=5, n_minor=4):
    loc = ticker.MaxNLocator(nbins=n_major, steps=[1, 2, 2.5, 5, 10])
    loc2 = ticker.MaxNLocator(nbins=n_major, steps=[1, 2, 2.5, 5, 10])
    if axis in ("x", "both"):
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))
    if axis in ("y", "both"):
        ax.yaxis.set_major_locator(loc2)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))


def panel_tag(ax, tag: str, layout: str):
    """Place a bold (a)/(b)/(c) tag in the upper-left corner of an axes panel."""
    # Use axes-fraction coordinates so it stays inside regardless of data range
    x_pos = -0.13 if layout == "side" else -0.11
    ax.text(
        x_pos, 1.02, tag,
        transform=ax.transAxes,
        fontsize=FONTSIZE_PANEL_TAG,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main plotting function
# ─────────────────────────────────────────────────────────────────────────────

def make_panel_figure(
    time: pd.Series,
    sensors: pd.DataFrame,
    labels: list[str],
    x_label: str,
    y_label: str,
    title: str | None,
    layout: str,          # "stacked" or "side"
    fig_width: float,
    panel_height: float,  # height of each individual panel (inches)
    smooth_window: int,
    show_raw: bool,
    shared_y: bool,
    output_path: Path,
) -> None:

    plt.rcParams.update(RC_BASE)

    n = len(sensors.columns)  # always 3

    if layout == "stacked":
        nrows, ncols = n, 1
        fig_height = panel_height * n
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_width, fig_height),
            sharex=True,
            sharey=shared_y,
        )
    else:  # side by side
        nrows, ncols = 1, n
        fig_height = panel_height
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_width, fig_height),
            sharex=True,
            sharey=shared_y,
        )

    axes = np.atleast_1d(axes).flatten()

    # Pre-compute y limits per panel (for consistent padding even without shared_y)
    y_ranges = []
    for col in sensors.columns:
        vals = sensors[col].values
        v_min, v_max = np.nanmin(vals), np.nanmax(vals)
        v_pad = (v_max - v_min) * 0.08
        y_ranges.append((v_min - v_pad, v_max + v_pad))

    if shared_y:
        global_min = min(r[0] for r in y_ranges)
        global_max = max(r[1] for r in y_ranges)
        y_ranges = [(global_min, global_max)] * n

    t_min, t_max = time.min(), time.max()
    t_pad = (t_max - t_min) * 0.01

    for i, (col, label, color, tag) in enumerate(
        zip(sensors.columns, labels, PALETTE, PANEL_TAGS)
    ):
        ax = axes[i]
        raw = sensors[col]
        smoothed = smooth_signal(raw, smooth_window)

        # Faint raw trace (optional)
        if smooth_window > 1 and show_raw:
            ax.plot(time, raw, color=color, lw=0.4, alpha=0.30, zorder=1)

        # Main signal line
        ax.plot(time, smoothed, color=color, lw=LINEWIDTH, alpha=ALPHA, zorder=2)

        # Axis limits
        ax.set_xlim(t_min - t_pad, t_max + t_pad)
        ax.set_ylim(*y_ranges[i])

        # Ticks
        auto_ticks(ax, "both")

        # Y-axis label on every panel (sensor name doubles as y-label)
        ax.set_ylabel(label, labelpad=4, color=color, fontweight="bold")

        # Panel tag (a), (b), (c)
        panel_tag(ax, tag, layout)

        # X-axis label: only on the last panel for stacked; all for side-by-side
        is_last = (i == n - 1)
        if layout == "stacked":
            if is_last:
                ax.set_xlabel(x_label, labelpad=4)
            else:
                # Hide x-tick labels on all but the bottom panel
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.tick_params(axis="x", which="both", length=0)
        else:
            ax.set_xlabel(x_label, labelpad=4)

    # Shared y-axis label for stacked layout (printed once on the left)
    if layout == "stacked" and not shared_y:
        # Individual y-labels already set above; add a global one too if requested
        pass  # sensor name already serves as the per-panel y-label

    if title:
        fig.suptitle(title, fontsize=FONTSIZE_AXIS_LABEL, fontweight="bold", y=1.01)

    # Layout spacing
    if layout == "stacked":
        fig.tight_layout(pad=0.4, h_pad=0.6)
    else:
        fig.tight_layout(pad=0.4, w_pad=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✓  Figure saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-panel Science-Robotics-quality sensor plots (one panel per sensor).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", help="Path to input CSV file")
    p.add_argument(
        "--output", "-o", default=None,
        help="Output PDF path (without extension). Defaults to <csv_stem>_panels.pdf"
    )
    p.add_argument(
        "--layout", choices=["stacked", "side"], default="stacked",
        help="Panel arrangement: 'stacked' = 3 rows × 1 col (default), 'side' = 1 row × 3 cols"
    )
    p.add_argument(
        "--width", choices=list(WIDTHS.keys()), default="2col",
        help="Figure width preset: 1col (3.5 in), 1.5col (5.5 in), 2col (7.2 in). Default: 2col"
    )
    p.add_argument(
        "--panel-height", type=float, default=1.4,
        help="Height of each individual panel in inches. Default: 1.4"
    )
    p.add_argument(
        "--smooth", type=int, default=1, metavar="WINDOW",
        help="Hann-window smoothing width (samples). 1 = off. Default: 1"
    )
    p.add_argument(
        "--show-raw", action="store_true",
        help="If --smooth > 1, overlay faint raw signal behind smoothed trace"
    )
    p.add_argument(
        "--labels", nargs=3, default=["Channel 1", "Channel 2", "Channel 3"],
        metavar=("L1", "L2", "L3"),
        help='Panel labels (used as y-axis titles). Default: "Channel 1" "Channel 2" "Channel 3"'
    )
    p.add_argument(
        "--shared-y", action="store_true",
        help="Force identical y-axis scale across all panels"
    )
    p.add_argument(
        "--xunit", default="ms",
        help="Unit label for time axis. Default: ms"
    )
    p.add_argument(
        "--yunit", default="ADC counts",
        help="Unit label for voltage/signal axis (appended to y-label if set). Default: ADC counts"
    )
    p.add_argument(
        "--xlabel", default=None,
        help="Override full x-axis label."
    )
    p.add_argument(
        "--title", default=None,
        help="Optional figure-level title (suptitle)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {csv_path} …")
    time, sensors = load_csv(str(csv_path))
    print(f"  {len(time)} samples, columns: {list(sensors.columns)}")

    output_stem = args.output or (csv_path.stem + "_panels")
    output_path = Path(output_stem).with_suffix(".pdf")

    x_label = args.xlabel or f"Time ({args.xunit})"

    make_panel_figure(
        time=time,
        sensors=sensors,
        labels=args.labels,
        x_label=x_label,
        y_label=args.yunit,
        title=args.title,
        layout=args.layout,
        fig_width=WIDTHS[args.width],
        panel_height=args.panel_height,
        smooth_window=args.smooth,
        show_raw=args.show_raw,
        shared_y=args.shared_y,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
