"""
plot_sensors.py
---------------
Generate publication-quality sensor voltage time-series figures
formatted for Science Robotics journal submission.

Usage
-----
    python plot_sensors.py data.csv
    python plot_sensors.py data.csv --output my_figure --width 2col
    python plot_sensors.py data.csv --output my_figure --width 1col
    python plot_sensors.py data.csv --smooth 5 --labels "Sensor A" "Sensor B" "Sensor C"
    python plot_sensors.py data.csv --xunit ms --yunit mV

CSV format expected
-------------------
    time, sensor1, sensor2, sensor3
    0.00, 0.12, 0.34, 0.56
    ...

Columns may also be named: time/t/Time, then any three remaining numeric columns
are used in order (auto-detection).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Science Robotics / AAAS figure style
# ─────────────────────────────────────────────────────────────────────────────

# Column widths as specified by Science family journals (inches)
WIDTHS = {
    "1col": 3.5,   # single-column  (89 mm)
    "2col": 7.2,   # double-column (183 mm)
    "1.5col": 5.5, # intermediate
}

# Color palette: high-contrast, print-safe, color-blind friendly
# Based on Wong (2011) Nature Methods palette
PALETTE = [
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # green
]

LINESTYLES = ["-", "-", "-"]
LINEWIDTH   = 1.2       # pt
ALPHA       = 0.92

# Typography – match AAAS/Science Robotics style
FONT_FAMILY = "Arial"   # falls back to DejaVu Sans if not installed
FONTSIZE_AXIS_LABEL = 8
FONTSIZE_TICK_LABEL = 7
FONTSIZE_LEGEND     = 7
FONTSIZE_PANEL_TAG  = 9   # bold "(a)" style tags if ever used

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
    "legend.handlelength":  1.5,
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
    "pdf.fonttype":         42,   # embed fonts as TrueType (required for many journals)
    "ps.fonttype":          42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> tuple[pd.Series, pd.DataFrame]:
    """Load CSV and return (time_series, sensor_dataframe)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Identify time column — supports: time, t, timestamp, run_ms, time_ms, time_s, etc.
    time_candidates = [
        c for c in df.columns
        if c.lower() in ("time", "t", "timestamp", "run_ms", "time_ms", "time_s",
                         "elapsed_ms", "elapsed_s", "ms", "seconds")
    ]
    if not time_candidates:
        # Fall back: pick the first column if it looks monotonically increasing
        first_col = df.columns[0]
        if df[first_col].is_monotonic_increasing:
            time_candidates = [first_col]
            print(f"  [auto] Using '{first_col}' as time column (first monotonic column).")
    if not time_candidates:
        raise ValueError(
            "Could not find a time column. Expected a column named 'time', 'run_ms', "
            "'timestamp', etc. — or ensure the first column is monotonically increasing."
        )
    time_col = time_candidates[0]
    time = df[time_col].astype(float)

    # Identify sensor columns.
    # Priority 1: explicit names (sensor1/2/3, v1/2/3, raw_ch*)
    # Priority 2: any numeric non-time column that is NOT a flat sentinel (i.e. not all -2)
    sensor_candidates = [c for c in df.columns if c.lower() in ("sensor1", "sensor2", "sensor3",
                                                                  "v1", "v2", "v3")]
    if len(sensor_candidates) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_time = [c for c in numeric_cols if c != time_col]

        # Drop columns whose unique value count == 1 (flat / disconnected, e.g. all -2)
        active_cols = [
            c for c in non_time
            if df[c].nunique() > 1
        ]
        if len(active_cols) < 3:
            raise ValueError(
                f"Found only {len(active_cols)} active sensor column(s) after filtering "
                f"flat/disconnected channels: {active_cols}. Need at least 3."
            )
        sensor_cols = active_cols[:3]
        print(f"  [auto] Active sensor columns selected: {sensor_cols}")
        print(f"  [auto] Skipped flat/disconnected columns: "
              f"{[c for c in non_time if c not in sensor_cols]}")
    else:
        sensor_cols = sensor_candidates[:3]

    sensors = df[sensor_cols].astype(float)
    return time, sensors


def smooth_signal(series: pd.Series, window: int) -> pd.Series:
    """Apply a centered rolling mean (Hann-weighted) for subtle smoothing."""
    if window <= 1:
        return series
    weights = np.hanning(window)
    weights /= weights.sum()
    padded = np.pad(series.values, window // 2, mode="edge")
    smoothed = np.convolve(padded, weights, mode="valid")[: len(series)]
    return pd.Series(smoothed, index=series.index)


def auto_ticks(ax, axis="both", n_major=5, n_minor=4):
    """Set clean auto-locators appropriate for journal figures."""
    loc = ticker.MaxNLocator(nbins=n_major, steps=[1, 2, 2.5, 5, 10])
    if axis in ("x", "both"):
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))
    loc2 = ticker.MaxNLocator(nbins=n_major, steps=[1, 2, 2.5, 5, 10])
    if axis in ("y", "both"):
        ax.yaxis.set_major_locator(loc2)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))


# ─────────────────────────────────────────────────────────────────────────────
# Main plotting function
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(
    time: pd.Series,
    sensors: pd.DataFrame,
    labels: list[str],
    x_label: str,
    y_label: str,
    title: str | None,
    fig_width: float,
    aspect: float,
    smooth_window: int,
    show_raw: bool,
    output_path: Path,
) -> None:

    plt.rcParams.update(RC_BASE)

    fig_height = fig_width / aspect
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for i, (col, label, color, ls) in enumerate(
        zip(sensors.columns, labels, PALETTE, LINESTYLES)
    ):
        raw = sensors[col]
        smoothed = smooth_signal(raw, smooth_window)

        if smooth_window > 1 and show_raw:
            ax.plot(time, raw, color=color, lw=0.4, alpha=0.35, linestyle=ls)

        ax.plot(
            time,
            smoothed,
            color=color,
            lw=LINEWIDTH,
            alpha=ALPHA,
            linestyle=ls,
            label=label,
            zorder=3 + i,
        )

    # Axes labels
    ax.set_xlabel(x_label, labelpad=4)
    ax.set_ylabel(y_label, labelpad=4)
    if title:
        ax.set_title(title, pad=6, fontsize=FONTSIZE_AXIS_LABEL, fontweight="bold")

    # Tick formatting
    auto_ticks(ax, "both")

    # Tight axis limits with small breathing room
    t_min, t_max = time.min(), time.max()
    t_pad = (t_max - t_min) * 0.01
    ax.set_xlim(t_min - t_pad, t_max + t_pad)

    all_vals = sensors.values.flatten()
    v_min, v_max = np.nanmin(all_vals), np.nanmax(all_vals)
    v_pad = (v_max - v_min) * 0.06
    ax.set_ylim(v_min - v_pad, v_max + v_pad)

    # Legend — placed inside upper right, no box
    legend_handles = [
        Line2D([0], [0], color=c, lw=LINEWIDTH, linestyle=ls, label=lbl)
        for c, ls, lbl in zip(PALETTE, LINESTYLES, labels)
    ]
    ax.legend(
        handles=legend_handles,
        loc="best",
        ncol=1,
        borderpad=0,
        columnspacing=0.8,
    )

    # Final layout
    fig.tight_layout(pad=0.3)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✓  Figure saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Science-Robotics-quality sensor voltage plots from a CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", help="Path to input CSV file")
    p.add_argument(
        "--output", "-o", default=None,
        help="Output PDF path (without extension). Defaults to <csv_stem>_figure.pdf"
    )
    p.add_argument(
        "--width", choices=list(WIDTHS.keys()), default="2col",
        help="Figure width preset: 1col (3.5 in), 1.5col (5.5 in), 2col (7.2 in). Default: 2col"
    )
    p.add_argument(
        "--aspect", type=float, default=2.5,
        help="Width-to-height aspect ratio. Default: 2.5 (wider than tall, typical for journals)"
    )
    p.add_argument(
        "--smooth", type=int, default=1, metavar="WINDOW",
        help="Hann-window smoothing width (samples). 1 = no smoothing. Default: 1"
    )
    p.add_argument(
        "--show-raw", action="store_true",
        help="If --smooth > 1, also plot the raw signal faintly behind the smoothed trace"
    )
    p.add_argument(
        "--labels", nargs=3, default=["Channel 1", "Channel 2", "Channel 3"],
        metavar=("L1", "L2", "L3"),
        help='Legend labels for the three sensors. Default: "Channel 1" "Channel 2" "Channel 3"'
    )
    p.add_argument(
        "--xunit", default="ms",
        help="Unit label for time axis. Default: ms (milliseconds)"
    )
    p.add_argument(
        "--yunit", default="ADC counts",
        help="Unit label for voltage/signal axis. Default: 'ADC counts'"
    )
    p.add_argument(
        "--xlabel", default=None,
        help="Override full x-axis label. Overrides --xunit."
    )
    p.add_argument(
        "--ylabel", default=None,
        help="Override full y-axis label. Overrides --yunit."
    )
    p.add_argument(
        "--title", default=None,
        help="Optional figure title (not recommended for final journal submission)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {csv_path} …")
    time, sensors = load_csv(str(csv_path))
    print(f"  {len(time)} samples, columns: {list(sensors.columns)}")

    output_stem = args.output or (csv_path.stem + "_figure")
    output_path = Path(output_stem).with_suffix(".pdf")

    x_label = args.xlabel or f"Time ({args.xunit})"
    y_label = args.ylabel or f"Voltage ({args.yunit})"

    make_figure(
        time=time,
        sensors=sensors,
        labels=args.labels,
        x_label=x_label,
        y_label=y_label,
        title=args.title,
        fig_width=WIDTHS[args.width],
        aspect=args.aspect,
        smooth_window=args.smooth,
        show_raw=args.show_raw,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
