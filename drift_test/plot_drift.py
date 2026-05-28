"""
plot_drift.py
-------------
Visualise drift-test CSVs produced by drift_logger.py.

By default, generates a figure for EVERY file in drift_test/artifacts/ and
saves each result to drift_test/figures/.

The x-axis always spans exactly how long the recording lasted.

Usage
-----
    # Process all CSVs in artifacts/ → figures saved to figures/
    python drift_test/plot_drift.py

    # Single specific file
    python drift_test/plot_drift.py drift_test/artifacts/drift_20260527_182448.csv

    # Options
    python drift_test/plot_drift.py --smooth-min 10
    python drift_test/plot_drift.py --delta          # show Δ from initial baseline
    python drift_test/plot_drift.py --no-raw         # hide faint raw trace
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
# Style constants  (matches plot_repeatability.py)
# ─────────────────────────────────────────────────────────────────────────────

CHANNELS    = ["sensor_1", "sensor_5", "sensor_7"]
NODE_LABELS = ["Node 1", "Node 5", "Node 7"]
PANEL_TAGS  = ["(a)", "(b)", "(c)"]

# Wong (2011) color-blind-safe palette
SENSOR_COLORS = ["#0072B2", "#D55E00", "#009E73"]

MEAN_LW   = 2.0
RAW_LW    = 0.5
RAW_ALPHA = 0.18

FONTSIZE_AXIS_LABEL = 14
FONTSIZE_TICK_LABEL = 12
FONTSIZE_PANEL_TAG  = 16

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


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = [c for c in ["elapsed_ms"] + CHANNELS if c not in df.columns]
    if missing:
        print(f"  Error: missing columns {missing} — skipping.", file=sys.stderr)
        return pd.DataFrame()

    df = df[["elapsed_ms"] + CHANNELS].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df["elapsed_h"] = df["elapsed_ms"] / 3_600_000.0
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def auto_ticks(ax, nx: int = 6, ny: int = 4, n_minor: int = 4) -> None:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nx, steps=[1, 2, 5, 10]))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=ny, steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))


def make_figure(
    df: pd.DataFrame,
    smooth_min: float,
    delta: bool,
    baseline_min: float,
    show_raw: bool,
    output_path: Path,
    fig_width: float = 7.2,
    panel_height: float = 2.0,
) -> None:

    plt.rcParams.update(RC_BASE)

    median_dt_ms   = float(df["elapsed_ms"].diff().median())
    samples_per_min = 60_000.0 / max(median_dt_ms, 1.0)
    smooth_samples  = max(1, int(smooth_min * samples_per_min))

    baseline_mask = df["elapsed_h"] <= baseline_min / 60.0
    baseline: dict[str, float] = {
        ch: float(df.loc[baseline_mask, ch].mean()) if baseline_mask.any() else 0.0
        for ch in CHANNELS
    }

    total_hours = float(df["elapsed_h"].max())

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
        raw = df[ch].values.astype(float)
        t_h = df["elapsed_h"].values

        if delta:
            raw = raw - baseline[ch]

        smoothed = pd.Series(raw).rolling(smooth_samples, center=True, min_periods=1).mean().values

        if show_raw:
            ax.plot(t_h, raw, color=color, lw=RAW_LW, alpha=RAW_ALPHA, zorder=2)

        ax.plot(t_h, smoothed, color=color, lw=MEAN_LW, zorder=3)

        if delta:
            ax.axhline(0, color="#AAAAAA", lw=0.9, ls="--", zorder=1)

        ax.set_xlim(0.0, total_hours)
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
        description="Plot drift figures from drift_test CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "csv", nargs="?", default=None,
        help="Single CSV to plot. If omitted, all files in artifacts/ are processed.",
    )
    p.add_argument(
        "--smooth-min", type=float, default=5.0, metavar="MIN",
        help="Rolling-mean window in minutes (default: 5)",
    )
    p.add_argument(
        "--delta", action="store_true",
        help="Subtract the initial baseline so y=0 at the start — makes drift obvious",
    )
    p.add_argument(
        "--baseline-min", type=float, default=5.0, metavar="MIN",
        help="Minutes at start used to compute the baseline for --delta (default: 5)",
    )
    p.add_argument(
        "--no-raw", action="store_true",
        help="Hide the faint raw trace; show rolling mean only",
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


def plot_one(csv_path: Path, out_dir: Path, args) -> None:
    print(f"\n{csv_path.name}  ({csv_path.stat().st_size // 1024} KB)")
    df = load_csv(csv_path)
    if df.empty:
        return
    print(f"  {len(df)} samples  |  duration: {df['elapsed_h'].max():.2f} h")
    make_figure(
        df=df,
        smooth_min=args.smooth_min,
        delta=args.delta,
        baseline_min=args.baseline_min,
        show_raw=not args.no_raw,
        output_path=out_dir / (csv_path.stem + "_figure.pdf"),
        fig_width=args.width,
        panel_height=args.panel_height,
    )


def main():
    args   = parse_args()
    here   = Path(__file__).parent          # drift_test/
    out_dir = here / "figures"

    if args.csv is not None:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
        plot_one(csv_path, out_dir, args)
    else:
        artifacts_dir = here / "artifacts"
        csvs = sorted(artifacts_dir.glob("drift_*.csv"))
        if not csvs:
            print(f"Error: no drift CSVs found in {artifacts_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(csvs)} file(s) in drift_test/artifacts/")
        for csv_path in csvs:
            plot_one(csv_path, out_dir, args)
        print(f"\nDone. Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
