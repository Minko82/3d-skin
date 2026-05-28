"""
plot_repeatability.py
---------------------
Repeatability comparison: 3 sensor builds × 10 trials each (30 total).

Layout  — 3 stacked panels, one per touch node.
          Each panel overlays 3 build mean ± 1 SD bands.

Signal pipeline per trial
--------------------------
  1. Percentile baseline  — 10th-pctile of each channel over the full
                            recording (robust even when recording starts
                            mid-touch or sensor offset drifted between builds).
  2. Peak-based alignment — smooth to find the sensor_1 peak; shift the
                            time axis so that peak sits at T_REF seconds.
                            This removes the artefact of different recording-
                            start offsets across builds while preserving the
                            true inter-touch timing (~10–12 s per touch).
  3. Clip to window       — [0, WINDOW_S] s; t = T_REF is the sensor_1 peak.
  4. Interpolate          — onto 500-point grid; NaN for out-of-range tails.
  5. Mean ± 1 SD          — per build per node across the 10 aligned trials.

Usage
-----
    # Use default data dir (touch_test/artifacts/) and save to touch_test/figures/
    python touch_test/plot_repeatability.py

    python touch_test/plot_repeatability.py --data-dir path/to/other/data
    python touch_test/plot_repeatability.py --smooth 5          # display smoothing
    python touch_test/plot_repeatability.py --show-traces       # faint individual lines
    python touch_test/plot_repeatability.py --shared-y
    python touch_test/plot_repeatability.py --width 1col --panel-height 1.8
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WIDTHS = {"1col": 3.5, "1.5col": 5.5, "2col": 7.2}

# Wong (2011) color-blind-safe palette
BUILD_COLORS  = ["#0072B2", "#D55E00", "#009E73"]
BUILD_LABELS  = ["Prototype 1", "Prototype 2", "Prototype 3"]

CHANNELS    = ["sensor_1", "sensor_5", "sensor_7"]
NODE_LABELS = ["Touch node 1", "Touch node 5", "Touch node 7"]

T_REF_S    = 4.0
WINDOW_S   = 32.0
PEAK_SMOOTH = 25

MEAN_LW     = 2.2
BAND_ALPHA  = 0.28
TRACE_LW    = 0.7
TRACE_ALPHA = 0.25

PANEL_TAGS = ["(a)", "(b)", "(c)"]

FONTSIZE_AXIS_LABEL = 14
FONTSIZE_TICK_LABEL = 12
FONTSIZE_LEGEND     = 12
FONTSIZE_PANEL_TAG  = 16
FONTSIZE_SD_LABEL   = 11

RC_BASE = {
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "font.weight":          "bold",
    "font.size":            FONTSIZE_TICK_LABEL,
    "axes.labelsize":       FONTSIZE_AXIS_LABEL,
    "axes.labelweight":     "bold",
    "axes.titlesize":       FONTSIZE_AXIS_LABEL,
    "xtick.labelsize":      FONTSIZE_TICK_LABEL,
    "ytick.labelsize":      FONTSIZE_TICK_LABEL,
    "legend.fontsize":      FONTSIZE_LEGEND,
    "legend.frameon":       False,
    "legend.handlelength":  1.4,
    "legend.handletextpad": 0.4,
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
    "figure.dpi":           300,
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.03,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & alignment
# ─────────────────────────────────────────────────────────────────────────────

def load_trial(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in ["elapsed_ms"] + CHANNELS:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path.name}")
    return df[["elapsed_ms"] + CHANNELS].astype(float).copy()


def display_smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    w = np.hanning(window)
    w /= w.sum()
    padded = np.pad(series.values, window // 2, mode="edge")
    return pd.Series(np.convolve(padded, w, mode="valid")[: len(series)], index=series.index)


def process_trial(path: Path, display_smooth_w: int,
                  subtract_baseline: bool = False) -> tuple[pd.DataFrame, float]:
    df = load_trial(path)

    s1_smooth = uniform_filter1d(df[CHANNELS[0]].values, PEAK_SMOOTH)
    peak_idx  = int(np.argmax(s1_smooth))
    peak_ms   = float(df["elapsed_ms"].iloc[peak_idx])

    if subtract_baseline:
        for ch in CHANNELS:
            base = float(np.percentile(df[ch], 10))
            df[ch] = df[ch] - base

    window_start_ms = peak_ms - T_REF_S * 1000.0
    df["t_s"] = (df["elapsed_ms"] - window_start_ms) / 1000.0

    if display_smooth_w > 1:
        for ch in CHANNELS:
            df[ch] = display_smooth(df[ch], display_smooth_w)

    return df, peak_ms


def compute_touch_peak_times(all_builds: list[list[pd.DataFrame]]) -> list[float]:
    peak_times: list[list[float]] = [[] for _ in CHANNELS]
    for trials in all_builds:
        for df in trials:
            mask = (df["t_s"] >= 0) & (df["t_s"] <= WINDOW_S)
            sub  = df.loc[mask].reset_index(drop=True)
            if len(sub) < 5:
                continue
            for ci, ch in enumerate(CHANNELS):
                smoothed = uniform_filter1d(sub[ch].values.astype(float), 25)
                idx = int(np.argmax(smoothed))
                peak_times[ci].append(float(sub["t_s"].iloc[idx]))
    return [float(np.mean(pt)) for pt in peak_times]


def compute_mean_std(
    trials: list[pd.DataFrame], channel: str, n_pts: int = 500
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_grid = np.linspace(0.0, WINDOW_S, n_pts)
    traces = []
    for df in trials:
        interp = np.interp(t_grid, df["t_s"].values, df[channel].values,
                           left=np.nan, right=np.nan)
        traces.append(interp)
    mat  = np.vstack(traces)
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0, ddof=1)
    return t_grid, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def auto_ticks(ax, nx=5, ny=4, n_minor=4):
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nx, steps=[1, 2, 5, 10]))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=ny, steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))


def make_figure(
    all_builds: list[list[pd.DataFrame]],
    fig_width: float,
    panel_height: float,
    shared_y: bool,
    show_traces: bool,
    subtract_baseline: bool,
    title: str,
    output_path: Path,
) -> None:

    plt.rcParams.update(RC_BASE)

    n_panels = len(CHANNELS)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(fig_width, panel_height * n_panels),
        sharex=True,
        sharey=shared_y,
    )

    stats: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for bi, trials in enumerate(all_builds):
        for ci, ch in enumerate(CHANNELS):
            stats[(bi, ci)] = compute_mean_std(trials, ch)

    touch_peak_times = compute_touch_peak_times(all_builds)
    print(f"  Mean touch peak times: "
          + "  ".join(f"{NODE_LABELS[i]}={touch_peak_times[i]:.1f}s"
                      for i in range(n_panels)))

    print("\n  σ at peak (sample SD, ddof=1):")
    for ci, node in enumerate(NODE_LABELS):
        for bi, blabel in enumerate(BUILD_LABELS):
            _, m, s = stats[(bi, ci)]
            print(f"    {node:16s}  {blabel}: σ = {s[int(np.nanargmax(m))]:.2f}")

    panel_extents: list[tuple[float, float]] = []
    for ci in range(n_panels):
        lo = 0.0
        hi = max(np.nanmax(stats[(bi, ci)][1] + stats[(bi, ci)][2])
                 for bi in range(len(all_builds)))
        panel_extents.append((lo, hi))

    y_label_prefix = "Δ ADC" if subtract_baseline else "ADC Count"

    if shared_y:
        g_lo = min(e[0] for e in panel_extents)
        g_hi = max(e[1] for e in panel_extents)
        panel_extents = [(g_lo, g_hi)] * n_panels

    for ci, (ax, node_label, tag) in enumerate(zip(axes, NODE_LABELS, PANEL_TAGS)):
        p_lo, p_hi = panel_extents[ci]
        p_pad = (p_hi - p_lo) * 0.07

        for bi, (trials, color, blabel) in enumerate(
            zip(all_builds, BUILD_COLORS, BUILD_LABELS)
        ):
            t_grid, mean, std = stats[(bi, ci)]

            if show_traces:
                for df in trials:
                    mask = (df["t_s"] >= 0) & (df["t_s"] <= WINDOW_S)
                    ax.plot(df.loc[mask, "t_s"], df.loc[mask, CHANNELS[ci]],
                            color=color, lw=TRACE_LW, alpha=TRACE_ALPHA, zorder=2)

            ax.fill_between(t_grid, np.maximum(0, mean - std), mean + std,
                            color=color, alpha=BAND_ALPHA, linewidth=0, zorder=3)

            legend_label = blabel if ci == 0 else "_nolegend_"
            ax.plot(t_grid, mean,
                    color=color, lw=MEAN_LW, alpha=1.0,
                    label=legend_label, zorder=4)

        ax.set_xlim(0.0, WINDOW_S)
        ax.set_ylim(p_lo, p_hi + p_pad)
        auto_ticks(ax)
        ax.yaxis.grid(True, color="#E8E8E8", lw=0.8, zorder=0)
        ax.set_axisbelow(True)

        ax.set_ylabel(f"{y_label_prefix}\nNode {ci + 1}",
                      rotation=0, ha="left", va="center",
                      fontsize=FONTSIZE_AXIS_LABEL)
        ax.yaxis.set_label_coords(-0.36, 0.5)

        ax.text(-0.36, 1.04, tag,
                transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL_TAG, fontweight="bold",
                va="bottom", ha="left", clip_on=False)

        if ci == n_panels - 1:
            ax.set_xlabel("Time (s)", labelpad=4)
        else:
            ax.tick_params(axis="x", which="both", length=0)

    legend_handles = [
        Line2D([0], [0], color=c, lw=MEAN_LW, label=lbl)
        for c, lbl in zip(BUILD_COLORS, BUILD_LABELS)
    ]

    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.3, h_pad=0.2, rect=[0.22, 0.07, 1, 0.93])

    ax_x1 = axes[0].get_position().x1
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        ncol=len(BUILD_LABELS),
        fontsize=FONTSIZE_LEGEND,
        bbox_to_anchor=(ax_x1, 0.0),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=2.0,
    )

    touch_labels = ["Node 1 Contact", "Node 2 Contact", "Node 3 Contact"]
    y_top_fig = axes[0].get_position().y1
    y_bot_fig = axes[-1].get_position().y0

    for t_touch, label in zip(touch_peak_times, touch_labels):
        x_px  = axes[0].transData.transform_point((t_touch, 0))[0]
        x_fig = fig.transFigure.inverted().transform_point((x_px, 0))[0]

        fig.add_artist(Line2D(
            [x_fig, x_fig], [y_bot_fig, y_top_fig],
            transform=fig.transFigure,
            color="#AAAAAA", lw=1.2, ls=":",
            clip_on=False, zorder=10,
        ))

        fig.text(x_fig, y_top_fig + 0.01, label,
                 transform=fig.transFigure,
                 fontsize=FONTSIZE_TICK_LABEL - 1,
                 fontweight="bold",
                 ha="center", va="bottom",
                 color="#222222",
                 bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Repeatability: mean ± 1 SD per build, peak-aligned, baseline-corrected.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Root dir with sensor_1/, sensor_2/, sensor_3/ subdirs. "
             "Default: touch_test/artifacts/"
    )
    p.add_argument("--output", "-o", default=None,
                   help="Output PDF stem (default: repeatability_figure). "
                        "Saved to touch_test/figures/.")
    p.add_argument("--smooth", type=int, default=1, metavar="WINDOW",
                   help="Display smoothing (Hann window, samples). 1 = off. Default: 1")
    p.add_argument("--width", choices=list(WIDTHS.keys()), default="2col",
                   help="Figure width preset. Default: 2col")
    p.add_argument("--panel-height", type=float, default=2.0,
                   help="Height per panel (inches). Default: 2.0")
    p.add_argument("--shared-y", action="store_true",
                   help="Force identical y-axis scale across all panels")
    p.add_argument("--show-traces", action="store_true",
                   help="Overlay individual trial traces (faint) behind mean band")
    p.add_argument("--baseline", action="store_true",
                   help="Subtract per-channel 10th-pctile baseline; y-axis shows Δ ADC")
    p.add_argument("--title", default=None,
                   help="Override the figure suptitle")
    return p.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).parent  # touch_test/

    data_root = Path(args.data_dir) if args.data_dir else here / "artifacts"
    if not data_root.exists():
        print(f"Error: data directory not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    build_dirs = [data_root / "sensor_1", data_root / "sensor_2", data_root / "sensor_3"]
    for bd in build_dirs:
        if not bd.exists():
            print(f"Error: {bd} not found", file=sys.stderr)
            sys.exit(1)

    all_builds: list[list[pd.DataFrame]] = []
    for bd, label in zip(build_dirs, BUILD_LABELS):
        print(f"\nLoading {label}  ({bd.name})")
        trials = []
        for f in sorted(bd.glob("*.csv")):
            df, peak_ms = process_trial(f, args.smooth, args.baseline)
            trials.append(df)
            print(f"    {f.name}  sensor_1 peak at {peak_ms/1000:.2f}s  "
                  f"→ aligned window 0–{WINDOW_S:.0f}s")
        all_builds.append(trials)

    out_stem   = args.output or "repeatability_figure"
    out_dir    = here / "figures"
    output_path = (out_dir / out_stem).with_suffix(".pdf")

    if args.title:
        title = args.title
    elif args.baseline:
        title = (
            "Repeatability of baseline-subtracted tactile sensor response "
            "across three fabricated prototypes (n = 10 trials per prototype, mean ± 1 s.d.)"
        )
    else:
        title = (
            "Repeatability of raw capacitive sensor output "
            "across three fabricated prototypes (n = 10 trials per prototype, mean ± 1 s.d.)"
        )

    make_figure(
        all_builds=all_builds,
        fig_width=WIDTHS[args.width],
        panel_height=args.panel_height,
        shared_y=True,
        show_traces=args.show_traces,
        subtract_baseline=args.baseline,
        title=title,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
