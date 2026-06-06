"""
plot_drift_skins.py
-------------------
Compares drift across multiple skin builds. Each subfolder inside
drift_test/artifacts/ is treated as one skin build. The CSVs in each
subfolder are averaged, and all skins are overlaid on a single figure
(one colored mean line + ±1 SD band per skin).

Add a new skin by dropping its drift CSVs into a new subfolder — the
script picks it up automatically next run.

Usage
-----
    python drift_test/plot_drift_skins.py

    python drift_test/plot_drift_skins.py --smooth-min 10
    python drift_test/plot_drift_skins.py --delta     # Δ from each run's own baseline
    python drift_test/plot_drift_skins.py --no-bands  # hide SD bands, show lines only
    python drift_test/plot_drift_skins.py --output my_comparison.pdf
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Style  (matches rest of project)
# ─────────────────────────────────────────────────────────────────────────────

CHANNELS    = ["sensor_1", "sensor_5", "sensor_7"]
NODE_LABELS = ["Node 1", "Node 5", "Node 7"]
PANEL_TAGS  = ["(a)", "(b)", "(c)"]

# Wong (2011) color-blind-safe palette — cycles if there are more than 7 skins
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]

MEAN_LW    = 2.2
BAND_ALPHA = 0.20
TRACE_LW   = 0.7

FONTSIZE_AXIS_LABEL = 14
FONTSIZE_TICK_LABEL = 12
FONTSIZE_LEGEND     = 11
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
    "legend.handlelength":  1.6,
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
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.03,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
}

N_GRID = 1000


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & averaging  (reused logic from plot_drift_average.py)
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


def average_skin(
    dfs: list[pd.DataFrame],
    t_grid: np.ndarray,
    smooth_min: float,
    delta: bool,
    baseline_min: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return (means, stds) for one skin's runs, interpolated onto t_grid."""
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
            interp   = np.interp(t_grid, t_h, smoothed, left=np.nan, right=np.nan)
            all_traces[ch].append(interp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = {ch: np.nanmean(np.vstack(all_traces[ch]), axis=0) for ch in CHANNELS}
        stds  = {
            ch: np.nanstd(np.vstack(all_traces[ch]), axis=0, ddof=1)
                if len(dfs) > 1 else np.zeros(len(t_grid))
            for ch in CHANNELS
        }
    return means, stds


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def auto_ticks(ax, nx: int = 6, ny: int = 4, n_minor: int = 4) -> None:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nx, steps=[1, 2, 5, 10]))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=ny, steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor))


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
    slope_per_h = float(np.polyfit(t_h, v, 1)[0])
    end_mask = t_h >= (t_h[-1] - win_h)
    end_val  = float(v[end_mask].mean()) if end_mask.any() else float(v[-1])
    return slope_per_h, end_val - baseline


def fmt_signed(x: float, decimals: int = 1) -> str:
    return f"{'+' if x >= 0 else ''}{x:.{decimals}f}"


def make_figure(
    t_grid: np.ndarray,
    skin_data: list[dict],   # [{name, means, stds, n_runs, color}, ...]
    delta: bool,
    show_bands: bool,
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

    for ci, (ax, ch, node_label, tag) in enumerate(
        zip(axes, CHANNELS, NODE_LABELS, PANEL_TAGS)
    ):
        for skin in skin_data:
            color     = skin["color"]
            mean_vals = skin["means"][ch]
            std_vals  = skin["stds"][ch]

            if show_bands:
                ax.fill_between(
                    t_grid,
                    mean_vals - std_vals,
                    mean_vals + std_vals,
                    color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2,
                )

            label = skin["name"] if ci == 0 else "_nolegend_"
            ax.plot(t_grid, mean_vals, color=color, lw=MEAN_LW,
                    label=label, zorder=3)

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

    # Legend below all panels, centred
    legend_handles = [
        Line2D([0], [0], color=s["color"], lw=MEAN_LW,
               label=f"{s['name'].replace('_', ' ').title()}  (n={s['n_runs']})")
        for s in skin_data
    ]
    ax_x1 = axes[0].get_position().x1

    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.3, h_pad=0.2, rect=[0.22, 0.07, 1.0, 1.0])

    fig.legend(
        handles=legend_handles,
        loc="lower right",
        ncol=min(len(skin_data), 3),
        fontsize=FONTSIZE_LEGEND,
        bbox_to_anchor=(ax_x1, 0.0),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path.name}")


def make_combined_figure(
    t_grid: np.ndarray,
    skin_data_abs: list[dict],
    skin_data_delta: list[dict],
    show_bands: bool,
    output_path: Path,
    baseline_min: float = 5.0,
    fig_width: float = 9.5,
    panel_height: float = 2.0,
) -> None:
    """3 rows (nodes) × 2 cols: left = absolute ADC, right = Δ from baseline."""

    plt.rcParams.update(RC_BASE)

    n_rows = len(CHANNELS)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(fig_width, panel_height * n_rows),
        sharex=True,
        squeeze=False,
    )

    col_specs = [
        (0, skin_data_abs,   "ADC Count",   False),
        (1, skin_data_delta, "Δ ADC Count", True),
    ]
    col_titles = ["Absolute", "Δ from baseline"]
    tag_iter = iter(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"])

    # Shared y-range per column (same units within a column), across all node rows
    col_ylim = {}
    for col, skin_data, _, _ in col_specs:
        vals = []
        for ch in CHANNELS:
            for skin in skin_data:
                m, s = skin["means"][ch], skin["stds"][ch]
                vals.append(m + s)
                vals.append(m - s)
        flat = np.concatenate(vals)
        flat = flat[np.isfinite(flat)]
        span = (flat.max() - flat.min()) or 1.0
        # extra top headroom for the drift-stat annotation in the delta column
        top = 0.30 * span if col == 1 else 0.10 * span
        col_ylim[col] = (flat.min() - 0.10 * span, flat.max() + top)

    for ri, (ch, node_label) in enumerate(zip(CHANNELS, NODE_LABELS)):
        for col, skin_data, y_label, is_delta in col_specs:
            ax = axes[ri][col]

            for skin in skin_data:
                color     = skin["color"]
                mean_vals = skin["means"][ch]
                std_vals  = skin["stds"][ch]
                if show_bands:
                    ax.fill_between(t_grid, mean_vals - std_vals, mean_vals + std_vals,
                                    color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
                label = skin["name"] if (ri == 0 and col == 0) else "_nolegend_"
                ax.plot(t_grid, mean_vals, color=color, lw=MEAN_LW,
                        label=label, zorder=3)

            if is_delta:
                ax.axhline(0, color="#AAAAAA", lw=0.9, ls="--", zorder=1)

                # Per-skin drift stats (slope ADC/h + net ADC), color-matched, stacked
                lines = []
                for skin in skin_data:
                    slope, net = drift_stats(t_grid, skin["means"][ch], baseline_min)
                    name = skin["name"].replace("_", " ").title()
                    lines.append((skin["color"],
                                  f"{name}: {fmt_signed(slope,1)} ADC/h, {fmt_signed(net,0)} net"))
                for k, (c, txt) in enumerate(lines):
                    ax.text(0.035, 0.93 - k * 0.135, txt,
                            transform=ax.transAxes, va="top", ha="left",
                            fontsize=9, color=c, fontweight="bold", zorder=6)

            ax.set_xlim(0.0, float(t_grid[-1]))
            ax.set_ylim(*col_ylim[col])
            auto_ticks(ax)
            ax.yaxis.grid(True, color="#E8E8E8", lw=0.8, zorder=0)
            ax.set_axisbelow(True)

            if ri == 0:
                ax.set_title(col_titles[col],
                             fontsize=FONTSIZE_AXIS_LABEL, fontweight="bold", pad=8)

            if col == 0:
                ax.set_ylabel(f"{y_label}\n{node_label}",
                              rotation=0, ha="right", va="center",
                              fontsize=FONTSIZE_AXIS_LABEL, labelpad=14)
            else:
                ax.set_ylabel(y_label, fontsize=FONTSIZE_TICK_LABEL)

            ax.text(-0.02, 1.04, next(tag_iter),
                    transform=ax.transAxes,
                    fontsize=FONTSIZE_PANEL_TAG, fontweight="bold",
                    va="bottom", ha="right", clip_on=False)

            if ri == n_rows - 1:
                ax.set_xlabel("Time (hours)", labelpad=4)
            else:
                ax.tick_params(axis="x", which="both", length=0)

    legend_handles = [
        Line2D([0], [0], color=s["color"], lw=MEAN_LW,
               label=f"{s['name'].replace('_', ' ').title()}  (n={s['n_runs']})")
        for s in skin_data_abs
    ]
    caption = (
        f"Δ measured from each run's initial baseline (mean of first {baseline_min:.0f} min).   "
        "ADC/h = drift rate (least-squares slope).   "
        "net = end level − baseline (total drift)."
    )
    fig.text(0.5, 0.015, caption, ha="center", va="bottom",
             fontsize=8.5, color="#444444", fontstyle="italic")

    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(pad=0.4, h_pad=0.4, w_pad=1.2, rect=[0.0, 0.075, 1.0, 1.0])

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(skin_data_abs), 3),
        fontsize=FONTSIZE_LEGEND,
        bbox_to_anchor=(0.5, 0.045),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.8,
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
        description="Compare drift across skin builds (one subfolder = one skin).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--smooth-min", type=float, default=5.0, metavar="MIN",
        help="Rolling-mean window applied before averaging (default: 5)",
    )
    p.add_argument(
        "--delta", action="store_true",
        help="Subtract each run's own initial baseline — y=0 at start of each run",
    )
    p.add_argument(
        "--baseline-min", type=float, default=5.0, metavar="MIN",
        help="Minutes used to compute the per-run baseline for --delta (default: 5)",
    )
    p.add_argument(
        "--no-bands", action="store_true",
        help="Hide the ±1 SD shaded bands; show mean lines only",
    )
    p.add_argument(
        "--side-by-side", action="store_true",
        help="Combined figure: absolute ADC (left) and Δ-from-baseline (right) side by side",
    )
    p.add_argument(
        "--data-dir", "-d", default=None,
        help="Folder whose subfolders are skins (default: drift_test/artifacts/). "
             "e.g. --data-dir artifacts/robot",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output PDF path (default: drift_test/figures/<data-dir-name>_skins[...]_figure.pdf)",
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
    args = parse_args()
    here = Path(__file__).parent  # drift_test/

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            # allow paths relative to drift_test/ or to the cwd
            data_dir = data_dir if data_dir.exists() else here / args.data_dir
    else:
        data_dir = here / "artifacts"

    if not data_dir.exists():
        print(f"Error: folder not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all subfolders that contain at least one drift CSV
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

    # Load all skins and find the global max duration for the shared time grid
    all_dfs_per_skin: list[list[pd.DataFrame]] = []
    max_h = 0.0

    for skin_dir in skin_dirs:
        csvs = sorted(skin_dir.glob("drift_*.csv"))
        dfs  = []
        for path in csvs:
            df = load_csv(path)
            if not df.empty:
                dfs.append(df)
                max_h = max(max_h, float(df["elapsed_h"].max()))
        all_dfs_per_skin.append(dfs)
        n_runs = len(dfs)
        print(f"  {skin_dir.name:<30} {n_runs} run(s)")

    t_grid = np.linspace(0.0, max_h, N_GRID)

    def build_skin_data(delta: bool) -> list[dict]:
        out = []
        for skin_dir, dfs in zip(skin_dirs, all_dfs_per_skin):
            if not dfs:
                continue
            color = PALETTE[len(out) % len(PALETTE)]
            means, stds = average_skin(dfs, t_grid,
                                       smooth_min=args.smooth_min,
                                       delta=delta,
                                       baseline_min=args.baseline_min)
            out.append({
                "name":   skin_dir.name,
                "means":  means,
                "stds":   stds,
                "n_runs": len(dfs),
                "color":  color,
            })
        return out

    # ── Side-by-side combined figure (absolute | delta) ──────────────────────
    if args.side_by_side:
        out_path = (
            Path(args.output).with_suffix(".pdf")
            if args.output
            else here / "figures" / f"{label}_skins_combined_figure.pdf"
        )
        print(f"\nGenerating combined absolute + Δ figure...")
        make_combined_figure(
            t_grid=t_grid,
            skin_data_abs=build_skin_data(delta=False),
            skin_data_delta=build_skin_data(delta=True),
            show_bands=not args.no_bands,
            output_path=out_path,
            baseline_min=args.baseline_min,
            fig_width=args.width if args.width != 7.2 else 9.5,
            panel_height=args.panel_height,
        )
        return

    # ── Single-variant figure ────────────────────────────────────────────────
    skin_data = build_skin_data(delta=args.delta)

    variant = "_delta" if args.delta else ""
    out_path = (
        Path(args.output).with_suffix(".pdf")
        if args.output
        else here / "figures" / f"{label}_skins{variant}_figure.pdf"
    )

    print(f"\nGenerating figure...")
    make_figure(
        t_grid=t_grid,
        skin_data=skin_data,
        delta=args.delta,
        show_bands=not args.no_bands,
        output_path=out_path,
        fig_width=args.width,
        panel_height=args.panel_height,
    )


if __name__ == "__main__":
    main()
