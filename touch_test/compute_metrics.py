"""
compute_metrics.py
------------------
Computes publication-ready repeatability metrics for the 3D-skin sensor.

Metrics
-------
1. Peak Height Consistency   — Mean ± SD of peak ΔADC for each node (all 30 trials)
2. Rise Time                 — Time from 10 % to 90 % of peak ΔADC (10–90 % rise)
3. Baseline Drift            — SD of signal in the 2 s pre-contact window
4. Contact Localization      — Cross-talk: non-primary node response / primary node peak
5. Inter-Prototype CV        — Coefficient of Variation of per-prototype peak means

Usage
-----
    python touch_test/compute_metrics.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# ── match constants from plot_repeatability.py ────────────────────────────────
CHANNELS    = ["sensor_1", "sensor_5", "sensor_7"]
NODE_NAMES  = ["Node 1",   "Node 2",   "Node 3"]
BUILD_DIRS  = ["sensor_1", "sensor_2", "sensor_3"]
BUILD_NAMES = ["Prototype 1", "Prototype 2", "Prototype 3"]

T_REF_S    = 4.0       # aligned peak time (s)
WINDOW_S   = 32.0      # total window length (s)
PEAK_SMOOTH = 25       # samples for peak-detection smoothing
BASELINE_PCT = 10      # percentile used for baseline subtraction

# Touch peak times (seconds in aligned window) — from plot output
# Node 1 peaks at 4.0 s, Node 2 at ~15.0 s, Node 3 at ~26.5 s
TOUCH_TIMES = [4.0, 15.0, 26.5]   # mean peak times per node

# Half-window around each touch peak used to find the local peak in that channel
SEARCH_HALF = 3.0   # s

DATA_DIR = Path(__file__).parent / "artifacts"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_and_process(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_s_aligned, delta_adc_matrix[3 x n_samples]).
    Baseline-subtracts via 10th-percentile, aligns sensor_1 peak to T_REF_S."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    data = df[CHANNELS].astype(float).values        # (n, 3)
    t_ms = df["elapsed_ms"].astype(float).values

    # baseline subtraction
    for ci in range(3):
        base = np.percentile(data[:, ci], BASELINE_PCT)
        data[:, ci] -= base

    # peak-align on sensor_1
    s1_smooth = uniform_filter1d(data[:, 0], PEAK_SMOOTH)
    peak_idx  = int(np.argmax(s1_smooth))
    peak_ms   = t_ms[peak_idx]

    t_s = (t_ms - (peak_ms - T_REF_S * 1000.0)) / 1000.0
    return t_s, data   # (n,), (n, 3)


def find_peak_in_window(t_s, signal, center_t, half=SEARCH_HALF):
    mask = (t_s >= center_t - half) & (t_s <= center_t + half)
    if mask.sum() < 3:
        return np.nan
    return float(np.max(signal[mask]))


def rise_time_10_90(t_s, signal, center_t, half=SEARCH_HALF):
    """10 %→90 % rise time (s) around a known peak center."""
    mask = (t_s >= center_t - half) & (t_s <= center_t + half)
    if mask.sum() < 3:
        return np.nan
    t_win   = t_s[mask]
    sig_win = signal[mask]

    peak_val = sig_win.max()
    if peak_val <= 0:
        return np.nan

    thr_lo = 0.10 * peak_val
    thr_hi = 0.90 * peak_val

    # find first crossing of 10 % (rising edge, before peak)
    peak_pos = int(np.argmax(sig_win))
    rising   = sig_win[:peak_pos + 1]
    t_rising = t_win[:peak_pos + 1]

    idx_10 = np.where(rising >= thr_lo)[0]
    idx_90 = np.where(rising >= thr_hi)[0]
    if len(idx_10) == 0 or len(idx_90) == 0:
        return np.nan

    t_10 = float(t_rising[idx_10[0]])
    t_90 = float(t_rising[idx_90[0]])
    return t_90 - t_10


def baseline_drift(t_s, signal, pre_contact_end):
    """SD of signal in the 2 s window just before the first contact."""
    mask = (t_s >= pre_contact_end - 2.0) & (t_s < pre_contact_end)
    if mask.sum() < 3:
        return np.nan
    return float(np.std(signal[mask], ddof=1))


# ── load all data ─────────────────────────────────────────────────────────────

# all_trials[build_idx][trial_idx] = (t_s, data[n,3])
all_trials: list[list[tuple]] = []
for bd in BUILD_DIRS:
    bd_path = DATA_DIR / bd
    csvs    = sorted(bd_path.glob("touch_*.csv"))
    trials  = [load_and_process(p) for p in csvs]
    all_trials.append(trials)
    print(f"Loaded {len(trials):2d} trials from {bd}")

n_builds = len(all_trials)
n_nodes  = len(CHANNELS)

print()

# ── METRIC 1 — Peak Height Consistency ───────────────────────────────────────
print("=" * 60)
print("METRIC 1 — Peak ΔADC Height (all 30 trials per node)")
print("=" * 60)

peak_matrix = np.full((n_nodes, n_builds, 10), np.nan)  # [node, build, trial]

for bi, build_trials in enumerate(all_trials):
    for ti, (t_s, data) in enumerate(build_trials):
        for ni in range(n_nodes):
            peak_matrix[ni, bi, ti] = find_peak_in_window(
                t_s, data[:, ni], TOUCH_TIMES[ni]
            )

for ni in range(n_nodes):
    vals = peak_matrix[ni].flatten()
    vals = vals[~np.isnan(vals)]
    print(f"  {NODE_NAMES[ni]}: {np.mean(vals):.1f} ± {np.std(vals, ddof=1):.1f} ADC")

all_vals = peak_matrix.flatten()
all_vals = all_vals[~np.isnan(all_vals)]
print(f"  Overall:  {np.mean(all_vals):.1f} ± {np.std(all_vals, ddof=1):.1f} ADC\n")

# ── METRIC 2 — Rise Time (10–90 %) ───────────────────────────────────────────
print("=" * 60)
print("METRIC 2 — Rise Time 10–90 % (seconds)")
print("=" * 60)

rise_matrix = np.full((n_nodes, n_builds, 10), np.nan)

for bi, build_trials in enumerate(all_trials):
    for ti, (t_s, data) in enumerate(build_trials):
        for ni in range(n_nodes):
            rise_matrix[ni, bi, ti] = rise_time_10_90(
                t_s, data[:, ni], TOUCH_TIMES[ni]
            )

for ni in range(n_nodes):
    vals = rise_matrix[ni].flatten()
    vals = vals[~np.isnan(vals)]
    print(f"  {NODE_NAMES[ni]}: {np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f} s")

all_rise = rise_matrix.flatten()
all_rise = all_rise[~np.isnan(all_rise)]
print(f"  Overall:  {np.mean(all_rise):.2f} ± {np.std(all_rise, ddof=1):.2f} s\n")

# ── METRIC 3 — Baseline Drift ─────────────────────────────────────────────────
print("=" * 60)
print("METRIC 3 — Pre-contact Baseline Drift (SD, ADC)")
print("=" * 60)

# Use 2 s window before first contact (Node 1 at 4.0 s → window 2–4 s)
pre_contact_end = TOUCH_TIMES[0]   # 4.0 s

drift_matrix = np.full((n_nodes, n_builds, 10), np.nan)

for bi, build_trials in enumerate(all_trials):
    for ti, (t_s, data) in enumerate(build_trials):
        for ni in range(n_nodes):
            drift_matrix[ni, bi, ti] = baseline_drift(t_s, data[:, ni], pre_contact_end)

for ni in range(n_nodes):
    vals = drift_matrix[ni].flatten()
    vals = vals[~np.isnan(vals)]
    print(f"  {NODE_NAMES[ni]}: {np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f} ADC")

all_drift = drift_matrix.flatten()
all_drift = all_drift[~np.isnan(all_drift)]
print(f"  Overall:  {np.mean(all_drift):.2f} ± {np.std(all_drift, ddof=1):.2f} ADC\n")

# ── METRIC 4 — Contact Localization / Cross-talk ─────────────────────────────
print("=" * 60)
print("METRIC 4 — Cross-talk (non-primary response / primary peak, %)")
print("=" * 60)

# For each touch event (node ni touched), measure response of all other nodes
crosstalk_vals = []

for ni_primary in range(n_nodes):
    non_primary = [n for n in range(n_nodes) if n != ni_primary]
    ct_per_touch = []

    for bi, build_trials in enumerate(all_trials):
        for ti, (t_s, data) in enumerate(build_trials):
            primary_peak = find_peak_in_window(t_s, data[:, ni_primary], TOUCH_TIMES[ni_primary])
            if np.isnan(primary_peak) or primary_peak <= 0:
                continue
            for ni_other in non_primary:
                other_peak = find_peak_in_window(t_s, data[:, ni_other], TOUCH_TIMES[ni_primary])
                if not np.isnan(other_peak):
                    ct_per_touch.append(max(0.0, other_peak) / primary_peak * 100.0)

    ct_arr = np.array(ct_per_touch)
    crosstalk_vals.extend(ct_per_touch)
    print(f"  When {NODE_NAMES[ni_primary]} touched: cross-talk = "
          f"{np.mean(ct_arr):.1f} ± {np.std(ct_arr, ddof=1):.1f} %")

all_ct = np.array(crosstalk_vals)
print(f"  Overall cross-talk: {np.mean(all_ct):.1f} ± {np.std(all_ct, ddof=1):.1f} %\n")

# ── METRIC 5 — Inter-Prototype CV ────────────────────────────────────────────
print("=" * 60)
print("METRIC 5 — Inter-Prototype Coefficient of Variation (CV = SD/mean, %)")
print("=" * 60)

cv_vals = []
for ni in range(n_nodes):
    # mean peak per prototype
    proto_means = []
    for bi in range(n_builds):
        vals = peak_matrix[ni, bi, :]
        vals = vals[~np.isnan(vals)]
        proto_means.append(np.mean(vals))
    proto_means = np.array(proto_means)
    cv = np.std(proto_means, ddof=1) / np.mean(proto_means) * 100.0
    cv_vals.append(cv)
    print(f"  {NODE_NAMES[ni]}: prototype means = "
          f"{', '.join(f'{v:.1f}' for v in proto_means)} ADC  →  CV = {cv:.1f} %")

print(f"  Mean CV across nodes: {np.mean(cv_vals):.1f} %\n")

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY (copy into paper)")
print("=" * 60)
print(f"  Peak signal strength:  {np.mean(all_vals):.0f} ± {np.std(all_vals, ddof=1):.0f} ADC")
print(f"  Rise time (10–90 %%):  {np.mean(all_rise):.2f} ± {np.std(all_rise, ddof=1):.2f} s")
print(f"  Pre-contact baseline drift: {np.mean(all_drift):.1f} ± {np.std(all_drift, ddof=1):.1f} ADC")
print(f"  Cross-talk (mean):     {np.mean(all_ct):.1f} ± {np.std(all_ct, ddof=1):.1f} %")
print(f"  Inter-prototype CV:    {np.mean(cv_vals):.1f} %")
