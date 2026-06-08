"""
Mag_Visualizer.py
-----------------
Real-time touch visualiser for the MLX90393 magnetometer skin.
Reads serial output from Magnetic_Sensor_Arduino.ino (text format).

Supports both single-sensor and mux (up to 8 sensors) modes.
Baseline is averaged from the first BASELINE_SAMPLES valid packets.
Touch is detected when the magnetic field deviation magnitude exceeds --threshold.

Usage:
    python Demo/Mag_Visualizer.py
    python Demo/Mag_Visualizer.py --port /dev/cu.usbmodem1101
    python Demo/Mag_Visualizer.py --sensors 3
    python Demo/Mag_Visualizer.py --threshold 400 --window 15

Keyboard:
    R — recalibrate baseline
"""

import argparse
import collections
import glob
import math
import re
import sys
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial not installed.  Run:  pip install pyserial")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation
except ImportError:
    print("ERROR: matplotlib not installed.  Run:  pip install matplotlib")
    sys.exit(1)


# ─── constants ────────────────────────────────────────────────────────────────

BAUD_DEFAULT      = 115200
WINDOW_S          = 10
HISTORY_PTS       = 600
FPS               = 20
BASELINE_SAMPLES  = 30      # packets averaged to lock baseline
TOUCH_THRESHOLD   = 500.0   # default |Δ| magnitude for touch detection
TOUCH_FLASH_TICKS = 6       # frames the "flash" stays on after touch

# Saturation cap on the magnitude bar (bar full = 2× threshold)
BAR_FULL_SCALE    = 2.0

SENSOR_COLORS = [
    "#EF5350", "#AB47BC", "#26C6DA", "#66BB6A",
    "#FFA726", "#EC407A", "#7E57C2", "#26A69A",
]
AXIS_COLORS = {"ΔX": "#EF5350", "ΔY": "#66BB6A", "ΔZ": "#42A5F5"}

# MLX90393 invalid-boot sentinel values (matches firmware filter)
INV_XY = 78000.0
INV_Z  = 126000.0

# Serial text patterns
RE_MULTI  = re.compile(r'CH(\d+):\s+X:([\d.\-]+)\s+Y:([\d.\-]+)\s+Z:([\d.\-]+)')
RE_SINGLE = re.compile(r'X:\s*([\d.\-]+)\s+Y:\s*([\d.\-]+)\s+Z:\s*([\d.\-]+)')


# ─── serial parsing ───────────────────────────────────────────────────────────

def parse_line(line):
    """Return list of (channel_id, x, y, z) from one serial line, filtering invalids."""
    if "[INVALID" in line:
        return []
    matches = RE_MULTI.findall(line)
    if matches:
        out = []
        for m in matches:
            ch, x, y, z = int(m[0]), float(m[1]), float(m[2]), float(m[3])
            if x < INV_XY and y < INV_XY and z < INV_Z:
                out.append((ch, x, y, z))
        return out
    m = RE_SINGLE.search(line)
    if m:
        x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if x < INV_XY and y < INV_XY and z < INV_Z:
            return [(0, x, y, z)]
    return []


# ─── port auto-detection ──────────────────────────────────────────────────────

def find_port():
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        mfr  = (p.manufacturer or "").lower()
        if any(kw in desc or kw in mfr for kw in
               ("cp210", "ch340", "ch9102", "ftdi", "esp",
                "usbserial", "acm", "usbmodem")):
            return p.device
    for pattern in (
        "/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*",
        "/dev/tty.usbmodem*", "/dev/tty.usbserial*",
        "/dev/ttyACM*", "/dev/ttyUSB*",
    ):
        m = sorted(glob.glob(pattern))
        if m:
            return m[0]
    return None


# ─── touch panel ─────────────────────────────────────────────────────────────

def draw_touch_panel(ax, sensor_idx, label, color,
                     dx, dy, dz, baseline_locked, calibrating_count,
                     threshold, flash_remaining):
    ax.cla()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── not yet calibrated ──
    if not baseline_locked:
        ax.set_facecolor("#141414")
        ax.patch.set_alpha(1.0)
        ax.text(0.5, 0.85, label, ha="center", va="top",
                transform=ax.transAxes, fontsize=11, fontweight="bold", color=color)
        pct = min(calibrating_count / BASELINE_SAMPLES, 1.0)
        ax.text(0.5, 0.58, "Calibrating…", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="#AAAAAA")
        bar_y, bar_h = 0.40, 0.06
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.10, bar_y), 0.80, bar_h, boxstyle="round,pad=0.005",
            facecolor="#2A2A2A", edgecolor="none", transform=ax.transAxes, clip_on=False))
        if pct > 0:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.10, bar_y), 0.80 * pct, bar_h, boxstyle="round,pad=0.005",
                facecolor="#FFA726", edgecolor="none", alpha=0.85,
                transform=ax.transAxes, clip_on=False))
        ax.text(0.5, bar_y - 0.07, f"{int(pct * 100)} %", ha="center", va="top",
                transform=ax.transAxes, fontsize=9, color="#AAAAAA")
        return

    mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    is_touch = mag >= threshold
    active   = is_touch or flash_remaining > 0
    fill_frac = min(mag / (threshold * BAR_FULL_SCALE), 1.0)

    if active:
        bg_color  = "#FF5252"
        bg_alpha  = 0.18
        ind_char  = "●"
        ind_text  = "TOUCH"
        ind_color = "#FF5252"
        bar_color = "#FF5252"
    else:
        bg_color  = "#141414"
        bg_alpha  = 1.0
        ind_char  = "○"
        ind_text  = "NO TOUCH"
        ind_color = "#555555"
        bar_color = "#26C6DA"

    ax.set_facecolor(bg_color)
    ax.patch.set_alpha(bg_alpha)

    # ── sensor label ──
    ax.text(0.5, 0.96, label, ha="center", va="top",
            transform=ax.transAxes, fontsize=11, fontweight="bold", color=color)

    # ── touch indicator ──
    ax.text(0.5, 0.80, f"{ind_char} {ind_text}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=15, fontweight="bold", color=ind_color)

    # ── magnitude ──
    ax.text(0.5, 0.66, f"|Δ| {mag:6.0f}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=16, fontweight="bold", color=ind_color if active else "#CCCCCC",
            fontfamily="monospace")

    # ── axis breakdown ──
    for j, (name, val) in enumerate([("ΔX", dx), ("ΔY", dy), ("ΔZ", dz)]):
        y_pos = 0.51 - j * 0.095
        sign  = "+" if val >= 0 else ""
        ax.text(0.30, y_pos, name, ha="right", va="center",
                transform=ax.transAxes, fontsize=9, color=AXIS_COLORS[name])
        ax.text(0.33, y_pos, f"{sign}{val:7.0f}", ha="left", va="center",
                transform=ax.transAxes, fontsize=9, color="#CCCCCC",
                fontfamily="monospace")

    # ── dominant axis badge ──
    max_axis = max(("ΔX", abs(dx)), ("ΔY", abs(dy)), ("ΔZ", abs(dz)), key=lambda t: t[1])
    ax.text(0.82, 0.54, max_axis[0], ha="center", va="center",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color=AXIS_COLORS[max_axis[0]], alpha=0.75)

    # ── magnitude bar ──
    bar_y, bar_h = 0.13, 0.055
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, bar_y), 0.90, bar_h, boxstyle="round,pad=0.005",
        facecolor="#2A2A2A", edgecolor="none", transform=ax.transAxes, clip_on=False))
    if fill_frac > 0.01:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, bar_y), 0.90 * fill_frac, bar_h, boxstyle="round,pad=0.005",
            facecolor=bar_color, edgecolor="none", alpha=0.90,
            transform=ax.transAxes, clip_on=False))

    # Threshold marker on bar (at the 50% mark = threshold / 2×threshold)
    thresh_bar_x = 0.05 + 0.90 * 0.5
    ax.plot([thresh_bar_x, thresh_bar_x], [bar_y - 0.01, bar_y + bar_h + 0.01],
            color="#FFFFFF", linewidth=1.5, transform=ax.transAxes,
            clip_on=False, alpha=0.55)
    ax.text(thresh_bar_x, bar_y - 0.04, "touch", ha="center", va="top",
            transform=ax.transAxes, fontsize=7, color="#888888")

    ax.text(0.05, bar_y - 0.04, "0", ha="left", va="top",
            transform=ax.transAxes, fontsize=7, color="#555555")
    ax.text(0.95, bar_y - 0.04, "max", ha="right", va="top",
            transform=ax.transAxes, fontsize=7, color="#555555")


# ─── build figure ─────────────────────────────────────────────────────────────

def build_figure(n_sensors, window_s, threshold):
    plt.style.use("dark_background")
    n = max(n_sensors, 1)
    fig_w = max(12, 4.5 * n)
    fig = plt.figure(figsize=(fig_w, 8.5), facecolor="#0D0D0D")
    fig.canvas.manager.set_window_title("Magnetic Skin – Touch Monitor")

    gs = gridspec.GridSpec(2, n, figure=fig,
                           left=0.07, right=0.98, top=0.90, bottom=0.08,
                           hspace=0.45, wspace=0.28,
                           height_ratios=[1.4, 1.0])

    fig.text(0.5, 0.955,
             f"Magnetic Skin  ·  Touch Monitor  (MLX90393 × {n})",
             ha="center", va="center", fontsize=16, fontweight="bold", color="white")
    fig.text(0.98, 0.955, "R = recalibrate",
             ha="right", va="center", fontsize=9, color="#555555")

    # ── magnitude time-series (top row, spans all columns) ──
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.set_facecolor("#1A1A1A")
    ax_ts.set_xlim(0, window_s)
    ax_ts.set_ylim(0, threshold * BAR_FULL_SCALE)
    ax_ts.set_xlabel("Time (s)", color="#AAAAAA", fontsize=10)
    ax_ts.set_ylabel("|Δ| Magnitude", color="#AAAAAA", fontsize=10)
    ax_ts.tick_params(colors="#AAAAAA", labelsize=9)
    for sp in ax_ts.spines.values():
        sp.set_edgecolor("#333333")
    ax_ts.grid(True, color="#2A2A2A", linewidth=0.5)

    # Touch threshold line
    ax_ts.axhline(threshold, color="#FF5252", linewidth=1.4,
                  linestyle="--", alpha=0.7, zorder=2)
    ax_ts.axhspan(threshold, threshold * BAR_FULL_SCALE,
                  alpha=0.06, color="#FF5252", zorder=0)
    ax_ts.text(window_s * 0.995, threshold * 1.03,
               "TOUCH THRESHOLD", ha="right", va="bottom",
               fontsize=8, color="#FF5252", alpha=0.7)

    ts_lines = []
    val_annotations = []
    for i in range(n):
        color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
        label = f"S{i}"
        line, = ax_ts.plot([], [], color=color, linewidth=2.0,
                           label=label, zorder=3)
        ts_lines.append(line)
        ann = ax_ts.annotate(
            "", xy=(window_s, 0), xytext=(4, 0),
            textcoords="offset points",
            fontsize=9, fontweight="bold", color=color,
            fontfamily="monospace", annotation_clip=False,
        )
        val_annotations.append(ann)

    patches = [mpatches.Patch(color=SENSOR_COLORS[i % len(SENSOR_COLORS)],
                               label=f"Sensor {i}") for i in range(n)]
    ax_ts.legend(handles=patches, loc="upper left", framealpha=0.3, fontsize=9)

    # ── touch panels (bottom row) ──
    touch_axes = [fig.add_subplot(gs[1, i]) for i in range(n)]
    for i, (ax, col) in enumerate(zip(touch_axes, SENSOR_COLORS)):
        draw_touch_panel(ax, i, f"Sensor {i}", col,
                         0, 0, 0, False, 0, threshold, 0)

    return fig, ax_ts, ts_lines, val_annotations, touch_axes


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",      "-p", default=None)
    p.add_argument("--baud",      "-b", type=int,   default=BAUD_DEFAULT)
    p.add_argument("--window",    "-w", type=float, default=WINDOW_S)
    p.add_argument("--sensors",   "-n", type=int,   default=3,
                   help="Expected sensor count (layout hint). Auto-adjusts if different data arrives.")
    p.add_argument("--threshold", "-t", type=float, default=TOUCH_THRESHOLD,
                   help="Magnitude threshold for touch detection.")
    return p.parse_args()


def main():
    args = parse_args()

    port = args.port or find_port()
    if port is None:
        print("ERROR: no serial port found. Plug in the board or use --port.")
        sys.exit(1)

    print(f"Opening {port} @ {args.baud} baud …")
    try:
        ser = serial.Serial(port, args.baud, timeout=0)
        time.sleep(1.5)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print("Serial open. Starting visualiser…")
    print(f"Touch threshold: |Δ| ≥ {args.threshold:.0f}")
    print("Waiting for first valid packet to detect sensor count…")

    n            = args.sensors
    window_s     = args.window
    threshold    = args.threshold

    # Per-sensor state (keyed by channel id, mapped to index 0..n-1)
    ch_to_idx    = {}            # maps hardware channel → panel index
    idx_order    = []            # insertion-order channel list

    # Calibration buffers
    cal_buf      = [[] for _ in range(8)]   # raw (x, y, z) samples per channel
    baselines    = [None] * 8               # (bx, by, bz) once locked

    # Runtime state
    times        = collections.deque(maxlen=HISTORY_PTS)
    mag_hist     = [collections.deque(maxlen=HISTORY_PTS) for _ in range(8)]
    latest_delta = [(0.0, 0.0, 0.0)] * 8   # last (dx, dy, dz)
    flash_count  = [0] * 8                  # frames remaining in touch-flash
    start        = [None]
    buf          = [""]

    fig, ax_ts, ts_lines, val_anns, touch_axes = build_figure(n, window_s, threshold)

    def recalibrate():
        for i in range(8):
            cal_buf[i].clear()
            baselines[i] = None
        print("Baseline cleared — recalibrating…")

    # 'R' key recalibrates
    def on_key(event):
        if event.key and event.key.lower() == 'r':
            recalibrate()

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(_frame):
        nonlocal n, fig, ax_ts, ts_lines, val_anns, touch_axes

        # ── drain serial ──
        try:
            raw = ser.read(ser.in_waiting or 1)
        except serial.SerialException:
            return

        if raw:
            buf[0] += raw.decode("utf-8", errors="replace")
            while "\n" in buf[0]:
                line, buf[0] = buf[0].split("\n", 1)
                readings = parse_line(line.strip())
                if not readings:
                    continue

                now = time.monotonic()
                if start[0] is None:
                    start[0] = now
                t = now - start[0]

                for (ch, x, y, z) in readings:
                    # Assign panel index on first sight
                    if ch not in ch_to_idx:
                        idx = len(idx_order)
                        if idx >= n:
                            continue   # more sensors than expected, skip
                        ch_to_idx[ch] = idx
                        idx_order.append(ch)

                    idx = ch_to_idx[ch]

                    # Calibration accumulation
                    if baselines[idx] is None:
                        cal_buf[idx].append((x, y, z))
                        if len(cal_buf[idx]) >= BASELINE_SAMPLES:
                            bx = sum(s[0] for s in cal_buf[idx]) / len(cal_buf[idx])
                            by = sum(s[1] for s in cal_buf[idx]) / len(cal_buf[idx])
                            bz = sum(s[2] for s in cal_buf[idx]) / len(cal_buf[idx])
                            baselines[idx] = (bx, by, bz)
                            print(f"Sensor {idx} (CH{ch}) baseline locked "
                                  f"→ X:{bx:.1f}  Y:{by:.1f}  Z:{bz:.1f}")
                        continue

                    bx, by, bz = baselines[idx]
                    dx, dy, dz = x - bx, y - by, z - bz
                    latest_delta[idx] = (dx, dy, dz)
                    mag = math.sqrt(dx*dx + dy*dy + dz*dz)

                    if mag >= threshold:
                        flash_count[idx] = TOUCH_FLASH_TICKS

                    times.append(t)
                    mag_hist[idx].append((t, mag))

        if not times:
            return

        now_t = times[-1] if times else 0.0
        t_min = max(0.0, now_t - window_s)

        # ── time-series ──
        for i in range(n):
            pts = [(t - t_min, m) for (t, m) in mag_hist[i] if t >= t_min]
            if pts:
                ft, fm = zip(*pts)
                ts_lines[i].set_data(ft, fm)
                val_anns[i].set_text(f" {fm[-1]:.0f}")
                val_anns[i].xy = (ft[-1], fm[-1])
                val_anns[i].set_visible(True)
            else:
                ts_lines[i].set_data([], [])
                val_anns[i].set_visible(False)

        ax_ts.set_xlim(0, window_s)
        ax_ts.set_ylim(0, threshold * BAR_FULL_SCALE)

        # ── touch panels ──
        for i in range(n):
            dx, dy, dz = latest_delta[i]
            if flash_count[i] > 0:
                flash_count[i] -= 1
            draw_touch_panel(
                touch_axes[i], i, f"Sensor {i}",
                SENSOR_COLORS[i % len(SENSOR_COLORS)],
                dx, dy, dz,
                baseline_locked=baselines[i] is not None,
                calibrating_count=len(cal_buf[i]),
                threshold=threshold,
                flash_remaining=flash_count[i],
            )

    ani = FuncAnimation(fig, update, interval=1000 // FPS,
                        blit=False, cache_frame_data=False)   # noqa: F841

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    ser.close()
    print("Done.")


if __name__ == "__main__":
    main()
