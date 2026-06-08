"""
TOF_Visualizer.py
-----------------
Real-time visualiser for the TOF skin — 3 × VL53L5CX sensors.
Reads dist0,dist1,dist2 lines from serial (flash tof_skin/tof_serial.ino first).

Usage:
    python Demo/TOF_Visualizer.py
    python Demo/TOF_Visualizer.py --port /dev/cu.usbmodem11101
    python Demo/TOF_Visualizer.py --baud 9600
    python Demo/TOF_Visualizer.py --window 15 --max-dist 2000
"""

import argparse
import collections
import glob
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
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("ERROR: matplotlib not installed.  Run:  pip install matplotlib")
    sys.exit(1)


# ─── constants ────────────────────────────────────────────────────────────────

SENSOR_LABELS   = ["Sensor 0", "Sensor 1", "Sensor 2"]
SENSOR_COLORS   = ["#EF5350", "#AB47BC", "#26C6DA"]

BAUD_DEFAULT    = 115200
WINDOW_S        = 10
HISTORY_PTS     = 2000
FPS             = 30
MAX_DIST_MM     = 4000
MIN_DIST_MM     = 20
ERROR_VAL       = 65535

COLOR_TOO_CLOSE = "#FF1744"
COLOR_NO_READ   = "#444444"

# Proximity zones (mm)
ZONE_CLOSE_MM   = 300
ZONE_MED_MM     = 800

# How many mm change counts as movement (filters noise)
TREND_THRESHOLD = 20

DIST_CMAP = LinearSegmentedColormap.from_list(
    "dist", ["#26C6DA", "#66BB6A", "#FFA726", "#EF5350"]
)


def dist_color(dist_mm, max_mm):
    frac = 1.0 - min(dist_mm / max_mm, 1.0)
    r, g, b, _ = DIST_CMAP(frac)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


def classify(v, max_mm):
    if v is None or v >= ERROR_VAL:
        return "no_reading", None
    if v < MIN_DIST_MM:
        return "too_close", None
    if v > max_mm:
        return "no_reading", None
    return "valid", int(v)


def zone_info(dist_mm):
    if dist_mm < ZONE_CLOSE_MM:
        return "CLOSE",  "#FF5252"
    if dist_mm < ZONE_MED_MM:
        return "MEDIUM", "#FFA726"
    return   "FAR",    "#66BB6A"


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


# ─── readout panel ────────────────────────────────────────────────────────────

def draw_readout(ax, dist_mm, prev_mm, max_mm, sensor_color, label):
    ax.cla()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    state, val   = classify(dist_mm, max_mm)
    _,    prev_v = classify(prev_mm, max_mm) if prev_mm is not None else ("no_reading", None)

    if state == "valid":
        fill_frac  = 1.0 - val / max_mm
        bar_color  = dist_color(val, max_mm)
        value_str  = str(val)
        unit_str   = "mm"
        label_color = bar_color
        zone, zone_color = zone_info(val)

        # Trend direction
        if prev_v is not None:
            delta = val - prev_v
            if delta < -TREND_THRESHOLD:
                arrow, trend_text, trend_color = "↓", "APPROACHING", "#FF5252"
            elif delta > TREND_THRESHOLD:
                arrow, trend_text, trend_color = "↑", "RECEDING",    "#66BB6A"
            else:
                arrow, trend_text, trend_color = "→", "STABLE",      "#AAAAAA"
        else:
            arrow, trend_text, trend_color = "→", "—", "#AAAAAA"

        # Background tint from zone color
        bg_alpha = 0.06 + 0.12 * fill_frac
        ax.set_facecolor(zone_color)
        ax.patch.set_alpha(bg_alpha)

    elif state == "too_close":
        fill_frac   = 1.0
        bar_color   = COLOR_TOO_CLOSE
        value_str   = "TOO"
        unit_str    = "CLOSE"
        label_color = COLOR_TOO_CLOSE
        zone, zone_color = "CLOSE", COLOR_TOO_CLOSE
        arrow, trend_text, trend_color = "!", "TOO CLOSE", COLOR_TOO_CLOSE
        ax.set_facecolor(COLOR_TOO_CLOSE)
        ax.patch.set_alpha(0.18)

    else:
        fill_frac   = 0.0
        bar_color   = COLOR_NO_READ
        value_str   = "—"
        unit_str    = ""
        label_color = "#666666"
        zone, zone_color = "", "#333333"
        arrow, trend_text, trend_color = "", "", "#444444"
        ax.set_facecolor("#141414")
        ax.patch.set_alpha(1.0)

    # ── sensor name ──
    ax.text(0.5, 0.96, label, ha="center", va="top", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=sensor_color)

    # ── trend arrow (left of number) ──
    if state == "valid":
        ax.text(0.14, 0.66, arrow, ha="center", va="center", transform=ax.transAxes,
                fontsize=26, fontweight="bold", color=trend_color)

    # ── big distance number ──
    ax.text(0.55, 0.66, value_str, ha="center", va="center", transform=ax.transAxes,
            fontsize=42, fontweight="bold", color=label_color, fontfamily="monospace")
    ax.text(0.55, 0.46, unit_str, ha="center", va="center", transform=ax.transAxes,
            fontsize=13, color=label_color, fontfamily="monospace")

    # ── trend + zone labels ──
    if state == "valid":
        ax.text(0.5, 0.35, trend_text, ha="center", va="center", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=trend_color, fontfamily="monospace")
        ax.text(0.5, 0.27, f"[ {zone} ]", ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=zone_color, fontfamily="monospace")

    # ── proximity bar ──
    bar_y, bar_h = 0.13, 0.055
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, bar_y), 0.90, bar_h, boxstyle="round,pad=0.005",
        facecolor="#2A2A2A", edgecolor="none", transform=ax.transAxes, clip_on=False))
    if fill_frac > 0.01:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, bar_y), 0.90 * fill_frac, bar_h, boxstyle="round,pad=0.005",
            facecolor=bar_color, edgecolor="none", alpha=0.90,
            transform=ax.transAxes, clip_on=False))

    ax.text(0.05, bar_y - 0.04, "CLOSE", ha="left", va="top",
            transform=ax.transAxes, fontsize=7, color="#EF5350")
    ax.text(0.95, bar_y - 0.04, f"FAR  {max_mm} mm", ha="right", va="top",
            transform=ax.transAxes, fontsize=7, color="#26C6DA")


# ─── build figure ─────────────────────────────────────────────────────────────

def build_figure(window_s, max_mm):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 8), facecolor="#0D0D0D")
    fig.canvas.manager.set_window_title("TOF Skin – Proximity Monitor")

    # 2 rows: chart (taller) + readout panels
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.07, right=0.98, top=0.90, bottom=0.09,
                           hspace=0.50, wspace=0.25,
                           height_ratios=[1.6, 1.0])

    fig.text(0.5, 0.955, "TOF Skin  ·  Proximity Monitor  (VL53L5CX × 3)",
             ha="center", va="center", fontsize=16, fontweight="bold", color="white")

    # ── time-series chart (full top row) ──
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.set_facecolor("#1A1A1A")
    ax_ts.set_xlim(0, window_s)
    ax_ts.set_ylim(0, max_mm)
    ax_ts.set_xlabel("Time (s)", color="#AAAAAA", fontsize=10)
    ax_ts.set_ylabel("Distance (mm)", color="#AAAAAA", fontsize=10)
    ax_ts.tick_params(colors="#AAAAAA", labelsize=9)
    for sp in ax_ts.spines.values():
        sp.set_edgecolor("#333333")
    ax_ts.grid(True, color="#2A2A2A", linewidth=0.5)

    # Zone bands
    ax_ts.axhspan(MIN_DIST_MM, ZONE_CLOSE_MM,
                  alpha=0.07, color="#FF5252", zorder=0)
    ax_ts.axhspan(ZONE_CLOSE_MM, ZONE_MED_MM,
                  alpha=0.05, color="#FFA726", zorder=0)
    ax_ts.axhspan(ZONE_MED_MM, max_mm,
                  alpha=0.03, color="#66BB6A", zorder=0)
    ax_ts.axhspan(0, MIN_DIST_MM,
                  alpha=0.12, color=COLOR_TOO_CLOSE, zorder=0)

    # Zone labels on right margin
    label_x = window_s * 0.995
    ax_ts.text(label_x, MIN_DIST_MM * 0.5, "TOO\nCLOSE",
               color=COLOR_TOO_CLOSE, fontsize=7, alpha=0.70, va="center", ha="right")
    ax_ts.text(label_x, (MIN_DIST_MM + ZONE_CLOSE_MM) / 2, "CLOSE",
               color="#FF5252", fontsize=7, alpha=0.70, va="center", ha="right")
    ax_ts.text(label_x, (ZONE_CLOSE_MM + ZONE_MED_MM) / 2, "MEDIUM",
               color="#FFA726", fontsize=7, alpha=0.70, va="center", ha="right")
    ax_ts.text(label_x, (ZONE_MED_MM + max_mm) / 2, "FAR",
               color="#66BB6A", fontsize=7, alpha=0.70, va="center", ha="right")

    # Dashed zone boundary lines
    for y_val in (MIN_DIST_MM, ZONE_CLOSE_MM, ZONE_MED_MM):
        ax_ts.axhline(y_val, color="#444444", linewidth=0.8, linestyle="--", zorder=1)

    ts_lines = []
    for color, label in zip(SENSOR_COLORS, SENSOR_LABELS):
        line, = ax_ts.plot([], [], color=color, linewidth=2.0, label=label, zorder=3)
        ts_lines.append(line)
    patches = [mpatches.Patch(color=SENSOR_COLORS[i], label=SENSOR_LABELS[i]) for i in range(3)]
    ax_ts.legend(handles=patches, loc="upper left", framealpha=0.3, fontsize=9)

    # Live value annotations at the right edge of each line
    val_annotations = []
    for color in SENSOR_COLORS:
        ann = ax_ts.annotate(
            "", xy=(window_s, 0), xytext=(4, 0),
            textcoords="offset points",
            fontsize=10, fontweight="bold", color=color,
            fontfamily="monospace",
            annotation_clip=False,
        )
        val_annotations.append(ann)

    # ── readout panels (bottom row) ──
    readout_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax, label, color in zip(readout_axes, SENSOR_LABELS, SENSOR_COLORS):
        draw_readout(ax, None, None, max_mm, color, label)

    return fig, ax_ts, ts_lines, val_annotations, readout_axes


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",     "-p", default=None)
    p.add_argument("--baud",     "-b", type=int,   default=BAUD_DEFAULT)
    p.add_argument("--window",   "-w", type=float, default=WINDOW_S)
    p.add_argument("--max-dist", "-m", type=int,   default=MAX_DIST_MM)
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

    times   = collections.deque(maxlen=HISTORY_PTS)
    vals    = [collections.deque(maxlen=HISTORY_PTS) for _ in range(3)]
    latest  = [None, None, None]
    prev    = [None, None, None]   # previous valid reading for trend
    start   = [None]
    n       = [0]
    buf     = [""]

    max_mm   = args.max_dist
    window_s = args.window

    fig, ax_ts, ts_lines, val_annotations, readout_axes = build_figure(window_s, max_mm)

    def update(_frame):
        # ── drain serial buffer ──
        try:
            raw = ser.read(ser.in_waiting or 1)
        except serial.SerialException as e:
            return

        if raw:
            buf[0] += raw.decode("utf-8", errors="replace")
            while "\n" in buf[0]:
                line, buf[0] = buf[0].split("\n", 1)
                line = line.strip()
                parts = line.split(",")
                if len(parts) != 3:
                    continue
                try:
                    v = [int(p) for p in parts]
                except ValueError:
                    continue

                now = time.monotonic()
                if start[0] is None:
                    start[0] = now
                t = now - start[0]
                times.append(t)
                for i in range(3):
                    vals[i].append(v[i])
                # Update prev before updating latest
                for i in range(3):
                    prev[i] = latest[i]
                latest[0], latest[1], latest[2] = v
                n[0] += 1

        if not times:
            return

        # ── time-series ──
        now_t = times[-1]
        t_min = max(0.0, now_t - window_s)

        for i, line in enumerate(ts_lines):
            ft, fv = [], []
            for t, v in zip(times, vals[i]):
                state, val = classify(v, max_mm)
                if state == "valid":
                    ft.append(t - t_min); fv.append(val)
                elif state == "too_close":
                    ft.append(t - t_min); fv.append(MIN_DIST_MM)
            line.set_data(ft, fv)

            # Live value label at right edge
            if ft:
                last_y = fv[-1]
                state, val = classify(latest[i], max_mm)
                if state == "valid":
                    val_annotations[i].set_text(f" {val} mm")
                    val_annotations[i].xy = (ft[-1], last_y)
                    val_annotations[i].set_visible(True)
                elif state == "too_close":
                    val_annotations[i].set_text(" <TOO CLOSE>")
                    val_annotations[i].xy = (ft[-1], MIN_DIST_MM)
                    val_annotations[i].set_visible(True)
                else:
                    val_annotations[i].set_visible(False)
            else:
                val_annotations[i].set_visible(False)

        ax_ts.set_xlim(0, window_s)

        # ── readout panels ──
        for i, (ax, color, label) in enumerate(zip(readout_axes, SENSOR_COLORS, SENSOR_LABELS)):
            draw_readout(ax, latest[i], prev[i], max_mm, color, label)

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
