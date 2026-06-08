"""
realtime_visualizer.py
----------------------
Real-time visualiser for the self-capacitant skin sensor (3 nodes).

The Arduino sketch streams one line per loop iteration:
    sensorValue1,sensorValue5,sensorValue7

Usage:
    python realtime_visualizer.py                        # auto-detect port
    python realtime_visualizer.py --port /dev/cu.usbmodem1101
    python realtime_visualizer.py --port COM3 --baud 115200
    python realtime_visualizer.py --window 15            # show last 15 s

Press Ctrl+C or close the window to exit.
"""

import argparse
import collections
import glob
import sys
import threading
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial not installed.  Run:  pip install pyserial")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("TkAgg")          # works on most systems; change to "Qt5Agg" if needed
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as mpatches
except ImportError:
    print("ERROR: matplotlib not installed.  Run:  pip install matplotlib")
    sys.exit(1)


# ─── constants ──────────────────────────────────────────────────────────────

NODES        = ["Node 1", "Node 5", "Node 7"]
NODE_COLORS  = ["#4FC3F7", "#81C784", "#FFB74D"]   # blue, green, amber
BAUD_DEFAULT = 9600
WINDOW_S     = 10          # seconds of history shown
HISTORY_PTS  = 2000        # max data points kept in ring-buffer
FPS          = 30          # animation target frame rate
BAR_YLIM     = (0, 1200)   # capacitance range — adjust if values exceed this


# ─── serial reader (runs in a background thread) ─────────────────────────────

class SensorReader(threading.Thread):
    def __init__(self, port: str, baud: int):
        super().__init__(daemon=True)
        self.port   = port
        self.baud   = baud
        self.lock   = threading.Lock()
        self._times = collections.deque(maxlen=HISTORY_PTS)
        self._vals  = [collections.deque(maxlen=HISTORY_PTS) for _ in range(3)]
        self._latest = [0, 0, 0]
        self._start  = None
        self._error  = None
        self._connected = False

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=2)
            time.sleep(0.5)
            ser.reset_input_buffer()
            self._connected = True
        except serial.SerialException as e:
            self._error = str(e)
            return

        while True:
            try:
                raw = ser.readline()
            except serial.SerialException as e:
                self._error = str(e)
                return

            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            parts = line.split(",")
            if len(parts) != 3:
                continue

            try:
                vals = [int(p) for p in parts]
            except ValueError:
                continue

            now = time.monotonic()
            with self.lock:
                if self._start is None:
                    self._start = now
                t = now - self._start
                self._times.append(t)
                for i, v in enumerate(vals):
                    self._vals[i].append(v)
                self._latest = vals

    def snapshot(self):
        """Return (times_list, [vals0, vals1, vals2], latest) — thread-safe copy."""
        with self.lock:
            return (
                list(self._times),
                [list(self._vals[i]) for i in range(3)],
                list(self._latest),
            )


# ─── port auto-detection ────────────────────────────────────────────────────

def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        mfr  = (p.manufacturer or "").lower()
        if any(kw in desc or kw in mfr for kw in
               ("cp210", "ch340", "ch9102", "ftdi", "esp", "usbserial", "acm", "usbmodem")):
            return p.device
    for pattern in (
        "/dev/cu.usbmodem*", "/dev/cu.usbserial*",
        "/dev/cu.wchusbserial*", "/dev/tty.usbmodem*", "/dev/tty.usbserial*",
        "/dev/ttyACM*", "/dev/ttyUSB*",
    ):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


# ─── argument parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Real-time visualiser for 3-node capacitive skin.")
    p.add_argument("--port",   "-p", default=None,         help="Serial port (auto-detected if omitted).")
    p.add_argument("--baud",   "-b", type=int, default=BAUD_DEFAULT, help=f"Baud rate (default {BAUD_DEFAULT}).")
    p.add_argument("--window", "-w", type=float, default=WINDOW_S,  help=f"Time window in seconds (default {WINDOW_S}).")
    return p.parse_args()


# ─── build the figure ───────────────────────────────────────────────────────

def build_figure(window_s: float):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 7), facecolor="#0D0D0D")
    fig.canvas.manager.set_window_title("3D Skin – Live Capacitance")

    gs = gridspec.GridSpec(
        2, 4,
        figure=fig,
        left=0.06, right=0.98,
        top=0.88,  bottom=0.10,
        hspace=0.55, wspace=0.45,
    )

    # ── title ──
    fig.text(
        0.5, 0.95,
        "Self-Capacitant Skin  ·  Real-Time Sensor Readout",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="white",
    )

    # ── time-series axes (top row, spans 3 columns) ──
    ax_ts = fig.add_subplot(gs[0, :3])
    ax_ts.set_facecolor("#1A1A1A")
    ax_ts.set_xlim(0, window_s)
    ax_ts.set_ylim(*BAR_YLIM)
    ax_ts.set_xlabel("Time (s)", color="#AAAAAA", fontsize=10)
    ax_ts.set_ylabel("Capacitance (counts)", color="#AAAAAA", fontsize=10)
    ax_ts.tick_params(colors="#AAAAAA")
    for spine in ax_ts.spines.values():
        spine.set_edgecolor("#333333")
    ax_ts.grid(True, color="#2A2A2A", linewidth=0.5)

    ts_lines = []
    for color, node in zip(NODE_COLORS, NODES):
        line, = ax_ts.plot([], [], color=color, linewidth=1.8, label=node)
        ts_lines.append(line)

    legend_patches = [
        mpatches.Patch(color=NODE_COLORS[i], label=NODES[i]) for i in range(3)
    ]
    ax_ts.legend(handles=legend_patches, loc="upper right",
                 framealpha=0.3, fontsize=9)

    # ── status label (top-right cell) ──
    ax_status = fig.add_subplot(gs[0, 3])
    ax_status.set_facecolor("#1A1A1A")
    ax_status.axis("off")
    status_txt = ax_status.text(
        0.5, 0.5, "Waiting for data…",
        ha="center", va="center",
        fontsize=10, color="#AAAAAA",
        transform=ax_status.transAxes,
        wrap=True,
    )

    # ── bar gauges (bottom row, one per node + one spacer) ──
    bar_axes = []
    bar_rects = []
    val_texts = []

    for i in range(3):
        ax_b = fig.add_subplot(gs[1, i])
        ax_b.set_facecolor("#1A1A1A")
        ax_b.set_xlim(-0.5, 0.5)
        ax_b.set_ylim(*BAR_YLIM)
        ax_b.set_xticks([])
        ax_b.set_title(NODES[i], color=NODE_COLORS[i], fontsize=11, fontweight="bold", pad=6)
        ax_b.tick_params(colors="#AAAAAA", labelsize=8)
        for spine in ax_b.spines.values():
            spine.set_edgecolor("#333333")
        ax_b.yaxis.set_tick_params(which="both", length=3)

        # background track
        ax_b.bar(0, BAR_YLIM[1], width=0.6, color="#2A2A2A", zorder=1)

        # live bar
        rect = ax_b.bar(0, 0, width=0.6, color=NODE_COLORS[i], alpha=0.9, zorder=2)[0]
        bar_rects.append(rect)

        # current-value label
        txt = ax_b.text(
            0, BAR_YLIM[1] * 0.97, "—",
            ha="center", va="top",
            fontsize=13, fontweight="bold", color="white", zorder=3,
        )
        val_texts.append(txt)
        bar_axes.append(ax_b)

    # ── hide 4th bottom cell ──
    fig.add_subplot(gs[1, 3]).set_visible(False)

    return fig, ax_ts, ts_lines, bar_rects, val_texts, status_txt


# ─── animation update ────────────────────────────────────────────────────────

def make_updater(reader, ax_ts, ts_lines, bar_rects, val_texts, status_txt, window_s):
    frame_count = [0]

    def update(_frame):
        frame_count[0] += 1

        if reader._error:
            status_txt.set_text(f"Serial error:\n{reader._error}")
            status_txt.set_color("#FF5252")
            return

        times, vals, latest = reader.snapshot()

        if not times:
            return  # still waiting

        now = times[-1]
        t_min = max(0.0, now - window_s)

        # update time-series
        for i, line in enumerate(ts_lines):
            # only show the points within the current window
            t_arr = times
            v_arr = vals[i]
            if len(t_arr) != len(v_arr):
                continue
            # shift so the right edge is always window_s
            shifted = [t - t_min for t in t_arr]
            line.set_data(shifted, v_arr)

        ax_ts.set_xlim(0, window_s)

        # update bars and value labels
        for i, (rect, txt) in enumerate(zip(bar_rects, val_texts)):
            v = latest[i]
            rect.set_height(max(0, v))
            txt.set_text(str(v))

        # status
        rate_hz = frame_count[0] / now if now > 0 else 0
        status_txt.set_text(
            f"Samples: {len(times)}\n"
            f"Time: {now:.1f} s\n"
            f"~{rate_hz:.0f} Hz"
        )
        status_txt.set_color("#66BB6A")

    return update


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    port = args.port
    if port is None:
        port = find_port()
        if port is None:
            print("ERROR: Could not auto-detect a serial port.")
            print("  Plug in the sensor and retry, or pass --port /dev/cu.usbmodemXXXX")
            sys.exit(1)
        print(f"Auto-detected port: {port}")

    print(f"Connecting to {port} @ {args.baud} baud …")
    reader = SensorReader(port, args.baud)
    reader.start()

    # give the thread a moment to open the port
    time.sleep(1.0)
    if reader._error:
        print(f"ERROR: {reader._error}")
        sys.exit(1)

    print("Serial open. Building display…")

    fig, ax_ts, ts_lines, bar_rects, val_texts, status_txt = build_figure(args.window)

    updater = make_updater(
        reader, ax_ts, ts_lines, bar_rects, val_texts, status_txt, args.window
    )

    ani = FuncAnimation(
        fig, updater,
        interval=1000 // FPS,
        blit=False,
        cache_frame_data=False,
    )

    print("Visualiser running. Close the window or press Ctrl+C to quit.")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    print("Done.")


if __name__ == "__main__":
    main()
