"""
Mag_3D_Visualizer.py
--------------------
3D touch visualiser for the MLX90393 magnetometer skin (5 chips per board).
Renders the PCB in OpenGL; each chip lights up and shows a field vector on touch.

Usage:
    python Demo/Mag_3D_Visualizer.py
    python Demo/Mag_3D_Visualizer.py --stl path/to/board.stl
    python Demo/Mag_3D_Visualizer.py --threshold 400
    python Demo/Mag_3D_Visualizer.py --port /dev/cu.usbmodem1101

Keyboard:
    R      – recalibrate baseline
    T      – toggle field-vector arrows
    Space  – reset camera
"""

import argparse
import collections
import glob
import math
import os
import re
import sys
import threading
import time

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("ERROR: pyserial not installed.  Run:  pip install pyserial")

try:
    import trimesh
except ImportError:
    sys.exit("ERROR: trimesh not installed.  Run:  pip install trimesh")

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except ImportError:
    sys.exit("ERROR: pyqtgraph / PyOpenGL not installed.\n"
             "  Run:  pip install pyqtgraph PyOpenGL")


# ─── board geometry (all units: metres) ──────────────────────────────────────

BOARD_W = 0.048    # 48 mm  (chips span ±20 mm → 4 mm margin each side)
BOARD_H = 0.034    # 34 mm
BOARD_T = 0.0025   # 2.5 mm thick

PAD_R   = 0.005    # sensor-pad cylinder radius  (5 mm)
PAD_H   = 0.0008   # sensor-pad cylinder height  (0.8 mm)
GLOW_R  = 0.013    # glow-disk max radius at full touch

# ── Physical chip layout ──────────────────────────────────────────────
# Each board has 1 chip in the centre + 4 in the corners (5 chips/board).
# CHIP_XY is indexed by HARDWARE MUX CHANNEL (CH0..CH7), so the pad that
# lights up sits at the same spot as the chip you physically touch.
#
# Single board on one 8-channel mux → channels 0..4 used below.
# To add the 2nd (right) board you need a 2nd mux / more channels; extend
# this table to channels 5..9 with a +X offset once that's wired.
#
#   corner geometry (relative to board centre, units: metres)
#
# Hardware (from Mux_Address_Scanner): 3 boards on mux channels 0,1,2, each
# with 5 chips at I2C 0x0C (centre) + 0x10/0x11/0x12/0x13 (corners).
# Firmware streams global id = channel*5 + chipIndex, so:
#   board 0 → CH0..CH4,  board 1 → CH5..CH9,  board 2 → CH10..CH14.
_HALF_W = 0.014    # 14 mm half-width  (each board ~28 mm)
_HALF_H = 0.010    # 10 mm half-height
_BOARD_GAP = 0.040 # centre-to-centre spacing between adjacent boards (40 mm)

# One board's local chip layout (centre + 4 corners), order matches the
# firmware's CHIP_ADDRS = [0x0C, 0x10, 0x11, 0x12, 0x13].
_BOARD_LOCAL = np.array([
    [ 0.000,  0.000],       # 0 – CENTRE          (0x0C)
    [-_HALF_W,  _HALF_H],   # 1 – top-left corner (0x10)
    [ _HALF_W,  _HALF_H],   # 2 – top-right       (0x11)
    [ _HALF_W, -_HALF_H],   # 3 – bottom-right    (0x12)
    [-_HALF_W, -_HALF_H],   # 4 – bottom-left     (0x13)
], dtype=np.float32)

CHIPS_PER_BOARD = len(_BOARD_LOCAL)   # 5

# ── Physical arrangement of the boards ────────────────────────────────
# Each board is identified by its MUX CHANNEL.  The firmware streams a global
# id = channel*5 + chipIndex, so:  channel 0 → CH0-4,  channel 2 → CH10-14,
# channel 3 → CH15-19, etc.
#
# BOARD_CHANNELS lists the channels LEFT → RIGHT as the boards physically sit
# on the skin.  EDIT THIS to match your hardware.
BOARD_CHANNELS = [0, 2, 3]   # left, middle, right
NUM_BOARDS = len(BOARD_CHANNELS)

# Channels with NO magnetic elastomer on top can't sense touch — they only see
# the ambient field and will saturate red from the neighbouring magnets. List
# such channels here to MUTE them (rendered dim, never register a touch).
# Remove a channel from this list once it has its magnet layer installed.
DISABLED_CHANNELS = [3]

# If the visualiser appears left-right mirrored vs the real skin (you press the
# left and the right lights up), flip this. It mirrors the whole scene's X axis.
MIRROR_X = True

# Arrays are sized to cover the highest channel in use (channel 3 → ids 15-19).
_MAX_CH   = max(BOARD_CHANNELS)
N_CHIPS   = (_MAX_CH + 1) * CHIPS_PER_BOARD   # some ids (unused channels) stay blank
MAX_CHIPS = N_CHIPS

# Lay out each real board into its physical slot; mark which ids are real.
CHIP_XY     = np.zeros((N_CHIPS, 2), dtype=np.float32)
CHIP_ACTIVE = np.zeros(N_CHIPS, dtype=bool)
BOARD_CENTRES = []
for _slot, _ch in enumerate(BOARD_CHANNELS):
    _cx = (_slot - (NUM_BOARDS - 1) / 2.0) * _BOARD_GAP
    _centre = np.array([_cx, 0.0], dtype=np.float32)
    BOARD_CENTRES.append(_centre)
    for _j in range(CHIPS_PER_BOARD):
        _gid = _ch * CHIPS_PER_BOARD + _j
        CHIP_XY[_gid]     = _BOARD_LOCAL[_j] + _centre
        CHIP_ACTIVE[_gid] = True

# Mirror the whole layout across X so physical-left maps to screen-left.
if MIRROR_X:
    CHIP_XY[:, 0] *= -1.0
    for _c in BOARD_CENTRES:
        _c[0] *= -1.0

PAD_Z_CTR = BOARD_T / 2 + PAD_H / 2    # cylinder centre Z
GLOW_Z    = BOARD_T / 2 + PAD_H + 3e-4  # glow disk Z
VEC_Z0    = BOARD_T / 2 + PAD_H + 1e-3  # vector base Z
LABEL_Z   = BOARD_T / 2 + PAD_H + 0.014 # label Z

# ── Curved skin surface ───────────────────────────────────────────────
# The skin is one continuous curved sheet (a section of a cylinder bowing
# around the Y axis). Each chip's flat (x, y) layout coordinate is wrapped
# onto the curve: the x coordinate becomes arc-length, y stays vertical.
SKIN_RADIUS = 0.075   # cylinder radius (m) — smaller = more curvature
PAD_LIFT    = 0.0012  # how far pads/glow sit above the surface (along normal)
LABEL_LIFT  = 0.016   # label offset above the surface


def _curve_point(cx, cy):
    """Map an unrolled (cx, cy) layout coord onto the curved skin.
    Returns (position_xyz, outward_unit_normal)."""
    theta = cx / SKIN_RADIUS
    px = SKIN_RADIUS * math.sin(theta)
    pz = SKIN_RADIUS * math.cos(theta) - SKIN_RADIUS   # x=0 → z=0
    pos = np.array([px, cy, pz], dtype=np.float32)
    nrm = np.array([math.sin(theta), 0.0, math.cos(theta)], dtype=np.float32)
    return pos, nrm


# Precompute every chip's 3D position and surface normal on the curve.
CHIP_POS3D = np.zeros((N_CHIPS, 3), dtype=np.float32)
CHIP_NRM3D = np.zeros((N_CHIPS, 3), dtype=np.float32)
for _i in range(N_CHIPS):
    _p, _n = _curve_point(float(CHIP_XY[_i, 0]), float(CHIP_XY[_i, 1]))
    CHIP_POS3D[_i] = _p
    CHIP_NRM3D[_i] = _n


def _tangent_basis(n):
    """Two orthonormal vectors spanning the plane perpendicular to n."""
    n = n / (np.linalg.norm(n) + 1e-9)
    ref = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(n[1]) < 0.9 \
        else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(ref, n); u /= (np.linalg.norm(u) + 1e-9)
    w = np.cross(n, u)
    return u.astype(np.float32), w.astype(np.float32)


def _oriented_disk(center, normal, r, sec=32):
    """Flat disk centred at `center`, lying in the plane with the given normal."""
    u, w = _tangent_basis(normal)
    ang = np.linspace(0, 2 * np.pi, sec, endpoint=False)
    rim = center[None, :] + r * (np.cos(ang)[:, None] * u[None, :]
                                 + np.sin(ang)[:, None] * w[None, :])
    v = np.vstack([center[None, :], rim]).astype(np.float32)
    # triangle fan from centre (vertex 0) around the rim (vertices 1..sec)
    f = np.array([[0, i, (i % sec) + 1] for i in range(1, sec + 1)], dtype=np.uint32)
    return v, f


def _build_skin_mesh():
    """Tessellate the curved skin sheet covering all chips (+ margin)."""
    xs, ys = CHIP_XY[:, 0], CHIP_XY[:, 1]
    x0, x1 = xs.min() - 0.013, xs.max() + 0.013
    y0, y1 = ys.min() - 0.013, ys.max() + 0.013
    nu, nv = 60, 20
    us = np.linspace(x0, x1, nu)
    vs = np.linspace(y0, y1, nv)
    verts = np.zeros((nv * nu, 3), dtype=np.float32)
    for j, vy in enumerate(vs):
        for k, ux in enumerate(us):
            p, _ = _curve_point(float(ux), float(vy))
            verts[j * nu + k] = p
    faces = []
    for j in range(nv - 1):
        for k in range(nu - 1):
            a = j * nu + k
            b = a + 1
            c = a + nu
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, np.array(faces, dtype=np.uint32)

# ─── colours ─────────────────────────────────────────────────────────────────

CHIP_COLORS = [
    (0.93, 0.33, 0.31, 1.0),   # red
    (0.67, 0.28, 0.74, 1.0),   # purple
    (0.15, 0.76, 0.85, 1.0),   # cyan
    (0.40, 0.73, 0.25, 1.0),   # green
    (1.00, 0.65, 0.15, 1.0),   # orange
]

PCB_COLOR = np.array([0.10, 0.28, 0.12, 0.95], dtype=np.float32)   # PCB green

# Magnitude colour ramp: (t_fraction, rgba)  t = mag / threshold
# Idle (t≈0) is a dim blue-grey so untouched pads recede; contact ramps to
# bright orange→red so touch points pop out clearly.
_RAMP = [
    (0.00, np.array([0.22, 0.26, 0.32, 0.65])),   # rest – dim slate
    (0.30, np.array([0.15, 0.70, 0.85, 0.80])),   # light touch – cyan
    (0.65, np.array([1.00, 0.65, 0.10, 0.92])),   # firm – orange
    (1.00, np.array([1.00, 0.07, 0.07, 1.00])),   # hard – red
]

BASELINE_SAMPLES = 30
HISTORY_LEN      = 400
FLASH_FRAMES     = 8
VEC_DISPLAY_SCALE = None   # set at runtime: 0.020 / threshold


def _mag_color(mag: float, threshold: float) -> np.ndarray:
    t = max(0.0, min(mag / threshold, 1.0))
    for i in range(len(_RAMP) - 1):
        t0, c0 = _RAMP[i]
        t1, c1 = _RAMP[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return (c0 + f * (c1 - c0)).astype(np.float32)
    return _RAMP[-1][1].copy()


# ─── mesh helpers ─────────────────────────────────────────────────────────────

def _flat_disk(cx: float, cy: float, z: float, r: float, sec: int = 32):
    """Flat disk mesh at (cx, cy, z) with radius r."""
    angles = np.linspace(0, 2 * np.pi, sec, endpoint=False)
    v = np.zeros((sec + 1, 3), dtype=np.float32)
    v[0] = [cx, cy, z]
    v[1:, 0] = cx + np.cos(angles) * r
    v[1:, 1] = cy + np.sin(angles) * r
    v[1:, 2] = z
    f = np.array([[0, i + 1, (i + 1) % sec + 1] for i in range(sec)],
                 dtype=np.uint32)
    return v, f


def _cylinder(cx: float, cy: float, z_ctr: float, r: float, h: float, sec: int = 32):
    """Short upright cylinder mesh."""
    try:
        mesh = trimesh.creation.cylinder(radius=r, height=h, sections=sec)
        v = np.array(mesh.vertices, dtype=np.float32)
        v[:, 0] += cx
        v[:, 1] += cy
        v[:, 2] += z_ctr
        return v, np.array(mesh.faces, dtype=np.uint32)
    except Exception:
        # fallback: two disks
        tv, tf = _flat_disk(cx, cy, z_ctr + h / 2, r, sec)
        bv, bf = _flat_disk(cx, cy, z_ctr - h / 2, r, sec)
        nv = len(tv)
        return (np.vstack([tv, bv]),
                np.vstack([tf, bf + nv]))


# ─── STL board generation / loading ──────────────────────────────────────────

_PLACEHOLDER_STL = os.path.join(os.path.dirname(__file__), "mag_skin_board.stl")


def _generate_board_stl(path: str = _PLACEHOLDER_STL) -> str:
    box = trimesh.creation.box((BOARD_W, BOARD_H, BOARD_T))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    box.export(path)
    print(f"Generated placeholder board STL → {path}")
    return path


def _load_board_mesh(stl_path: str):
    """Load STL, auto-detect mm vs m, centre at origin. Returns (verts, faces)."""
    raw = trimesh.load(stl_path, force='mesh')
    v = np.array(raw.vertices, dtype=np.float32)
    span = v.max(axis=0) - v.min(axis=0)
    if span.max() > 1.0:          # assume mm → convert
        v /= 1000.0
    v[:, 0] -= (v[:, 0].max() + v[:, 0].min()) / 2
    v[:, 1] -= (v[:, 1].max() + v[:, 1].min()) / 2
    v[:, 2] -= (v[:, 2].max() + v[:, 2].min()) / 2
    return v, np.array(raw.faces, dtype=np.uint32)


# ─── serial helpers ───────────────────────────────────────────────────────────

RE_MULTI  = re.compile(r'CH(\d+):\s+X:([\d.\-]+)\s+Y:([\d.\-]+)\s+Z:([\d.\-]+)')
RE_SINGLE = re.compile(r'X:\s*([\d.\-]+)\s+Y:\s*([\d.\-]+)\s+Z:\s*([\d.\-]+)')
INV_XY, INV_Z = 78000.0, 126000.0


def _parse_line(line: str):
    # Multi-sensor format: "CH0: X:1234.56 Y:... Z:...   |   CH1: ..."
    # Only skip channels that are individually flagged [INVALID], not the whole line.
    matches = RE_MULTI.findall(line)
    if matches:
        out = []
        for m in matches:
            ch, x, y, z = int(m[0]), float(m[1]), float(m[2]), float(m[3])
            if x < INV_XY and y < INV_XY and z < INV_Z:
                out.append((ch, x, y, z))
        return out
    # Single-sensor format: "X: 1234.56  Y: ...  Z: ..."
    if "[INVALID" in line:
        return []
    m = RE_SINGLE.search(line)
    if m:
        x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if x < INV_XY and y < INV_XY and z < INV_Z:
            return [(0, x, y, z)]
    return []


def _find_port(prefer=None):
    if prefer and glob.glob(prefer):
        return prefer
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        mfr  = (p.manufacturer or "").lower()
        if any(k in desc or k in mfr for k in
               ("cp210", "ch340", "ch9102", "ftdi", "esp", "usbserial", "acm", "usbmodem")):
            return p.device
    for pat in ("/dev/cu.usbmodem*", "/dev/cu.usbserial*",
                "/dev/tty.usbmodem*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


# ─── main visualiser ──────────────────────────────────────────────────────────

class MagSkin3DVisualizer:

    def __init__(self, args):
        self.args       = args
        # threshold <= 0  → adaptive (auto-tuned to each chip's noise)
        self.auto_thr   = (args.threshold is None) or (args.threshold <= 0)
        self.threshold  = args.threshold if not self.auto_thr else 50.0
        self.sensitivity = args.sensitivity
        self.floor       = args.floor   # min adaptive threshold (µT)
        self.show_vecs  = True

        global VEC_DISPLAY_SCALE
        VEC_DISPLAY_SCALE = 0.020 / max(self.threshold, 1.0)

        # calibration
        self.ch_to_idx  = {}
        self.cal_buf    = [[] for _ in range(N_CHIPS)]
        self.baselines  = [None] * N_CHIPS
        self.noise      = [None] * N_CHIPS   # per-chip rest noise level (µT)

        # live state (fresh array per channel — no shared aliasing)
        self.deltas     = [np.zeros(3, dtype=np.float32) for _ in range(N_CHIPS)]
        self.mags       = [0.0]   * N_CHIPS
        self.is_touch   = [False] * N_CHIPS
        self.flash_left = [0]     * N_CHIPS

        # history for chart
        self.mag_hist   = [collections.deque(maxlen=HISTORY_LEN) for _ in range(N_CHIPS)]
        self.sample_idx = 0

        # serial (read on a background thread; UI consumes from _rx_queue)
        self.ser           = None
        self._port         = None
        self._reconnecting = False
        self._bytes_recv   = 0
        self._lines_parsed = 0
        self._diag_printed = 0    # how many raw lines we've echoed for diagnosis
        self._last_raw     = ""   # most recent raw serial line (for on-screen debug)
        self._last_parsed  = False
        self._rx_queue     = collections.deque(maxlen=4000)  # thread-safe: raw lines
        self._stop_evt     = threading.Event()
        self._rx_thread    = None

        # GL items (filled in _setup_scene)
        self._pad_items   = []
        self._pad_nfaces  = []
        self._glow_items  = []
        self._vec_items   = []
        self._label_items = []

    # ─── entry point ──────────────────────────────────────────────────

    def run(self):
        self._resolve_port()

        pg.setConfigOptions(antialias=True)
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        self._build_window()
        self._setup_scene()
        self._setup_shortcuts()
        self._setup_timer()

        # Start the background serial reader (blocking reads in its own thread)
        self._rx_thread = threading.Thread(target=self._serial_loop, daemon=True)
        self._rx_thread.start()

        self.win.show()
        print(f"Touch threshold |Δ| ≥ {self.threshold:.0f}  |  "
              "R = recalibrate  T = vectors  Space = reset camera")
        try:
            self.app.exec()
        finally:
            self._stop_evt.set()
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass

    # ─── window layout ────────────────────────────────────────────────

    def _build_window(self):
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Magnetic Skin – 3D Touch Visualiser")
        self.win.resize(1300, 840)
        self.win.setStyleSheet("background:#0D0D0D;")

        root = QtWidgets.QWidget()
        self.win.setCentralWidget(root)
        vbox = QtWidgets.QVBoxLayout(root)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        # ── top row: 3D view + status panel ──
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor('#0D0D0D')
        self.gl_view.setCameraPosition(distance=0.15, elevation=38, azimuth=35)
        splitter.addWidget(self.gl_view)

        # right status panel
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(230)
        panel.setStyleSheet("background:#0F0F0F;")
        playout = QtWidgets.QVBoxLayout(panel)
        playout.setContentsMargins(8, 8, 8, 8)
        playout.setSpacing(5)

        title = QtWidgets.QLabel(f"MLX90393 · {NUM_BOARDS} boards × {CHIPS_PER_BOARD}")
        title.setStyleSheet("color:#AAAAAA; font-size:12px; font-weight:bold;"
                            " font-family:monospace; padding:2px;")
        playout.addWidget(title)

        # Scrollable list of compact per-chip rows, grouped by board.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border:none; background:#0F0F0F;")
        inner = QtWidgets.QWidget()
        ilayout = QtWidgets.QVBoxLayout(inner)
        ilayout.setContentsMargins(0, 0, 0, 0)
        ilayout.setSpacing(2)

        self._chip_labels = []
        for i in range(N_CHIPS):
            board = i // CHIPS_PER_BOARD
            chip  = i % CHIPS_PER_BOARD
            if chip == 0:
                hdr = QtWidgets.QLabel(f"── Board {board} ──")
                hdr.setStyleSheet("color:#777777; font-size:9px; font-weight:bold;"
                                  " font-family:monospace; padding-top:3px;")
                ilayout.addWidget(hdr)
            c = CHIP_COLORS[chip % len(CHIP_COLORS)]
            hex_c = "#{:02X}{:02X}{:02X}".format(
                int(c[0]*255), int(c[1]*255), int(c[2]*255))
            lbl = QtWidgets.QLabel(f"CH{i}  —")
            lbl.setStyleSheet(
                f"color:{hex_c}; font-size:10px; font-family:monospace;"
                " background:#1A1A1A; border-radius:3px; padding:3px;")
            ilayout.addWidget(lbl)
            self._chip_labels.append(lbl)

        ilayout.addStretch()
        scroll.setWidget(inner)
        playout.addWidget(scroll, stretch=1)

        # Raw serial line debug — shows exactly what the firmware sends
        raw_title = QtWidgets.QLabel("RAW SERIAL:")
        raw_title.setStyleSheet("color:#666666; font-size:8px; font-family:monospace;")
        playout.addWidget(raw_title)
        self._raw_lbl = QtWidgets.QLabel("(none yet)")
        self._raw_lbl.setWordWrap(True)
        self._raw_lbl.setStyleSheet(
            "color:#888888; font-size:8px; font-family:monospace;"
            " background:#1A1A1A; border-radius:3px; padding:4px;")
        playout.addWidget(self._raw_lbl)

        self._footer_lbl = QtWidgets.QLabel("Waiting for data…")
        self._footer_lbl.setStyleSheet(
            "color:#555555; font-size:9px; font-family:monospace;")
        playout.addWidget(self._footer_lbl)
        splitter.addWidget(panel)
        splitter.setSizes([1060, 230])
        vbox.addWidget(splitter, stretch=4)

        # ── bottom: magnitude chart ──
        self.plot_w = pg.PlotWidget(background='#141414')
        self.plot_w.setMaximumHeight(195)
        self.plot_w.setLabel('left',   '|Δ| Magnitude', color='#AAAAAA', size='10pt')
        self.plot_w.setLabel('bottom', 'Samples',        color='#AAAAAA', size='10pt')
        self.plot_w.showGrid(x=True, y=True, alpha=0.25)
        # Magnitudes are unknown a-priori (depends on the sensor) → auto-range Y.
        self.plot_w.enableAutoRange('y', True)
        self.plot_w.setYRange(0, 100)

        self._curves = []
        for i in range(N_CHIPS):
            c = CHIP_COLORS[i % len(CHIP_COLORS)]
            pen = pg.mkPen(color=(int(c[0]*255), int(c[1]*255), int(c[2]*255)), width=2)
            curve = self.plot_w.plot([], [], pen=pen, name=f"S{i}")
            self._curves.append(curve)

        vbox.addWidget(self.plot_w, stretch=1)

    # ─── 3D scene ─────────────────────────────────────────────────────

    def _setup_scene(self):
        grid = gl.GLGridItem()
        grid.setSize(0.30, 0.22)
        grid.setSpacing(0.01, 0.01)
        grid.translate(0, 0, -(SKIN_RADIUS * 0.25 + 0.01))
        self.gl_view.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(0.012, 0.012, 0.012)
        self.gl_view.addItem(axis)

        self._build_skin()

        for i in range(N_CHIPS):
            pos = CHIP_POS3D[i]
            nrm = CHIP_NRM3D[i]
            self._make_pad(i, pos, nrm)
            self._make_glow(i, pos, nrm)
            self._make_vector(i, pos, nrm)
            self._make_label(i, pos, nrm)

    def _build_skin(self):
        """One continuous curved skin surface with all chips embedded on it."""
        verts, faces = _build_skin_mesh()
        fc = np.full((len(faces), 4), PCB_COLOR, dtype=np.float32)
        item = gl.GLMeshItem(
            vertexes=verts, faces=faces, faceColors=fc,
            drawEdges=True, edgeColor=(0.0, 0.45, 0.15, 0.30),
            smooth=True, glOptions='translucent')
        self.gl_view.addItem(item)
        print(f"Curved skin: {len(faces)} triangles, radius {SKIN_RADIUS*1000:.0f} mm")

    def _pad_centre(self, pos, nrm):
        """Point sitting just above the surface along its normal."""
        return (pos + nrm * PAD_LIFT).astype(np.float32)

    def _make_pad(self, idx: int, pos, nrm):
        c = self._pad_centre(pos, nrm)
        v, f = _oriented_disk(c, nrm, PAD_R)
        col = np.array(CHIP_COLORS[idx % len(CHIP_COLORS)], dtype=np.float32)
        col[3] = 0.55
        fc = np.full((len(f), 4), col, dtype=np.float32)
        item = gl.GLMeshItem(vertexes=v, faces=f, faceColors=fc,
                             drawEdges=False, smooth=True, glOptions='translucent')
        self.gl_view.addItem(item)
        self._pad_items.append(item)
        self._pad_nfaces.append(len(f))

    def _make_glow(self, idx: int, pos, nrm):
        c = self._pad_centre(pos, nrm)
        v, f = _oriented_disk(c, nrm, PAD_R)
        fc = np.zeros((len(f), 4), dtype=np.float32)   # invisible at start
        item = gl.GLMeshItem(vertexes=v, faces=f, faceColors=fc,
                             drawEdges=False, smooth=True, glOptions='translucent')
        self.gl_view.addItem(item)
        self._glow_items.append(item)

    def _make_vector(self, idx: int, pos, nrm):
        c = self._pad_centre(pos, nrm)
        pts = np.array([c, c + nrm * 0.001], dtype=np.float32)
        col = CHIP_COLORS[idx % len(CHIP_COLORS)]
        item = gl.GLLinePlotItem(pos=pts, color=col, width=3.0,
                                 antialias=True, mode='lines')
        self.gl_view.addItem(item)
        self._vec_items.append(item)

    def _make_label(self, idx: int, pos, nrm):
        c = CHIP_COLORS[idx % len(CHIP_COLORS)]
        qc = pg.mkColor(int(c[0]*255), int(c[1]*255), int(c[2]*255))
        lpos = (pos + nrm * LABEL_LIFT).astype(np.float32)
        try:
            lbl = gl.GLTextItem(
                pos=lpos, text=f"CH{idx}", color=qc)
            lbl.setData(font=QtGui.QFont("Helvetica", 13, QtGui.QFont.Weight.Bold))
            self.gl_view.addItem(lbl)
            self._label_items.append(lbl)
        except Exception:
            self._label_items.append(None)

    # ─── shortcuts ────────────────────────────────────────────────────

    def _setup_shortcuts(self):
        pg.QtGui.QShortcut(pg.QtGui.QKeySequence('R'), self.win).activated.connect(
            self._recalibrate)
        pg.QtGui.QShortcut(pg.QtGui.QKeySequence('T'), self.win).activated.connect(
            self._toggle_vectors)
        pg.QtGui.QShortcut(pg.QtGui.QKeySequence('Space'), self.win).activated.connect(
            self._reset_camera)

    # ─── timer ────────────────────────────────────────────────────────

    def _setup_timer(self):
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)   # ~30 Hz

    # ─── serial ───────────────────────────────────────────────────────

    def _resolve_port(self):
        all_ports = serial.tools.list_ports.comports()
        if all_ports:
            print("Available serial ports:")
            for pp in all_ports:
                print(f"  {pp.device:<25} {pp.description}")
        else:
            print("No serial ports found at all.")

        port = self.args.port or _find_port()
        if port is None:
            sys.exit("ERROR: no port auto-detected. Re-run with --port /dev/...")
        self._port = port
        print(f"\nUsing: {port}  (override with --port if wrong)")
        print("Reading serial on a background thread.  "
              "Press RESET on the ESP32 if nothing streams.\n")

    def _serial_loop(self):
        """Background thread: blocking reads (works on ESP32-C6 USB-serial-JTAG),
        push raw lines into a thread-safe queue. Auto-reconnects on USB dropout."""
        buf = ""
        while not self._stop_evt.is_set():
            # (re)open
            if self.ser is None:
                self._reconnecting = True
                port = _find_port(self._port) or self._port
                try:
                    self.ser = serial.Serial(port, self.args.baud, timeout=0.2)
                    self._port = port
                    time.sleep(0.3)
                    self.ser.reset_input_buffer()
                    buf = ""
                    self._reconnecting = False
                    print(f"[serial] connected to {port}")
                except (serial.SerialException, OSError):
                    self.ser = None
                    time.sleep(0.5)
                    continue

            # blocking read with timeout — returns whatever arrived
            try:
                chunk = self.ser.read(4096)
            except (serial.SerialException, OSError):
                print("[serial] dropout — reconnecting…")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                continue

            if not chunk:
                continue
            self._bytes_recv += len(chunk)
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                stripped = line.strip()
                if stripped:
                    self._rx_queue.append(stripped)

    # ─── main tick ────────────────────────────────────────────────────

    def _tick(self):
        self._consume_queue()
        self._refresh_scene()
        self._refresh_chart()
        self._refresh_status()

    def _consume_queue(self):
        # Drain everything the serial thread has queued since the last tick.
        n = len(self._rx_queue)
        for _ in range(n):
            try:
                stripped = self._rx_queue.popleft()
            except IndexError:
                break

            # Print first 6 raw lines so the user can verify the format in terminal
            if self._diag_printed < 6:
                print(f"[serial] {repr(stripped)}")
                self._diag_printed += 1

            self._last_raw = stripped[:90]   # for on-screen debug overlay
            readings = _parse_line(stripped)
            self._last_parsed = bool(readings)
            if not readings:
                continue
            self._lines_parsed += 1

            for (ch, x, y, z) in readings:
                # Pad position is tied directly to the hardware mux channel,
                # so the pad that lights up is the chip you physically touch.
                if ch >= N_CHIPS:
                    continue          # channel beyond our layout table — ignore
                idx = ch
                self.ch_to_idx[ch] = idx
                xyz = np.array([x, y, z], dtype=np.float32)

                # ── Baseline calibration ──
                #   Phase 1 (first BASELINE_SAMPLES): average the raw field with
                #     the skin at rest → the "untouched" reference. While this
                #     runs we force no-touch so a stray press can't register.
                #   Phase 2 (after): baseline is locked. When idle (small Δ) we
                #     let it drift VERY slowly toward the live value, so sensor
                #     drift keeps the rest state at ~0 — the display only lights
                #     up when you actually squish it.
                if len(self.cal_buf[idx]) < BASELINE_SAMPLES:
                    self.cal_buf[idx].append(xyz)
                    self.baselines[idx] = np.mean(self.cal_buf[idx], axis=0).astype(np.float32)
                    # calibrating → report rest state, never a touch
                    self.deltas[idx]   = np.zeros(3, dtype=np.float32)
                    self.mags[idx]     = 0.0
                    self.is_touch[idx] = False
                    if len(self.cal_buf[idx]) == BASELINE_SAMPLES:
                        b = self.baselines[idx]
                        # per-chip rest noise = mean |sample - baseline| over the buffer
                        diffs = np.array(self.cal_buf[idx]) - b
                        self.noise[idx] = float(np.mean(np.linalg.norm(diffs, axis=1))) + 1e-3
                        print(f"CH{ch} baseline locked  noise≈{self.noise[idx]:.1f}  "
                              f"touch≥{self._thr(idx):.0f}")
                    self.mag_hist[idx].append(0.0)
                    self.sample_idx += 1
                    continue

                d   = xyz - self.baselines[idx]
                mag = float(np.linalg.norm(d))
                thr = self._thr(idx)

                self.deltas[idx]   = d
                self.mags[idx]     = mag
                self.is_touch[idx] = mag >= thr
                if self.is_touch[idx]:
                    self.flash_left[idx] = FLASH_FRAMES
                elif mag < thr * 0.4:
                    # idle → very slow baseline drift comp (won't absorb real presses)
                    self.baselines[idx] = (self.baselines[idx] * 0.999
                                           + xyz * 0.001).astype(np.float32)

                self.mag_hist[idx].append(mag)
                self.sample_idx += 1

    def _thr(self, idx):
        """Touch threshold for chip idx — adaptive (noise×sensitivity) or fixed.
        Disabled channels get an effectively infinite threshold so they stay dim."""
        if (idx // CHIPS_PER_BOARD) in DISABLED_CHANNELS:
            return float('inf')
        if self.auto_thr and self.noise[idx] is not None:
            return max(self.noise[idx] * self.sensitivity, self.floor)
        return self.threshold

    # ─── scene refresh ────────────────────────────────────────────────

    def _refresh_scene(self):
        for i in range(N_CHIPS):
            # Hide ids that don't belong to a real board (unused mux channels)
            if not CHIP_ACTIVE[i]:
                self._pad_items[i].setVisible(False)
                self._glow_items[i].setVisible(False)
                self._vec_items[i].setVisible(False)
                if self._label_items[i] is not None:
                    self._label_items[i].setVisible(False)
                continue

            wired      = i in self.ch_to_idx
            calibrated = self.baselines[i] is not None

            # Real-board pads are always visible so the layout is clear.
            self._pad_items[i].setVisible(True)
            if self._label_items[i] is not None:
                self._label_items[i].setVisible(True)

            pos = CHIP_POS3D[i]
            nrm = CHIP_NRM3D[i]
            centre = (pos + nrm * PAD_LIFT).astype(np.float32)
            lpos   = (pos + nrm * LABEL_LIFT).astype(np.float32)

            # Idle pads (no data yet or still calibrating) → dim grey, no glow
            if not wired or not calibrated:
                grey = np.array([0.55, 0.55, 0.55, 0.90], dtype=np.float32)
                fc = np.full((self._pad_nfaces[i], 4), grey, dtype=np.float32)
                self._pad_items[i].setMeshData(faceColors=fc)
                self._glow_items[i].setVisible(False)
                self._vec_items[i].setVisible(False)
                if self._label_items[i] is not None:
                    state = "…" if wired else "—"
                    self._label_items[i].setData(pos=lpos, text=f"CH{i} {state}")
                continue

            mag   = self.mags[i]
            thr   = self._thr(i)
            touch = self.is_touch[i] or self.flash_left[i] > 0
            if self.flash_left[i] > 0:
                self.flash_left[i] -= 1

            t_frac = min(mag / thr, 1.0)   # 0 = rest, 1 = at/over touch threshold

            # Pad: dark when idle, bright hot colour on contact.
            col = _mag_color(mag, thr)
            fc = np.full((self._pad_nfaces[i], 4), col, dtype=np.float32)
            self._pad_items[i].setMeshData(faceColors=fc)

            # Glow only appears with real contact — expands & brightens with force.
            if t_frac > 0.25:
                self._glow_items[i].setVisible(True)
                g_radius = PAD_R + (GLOW_R - PAD_R) * t_frac
                gv, gf   = _oriented_disk(centre, nrm, g_radius)
                gcol     = col.copy()
                gcol[3]  = 0.65 * t_frac
                gfc      = np.full((len(gf), 4), gcol, dtype=np.float32)
                self._glow_items[i].setMeshData(vertexes=gv, faces=gf, faceColors=gfc)
            else:
                self._glow_items[i].setVisible(False)

            # field vector — base at the pad, tip along the surface normal
            if self.show_vecs and mag > thr * 0.3:
                d   = self.deltas[i]
                u, w = _tangent_basis(nrm)
                vscale = 0.020 / max(thr, 1.0)
                tip = (centre
                       + nrm * (mag * vscale)
                       + u * (d[0] * vscale)
                       + w * (d[1] * vscale)).astype(np.float32)
                vcol = tuple(col.tolist())
                self._vec_items[i].setData(pos=np.stack([centre, tip]), color=vcol)
                self._vec_items[i].setVisible(True)
            else:
                self._vec_items[i].setVisible(False)

            # label — show the hardware channel + live magnitude
            if self._label_items[i] is not None:
                text = f"CH{i}  {mag:.0f}"
                if touch:
                    text += "  ● TOUCH"
                self._label_items[i].setData(pos=lpos, text=text)

    # ─── chart refresh ────────────────────────────────────────────────

    def _refresh_chart(self):
        for i in range(N_CHIPS):
            h = self.mag_hist[i]
            if len(h) < 2:
                continue
            arr = np.array(h)
            self._curves[i].setData(np.arange(len(arr)), arr)

    # ─── status panel refresh ─────────────────────────────────────────

    def _refresh_status(self):
        # Raw-line debug overlay
        if self._reconnecting or self.ser is None:
            self._raw_lbl.setText("⟳ reconnecting… press RESET on ESP32")
            self._raw_lbl.setStyleSheet(
                "color:#FFA726; font-size:8px; font-family:monospace;"
                " background:#1A1A1A; border-radius:3px; padding:4px;")
        elif self._last_raw:
            ok = "✓ parsed" if self._last_parsed else "✗ NOT parsed"
            ok_col = "#66BB6A" if self._last_parsed else "#FF5252"
            self._raw_lbl.setText(f"{self._last_raw}\n[{ok}]")
            self._raw_lbl.setStyleSheet(
                f"color:{ok_col}; font-size:8px; font-family:monospace;"
                " background:#1A1A1A; border-radius:3px; padding:4px;")

        # Active channels = the mux channels we've actually received data on.
        active   = sorted(self.ch_to_idx.keys())   # idx == channel number
        n_seen   = len(active)
        cal_done = sum(1 for ch in active if self.baselines[ch] is not None)

        # Footer: bytes counter + state
        if self._bytes_recv == 0:
            footer = "⚠  0 bytes received — check --port and baud"
            self._footer_lbl.setStyleSheet(
                "color:#FF5252; font-size:9px; font-family:monospace;")
        elif n_seen == 0:
            footer = f"Connected — waiting for first sensor packet…  ({self._bytes_recv} B)"
            self._footer_lbl.setStyleSheet(
                "color:#FFA726; font-size:9px; font-family:monospace;")
        elif cal_done < n_seen:
            prog  = sum(len(self.cal_buf[ch]) for ch in active
                        if self.baselines[ch] is None)
            total = (n_seen - cal_done) * BASELINE_SAMPLES
            footer = (f"Calibrating {cal_done}/{n_seen} chips  "
                      f"({min(prog, total)}/{total} samples)  "
                      f"| {self._bytes_recv} B")
            self._footer_lbl.setStyleSheet(
                "color:#FFA726; font-size:9px; font-family:monospace;")
        else:
            footer = (f"● {n_seen} chips live  |  "
                      f"{self._lines_parsed} pkts  |  R to recalibrate")
            self._footer_lbl.setStyleSheet(
                "color:#66BB6A; font-size:9px; font-family:monospace;")
        self._footer_lbl.setText(footer)

        for i in range(N_CHIPS):
            lbl = self._chip_labels[i]
            if i not in self.ch_to_idx:
                lbl.setText(f"CH{i}  —")
                continue
            if self.baselines[i] is None:
                pct = int(len(self.cal_buf[i]) / BASELINE_SAMPLES * 100)
                lbl.setText(f"CH{i}  cal {pct}%")
            else:
                mag  = self.mags[i]
                flag = "●" if self.is_touch[i] else "○"
                lbl.setText(f"CH{i}  {flag}  |Δ| {mag:5.0f}")

    # ─── actions ──────────────────────────────────────────────────────

    def _recalibrate(self):
        active = sorted(self.ch_to_idx.keys())
        for ch in active:
            self.cal_buf[ch].clear()
            self.baselines[ch]  = None
            self.noise[ch]      = None
            self.deltas[ch]     = np.zeros(3, dtype=np.float32)
            self.mags[ch]       = 0.0
            self.is_touch[ch]   = False
        print(f"Baseline cleared for {len(active)} chips — recalibrating… "
              "(keep hands off the skin)")

    def _toggle_vectors(self):
        self.show_vecs = not self.show_vecs
        for item in self._vec_items:
            item.setVisible(self.show_vecs)
        print(f"Field vectors {'ON' if self.show_vecs else 'OFF'}")

    def _reset_camera(self):
        self.gl_view.setCameraPosition(distance=0.15, elevation=38, azimuth=35)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="3D magnetometer skin touch visualiser")
    p.add_argument("--port",      "-p", default=None)
    p.add_argument("--baud",      "-b", type=int,   default=115200)
    p.add_argument("--stl",       "-s", default=None,
                   help="Board STL path. Omit to auto-generate a placeholder.")
    p.add_argument("--threshold", "-t", type=float, default=0.0,
                   help="Fixed touch threshold |Δ|. 0 = adaptive (auto-tune per chip).")
    p.add_argument("--sensitivity", type=float, default=5.0,
                   help="Adaptive mode: touch fires at noise × this (lower = more sensitive).")
    p.add_argument("--floor", type=float, default=12.0,
                   help="Adaptive mode: minimum touch threshold |Δ| (µT). Raise if pads are twitchy.")
    args = p.parse_args()
    MagSkin3DVisualizer(args).run()


if __name__ == "__main__":
    main()
