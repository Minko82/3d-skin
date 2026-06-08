"""
Shared constants, geometry utilities, FK/xacro parsing, and ray pre-computation
used across the proximity sensor visualizer modules.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np

# ── Sensor / pixel constants ─────────────────────────────────────────
IMAGE_WIDTH = 8
NUM_PIXELS = IMAGE_WIDTH * IMAGE_WIDTH
FOV_DEG = 45.0
INVALID_DEPTH = 4.0         # metres — threshold for invalid sensor readings
MAX_RANGE = 4.0             # metres — maximum sensor range

# ── History / rendering constants ────────────────────────────────────
HISTORY_SENSOR = 10
HISTORY_TIME = 50
HISTORY_TRAIL = 50
TRAIL_MAX_RANGE = 1.0       # metres — trail alpha fade distance
MAX_HISTORY_LEN = max(HISTORY_SENSOR, HISTORY_TIME, HISTORY_TRAIL)
POINT_SIZE = 6
COLOR_SCALE = 0.7

# ── Mesh / skin visual constants ────────────────────────────────────
ROBOT_COLOR = (0.55, 0.55, 0.6, 1.0)
ROBOT_EDGE_COLOR = (0.7, 0.7, 0.7, 0.0)
SKIN_COLOR = (0.25, 0.7, 0.9, 0.3)
SKIN_EDGE_COLOR = (0.4, 0.8, 1.0, 0.1)

# ── Standard FR3 joint poses ────────────────────────────────────────
FR3_READY_POSE = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
FR3_ZERO_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ── Per-sensor hue palette ──────────────────────────────────────────
SENSOR_HUES = [
    np.array([1.0, 0.15, 0.15]),   # 0  red
    np.array([0.15, 1.0, 0.15]),   # 1  green
    np.array([0.15, 0.5, 1.0]),    # 2  blue
    np.array([1.0, 0.15, 1.0]),    # 3  magenta
    np.array([1.0, 0.8, 0.15]),    # 4  yellow
    np.array([0.2, 1.0, 1.0]),     # 5  cyan
    np.array([1.0, 0.5, 0.0]),     # 6  orange
    np.array([0.5, 0.0, 1.0]),     # 7  purple
    np.array([0.0, 1.0, 0.5]),     # 8  mint
    np.array([1.0, 0.4, 0.4]),     # 9  salmon
    np.array([0.8, 1.0, 0.2]),     # 10 lime
    np.array([0.2, 0.6, 1.0]),     # 11 sky blue
    np.array([1.0, 0.2, 0.6]),     # 12 hot pink
    np.array([0.6, 0.8, 0.0]),     # 13 olive
    np.array([0.9, 0.3, 0.9]),     # 14 orchid
    np.array([0.3, 0.9, 0.6]),     # 15 emerald
    np.array([0.9, 0.6, 0.3]),     # 16 tangerine
    np.array([0.3, 0.3, 0.9]),     # 17 indigo
    np.array([0.9, 0.9, 0.3]),     # 18 gold
    np.array([0.3, 0.9, 0.9]),     # 19 teal
    np.array([1.0, 0.6, 0.6]),     # 20 coral
    np.array([0.6, 0.4, 1.0]),     # 21 violet
    np.array([0.4, 1.0, 0.4]),     # 22 spring
    np.array([1.0, 0.8, 0.5]),     # 23 peach
    np.array([0.5, 0.8, 1.0]),     # 24 periwinkle
    np.array([1.0, 0.4, 0.8]),     # 25 fuchsia
    np.array([0.7, 0.3, 0.0]),     # 26 brown
    np.array([0.0, 0.7, 0.3]),     # 27 jade
    np.array([0.5, 0.5, 1.0]),     # 28 lavender
    np.array([1.0, 0.7, 0.7]),     # 29 rose
    np.array([0.4, 0.9, 0.2]),     # 30 chartreuse
]

# Legacy NPZ skin-label counts (independent of xacro sensor counts)
NPZ_SKIN_COUNTS = [
    ('skin1', 7), ('skin2', 5), ('skin3', 4), ('skin5', 4),
    ('skin5_part2', 6), ('skin6', 6), ('skin4', 5),
]

# ── Pre-compute per-zone ray directions (VL53L5CX spherical projection) ─
half_fov = np.radians(FOV_DEG / 2.0)
zone_angles = np.linspace(half_fov, -half_fov, IMAGE_WIDTH)
az_grid, el_grid = np.meshgrid(zone_angles, zone_angles)
ray_x_flat = (np.sin(az_grid) * np.cos(el_grid)).flatten()
ray_y_flat = np.sin(el_grid).flatten()
ray_z_flat = (np.cos(az_grid) * np.cos(el_grid)).flatten()


# ── Geometry utilities ───────────────────────────────────────────────

def rpy_to_rotation_matrix(roll, pitch, yaw):
    """ZYX-intrinsic (URDF convention) rotation matrix from roll/pitch/yaw."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)


def make_transform(xyz, rpy):
    """Build a 4x4 homogeneous transform from translation + RPY (metres)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_rotation_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def rz_matrix(angle):
    """4x4 homogeneous rotation about Z."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=np.float64)


# ── FK chain parsing ────────────────────────────────────────────────

def parse_fk_chain_from_xacro(filepath):
    """Parse joint origins from the FR3 xacro to build the FK chain."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    joint_names = ['fr3_base_joint'] + [f'fr3_joint{i}' for i in range(1, 8)]
    joints_by_name = {}
    for joint in root.iter('joint'):
        name = joint.get('name')
        if name in joint_names:
            origin = joint.find('origin')
            if origin is not None:
                xyz = tuple(float(v) for v in origin.get('xyz', '0 0 0').split())
                rpy = tuple(float(v) for v in origin.get('rpy', '0 0 0').split())
            else:
                xyz, rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            joints_by_name[name] = (name, xyz, rpy)
    missing = [n for n in joint_names if n not in joints_by_name]
    if missing:
        raise RuntimeError(f"Joints not found in {filepath}: {missing}")
    return [joints_by_name[n] for n in joint_names]


def compute_link_transforms(joint_angles, fk_chain):
    """Compute cumulative base->link_i transforms for the 7-DOF FK chain."""
    transforms = []
    T_accum = np.eye(4, dtype=np.float64)
    for idx, (_, xyz, rpy) in enumerate(fk_chain):
        T_accum = T_accum @ make_transform(xyz, rpy)
        if idx > 0:
            T_accum = T_accum @ rz_matrix(joint_angles[idx - 1])
        transforms.append(T_accum.copy())
    return transforms


# ── Skins-xacro parsing ─────────────────────────────────────────────

def _parse_sensors_by_link_from_xacro(root, macro_names, current_macro,
                                       out_sensors, out_meshes, out_dermis):
    tag_local = root.tag.split('}')[-1] if '}' in root.tag else root.tag
    if root.get('name') in macro_names:
        current_macro = root.get('name')
    if root.get('name') == 'mesh_file' and root.get('value') and current_macro:
        val = root.get('value', '')
        if val.startswith('package://proximity_point_motion/'):
            rel = val.replace('package://proximity_point_motion/', '')
            out_meshes[current_macro] = os.path.join(os.path.dirname(__file__), rel)
        else:
            out_meshes[current_macro] = val
    if root.get('sensor_number') is not None and current_macro:
        origin = root.find('origin') or root.find('{*}origin')
        if origin is not None:
            xyz = tuple(float(v) for v in origin.get('xyz', '0 0 0').split())
            rpy = tuple(float(v) for v in origin.get('rpy', '0 0 0').split())
            out_sensors[current_macro].append(
                (int(root.get('sensor_number')), {'xyz': xyz, 'rpy': rpy}))
    if tag_local == 'dermis_base_macro' and current_macro and current_macro not in out_dermis:
        origin = root.find('origin') or root.find('{*}origin')
        if origin is not None:
            xyz = tuple(float(v) for v in origin.get('xyz', '0 0 0').split())
            rpy = tuple(float(v) for v in origin.get('rpy', '0 0 0').split())
            out_dermis[current_macro] = {'xyz': xyz, 'rpy': rpy}
    for child in root:
        _parse_sensors_by_link_from_xacro(
            child, macro_names, current_macro, out_sensors, out_meshes, out_dermis)


def parse_skins_xacro(filepath, macro_names):
    """Return (sensors_by_link, meshes_by_link, dermis_by_link) from skins.xacro."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    out_sensors = {k: [] for k in macro_names}
    out_meshes = {}
    out_dermis = {}
    _parse_sensors_by_link_from_xacro(
        root, macro_names, None, out_sensors, out_meshes, out_dermis)
    for k in out_sensors:
        out_sensors[k].sort(key=lambda t: t[0])
        out_sensors[k] = [s for _, s in out_sensors[k]]
    return out_sensors, out_meshes, out_dermis


# ── NPZ legacy skin mapping ─────────────────────────────────────────

def sensor_id_to_skin(sensor_id):
    """Return skin label for NPZ recording."""
    acc = 0
    for label, count in NPZ_SKIN_COUNTS:
        if sensor_id < acc + count:
            return label
        acc += count
    return NPZ_SKIN_COUNTS[-1][0]


def sensor_id_to_skin_index(sensor_id):
    """Return index within skin for NPZ recording."""
    acc = 0
    for _, count in NPZ_SKIN_COUNTS:
        if sensor_id < acc + count:
            return sensor_id - acc
        acc += count
    return sensor_id - acc
