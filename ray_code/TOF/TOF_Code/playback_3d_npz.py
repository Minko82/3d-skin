"""
Replay NPZ recordings with the same 3D visualization as visualizer_3d_combined:
robot mesh, skin meshes, sensor locations, and point clouds.
"""

import argparse
import os
import time
import numpy as np
import xml.etree.ElementTree as ET
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg
import trimesh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CLI ─────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Replay NPZ recordings with robot/skin visualization")
_parser.add_argument('npz', help='Path to .npz recording file or directory containing .npz files')
_parser.add_argument('--config', '-c', default=None, help='Path to sensors_config.yaml')
_parser.add_argument('--start-pose', choices=['ready', 'zero'], default='ready')
_parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
_parser.add_argument('--color-mode', choices=['sensor', 'time', 'trail'], default='sensor')
_parser.add_argument('--no-labels', action='store_true', help='Hide sensor labels in the scene')
_args = _parser.parse_args()

# ── Constants (match visualizer) ─────────────────────────────────────
IMAGE_WIDTH = 8
NUM_PIXELS = IMAGE_WIDTH * IMAGE_WIDTH
FOV_DEG = 45.0
INVALID_VALUE = 4.0
HISTORY_SENSOR = 6
HISTORY_TIME = 50
HISTORY_TRAIL = 50
TRAIL_MAX_RANGE = 4.0
MAX_HISTORY_LEN = max(HISTORY_SENSOR, HISTORY_TIME, HISTORY_TRAIL)
POINT_SIZE = 10
POINT_ALPHA = 0.8
COLOR_SCALE = 0.7
ROBOT_MESH_DIR = os.path.join(SCRIPT_DIR, 'robot_arms', 'fr3', 'visual')
ROBOT_COLOR = (0.55, 0.55, 0.6, 1.0)
ROBOT_EDGE_COLOR = (0.7, 0.7, 0.7, 0.0)
SKIN_MESH_DIR = os.path.join(SCRIPT_DIR, 'meshes', 'skin')
SKIN_COLOR = (0.25, 0.7, 0.9, 0.3)
SKIN_EDGE_COLOR = (0.4, 0.8, 1.0, 0.1)
SKIN_LINK1_STL = os.path.join(SKIN_MESH_DIR, 'link1_hybrid_dermis.stl')
SKIN_LINK3_STL = os.path.join(SKIN_MESH_DIR, 'link2_hybrid_dermis.stl')
SKIN_LINK2_STL = os.path.join(SKIN_MESH_DIR, 'link3_hybrid_dermis.stl')
SKIN_LINK4_STL = os.path.join(SKIN_MESH_DIR, 'link4_hybrid_dermis.stl')
SKIN_LINK5_STL = os.path.join(SKIN_MESH_DIR, 'link5_part1_hybrid_dermis.stl')
SKIN_LINK5_PART2_STL = os.path.join(SKIN_MESH_DIR, 'link5_part2_hybrid_dermis.stl')
SKIN_LINK6_STL = os.path.join(SKIN_MESH_DIR, 'link6_hybrid_dermis.stl')

SENSOR_HUES = [
    np.array([1.0, 0.15, 0.15]), np.array([0.15, 1.0, 0.15]), np.array([0.15, 0.5, 1.0]),
    np.array([1.0, 0.15, 1.0]), np.array([1.0, 0.8, 0.15]), np.array([0.2, 1.0, 1.0]),
    np.array([1.0, 0.5, 0.0]), np.array([0.5, 0.0, 1.0]), np.array([0.0, 1.0, 0.5]),
    np.array([1.0, 0.4, 0.4]), np.array([0.8, 1.0, 0.2]), np.array([0.2, 0.6, 1.0]),
    np.array([1.0, 0.2, 0.6]), np.array([0.6, 0.8, 0.0]),
    np.array([0.9, 0.3, 0.9]), np.array([0.3, 0.9, 0.6]),
    np.array([0.9, 0.6, 0.3]), np.array([0.3, 0.3, 0.9]),
    np.array([0.9, 0.9, 0.3]), np.array([0.3, 0.9, 0.9]),
    np.array([1.0, 0.6, 0.6]), np.array([0.6, 0.4, 1.0]),
    np.array([0.4, 1.0, 0.4]), np.array([1.0, 0.8, 0.5]),
    np.array([0.5, 0.8, 1.0]), np.array([1.0, 0.4, 0.8]),
]

# ── Parse skins.xacro ───────────────────────────────────────────────
def _parse_sensors_by_link_from_xacro(root, macro_names, current_macro, out_sensors, out_meshes, out_dermis):
    tag_local = root.tag.split('}')[-1] if '}' in root.tag else root.tag
    if root.get('name') in macro_names:
        current_macro = root.get('name')
    if root.get('name') == 'mesh_file' and root.get('value') and current_macro:
        val = root.get('value', '')
        if val.startswith('package://proximity_point_motion/'):
            rel = val.replace('package://proximity_point_motion/', '')
            out_meshes[current_macro] = os.path.join(SCRIPT_DIR, rel)
        else:
            out_meshes[current_macro] = val
    if root.get('sensor_number') is not None and current_macro:
        origin = root.find('origin')
        if origin is None:
            origin = root.find('{*}origin')
        if origin is not None:
            xyz = tuple(float(v) for v in origin.get('xyz', '0 0 0').split())
            rpy = tuple(float(v) for v in origin.get('rpy', '0 0 0').split())
            out_sensors[current_macro].append((int(root.get('sensor_number')), {'xyz': xyz, 'rpy': rpy}))
    if tag_local == 'dermis_base_macro' and current_macro and current_macro not in out_dermis:
        origin = root.find('origin')
        if origin is None:
            origin = root.find('{*}origin')
        if origin is not None:
            xyz = tuple(float(v) for v in origin.get('xyz', '0 0 0').split())
            rpy = tuple(float(v) for v in origin.get('rpy', '0 0 0').split())
            out_dermis[current_macro] = {'xyz': xyz, 'rpy': rpy}
    for child in root:
        _parse_sensors_by_link_from_xacro(child, macro_names, current_macro, out_sensors, out_meshes, out_dermis)

def _parse_skins_xacro(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    macro_names = ('link1_skin', 'link2_skin', 'link3_skin', 'link4_skin', 'link5_skin', 'link5_part2_skin', 'link6_skin')
    out_sensors = {k: [] for k in macro_names}
    out_meshes = {}
    out_dermis = {}
    _parse_sensors_by_link_from_xacro(root, macro_names, None, out_sensors, out_meshes, out_dermis)
    for k in out_sensors:
        out_sensors[k].sort(key=lambda t: t[0])
        out_sensors[k] = [s for _, s in out_sensors[k]]
    return out_sensors, out_meshes, out_dermis

_SKINS_XACRO = os.path.join(SCRIPT_DIR, 'skins.xacro')
_sensors_by_link, _meshes_by_link, _dermis_by_link = _parse_skins_xacro(_SKINS_XACRO)
SENSOR_LINK1_XACRO = _sensors_by_link['link1_skin']
SENSOR_LINK3_XACRO = {i: t for i, t in enumerate(_sensors_by_link['link2_skin'])}
SENSOR_LINK2_XACRO = _sensors_by_link['link3_skin']
SENSOR_LINK4_XACRO = _sensors_by_link['link4_skin']
SENSOR_LINK5_XACRO = _sensors_by_link['link5_skin']
SENSOR_LINK5_PART2_XACRO = _sensors_by_link['link5_part2_skin']
SENSOR_LINK6_XACRO = _sensors_by_link['link6_skin']
NUM_SENSORS_LINK1 = len(SENSOR_LINK1_XACRO)
NUM_SENSORS_LINK3 = len(SENSOR_LINK3_XACRO)
NUM_SENSORS_LINK2 = len(SENSOR_LINK2_XACRO)
NUM_SENSORS_LINK4 = len(SENSOR_LINK4_XACRO)
NUM_SENSORS_LINK5 = len(SENSOR_LINK5_XACRO)
NUM_SENSORS_LINK5_PART2 = len(SENSOR_LINK5_PART2_XACRO)
NUM_SENSORS_LINK6 = len(SENSOR_LINK6_XACRO)
NUM_SENSORS = NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2 + NUM_SENSORS_LINK5 + NUM_SENSORS_LINK5_PART2 + NUM_SENSORS_LINK6 + NUM_SENSORS_LINK4

if _meshes_by_link.get('link1_skin'):
    _skin_link1 = _meshes_by_link['link1_skin']
else:
    _skin_link1 = SKIN_LINK1_STL
if _meshes_by_link.get('link2_skin'):
    _skin_link3 = _meshes_by_link['link2_skin']
else:
    _skin_link3 = SKIN_LINK3_STL
if _meshes_by_link.get('link3_skin'):
    _skin_link2 = _meshes_by_link['link3_skin']
else:
    _skin_link2 = SKIN_LINK2_STL
if _meshes_by_link.get('link5_skin'):
    _skin_link5 = _meshes_by_link['link5_skin']
else:
    _skin_link5 = SKIN_LINK5_STL
if _meshes_by_link.get('link5_part2_skin'):
    _skin_link5_part2 = _meshes_by_link['link5_part2_skin']
else:
    _skin_link5_part2 = SKIN_LINK5_PART2_STL
if _meshes_by_link.get('link4_skin'):
    _skin_link4 = _meshes_by_link['link4_skin']
else:
    _skin_link4 = SKIN_LINK4_STL
if _meshes_by_link.get('link6_skin'):
    _skin_link6 = _meshes_by_link['link6_skin']
else:
    _skin_link6 = SKIN_LINK6_STL

# ── Config ──────────────────────────────────────────────────────────
def _load_config():
    config_path = _args.config or os.path.join(SCRIPT_DIR, 'sensors_config.yaml')
    rotation_offsets, translation_offsets = {}, {}
    enabled_links = {'link1', 'link3', 'link2', 'link4', 'link5', 'link6'}
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            if (cfg or {}).get('enabled_links'):
                enabled_links = {str(n).strip() for n in cfg['enabled_links'] if isinstance(n, str) and n.strip()}
            for k, v in ((cfg or {}).get('rotation_offsets') or {}).items():
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    rotation_offsets[k] = tuple(np.radians(float(x)) for x in v)
            for k, v in ((cfg or {}).get('translation_offsets') or {}).items():
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    translation_offsets[k] = tuple(float(x) / 1000.0 for x in v)
        except Exception:
            pass
    return enabled_links, rotation_offsets, translation_offsets

ENABLED_LINKS, ROTATION_OFFSETS, TRANSLATION_OFFSETS = _load_config()

# ── Ray directions ──────────────────────────────────────────────────
half_fov = np.radians(FOV_DEG / 2.0)
zone_angles = np.linspace(half_fov, -half_fov, IMAGE_WIDTH)
az_grid, el_grid = np.meshgrid(zone_angles, zone_angles)
ray_x = np.sin(az_grid) * np.cos(el_grid)
ray_y = np.sin(el_grid)
ray_z = np.cos(az_grid) * np.cos(el_grid)
ray_x_flat = ray_x.flatten()
ray_y_flat = ray_y.flatten()
ray_z_flat = ray_z.flatten()

# ── Forward kinematics ──────────────────────────────────────────────
def rpy_to_rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr],
    ], dtype=np.float64)

def make_transform(xyz, rpy):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_rotation_matrix(*rpy)
    T[:3, 3] = xyz
    return T

def rz_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

XACRO_PATH = os.path.join(SCRIPT_DIR, 'fr3_full_skin.xacro')
def parse_fk_chain(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    joint_names = ['fr3_base_joint'] + [f'fr3_joint{i}' for i in range(1, 8)]
    joints = {}
    for joint in root.iter('joint'):
        if joint.get('name') in joint_names:
            origin = joint.find('origin')
            if origin is None:
                origin = joint.find('{*}origin')
            xyz = tuple(float(v) for v in origin.get('xyz', '0 0 0').split()) if origin is not None else (0, 0, 0)
            rpy = tuple(float(v) for v in origin.get('rpy', '0 0 0').split()) if origin is not None else (0, 0, 0)
            joints[joint.get('name')] = (joint.get('name'), xyz, rpy)
    return [joints[n] for n in joint_names]

FR3_READY_POSE = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]
FR3_ZERO_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
JOINT_ANGLES = FR3_READY_POSE if _args.start_pose == 'ready' else FR3_ZERO_POSE
FK_CHAIN = parse_fk_chain(XACRO_PATH)

def compute_link_transforms(joint_angles):
    transforms = []
    T = np.eye(4, dtype=np.float64)
    for idx, (_, xyz, rpy) in enumerate(FK_CHAIN):
        T = T @ make_transform(xyz, rpy)
        if idx > 0:
            T = T @ rz_matrix(joint_angles[idx - 1])
        transforms.append(T.copy())
    return transforms

_link_transforms = compute_link_transforms(JOINT_ANGLES)
T_base_to_link1 = _link_transforms[1]
T_base_to_link2 = _link_transforms[2]
T_base_to_link3 = _link_transforms[3]
T_base_to_link4 = _link_transforms[4]
T_base_to_link5 = _link_transforms[5]
T_base_to_link6 = _link_transforms[6]
_d = lambda k: _dermis_by_link.get(k, {'xyz': (0, 0, 0), 'rpy': (0, 0, 0)})
T_dermis_link1_pb = make_transform(_d('link1_skin')['xyz'], _d('link1_skin')['rpy'])
_link2_hybrid_rpy = ROTATION_OFFSETS.get('link2_hybrid', (0, 0, 0))
_link2_hybrid_xyz = TRANSLATION_OFFSETS.get('link2_hybrid', (0, 0, 0))
T_dermis = make_transform(_d('link2_skin')['xyz'], _d('link2_skin')['rpy']) @ make_transform(_link2_hybrid_xyz, _link2_hybrid_rpy)
T_dermis_link3 = make_transform(_d('link3_skin')['xyz'], _d('link3_skin')['rpy'])

def _np4x4_to_qmatrix(T_m):
    m = T_m.astype(float).flatten()
    from pyqtgraph.Qt import QtGui
    return QtGui.QMatrix4x4(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                            m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15])

_sensor_R = {}
_sensor_t = {}
T_skin1_world_pb = T_base_to_link1 @ T_dermis_link1_pb
for _i, _tf in enumerate(SENSOR_LINK1_XACRO):
    _sid = _i
    T_full = T_skin1_world_pb @ make_transform(_tf['xyz'], _tf['rpy'])
    _sensor_R[_sid] = T_full[:3, :3].astype(np.float32)
    _sensor_t[_sid] = T_full[:3, 3].astype(np.float32)
for _sid, _tf in SENSOR_LINK3_XACRO.items():
    _sid_offset = NUM_SENSORS_LINK1 + _sid
    T_full = T_base_to_link2 @ T_dermis @ make_transform(_tf['xyz'], _tf['rpy'])
    _sensor_R[_sid_offset] = T_full[:3, :3].astype(np.float32)
    _sensor_t[_sid_offset] = T_full[:3, 3].astype(np.float32)
T_skin3_world = T_base_to_link3 @ T_dermis_link3
for _i, _tf in enumerate(SENSOR_LINK2_XACRO):
    _sid = NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + _i
    T_full = T_skin3_world @ make_transform(_tf['xyz'], _tf['rpy'])
    _sensor_R[_sid] = T_full[:3, :3].astype(np.float32)
    _sensor_t[_sid] = T_full[:3, 3].astype(np.float32)
for _i, _tf in enumerate(SENSOR_LINK5_XACRO):
    _sid = NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2 + _i
    T_full = T_base_to_link5 @ make_transform(_tf['xyz'], _tf['rpy'])
    _sensor_R[_sid] = T_full[:3, :3].astype(np.float32)
    _sensor_t[_sid] = T_full[:3, 3].astype(np.float32)
for _i, _tf in enumerate(SENSOR_LINK5_PART2_XACRO):
    _sid = NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2 + NUM_SENSORS_LINK5 + _i
    T_full = T_base_to_link5 @ make_transform(_tf['xyz'], _tf['rpy'])
    _sensor_R[_sid] = T_full[:3, :3].astype(np.float32)
    _sensor_t[_sid] = T_full[:3, 3].astype(np.float32)
_link6_hybrid_rpy_pb = ROTATION_OFFSETS.get('link6_hybrid', (0, 0, 0))
_link6_hybrid_xyz_pb = TRANSLATION_OFFSETS.get('link6_hybrid', (0, 0, 0))
T_dermis_link6_pb = make_transform(_d('link6_skin')['xyz'], _d('link6_skin')['rpy']) @ make_transform(_link6_hybrid_xyz_pb, _link6_hybrid_rpy_pb)
T_skin6_world_pb = T_base_to_link6 @ T_dermis_link6_pb
for _i, _tf in enumerate(SENSOR_LINK6_XACRO):
    _sid = NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2 + NUM_SENSORS_LINK5 + NUM_SENSORS_LINK5_PART2 + _i
    T_full = T_skin6_world_pb @ make_transform(_tf['xyz'], _tf['rpy'])
    _sensor_R[_sid] = T_full[:3, :3].astype(np.float32)
    _sensor_t[_sid] = T_full[:3, 3].astype(np.float32)
T_dermis_link4_pb = make_transform(_d('link4_skin')['xyz'], _d('link4_skin')['rpy'])
T_skin4_world_pb = T_base_to_link4 @ T_dermis_link4_pb
for _i, _tf in enumerate(SENSOR_LINK4_XACRO):
    _sid = NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2 + NUM_SENSORS_LINK5 + NUM_SENSORS_LINK5_PART2 + NUM_SENSORS_LINK6 + _i
    T_full = T_skin4_world_pb @ make_transform(_tf['xyz'], _tf['rpy'])
    _sensor_R[_sid] = T_full[:3, :3].astype(np.float32)
    _sensor_t[_sid] = T_full[:3, 3].astype(np.float32)

def _sensor_id_to_name(s):
    if s < NUM_SENSORS_LINK1:
        return f"L1-{s}"
    elif s < NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3:
        return f"L2-{s - NUM_SENSORS_LINK1}"
    elif s < NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2:
        return f"L3-{s - NUM_SENSORS_LINK1 - NUM_SENSORS_LINK3}"
    elif s < NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2 + NUM_SENSORS_LINK5:
        return f"L5-{s - NUM_SENSORS_LINK1 - NUM_SENSORS_LINK3 - NUM_SENSORS_LINK2}"
    elif s < NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2 + NUM_SENSORS_LINK5 + NUM_SENSORS_LINK5_PART2:
        return f"L5P2-{s - NUM_SENSORS_LINK1 - NUM_SENSORS_LINK3 - NUM_SENSORS_LINK2 - NUM_SENSORS_LINK5}"
    elif s < NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2 + NUM_SENSORS_LINK5 + NUM_SENSORS_LINK5_PART2 + NUM_SENSORS_LINK6:
        return f"L6-{s - NUM_SENSORS_LINK1 - NUM_SENSORS_LINK3 - NUM_SENSORS_LINK2 - NUM_SENSORS_LINK5 - NUM_SENSORS_LINK5_PART2}"
    else:
        return f"L4-{s - NUM_SENSORS_LINK1 - NUM_SENSORS_LINK3 - NUM_SENSORS_LINK2 - NUM_SENSORS_LINK5 - NUM_SENSORS_LINK5_PART2 - NUM_SENSORS_LINK6}"

COLOR_MODE = _args.color_mode
def _initial_history_len(mode):
    if mode == 'time': return HISTORY_TIME
    if mode == 'trail': return HISTORY_TRAIL
    return HISTORY_SENSOR
HISTORY_LEN = _initial_history_len(COLOR_MODE)
MAX_POINTS_PER_SENSOR = MAX_HISTORY_LEN * NUM_PIXELS

def build_colormap_sensor(distances, sensor_id):
    colors = np.zeros((len(distances), 4), dtype=np.float32)
    valid = distances < INVALID_VALUE
    if not np.any(valid):
        return colors
    d = distances[valid].astype(np.float32)
    lo, hi = d.min(), d.max()
    t = np.zeros_like(d) if hi - lo < 1e-3 else (d - lo) / (hi - lo)
    hue = SENSOR_HUES[sensor_id % len(SENSOR_HUES)]
    brightness = (0.3 + 0.7 * (1.0 - t)) * COLOR_SCALE
    colors[valid, 0] = hue[0] * brightness
    colors[valid, 1] = hue[1] * brightness
    colors[valid, 2] = hue[2] * brightness
    colors[valid, 3] = POINT_ALPHA
    return colors

def build_colormap_time(distances, _sensor_id=None):
    colors = np.zeros((len(distances), 4), dtype=np.float32)
    valid = distances < INVALID_VALUE
    if not np.any(valid):
        return colors
    d = distances[valid].astype(np.float32)
    lo, hi = d.min(), d.max()
    t = np.zeros_like(d) if hi - lo < 1e-3 else (d - lo) / (hi - lo)
    colors[valid, 0] = (1.0 - t) * COLOR_SCALE
    colors[valid, 1] = (1.0 - np.abs(t - 0.5) * 2) * COLOR_SCALE
    colors[valid, 2] = t * COLOR_SCALE
    colors[valid, 3] = POINT_ALPHA
    return colors

def build_colormap_trail(distances, sensor_id):
    colors = np.zeros((len(distances), 4), dtype=np.float32)
    valid = distances < INVALID_VALUE
    if not np.any(valid):
        return colors
    d = distances[valid].astype(np.float32)
    lo, hi = d.min(), d.max()
    t = np.zeros_like(d) if hi - lo < 1e-3 else (d - lo) / (hi - lo)
    hue = SENSOR_HUES[sensor_id % len(SENSOR_HUES)]
    brightness = (0.3 + 0.7 * (1.0 - t)) * COLOR_SCALE
    colors[valid, 0] = hue[0] * brightness
    colors[valid, 1] = hue[1] * brightness
    colors[valid, 2] = hue[2] * brightness
    colors[valid, 3] = np.clip(1.0 - d / TRAIL_MAX_RANGE, 0.0, 1.0) * POINT_ALPHA
    return colors

def build_colormap(distances, sensor_id):
    if COLOR_MODE == 'time':
        return build_colormap_time(distances, sensor_id)
    if COLOR_MODE == 'trail':
        return build_colormap_trail(distances, sensor_id)
    return build_colormap_sensor(distances, sensor_id)

def packet_to_points(packet, sensor_id):
    if len(packet) < 1 + NUM_PIXELS * 2:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    raw_mm = np.frombuffer(packet[1:1 + NUM_PIXELS * 2], dtype=np.uint16)
    dist = raw_mm.astype(np.float32) / 1000.0
    valid = dist < INVALID_VALUE
    x = dist * ray_x_flat
    y = dist * ray_y_flat
    z = dist * ray_z_flat
    x[~valid] = 0
    y[~valid] = 0
    z[~valid] = 0
    pos_local = np.column_stack([x, y, z])
    pos = (pos_local @ _sensor_R[sensor_id].T) + _sensor_t[sensor_id] if sensor_id in _sensor_R else pos_local
    colors = build_colormap(dist, sensor_id)
    return pos.astype(np.float32), colors

# ── Load NPZ ─────────────────────────────────────────────────────────
FRAME_GAP_NS = 30_000_000  # 30 ms — packets within this gap are one frame

def load_npz(path):
    data = np.load(path, allow_pickle=False)
    timestamps = np.asarray(data['timestamps'])
    packets = data['packets']
    if 'sensor_ids' in data:
        sensor_ids = np.asarray(data['sensor_ids'])
    else:
        print("Warning: NPZ has no sensor_ids; assuming sensor 0")
        sensor_ids = np.zeros(len(timestamps), dtype=np.uint8)
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    sensor_ids = sensor_ids[order]
    packets = packets[order] if packets.ndim == 2 else np.array([packets[i] for i in order])

    # Build frame boundaries: a frame = burst of packets with gaps < FRAME_GAP_NS
    frame_starts = [0]
    for i in range(1, len(timestamps)):
        if int(timestamps[i]) - int(timestamps[i - 1]) > FRAME_GAP_NS:
            frame_starts.append(i)
    frame_starts.append(len(timestamps))  # sentinel

    return timestamps, sensor_ids, packets, frame_starts

# ── Episode discovery ────────────────────────────────────────────────
def _discover_episodes(path):
    """Find all .npz files in the same directory, sorted by name."""
    d = os.path.dirname(path) or '.'
    npz_files = sorted(
        os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith('.npz')
    )
    if not npz_files:
        npz_files = [path]
    return npz_files


# ── Main ────────────────────────────────────────────────────────────
def main():
    npz_path = os.path.abspath(_args.npz)
    if os.path.isdir(npz_path):
        npz_files = sorted(
            os.path.join(npz_path, f) for f in os.listdir(npz_path) if f.lower().endswith('.npz')
        )
        if not npz_files:
            print(f"Error: no .npz files in {npz_path}")
            return 1
        npz_path = npz_files[0]
    elif not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found")
        return 1

    episodes = _discover_episodes(npz_path)
    ep_idx = [episodes.index(npz_path) if npz_path in episodes else 0]

    timestamps, sensor_ids, packets, frame_starts = load_npz(npz_path)
    if len(timestamps) == 0:
        print("Error: NPZ contains no packets")
        return 1

    t0_ns = int(timestamps[0])
    duration_ns = int(timestamps[-1]) - t0_ns
    num_frames = len(frame_starts) - 1  # last entry is sentinel
    print(f"Loaded {len(timestamps)} packets in {num_frames} frames, duration {duration_ns / 1e9:.2f}s")

    # Mutable state shared by update/callbacks
    state = {
        'timestamps': timestamps,
        'sensor_ids': sensor_ids,
        'packets': packets,
        'frame_starts': frame_starts,
        'num_frames': num_frames,
        't0_ns': t0_ns,
        'duration_ns': duration_ns,
        'index': 0,
        'start_time': None,
        'paused': False,
        'speed': _args.speed,
        'frame': 0,
    }

    app = QtWidgets.QApplication([])
    win = gl.GLViewWidget()
    win.setGeometry(100, 100, 1400, 900)
    win.setCameraPosition(distance=2.0, elevation=25, azimuth=-45)
    win.show()

    grid = gl.GLGridItem()
    grid.setSize(6.0, 4.0, 1)
    grid.setSpacing(0.5, 0.5, 1)
    win.addItem(grid)
    axis = gl.GLAxisItem()
    axis.setSize(1.0, 1.0, 1.0)
    win.addItem(axis)

    # Robot meshes (match visualizer)
    for i in range(8):
        mesh_path = os.path.join(ROBOT_MESH_DIR, f'link{i}.dae')
        if not os.path.exists(mesh_path):
            continue
        raw = trimesh.load(mesh_path, force='mesh')
        verts = np.array(raw.vertices, dtype=np.float32)
        faces = np.array(raw.faces, dtype=np.uint32)
        fc = np.full((len(faces), 4), ROBOT_COLOR, dtype=np.float32)
        item = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=fc,
                            edgeColor=ROBOT_EDGE_COLOR, drawEdges=True, smooth=True, glOptions='translucent')
        item.setTransform(_np4x4_to_qmatrix(_link_transforms[i]))
        win.addItem(item)
    print(f"Robot loaded from {ROBOT_MESH_DIR}")

    # Skin meshes (match visualizer)
    if 'link1' in ENABLED_LINKS and os.path.exists(_skin_link1):
        T_skin1 = T_base_to_link1 @ T_dermis_link1_pb
        raw = trimesh.load(_skin_link1, force='mesh')
        verts = np.array(raw.vertices, dtype=np.float32) / 1000.0
        faces = np.array(raw.faces, dtype=np.uint32)
        fc = np.full((len(faces), 4), SKIN_COLOR, dtype=np.float32)
        item = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=fc,
                             edgeColor=SKIN_EDGE_COLOR, drawEdges=True, smooth=True, glOptions='translucent')
        item.setTransform(_np4x4_to_qmatrix(T_skin1))
        win.addItem(item)
        _link1_pos_m = np.array([tf['xyz'] for tf in SENSOR_LINK1_XACRO], dtype=np.float64)
        _link1_pos = (T_skin1[:3, :3] @ _link1_pos_m.T).T + T_skin1[:3, 3]
        scatter = gl.GLScatterPlotItem(pos=_link1_pos.astype(np.float32),
            color=np.array([[h[0], h[1], h[2], 1.0] for h in [SENSOR_HUES[i % len(SENSOR_HUES)] for i in range(len(SENSOR_LINK1_XACRO))]], dtype=np.float32),
            size=12, pxMode=True, glOptions='translucent')
        win.addItem(scatter)
    if 'link3' in ENABLED_LINKS and os.path.exists(_skin_link3):
        raw = trimesh.load(_skin_link3, force='mesh')
        verts = np.array(raw.vertices, dtype=np.float32) / 1000.0
        faces = np.array(raw.faces, dtype=np.uint32)
        fc = np.full((len(faces), 4), SKIN_COLOR, dtype=np.float32)
        item = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=fc,
                             edgeColor=SKIN_EDGE_COLOR, drawEdges=True, smooth=True, glOptions='translucent')
        item.setTransform(_np4x4_to_qmatrix(T_base_to_link2 @ T_dermis))
        win.addItem(item)
    if 'link2' in ENABLED_LINKS and os.path.exists(_skin_link2):
        T_skin2 = T_base_to_link3 @ T_dermis_link3
        raw = trimesh.load(_skin_link2, force='mesh')
        verts = np.array(raw.vertices, dtype=np.float32) / 1000.0
        faces = np.array(raw.faces, dtype=np.uint32)
        fc = np.full((len(faces), 4), SKIN_COLOR, dtype=np.float32)
        item = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=fc,
                             edgeColor=SKIN_EDGE_COLOR, drawEdges=True, smooth=True, glOptions='translucent')
        item.setTransform(_np4x4_to_qmatrix(T_skin2))
        win.addItem(item)
        _link2_pos_m = np.array([tf['xyz'] for tf in SENSOR_LINK2_XACRO], dtype=np.float64)
        _link2_pos = (T_skin2[:3, :3] @ _link2_pos_m.T).T + T_skin2[:3, 3]
        scatter = gl.GLScatterPlotItem(pos=_link2_pos.astype(np.float32),
            color=np.array([[h[0], h[1], h[2], 1.0] for h in [SENSOR_HUES[i % len(SENSOR_HUES)] for i in range(len(SENSOR_LINK2_XACRO))]], dtype=np.float32),
            size=12, pxMode=True, glOptions='translucent')
        win.addItem(scatter)
    if 'link5' in ENABLED_LINKS and os.path.exists(_skin_link5):
        T_skin5 = T_base_to_link5
        raw = trimesh.load(_skin_link5, force='mesh')
        verts = np.array(raw.vertices, dtype=np.float32) / 1000.0
        faces = np.array(raw.faces, dtype=np.uint32)
        fc = np.full((len(faces), 4), SKIN_COLOR, dtype=np.float32)
        item = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=fc,
                             edgeColor=SKIN_EDGE_COLOR, drawEdges=True, smooth=True, glOptions='translucent')
        item.setTransform(_np4x4_to_qmatrix(T_skin5))
        win.addItem(item)
        _link5_pos_m = np.array([tf['xyz'] for tf in SENSOR_LINK5_XACRO], dtype=np.float64)
        _link5_pos = (T_skin5[:3, :3] @ _link5_pos_m.T).T + T_skin5[:3, 3]
        scatter = gl.GLScatterPlotItem(pos=_link5_pos.astype(np.float32),
            color=np.array([[h[0], h[1], h[2], 1.0] for h in [SENSOR_HUES[i % len(SENSOR_HUES)] for i in range(len(SENSOR_LINK5_XACRO))]], dtype=np.float32),
            size=12, pxMode=True, glOptions='translucent')
        win.addItem(scatter)
    if 'link5' in ENABLED_LINKS and os.path.exists(_skin_link5_part2):
        T_skin5p2 = T_base_to_link5
        raw = trimesh.load(_skin_link5_part2, force='mesh')
        verts = np.array(raw.vertices, dtype=np.float32) / 1000.0
        faces = np.array(raw.faces, dtype=np.uint32)
        fc = np.full((len(faces), 4), SKIN_COLOR, dtype=np.float32)
        item = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=fc,
                             edgeColor=SKIN_EDGE_COLOR, drawEdges=True, smooth=True, glOptions='translucent')
        item.setTransform(_np4x4_to_qmatrix(T_skin5p2))
        win.addItem(item)
        _hue_offset = NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2 + NUM_SENSORS_LINK5
        _link5p2_pos_m = np.array([tf['xyz'] for tf in SENSOR_LINK5_PART2_XACRO], dtype=np.float64)
        _link5p2_pos = (T_skin5p2[:3, :3] @ _link5p2_pos_m.T).T + T_skin5p2[:3, 3]
        scatter = gl.GLScatterPlotItem(pos=_link5p2_pos.astype(np.float32),
            color=np.array([[h[0], h[1], h[2], 1.0] for h in [SENSOR_HUES[(_hue_offset + i) % len(SENSOR_HUES)] for i in range(len(SENSOR_LINK5_PART2_XACRO))]], dtype=np.float32),
            size=12, pxMode=True, glOptions='translucent')
        win.addItem(scatter)
    if 'link6' in ENABLED_LINKS and os.path.exists(_skin_link6):
        T_skin6 = T_skin6_world_pb
        raw = trimesh.load(_skin_link6, force='mesh')
        verts = np.array(raw.vertices, dtype=np.float32) / 1000.0
        faces = np.array(raw.faces, dtype=np.uint32)
        fc = np.full((len(faces), 4), SKIN_COLOR, dtype=np.float32)
        item = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=fc,
                             edgeColor=SKIN_EDGE_COLOR, drawEdges=True, smooth=True, glOptions='translucent')
        item.setTransform(_np4x4_to_qmatrix(T_skin6))
        win.addItem(item)
        _l6_hue_offset = NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2 + NUM_SENSORS_LINK5 + NUM_SENSORS_LINK5_PART2
        _link6_pos_m = np.array([tf['xyz'] for tf in SENSOR_LINK6_XACRO], dtype=np.float64)
        _link6_pos = (T_skin6[:3, :3] @ _link6_pos_m.T).T + T_skin6[:3, 3]
        scatter = gl.GLScatterPlotItem(pos=_link6_pos.astype(np.float32),
            color=np.array([[h[0], h[1], h[2], 1.0] for h in [SENSOR_HUES[(_l6_hue_offset + i) % len(SENSOR_HUES)] for i in range(len(SENSOR_LINK6_XACRO))]], dtype=np.float32),
            size=12, pxMode=True, glOptions='translucent')
        win.addItem(scatter)
    if 'link4' in ENABLED_LINKS and os.path.exists(_skin_link4):
        T_skin4 = T_base_to_link4 @ T_dermis_link4_pb
        raw = trimesh.load(_skin_link4, force='mesh')
        verts = np.array(raw.vertices, dtype=np.float32) / 1000.0
        faces = np.array(raw.faces, dtype=np.uint32)
        fc = np.full((len(faces), 4), SKIN_COLOR, dtype=np.float32)
        item = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=fc,
                             edgeColor=SKIN_EDGE_COLOR, drawEdges=True, smooth=True, glOptions='translucent')
        item.setTransform(_np4x4_to_qmatrix(T_skin4))
        win.addItem(item)
        _l4_hue_offset = NUM_SENSORS_LINK1 + NUM_SENSORS_LINK3 + NUM_SENSORS_LINK2 + NUM_SENSORS_LINK5 + NUM_SENSORS_LINK5_PART2 + NUM_SENSORS_LINK6
        _link4_pos_m = np.array([tf['xyz'] for tf in SENSOR_LINK4_XACRO], dtype=np.float64)
        _link4_pos = (T_skin4[:3, :3] @ _link4_pos_m.T).T + T_skin4[:3, 3]
        scatter = gl.GLScatterPlotItem(pos=_link4_pos.astype(np.float32),
            color=np.array([[h[0], h[1], h[2], 1.0] for h in [SENSOR_HUES[(_l4_hue_offset + i) % len(SENSOR_HUES)] for i in range(len(SENSOR_LINK4_XACRO))]], dtype=np.float32),
            size=12, pxMode=True, glOptions='translucent')
        win.addItem(scatter)
    print("Skin meshes and sensor markers loaded")

    # Sensor labels (all sensors)
    for s in range(NUM_SENSORS):
        if s not in _sensor_t:
            continue
        hue = SENSOR_HUES[s % len(SENSOR_HUES)]
        pos = _sensor_t[s].copy()
        pos[1] += 0.02
        if not _args.no_labels:
            lbl = gl.GLTextItem(pos=pos, text=_sensor_id_to_name(s),
                                color=pg.mkColor(int(hue[0]*255), int(hue[1]*255), int(hue[2]*255)))
            lbl.setData(font=pg.QtGui.QFont("Helvetica", 16))
            win.addItem(lbl)

    # Point cloud scatters (match visualizer)
    scatters = []
    _hist_pos = [np.zeros((MAX_HISTORY_LEN, NUM_PIXELS, 3), dtype=np.float32) for _ in range(NUM_SENSORS)]
    _hist_col = [np.zeros((MAX_HISTORY_LEN, NUM_PIXELS, 4), dtype=np.float32) for _ in range(NUM_SENSORS)]
    _hist_count = [0] * NUM_SENSORS
    _combined_pos = [np.zeros((MAX_POINTS_PER_SENSOR, 3), dtype=np.float32) for _ in range(NUM_SENSORS)]
    _combined_col = [np.zeros((MAX_POINTS_PER_SENSOR, 4), dtype=np.float32) for _ in range(NUM_SENSORS)]
    for s in range(NUM_SENSORS):
        scatter = gl.GLScatterPlotItem(pos=_combined_pos[s], color=_combined_col[s],
                                      size=POINT_SIZE, pxMode=True, glOptions='translucent')
        win.addItem(scatter)
        scatters.append(scatter)

    def _current_frame_idx():
        """Return which frame the current packet index falls in."""
        fs = state['frame_starts']
        idx = state['index']
        lo, hi = 0, len(fs) - 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fs[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _update_title():
        ts = state['timestamps']
        total_pkts = len(ts)
        pkt_idx = min(state['index'], total_pkts - 1) if total_pkts else 0
        elapsed_s = (int(ts[pkt_idx]) - state['t0_ns']) / 1e9 if total_pkts else 0.0
        dur_s = state['duration_ns'] / 1e9
        cur_frame = _current_frame_idx()
        nf = state['num_frames']
        prefix = "[PAUSED] " if state['paused'] else ""
        ep_label = f"Ep {ep_idx[0]+1}/{len(episodes)}" if len(episodes) > 1 else ""
        ep_sep = " | " if ep_label else ""
        fname = os.path.basename(episodes[ep_idx[0]])
        win.setWindowTitle(
            f"{prefix}{fname} | "
            f"Frame {cur_frame + 1}/{nf} | "
            f"{elapsed_s:.1f}s / {dur_s:.1f}s | "
            f"Speed {state['speed']:.1f}x"
            f"{ep_sep}{ep_label}"
        )

    def _reset_playback():
        """Clear history buffers and reset playback counters."""
        state['index'] = 0
        state['start_time'] = None
        state['frame'] = 0
        for s in range(NUM_SENSORS):
            _hist_count[s] = 0
            _hist_pos[s][:] = 0
            _hist_col[s][:] = 0
            _combined_pos[s][:] = 0
            _combined_col[s][:] = 0
            scatters[s].setData(pos=_combined_pos[s][:0], color=_combined_col[s][:0])
        _update_title()

    def _load_episode(new_idx):
        """Load a new episode by index and reset playback."""
        new_idx = new_idx % len(episodes)
        ep_idx[0] = new_idx
        path = episodes[new_idx]
        ts, sids, pkts, fstarts = load_npz(path)
        if len(ts) == 0:
            print(f"Warning: {path} contains no packets, skipping")
            return
        state['timestamps'] = ts
        state['sensor_ids'] = sids
        state['packets'] = pkts
        state['frame_starts'] = fstarts
        state['num_frames'] = len(fstarts) - 1
        state['t0_ns'] = int(ts[0])
        state['duration_ns'] = int(ts[-1]) - state['t0_ns']
        nf = state['num_frames']
        print(f"Episode {new_idx+1}/{len(episodes)}: {os.path.basename(path)} "
              f"({len(ts)} packets, {nf} frames, {state['duration_ns']/1e9:.2f}s)")
        _reset_playback()

    _update_title()

    def _render_packet(idx):
        """Process and render a single packet by its index."""
        sids = state['sensor_ids']
        pkts = state['packets']
        sensor_id = int(sids[idx])
        if sensor_id >= NUM_SENSORS:
            return
        pkt = np.asarray(pkts[idx], dtype=np.uint8)
        if pkt.ndim > 1:
            pkt = pkt.flatten()
        if len(pkt) < 1 + NUM_PIXELS * 2:
            return
        pos, colors = packet_to_points(pkt, sensor_id)
        if pos.shape[0] == 0:
            return
        if _hist_count[sensor_id] == HISTORY_LEN:
            _hist_pos[sensor_id][:-1] = _hist_pos[sensor_id][1:]
            _hist_col[sensor_id][:-1] = _hist_col[sensor_id][1:]
        else:
            _hist_count[sensor_id] += 1
        n = _hist_count[sensor_id]
        _hist_pos[sensor_id][n - 1] = pos
        _hist_col[sensor_id][n - 1] = colors
        age_frac = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(n, 1, 1) if n > 1 else np.ones((1, 1, 1), dtype=np.float32)
        alpha_mult = 0.1 + 0.9 * age_frac
        src = _hist_col[sensor_id][:n]
        total = n * NUM_PIXELS
        if COLOR_MODE == 'time':
            blue_shift = 1.0 - age_frac
            _combined_col[sensor_id][:total, 0] = (src[:, :, 0] * age_frac[:, :, 0]).ravel()
            _combined_col[sensor_id][:total, 1] = (src[:, :, 1] * age_frac[:, :, 0]).ravel()
            _combined_col[sensor_id][:total, 2] = (src[:, :, 2] + (1.0 - src[:, :, 2]) * blue_shift[:, :, 0]).ravel()
            _combined_col[sensor_id][:total, 3] = (src[:, :, 3] * alpha_mult[:, :, 0]).ravel()
        elif COLOR_MODE == 'trail':
            _combined_col[sensor_id][:total, 0] = (src[:, :, 0] * age_frac[:, :, 0]).ravel()
            _combined_col[sensor_id][:total, 1] = (src[:, :, 1] * age_frac[:, :, 0]).ravel()
            _combined_col[sensor_id][:total, 2] = (src[:, :, 2] * age_frac[:, :, 0]).ravel()
            _combined_col[sensor_id][:total, 3] = (src[:, :, 3] * alpha_mult[:, :, 0]).ravel()
        else:
            _combined_col[sensor_id][:total, 0] = (src[:, :, 0] * age_frac[:, :, 0]).ravel()
            _combined_col[sensor_id][:total, 1] = (src[:, :, 1] * age_frac[:, :, 0]).ravel()
            _combined_col[sensor_id][:total, 2] = (src[:, :, 2] * age_frac[:, :, 0]).ravel()
            _combined_col[sensor_id][:total, 3] = (src[:, :, 3] * alpha_mult[:, :, 0]).ravel()
        _combined_pos[sensor_id][:total] = _hist_pos[sensor_id][:n].reshape(total, 3)
        scatters[sensor_id].setData(pos=_combined_pos[sensor_id][:total], color=_combined_col[sensor_id][:total])

    def _play_frame(frame_idx):
        """Render all packets in a single frame."""
        fs = state['frame_starts']
        if frame_idx < 0 or frame_idx >= state['num_frames']:
            return
        start_pkt = fs[frame_idx]
        end_pkt = fs[frame_idx + 1]
        for pkt_i in range(start_pkt, end_pkt):
            _render_packet(pkt_i)
        state['index'] = end_pkt

    def update():
        if state['paused']:
            return
        if state['start_time'] is None:
            state['start_time'] = time.perf_counter()
        ts = state['timestamps']
        elapsed_s = (time.perf_counter() - state['start_time']) * state['speed']
        current_t = state['t0_ns'] + int(elapsed_s * 1e9)
        while state['index'] < len(ts) and int(ts[state['index']]) <= current_t:
            _render_packet(state['index'])
            state['index'] += 1
        _update_title()
        if state['index'] >= len(ts):
            _reset_playback()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(16)

    # ── Keyboard shortcuts ───────────────────────────────────────────
    def toggle_pause():
        state['paused'] = not state['paused']
        if not state['paused']:
            state['start_time'] = time.perf_counter() - (
                (int(state['timestamps'][min(state['index'], len(state['timestamps'])-1)]) - state['t0_ns'])
                / 1e9 / state['speed']
            ) if state['index'] > 0 else None
        _update_title()

    def speed_up():
        state['speed'] = min(state['speed'] * 2.0, 64.0)
        state['start_time'] = time.perf_counter() - (
            (int(state['timestamps'][min(state['index'], len(state['timestamps'])-1)]) - state['t0_ns'])
            / 1e9 / state['speed']
        ) if state['index'] > 0 else None
        print(f"Speed: {state['speed']:.1f}x")
        _update_title()

    def speed_down():
        state['speed'] = max(state['speed'] / 2.0, 0.125)
        state['start_time'] = time.perf_counter() - (
            (int(state['timestamps'][min(state['index'], len(state['timestamps'])-1)]) - state['t0_ns'])
            / 1e9 / state['speed']
        ) if state['index'] > 0 else None
        print(f"Speed: {state['speed']:.1f}x")
        _update_title()

    def next_episode():
        if len(episodes) <= 1:
            print("No other episodes in this directory")
            return
        _load_episode(ep_idx[0] + 1)

    def prev_episode():
        if len(episodes) <= 1:
            print("No other episodes in this directory")
            return
        _load_episode(ep_idx[0] - 1)

    def step_forward():
        """Advance one frame while paused."""
        if not state['paused']:
            return
        fi = _current_frame_idx()
        next_fi = fi + 1
        if next_fi >= state['num_frames']:
            return
        _play_frame(next_fi)
        _update_title()

    def step_backward():
        """Go back one frame while paused (replays from start up to that frame)."""
        if not state['paused']:
            return
        fi = _current_frame_idx()
        if fi <= 0:
            return
        target = fi - 1
        for s in range(NUM_SENSORS):
            _hist_count[s] = 0
            _hist_pos[s][:] = 0
            _hist_col[s][:] = 0
        replay_from = max(0, target - HISTORY_LEN + 1)
        for f in range(replay_from, target + 1):
            _play_frame(f)
        _update_title()

    pg.QtGui.QShortcut(pg.QtGui.QKeySequence(' '), win).activated.connect(toggle_pause)
    pg.QtGui.QShortcut(pg.QtGui.QKeySequence('+'), win).activated.connect(speed_up)
    pg.QtGui.QShortcut(pg.QtGui.QKeySequence('='), win).activated.connect(speed_up)
    pg.QtGui.QShortcut(pg.QtGui.QKeySequence('-'), win).activated.connect(speed_down)
    pg.QtGui.QShortcut(pg.QtGui.QKeySequence(']'), win).activated.connect(next_episode)
    pg.QtGui.QShortcut(pg.QtGui.QKeySequence('['), win).activated.connect(prev_episode)
    pg.QtGui.QShortcut(pg.QtGui.QKeySequence('.'), win).activated.connect(step_forward)
    pg.QtGui.QShortcut(pg.QtGui.QKeySequence(','), win).activated.connect(step_backward)
    pg.QtGui.QShortcut(pg.QtGui.QKeySequence('R'), win).activated.connect(_reset_playback)

    print(f"Playback started (speed={state['speed']:.1f}x, {state['num_frames']} frames)")
    print(f"  Space=pause  +/-=speed  [/]=episodes  .=next frame  ,=prev frame  R=restart")
    if len(episodes) > 1:
        print(f"  {len(episodes)} episodes found in {os.path.dirname(episodes[0])}")
    app.exec()
    return 0

if __name__ == '__main__':
    exit(main() or 0)
