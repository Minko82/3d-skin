"""
ProximitySuit — manages all skins, config, UDP data, and recording.
"""

import os
import socket
import struct
import threading
import time
from datetime import datetime
import numpy as np

try:
    from mcap.writer import Writer as McapWriter
    MCAP_AVAILABLE = True
except ImportError:
    MCAP_AVAILABLE = False

from rosbags.typesys import get_typestore, Stores
from ros.ros2_schemas import FLOAT32_MULTIARRAY_SCHEMA, POINTCLOUD2_SCHEMA, TFMESSAGE_SCHEMA

from proximity_common import (
    IMAGE_WIDTH, NUM_PIXELS, INVALID_DEPTH,
    FR3_READY_POSE, FR3_ZERO_POSE,
    NPZ_SKIN_COUNTS,
    ray_x_flat, ray_y_flat, ray_z_flat,
    make_transform, parse_fk_chain_from_xacro, compute_link_transforms,
    parse_skins_xacro, sensor_id_to_skin, sensor_id_to_skin_index,
)
from skin import Skin
from proximity_visualizer import ProximityVisualizer

MULTICAST_GROUP = '239.0.0.1'   # Must match Arduino's target IP
USE_MULTICAST = True            # Set False to revert to unicast

# ── ROS2 type definitions ────────────────────────────────────────────
typestore = get_typestore(Stores.ROS2_HUMBLE)
serialize_cdr = typestore.serialize_cdr
Float32MultiArray = typestore.types['std_msgs/msg/Float32MultiArray']
MultiArrayLayout = typestore.types['std_msgs/msg/MultiArrayLayout']
MultiArrayDimension = typestore.types['std_msgs/msg/MultiArrayDimension']
Header = typestore.types['std_msgs/msg/Header']
PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
PointField = typestore.types['sensor_msgs/msg/PointField']
TFMessage = typestore.types['tf2_msgs/msg/TFMessage']
TransformStamped = typestore.types['geometry_msgs/msg/TransformStamped']
Transform = typestore.types['geometry_msgs/msg/Transform']
Vector3 = typestore.types['geometry_msgs/msg/Vector3']
Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
Time = typestore.types['builtin_interfaces/msg/Time']


# ── ROS CDR helpers ──────────────────────────────────────────────────

def make_static_tf_cdr(ts_ns, parent_frame='map', child_frame='sensor'):
    sec = int(ts_ns // 1_000_000_000)
    nanosec = int(ts_ns % 1_000_000_000)
    msg = TFMessage(transforms=[
        TransformStamped(
            header=Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=parent_frame),
            child_frame_id=child_frame,
            transform=Transform(
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
    ])
    return serialize_cdr(msg, msg.__class__.__msgtype__)


def packet_to_float32multiarray_cdr(raw_packet):
    """Serialize raw UDP depth packet as Float32MultiArray in metres."""
    raw_mm = np.frombuffer(raw_packet[1:1 + NUM_PIXELS * 2], dtype=np.uint16)
    depth_m = raw_mm.astype(np.float32) / 1000.0
    msg = Float32MultiArray(
        layout=MultiArrayLayout(
            dim=[
                MultiArrayDimension(label='row', size=IMAGE_WIDTH, stride=NUM_PIXELS),
                MultiArrayDimension(label='col', size=IMAGE_WIDTH, stride=IMAGE_WIDTH),
            ],
            data_offset=0,
        ),
        data=depth_m,
    )
    return serialize_cdr(msg, msg.__class__.__msgtype__)


def packet_to_pointcloud2_cdr(raw_packet, ts_ns, sensor_id=0):
    """Serialize raw UDP depth packet as PointCloud2 in metres."""
    raw_mm = np.frombuffer(raw_packet[1:1 + NUM_PIXELS * 2], dtype=np.uint16)
    depth_m = raw_mm.astype(np.float32) / 1000.0
    valid = depth_m < INVALID_DEPTH
    x = depth_m * ray_x_flat
    y = depth_m * ray_y_flat
    z = depth_m * ray_z_flat
    x[~valid] = 0.0
    y[~valid] = 0.0
    z[~valid] = 0.0
    pts = np.column_stack([x, y, z, depth_m]).astype(np.float32)
    data_bytes = np.frombuffer(pts.tobytes(), dtype=np.uint8)
    sec = int(ts_ns // 1_000_000_000)
    nanosec = int(ts_ns % 1_000_000_000)
    FLOAT32 = 7
    msg = PointCloud2(
        header=Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=f'sensor_{sensor_id}'),
        height=1, width=NUM_PIXELS,
        fields=[
            PointField(name='x',         offset=0,  datatype=FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=FLOAT32, count=1),
        ],
        is_bigendian=False, point_step=16, row_step=16 * NUM_PIXELS,
        data=data_bytes, is_dense=False,
    )
    return serialize_cdr(msg, msg.__class__.__msgtype__)


# ── UDP socket helper ────────────────────────────────────────────────

def _make_udp_socket(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    sock.setblocking(False)
    sock.bind(("0.0.0.0", port))

    if USE_MULTICAST:
        mreq = struct.pack('4sL', socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        print(f"  [multicast] Joined group {MULTICAST_GROUP} on port {port}")
    return sock


class ProximitySuit:
    """Top-level container that holds every Skin, manages UDP data, and recording."""

    DEFAULTS = {
        'skins': [],  # Loaded from sensors_config.yaml only
        'disabled_sensors': set(),
        'start_pose': 'ready',
        'color_mode': 'sensor',
        'show_labels': True,
        'camera_pan': False,
        'pan_speed': 0.3,
        'replay': None,
        'episode': 0,
        'playback_speed': 1.0,
        'loop': True,
        'replay_joints_only': True,
        'record_data': True,
        'live_joints': False,
        'joint_states_topic': '/joint_states',
    }

    def __init__(self, xacro_name, config_path=None, fk_xacro_path=None,
                 **overrides):
        self.config = self._load_config(config_path)
        for key, val in overrides.items():
            if val is not None:
                self.config[key] = val

        self.xacro_name = os.path.join(os.path.dirname(__file__), xacro_name)
        self.fk_xacro_path = fk_xacro_path or os.path.join(
            os.path.dirname(__file__), 'fr3_full_skin.xacro')

        self.skin_defs = self.config['skins']
        self.disabled_sensors = self.config['disabled_sensors']

        start_pose = self.config['start_pose']
        self.fk_chain = parse_fk_chain_from_xacro(self.fk_xacro_path)
        self.joint_angles = list(
            FR3_READY_POSE if start_pose == 'ready' else FR3_ZERO_POSE)
        self.link_transforms = compute_link_transforms(
            self.joint_angles, self.fk_chain)

        self.skins = []
        self._sensors_by_id = {}
        self.num_sensors = 0
        self.generate_skins()

        self.enabled_sensor_ids = self._build_enabled_sensor_ids()
        disabled_skins = [sd['skin_name'] for sd in self.skin_defs if not sd.get('enabled', True)]
        if self.disabled_sensors or disabled_skins:
            print(f"Config: disabled_skins={disabled_skins or 'none'}, "
                  f"disabled_sensors={sorted(self.disabled_sensors) or 'none'}")

        self.data_lock = threading.Lock()
        self.new_data = threading.Event()

        self._pkt_counts = {}
        self._pkt_count_lock = threading.Lock()
        self._last_report = 0.0

        self._record_lock = threading.Lock()
        self._record_packets = []
        self._record_mcap = None
        self._record_mcap_file = None
        self._record_base = None
        self._mcap_channels = {}
        self.record_data = self.config['record_data']

        self._seen_ips = set()
        self._seen_ips_lock = threading.Lock()

    # ── config loading ───────────────────────────────────────────────

    @classmethod
    def _load_config(cls, config_path):
        """Load sensors_config.yaml and return a merged config dict."""
        out = dict(cls.DEFAULTS)
        path = config_path or os.path.join(
            os.path.dirname(__file__), 'sensors_config.yaml')
        if not os.path.exists(path):
            print(f"Warning: {path} not found — no skins configured.")
            return out
        try:
            import yaml
        except ImportError:
            print("Warning: PyYAML not installed — using defaults.")
            return out
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: could not load {path}: {e} — using defaults.")
            return out

        # ── skin definitions ─────────────────────────────────────────
        if 'skins' in cfg and isinstance(cfg['skins'], list):
            skin_list = []
            for entry in cfg['skins']:
                if not isinstance(entry, dict) or 'skin_name' not in entry:
                    continue
                skin_list.append({
                    'skin_name':     str(entry['skin_name']),
                    'link_attached': int(entry.get('link_attached', 0)),
                    'port':          int(entry.get('port', 5000)),
                    'label_prefix':  str(entry.get('label_prefix', 'S')),
                    'enabled':       bool(entry.get('enabled', True)),
                })
            if skin_list:
                out['skins'] = skin_list

        disabled_sensors = set()
        for name in cfg.get('disabled_sensors') or []:
            if isinstance(name, str) and name.strip():
                disabled_sensors.add(name.strip())
        out['disabled_sensors'] = disabled_sensors

        # ── visualizer settings ──────────────────────────────────────
        if 'start_pose' in cfg:
            out['start_pose'] = str(cfg['start_pose'])
        if 'color_mode' in cfg:
            out['color_mode'] = str(cfg['color_mode'])
        if 'show_labels' in cfg:
            out['show_labels'] = bool(cfg['show_labels'])
        if 'camera_pan' in cfg:
            out['camera_pan'] = bool(cfg['camera_pan'])
        if 'pan_speed' in cfg:
            out['pan_speed'] = float(cfg['pan_speed'])

        # ── replay settings ──────────────────────────────────────────
        if 'replay' in cfg and cfg['replay'] is not None:
            out['replay'] = str(cfg['replay'])
        if 'episode' in cfg:
            out['episode'] = int(cfg['episode'])
        if 'playback_speed' in cfg:
            out['playback_speed'] = float(cfg['playback_speed'])
        if 'loop' in cfg:
            out['loop'] = bool(cfg['loop'])
        if 'replay_joints_only' in cfg:
            out['replay_joints_only'] = bool(cfg['replay_joints_only'])

        # ── recording ────────────────────────────────────────────────
        if 'record_data' in cfg:
            out['record_data'] = bool(cfg['record_data'])

        # ── live joint tracking ──────────────────────────────────────
        if 'live_joints' in cfg:
            out['live_joints'] = bool(cfg['live_joints'])
        if 'joint_states_topic' in cfg:
            out['joint_states_topic'] = str(cfg['joint_states_topic'])

        return out

    # ── skin generation ──────────────────────────────────────────────

    def generate_skins(self):
        """Parse xacro, create Skin objects with their Sensors, compute initial transforms."""
        macro_names = tuple(sd['skin_name'] for sd in self.skin_defs)
        sensors_by_link, meshes_by_link, dermis_by_link = \
            parse_skins_xacro(self.xacro_name, macro_names)

        global_id_offset = 0
        for skin_id, sd in enumerate(self.skin_defs):
            xacro_name = sd['skin_name']
            link_idx = sd['link_attached']

            d = dermis_by_link.get(xacro_name, {'xyz': (0, 0, 0), 'rpy': (0, 0, 0)})
            dermis_tf = make_transform(d['xyz'], d['rpy'])

            stl_path = meshes_by_link.get(xacro_name)
            sensor_defs = sensors_by_link.get(xacro_name, [])

            skin = Skin(
                skin_name=xacro_name,
                link_attached=link_idx,
                id=skin_id,
                port=sd['port'],
                link_to_skin_tf=dermis_tf,
                sensor_defs=sensor_defs,
                global_id_offset=global_id_offset,
                stl_path=stl_path,
                label_prefix=sd.get('label_prefix', 'S'),
                enabled=sd.get('enabled', True),
            )
            skin.compute_sensor_transforms(self.link_transforms[link_idx])

            self.skins.append(skin)
            for sensor in skin.sensors:
                self._sensors_by_id[sensor.id] = sensor

            global_id_offset += skin.num_sensors

        self.num_sensors = global_id_offset

    # ── sensor access ────────────────────────────────────────────────

    def get_sensor(self, sensor_id):
        return self._sensors_by_id.get(sensor_id)

    def get_all_sensors(self):
        return list(self._sensors_by_id.values())

    def get_latest_depth(self):
        """Return dict of sensor_id -> depth array (float32, metres)."""
        result = {}
        with self.data_lock:
            for sid, sensor in self._sensors_by_id.items():
                result[sid] = sensor.packet_buffer.copy()
        return result

    def sensor_id_to_name(self, sensor_id):
        for skin in self.skins:
            if skin.global_id_offset <= sensor_id < skin.global_id_offset + skin.num_sensors:
                return skin.sensor_name(self._sensors_by_id[sensor_id])
        return f"S-{sensor_id}"

    def skin_for_sensor(self, sensor_id):
        for skin in self.skins:
            if skin.global_id_offset <= sensor_id < skin.global_id_offset + skin.num_sensors:
                return skin
        return None

    def _build_enabled_sensor_ids(self):
        ids = set()
        for skin in self.skins:
            if skin.enabled:
                for sensor in skin.sensors:
                    if skin.sensor_name(sensor) not in self.disabled_sensors:
                        ids.add(sensor.id)
        return ids

    # ── joint angle updates ──────────────────────────────────────────

    def update_joint_angles(self, joint_angles):
        self.joint_angles = list(joint_angles)
        self.link_transforms = compute_link_transforms(
            self.joint_angles, self.fk_chain)
        for skin in self.skins:
            skin.compute_sensor_transforms(self.link_transforms[skin.link_attached])

    # ── UDP reception ────────────────────────────────────────────────

    def _process_packet(self, data, sensor_id):
        ts_ns = time.time_ns()

        with self._pkt_count_lock:
            self._pkt_counts[sensor_id] = self._pkt_counts.get(sensor_id, 0) + 1
            now = time.monotonic()
            if now - self._last_report >= 3.0:
                self._last_report = now
                counts = '  '.join(
                    f'{self.sensor_id_to_name(i)}:{self._pkt_counts.get(i, 0)}'
                    for i in sorted(self.enabled_sensor_ids)
                )
                print(f"[packets] {counts}")

        if self.record_data:
            with self._record_lock:
                if self._record_base is None:
                    self._init_mcap(ts_ns)
                self._record_packets.append((ts_ns, data, sensor_id))
                if self._record_mcap is not None and sensor_id in self._mcap_channels:
                    ch = self._mcap_channels[sensor_id]
                    self._record_mcap.add_message(
                        channel_id=ch['depth'], log_time=ts_ns,
                        data=packet_to_float32multiarray_cdr(data),
                        publish_time=ts_ns,
                    )
                    self._record_mcap.add_message(
                        channel_id=ch['pc2'], log_time=ts_ns,
                        data=packet_to_pointcloud2_cdr(data, ts_ns, sensor_id),
                        publish_time=ts_ns,
                    )

        raw_mm = np.frombuffer(
            data[1:1 + NUM_PIXELS * 2], dtype=np.uint16)
        sensor = self._sensors_by_id.get(sensor_id)
        if sensor:
            with self.data_lock:
                sensor.packet_buffer = raw_mm.astype(np.float32) / 1000.0
                sensor.dirty = True
            self.new_data.set()

    def _udp_receiver_loop(self, port, sensor_id_offset, num_on_port):
        import select
        sock = _make_udp_socket(port)
        expected_len = 1 + NUM_PIXELS * 2
        while True:
            select.select([sock], [], [], 1.0)
            while True:
                try:
                    data, addr = sock.recvfrom(4096)
                    if len(data) < expected_len:
                        continue
                    with self._seen_ips_lock:
                        if addr[0] not in self._seen_ips:
                            self._seen_ips.add(addr[0])
                            print(f"Discovered ESP32 at {addr[0]} (port {port})")
                    pkt_sensor_id = data[0]
                    if pkt_sensor_id >= num_on_port:
                        continue
                    self._process_packet(data, sensor_id_offset + pkt_sensor_id)
                except BlockingIOError:
                    break

    def start_udp(self):
        """Spawn one UDP listener thread per skin port."""
        for skin in self.skins:
            t = threading.Thread(
                target=self._udp_receiver_loop,
                args=(skin.port, skin.global_id_offset, skin.num_sensors),
                daemon=True,
            )
            t.start()
            print(f"Listening on port {skin.port} for {skin.skin_name} sensors "
                  f"(IDs {skin.global_id_offset}-"
                  f"{skin.global_id_offset + skin.num_sensors - 1})")

    # ── MCAP recording ───────────────────────────────────────────────

    def _init_mcap(self, ts_ns):
        self._record_base = f"raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not MCAP_AVAILABLE:
            print(f"Recording to {self._record_base}.npz (mcap not installed)")
            return
        self._record_mcap_file = open(self._record_base + ".mcap", "wb")
        self._record_mcap = McapWriter(self._record_mcap_file)
        self._record_mcap.start()

        depth_sid = self._record_mcap.register_schema(
            name="std_msgs/msg/Float32MultiArray",
            encoding="ros2msg", data=FLOAT32_MULTIARRAY_SCHEMA)
        pc2_sid = self._record_mcap.register_schema(
            name="sensor_msgs/msg/PointCloud2",
            encoding="ros2msg", data=POINTCLOUD2_SCHEMA)
        tf_sid = self._record_mcap.register_schema(
            name="tf2_msgs/msg/TFMessage",
            encoding="ros2msg", data=TFMESSAGE_SCHEMA)

        for s in range(self.num_sensors):
            depth_ch = self._record_mcap.register_channel(
                topic=f"/sensor_{s}/tof_depth",
                message_encoding="cdr", schema_id=depth_sid)
            pc2_ch = self._record_mcap.register_channel(
                topic=f"/sensor_{s}/tof_pointcloud",
                message_encoding="cdr", schema_id=pc2_sid)
            self._mcap_channels[s] = {'depth': depth_ch, 'pc2': pc2_ch}
            tf_ch = self._record_mcap.register_channel(
                topic=f"/sensor_{s}/tf_static",
                message_encoding="cdr", schema_id=tf_sid)
            tf_cdr = make_static_tf_cdr(ts_ns, child_frame=f'sensor_{s}')
            self._record_mcap.add_message(
                channel_id=tf_ch, log_time=ts_ns,
                data=tf_cdr, publish_time=ts_ns)

        print(f"Recording to {self._record_base}.npz and {self._record_base}.mcap")

    def save_recording(self):
        """Flush recorded packets to NPZ + MCAP (called on exit)."""
        if self._record_base is None or not self._record_packets:
            return
        with self._record_lock:
            timestamps = np.array([p[0] for p in self._record_packets], dtype=np.uint64)
            packets_list = [np.frombuffer(p[1], dtype=np.uint8) for p in self._record_packets]
            sensor_ids = np.array([p[2] for p in self._record_packets], dtype=np.uint8)
            skins = np.array([sensor_id_to_skin(p[2]) for p in self._record_packets], dtype='U6')
            skin_indices = np.array([sensor_id_to_skin_index(p[2]) for p in self._record_packets], dtype=np.uint8)
            max_len = max(len(p) for p in packets_list)
            padded = np.zeros((len(packets_list), max_len), dtype=np.uint8)
            for i, p in enumerate(packets_list):
                padded[i, :len(p)] = p
            np.savez(
                self._record_base + ".npz",
                timestamps=timestamps, packets=padded,
                sensor_ids=sensor_ids, skins=skins,
                skin_indices=skin_indices, allow_pickle=False)
            print(f"Saved {len(self._record_packets)} packets to {self._record_base}.npz")
            if self._record_mcap is not None:
                self._record_mcap.finish()
            if self._record_mcap_file is not None:
                self._record_mcap_file.close()
                print(f"Recording stopped. {self._record_base}.mcap closed.")

    # ── visualizer ───────────────────────────────────────────────────

    def visualize(self, robot_mesh_name='fr3', **overrides):
        """Create and run the 3D visualizer (blocks until window is closed).

        Any keyword argument overrides the corresponding value from
        sensors_config.yaml.  Unset keys fall back to the YAML / defaults.
        """
        c = self.config
        viz = ProximityVisualizer(
            self,
            robot_mesh_name=robot_mesh_name,
            color_mode=overrides.get('color_mode', c['color_mode']),
            show_labels=overrides.get('show_labels', c['show_labels']),
            camera_pan=overrides.get('camera_pan', c['camera_pan']),
            pan_speed=overrides.get('pan_speed', c['pan_speed']),
            replay_path=overrides.get('replay_path', c['replay']),
            replay_episode=overrides.get('replay_episode', c['episode']),
            playback_speed=overrides.get('playback_speed', c['playback_speed']),
            replay_loop=overrides.get('replay_loop', c['loop']),
            replay_joints_only=overrides.get('replay_joints_only', c['replay_joints_only']),
            live_joints=overrides.get('live_joints', c['live_joints']),
            joint_states_topic=overrides.get('joint_states_topic', c['joint_states_topic']),
        )
        viz.visualize()
