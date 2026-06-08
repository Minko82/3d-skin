"""
ProximityVisualizer — 3D OpenGL point-cloud visualizer for proximity sensors.
"""

import os
import numpy as np

import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import trimesh

from proximity_replay import H5ReplayController
from ros.joint_state_subscriber import JointStateSubscriber, RCLPY_AVAILABLE
from proximity_common import (
    NUM_PIXELS, INVALID_DEPTH,
    HISTORY_SENSOR, HISTORY_TIME, HISTORY_TRAIL, TRAIL_MAX_RANGE,
    MAX_HISTORY_LEN, POINT_SIZE, COLOR_SCALE,
    ROBOT_COLOR, ROBOT_EDGE_COLOR, SKIN_COLOR, SKIN_EDGE_COLOR,
    FR3_READY_POSE,
    SENSOR_HUES,
    ray_x_flat, ray_y_flat, ray_z_flat,
)


def _np4x4_to_qmatrix(T_m):
    """Convert a 4x4 numpy transform to QMatrix4x4."""
    m = T_m.astype(float).flatten()
    return QtGui.QMatrix4x4(
        m[0], m[1], m[2], m[3],
        m[4], m[5], m[6], m[7],
        m[8], m[9], m[10], m[11],
        m[12], m[13], m[14], m[15],
    )


class ProximityVisualizer:
    """3D point-cloud visualizer backed by pyqtgraph/OpenGL."""

    def __init__(self, suit, robot_mesh_name='fr3', color_mode='sensor',
                 show_labels=True, camera_pan=False, pan_speed=0.3,
                 replay_path=None, replay_episode=0,
                 playback_speed=1.0, replay_loop=True, replay_joints_only=True,
                 live_joints=False, joint_states_topic='/joint_states'):
        self.suit = suit
        self.robotMeshName = robot_mesh_name
        self.robot_mesh_dir = os.path.join(
            os.path.dirname(__file__), 'robot_arms', robot_mesh_name, 'visual')
        self.color_mode = color_mode
        self.show_labels = show_labels
        self.camera_pan = camera_pan
        self.pan_speed = pan_speed
        self._pan_azimuth = 0.0

        self.history_len = self._history_len_for_mode(color_mode)

        self._replay = H5ReplayController(
            replay_path, suit, episode=replay_episode,
            playback_speed=playback_speed, loop=replay_loop)
        self._replay_joints_only = replay_joints_only
        self._replay_timer = None

        self._live_joints = live_joints
        self._joint_states_topic = joint_states_topic
        self._joint_sub = None

        self._robot_link_items = []
        self._skin_items = []
        self._sensor_scatter_data = []
        self._sensor_label_items = []
        self.scatters = []
        self._hist_pos = []
        self._hist_col = []
        self._hist_count = []
        self._combined_pos = []
        self._combined_col = []

    @staticmethod
    def _history_len_for_mode(mode):
        if mode == 'time':
            return HISTORY_TIME
        if mode == 'trail':
            return HISTORY_TRAIL
        return HISTORY_SENSOR

    # ── main entry point ─────────────────────────────────────────────

    def visualize(self):
        """Build the scene, start UDP + timers, and run the Qt event loop."""
        self.app = QtWidgets.QApplication([])
        self.win = gl.GLViewWidget()
        self.win.setWindowTitle(
            f"VL53L5CX — Combined 3D Point Cloud [{self.color_mode} mode]")
        self.win.setGeometry(100, 100, 1400, 900)
        if self.camera_pan:
            self.win.setCameraPosition(distance=7.0, elevation=55, azimuth=0)
        else:
            self.win.setCameraPosition(distance=2.0, elevation=25, azimuth=-45)
        self.win.show()

        self._setup_scene()
        self._setup_shortcuts()
        self._load_replay()
        self._start_live_joints()
        self._setup_timers()
        self.suit.start_udp()

        print(f"Robot pose: {'ready' if self.suit.joint_angles == FR3_READY_POSE else 'zero'} — "
              f"q = [{', '.join(f'{a:.4f}' for a in self.suit.joint_angles)}]")
        if self._replay.is_active:
            mode = "joints only (live sensors)" if self._replay_joints_only else "full (recorded sensors)"
            print(f"Replay active ({mode}). Press Space to pause, Left/Right to step, "
                  "[/] to change episode.")
        if self._joint_sub is not None:
            print(f"Live joint tracking active on {self._joint_states_topic}")
        print(f"Combined visualizer running (color mode: {self.color_mode}). "
              "Press T to toggle color. Close the window to exit.")

        try:
            self.app.exec()
        finally:
            if self._joint_sub is not None:
                self._joint_sub.shutdown()
            self.suit.save_recording()

    # ── scene construction ───────────────────────────────────────────

    def _setup_scene(self):
        grid = gl.GLGridItem()
        grid.setSize(6.0, 4.0, 1)
        grid.setSpacing(0.5, 0.5, 1)
        self.win.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(1.0, 1.0, 1.0)
        self.win.addItem(axis)

        self._load_robot_meshes()
        self._load_skin_meshes()
        self._setup_point_cloud_items()
        self._setup_sensor_labels()

    def _load_robot_meshes(self):
        for i in range(8):
            mesh_path = os.path.join(self.robot_mesh_dir, f'link{i}.dae')
            if not os.path.exists(mesh_path):
                print(f"Warning: {mesh_path} not found, skipping link{i}")
                continue
            raw = trimesh.load(mesh_path, force='mesh')
            verts = np.array(raw.vertices, dtype=np.float32)
            faces = np.array(raw.faces, dtype=np.uint32)
            fc = np.empty((len(faces), 4), dtype=np.float32)
            fc[:] = ROBOT_COLOR
            item = gl.GLMeshItem(
                vertexes=verts, faces=faces, faceColors=fc,
                edgeColor=ROBOT_EDGE_COLOR, drawEdges=True,
                smooth=True, glOptions='translucent')
            item.setTransform(_np4x4_to_qmatrix(self.suit.link_transforms[i]))
            self.win.addItem(item)
            self._robot_link_items.append((item, i))
            print(f"Loaded link{i}: {len(faces)} triangles")
        print(f"Robot arm loaded from {self.robot_mesh_dir}")

    def _load_skin_meshes(self):
        """Load skin STL meshes and sensor-position scatter markers for each enabled skin."""
        for skin in self.suit.skins:
            if not skin.enabled:
                print(f"{skin.skin_name} disabled in sensors_config.yaml, skipping")
                continue
            if not skin.stl_path or not os.path.exists(skin.stl_path):
                if skin.stl_path:
                    print(f"Warning: {skin.skin_name} skin not found at {skin.stl_path}, skipping")
                continue

            raw = trimesh.load(skin.stl_path, force='mesh')
            verts = np.array(raw.vertices, dtype=np.float32) / 1000.0
            faces = np.array(raw.faces, dtype=np.uint32)
            fc = np.empty((len(faces), 4), dtype=np.float32)
            fc[:] = SKIN_COLOR

            T_skin_world = self.suit.link_transforms[skin.link_attached] @ skin.link_to_skin_tf
            item = gl.GLMeshItem(
                vertexes=verts, faces=faces, faceColors=fc,
                edgeColor=SKIN_EDGE_COLOR, drawEdges=True,
                smooth=True, glOptions='translucent')
            item.setTransform(_np4x4_to_qmatrix(T_skin_world))
            self.win.addItem(item)
            self._skin_items.append((item, skin.link_attached, skin.link_to_skin_tf.copy()))
            print(f"Loaded {skin.skin_name} skin: {len(faces)} triangles from {skin.stl_path}")

            local_pos_m = np.array([s.pose for s in skin.sensors], dtype=np.float64)
            world_pos = (T_skin_world[:3, :3] @ local_pos_m.T).T + T_skin_world[:3, 3]
            scatter_col = np.array(
                [[h[0], h[1], h[2], 1.0]
                 for h in [SENSOR_HUES[(skin.global_id_offset + i) % len(SENSOR_HUES)]
                           for i in range(skin.num_sensors)]],
                dtype=np.float32)
            scatter = gl.GLScatterPlotItem(
                pos=world_pos.astype(np.float32), color=scatter_col,
                size=12, pxMode=True, glOptions='translucent')
            self.win.addItem(scatter)
            self._sensor_scatter_data.append(
                (scatter, local_pos_m, skin.link_attached, skin.link_to_skin_tf.copy()))
            print(f"Loaded {skin.num_sensors} {skin.skin_name} sensor markers")

    def _setup_point_cloud_items(self):
        ns = self.suit.num_sensors
        max_pts = MAX_HISTORY_LEN * NUM_PIXELS

        for s in range(ns):
            cp = np.zeros((max_pts, 3), dtype=np.float32)
            cc = np.zeros((max_pts, 4), dtype=np.float32)
            self._combined_pos.append(cp)
            self._combined_col.append(cc)

            scatter = gl.GLScatterPlotItem(
                pos=cp, color=cc, size=POINT_SIZE,
                pxMode=True, glOptions='translucent')
            if s in self.suit.enabled_sensor_ids:
                self.win.addItem(scatter)
            self.scatters.append(scatter)

            self._hist_pos.append(np.zeros((MAX_HISTORY_LEN, NUM_PIXELS, 3), dtype=np.float32))
            self._hist_col.append(np.zeros((MAX_HISTORY_LEN, NUM_PIXELS, 4), dtype=np.float32))
        self._hist_count = [0] * ns

    def _setup_sensor_labels(self):
        for s in range(self.suit.num_sensors):
            if s not in self.suit.enabled_sensor_ids:
                continue
            sensor = self.suit.get_sensor(s)
            hue = SENSOR_HUES[s % len(SENSOR_HUES)]
            label_pos = sensor.get_pose().copy()
            label_pos[1] += 0.02
            label = gl.GLTextItem(
                pos=label_pos,
                text=self.suit.sensor_id_to_name(s),
                color=pg.mkColor(int(hue[0] * 255), int(hue[1] * 255), int(hue[2] * 255)),
            )
            label.setData(font=pg.QtGui.QFont("Helvetica", 16))
            if self.show_labels:
                self.win.addItem(label)
            self._sensor_label_items.append((label, s))

    # ── live joint tracking ──────────────────────────────────────────

    def _start_live_joints(self):
        if not self._live_joints:
            return
        if not RCLPY_AVAILABLE:
            print("Warning: rclpy not available — live joint tracking disabled. "
                  "Source your ROS2 workspace first.")
            return
        self._joint_sub = JointStateSubscriber(topic=self._joint_states_topic)
        self._joint_sub.start()

    # ── shortcuts ────────────────────────────────────────────────────

    def _setup_shortcuts(self):
        sc_t = pg.QtGui.QShortcut(pg.QtGui.QKeySequence('T'), self.win)
        sc_t.activated.connect(self._toggle_color_mode)

    # ── timers ───────────────────────────────────────────────────────

    def _setup_timers(self):
        self._update_timer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._update)
        self._update_timer.start(16)

        if self.camera_pan:
            self._pan_timer = QtCore.QTimer()
            self._pan_timer.timeout.connect(self._camera_pan_tick)
            self._pan_timer.start(33)
            print(f"Camera pan enabled (speed: {self.pan_speed} deg/tick)")

        if self._replay.is_active:
            self._replay_timer = QtCore.QTimer()
            self._replay_timer.timeout.connect(self._replay_step)
            self._replay_timer.start(self._replay.timer_interval_ms)

            sc_space = pg.QtGui.QShortcut(pg.QtGui.QKeySequence('Space'), self.win)
            sc_space.activated.connect(self._replay_toggle)
            sc_right = pg.QtGui.QShortcut(pg.QtGui.QKeySequence('Right'), self.win)
            sc_right.activated.connect(self._replay_step_fwd)
            sc_left = pg.QtGui.QShortcut(pg.QtGui.QKeySequence('Left'), self.win)
            sc_left.activated.connect(self._replay_step_bwd)
            sc_next = pg.QtGui.QShortcut(pg.QtGui.QKeySequence(']'), self.win)
            sc_next.activated.connect(self._replay_next_ep)
            sc_prev = pg.QtGui.QShortcut(pg.QtGui.QKeySequence('['), self.win)
            sc_prev.activated.connect(self._replay_prev_ep)
            sc_faster = pg.QtGui.QShortcut(pg.QtGui.QKeySequence('+'), self.win)
            sc_faster.activated.connect(self._replay_speed_up)
            sc_slower = pg.QtGui.QShortcut(pg.QtGui.QKeySequence('-'), self.win)
            sc_slower.activated.connect(self._replay_speed_down)
            self._replay_shortcuts = [
                sc_space, sc_right, sc_left, sc_next, sc_prev, sc_faster, sc_slower]

    # ── camera orbit ─────────────────────────────────────────────────

    def _camera_pan_tick(self):
        self._pan_azimuth = (self._pan_azimuth + self.pan_speed) % 360.0
        self.win.setCameraPosition(distance=7.0, elevation=55, azimuth=self._pan_azimuth)

    # ── color modes ──────────────────────────────────────────────────

    def _toggle_color_mode(self):
        cycle = ['sensor', 'time', 'trail']
        self.color_mode = cycle[(cycle.index(self.color_mode) + 1) % len(cycle)]
        self.history_len = self._history_len_for_mode(self.color_mode)
        for s in range(self.suit.num_sensors):
            self._hist_count[s] = 0
            self._hist_pos[s][:] = 0
            self._hist_col[s][:] = 0
        self.win.setWindowTitle(
            f"VL53L5CX — Combined 3D Point Cloud [{self.color_mode} mode]")
        print(f"Toggled color mode -> {self.color_mode}  (history: {self.history_len} frames)")

    def _build_colormap(self, depths, sensor_id):
        if self.color_mode == 'time':
            return self._colormap_time(depths)
        if self.color_mode == 'trail':
            return self._colormap_trail(depths, sensor_id)
        return self._colormap_sensor(depths, sensor_id)

    @staticmethod
    def _colormap_sensor(depths, sensor_id):
        colors = np.zeros((len(depths), 4), dtype=np.float32)
        valid = depths < INVALID_DEPTH
        if not np.any(valid):
            return colors
        d = depths[valid]
        lo, hi = d.min(), d.max()
        t = np.zeros_like(d) if hi - lo < 1e-6 else (d - lo) / (hi - lo)
        hue = SENSOR_HUES[sensor_id % len(SENSOR_HUES)]
        brightness = (0.3 + 0.7 * (1.0 - t)) * COLOR_SCALE
        colors[valid, 0] = hue[0] * brightness
        colors[valid, 1] = hue[1] * brightness
        colors[valid, 2] = hue[2] * brightness
        colors[valid, 3] = 1.0
        return colors

    @staticmethod
    def _colormap_time(depths):
        colors = np.zeros((len(depths), 4), dtype=np.float32)
        valid = depths < INVALID_DEPTH
        if not np.any(valid):
            return colors
        d = depths[valid]
        lo, hi = d.min(), d.max()
        t = np.zeros_like(d) if hi - lo < 1e-6 else (d - lo) / (hi - lo)
        colors[valid, 0] = (1.0 - t) * COLOR_SCALE
        colors[valid, 1] = (1.0 - np.abs(t - 0.5) * 2) * COLOR_SCALE
        colors[valid, 2] = t * COLOR_SCALE
        colors[valid, 3] = 1.0
        return colors

    @staticmethod
    def _colormap_trail(depths, sensor_id):
        colors = np.zeros((len(depths), 4), dtype=np.float32)
        valid = depths < INVALID_DEPTH
        if not np.any(valid):
            return colors
        d = depths[valid]
        lo, hi = d.min(), d.max()
        t = np.zeros_like(d) if hi - lo < 1e-6 else (d - lo) / (hi - lo)
        hue = SENSOR_HUES[sensor_id % len(SENSOR_HUES)]
        brightness = (0.3 + 0.7 * (1.0 - t)) * COLOR_SCALE
        colors[valid, 0] = hue[0] * brightness
        colors[valid, 1] = hue[1] * brightness
        colors[valid, 2] = hue[2] * brightness
        colors[valid, 3] = np.clip(1.0 - d / TRAIL_MAX_RANGE, 0.0, 1.0)
        return colors

    # ── frame update ─────────────────────────────────────────────────

    def _update(self):
        if self._joint_sub is not None and not self._replay.is_active:
            positions, is_new = self._joint_sub.get_latest()
            if is_new and positions is not None:
                self._update_robot_pose(positions)

        if not self.suit.new_data.is_set():
            return
        self.suit.new_data.clear()

        snapshots = {}
        dirty = {}
        with self.suit.data_lock:
            for sid, sensor in self.suit._sensors_by_id.items():
                snapshots[sid] = sensor.packet_buffer.copy()
                dirty[sid] = sensor.dirty
                sensor.dirty = False

        for s in sorted(snapshots):
            if s not in self.suit.enabled_sensor_ids or not dirty.get(s):
                continue

            dist = snapshots[s]
            valid = dist < INVALID_DEPTH

            x = dist * ray_x_flat
            y = dist * ray_y_flat
            z = dist * ray_z_flat
            x[~valid] = 0
            y[~valid] = 0
            z[~valid] = 0
            pos_local = np.column_stack([x, y, z])

            sensor = self.suit.get_sensor(s)
            if sensor.rotation is not None:
                pos = (pos_local @ sensor.rotation.T) + sensor.translation
            else:
                pos = pos_local
            colors = self._build_colormap(dist, s)

            hl = self.history_len
            if self._hist_count[s] == hl:
                self._hist_pos[s][:-1] = self._hist_pos[s][1:]
                self._hist_col[s][:-1] = self._hist_col[s][1:]
            else:
                self._hist_count[s] += 1

            idx = self._hist_count[s] - 1
            self._hist_pos[s][idx] = pos
            self._hist_col[s][idx] = colors

            n = self._hist_count[s]
            age_frac = (np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(n, 1, 1)
                        if n > 1 else np.ones((1, 1, 1), dtype=np.float32))
            alpha_mult = 0.1 + 0.9 * age_frac

            src = self._hist_col[s][:n]
            total = n * NUM_PIXELS

            if self.color_mode == 'time':
                blue_shift = 1.0 - age_frac
                self._combined_col[s][:total, 0] = (src[:, :, 0] * age_frac[:, :, 0]).ravel()
                self._combined_col[s][:total, 1] = (src[:, :, 1] * age_frac[:, :, 0]).ravel()
                self._combined_col[s][:total, 2] = (src[:, :, 2] + (1.0 - src[:, :, 2]) * blue_shift[:, :, 0]).ravel()
                self._combined_col[s][:total, 3] = (src[:, :, 3] * alpha_mult[:, :, 0]).ravel()
            else:
                self._combined_col[s][:total, 0] = (src[:, :, 0] * age_frac[:, :, 0]).ravel()
                self._combined_col[s][:total, 1] = (src[:, :, 1] * age_frac[:, :, 0]).ravel()
                self._combined_col[s][:total, 2] = (src[:, :, 2] * age_frac[:, :, 0]).ravel()
                self._combined_col[s][:total, 3] = (src[:, :, 3] * alpha_mult[:, :, 0]).ravel()

            self._combined_pos[s][:total] = self._hist_pos[s][:n].reshape(total, 3)
            self.scatters[s].setData(
                pos=self._combined_pos[s][:total],
                color=self._combined_col[s][:total])

    # ── pose update helpers ──────────────────────────────────────────

    def _update_robot_pose(self, joint_angles):
        self.suit.update_joint_angles(joint_angles)
        transforms = self.suit.link_transforms

        for item, link_idx in self._robot_link_items:
            item.setTransform(_np4x4_to_qmatrix(transforms[link_idx]))

        for item, link_idx, local_T in self._skin_items:
            item.setTransform(_np4x4_to_qmatrix(transforms[link_idx] @ local_T))

        for scatter_item, pos_local_m, link_idx, local_T in self._sensor_scatter_data:
            T_full = transforms[link_idx] @ local_T
            pos_world_m = (T_full[:3, :3] @ pos_local_m.T).T + T_full[:3, 3]
            scatter_item.setData(pos=pos_world_m.astype(np.float32))

        for label_item, sensor_id in self._sensor_label_items:
            sensor = self.suit.get_sensor(sensor_id)
            pos = sensor.get_pose()
            pos[1] += 0.02
            label_item.setData(pos=pos)

    # ── replay ───────────────────────────────────────────────────────

    def _load_replay(self):
        """Wire replay controller callback and emit initial frame."""
        if not self._replay.is_active:
            return
        self._replay.on_frame_change = self._on_replay_frame_change
        self._replay.emit_current_frame()

    def _on_replay_frame_change(self, ep, frame, joint_positions, sensor_data):
        """Callback: update pose, optionally inject sensors, refresh window title."""
        self._update_robot_pose(joint_positions)
        if not self._replay_joints_only:
            with self.suit.data_lock:
                for sid, depth in sensor_data.items():
                    sensor = self.suit.get_sensor(sid)
                    if sensor:
                        sensor.packet_buffer = depth
                        sensor.dirty = True
            self.suit.new_data.set()
        status = "[paused]" if not self._replay.playing else f"[{self._replay.speed:.1f}x]"
        self.win.setWindowTitle(
            f"VL53L5CX — Replay ep {ep}/{self._replay.num_episodes - 1}  "
            f"frame {frame}/{self._replay.num_frames - 1}  "
            f"{status} [{self.color_mode}]")

    def _replay_step(self):
        self._replay.step()

    def _replay_toggle(self):
        self._replay.toggle()

    def _replay_step_fwd(self):
        self._replay.step_fwd()

    def _replay_step_bwd(self):
        self._replay.step_bwd()

    def _replay_next_ep(self):
        self._replay.next_ep()

    def _replay_prev_ep(self):
        self._replay.prev_ep()

    def _replay_speed_up(self):
        self._replay.speed_up()
        if self._replay_timer is not None:
            self._replay_timer.setInterval(self._replay.timer_interval_ms)

    def _replay_speed_down(self):
        self._replay.speed_down()
        if self._replay_timer is not None:
            self._replay_timer.setInterval(self._replay.timer_interval_ms)
