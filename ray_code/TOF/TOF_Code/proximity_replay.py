"""
H5ReplayController — loads H5 trajectory files and drives playback via callbacks.
"""

import re
import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

from proximity_common import INVALID_DEPTH


class H5ReplayController:
    """
    Loads H5 trajectory files, maps H5 sensor IDs to viz IDs, and tracks playback.
    Uses on_frame_change(ep, frame, joint_positions, sensor_data) for updates.
    """

    def __init__(self, path, suit, episode=0, playback_speed=1.0, loop=True):
        self.path = path
        self.suit = suit
        self._episode = episode
        self._frame = 0
        self._playing = False
        self._speed = playback_speed
        self._loop = loop

        self._episodes = []
        self._sensor_episodes = []
        self._num_frames = 0

        self.on_frame_change = None  # Callable[[int, int, np.ndarray, dict], None]

        if path:
            if H5PY_AVAILABLE:
                self._load()
            else:
                print("Error: h5py not installed. Run: pip install h5py")

    @property
    def is_active(self):
        return len(self._episodes) > 0

    @property
    def ep(self):
        return self._episode

    @property
    def frame(self):
        return self._frame

    @property
    def num_episodes(self):
        return len(self._episodes)

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def playing(self):
        return self._playing

    @property
    def speed(self):
        return self._speed

    @property
    def timer_interval_ms(self):
        return int(200.0 / self._speed)

    def _load(self):
        if not self.path or not H5PY_AVAILABLE:
            return

        h5f = h5py.File(self.path, 'r')
        episodes = []
        sensor_episodes = []
        traj_idx = 0
        while f'traj_data/traj_{traj_idx:06d}/robot/joint_positions' in h5f:
            traj_key = f'traj_data/traj_{traj_idx:06d}'
            jp = h5f[f'{traj_key}/robot/joint_positions'][:]
            episodes.append(jp[:, 0, :])
            sensor_episodes.append(self._load_h5_sensor_data(h5f, traj_key))
            traj_idx += 1
        h5f.close()

        if not episodes:
            print(f"Warning: no trajectory data found in {self.path}")
            return

        self._episodes = episodes
        self._sensor_episodes = sensor_episodes
        self._episode = min(self._episode, len(episodes) - 1)
        self._num_frames = len(self._episodes[self._episode])
        self._frame = 0
        self._playing = True

        n_sensor_streams = sum(1 for ep in sensor_episodes if ep for _ in ep)
        print(f"Replay: loaded {len(episodes)} episodes from {self.path}")
        if n_sensor_streams:
            print(f"  Sensor data: {n_sensor_streams} sensor streams across all episodes")
        print(f"  Starting at episode {self._episode} ({self._num_frames} frames)")
        print(f"  Controls: Space=play/pause, Left/Right=step, [/]=episode, +/-=speed")

    def _load_h5_sensor_data(self, h5f, traj_key):
        obs_key = f'{traj_key}/observations'
        if obs_key not in h5f:
            return None
        sensor_map = {}
        pat = re.compile(r'^depth_sensor_link(\d+)_sensor_(\d+)$')
        for name in h5f[obs_key]:
            m = pat.match(name)
            if not m:
                continue
            link_num, sensor_idx = int(m.group(1)), int(m.group(2))
            viz_id = self._h5_sensor_to_viz_id(link_num, sensor_idx)
            if viz_id is None or viz_id >= self.suit.num_sensors:
                continue
            depth_key = f'{obs_key}/{name}/depth_to_camera'
            if depth_key not in h5f:
                continue
            raw = h5f[depth_key][:]
            depth = raw[:, 0, :, :, 0].astype(np.float32)
            depth[depth >= INVALID_DEPTH] = INVALID_DEPTH
            sensor_map[viz_id] = depth.reshape(depth.shape[0], -1)
        return sensor_map if sensor_map else None

    def _h5_sensor_to_viz_id(self, link_num, sensor_idx):
        base_offset = None
        total_on_link = 0
        acc = 0
        for skin in self.suit.skins:
            if skin.link_attached == link_num:
                if base_offset is None:
                    base_offset = acc
                total_on_link += skin.num_sensors
            acc += skin.num_sensors
        if base_offset is None or sensor_idx >= total_on_link:
            return None
        return base_offset + sensor_idx

    def _emit_frame_change(self):
        if self.on_frame_change is None or not self.is_active:
            return
        jp = self._episodes[self._episode][self._frame]
        ep_sensors = self._sensor_episodes[self._episode]
        sensor_data = {}
        if ep_sensors:
            for sid, depth_frames in ep_sensors.items():
                if self._frame < len(depth_frames):
                    sensor_data[sid] = depth_frames[self._frame].copy()
        self.on_frame_change(self._episode, self._frame, jp, sensor_data)

    def emit_current_frame(self):
        """Emit the current frame to the callback (e.g. after load or seek)."""
        self._emit_frame_change()

    def step(self):
        """Advance one frame (for timer tick). Returns True if frame changed."""
        if not self.is_active or not self._playing:
            return False
        self._frame += 1
        if self._frame >= self._num_frames:
            if self._loop:
                self._episode = (self._episode + 1) % self.num_episodes
                self._num_frames = len(self._episodes[self._episode])
                self._frame = 0
                print(f"Replay: episode {self._episode}/{self.num_episodes - 1}")
            else:
                self._frame = self._num_frames - 1
                self._playing = False
                return False
        self._emit_frame_change()
        return True

    def step_fwd(self):
        if not self.is_active:
            return
        self._playing = False
        self._frame = min(self._frame + 1, self._num_frames - 1)
        self._emit_frame_change()

    def step_bwd(self):
        if not self.is_active:
            return
        self._playing = False
        self._frame = max(self._frame - 1, 0)
        self._emit_frame_change()

    def next_ep(self):
        if not self.is_active:
            return
        self._episode = (self._episode + 1) % self.num_episodes
        self._num_frames = len(self._episodes[self._episode])
        self._frame = 0
        self._emit_frame_change()
        print(f"Replay: episode {self._episode}/{self.num_episodes - 1} "
              f"({self._num_frames} frames)")

    def prev_ep(self):
        if not self.is_active:
            return
        self._episode = (self._episode - 1) % self.num_episodes
        self._num_frames = len(self._episodes[self._episode])
        self._frame = 0
        self._emit_frame_change()
        print(f"Replay: episode {self._episode}/{self.num_episodes - 1} "
              f"({self._num_frames} frames)")

    def toggle(self):
        if not self.is_active:
            return
        self._playing = not self._playing
        print(f"Replay: {'playing' if self._playing else 'paused'}")

    def speed_up(self):
        self._speed = min(self._speed * 1.5, 10.0)
        print(f"Replay: speed {self._speed:.1f}x")

    def speed_down(self):
        self._speed = max(self._speed / 1.5, 0.1)
        print(f"Replay: speed {self._speed:.1f}x")
