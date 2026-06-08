"""
Sensor — individual VL53L5CX time-of-flight sensor.
"""

import numpy as np

from proximity_common import (
    NUM_PIXELS, INVALID_DEPTH,
    ray_x_flat, ray_y_flat, ray_z_flat,
)


class Sensor:
    """Individual VL53L5CX ToF sensor with pose, depth data, and world-frame transforms."""

    def __init__(self, id, pose, rpy, normal=None):
        self.id = id
        self.pose = np.array(pose, dtype=np.float64)
        self.normal = np.array(normal, dtype=np.float64) if normal is not None else None
        self.rpy = tuple(rpy)
        self.packet_buffer = np.full(NUM_PIXELS, INVALID_DEPTH, dtype=np.float32)
        self.camera_calibration_matrix = np.column_stack(
            [ray_x_flat, ray_y_flat, ray_z_flat]
        ).astype(np.float32)
        self._rotation = None   # (3,3) float32, base-frame rotation
        self._translation = None  # (3,) float32, base-frame translation in metres
        self.dirty = False

    # ── public API ───────────────────────────────────────────────────

    def get_sensorID(self):
        return self.id

    def get_depth(self):
        """Return depth values (float32, metres)."""
        return self.packet_buffer.copy()

    def get_undistorted_depth(self):
        """Return 3D points in sensor-local frame (spherical projection)."""
        dist = self.packet_buffer
        valid = dist < INVALID_DEPTH
        x = dist * ray_x_flat
        y = dist * ray_y_flat
        z = dist * ray_z_flat
        x[~valid] = 0.0
        y[~valid] = 0.0
        z[~valid] = 0.0
        return np.column_stack([x, y, z])

    def get_pose(self):
        """Return sensor position in base frame (metres)."""
        if self._translation is not None:
            return self._translation.copy()
        return np.zeros(3, dtype=np.float32)

    # ── internal helpers ─────────────────────────────────────────────

    def set_world_transform(self, R, t):
        self._rotation = R
        self._translation = t

    @property
    def rotation(self):
        return self._rotation

    @property
    def translation(self):
        return self._translation
