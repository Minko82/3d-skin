"""
Skin — group of sensors attached to a single robot link.
"""

import numpy as np

from proximity_common import make_transform
from sensor import Sensor


class Skin:
    """Skin segment with multiple sensors attached to a robot link."""

    def __init__(self, skin_name, link_attached, id, port,
                 link_to_skin_tf, sensor_defs, global_id_offset,
                 stl_path=None, label_prefix='S', enabled=True):
        self.skin_name = skin_name
        self.link_attached = link_attached
        self.id = id
        self.port = port
        self.link_to_skin_tf = np.array(link_to_skin_tf, dtype=np.float64)
        self.stl_path = stl_path
        self.label_prefix = label_prefix
        self.enabled = enabled
        self.global_id_offset = global_id_offset
        self.num_sensors = len(sensor_defs)
        self.sensors = [
            Sensor(id=global_id_offset + i, pose=sdef['xyz'], rpy=sdef['rpy'])
            for i, sdef in enumerate(sensor_defs)
        ]

    def getSkinID(self):
        return self.id

    def compute_sensor_transforms(self, link_transform):
        """Recompute world-frame transforms for all sensors given the FK link transform."""
        T_skin = link_transform @ self.link_to_skin_tf
        for sensor in self.sensors:
            T_local = make_transform(sensor.pose, sensor.rpy)
            T_full = T_skin @ T_local
            sensor.set_world_transform(
                T_full[:3, :3].astype(np.float32),
                T_full[:3, 3].astype(np.float32),
            )
        return T_skin

    def sensor_name(self, sensor):
        """Human-readable label like 'L1-0'."""
        return f"{self.label_prefix}-{sensor.id - self.global_id_offset}"
