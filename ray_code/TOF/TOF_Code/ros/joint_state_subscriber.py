"""
ROS2 JointState subscriber for real-time FR3 joint tracking.

Runs rclpy in a background thread and exposes the latest joint positions
via a thread-safe interface for the visualizer to poll.
"""

import threading
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False

FR3_JOINT_NAMES = [f'fr3_joint{i}' for i in range(1, 8)]


class JointStateSubscriber:
    """Subscribe to /joint_states and provide latest FR3 joint positions."""

    def __init__(self, topic='/joint_states'):
        if not RCLPY_AVAILABLE:
            raise RuntimeError(
                'rclpy not available. Source your ROS2 workspace before running.')

        self._topic = topic
        self._lock = threading.Lock()
        self._latest_positions = None
        self._has_new = False
        self._node = None
        self._thread = None

    def start(self):
        """Initialise rclpy and spin in a background daemon thread."""
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node('proximity_joint_listener')
        self._node.create_subscription(
            JointState, self._topic, self._callback, 10)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        print(f"[JointStateSubscriber] Listening on {self._topic}")

    def _spin(self):
        try:
            rclpy.spin(self._node)
        except Exception:
            pass

    def _callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        positions = []
        for jn in FR3_JOINT_NAMES:
            if jn not in name_to_pos:
                return
            positions.append(name_to_pos[jn])

        with self._lock:
            self._latest_positions = np.array(positions, dtype=np.float64)
            self._has_new = True

    def get_latest(self):
        """Return (positions, is_new).  positions is None until the first message."""
        with self._lock:
            is_new = self._has_new
            self._has_new = False
            if self._latest_positions is not None:
                return self._latest_positions.copy(), is_new
            return None, False

    def shutdown(self):
        if self._node is not None:
            self._node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
