"""
Unified real-time 3D point cloud visualizer for all VL53L5CX ToF sensors.

All settings live in sensors_config.yaml.  CLI arguments override YAML values
when explicitly provided.

Run:  python3 visualizer_3d_combined.py [--color-mode sensor|time|trail] ...
"""

import argparse
from proximity_suit import ProximitySuit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VL53L5CX combined 3D visualizer")
    parser.add_argument('--config', '-c', default=None,
                        help='Path to sensors_config.yaml')
    parser.add_argument('--start-pose', choices=['ready', 'zero'], default=None,
                        help='Initial robot joint configuration')
    parser.add_argument('--color-mode', choices=['sensor', 'time', 'trail'], default=None,
                        help='Point-cloud color mode')
    parser.add_argument('--no-labels', action='store_true', default=None,
                        help='Hide sensor labels')
    parser.add_argument('--camera-pan', action='store_true', default=None,
                        help='Orbit the camera')
    parser.add_argument('--pan-speed', type=float, default=None,
                        help='Camera pan speed (deg/tick)')
    parser.add_argument('--replay', default=None,
                        help='Path to HDF5 replay file')
    parser.add_argument('--episode', type=int, default=None,
                        help='Episode index for replay')
    parser.add_argument('--playback-speed', type=float, default=None,
                        help='Playback speed multiplier')
    parser.add_argument('--no-loop', action='store_true', default=None,
                        help='Disable episode looping')
    parser.add_argument('--replay-sensors', action='store_true', default=None,
                        help='Use recorded sensors from HDF5 (default: live sensors)')
    parser.add_argument('--live-joints', action='store_true', default=None,
                        help='Subscribe to /joint_states for real-time robot pose')
    parser.add_argument('--joint-topic', default=None,
                        help='ROS2 topic for JointState messages (default: /joint_states)')
    args = parser.parse_args()

    suit_overrides = {}
    if args.start_pose is not None:
        suit_overrides['start_pose'] = args.start_pose

    suit = ProximitySuit(
        xacro_name='skins.xacro',
        config_path=args.config,
        **suit_overrides,
    )

    viz_overrides = {}
    if args.color_mode is not None:
        viz_overrides['color_mode'] = args.color_mode
    if args.no_labels:
        viz_overrides['show_labels'] = False
    if args.camera_pan:
        viz_overrides['camera_pan'] = True
    if args.pan_speed is not None:
        viz_overrides['pan_speed'] = args.pan_speed
    if args.replay is not None:
        viz_overrides['replay_path'] = args.replay
    if args.episode is not None:
        viz_overrides['replay_episode'] = args.episode
    if args.playback_speed is not None:
        viz_overrides['playback_speed'] = args.playback_speed
    if args.no_loop:
        viz_overrides['replay_loop'] = False
    if args.replay_sensors:
        viz_overrides['replay_joints_only'] = False
    if args.live_joints:
        viz_overrides['live_joints'] = True
    if args.joint_topic is not None:
        viz_overrides['joint_states_topic'] = args.joint_topic

    suit.visualize(**viz_overrides)
