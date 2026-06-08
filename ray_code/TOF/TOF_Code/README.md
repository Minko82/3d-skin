# VL53L5CX 3D Point Cloud Visualizer

Real-time 3D visualization and playback for VL53L5CX ToF sensor data from ESP32-C6.

---

## Install

```bash
pip install -r requirements.txt
```

---

## Live Visualizer — `visualizer_3d_combined.py`

Receives live UDP sensor data, renders the FR3 robot mesh, skin meshes, sensor positions, and point clouds in real time. Records to `.npz` and `.mcap` automatically.

```bash
python3 visualizer_3d_combined.py [OPTIONS]
```

### Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--color-mode` | `sensor` `time` `trail` | `sensor` | Point cloud coloring mode (see below) |
| `--start-pose` | `ready` `zero` | `ready` | Initial robot joint configuration |
| `--config`, `-c` | path | `sensors_config.yaml` | Path to sensor config YAML |
| `--replay` | path | — | Path to HDF5 file with recorded joint trajectories to replay |
| `--episode` | int | `0` | Episode/trajectory index to start replay from |
| `--playback-speed` | float | `1.0` | Speed multiplier for HDF5 trajectory replay |
| `--no-loop` | flag | off | Disable looping through HDF5 episodes |
| `--no-labels` | flag | off | Hide sensor labels (L3-0, L2-0, etc.) in the scene |

### Color Modes

| Mode | Description |
|------|-------------|
| `sensor` | Each sensor gets a distinct fixed color. Brightness varies with relative distance. Short history (10 frames). |
| `time` | Distance colormap (red=near → green=mid → blue=far). Older frames shift toward blue. Long history (50 frames). |
| `trail` | Per-sensor color like `sensor`, but alpha fades linearly from fully opaque at 0 mm to fully transparent at 4000 mm. Long history (50 frames). |

Press **T** at runtime to cycle through all three modes: `sensor → time → trail → sensor`.

### Runtime Controls (Visualizer)

| Key | Action |
|-----|--------|
| `T` | Cycle color mode |
| `Space` | Pause/resume HDF5 trajectory replay |
| `Left` / `Right` | Step backward/forward one frame in HDF5 replay |
| `[` / `]` | Previous/next HDF5 episode |
| `+` / `-` | Speed up/slow down HDF5 replay |

### Examples

```bash
# Default live visualization
python3 visualizer_3d_combined.py

# Trail mode with zero pose
python3 visualizer_3d_combined.py --color-mode trail --start-pose zero

# Hide sensor labels
python3 visualizer_3d_combined.py --no-labels

# Replay joint trajectory from HDF5 while receiving live sensor data
python3 visualizer_3d_combined.py --replay data.h5 --episode 2 --playback-speed 0.5
```

---

## NPZ Playback — `playback_3d_npz.py`

Replays a recorded `.npz` file with the full 3D scene: robot mesh, skin meshes, sensor positions, and point clouds.

```bash
python3 playback_3d_npz.py <recording.npz> [OPTIONS]
```

### Positional Argument

| Argument | Description |
|----------|-------------|
| `npz` | Path to the `.npz` recording file (required) |

### Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--color-mode` | `sensor` `time` `trail` | `sensor` | Point cloud coloring mode (see above) |
| `--start-pose` | `ready` `zero` | `ready` | Robot joint configuration shown in the scene |
| `--speed` | float | `1.0` | Playback speed multiplier |
| `--config`, `-c` | path | `sensors_config.yaml` | Path to sensor config YAML |
| `--no-labels` | flag | off | Hide sensor labels in the scene |

### Runtime Controls (Playback)

| Key | Action |
|-----|--------|
| `Space` | Pause/resume playback |
| `+` / `-` | Speed up / slow down |
| `[` / `]` | Previous / next episode |
| `.` | Step forward one frame (all sensor packets in that burst) |
| `,` | Step backward one frame (replays from history) |
| `R` | Restart current episode from the beginning |

Playback automatically loops back to the start when it reaches the end.

### Examples

```bash
# Basic playback
python3 playback_3d_npz.py raw_data_20260222_122747.npz

# 2× speed, trail mode, no labels
python3 playback_3d_npz.py raw_data_20260222_122747.npz --speed 2.0 --color-mode trail --no-labels

# Replay with time colormap
python3 playback_3d_npz.py raw_data_20260222_122747.npz --color-mode time
```

---

## NPZ File Format

Recordings saved by `visualizer_3d_combined.py` contain:

| Array | dtype | Description |
|-------|-------|-------------|
| `timestamps` | `uint64` | Packet timestamps in nanoseconds |
| `packets` | `uint8` | Raw sensor packet bytes `(N, packet_len)` |
| `sensor_ids` | `uint8` | Global sensor index per packet (0–12) |
| `skins` | `U6` | Skin label per packet: `skin2`, `skin3`, `skin5` |
| `skin_indices` | `uint8` | Index within the skin (0–4 for skin2, 0–3 for skin3/skin5) |

Skin mapping:
- **skin2** — 5 sensors (global IDs 0–4), on robot link2, port 5005
- **skin3** — 4 sensors (global IDs 5–8), on robot link3, port 5006
- **skin5** — 4 sensors (global IDs 9–12), on robot link5, port 5007

---

## `sensors_config.yaml`

Controls which links and sensors are shown in the visualizer and playback.

```yaml
# Links to display. Comment out a line to hide that link (skin + sensors).
enabled_links:
  - link3
  - link2
  - link5
  # - link6

# Disable individual sensors by name.
# Names: L3-0..L3-6 (link3 skin), L2-0..L2-4 (link3 skin), L5-0..L5-3 (link5 skin)
disabled_sensors: []
# disabled_sensors:
#   - L2-2

# Rotation offsets for skin assemblies, in degrees [roll, pitch, yaw].
rotation_offsets:
  # link2_hybrid: [0, -45, 0]
  link6_hybrid: [0, 180, 0]

# Translation offsets for skin assemblies, in mm [x, y, z].
# translation_offsets:
#   link2_hybrid: [0, 0, 5]
```

---

## UDP Ports

| Port | Skin | Sensors |
|------|------|---------|
| 5005 | skin2 (link2) | L3-0 … L3-6 |
| 5006 | skin3 (link3) | L2-0 … L2-4 |
| 5007 | skin5 (link5) | L5-0 … L5-3 |

---

## Franka Controller

```bash
cd ~/franka_ros2_ws
source install/setup.bash
source ../ros2_ws/install/setup.bash

ros2 launch franka_bringup example.launch.py \
  robot_ip:=192.168.0.100 \
  controller_name:=move_to_start_example_controller
```

---

## ROS2 Pipeline

```bash
# Terminal 1 — send sensor data over UDP
local_ws
ros2 run udp_tof_listener udp_grid_listener_array

# Terminal 2 — transform to point clouds
local_ws
ros2 run pointcloud talker

# Terminal 3 — load robot description and skin transforms
local_ws
ros2 launch gentact_ros_tools spad.launch.py

# Terminal 4 — Franka controller (see above)

# Terminal 5 — Foxglove bridge for visualization
source ~/franka_ros2_ws/install/setup.bash
source ../ros2_ws/install/setup.bash
ros2 run foxglove_bridge foxglove_bridge
```
