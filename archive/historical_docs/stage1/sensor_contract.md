# Stage 1: Canonical Sensor Contract

## 1. SensorFrame Specification

The `SensorFrame` (`src/scene_graph/data/sensor_frame.py`) is the immutable acquisition-level representation of a multi-modal sensor event.

### Fields and Semantic Invariants

| Field | Type | Description | Invariant / Validation |
| :--- | :--- | :--- | :--- |
| `session_id` | `str` | Sensor session identifier | Non-empty string; unique per hardware connect |
| `sequence_number` | `int` | Monotonic frame sequence counter | Monotonically increasing ($\ge 0$) |
| `timestamp` | `Timestamp` | Authoritative acquisition timestamp | Immutable; carries explicit `TimestampDomain` |
| `rgb` | `np.ndarray` | Color image matrix | Shape `(H, W, 3)`, `uint8` dtype |
| `camera_intrinsics` | `CameraIntrinsics` | Pinhole optical intrinsics | `fx, fy, cx, cy > 0`, matches stream resolution |
| `depth` | `Optional[np.ndarray]`| Metric depth matrix in meters | `float32` dtype (or `None`). Reject integer depth |
| `depth_scale` | `float` | Meters per raw depth unit | Strictly positive and finite (D455 $\approx 0.001$) |
| `distortion` | `tuple[float, ...]` | Lens distortion coefficients | Preserved from active stream profile |
| `distortion_model` | `str` | OpenCV distortion model | e.g. `"plumb_bob"` or `"brown_conrady"` |
| `imu_samples` | `tuple[IMUSample, ...]`| Batch of high-rate inertial samples | Windowed samples in $(t_{k-1}, t_k]$ |
| `status` | `StreamStatus` | Operational sensor stream health | `OK`, `DEGRADED`, or `DEPTH_DROPPED` |

### Excluded Downstream Concepts

`SensorFrame` strictly excludes:
- `world_T_camera` (world pose matrix)
- `pose_timestamp` and `pose_age`
- `relation_frame` (gravity / heading reference frames)
- Track IDs, detections, or semantic graphs

Downstream concepts are injected via the compatibility bridge `FramePacket.from_sensor_frame()`.

## 2. IMUSample Specification

The `IMUSample` (`src/scene_graph/data/sensor_frame.py`) represents 6-axis inertial motion:
- `timestamp`: Sensor timestamp in seconds
- `accel`: `(3,)` linear acceleration in $m/s^2$ $[a_x, a_y, a_z]$
- `gyro`: `(3,)` angular velocity in $rad/s$ $[\omega_x, \omega_y, \omega_z]$
- `domain`: `TimestampDomain` (typically `HARDWARE_CLOCK`)
- `sequence`: Hardware IMU sample sequence counter
