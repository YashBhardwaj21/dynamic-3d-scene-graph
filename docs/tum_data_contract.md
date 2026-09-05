# Stage 1: TUM Data Contract

This document explicitly defines the boundaries and assumptions for data ingestion, sensor synchronization, and camera geometry for the Dynamic 3D Scene Graph project.

## Dataset Structure
- Dataset must follow the TUM RGB-D format (e.g. `rgbd_dataset_freiburg1_desk`).
- Core streams: `rgb.txt`, `depth.txt`, `groundtruth.txt`.
- Space-delimited lists. Comment lines starting with `#` are ignored.

## Frame Indexing and Time
- `frame_index` strictly reflects the original global line index in the parsed `rgb.txt` stream.
- RGB is the master temporal stream. Depth and ground truth (pose) are synchronized against the RGB anchors.
- All timestamps are preserved as double-precision `float` UNIX seconds.

## Data Modes & Fallbacks
- **RGB is mandatory**: If an RGB image fails to load for a targeted frame, the pipeline raises an exception.
- **Depth is optional**: If no depth map matches within the threshold, `has_depth = False`.
- **Pose is optional**: If no ground truth pose matches within the threshold, `has_pose = False`.
- The pipeline intentionally emits packets for all targeted frames to preserve sequential continuity, leaving missing data explicitly flagged.

## Sensor Formats
### RGB
- **Format**: `(480, 640, 3)`
- **Dtype**: `uint8`
- **Color space**: Strictly `RGB` order. Open-CV standard `BGR` is converted immediately upon loading.

### Depth
- **Format**: `(480, 640)`
- **Dtype**: `uint16` raw.
- **Scale**: TUM uses a static configuration where 5000 units = 1 meter. 
- **Processing**: The geometry pipeline abstracts this entirely. Point clouds strictly request `float64` *metric* depth.

### Pose (Ground Truth)
- **Convention**: Provides `world_T_camera` (transformation from camera coordinate space into the global world reference space).
- **Quaternion**: TUM uses `qx, qy, qz, qw`. These are extracted, normalized (failing if zero), and converted to a 3x3 rotation matrix.
- **Matrix**: Built into a homogeneous 4x4 `float64` SE(3) matrix.

## Camera Intrinsics
- The default pipeline assumes TUM `fr1` intrinsics:
  - `fx = 525.0`
  - `fy = 525.0`
  - `cx = 319.5`
  - `cy = 239.5`
  - `width = 640`, `height = 480`

## Synchronization Algorithm
- Uses an O(N+M) *monotonic nearest-neighbor* association logic, prioritizing sequential alignment. 
- **Configurable Thresholds**: RGB-to-Depth and RGB-to-Pose have independently configurable `max_dt` thresholds (default `0.02` seconds).

## Coordinate Systems & Reference Frames
- **World Frame**: The direct result of applying the `world_T_camera` pose.
- **Alignment (Gravity)**: A calibration module can compute an alignment transform by estimating the principal support plane (via SVD/PCA on 3D points) and mapping it to the `[0, 0, 1]` Z-axis. This establishes a "Stable World" frame for evaluating spatial relations (e.g., `ON`, `ABOVE`).
