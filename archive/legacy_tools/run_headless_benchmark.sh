#!/usr/bin/env bash
set -e

SOURCE_SEQ="$1"
START_FRAME="${2:-0}"
END_FRAME="${3:-100}"
OUT_DIR="${4:-results/stage3/tum_fr1_desk}"
RATE_MULT="${5:-3.0}"

mkdir -p "$OUT_DIR"

source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash

DEBUG_MODE="${6:-true}"

if [ "$SOURCE_SEQ" = "my_desk_sequence" ]; then
    DATASET_ROOT="data/raw/my_desk_sequence"
    DEPTH_SCALE="1000.0"
else
    DATASET_ROOT="data/raw/rgbd_dataset_freiburg1_desk"
    DEPTH_SCALE="5000.0"
fi

echo "Starting headless replay for $SOURCE_SEQ ($DATASET_ROOT) frames $START_FRAME..$END_FRAME into $OUT_DIR (debug_frame_packet_only=$DEBUG_MODE)..."
ros2 launch scene_graph_ros tum_scene_graph.launch.py \
    use_rviz:=false \
    use_viewer:=false \
    use_rtabmap_viz:=false \
    config_path:="configs/${SOURCE_SEQ}.yaml" \
    dataset_root:="$DATASET_ROOT" \
    depth_scale:="$DEPTH_SCALE" \
    start_frame:="$START_FRAME" \
    end_frame:="$END_FRAME" \
    rate_multiplier:="$RATE_MULT" \
    shutdown_on_finish:=true \
    drop_old_frames:=false \
    debug_frame_packet_only:="$DEBUG_MODE" \
    telemetry_log_path:="${OUT_DIR}/estimator_state.csv"
