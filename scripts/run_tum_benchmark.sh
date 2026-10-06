#!/usr/bin/env bash
# Benchmark runner for TUM Freiburg 1 Desk replay and RTAB-Map SLAM
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

START_FRAME="${1:-100}"
END_FRAME="${2:-133}"
OUT_DIR="${3:-results/stage3/tum_fr1_desk}"
RATE_MULT="${4:-3.0}"

mkdir -p "$OUT_DIR"

if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

if [ -f "$SCRIPT_DIR/ros2_ws/install/setup.bash" ]; then
    source "$SCRIPT_DIR/ros2_ws/install/setup.bash"
fi

cd "$SCRIPT_DIR"

echo "Running TUM Benchmark Replay (frames $START_FRAME..$END_FRAME) at ${RATE_MULT}x rate"
echo "Dataset: data/raw/rgbd_dataset_freiburg1_desk"
echo "Output Directory: $OUT_DIR"
echo "Simulated Time: use_sim_time=true (/clock)"
echo "Mode: Auto-shutdown on sequence completion"

ros2 launch scene_graph_ros tum_scene_graph.launch.py \
    use_rviz:=false \
    use_viewer:=false \
    use_rtabmap_viz:=false \
    config_path:="configs/tum_fr1_desk.yaml" \
    dataset_root:="data/raw/rgbd_dataset_freiburg1_desk" \
    depth_scale:=5000.0 \
    start_frame:="$START_FRAME" \
    end_frame:="$END_FRAME" \
    rate_multiplier:="$RATE_MULT" \
    shutdown_on_finish:=true \
    drop_old_frames:=false \
    debug_frame_packet_only:=true \
    telemetry_log_path:="${OUT_DIR}/estimator_state.csv"

echo "Benchmark replay complete. Artifacts written to $OUT_DIR."
