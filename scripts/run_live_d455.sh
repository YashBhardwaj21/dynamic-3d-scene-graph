#!/usr/bin/env bash
# Production live runner for Intel RealSense D455 with RTAB-Map SLAM and SceneGraph
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source ROS 2 Humble
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

# Source workspace install
if [ -f "$SCRIPT_DIR/ros2_ws/install/setup.bash" ]; then
    source "$SCRIPT_DIR/ros2_ws/install/setup.bash"
fi

# Detect Python virtualenv and dependencies
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.10")"
EXTRA_SITE_PACKAGES=""
if [ -n "$VIRTUAL_ENV" ] && [ -d "$VIRTUAL_ENV/lib/python$PY_VER/site-packages" ]; then
    EXTRA_SITE_PACKAGES=":$VIRTUAL_ENV/lib/python$PY_VER/site-packages"
elif [ -d "$HOME/myenv/lib/python$PY_VER/site-packages" ]; then
    EXTRA_SITE_PACKAGES=":$HOME/myenv/lib/python$PY_VER/site-packages"
elif [ -d "$SCRIPT_DIR/.venv/lib/python$PY_VER/site-packages" ]; then
    EXTRA_SITE_PACKAGES=":$SCRIPT_DIR/.venv/lib/python$PY_VER/site-packages"
elif [ -d "$SCRIPT_DIR/myenv/lib/python$PY_VER/site-packages" ]; then
    EXTRA_SITE_PACKAGES=":$SCRIPT_DIR/myenv/lib/python$PY_VER/site-packages"
fi

export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR/ros2_ws/src/scene_graph_ros:$SCRIPT_DIR/ros2_ws/src/d455_bridge${EXTRA_SITE_PACKAGES}:$PYTHONPATH"

# Set display for WSLg GUI window rendering
export DISPLAY="${DISPLAY:-:0}"
if [ -z "$WAYLAND_DISPLAY" ] && [ -e "/mnt/wslg/runtime-dir/wayland-0" ]; then
    export WAYLAND_DISPLAY="wayland-0"
fi

cd "$SCRIPT_DIR"

echo "Starting Intel RealSense D455 Live Runtime"
echo "Pipeline: RealSense D455 -> RTAB-Map SLAM -> SceneGraph"
echo "Time Domain: System wall clock (use_sim_time=false)"
echo "Freshness: High-accuracy live SLAM (pose_max_age=80ms)"
echo "Visualization: RViz2 (Window 3) + 2D Detections & Tracks (Windows 1 & 2)"

exec ros2 launch scene_graph_ros live_scene_graph.launch.py \
    use_bridge:=false \
    use_rtabmap:=true \
    use_scenegraph:=true \
    use_rviz:=true \
    use_viewer:=true \
    config_path:=configs/runtime/live_d455.yaml \
    localization_mode:=slam \
    "$@"
