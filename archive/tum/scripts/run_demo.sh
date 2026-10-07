#!/usr/bin/env bash
set -e

# Resolve repository root directory dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source ROS 2 Humble if present
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

# Ensure workspace install is sourced if present
if [ -f "$SCRIPT_DIR/ros2_ws/install/setup.bash" ]; then
    source "$SCRIPT_DIR/ros2_ws/install/setup.bash"
fi

# Dynamically detect Python version and site-packages from active/local virtualenvs
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

# Set PYTHONPATH to include perception core, ROS nodes, and virtualenv dependencies
export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR/ros2_ws/src/scene_graph_ros:$SCRIPT_DIR/ros2_ws/src/d455_bridge${EXTRA_SITE_PACKAGES}:$PYTHONPATH"

# Set DISPLAY for WSLg GUI window rendering (RViz2 + OpenCV 2D Viewer)
export DISPLAY="${DISPLAY:-:0}"
if [ -z "$WAYLAND_DISPLAY" ] && [ -e "/mnt/wslg/runtime-dir/wayland-0" ]; then
    export WAYLAND_DISPLAY="wayland-0"
fi

cd "$SCRIPT_DIR"

echo "Starting Dynamic 3D Scene Graph Real-time Demo..."
echo "Display: $DISPLAY | Wayland: $WAYLAND_DISPLAY"
echo "Ingesting TUM RGB-D Sequence -> YOLOE Perception -> Geometry"
echo "-> Causal Tracker -> Spatial Hierarchy -> Temporal Relations"
echo "-> 3D RViz Visualization + 2D Perception Dashboard"

exec ros2 launch scene_graph_ros tum_scene_graph.launch.py "$@"
