#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi
if [ -f "$SCRIPT_DIR/ros2_ws/install/setup.bash" ]; then
    source "$SCRIPT_DIR/ros2_ws/install/setup.bash"
fi
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.10")"
EXTRA_SITE_PACKAGES=""
if [ -d "$HOME/myenv/lib/python$PY_VER/site-packages" ]; then
    EXTRA_SITE_PACKAGES=":$HOME/myenv/lib/python$PY_VER/site-packages"
fi
export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR/ros2_ws/src/scene_graph_ros:$SCRIPT_DIR/ros2_ws/src/d455_bridge${EXTRA_SITE_PACKAGES}:$PYTHONPATH"
cd "$SCRIPT_DIR"
python3 -m pytest "$@"
