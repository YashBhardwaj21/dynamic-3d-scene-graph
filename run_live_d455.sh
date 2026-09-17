#!/usr/bin/env bash
set -e

# Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# Ensure workspace install is sourced if present
if [ -f "/mnt/c/Users/Yash Bhardwaj/Desktop/perception/ros2_ws/install/setup.bash" ]; then
    source "/mnt/c/Users/Yash Bhardwaj/Desktop/perception/ros2_ws/install/setup.bash"
fi

# Set PYTHONPATH to include perception core, ROS nodes, and virtualenv dependencies
export PYTHONPATH="/mnt/c/Users/Yash Bhardwaj/Desktop/perception/src:/mnt/c/Users/Yash Bhardwaj/Desktop/perception/ros2_ws/src/scene_graph_ros:/mnt/c/Users/Yash Bhardwaj/Desktop/perception/ros2_ws/src/d455_bridge:/home/saturn/myenv/lib/python3.10/site-packages:$PYTHONPATH"

# Set DISPLAY for WSLg GUI window rendering (RViz2 + OpenCV 2D Viewer)
export DISPLAY="${DISPLAY:-:0}"
if [ -z "$WAYLAND_DISPLAY" ] && [ -e "/mnt/wslg/runtime-dir/wayland-0" ]; then
    export WAYLAND_DISPLAY="wayland-0"
fi

cd "/mnt/c/Users/Yash Bhardwaj/Desktop/perception"

echo "============================================================"
echo " Starting Real-Time Intel RealSense D455 Scene Graph..."
echo " Listening on TCP Port 5000 for Windows D455 Sender"
echo " Display: $DISPLAY | Wayland: $WAYLAND_DISPLAY"
echo " Ingesting Live RGB-D -> YOLOE Perception -> Geometry"
echo " -> Causal Tracker -> Spatial Hierarchy -> Temporal Relations"
echo " -> 3D RViz Visualization + 2D Perception Dashboard"
echo "============================================================"

exec ros2 launch scene_graph_ros live_scene_graph.launch.py use_bridge:=true use_rviz:=true use_viewer:=true "$@"
