"""Pytest configuration and global sys.path fixture."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src_path = str(ROOT / "src")
ros_pkg_path = str(ROOT / "ros2_ws" / "src" / "scene_graph_ros")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

if ros_pkg_path not in sys.path:
    sys.path.insert(0, ros_pkg_path)
