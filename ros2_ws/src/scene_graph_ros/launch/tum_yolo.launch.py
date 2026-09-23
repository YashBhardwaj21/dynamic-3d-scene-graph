"""Launch YOLO 2D Perception, 3D Tracking, Spatial Relations, and 2D Viewer.

Dedicated to the YOLO perception pipeline:
- tum_player (replaying RGB, depth, camera_info, and ground-truth TF)
- scene_graph_node (YOLO object detection + causal 3D tracking + spatial relations)
- live_2d_viewer (Window 1: YOLO detections & masks, Window 2: Tracks & 3D relations)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    tum_player_node = Node(
        package="scene_graph_ros",
        executable="tum_player",
        name="tum_player",
        output="screen",
        parameters=[
            {
                "config_path": LaunchConfiguration("config_path"),
                "dataset_root": LaunchConfiguration("dataset_root"),
                "rate_multiplier": LaunchConfiguration("rate_multiplier"),
                "start_frame": LaunchConfiguration("start_frame"),
                "end_frame": LaunchConfiguration("end_frame"),
                "frame_stride": LaunchConfiguration("frame_stride"),
                "world_frame": LaunchConfiguration("world_frame"),
                "sensor_frame": LaunchConfiguration("sensor_frame"),
                "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
                "loop": LaunchConfiguration("loop"),
                "publish_groundtruth_tf": True,
                "depth_scale": LaunchConfiguration("depth_scale"),
                "min_subscribers": 1,
            }
        ],
        condition=IfCondition(LaunchConfiguration("launch_player")),
    )

    scene_graph_node = Node(
        package="scene_graph_ros",
        executable="scene_graph_node",
        name="scene_graph_node",
        output="screen",
        parameters=[
            {
                "config_path": LaunchConfiguration("config_path"),
                "world_frame": LaunchConfiguration("world_frame"),
                "sensor_frame": LaunchConfiguration("sensor_frame"),
                "depth_scale": LaunchConfiguration("depth_scale"),
                "queue_size": LaunchConfiguration("queue_size"),
                "drop_old_frames": LaunchConfiguration("drop_old_frames"),
                "localization_mode": LaunchConfiguration("localization_mode"),
                "use_latest_tf": True,
                "slam_pose_max_age": 2.0,
                "pose_max_dt": 0.05,
                "rgb_topic": "/tum/rgb/image_raw",
                "depth_topic": "/tum/depth/image_raw",
                "camera_info_topic": "/tum/rgb/camera_info",
                "map_voxel_size_m": LaunchConfiguration("map_voxel_size_m"),
                "map_max_points": LaunchConfiguration("map_max_points"),
                "map_cloud_stride": LaunchConfiguration("map_cloud_stride"),
            }
        ],
    )

    viewer_node = Node(
        package="scene_graph_ros",
        executable="live_2d_viewer",
        name="live_2d_viewer",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "split_windows": LaunchConfiguration("split_viewer_windows"),
            }
        ],
        condition=IfCondition(LaunchConfiguration("use_viewer")),
    )

    return [tum_player_node, scene_graph_node, viewer_node]


def generate_launch_description():
    pkg_share = get_package_share_directory("scene_graph_ros")
    default_config = os.path.join(pkg_share, "configs", "tum_fr1_desk.yaml")
    if not os.path.exists(default_config):
        default_config = "configs/tum_fr1_desk.yaml"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "launch_player",
                default_value="true",
                description="Whether to launch TUM player sequence streamer",
            ),
            DeclareLaunchArgument(
                "dataset_root",
                default_value="data/raw/rgbd_dataset_freiburg1_desk",
                description="Path to TUM RGB-D sequence directory",
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value=default_config,
                description="Path to SceneGraph configuration YAML",
            ),
            DeclareLaunchArgument(
                "publish_rate_hz",
                default_value="2.0",
                description="Replay input rate in Hz",
            ),
            DeclareLaunchArgument(
                "rate_multiplier",
                default_value="1.0",
                description="Playback rate multiplier",
            ),
            DeclareLaunchArgument(
                "start_frame",
                default_value="0",
                description="Starting frame index",
            ),
            DeclareLaunchArgument(
                "end_frame",
                default_value="-1",
                description="Ending frame index (-1 = end of sequence)",
            ),
            DeclareLaunchArgument(
                "frame_stride",
                default_value="1",
                description="Step between published frames",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="true",
                description="Loop playback continuously",
            ),
            DeclareLaunchArgument(
                "localization_mode",
                default_value="world",
                description="Localization mode: 'world' (uses player TF) or 'camera_local'",
            ),
            DeclareLaunchArgument(
                "world_frame",
                default_value="world",
                description="World reference frame ID",
            ),
            DeclareLaunchArgument(
                "sensor_frame",
                default_value="camera_optical_frame",
                description="Camera optical frame ID",
            ),
            DeclareLaunchArgument(
                "depth_scale",
                default_value="5000.0",
                description="Raw depth units per metre",
            ),
            DeclareLaunchArgument(
                "queue_size",
                default_value="5",
                description="SceneGraph worker queue size",
            ),
            DeclareLaunchArgument(
                "drop_old_frames",
                default_value="true",
                description="Drop queued frames when processing falls behind",
            ),
            DeclareLaunchArgument(
                "use_viewer",
                default_value="true",
                description="Launch 2D perception viewer",
            ),
            DeclareLaunchArgument(
                "split_viewer_windows",
                default_value="true",
                description="Split 2D viewer into two independent windows (YOLO and Tracks/Relations)",
            ),
            DeclareLaunchArgument(
                "map_voxel_size_m",
                default_value="0.008",
                description="Voxel size for the growing map cloud",
            ),
            DeclareLaunchArgument(
                "map_max_points",
                default_value="2000000",
                description="Maximum accumulated map points",
            ),
            DeclareLaunchArgument(
                "map_cloud_stride",
                default_value="2",
                description="Pixel stride for map cloud generation",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
