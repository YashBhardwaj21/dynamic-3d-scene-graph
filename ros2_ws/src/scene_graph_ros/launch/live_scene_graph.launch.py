"""Launch file for live perception with RealSense, RTAB-Map, SceneGraph, and RViz."""

from __future__ import annotations

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("scene_graph_ros")
    default_rviz_config = os.path.join(pkg_share, "rviz", "scene_graph.rviz")

    use_realsense_arg = DeclareLaunchArgument(
        "use_realsense",
        default_value="true",
        description="Whether to launch realsense2_camera driver",
    )

    use_rtabmap_arg = DeclareLaunchArgument(
        "use_rtabmap",
        default_value="true",
        description="Whether to launch RTAB-Map RGB-D visual odometry & SLAM",
    )

    use_scenegraph_arg = DeclareLaunchArgument(
        "use_scenegraph",
        default_value="true",
        description="Whether to launch SceneGraph perception & tracking node",
    )

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Whether to launch RViz2 3D visualization",
    )

    use_viewer_arg = DeclareLaunchArgument(
        "use_viewer",
        default_value="true",
        description="Whether to launch 2D live perception dashboard",
    )

    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value="configs/default.yaml",
        description="Path to SceneGraph configuration YAML",
    )

    world_frame_arg = DeclareLaunchArgument(
        "world_frame",
        default_value="map",
        description="Global map reference frame (map from RTAB-Map or world)",
    )

    sensor_frame_arg = DeclareLaunchArgument(
        "sensor_frame",
        default_value="camera_color_optical_frame",
        description="Camera optical frame (Z forward, X right, Y down)",
    )

    rgb_topic_arg = DeclareLaunchArgument(
        "rgb_topic",
        default_value="/camera/camera/color/image_raw",
        description="RGB image topic from sensor or replay",
    )

    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera/camera/aligned_depth_to_color/image_raw",
        description="Aligned depth image topic",
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/camera/color/camera_info",
        description="Camera calibration topic",
    )

    queue_size_arg = DeclareLaunchArgument(
        "queue_size",
        default_value="2",
        description="Queue size for live mode (small buffer to prevent latency accumulation)",
    )

    drop_old_frames_arg = DeclareLaunchArgument(
        "drop_old_frames",
        default_value="true",
        description="Drop older frames when busy to keep live processing real-time",
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            ])
        ),
        launch_arguments={
            "enable_color": "true",
            "enable_depth": "true",
            "align_depth.enable": "true",
            "enable_gyro": "false",
            "enable_accel": "false",
            "rgb_camera.profile": "640x480x30",
            "depth_module.profile": "640x480x30",
        }.items(),
        condition=IfCondition(LaunchConfiguration("use_realsense")),
    )

    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory("rtabmap_launch"),
                "launch",
                "rtabmap.launch.py",
            ])
        ),
        launch_arguments={
            "rgb_topic": LaunchConfiguration("rgb_topic"),
            "depth_topic": LaunchConfiguration("depth_topic"),
            "camera_info_topic": LaunchConfiguration("camera_info_topic"),
            "frame_id": "camera_link",
            "approx_sync": "true",
            "wait_imu_to_init": "false",
            "rtabmap_viz": "false",
            "rviz": "false",
        }.items(),
        condition=IfCondition(LaunchConfiguration("use_rtabmap")),
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
                "rgb_topic": LaunchConfiguration("rgb_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "queue_size": LaunchConfiguration("queue_size"),
                "drop_old_frames": LaunchConfiguration("drop_old_frames"),
            }
        ],
        condition=IfCondition(LaunchConfiguration("use_scenegraph")),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", default_rviz_config],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    viewer_node = Node(
        package="scene_graph_ros",
        executable="live_2d_viewer",
        name="live_2d_viewer",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_viewer")),
    )

    return LaunchDescription([
        use_realsense_arg,
        use_rtabmap_arg,
        use_scenegraph_arg,
        use_rviz_arg,
        use_viewer_arg,
        config_path_arg,
        world_frame_arg,
        sensor_frame_arg,
        rgb_topic_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        queue_size_arg,
        drop_old_frames_arg,
        realsense_launch,
        rtabmap_launch,
        scene_graph_node,
        rviz_node,
        viewer_node,
    ])
