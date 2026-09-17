"""Launch file for live perception with RealSense, RTAB-Map, SceneGraph, and RViz."""

from __future__ import annotations

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("scene_graph_ros")
    default_rviz_config = os.path.join(pkg_share, "rviz", "scene_graph.rviz")

    use_bridge_arg = DeclareLaunchArgument(
        "use_bridge",
        default_value="true",
        description="Whether to launch D455 TCP receiver bridge node",
    )

    use_rtabmap_arg = DeclareLaunchArgument(
        "use_rtabmap",
        default_value="false",
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
        default_value="configs/live_d455.yaml",
        description="Path to SceneGraph configuration YAML",
    )

    min_hits_arg = DeclareLaunchArgument(
        "min_hits",
        default_value="1",
        description="Minimum consecutive hits before promoting track to ACTIVE",
    )

    max_missing_seconds_arg = DeclareLaunchArgument(
        "max_missing_seconds",
        default_value="5.0",
        description="Maximum seconds before unobserved track is declared LOST",
    )

    world_frame_arg = DeclareLaunchArgument(
        "world_frame",
        default_value="world",
        description="Global map reference frame (world or map from RTAB-Map)",
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

    localization_mode_arg = DeclareLaunchArgument(
        "localization_mode",
        default_value="camera_local",
        description="Localization mode: 'world' (requires active TF/SLAM) or 'camera_local' (transient non-persistent frame)",
    )

    depth_scale_arg = DeclareLaunchArgument(
        "depth_scale",
        default_value="1000.0",
        description="Depth scale conversion factor (raw integer depth units per meter; 1000.0 for RealSense D455 mm, 5000.0 for TUM)",
    )

    d455_bridge_node = Node(
        package="d455_bridge",
        executable="d455_receiver",
        name="d455_receiver",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_bridge")),
    )

    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_map_to_camera",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", LaunchConfiguration("world_frame"),
            "--child-frame-id", LaunchConfiguration("sensor_frame"),
        ],
        condition=UnlessCondition(LaunchConfiguration("use_rtabmap")),
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
            "frame_id": LaunchConfiguration("sensor_frame"),
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
                "depth_scale": LaunchConfiguration("depth_scale"),
                "queue_size": LaunchConfiguration("queue_size"),
                "drop_old_frames": LaunchConfiguration("drop_old_frames"),
                "localization_mode": LaunchConfiguration("localization_mode"),
                "min_hits": LaunchConfiguration("min_hits"),
                "max_missing_seconds": LaunchConfiguration("max_missing_seconds"),
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
        use_bridge_arg,
        use_rtabmap_arg,
        use_scenegraph_arg,
        use_rviz_arg,
        use_viewer_arg,
        config_path_arg,
        min_hits_arg,
        max_missing_seconds_arg,
        world_frame_arg,
        sensor_frame_arg,
        rgb_topic_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        depth_scale_arg,
        queue_size_arg,
        drop_old_frames_arg,
        localization_mode_arg,
        d455_bridge_node,
        static_tf_node,
        rtabmap_launch,
        scene_graph_node,
        rviz_node,
        viewer_node,
    ])
