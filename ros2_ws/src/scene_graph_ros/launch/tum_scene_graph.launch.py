"""Launch file for TUM dataset replay with SceneGraph ROS Node and RViz."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("scene_graph_ros")
    default_rviz_config = os.path.join(pkg_share, "rviz", "scene_graph.rviz")

    # Launch arguments
    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value="configs/tum_fr1_desk.yaml",
        description="Path to SceneGraph configuration YAML",
    )

    dataset_root_arg = DeclareLaunchArgument(
        "dataset_root",
        default_value="data/raw/rgbd_dataset_freiburg1_desk",
        description="Path to TUM RGB-D sequence directory",
    )

    rate_multiplier_arg = DeclareLaunchArgument(
        "rate_multiplier",
        default_value="1.0",
        description="Playback rate multiplier (1.0 = real-time, 2.0 = 2x, 0.0 = as fast as possible)",
    )

    start_frame_arg = DeclareLaunchArgument(
        "start_frame",
        default_value="0",
        description="Starting frame index",
    )

    end_frame_arg = DeclareLaunchArgument(
        "end_frame",
        default_value="-1",
        description="Ending frame index (-1 for end of sequence)",
    )

    debug_frame_packet_only_arg = DeclareLaunchArgument(
        "debug_frame_packet_only",
        default_value="false",
        description="Run in FramePacket diagnostic isolation mode (Acceptance Test 2)",
    )

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Whether to launch RViz2",
    )

    world_frame_arg = DeclareLaunchArgument(
        "world_frame",
        default_value="world",
        description="World reference coordinate frame ID",
    )

    sensor_frame_arg = DeclareLaunchArgument(
        "sensor_frame",
        default_value="camera_optical_frame",
        description="Camera optical coordinate frame ID (X right, Y down, Z forward)",
    )

    queue_size_arg = DeclareLaunchArgument(
        "queue_size",
        default_value="64",
        description="Maximum frames buffered in worker queue (64 for offline replay, 2 for live)",
    )

    drop_old_frames_arg = DeclareLaunchArgument(
        "drop_old_frames",
        default_value="false",
        description="Whether to drop incoming frames when the queue is full (false for offline, true for live)",
    )

    # 1. TUM Player Node
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
                "world_frame": LaunchConfiguration("world_frame"),
                "sensor_frame": LaunchConfiguration("sensor_frame"),
            }
        ],
    )

    # 2. SceneGraph Generic Node
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
                "debug_frame_packet_only": LaunchConfiguration("debug_frame_packet_only"),
                "queue_size": LaunchConfiguration("queue_size"),
                "drop_old_frames": LaunchConfiguration("drop_old_frames"),
            }
        ],
    )

    # 3. RViz2 Node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", default_rviz_config],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    return LaunchDescription(
        [
            config_path_arg,
            dataset_root_arg,
            rate_multiplier_arg,
            start_frame_arg,
            end_frame_arg,
            debug_frame_packet_only_arg,
            use_rviz_arg,
            world_frame_arg,
            sensor_frame_arg,
            queue_size_arg,
            drop_old_frames_arg,
            tum_player_node,
            scene_graph_node,
            rviz_node,
        ]
    )
