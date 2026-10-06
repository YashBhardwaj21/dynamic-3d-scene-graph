"""Production Launch file for live Intel RealSense D455 perception with RTAB-Map SLAM and SceneGraph."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter


def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory("scene_graph_ros")
    default_rviz_config = os.path.join(pkg_share, "rviz", "scene_graph.rviz")

    # Resolve localization mode (strictly live-safe: slam vs camera_local)
    raw_loc_mode = context.launch_configurations.get("localization_mode", "slam").strip().lower()
    raw_use_rtabmap = context.launch_configurations.get("use_rtabmap", "true").strip().lower()

    if raw_loc_mode in ("slam", "world"):
        launch_rtabmap = True
        node_localization_mode = "slam"
        use_latest_tf = True
        publish_static_tf = False
    elif raw_loc_mode in ("camera_local", "local", "camera"):
        launch_rtabmap = False
        node_localization_mode = "camera_local"
        use_latest_tf = False
        publish_static_tf = True
    elif raw_use_rtabmap in ("true", "1", "yes"):
        launch_rtabmap = True
        node_localization_mode = "slam"
        use_latest_tf = True
        publish_static_tf = False
    else:
        launch_rtabmap = False
        node_localization_mode = "camera_local"
        use_latest_tf = False
        publish_static_tf = True

    # Real-world runtime invariant: use_sim_time is strictly False
    actions = [
        SetParameter(name="use_sim_time", value=False),
    ]

    # 1. D455 TCP Receiver Bridge (Windows pyrealsense2 sender -> WSL2 ROS 2 bridge)
    d455_bridge_node = Node(
        package="d455_bridge",
        executable="d455_receiver",
        name="d455_receiver",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_bridge")),
    )
    actions.append(d455_bridge_node)

    # 2. Static transform publisher for camera_local non-SLAM mode
    if publish_static_tf:
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
        )
        actions.append(static_tf_node)

    # 3. RTAB-Map SLAM and visual odometry
    if launch_rtabmap:
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
                "map_frame_id": LaunchConfiguration("world_frame"),
                "odom_frame_id": "odom",
                "approx_sync": "true",
                "approx_sync_max_interval": "0.05",
                "wait_imu_to_init": "false",
                "subscribe_depth": "true",
                "subscribe_rgb": "true",
                "subscribe_odom_info": "true",
                "rtabmap_viz": LaunchConfiguration("use_rtabmap_viz"),
                "rviz": "false",
                "odom_always_process_most_recent_frame": "true",
                "use_sim_time": "false",
            }.items(),
        )
        actions.append(rtabmap_launch)

    # 4. SceneGraph perception, tracking, and spatial relation inference node
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
                "localization_mode": node_localization_mode,
                "use_latest_tf": use_latest_tf,
                "slam_pose_max_age": LaunchConfiguration("slam_pose_max_age"),
                "pose_max_dt": LaunchConfiguration("pose_max_dt"),
                "telemetry_max_age": LaunchConfiguration("telemetry_max_age"),
                "telemetry_log_path": LaunchConfiguration("telemetry_log_path"),
                "debug_frame_packet_only": False,
                "use_sim_time": False,
                "min_hits": LaunchConfiguration("min_hits"),
                "max_missing_seconds": LaunchConfiguration("max_missing_seconds"),
                "publish_scene_cloud": LaunchConfiguration("publish_scene_cloud"),
                "scene_cloud_stride": LaunchConfiguration("scene_cloud_stride"),
                "map_voxel_size_m": LaunchConfiguration("map_voxel_size_m"),
                "map_max_points": LaunchConfiguration("map_max_points"),
                "map_cloud_stride": LaunchConfiguration("map_cloud_stride"),
            }
        ],
        condition=IfCondition(LaunchConfiguration("use_scenegraph")),
    )
    actions.append(scene_graph_node)

    # 5. RViz2 3D visualization (Window 3)
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", default_rviz_config],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )
    actions.append(rviz_node)

    # 6. Live 2D perception dashboard (Windows 1 & 2)
    viewer_node = Node(
        package="scene_graph_ros",
        executable="live_2d_viewer",
        name="live_2d_viewer",
        output="screen",
        parameters=[
            {
                "split_windows": LaunchConfiguration("split_viewer_windows"),
                "detections_topic": "/scene_graph/overlay_detections",
                "tracks_topic": "/scene_graph/overlay_tracks",
                "state_topic": "/scene_graph/state",
            }
        ],
        condition=IfCondition(LaunchConfiguration("use_viewer")),
    )
    actions.append(viewer_node)

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "use_bridge",
            default_value="true",
            description="Whether to launch D455 TCP receiver bridge node",
        ),
        DeclareLaunchArgument(
            "use_rtabmap",
            default_value="true",
            description="Whether to launch RTAB-Map RGB-D visual odometry & SLAM",
        ),
        DeclareLaunchArgument(
            "use_scenegraph",
            default_value="true",
            description="Whether to launch SceneGraph perception & tracking node",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Whether to launch RViz2 3D visualization (Window 3)",
        ),
        DeclareLaunchArgument(
            "use_viewer",
            default_value="true",
            description="Whether to launch 2D live perception dashboard (Windows 1 & 2)",
        ),
        DeclareLaunchArgument(
            "use_rtabmap_viz",
            default_value="false",
            description="Whether to launch native RTAB-Map 3D visualization GUI (Window 4)",
        ),
        DeclareLaunchArgument(
            "split_viewer_windows",
            default_value="true",
            description="Split 2D viewer into two independent windows (YOLO and Tracks/Relations)",
        ),
        DeclareLaunchArgument(
            "config_path",
            default_value="configs/live_d455.yaml",
            description="Path to SceneGraph configuration YAML",
        ),
        DeclareLaunchArgument(
            "localization_mode",
            default_value="slam",
            description="Localization mode: 'slam' (RTAB-Map active) or 'camera_local' (transient non-persistent frame)",
        ),
        DeclareLaunchArgument(
            "world_frame",
            default_value="world",
            description="Global map reference frame (world or map from RTAB-Map)",
        ),
        DeclareLaunchArgument(
            "sensor_frame",
            default_value="camera_color_optical_frame",
            description="Camera optical frame (Z forward, X right, Y down)",
        ),
        DeclareLaunchArgument(
            "rgb_topic",
            default_value="/camera/camera/color/image_raw",
            description="RGB image topic from D455 sensor bridge",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw",
            description="Aligned depth image topic from D455 sensor bridge",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera/color/camera_info",
            description="Camera calibration topic from D455 sensor bridge",
        ),
        DeclareLaunchArgument(
            "depth_scale",
            default_value="1000.0",
            description="Depth scale conversion factor (1000.0 for RealSense D455 raw mm depth)",
        ),
        DeclareLaunchArgument(
            "queue_size",
            default_value="2",
            description="Queue size for live mode (small buffer to prevent latency accumulation)",
        ),
        DeclareLaunchArgument(
            "drop_old_frames",
            default_value="true",
            description="Drop older frames when busy to keep live processing real-time",
        ),
        DeclareLaunchArgument(
            "slam_pose_max_age",
            default_value="0.08",
            description="Tighter live freshness limit in seconds (80 ms; rejects stale pose backlog)",
        ),
        DeclareLaunchArgument(
            "pose_max_dt",
            default_value="0.05",
            description="Maximum timestamp delta for pose lookup (50 ms)",
        ),
        DeclareLaunchArgument(
            "telemetry_max_age",
            default_value="0.10",
            description="Maximum age of RTAB-Map registration telemetry (100 ms)",
        ),
        DeclareLaunchArgument(
            "telemetry_log_path",
            default_value="",
            description="Optional path to log live CSV telemetry and estimated.tum trajectory",
        ),
        DeclareLaunchArgument(
            "min_hits",
            default_value="5",
            description="Minimum consecutive hits before promoting track to ACTIVE",
        ),
        DeclareLaunchArgument(
            "max_missing_seconds",
            default_value="-1.0",
            description="Maximum seconds before unobserved track is declared LOST (-1.0 to use config YAML)",
        ),
        DeclareLaunchArgument(
            "publish_scene_cloud",
            default_value="false",
            description="Enable ephemeral per-frame scene cloud (disabled by default; use map cloud instead)",
        ),
        DeclareLaunchArgument(
            "scene_cloud_stride",
            default_value="4",
            description="Subsampling stride for per-frame scene cloud",
        ),
        DeclareLaunchArgument(
            "map_voxel_size_m",
            default_value="0.02",
            description="Voxel size (m) for growing map cloud downsampling",
        ),
        DeclareLaunchArgument(
            "map_max_points",
            default_value="500000",
            description="Maximum points in map before voxel downsampling triggers",
        ),
        DeclareLaunchArgument(
            "map_cloud_stride",
            default_value="4",
            description="Subsampling stride for map cloud unprojection",
        ),
        OpaqueFunction(function=launch_setup),
    ])
