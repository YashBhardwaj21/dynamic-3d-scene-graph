"""Launch TUM RGB-D replay with SLAM, ground-truth, or camera-local localization.

tum_player uses wall time to pace playback and publishes dataset timestamps on /clock.
RTAB-Map, SceneGraph, and RViz run with use_sim_time=True.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter


def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory("scene_graph_ros")
    default_rviz_config = os.path.join(pkg_share, "rviz", "scene_graph.rviz")

    # Resolve the authoritative localization mode.
    raw_loc_mode = (
        context.launch_configurations.get("localization_mode", "")
        .strip()
        .lower()
    )
    raw_use_rtabmap = (
        context.launch_configurations.get("use_rtabmap", "")
        .strip()
        .lower()
    )
    raw_publish_gt = (
        context.launch_configurations.get("publish_groundtruth_tf", "")
        .strip()
        .lower()
    )

    if raw_loc_mode in ("slam", "ground_truth", "camera_local"):
        resolved_mode = raw_loc_mode
    elif raw_use_rtabmap == "true":
        resolved_mode = "slam"
    elif raw_publish_gt == "true":
        resolved_mode = "ground_truth"
    elif raw_loc_mode in ("local", "camera"):
        resolved_mode = "camera_local"
    else:
        resolved_mode = "slam"

    # Enable exactly one localization source.
    if resolved_mode == "slam":
        launch_rtabmap = True
        publish_gt_tf = False
        publish_static_tf = False
        node_localization_mode = "slam"
        use_latest_tf = True
    elif resolved_mode == "ground_truth":
        launch_rtabmap = False
        publish_gt_tf = True
        publish_static_tf = False
        node_localization_mode = "ground_truth"
        use_latest_tf = False
    else:
        launch_rtabmap = False
        publish_gt_tf = False
        publish_static_tf = True
        node_localization_mode = "camera_local"
        use_latest_tf = False

    # tum_player uses wall-clock pacing and publishes /clock from dataset timestamps.
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
                "publish_groundtruth_tf": publish_gt_tf,
                "depth_scale": LaunchConfiguration("depth_scale"),
                "slam_depth_topic": "/tum/depth_m/image_raw",
                "min_subscribers": 3 if launch_rtabmap else 1,
            }
        ],
    )

    # Downstream nodes use the dataset clock.
    downstream_actions = [
        SetParameter(name="use_sim_time", value=True),
    ]

    # Launch one localization source.
    if launch_rtabmap:
        downstream_actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            get_package_share_directory("rtabmap_launch"),
                            "launch",
                            "rtabmap.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "rgb_topic": "/tum/rgb/image_raw",
                    "depth_topic": "/tum/depth_m/image_raw",
                    "camera_info_topic": "/tum/rgb/camera_info",
                    "frame_id": LaunchConfiguration("sensor_frame"),
                    "map_frame_id": LaunchConfiguration("world_frame"),

                    # Use bounded approximate synchronization for offline replay.
                    "approx_sync": "true",
                    "approx_sync_max_interval": "0.05",

                    # Keep input queues bounded.
                    "topic_queue_size": "5",
                    "sync_queue_size": "10",

                    # Do not let CPU odometry accumulate a frame backlog.
                    "odom_always_process_most_recent_frame": "false",

                    # Depth is already float32 metres.
                    "depth_scale": "1.0",

                    "wait_imu_to_init": "false",
                    "use_sim_time": "true",
                    "rtabmap_viz": LaunchConfiguration("use_rtabmap_viz"),
                    "rviz": "false",
                    "args": (
                        "-d "
                        "--Grid/CellSize 0.01 "
                        "--Grid/VoxelSize 0.005 "
                        "--Grid/RangeMax 4.0 "
                        "--RGBD/LinearUpdate 0.01 "
                        "--RGBD/AngularUpdate 0.01 "
                        "--Rtabmap/DetectionRate 2.0"
                    ),
                }.items(),
            )
        )
    elif publish_static_tf:
        downstream_actions.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_world_to_camera",
                arguments=[
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--roll",
                    "0",
                    "--pitch",
                    "0",
                    "--yaw",
                    "0",
                    "--frame-id",
                    LaunchConfiguration("world_frame"),
                    "--child-frame-id",
                    LaunchConfiguration("sensor_frame"),
                ],
            )
        )

    downstream_actions.append(
        Node(
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
                    "debug_frame_packet_only": LaunchConfiguration(
                        "debug_frame_packet_only"
                    ),
                    "queue_size": LaunchConfiguration("queue_size"),
                    "drop_old_frames": LaunchConfiguration("drop_old_frames"),
                    "localization_mode": node_localization_mode,
                    "use_latest_tf": use_latest_tf,
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
    )

    downstream_actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", default_rviz_config],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        )
    )

    downstream_actions.append(
        Node(
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
    )

    sim_group = GroupAction(downstream_actions)

    return [tum_player_node, sim_group]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "localization_mode",
                default_value="slam",
                description=(
                    "Localization mode: 'slam', 'ground_truth', or 'camera_local'"
                ),
            ),
            DeclareLaunchArgument(
                "use_rtabmap",
                default_value="false",
                description="[Compatibility] Enable SLAM when no localization_mode is supplied",
            ),
            DeclareLaunchArgument(
                "publish_groundtruth_tf",
                default_value="false",
                description=(
                    "[Compatibility] Enable ground-truth TF when no "
                    "localization_mode is supplied"
                ),
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value="configs/tum_fr1_desk.yaml",
                description="Path to SceneGraph configuration YAML",
            ),
            DeclareLaunchArgument(
                "dataset_root",
                default_value="data/raw/rgbd_dataset_freiburg1_desk",
                description="Path to TUM RGB-D sequence directory",
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
                "debug_frame_packet_only",
                default_value="false",
                description="Run FramePacket diagnostic isolation mode",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Launch RViz2",
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
                "use_rtabmap_viz",
                default_value="true",
                description="Launch native RTAB-Map 3D visualization GUI (Window 4)",
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
                description=(
                    "Raw depth units per metre: 5000.0 for TUM, "
                    "1000.0 for D455 recordings"
                ),
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
                "publish_rate_hz",
                default_value="2.0",
                description="Replay input rate in Hz",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="false",
                description="Loop playback; keep false for SLAM",
            ),
            DeclareLaunchArgument(
                "map_voxel_size_m",
                default_value="0.008",
                description="Voxel size for the growing map cloud (0.008 = 8mm for high density)",
            ),
            DeclareLaunchArgument(
                "map_max_points",
                default_value="2000000",
                description="Maximum accumulated map points (2 million)",
            ),
            DeclareLaunchArgument(
                "map_cloud_stride",
                default_value="2",
                description="Pixel stride for map cloud generation (2 = 4x denser)",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )