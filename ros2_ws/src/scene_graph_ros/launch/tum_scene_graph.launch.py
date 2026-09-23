"""Launch file for TUM dataset replay with mutually exclusive localization modes.

Clock architecture
------------------
tum_player publishes /clock from the RGB header stamp.
RTAB-Map, scene_graph_node, and RViz all run with use_sim_time=True via SetParameter,
so they live entirely in the TUM timestamp domain (~1305031xxx seconds).
tum_player does NOT use use_sim_time — its wall-clock timer drives playback.

Authoritative Localization Modes
--------------------------------
1. 'slam' (default):
   TUM RGB-D + /clock → RTAB-Map SLAM → TF poses → SceneGraph → RViz
   - RTAB-Map is launched.
   - Ground-truth TF is NOT published.
   - Static world->camera TF is NOT published.
   - SceneGraph uses causal SLAM TF lookup (exact if available, causal past TF if processing lag).

2. 'ground_truth':
   TUM RGB-D + /clock + GT TF (from groundtruth.txt) → SceneGraph → RViz
   - tum_player publishes ground-truth TF.
   - RTAB-Map is NOT launched.
   - Static world->camera TF is NOT published.
   - SceneGraph uses exact-timestamp TF lookup.

3. 'camera_local':
   TUM RGB-D + /clock + static identity TF → SceneGraph (camera-local frame) → RViz
   - RTAB-Map is NOT launched.
   - Ground-truth TF is NOT published.
   - Static identity TF is published for RViz frame resolution.
   - SceneGraph processes objects in camera-local coordinate frame.
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

    # Resolve authoritative localization mode with backward compatibility
    raw_loc_mode = context.launch_configurations.get("localization_mode", "").strip().lower()
    raw_use_rtabmap = context.launch_configurations.get("use_rtabmap", "").strip().lower()
    raw_publish_gt = context.launch_configurations.get("publish_groundtruth_tf", "").strip().lower()

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

    # Derive mutually exclusive component flags
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
    else:  # camera_local
        launch_rtabmap = False
        publish_gt_tf = False
        publish_static_tf = True
        node_localization_mode = "camera_local"
        use_latest_tf = False

    # -------------------------------------------------------------------------
    # TUM Player — does NOT use use_sim_time.
    # Its wall-clock timer drives playback and publishes /clock for downstream nodes.
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # All downstream nodes run with use_sim_time=True via SetParameter.
    # -------------------------------------------------------------------------
    downstream_actions = [
        SetParameter(name="use_sim_time", value=True),
    ]

    # Mutually exclusive TF sources:
    if launch_rtabmap:
        downstream_actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        get_package_share_directory("rtabmap_launch"),
                        "launch",
                        "rtabmap.launch.py",
                    ])
                ),
                launch_arguments={
                    "rgb_topic": "/tum/rgb/image_raw",
                    "depth_topic": "/tum/depth_m/image_raw",
                    "camera_info_topic": "/tum/rgb/camera_info",
                    "frame_id": LaunchConfiguration("sensor_frame"),
                    "map_frame_id": LaunchConfiguration("world_frame"),
                    "approx_sync": "false",
                    "wait_imu_to_init": "false",
                    "use_sim_time": "true",
                    "rtabmap_viz": "false",
                    "rviz": "false",
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
                    "--x", "0", "--y", "0", "--z", "0",
                    "--roll", "0", "--pitch", "0", "--yaw", "0",
                    "--frame-id", LaunchConfiguration("world_frame"),
                    "--child-frame-id", LaunchConfiguration("sensor_frame"),
                ],
            )
        )

    # SceneGraph perception node
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
                    "debug_frame_packet_only": LaunchConfiguration("debug_frame_packet_only"),
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
            condition=IfCondition(LaunchConfiguration("use_viewer")),
        )
    )

    sim_group = GroupAction(downstream_actions)

    return [tum_player_node, sim_group]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "localization_mode",
            default_value="slam",
            description="Authoritative localization mode: 'slam' (RTAB-Map), 'ground_truth' (tum_player GT TF), or 'camera_local' (identity)",
        ),
        DeclareLaunchArgument(
            "use_rtabmap",
            default_value="false",
            description="[Compatibility] True sets localization_mode to slam if localization_mode is not explicitly passed",
        ),
        DeclareLaunchArgument(
            "publish_groundtruth_tf",
            default_value="false",
            description="[Compatibility] True sets localization_mode to ground_truth if localization_mode is not explicitly passed",
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
            description="Step between published frames (1=every frame)",
        ),
        DeclareLaunchArgument(
            "debug_frame_packet_only",
            default_value="false",
            description="Run in FramePacket diagnostic isolation mode",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Whether to launch RViz2",
        ),
        DeclareLaunchArgument(
            "use_viewer",
            default_value="true",
            description="Whether to launch 2D live perception viewer",
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
            description="Depth scale (raw units per metre). 5000.0 for TUM standard; 1000.0 for my_desk_sequence (D455).",
        ),
        DeclareLaunchArgument(
            "queue_size",
            default_value="5",
            description="Max frames buffered in worker queue (keep small for SLAM to avoid latency)",
        ),
        DeclareLaunchArgument(
            "drop_old_frames",
            default_value="true",
            description="Drop incoming frames when the queue is full (true = avoid latency buildup)",
        ),
        DeclareLaunchArgument(
            "publish_rate_hz",
            default_value="30.0",
            description="Frame publishing rate in Hz",
        ),
        DeclareLaunchArgument(
            "loop",
            default_value="false",
            description="Whether to loop the sequence. Must be false with SLAM (clock would rewind).",
        ),
        DeclareLaunchArgument(
            "map_voxel_size_m",
            default_value="0.02",
            description="Voxel size (m) for growing map cloud",
        ),
        DeclareLaunchArgument(
            "map_max_points",
            default_value="500000",
            description="Maximum points in map before voxel downsampling",
        ),
        DeclareLaunchArgument(
            "map_cloud_stride",
            default_value="4",
            description="Subsampling stride for map cloud unprojection",
        ),
        OpaqueFunction(function=launch_setup),
    ])
