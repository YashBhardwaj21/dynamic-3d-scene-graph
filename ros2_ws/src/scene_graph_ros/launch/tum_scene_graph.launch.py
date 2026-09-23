"""Launch file for TUM dataset replay with optional RTAB-Map SLAM or ground-truth TF.

Clock architecture
------------------
tum_player publishes /clock from the RGB header stamp.
RTAB-Map, scene_graph_node, and RViz all run with use_sim_time=True via SetParameter,
so they live entirely in the TUM timestamp domain (~1305031xxx seconds).
tum_player does NOT use use_sim_time — its wall-clock timer drives playback.

Modes
-----
1. RTAB-Map SLAM  (use_rtabmap:=true  publish_groundtruth_tf:=false)
   TUM RGB-D + /clock → RTAB-Map → estimated TF → SceneGraph → RViz

2. Ground-truth TF  (use_rtabmap:=false  publish_groundtruth_tf:=true)
   TUM RGB-D + /clock + GT TF → SceneGraph → RViz
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    pkg_share = get_package_share_directory("scene_graph_ros")
    default_rviz_config = os.path.join(pkg_share, "rviz", "scene_graph.rviz")

    # -------------------------------------------------------------------------
    # Launch arguments
    # -------------------------------------------------------------------------

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
        description="Playback rate multiplier",
    )

    start_frame_arg = DeclareLaunchArgument(
        "start_frame",
        default_value="0",
        description="Starting frame index",
    )

    end_frame_arg = DeclareLaunchArgument(
        "end_frame",
        default_value="-1",
        description="Ending frame index (-1 = end of sequence)",
    )

    frame_stride_arg = DeclareLaunchArgument(
        "frame_stride",
        default_value="1",
        description="Step between published frames (1=every frame)",
    )

    debug_frame_packet_only_arg = DeclareLaunchArgument(
        "debug_frame_packet_only",
        default_value="false",
        description="Run in FramePacket diagnostic isolation mode",
    )

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Whether to launch RViz2",
    )

    use_viewer_arg = DeclareLaunchArgument(
        "use_viewer",
        default_value="true",
        description="Whether to launch 2D live perception viewer",
    )

    use_rtabmap_arg = DeclareLaunchArgument(
        "use_rtabmap",
        default_value="false",
        description="Launch RTAB-Map SLAM. When true, set publish_groundtruth_tf:=false.",
    )

    publish_groundtruth_tf_arg = DeclareLaunchArgument(
        "publish_groundtruth_tf",
        default_value="false",
        description=(
            "Broadcast ground-truth world→sensor TF from groundtruth.txt. "
            "Set to false when use_rtabmap:=true."
        ),
    )

    localization_mode_arg = DeclareLaunchArgument(
        "localization_mode",
        default_value="world",
        description="'world' = TF-based | 'camera_local' = identity (debug only)",
    )

    world_frame_arg = DeclareLaunchArgument(
        "world_frame",
        default_value="world",
        description="World reference frame ID",
    )

    sensor_frame_arg = DeclareLaunchArgument(
        "sensor_frame",
        default_value="camera_optical_frame",
        description="Camera optical frame ID",
    )

    depth_scale_arg = DeclareLaunchArgument(
        "depth_scale",
        default_value="5000.0",
        description=(
            "Depth scale (raw units per metre). "
            "5000.0 for TUM standard; 1000.0 for my_desk_sequence (D455)."
        ),
    )

    queue_size_arg = DeclareLaunchArgument(
        "queue_size",
        default_value="5",
        description="Max frames buffered in worker queue (keep small for SLAM to avoid latency)",
    )

    drop_old_frames_arg = DeclareLaunchArgument(
        "drop_old_frames",
        default_value="true",
        description="Drop incoming frames when the queue is full (true = avoid latency buildup)",
    )

    # 30 Hz: fast enough for RTAB-Map to build a good map; tum_player timer runs on wall clock
    publish_rate_hz_arg = DeclareLaunchArgument(
        "publish_rate_hz",
        default_value="30.0",
        description="Frame publishing rate in Hz",
    )

    # Do NOT loop: /clock would jump backwards, breaking TF ordering
    loop_arg = DeclareLaunchArgument(
        "loop",
        default_value="false",
        description="Whether to loop the sequence. Must be false with SLAM (clock would rewind).",
    )

    map_voxel_size_m_arg = DeclareLaunchArgument(
        "map_voxel_size_m",
        default_value="0.02",
        description="Voxel size (m) for growing map cloud",
    )

    map_max_points_arg = DeclareLaunchArgument(
        "map_max_points",
        default_value="500000",
        description="Maximum points in map before voxel downsampling",
    )

    map_cloud_stride_arg = DeclareLaunchArgument(
        "map_cloud_stride",
        default_value="4",
        description="Subsampling stride for map cloud unprojection",
    )

    # -------------------------------------------------------------------------
    # TUM Player — does NOT use use_sim_time.
    # Its wall-clock timer drives playback and publishes /clock for downstream nodes.
    # -------------------------------------------------------------------------
    tum_player_node = Node(
        package="scene_graph_ros",
        executable="tum_player",
        name="tum_player",
        output="screen",
        # NOTE: no use_sim_time here — tum_player must run on wall clock
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
                "publish_groundtruth_tf": LaunchConfiguration("publish_groundtruth_tf"),
                "depth_scale": LaunchConfiguration("depth_scale"),
                "slam_depth_topic": "/tum/depth_m/image_raw",
            }
        ],
    )

    # -------------------------------------------------------------------------
    # All SLAM / visualization nodes run with use_sim_time=True via SetParameter.
    # SetParameter applies to every Node and IncludeLaunchDescription in the group.
    # This puts RTAB-Map, scene_graph_node, and RViz into the TUM time domain.
    # -------------------------------------------------------------------------
    slam_nodes = GroupAction([
        # Apply use_sim_time=True to ALL nodes within this group
        SetParameter(name="use_sim_time", value=True),

        # RTAB-Map SLAM — only launched when use_rtabmap:=true.
        # Uses 32FC1 float metres depth so no depth_scale confusion.
        # use_sim_time propagated via SetParameter above.
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
                "approx_sync": "true",
                "wait_imu_to_init": "false",
                "rtabmap_viz": "false",
                "rviz": "false",
            }.items(),
            condition=IfCondition(LaunchConfiguration("use_rtabmap")),
        ),

        # Static TF for camera-local/debug mode (no SLAM, no GT TF).
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
            condition=UnlessCondition(LaunchConfiguration("use_rtabmap")),
        ),

        # SceneGraph node — uses exact-timestamp TF lookup.
        # With use_sim_time=True and /clock from tum_player, its TF lookups are
        # at TUM timestamps and RTAB-Map's TF is also at TUM timestamps → match.
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
                    "localization_mode": LaunchConfiguration("localization_mode"),
                    "rgb_topic": "/tum/rgb/image_raw",
                    "depth_topic": "/tum/depth/image_raw",
                    "camera_info_topic": "/tum/rgb/camera_info",
                    "map_voxel_size_m": LaunchConfiguration("map_voxel_size_m"),
                    "map_max_points": LaunchConfiguration("map_max_points"),
                    "map_cloud_stride": LaunchConfiguration("map_cloud_stride"),
                    # RTAB-Map publishes TF *after* processing the frame, so the exact
                    # image timestamp is never in the TF buffer when scene_graph_node asks.
                    # use_latest_tf=True → lookup_transform(Time(0)) = latest available pose.
                    # slam_pose_max_age rejects transforms older than 2 s (pre-init frames).
                    "use_latest_tf": True,
                    "slam_pose_max_age": 2.0,
                }
            ],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", default_rviz_config],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),

        Node(
            package="scene_graph_ros",
            executable="live_2d_viewer",
            name="live_2d_viewer",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_viewer")),
        ),
    ])

    return LaunchDescription(
        [
            # Args
            config_path_arg,
            dataset_root_arg,
            rate_multiplier_arg,
            start_frame_arg,
            end_frame_arg,
            frame_stride_arg,
            debug_frame_packet_only_arg,
            use_rviz_arg,
            use_viewer_arg,
            use_rtabmap_arg,
            publish_groundtruth_tf_arg,
            localization_mode_arg,
            world_frame_arg,
            sensor_frame_arg,
            depth_scale_arg,
            queue_size_arg,
            drop_old_frames_arg,
            publish_rate_hz_arg,
            loop_arg,
            map_voxel_size_m_arg,
            map_max_points_arg,
            map_cloud_stride_arg,
            # tum_player: NO use_sim_time — wall clock timer drives playback
            tum_player_node,
            # All SLAM/viz nodes: use_sim_time=True via SetParameter
            slam_nodes,
        ]
    )
