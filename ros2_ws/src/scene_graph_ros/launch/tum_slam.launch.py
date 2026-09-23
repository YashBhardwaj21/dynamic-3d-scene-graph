"""Launch TUM RGB-D Replay with RTAB-Map SLAM and High-Density 3D Point Cloud.

Dedicated to:
- tum_player (replaying RGB, depth, camera_info, and /clock)
- RTAB-Map visual odometry + dense SLAM + TF
- RViz2 (with dense growing map cloud and camera trajectory)
- rtabmap_viz (native RTAB-Map 3D visualization)
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
                "publish_groundtruth_tf": False,
                "depth_scale": LaunchConfiguration("depth_scale"),
                "slam_depth_topic": "/tum/depth_m/image_raw",
                "min_subscribers": 1,
            }
        ],
    )

    downstream_actions = [
        SetParameter(name="use_sim_time", value=True),
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
                "approx_sync": "true",
                "approx_sync_max_interval": "0.05",
                "topic_queue_size": "5",
                "sync_queue_size": "10",
                "odom_always_process_most_recent_frame": "false",
                "depth_scale": "1.0",
                "wait_imu_to_init": "false",
                "use_sim_time": "true",
                "rtabmap_viz": LaunchConfiguration("use_rtabmap_viz"),
                "rviz": "false",
                # High-density point cloud SLAM parameters:
                # - Grid/CellSize 0.01: 1cm grid cells (default is 5cm)
                # - Grid/VoxelSize 0.005: 5mm voxel downsampling for dense features
                # - Grid/RangeMax 4.0: preserve depth points up to 4 metres
                # - RGBD updates on small motion: 1cm translation, 10mrad rotation
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
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", default_rviz_config],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),
    ]

    sim_group = GroupAction(downstream_actions)
    return [tum_player_node, sim_group]


def generate_launch_description():
    pkg_share = get_package_share_directory("scene_graph_ros")
    default_config = os.path.join(pkg_share, "configs", "tum_fr1_desk.yaml")
    if not os.path.exists(default_config):
        default_config = "configs/tum_fr1_desk.yaml"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value=default_config,
                description="Path to SceneGraph configuration YAML",
            ),
            DeclareLaunchArgument(
                "dataset_root",
                default_value="data/raw/rgbd_dataset_freiburg1_desk",
                description="Path to TUM RGB-D sequence directory",
            ),
            DeclareLaunchArgument(
                "publish_rate_hz",
                default_value="3.0",
                description="Replay input rate in Hz",
            ),
            DeclareLaunchArgument(
                "rate_multiplier",
                default_value="1.0",
                description="Playback rate multiplier",
            ),
            DeclareLaunchArgument(
                "start_frame",
                default_value="50",
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
                "use_rviz",
                default_value="true",
                description="Launch RViz2",
            ),
            DeclareLaunchArgument(
                "use_rtabmap_viz",
                default_value="true",
                description="Launch native RTAB-Map 3D visualization GUI",
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
                "loop",
                default_value="true",
                description="Loop playback",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
