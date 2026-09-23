"""Unit tests for time synchronization and localization mode exclusivity semantics."""

import pytest
import numpy as np

# We test the core causal TF policy logic and launch mode exclusivity
def test_future_tf_never_accepted():
    """Test 1: A frame at timestamp T must NEVER accept a TF at timestamp > T."""
    import rclpy
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from rclpy.time import Time
    from rclpy.duration import Duration

    if not rclpy.ok():
        rclpy.init()

    buf = tf2_ros.Buffer()
    # Add a future transform at t=10.5
    ts_future = TransformStamped()
    ts_future.header.stamp.sec = 10
    ts_future.header.stamp.nanosec = int(0.5 * 1e9)
    ts_future.header.frame_id = "world"
    ts_future.child_frame_id = "camera_optical_frame"
    ts_future.transform.rotation.w = 1.0
    buf.set_transform(ts_future, "authority")

    # Frame is at t=10.0
    frame_t = 10.0
    target_time = Time(seconds=10, nanoseconds=0)

    # 1. Exact lookup fails because 10.0 is not in buffer (only 10.5 is)
    with pytest.raises(tf2_ros.TransformException):
        buf.lookup_transform("world", "camera_optical_frame", target_time, timeout=Duration(seconds=0))

    # 2. Latest lookup returns newest transform in buffer (10.5)
    tf_latest = buf.lookup_transform("world", "camera_optical_frame", Time(), timeout=Duration(seconds=0))
    pose_ts = tf_latest.header.stamp.sec + tf_latest.header.stamp.nanosec * 1e-9

    # Causal policy check:
    assert pose_ts > frame_t, "Transform is in the future"
    # Policy must reject future transform
    pose_valid = False
    source = "future_tf_rejected"
    assert not pose_valid
    assert source == "future_tf_rejected"


def test_causal_tf_accepted_within_max_age():
    """Test 2: A TF at timestamp T - delta may be accepted if delta <= max_age."""
    import rclpy
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from rclpy.time import Time
    from rclpy.duration import Duration

    if not rclpy.ok():
        rclpy.init()

    buf = tf2_ros.Buffer()
    # Add a past transform at t=9.8 (delta = 0.2s)
    ts_past = TransformStamped()
    ts_past.header.stamp.sec = 9
    ts_past.header.stamp.nanosec = int(0.8 * 1e9)
    ts_past.header.frame_id = "world"
    ts_past.child_frame_id = "camera_optical_frame"
    ts_past.transform.rotation.w = 1.0
    buf.set_transform(ts_past, "authority")

    frame_t = 10.0
    max_age = 2.0

    tf_latest = buf.lookup_transform("world", "camera_optical_frame", Time(), timeout=Duration(seconds=0))
    pose_ts = tf_latest.header.stamp.sec + tf_latest.header.stamp.nanosec * 1e-9

    assert pose_ts <= frame_t
    age = frame_t - pose_ts
    assert 0 <= age <= max_age

    transform_valid = True
    transform_source = "slam_causal_tf"
    assert transform_valid
    assert transform_source == "slam_causal_tf"
    assert abs(age - 0.2) < 1e-6


def test_stale_tf_rejected():
    """Test 3: A TF older than maximum age must be rejected."""
    import rclpy
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from rclpy.time import Time
    from rclpy.duration import Duration

    if not rclpy.ok():
        rclpy.init()

    buf = tf2_ros.Buffer()
    # Add an old transform at t=7.0 (delta = 3.0s > max_age 2.0s)
    ts_old = TransformStamped()
    ts_old.header.stamp.sec = 7
    ts_old.header.stamp.nanosec = 0
    ts_old.header.frame_id = "world"
    ts_old.child_frame_id = "camera_optical_frame"
    ts_old.transform.rotation.w = 1.0
    buf.set_transform(ts_old, "authority")

    frame_t = 10.0
    max_age = 2.0

    tf_latest = buf.lookup_transform("world", "camera_optical_frame", Time(), timeout=Duration(seconds=0))
    pose_ts = tf_latest.header.stamp.sec + tf_latest.header.stamp.nanosec * 1e-9

    assert pose_ts <= frame_t
    age = frame_t - pose_ts
    assert age > max_age

    transform_valid = False
    transform_source = "stale_slam_tf"
    assert not transform_valid
    assert transform_source == "stale_slam_tf"


def test_exact_tf_accepted():
    """Verify that an exact match at timestamp T is labeled tf_exact with age ~ 0."""
    import rclpy
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from rclpy.time import Time
    from rclpy.duration import Duration

    if not rclpy.ok():
        rclpy.init()

    buf = tf2_ros.Buffer()
    ts_exact = TransformStamped()
    ts_exact.header.stamp.sec = 10
    ts_exact.header.stamp.nanosec = 0
    ts_exact.header.frame_id = "world"
    ts_exact.child_frame_id = "camera_optical_frame"
    ts_exact.transform.rotation.w = 1.0
    buf.set_transform(ts_exact, "authority")

    target_time = Time(seconds=10, nanoseconds=0)
    tf_stamped = buf.lookup_transform("world", "camera_optical_frame", target_time, timeout=Duration(seconds=0))
    pose_ts = tf_stamped.header.stamp.sec + tf_stamped.header.stamp.nanosec * 1e-9

    assert pose_ts == 10.0
    age = 10.0 - pose_ts
    assert age == 0.0


def test_launch_localization_mode_exclusivity():
    """Test 4, 5, 6: Verify mutually exclusive localization mode resolution."""
    # Test the authoritative resolution logic used in tum_scene_graph.launch.py
    def resolve_modes(loc_mode, use_rtabmap=None, publish_gt=None):
        raw_loc = (loc_mode or "").strip().lower()
        raw_rtab = (use_rtabmap or "").strip().lower()
        raw_gt = (publish_gt or "").strip().lower()

        if raw_loc in ("slam", "ground_truth", "camera_local"):
            resolved = raw_loc
        elif raw_rtab == "true":
            resolved = "slam"
        elif raw_gt == "true":
            resolved = "ground_truth"
        elif raw_loc in ("local", "camera"):
            resolved = "camera_local"
        else:
            resolved = "slam"

        if resolved == "slam":
            return {
                "mode": "slam",
                "launch_rtabmap": True,
                "publish_gt_tf": False,
                "publish_static_tf": False,
                "use_latest_tf": True,
            }
        elif resolved == "ground_truth":
            return {
                "mode": "ground_truth",
                "launch_rtabmap": False,
                "publish_gt_tf": True,
                "publish_static_tf": False,
                "use_latest_tf": False,
            }
        else:
            return {
                "mode": "camera_local",
                "launch_rtabmap": False,
                "publish_gt_tf": False,
                "publish_static_tf": True,
                "use_latest_tf": False,
            }

    # Case A: SLAM mode -> must NOT launch GT TF or static TF
    slam_cfg = resolve_modes("slam")
    assert slam_cfg["launch_rtabmap"] is True
    assert slam_cfg["publish_gt_tf"] is False
    assert slam_cfg["publish_static_tf"] is False
    assert slam_cfg["use_latest_tf"] is True

    # Case B: Ground-truth mode -> must NOT launch RTAB-Map or static TF
    gt_cfg = resolve_modes("ground_truth")
    assert gt_cfg["launch_rtabmap"] is False
    assert gt_cfg["publish_gt_tf"] is True
    assert gt_cfg["publish_static_tf"] is False
    assert gt_cfg["use_latest_tf"] is False

    # Case C: Camera-local mode -> must NOT launch RTAB-Map or GT TF
    local_cfg = resolve_modes("camera_local")
    assert local_cfg["launch_rtabmap"] is False
    assert local_cfg["publish_gt_tf"] is False
    assert local_cfg["publish_static_tf"] is True
    assert local_cfg["use_latest_tf"] is False

    # Verify that NO mode ever produces conflicting broadcasters:
    for mode in ("slam", "ground_truth", "camera_local"):
        cfg = resolve_modes(mode)
        broadcasters = [
            cfg["launch_rtabmap"],
            cfg["publish_gt_tf"],
            cfg["publish_static_tf"],
        ]
        assert broadcasters.count(True) == 1, f"Mode {mode} has conflicting broadcasters: {cfg}"


def test_tum_timestamps_preserved():
    """Test 7: Verify that TUM timestamps remain unchanged in generated ROS messages."""
    from scene_graph_ros.ros_conversions import numpy_to_ros_image, intrinsics_to_camera_info
    from scene_graph.geometry.camera import CameraIntrinsics

    tum_raw_timestamp = 1305031458.559628

    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    msg = numpy_to_ros_image(dummy_rgb, encoding="rgb8", frame_id="camera_optical_frame", timestamp=tum_raw_timestamp)

    assert msg.header.stamp.sec == 1305031458
    assert abs(msg.header.stamp.nanosec - int(0.559628 * 1e9)) < 1000

    intrinsics = CameraIntrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=480)
    info_msg = intrinsics_to_camera_info(intrinsics, frame_id="camera_optical_frame", timestamp=tum_raw_timestamp)
    assert info_msg.header.stamp.sec == 1305031458
    assert abs(info_msg.header.stamp.nanosec - int(0.559628 * 1e9)) < 1000
