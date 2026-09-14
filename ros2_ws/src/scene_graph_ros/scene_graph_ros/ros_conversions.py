"""ROS 2 message conversion utilities for Dynamic 3D Scene Graph."""

from typing import Optional, Union
import numpy as np

try:
    from sensor_msgs.msg import Image, CameraInfo, Imu, PointCloud2, PointField
    from geometry_msgs.msg import TransformStamped, Transform
    from builtin_interfaces.msg import Time
    HAS_ROS2_MSGS = True
except ImportError:
    HAS_ROS2_MSGS = False

    class PointField:  # type: ignore
        FLOAT32 = 7
        UINT32 = 6
        def __init__(self, name="", offset=0, datatype=0, count=1):
            self.name = name
            self.offset = offset
            self.datatype = datatype
            self.count = count

    class Header:  # type: ignore
        def __init__(self, stamp=None, frame_id=""):
            self.stamp = stamp
            self.frame_id = frame_id

    class Time:  # type: ignore
        def __init__(self, sec=0, nanosec=0):
            self.sec = sec
            self.nanosec = nanosec

    class PointCloud2:  # type: ignore
        def __init__(self):
            self.header = Header()
            self.height = 0
            self.width = 0
            self.fields = []
            self.is_bigendian = False
            self.point_step = 0
            self.row_step = 0
            self.data = b""
            self.is_dense = True

    class Image:  # type: ignore
        def __init__(self):
            self.header = Header()
            self.height = 0
            self.width = 0
            self.encoding = ""
            self.is_bigendian = 0
            self.step = 0
            self.data = b""

try:
    from cv_bridge import CvBridge
    _CV_BRIDGE = CvBridge()
except ImportError:
    _CV_BRIDGE = None

from scene_graph.data.frame_packet import IMUSample
from scene_graph.geometry.camera import CameraIntrinsics


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) into a 3x3 orthonormal rotation matrix."""
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero quaternion")
    
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)

    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)

    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    return np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22],
    ], dtype=np.float64)


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 orthonormal rotation matrix into a unit quaternion (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return float(qx), float(qy), float(qz), float(qw)


def ros_image_to_numpy(msg) -> np.ndarray:
    """Convert sensor_msgs/Image to a numpy array."""
    encoding = msg.encoding.lower()

    if _CV_BRIDGE is not None:
        try:
            if encoding in ("rgb8", "bgr8", "8uc3"):
                return _CV_BRIDGE.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            elif encoding in ("16uc1", "mono16"):
                return _CV_BRIDGE.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            elif encoding in ("32fc1",):
                return _CV_BRIDGE.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception:
            pass

    raw_data = bytes(msg.data)
    h, w = msg.height, msg.width
    step = getattr(msg, "step", None)

    if encoding in ("rgb8", "bgr8"):
        expected_row = w * 3
        if step is not None and step > expected_row:
            arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((h, step))[:, :expected_row].reshape((h, w, 3))
        else:
            arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((h, w, 3))
        if encoding == "bgr8":
            return arr[:, :, ::-1].copy()
        return arr.copy()
    elif encoding in ("16uc1", "mono16"):
        expected_row = w * 2
        if step is not None and step > expected_row:
            arr = np.frombuffer(raw_data, dtype=np.uint16).reshape((h, step // 2))[:, :w]
        else:
            arr = np.frombuffer(raw_data, dtype=np.uint16).reshape((h, w))
        return arr.copy()
    elif encoding == "32fc1":
        expected_row = w * 4
        if step is not None and step > expected_row:
            arr = np.frombuffer(raw_data, dtype=np.float32).reshape((h, step // 4))[:, :w]
        else:
            arr = np.frombuffer(raw_data, dtype=np.float32).reshape((h, w))
        return arr.copy()
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def numpy_to_ros_image(
    arr: np.ndarray,
    encoding: str,
    frame_id: str,
    timestamp: float,
) -> "Image":
    msg = Image()
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1e9)
    msg.header.stamp = Time(sec=sec, nanosec=nanosec)
    msg.header.frame_id = frame_id
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0

    if arr.ndim == 2:
        channels = 1
        itemsize = arr.itemsize
    else:
        channels = arr.shape[2]
        itemsize = arr.itemsize

    msg.step = msg.width * channels * itemsize
    msg.data = arr.tobytes()
    return msg


def camera_info_to_intrinsics(msg: "CameraInfo") -> CameraIntrinsics:
    """Extract CameraIntrinsics from sensor_msgs/CameraInfo."""
    width = int(msg.width)
    height = int(msg.height)

    if len(msg.k) == 9 and msg.k[0] > 0.0:
        fx = float(msg.k[0])
        cx = float(msg.k[2])
        fy = float(msg.k[4])
        cy = float(msg.k[5])
    elif len(msg.p) == 12 and msg.p[0] > 0.0:
        fx = float(msg.p[0])
        cx = float(msg.p[2])
        fy = float(msg.p[5])
        cy = float(msg.p[6])
    else:
        raise ValueError("CameraInfo contains invalid zero focal length in both K and P")

    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
    )


def intrinsics_to_camera_info(
    intrinsics: CameraIntrinsics,
    frame_id: str,
    timestamp: float,
) -> "CameraInfo":
    """Construct sensor_msgs/CameraInfo from CameraIntrinsics."""
    if not HAS_ROS2_MSGS:
        raise RuntimeError("sensor_msgs is not available in this environment")

    msg = CameraInfo()
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1e9)
    msg.header.stamp = Time(sec=sec, nanosec=nanosec)
    msg.header.frame_id = frame_id
    msg.width = intrinsics.width
    msg.height = intrinsics.height
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

    msg.k = [
        intrinsics.fx, 0.0, intrinsics.cx,
        0.0, intrinsics.fy, intrinsics.cy,
        0.0, 0.0, 1.0,
    ]

    msg.r = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]

    msg.p = [
        intrinsics.fx, 0.0, intrinsics.cx, 0.0,
        0.0, intrinsics.fy, intrinsics.cy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]

    return msg


def transform_to_matrix(msg: Union["TransformStamped", "Transform"]) -> np.ndarray:
    """Convert a TransformStamped or Transform message into a 4x4 float64 transform matrix."""
    if hasattr(msg, "transform"):
        transform = msg.transform
    else:
        transform = msg

    tx = float(transform.translation.x)
    ty = float(transform.translation.y)
    tz = float(transform.translation.z)

    qx = float(transform.rotation.x)
    qy = float(transform.rotation.y)
    qz = float(transform.rotation.z)
    qw = float(transform.rotation.w)

    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def matrix_to_transform_stamped(
    T: np.ndarray,
    parent_frame: str,
    child_frame: str,
    timestamp: float,
) -> "TransformStamped":
    """Convert a 4x4 matrix into geometry_msgs/TransformStamped."""
    if not HAS_ROS2_MSGS:
        raise RuntimeError("geometry_msgs is not available in this environment")

    msg = TransformStamped()
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1e9)
    msg.header.stamp = Time(sec=sec, nanosec=nanosec)
    msg.header.frame_id = parent_frame
    msg.child_frame_id = child_frame

    msg.transform.translation.x = float(T[0, 3])
    msg.transform.translation.y = float(T[1, 3])
    msg.transform.translation.z = float(T[2, 3])

    qx, qy, qz, qw = rotation_matrix_to_quaternion(T[:3, :3])
    msg.transform.rotation.x = qx
    msg.transform.rotation.y = qy
    msg.transform.rotation.z = qz
    msg.transform.rotation.w = qw

    return msg


def imu_msg_to_sample(msg: "Imu") -> IMUSample:
    """Convert sensor_msgs/Imu into an immutable IMUSample."""
    timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
    accel = np.array([
        msg.linear_acceleration.x,
        msg.linear_acceleration.y,
        msg.linear_acceleration.z,
    ], dtype=np.float64)

    gyro = np.array([
        msg.angular_velocity.x,
        msg.angular_velocity.y,
        msg.angular_velocity.z,
    ], dtype=np.float64)

    return IMUSample(
        timestamp=timestamp,
        accel=accel,
        gyro=gyro,
    )


def numpy_to_point_cloud2(
    points: np.ndarray,
    frame_id: str,
    timestamp: float,
    colors: Optional[np.ndarray] = None,
) -> "PointCloud2":
    msg = PointCloud2()
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1e9)
    msg.header.stamp = Time(sec=sec, nanosec=nanosec)
    msg.header.frame_id = frame_id

    n_points = int(points.shape[0]) if points is not None else 0
    msg.height = 1
    msg.width = n_points
    msg.is_bigendian = False
    msg.is_dense = True

    if n_points == 0:
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = 0
        msg.data = b""
        return msg

    points_f32 = np.ascontiguousarray(points[:, :3], dtype=np.float32)

    if colors is not None and colors.shape[0] == n_points:
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = 16 * n_points

        cloud_data = np.empty(n_points, dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgb", np.uint32),
        ])
        cloud_data["x"] = points_f32[:, 0]
        cloud_data["y"] = points_f32[:, 1]
        cloud_data["z"] = points_f32[:, 2]

        c_u32 = colors.astype(np.uint32)
        r = c_u32[:, 0]
        g = c_u32[:, 1]
        b = c_u32[:, 2]
        cloud_data["rgb"] = (r << 16) | (g << 8) | b
        msg.data = cloud_data.tobytes()
    else:
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = 12 * n_points
        msg.data = points_f32.tobytes()

    return msg

