from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.reference_frame import RelationReferenceFrame


@dataclass(frozen=True)
class IMUSample:
    """Inertial Measurement Unit sample representing linear acceleration and angular velocity."""
    timestamp: float
    accel: np.ndarray  # (3,) m/s^2
    gyro: np.ndarray   # (3,) rad/s


@dataclass
class FramePacket:
    """A synchronized, per-frame container for downstream scene graph tasks.
    
    Invariant:
    - depth is ALWAYS metric float32 depth in meters (or None).
    - Raw integer sensor depth is converted by the data source adapter before reaching FramePacket.
    - Core geometry never performs raw depth scale conversion.
    """
    frame_index: int            
    timestamp: float            
    rgb: np.ndarray             # HxWx3 uint8 RGB format
    depth: Optional[np.ndarray] # HxW metric float32 depth in meters
    world_T_camera: Optional[np.ndarray]  # (4, 4) pose matrix
    camera_intrinsics: CameraIntrinsics
    depth_model: Optional[DepthModel] = None  # Deprecated: adapters own scale
    relation_frame: Optional[RelationReferenceFrame] = None
    imu_samples: tuple[IMUSample, ...] = ()  # IMU samples in (t_{k-1}, t_k]
    frame_id: str = "camera_color_optical_frame"
    optical_frame_id: Optional[str] = "camera_depth_optical_frame"
    pose_source: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not np.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(f"FramePacket timestamp must be finite and non-negative, got {self.timestamp}")

        if self.rgb is not None and isinstance(self.rgb, np.ndarray):
            if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
                raise ValueError(f"FramePacket.rgb must be (H, W, 3), got shape {self.rgb.shape}")
            if self.rgb.dtype != np.uint8:
                raise TypeError(f"FramePacket.rgb must have uint8 dtype, got {self.rgb.dtype}")

        if self.depth is not None:
            if isinstance(self.depth, np.ndarray):
                if np.issubdtype(self.depth.dtype, np.integer):
                    raise ValueError(
                        f"FramePacket.depth must be metric depth in meters (float32), got integer dtype {self.depth.dtype}. "
                        "Raw depth conversion belongs in the sensor adapter, not inside the core pipeline."
                    )
                if self.depth.ndim != 2:
                    raise ValueError(f"FramePacket.depth must be (H, W) 2D array, got shape {self.depth.shape}")
                if self.depth.dtype != np.float32:
                    self.depth = self.depth.astype(np.float32)

        if self.world_T_camera is not None and isinstance(self.world_T_camera, np.ndarray):
            if self.world_T_camera.shape != (4, 4):
                raise ValueError(f"FramePacket.world_T_camera must be (4, 4) pose matrix")

        if self.metadata is None:
            self.metadata = {}

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    @property
    def has_pose(self) -> bool:
        return self.world_T_camera is not None

    @property
    def has_imu(self) -> bool:
        return len(self.imu_samples) > 0
