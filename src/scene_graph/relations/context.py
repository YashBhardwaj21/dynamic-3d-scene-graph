from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.reference_frame import RelationReferenceFrame


@dataclass
class ObservationGeometry:
    """Cached per-observation. Computed ONCE, consumed by all relation modules."""
    obs_id: str
    track_id: str
    centroid_world: np.ndarray
    bbox_min_world: np.ndarray
    bbox_max_world: np.ndarray
    points_world: Optional[np.ndarray]  # (N, 3) — only when mask available
    points_camera: Optional[np.ndarray]
    mask: Optional[np.ndarray]          # (H, W) bool — decoded from RLE
    valid_point_count: int


@dataclass
class FrameContext:
    frame_index: int
    timestamp: float
    intrinsics: CameraIntrinsics
    world_T_camera: np.ndarray
    reference_frame: RelationReferenceFrame
    depth_image: Optional[np.ndarray]
    observation_geometry: Dict[str, ObservationGeometry]
