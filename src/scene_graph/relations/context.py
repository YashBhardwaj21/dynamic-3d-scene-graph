from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.reference_frame import (
    RelationReferenceFrame,
    CameraFrame,
)


@dataclass
class ObservationGeometry:
    """Cached per-observation. Computed ONCE, consumed by all relation modules."""

    obs_id: str
    track_id: str
    centroid_world: np.ndarray
    bbox_min_world: np.ndarray
    bbox_max_world: np.ndarray
    depth_stats: Optional[Dict[str, float]]
    points_world_sampled: Optional[np.ndarray]
    points_world: Optional[np.ndarray]
    points_camera: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    valid_point_count: int
    position_covariance_world: Optional[np.ndarray] = None
    obb_center_world: Optional[np.ndarray] = None
    obb_axes_world: Optional[np.ndarray] = None
    obb_extents_world: Optional[np.ndarray] = None


@dataclass
class FrameContext:
    frame_index: int
    timestamp: float
    intrinsics: CameraIntrinsics
    world_T_camera: np.ndarray
    reference_frame: RelationReferenceFrame
    camera_frame: CameraFrame
    depth_image: Optional[np.ndarray]
    observation_geometry: Dict[str, ObservationGeometry]