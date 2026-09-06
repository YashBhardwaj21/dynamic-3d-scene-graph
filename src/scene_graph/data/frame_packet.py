"""FramePacket dataclass and stream builder."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.reference_frame import RelationReferenceFrame




@dataclass
class FramePacket:
    """A synchronized, per-frame container for downstream scene graph tasks.
    
    Contains NO dataset-specific fields.
    """
    frame_index: int            
    timestamp: float            
    rgb: np.ndarray             # HxWx3 uint8 RGB format
    depth: Optional[np.ndarray] # HxW raw depth units
    world_T_camera: Optional[np.ndarray]  # (4, 4) pose matrix
    camera_intrinsics: CameraIntrinsics
    depth_model: DepthModel
    relation_frame: Optional[RelationReferenceFrame] = None
    metadata: dict = None             # Any other dataset-independent metadata

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    @property
    def has_pose(self) -> bool:
        return self.world_T_camera is not None



