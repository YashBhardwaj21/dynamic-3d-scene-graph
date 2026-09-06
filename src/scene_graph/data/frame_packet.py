"""FramePacket dataclass and stream builder."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np




@dataclass
class FramePacket:
    """A synchronized, per-frame container for downstream scene graph tasks.
    
    RGB is mandatory. Depth and Pose are optional.
    """
    frame_index: int            # original RGB stream index
    timestamp: float            # RGB timestamp
    rgb: np.ndarray             # (480, 640, 3) uint8 RGB format
    depth: Optional[np.ndarray] # (480, 640) uint16
    pose: Optional[np.ndarray]  # (4, 4) world_T_camera
    has_depth: bool
    has_pose: bool



