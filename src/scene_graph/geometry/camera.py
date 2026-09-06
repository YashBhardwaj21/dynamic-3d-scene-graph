"""Camera intrinsics and pixel projection."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsics (focal length and principal point)."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    
    def __post_init__(self):
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths (fx, fy) must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions (width, height) must be > 0")

    def pixel_to_camera(self, u: float, v: float, depth_m: float) -> tuple[float, float, float]:
        """Project a 2D pixel with depth into 3D camera coordinates.
        
        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.
            depth_m: Depth in meters along the Z axis.
            
        Returns:
            (x, y, z) tuple in camera coordinate space.
        """
        if not np.isfinite(depth_m) or depth_m <= 0:
            raise ValueError(f"Invalid depth: {depth_m}")
            
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        return x, y, z


@dataclass(frozen=True)
class DepthModel:
    """Depth scaling model.
    
    Converts raw depth units to meters.
    """
    scale: float
    
    def __post_init__(self):
        if self.scale <= 0:
            raise ValueError("Scale must be > 0")
    
    def depth_to_meters(self, depth_raw: np.ndarray) -> np.ndarray:
        """Convert raw uint16 depth image to metric float depth."""
        return depth_raw.astype(np.float64) / self.scale

