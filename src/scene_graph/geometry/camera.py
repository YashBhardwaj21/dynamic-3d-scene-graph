"""Camera intrinsics and pixel projection."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsics model.
    
    Defaults match TUM RGB-D Kinect dataset constraints.
    """
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5
    width: int = 640
    height: int = 480

    def pixel_to_camera(self, u: float, v: float, depth_m: float) -> tuple[float, float, float]:
        """Project a 2D pixel with depth into 3D camera coordinates.
        
        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.
            depth_m: Depth in meters along the Z axis.
            
        Returns:
            (x, y, z) tuple in camera coordinate space.
        """
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        return x, y, z


@dataclass(frozen=True)
class DepthModel:
    """Depth scaling model.
    
    TUM uses a scale of 5000 (i.e. 5000 uint16 = 1.0 meters).
    """
    scale: float = 5000.0
    
    def depth_to_meters(self, depth_raw: np.ndarray) -> np.ndarray:
        """Convert raw uint16 depth image to metric float depth."""
        return depth_raw.astype(np.float64) / self.scale

