"""Camera intrinsics and pixel projection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsics model.
    
    Defaults match TUM RGB-D Kinect dataset constraints.
    """
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5

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
