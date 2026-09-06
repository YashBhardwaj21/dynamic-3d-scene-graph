from dataclasses import dataclass
import numpy as np


@dataclass
class RelationReferenceFrame:
    """The canonical reference frame for relation evaluation.
    
    All geometric relation checks (left/right, above/below, front/back) 
    MUST be evaluated in this frame to ensure mathematical consistency.
    """
    origin_world: np.ndarray        # (3,) World coordinate of the origin (e.g. camera center)
    up_axis_world: np.ndarray       # (3,) Unit vector for 'up' (positive Y in camera frame, usually)
    horizontal_axis_world: np.ndarray # (3,) Unit vector for 'right' (positive X in camera frame)
    depth_axis_world: np.ndarray    # (3,) Unit vector for 'depth' (positive Z in camera frame)
    
    def __post_init__(self):
        # Validate unit vectors
        for axis in [self.up_axis_world, self.horizontal_axis_world, self.depth_axis_world]:
            if not np.all(np.isfinite(axis)):
                raise ValueError("Reference frame axes must be finite")
            norm = np.linalg.norm(axis)
            if not np.isclose(norm, 1.0, atol=1e-4):
                raise ValueError(f"Reference frame axes must be unit vectors, got norm {norm}")
                
        # Validate orthogonality
        if not np.isclose(np.dot(self.up_axis_world, self.horizontal_axis_world), 0.0, atol=1e-4):
            raise ValueError("Up and horizontal axes must be orthogonal")
        if not np.isclose(np.dot(self.up_axis_world, self.depth_axis_world), 0.0, atol=1e-4):
            raise ValueError("Up and depth axes must be orthogonal")
        if not np.isclose(np.dot(self.horizontal_axis_world, self.depth_axis_world), 0.0, atol=1e-4):
            raise ValueError("Horizontal and depth axes must be orthogonal")

    @classmethod
    def from_camera_pose(cls, world_T_camera: np.ndarray) -> "RelationReferenceFrame":
        """Construct a reference frame from a camera pose matrix.
        
        Assumes OpenCV camera convention: X right, Y down, Z forward.
        For relations, we often want 'up' to be negative Y camera, 
        'right' to be positive X camera, and 'depth' to be positive Z camera.
        """
        from scene_graph.geometry.transforms import validate_se3_transform
        validate_se3_transform(world_T_camera)
        
        R = world_T_camera[:3, :3]
        T = world_T_camera[:3, 3]
        
        # Camera axes in world coordinates
        # R columns are the camera axes expressed in world coordinates
        cam_x = R[:, 0]  # Right
        cam_y = R[:, 1]  # Down
        cam_z = R[:, 2]  # Forward
        
        return cls(
            origin_world=T,
            up_axis_world=-cam_y,         # 'Up' is negative camera Y
            horizontal_axis_world=cam_x,  # 'Right' is positive camera X
            depth_axis_world=cam_z        # 'Depth' is positive camera Z
        )
