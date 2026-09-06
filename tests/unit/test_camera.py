"""Unit tests for Camera Intrinsics and Depth Model (Stage 1 Freeze)."""

import numpy as np
import pytest

from scene_graph.geometry.camera import CameraIntrinsics, DepthModel


def test_depth_model_conversion():
    """Verify depth scale conversion."""
    model = DepthModel(scale=5000.0)
    
    # Test values
    raw_depth = np.array([5000, 10000, 0, 2500], dtype=np.uint16)
    metric_depth = model.depth_to_meters(raw_depth)
    
    np.testing.assert_allclose(metric_depth, [1.0, 2.0, 0.0, 0.5])


def test_depth_model_custom_scale():
    """Verify custom scale factor."""
    model = DepthModel(scale=1000.0)
    
    raw_depth = np.array([1000, 2000], dtype=np.uint16)
    metric_depth = model.depth_to_meters(raw_depth)
    
    np.testing.assert_allclose(metric_depth, [1.0, 2.0])


def test_camera_intrinsics_validation():
    """Verify CameraIntrinsics rejects invalid values."""
    with pytest.raises(ValueError, match="must be > 0"):
        CameraIntrinsics(fx=-5.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=480)
        
    with pytest.raises(ValueError, match="must be > 0"):
        CameraIntrinsics(fx=525.0, fy=0.0, cx=319.5, cy=239.5, width=640, height=480)
        
    with pytest.raises(ValueError, match="must be > 0"):
        CameraIntrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=0, height=480)
        
    with pytest.raises(ValueError, match="must be > 0"):
        CameraIntrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=-480)
        
def test_depth_model_validation():
    """Verify DepthModel rejects invalid scale."""
    with pytest.raises(ValueError, match="Scale must be > 0"):
        DepthModel(scale=0)
    with pytest.raises(ValueError, match="Scale must be > 0"):
        DepthModel(scale=-1000.0)
