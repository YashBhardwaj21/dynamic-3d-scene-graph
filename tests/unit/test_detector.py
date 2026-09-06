"""Unit tests for detector implementations (Substage 2.3 & 2.4)."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.perception.stored_loader import StoredObservationLoader


@pytest.fixture
def dummy_observations_dir():
    """Create a temporary directory with metadata and chunked dummy observations."""
    metadata = {
        "dataset": "test",
        "total_frames": 2,
        "chunk_size": 100,
        "chunks": ["chunk_0000.json"],
        "detector": {"model": "test_model"}
    }
    
    chunk_0000 = {
        "frames": [
            {
                "frame_index": 100,
                "timestamp": 1.0,
                "observations": [
                    {
                        "obs_id": "obs_1",
                        "frame_index": 100,
                        "timestamp": 1.0,
                        "class_name": "cup",
                        "confidence": 0.95,
                        "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
                        "centroid_world": [1.0, 2.0, 3.0],
                        "valid_point_count": 100
                    }
                ]
            },
            {
                "frame_index": 101,
                "timestamp": 1.1,
                "observations": []
            }
        ]
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Write metadata
        with open(temp_path / "metadata.json", 'w') as f:
            json.dump(metadata, f)
            
        # Write chunks
        obs_dir = temp_path / "observations_controlled"
        obs_dir.mkdir()
        
        with open(obs_dir / "chunk_0000.json", 'w') as f:
            json.dump(chunk_0000, f)
            
        yield str(temp_path)


# Dummy intrinsics/depth for FramePacket construction
_DUMMY_INTRINSICS = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)
_DUMMY_DEPTH_MODEL = DepthModel(scale=5000.0)


def test_stored_observation_loader(dummy_observations_dir):
    """Verify StoredObservationLoader correctly parses chunked JSON and returns Observations."""
    loader = StoredObservationLoader(dummy_observations_dir)
    
    # Test metadata
    assert loader.get_model_info()["model"] == "test_model"
    
    # Create dummy packet for frame 100
    packet_100 = FramePacket(
        frame_index=100,
        timestamp=1.0,
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=None,
        world_T_camera=None,
        camera_intrinsics=_DUMMY_INTRINSICS,
        depth_model=_DUMMY_DEPTH_MODEL
    )
    
    observations_100 = loader.detect(packet_100)
    assert len(observations_100) == 1
    
    obs = observations_100[0]
    assert obs.obs_id == "obs_1"
    assert obs.class_name == "cup"
    assert obs.confidence == 0.95
    np.testing.assert_array_equal(obs.bbox_xyxy, [10.0, 10.0, 50.0, 50.0])
    # centroid_world now lives on object_geometry, populated by the loader
    assert obs.object_geometry is not None
    np.testing.assert_array_almost_equal(obs.object_geometry.centroid_world, [1.0, 2.0, 3.0])
    assert obs.object_geometry.valid_point_count == 100
    assert obs.mask_rle is None
    
    # Test frame 101 (empty observations)
    packet_101 = FramePacket(
        frame_index=101,
        timestamp=1.1,
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=None,
        world_T_camera=None,
        camera_intrinsics=_DUMMY_INTRINSICS,
        depth_model=_DUMMY_DEPTH_MODEL
    )
    
    observations_101 = loader.detect(packet_101)
    assert len(observations_101) == 0
    
    # Test unknown frame (should return empty list)
    packet_102 = FramePacket(
        frame_index=102,
        timestamp=1.2,
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=None,
        world_T_camera=None,
        camera_intrinsics=_DUMMY_INTRINSICS,
        depth_model=_DUMMY_DEPTH_MODEL
    )
    
    observations_102 = loader.detect(packet_102)
    assert len(observations_102) == 0
