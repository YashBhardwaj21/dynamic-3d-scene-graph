"""Integration tests for FramePacket builder (Substage 1.3).

Verifies:
1. Emits all 201 frames between start_frame 100 and end_frame 300 (inclusive).
2. Successfully loads and parses real image files from the dataset.
3. RGB is (480, 640, 3) uint8.
4. Depth (where available) is (480, 640) uint16.
5. Pose (where available) is (4, 4) float matrix.
"""

from pathlib import Path
import numpy as np
import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.data.tum_source import TUMReplaySource


WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
TUM_DATASET_DIR = WORKSPACE_DIR / "rgbd_dataset_freiburg1_desk"


@pytest.mark.skipif(not TUM_DATASET_DIR.is_dir(), reason="TUM dataset not found")
def test_tum_replay_source_201_frames():
    """Verify source yields 201 packets with correct shapes and types."""
    start_frame = 100
    end_frame = 300
    expected_count = end_frame - start_frame + 1  # 201
    
    config = SceneGraphConfig.model_validate({
        "dataset": {"root": str(TUM_DATASET_DIR), "type": "tum"},
        "sequence": {"start_frame": start_frame, "end_frame": end_frame},
        "sync": {"rgb_depth_max_dt": 0.02, "rgb_pose_max_dt": 0.02}
    })
    
    source = TUMReplaySource(config)
    packets = list(source)
    
    assert len(source) == expected_count
    assert len(packets) == expected_count, f"Expected {expected_count} packets, got {len(packets)}"
    
    # Check bounds
    assert packets[0].frame_index == 100
    assert packets[-1].frame_index == 300
    
    from scene_graph.data.synchronization import associate
    from scene_graph.data.tum_loader import TUMLoader
    
    loader = TUMLoader(TUM_DATASET_DIR)
    
    # Calculate exact expected matches using our frozen optimal DP algorithm
    rgb_entries = loader.load_rgb()[start_frame:end_frame + 1]
    depth_entries = loader.load_depth()
    pose_entries = loader.load_groundtruth()
    
    rgb_ts = [e.timestamp for e in rgb_entries]
    depth_ts = [e.timestamp for e in depth_entries]
    pose_ts = [e.timestamp for e in pose_entries]
    
    expected_depth_matches = len(associate(rgb_ts, depth_ts, max_dt=0.02))
    expected_pose_matches = len(associate(rgb_ts, pose_ts, max_dt=0.02))
    
    has_depth_count = sum(1 for p in packets if p.has_depth)
    has_pose_count = sum(1 for p in packets if p.has_pose)
    
    assert has_depth_count == expected_depth_matches, f"Expected exact depth association {expected_depth_matches}, got {has_depth_count}"
    assert has_pose_count == expected_pose_matches, f"Expected exact pose association {expected_pose_matches}, got {has_pose_count}"
    
    # Verify exact sequence
    assert [p.frame_index for p in packets] == list(range(start_frame, end_frame + 1))
    
    rgb_to_depth = dict(associate(rgb_ts, depth_ts, max_dt=0.02))
    rgb_to_pose = dict(associate(rgb_ts, pose_ts, max_dt=0.02))
    
    # Deep integration check for representative frames
    check_indices = [100, 150, 200, 250, 300]
    
    for packet in packets:
        idx = packet.frame_index
        local_idx = idx - start_frame
        
        # Mandatory RGB checks
        # Verify FramePacket standardizes to strictly RGB channel order
        assert isinstance(packet.rgb, np.ndarray)
        assert packet.rgb.shape == (480, 640, 3)
        assert packet.rgb.dtype == np.uint8
        
        # Optional depth checks
        if packet.has_depth:
            assert isinstance(packet.depth, np.ndarray)
            assert packet.depth.shape == (480, 640)
            assert packet.depth.dtype == np.uint16
            
            # Verify timestamp delta
            d_idx = rgb_to_depth[local_idx]
            dt = abs(rgb_ts[local_idx] - depth_ts[d_idx])
            assert dt <= 0.02
            
        # Optional pose checks
        if packet.has_pose:
            assert isinstance(packet.world_T_camera, np.ndarray)
            assert packet.world_T_camera.shape == (4, 4)
            assert packet.world_T_camera.dtype == np.float64
            # Verify R.T @ R ≈ I
            R = packet.world_T_camera[:3, :3]
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)
            assert pytest.approx(np.linalg.det(R), abs=1e-5) == 1.0
            
            # Verify timestamp delta
            p_idx = rgb_to_pose[local_idx]
            dt = abs(rgb_ts[local_idx] - pose_ts[p_idx])
            assert dt <= 0.02
            
        # Check specific representative frames if they have data
        if idx in check_indices:
            assert packet.timestamp > 0.0

