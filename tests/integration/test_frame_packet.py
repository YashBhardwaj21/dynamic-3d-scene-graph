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

from scene_graph.data.frame_packet import build_frame_packets


WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
TUM_DATASET_DIR = WORKSPACE_DIR / "rgbd_dataset_freiburg1_desk"


@pytest.mark.skipif(not TUM_DATASET_DIR.is_dir(), reason="TUM dataset not found")
def test_build_frame_packets_201_frames():
    """Verify stream builder yields 201 packets with correct shapes and types."""
    start_frame = 100
    end_frame = 300
    expected_count = end_frame - start_frame + 1  # 201
    
    packets = build_frame_packets(
        sequence_dir=TUM_DATASET_DIR,
        start_frame=start_frame,
        end_frame=end_frame
    )
    
    assert len(packets) == expected_count, f"Expected {expected_count} packets, got {len(packets)}"
    
    # Check bounds
    assert packets[0].frame_index == 100
    assert packets[-1].frame_index == 300
    
    has_depth_count = sum(1 for p in packets if p.has_depth)
    has_pose_count = sum(1 for p in packets if p.has_pose)
    
    assert has_depth_count > 150, f"Expected high depth association, got {has_depth_count}"
    assert has_pose_count > 150, f"Expected high pose association, got {has_pose_count}"
    
    # Deep integration check for representative frames
    check_indices = [100, 150, 200, 250, 300]
    
    for packet in packets:
        # Mandatory RGB checks
        assert isinstance(packet.rgb, np.ndarray)
        assert packet.rgb.shape == (480, 640, 3)
        assert packet.rgb.dtype == np.uint8
        
        # Optional depth checks
        if packet.has_depth:
            assert isinstance(packet.depth, np.ndarray)
            assert packet.depth.shape == (480, 640)
            assert packet.depth.dtype == np.uint16
            
        # Optional pose checks
        if packet.has_pose:
            assert isinstance(packet.pose, np.ndarray)
            assert packet.pose.shape == (4, 4)
            assert packet.pose.dtype == np.float64
            # Verify R.T @ R ≈ I
            R = packet.pose[:3, :3]
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)
            # Verify det(R) ≈ 1
            assert pytest.approx(np.linalg.det(R), abs=1e-5) == 1.0
            
        # Check specific representative frames if they have data
        if packet.frame_index in check_indices:
            assert packet.timestamp > 0.0
