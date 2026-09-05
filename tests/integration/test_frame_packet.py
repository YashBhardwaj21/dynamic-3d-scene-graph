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

from scene_graph.data.frame_packet import build_frame_packets, FramePacket

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
TUM_DATASET_DIR = WORKSPACE_DIR / "rgbd_dataset_freiburg1_desk"

@pytest.mark.skipif(not TUM_DATASET_DIR.is_dir(), reason="TUM dataset not found")
def test_build_frame_packets_201_frames():
    """Verify stream builder yields 201 packets with correct shapes and types."""
    # From the frozen parameters, frames [100, 300] inclusive should yield 201 frames
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
    
    # Check data formats on the first packet
    first = packets[0]
    
    # RGB checks
    assert isinstance(first.rgb, np.ndarray)
    assert first.rgb.shape == (480, 640, 3)
    assert first.rgb.dtype == np.uint8
    
    # Most frames in this dense sequence should have depth and pose
    # We verify the data structure for whichever packet has depth
    has_depth_count = 0
    has_pose_count = 0
    
    for packet in packets:
        if packet.has_depth:
            has_depth_count += 1
            assert isinstance(packet.depth, np.ndarray)
            assert packet.depth.shape == (480, 640)
            assert packet.depth.dtype == np.uint16
            
        if packet.has_pose:
            has_pose_count += 1
            assert isinstance(packet.pose, np.ndarray)
            assert packet.pose.shape == (4, 4)
            assert packet.pose.dtype == np.float64
            
    # In TUM fr1_desk, almost all frames in this segment have depth and pose.
    # We just ensure it's successfully matching the vast majority.
    assert has_depth_count > 170, f"Expected high depth association, got {has_depth_count}"
    assert has_pose_count > 170, f"Expected high pose association, got {has_pose_count}"
