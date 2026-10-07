import pytest
import numpy as np

from scene_graph.data.frame_packet import FramePacket, IMUSample
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.reference_frame import RelationReferenceFrame


def _dummy_intrinsics():
    return CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)


def test_frame_packet_accepts_metric_float_depth():
    """Verify FramePacket accepts float32 depth in meters."""
    intrinsics = _dummy_intrinsics()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_m = np.ones((480, 640), dtype=np.float32) * 1.5  # 1.5 meters

    packet = FramePacket(
        frame_index=0,
        timestamp=100.0,
        rgb=rgb,
        depth=depth_m,
        world_T_camera=np.eye(4),
        camera_intrinsics=intrinsics,
        frame_id="camera_color_optical_frame",
        optical_frame_id="camera_depth_optical_frame",
        pose_source="groundtruth",
    )

    assert packet.has_depth
    assert packet.depth.dtype == np.float32
    assert packet.depth[240, 320] == 1.5
    assert packet.frame_id == "camera_color_optical_frame"
    assert packet.optical_frame_id == "camera_depth_optical_frame"
    assert packet.pose_source == "groundtruth"


def test_frame_packet_rejects_raw_integer_depth():
    """Verify FramePacket strictly rejects integer/uint16 depth (adapter invariant)."""
    intrinsics = _dummy_intrinsics()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    raw_uint16_depth = np.ones((480, 640), dtype=np.uint16) * 1000  # 1000 raw units

    with pytest.raises(ValueError, match="FramePacket.depth must be metric depth in meters"):
        FramePacket(
            frame_index=0,
            timestamp=100.0,
            rgb=rgb,
            depth=raw_uint16_depth,
            world_T_camera=np.eye(4),
            camera_intrinsics=intrinsics,
        )


def test_frame_packet_timestamp_validation():
    """Verify FramePacket validates timestamps (non-negative and finite)."""
    intrinsics = _dummy_intrinsics()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="timestamp must be finite and non-negative"):
        FramePacket(
            frame_index=0,
            timestamp=-0.5,
            rgb=rgb,
            depth=None,
            world_T_camera=np.eye(4),
            camera_intrinsics=intrinsics,
        )

    with pytest.raises(ValueError, match="timestamp must be finite and non-negative"):
        FramePacket(
            frame_index=0,
            timestamp=float("nan"),
            rgb=rgb,
            depth=None,
            world_T_camera=np.eye(4),
            camera_intrinsics=intrinsics,
        )


def test_adapter_raw_to_metric_equivalence():
    """Verify TUM and RealSense adapters produce equivalent metric depth in FramePackets.
    
    1000 raw units in RealSense (scale 1000.0) == 1.0 meter
    5000 raw units in TUM (scale 5000.0) == 1.0 meter
    Both yield depth array with value 1.000 float32.
    """
    intrinsics = _dummy_intrinsics()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    # Raw RealSense depth: 1000 mm -> 1.0 meter
    realsense_raw = np.full((480, 640), 1000, dtype=np.uint16)
    realsense_scale = 1000.0
    realsense_depth_m = realsense_raw.astype(np.float32) / realsense_scale

    # Raw TUM depth: 5000 units -> 1.0 meter
    tum_raw = np.full((480, 640), 5000, dtype=np.uint16)
    tum_scale = 5000.0
    tum_depth_m = tum_raw.astype(np.float32) / tum_scale

    packet_rs = FramePacket(
        frame_index=0,
        timestamp=1.0,
        rgb=rgb,
        depth=realsense_depth_m,
        world_T_camera=np.eye(4),
        camera_intrinsics=intrinsics,
    )

    packet_tum = FramePacket(
        frame_index=0,
        timestamp=1.0,
        rgb=rgb,
        depth=tum_depth_m,
        world_T_camera=np.eye(4),
        camera_intrinsics=intrinsics,
    )

    assert packet_rs.depth.dtype == np.float32
    assert packet_tum.depth.dtype == np.float32
    np.testing.assert_allclose(packet_rs.depth, 1.0, atol=1e-6)
    np.testing.assert_allclose(packet_tum.depth, 1.0, atol=1e-6)
    np.testing.assert_array_equal(packet_rs.depth, packet_tum.depth)
