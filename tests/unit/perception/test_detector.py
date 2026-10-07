"""Unit tests for ObservationProducer and ObservationSource interfaces."""

from typing import List
import numpy as np
import pytest

from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.perception.observation import Observation
from scene_graph.perception.observation_source import ObservationProducer, ObservationSource


class MockObservationProducer(ObservationProducer):
    def detect(self, packet: FramePacket) -> List[Observation]:
        if packet.frame_index == 100:
            return [
                Observation(
                    obs_id="obs_1",
                    frame_index=100,
                    timestamp=packet.timestamp,
                    class_name="cup",
                    confidence=0.95,
                    bbox_xyxy=np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32),
                )
            ]
        return []


def test_observation_producer_interface():
    producer = MockObservationProducer()
    intrinsics = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)
    depth_model = DepthModel(scale=1000.0)

    packet = FramePacket(
        frame_index=100,
        timestamp=1.0,
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=None,
        world_T_camera=None,
        camera_intrinsics=intrinsics,
        depth_model=depth_model,
    )

    observations = producer.detect(packet)
    assert len(observations) == 1
    assert observations[0].obs_id == "obs_1"
    assert observations[0].class_name == "cup"
    assert observations[0].confidence == pytest.approx(0.95)
