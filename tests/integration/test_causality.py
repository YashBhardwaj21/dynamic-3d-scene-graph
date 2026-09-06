import numpy as np
from unittest.mock import patch

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.pipeline.online_pipeline import OnlinePipeline
from scene_graph.data.tum_source import TUMReplaySource


CAMERA_INTRINSICS = CameraIntrinsics(
    fx=525.0,
    fy=525.0,
    cx=319.5,
    cy=239.5,
    width=640,
    height=480,
)

DEPTH_MODEL = DepthModel(scale=5000.0)


def dummy_packet(index: int) -> FramePacket:
    return FramePacket(
        frame_index=index,
        timestamp=index * 0.1,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=np.zeros((480, 640), dtype=np.uint16),
        world_T_camera=np.eye(4),
        camera_intrinsics=CAMERA_INTRINSICS,
        depth_model=DEPTH_MODEL,
    )


def dummy_source(config):
    return [dummy_packet(i) for i in range(20)]


def run_pipeline(config, max_frames: int):
    pipeline = OnlinePipeline(config)
    source = TUMReplaySource(config)

    for i, packet in enumerate(source):
        if i >= max_frames:
            break
        graph = pipeline.update(packet)

    return graph.get_active_edges()


def test_prefix_equivalence():
    with patch(
        "scene_graph.pipeline.online_pipeline.YOLOEDetector"
    ) as MockDetector, patch(
        "tests.integration.test_causality.TUMReplaySource",
        side_effect=dummy_source,
    ):
        MockDetector.return_value.detect.return_value = []

        config = SceneGraphConfig.from_files("configs/tum_fr1_desk.yaml")

        edges_prefix = run_pipeline(config, max_frames=10)

        pipeline_full = OnlinePipeline(config)
        source_full = TUMReplaySource(config)

        edges_at_10 = None

        for i, packet in enumerate(source_full):
            if i >= 15:
                break

            graph = pipeline_full.update(packet)

            if i == 9:
                edges_at_10 = graph.get_active_edges()

        assert edges_at_10 is not None

        prefix_relations = {
            (edge.subject_id, edge.predicate, edge.object_id)
            for edge in edges_prefix
        }

        full_relations = {
            (edge.subject_id, edge.predicate, edge.object_id)
            for edge in edges_at_10
        }

        assert prefix_relations == full_relations
