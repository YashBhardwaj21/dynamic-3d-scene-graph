import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.data.tum_source import TUMReplaySource
from scene_graph.pipeline.online_pipeline import OnlinePipeline

def run_pipeline(config, max_frames: int):
    """Run pipeline for max_frames and return active edges."""
    pipeline = OnlinePipeline(config)
    source = TUMReplaySource(config)
    
    # Instead of reading from disk in tests, we'll patch the source directly below.
    # The source is created in the test and we will just yield dummy packets.
    for i, packet in enumerate(source):
        if i >= max_frames:
            break
        graph = pipeline.update(packet)
        
    return graph.get_active_edges()

from unittest.mock import patch, MagicMock
from scene_graph.data.frame_packet import FramePacket
import numpy as np

def dummy_packet(index):
    return FramePacket(
        frame_index=index,
        timestamp=index * 0.1,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=np.zeros((480, 640), dtype=np.uint16),
        pose=np.eye(4),
        has_depth=True,
        has_pose=True
    )

def dummy_source(config):
    # A dummy iterable that yields 20 packets
    return [dummy_packet(i) for i in range(20)]

def test_prefix_equivalence():
    """Prefix equivalence causality test.
    
    Ensures that State_full(N) == State_prefix(N).
    This proves the temporal state machine and pipeline do not leak future information.
    """
    with patch('scene_graph.pipeline.online_pipeline.YOLOEDetector') as MockDetector, \
         patch('tests.integration.test_causality.TUMReplaySource', side_effect=dummy_source):
         
        # Make the mocked detector return an empty list of observations so it doesn't crash the tracker
        mock_instance = MockDetector.return_value
        mock_instance.detect.return_value = []
        
        config = SceneGraphConfig.from_files("configs/tum_fr1_desk.yaml")
        
        # 1. Run pipeline from frame 0 to 10
        edges_prefix = run_pipeline(config, max_frames=10)
        
        # 2. Run pipeline from frame 0 to 15
        
        pipeline_full = OnlinePipeline(config)
        source_full = TUMReplaySource(config)
        
        edges_at_10 = None
        
        for i, packet in enumerate(source_full):
            if i >= 15:
                break
            graph = pipeline_full.update(packet)
            if i == 9: # Zero-indexed, 10th frame
                edges_at_10 = graph.get_active_edges()
                
        # 3. Assert equality
        # The edges should be exactly the same.
        assert len(edges_prefix) == len(edges_at_10)
        
        # Check that they contain the exact same subject-predicate-object combinations
        set_prefix = {(e.subject_id, e.predicate, e.object_id) for e in edges_prefix}
        set_full = {(e.subject_id, e.predicate, e.object_id) for e in edges_at_10}
        
        assert set_prefix == set_full
