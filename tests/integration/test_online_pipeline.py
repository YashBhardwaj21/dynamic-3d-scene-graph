import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.pipeline.online_pipeline import OnlinePipeline


from unittest.mock import patch

def test_online_pipeline_empty_init():
    """Verify the causal online pipeline starts completely empty."""
    config = SceneGraphConfig({
        "dataset": {"name": "test"}
    })
    
    with patch('scene_graph.pipeline.online_pipeline.YOLOEDetector'):
        pipeline = OnlinePipeline(config)
        
        # Verify it was instantiated
        assert pipeline.config.get("dataset.name") == "test"
        
        # Verify graph is empty
        assert len(pipeline.graph.nodes) == 0
        assert len(pipeline.graph.edges) == 0
