import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.pipeline.online_pipeline import OnlinePipeline


from unittest.mock import patch

def test_online_pipeline_empty_init():
    """Verify the causal online pipeline starts completely empty."""
    config = SceneGraphConfig.model_validate({
        "dataset": {"name": "test", "root": ".", "type": "tum"},
        "perception": {"classes": ["cup", "book"]}
    })
    
    with patch('scene_graph.pipeline.online_pipeline.YOLOEDetector'):
        pipeline = OnlinePipeline(config)
        
        # Verify it was instantiated
        assert pipeline.config.dataset.name == "test"
        
        # Verify graph is empty (graph lives on the core pipeline)
        assert len(pipeline.core.graph.nodes) == 0
        assert len(pipeline.core.graph.edges) == 0
