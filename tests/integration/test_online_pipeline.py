import pytest

from scene_graph.config import SceneGraphConfig
from scene_graph.pipeline.online_pipeline import OnlinePipeline


def test_online_pipeline_empty_init():
    """Verify the causal online pipeline starts completely empty."""
    config = SceneGraphConfig({
        "dataset": {"name": "test"}
    })
    
    pipeline = OnlinePipeline(config)
    
    # Verify core was instantiated
    assert pipeline.core is not None
    assert pipeline.core.config.get("dataset.name") == "test"
    
    # As the system is built, we should verify that at frame 1, 
    # there are NO precomputed tracks and NO pre-existing graph relations.
    # We leave this as a skeleton test for the causality constraint requirement.
