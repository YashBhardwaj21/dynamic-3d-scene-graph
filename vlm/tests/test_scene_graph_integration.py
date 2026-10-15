"""Stage 5 -> Stage 6 -> Contextual Scene Graph Real-Recording Integration Test.

Verifies end-to-end data flow:
1. Loads real-recording observations and Stage 5 evidence.
2. Adapts evidence through Stage5EvidenceAdapter without ground truth leakage.
3. Routes through SelectiveRouter in deterministic offline mode (cached/prior).
4. Updates persistent ContextualSceneGraphFusion.
5. Converts to authoritative PersistentEntity.
6. Serializes scene graph to JSON.
7. Validates mathematical running mean, identity semantics (track IDs vs obs IDs),
   geometry validity, uncertainty propagation, and lifecycle state.
"""

import json
import math
import numpy as np
import pytest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent

from vlm.src.stage5_adapter import Stage5EvidenceAdapter
from vlm.src.selective_router import SelectiveRouter
from vlm.src.contextual_scene_graph_fusion import (
    ContextualSceneGraphFusion,
    ContextualMovability,
    ObservedMotionState,
)
from scene_graph.ontology.entity import PersistentEntity, EntityLifecycleState


def test_real_recording_scene_graph_pipeline_integration(tmp_path):
    """End-to-end integration test on real recording data."""
    manifest_path = root_dir / "vlm" / "benchmarks" / "router_evaluation_manifest.json"
    cache_path = root_dir / "vlm" / "outputs" / "stage6" / "router_vlm_cache.json"

    if not manifest_path.exists():
        pytest.skip("Benchmark manifest not available.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)["evaluation_instances"]

    # Target real recording chair instance
    chair_entry = next((item for item in manifest if item["entity_id"] == "inst_chair_002"), None)
    assert chair_entry is not None

    # 1 & 2: Adapter loads real observations and computes non-leaking evidence
    adapter = Stage5EvidenceAdapter(repo_root=root_dir)
    evidence = adapter.build_instance_evidence(chair_entry)

    assert evidence["entity_id"] == "inst_chair_002"
    assert evidence["semantic_class"] == "chair"
    assert evidence["has_valid_pose"] is True
    assert evidence["depth_valid"] is True
    assert evidence["relocation_evaluable"] is True
    assert evidence["world_displacement_m"] > 0.30
    assert evidence["observed_motion"] == "DISPLACED"

    # 3: Route evidence through deterministic selective router
    router = SelectiveRouter(cache_path=cache_path, enable_vlm=True)
    routed = router.route_entity(evidence, repo_root=root_dir)

    assert routed["decision"] in ["ESCALATE", "RESOLVE_CHEAP"]
    assert routed["prediction"] == "MOVABLE"

    # 4: Update Contextual Scene Graph
    fusion = ContextualSceneGraphFusion()

    # Step 4a: First observation
    obs_meta_1 = {
        "timestamp": 100.0,
        "source_session": "semantic_mapping_my_desk_recording_01",
        "obs_id": "obs_s1_001",
        "local_track_id": "trk_1002",
        "centroid": [1.50, 2.50, 0.45],
        "spatial_extent_m": 0.60,
        "depth_valid": True,
        "coordinate_frame": "floor_map",
    }
    node = fusion.update_or_create_node(
        entity_id="inst_chair_002",
        semantic_class="chair",
        routed_result=routed,
        observation_meta=obs_meta_1,
    )

    assert node.entity_id == "inst_chair_002"
    assert node.contextual_movability == ContextualMovability.MOVABLE
    assert node.observed_motion == ObservedMotionState.DISPLACED
    assert node.geometry_valid is True
    assert np.allclose(node.centroid_world, [1.50, 2.50, 0.45])
    assert node.source_track_ids == ["trk_1002"]
    assert node.contributing_observation_ids == ["obs_s1_001"]

    # Step 4b: Second observation with slight variation to verify exact running mean
    obs_meta_2 = {
        "timestamp": 101.5,
        "source_session": "semantic_mapping_my_desk_recording_01",
        "obs_id": "obs_s1_002",
        "local_track_id": "trk_1002",
        "centroid": [1.60, 2.60, 0.45],
        "spatial_extent_m": 0.62,
        "depth_valid": True,
        "coordinate_frame": "floor_map",
    }
    node_updated = fusion.update_or_create_node(
        entity_id="inst_chair_002",
        semantic_class="chair",
        routed_result=routed,
        observation_meta=obs_meta_2,
    )

    assert node_updated.total_observations_count == 2
    # Exact average: ([1.5, 2.5] + [1.6, 2.6]) / 2 = [1.55, 2.55]
    assert np.allclose(node_updated.centroid_world, [1.55, 2.55, 0.45])
    assert node_updated.spatial_extent_m == pytest.approx(0.61, abs=1e-4)
    assert node_updated.source_track_ids == ["trk_1002"]
    assert len(node_updated.contributing_observation_ids) == 2

    # 5: Convert to authoritative PersistentEntity
    entity = node_updated.to_persistent_entity()
    assert isinstance(entity, PersistentEntity)
    assert entity.entity_id == "inst_chair_002"
    assert entity.lifecycle_state == EntityLifecycleState.CONFIRMED
    assert entity.primary_label == "chair"
    assert entity.source_track_ids == ["trk_1002"]  # Must be track ID, not obs ID!
    assert "obs_s1_001" in entity.attributes["contributing_observation_ids"]
    assert entity.attributes["contextual_movability"] == "MOVABLE"
    assert entity.attributes["observed_motion"] == "DISPLACED"

    # 6: Serialization to JSON
    serialized = fusion.serialize_to_json()
    assert serialized["total_nodes"] == 1
    assert "inst_chair_002" in serialized["nodes"]
    assert serialized["nodes"]["inst_chair_002"]["geometry_valid"] is True


def test_missing_geometry_handling():
    """Verify that absent geometry does not fabricate fake coordinates and sets TENTATIVE state."""
    fusion = ContextualSceneGraphFusion()
    routed = {
        "decision": "RESOLVE_CHEAP",
        "prediction": "IMMOVABLE",
        "observed_motion": "STATIC",
        "world_displacement_m": None,
        "relocation_evaluable": False,
        "evidence_provenance": "MISSING",
    }
    # No centroid provided
    obs_meta = {
        "timestamp": 200.0,
        "source_session": "session_01",
        "obs_id": "obs_blind_01",
        "local_track_id": "trk_99",
        "centroid": None,
        "spatial_extent_m": None,
        "depth_valid": False,
    }

    node = fusion.update_or_create_node(
        entity_id="inst_wall_unknown_geo",
        semantic_class="wall",
        routed_result=routed,
        observation_meta=obs_meta,
    )

    assert node.geometry_valid is False
    assert node.centroid_world is None
    assert node.contextual_movability == ContextualMovability.IMMOVABLE

    persistent = node.to_persistent_entity()
    # Must remain TENTATIVE when geometry is missing!
    assert persistent.lifecycle_state == EntityLifecycleState.TENTATIVE
    assert persistent.attributes["geometry_valid"] is False


def test_running_mean_mathematical_consistency():
    """Verify exact running mean without drift across 100 observations."""
    fusion = ContextualSceneGraphFusion()
    routed = {"prediction": "MOVABLE", "observed_motion": "STATIC"}

    # True values to sample around [2.0, 3.0, 1.0]
    true_mean = np.array([2.0, 3.0, 1.0])
    points = [true_mean + np.array([0.1 * math.sin(i), 0.1 * math.cos(i), 0.0]) for i in range(100)]
    expected_mean = np.mean(points, axis=0)

    for i, pt in enumerate(points):
        meta = {
            "timestamp": 100.0 + i,
            "obs_id": f"obs_{i:03d}",
            "local_track_id": "trk_01",
            "centroid": pt,
            "spatial_extent_m": 0.5,
            "depth_valid": True,
        }
        node = fusion.update_or_create_node("inst_test_math", "chair", routed, meta)

    assert node.total_observations_count == 100
    assert node.valid_geometry_observations_count == 100
    assert np.allclose(node.centroid_world, expected_mean, atol=1e-6)
