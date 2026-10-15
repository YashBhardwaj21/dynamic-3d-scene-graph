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


def test_cross_session_physical_identity_scene_graph_fusion():
    """Stage 9C Integration Test: Associate real observations across sessions into persistent scene graph."""
    from vlm.src.cross_session_association import CrossSessionCandidateMatcher
    from vlm.src.stage5_adapter import load_registration_audit

    obs_path = root_dir / "vlm" / "outputs" / "object_observations.jsonl"
    reg_path = root_dir / "vlm" / "outputs" / "stage5_registration_audit.json"

    if not obs_path.exists() or not reg_path.exists():
        pytest.skip("Real observations or registration audit missing.")

    reg = load_registration_audit(reg_path)
    assert reg.is_valid, f"Registration must be valid: {reg.status}"

    observations = []
    with open(obs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))

    # Aggregate clusters
    s1_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(observations, "recording_01", min_obs=2)
    s2_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(observations, "recording_02", min_obs=2)

    matcher = CrossSessionCandidateMatcher(max_static_distance_m=0.45)
    assoc_res = matcher.match_entities(s1_clusters, s2_clusters, reg)

    assert assoc_res.is_valid
    assert len(assoc_res.matched_pairs) > 0

    # Pick a verified static match, e.g. desk_001 <-> desk_001 or desk_005 <-> desk_014
    target_match = next((m for m in assoc_res.matched_pairs if m.s1_entity_id == "desk_001" and m.s2_entity_id == "desk_001"), None)
    if not target_match:
        target_match = assoc_res.matched_pairs[0]

    s1_eid = target_match.s1_entity_id
    s2_eid = target_match.s2_entity_id
    persistent_id = target_match.persistent_entity_id

    # Create Scene Graph Fusion
    fusion = ContextualSceneGraphFusion()

    # Feed Session 1 observation
    c1 = s1_clusters[s1_eid]
    obs_meta_s1 = {
        "timestamp": 10.0,
        "source_session": "semantic_mapping_my_desk_recording_01",
        "obs_id": c1.observation_ids[0] if c1.observation_ids else "obs_s1_001",
        "local_track_id": s1_eid,
        "centroid": [c1.centroid_floor[0], c1.centroid_floor[1], 0.72],
        "spatial_extent_m": c1.spatial_extent_m,
        "coordinate_frame": "reference_frame",
        "registration": reg,
    }
    routed_s1 = {
        "decision": "RESOLVE_CHEAP",
        "prediction": "MOVABLE",
        "observed_motion": "STATIC",
        "evidence_provenance": "MEASURED_FULL",
    }
    node = fusion.update_or_create_node(persistent_id, c1.semantic_class, routed_s1, obs_meta_s1)

    assert node.entity_id == persistent_id
    assert node.total_observations_count == 1
    assert "semantic_mapping_my_desk_recording_01" in node.source_sessions
    assert s1_eid in node.source_track_ids

    # Feed Session 2 observation (in S2 local coordinates, should be transformed to reference frame via reg)
    c2 = s2_clusters[s2_eid]
    obs_meta_s2 = {
        "timestamp": 25.0,
        "source_session": "semantic_mapping_my_desk_recording_02",
        "obs_id": c2.observation_ids[0] if c2.observation_ids else "obs_s2_001",
        "local_track_id": s2_eid,
        "centroid": [c2.centroid_floor[0], c2.centroid_floor[1], 0.72],
        "spatial_extent_m": c2.spatial_extent_m,
        "coordinate_frame": "session_02_local",
        "registration": reg,
    }
    routed_s2 = {
        "decision": "RESOLVE_CHEAP",
        "prediction": "MOVABLE",
        "observed_motion": "STATIC",
        "evidence_provenance": "MEASURED_FULL",
    }
    node_merged = fusion.update_or_create_node(persistent_id, c2.semantic_class, routed_s2, obs_meta_s2)

    # Verify identity continuity across sessions
    assert node_merged.total_observations_count == 2
    assert len(node_merged.source_sessions) == 2
    assert "semantic_mapping_my_desk_recording_01" in node_merged.source_sessions
    assert "semantic_mapping_my_desk_recording_02" in node_merged.source_sessions
    assert s1_eid in node_merged.source_track_ids
    assert s2_eid in node_merged.source_track_ids

    # Geometry preservation: merged centroid should be in reference frame within small delta of S1 centroid
    dist_to_s1 = np.linalg.norm(node_merged.centroid_world[:2] - c1.centroid_floor[:2])
    assert dist_to_s1 < 0.25, f"Merged centroid drifted too far from reference frame: {dist_to_s1}m"

    # Serialize to JSON and PersistentEntity
    serialized = fusion.serialize_to_json()
    assert persistent_id in serialized["nodes"]

    persistent_entity = node_merged.to_persistent_entity()
    assert persistent_entity.entity_id == persistent_id
    assert s1_eid in persistent_entity.source_track_ids
    assert s2_eid in persistent_entity.source_track_ids
    assert persistent_entity.lifecycle_state == EntityLifecycleState.CONFIRMED

