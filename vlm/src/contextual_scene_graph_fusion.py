"""Stage 8: Contextual Semantic Map and Dynamic 3D Scene Graph Fusion.

Influenced by Rüdt et al. (Towards Dynamic 3D Scene Graphs):
Integrates estimator-aware evidence routing into persistent scene graph entities.

Maintains strict separation between:
1. Contextual movability: IMMOVABLE (fixed/mounted), MOVABLE (transportable furniture),
   MOBILE (self-propelled agent or carrier).
2. Observed motion: STATIC, DISPLACED, DYNAMIC, UNCERTAIN during observed interval.
3. Physical relocation: verified SE(2)/SE(3) measured change in object position.

Preserves geometric state even when contextual semantics is UNKNOWN or ambiguous.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Import repository scene graph ontology
try:
    from scene_graph.ontology.entity import (
        EntityType,
        EntityLifecycleState,
        VisibilityState,
        SemanticHypothesis,
        PersistentEntity,
    )
except ImportError:
    # Standalone fallback if invoked outside package root
    pass


class ContextualMovability(str, Enum):
    IMMOVABLE = "IMMOVABLE"
    MOVABLE = "MOVABLE"
    MOBILE = "MOBILE"
    UNKNOWN = "UNKNOWN"


class ObservedMotionState(str, Enum):
    STATIC = "STATIC"
    DISPLACED = "DISPLACED"
    DYNAMIC = "DYNAMIC"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ContextualSceneGraphNode:
    """Persistent entity node in the contextual dynamic 3D scene graph."""
    entity_id: str
    semantic_class: str
    contextual_movability: ContextualMovability = ContextualMovability.UNKNOWN
    observed_motion: ObservedMotionState = ObservedMotionState.STATIC
    physical_support: Optional[str] = None  # "mounted_to_structure", "freestanding_on_surface", "self_propelled"
    
    # 3D spatial geometry in verified reference coordinate frame
    centroid_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    spatial_extent_m: float = 0.0
    bounding_box_min: Optional[np.ndarray] = None
    bounding_box_max: Optional[np.ndarray] = None
    coordinate_frame: str = "floor_map"
    
    # Evidence confidence & tracking
    semantic_confidence: float = 0.80
    geometric_confidence: float = 0.80
    evidence_provenance: str = "MEASURED_FULL"
    source_sessions: List[str] = field(default_factory=list)
    contributing_observation_ids: List[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    total_observations_count: int = 0
    
    # Selective router metadata
    vlm_query_requested: bool = False
    vlm_inference_executed: bool = False
    vlm_decision: str = "RESOLVE_CHEAP"
    vlm_reason_codes: List[str] = field(default_factory=list)
    vlm_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "semantic_class": self.semantic_class,
            "contextual_movability": self.contextual_movability.value if isinstance(self.contextual_movability, ContextualMovability) else str(self.contextual_movability),
            "observed_motion": self.observed_motion.value if isinstance(self.observed_motion, ObservedMotionState) else str(self.observed_motion),
            "physical_support": self.physical_support,
            "centroid_world": [round(float(x), 4) for x in self.centroid_world],
            "spatial_extent_m": round(float(self.spatial_extent_m), 4),
            "coordinate_frame": self.coordinate_frame,
            "semantic_confidence": round(float(self.semantic_confidence), 4),
            "geometric_confidence": round(float(self.geometric_confidence), 4),
            "evidence_provenance": self.evidence_provenance,
            "source_sessions": list(set(self.source_sessions)),
            "contributing_observations_count": len(self.contributing_observation_ids),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "vlm_decision": self.vlm_decision,
            "vlm_query_requested": self.vlm_query_requested,
            "vlm_reason_codes": self.vlm_reason_codes,
        }

    def to_persistent_entity(self) -> Any:
        """Convert to authoritative PersistentEntity object for scene graph integration."""
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = self.centroid_world

        ent_type = EntityType.OBJECT
        if self.contextual_movability == ContextualMovability.MOBILE or self.semantic_class in ["person", "robot"]:
            ent_type = EntityType.AGENT

        hypotheses = [
            SemanticHypothesis(
                label=self.semantic_class,
                confidence=float(self.semantic_confidence),
                source="contextual_router_fusion",
            )
        ]

        attrs = {
            "contextual_movability": self.contextual_movability.value if isinstance(self.contextual_movability, ContextualMovability) else str(self.contextual_movability),
            "observed_motion": self.observed_motion.value if isinstance(self.observed_motion, ObservedMotionState) else str(self.observed_motion),
            "physical_support": self.physical_support,
            "spatial_extent_m": self.spatial_extent_m,
            "evidence_provenance": self.evidence_provenance,
            "vlm_decision": self.vlm_decision,
            "vlm_reason_codes": self.vlm_reason_codes,
        }

        return PersistentEntity(
            entity_id=self.entity_id,
            entity_type=ent_type,
            semantic_hypotheses=hypotheses,
            pose=pose,
            uncertainty=np.eye(3, dtype=np.float64) * (1.0 - min(0.9, self.geometric_confidence)),
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            last_observed=self.last_seen,
            visibility_state=VisibilityState.VISIBLE,
            lifecycle_state=EntityLifecycleState.CONFIRMED,
            source_track_ids=list(self.contributing_observation_ids[:10]),
            attributes=attrs,
        )


class ContextualSceneGraphFusion:
    """Manages persistent contextual nodes and updates them with routed evidence."""

    def __init__(self):
        self.nodes: Dict[str, ContextualSceneGraphNode] = {}

    def update_or_create_node(
        self,
        entity_id: str,
        semantic_class: str,
        routed_result: Dict[str, Any],
        observation_meta: Optional[Dict[str, Any]] = None,
    ) -> ContextualSceneGraphNode:
        """Update existing node or instantiate a new persistent scene graph node.

        Guarantees:
        - Geometric centroid and extent are preserved even if semantic movability is UNKNOWN.
        - Movability and observed physical motion remain distinct attributes.
        - Contradictory VLM observations decay confidence or abstain without destroying geometry.
        """
        obs_meta = observation_meta or {}
        ts = float(obs_meta.get("timestamp", time.time()))
        session = obs_meta.get("source_session", "session_01")
        obs_id = obs_meta.get("obs_id", "")
        centroid = np.array(obs_meta.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float64)
        extent = float(obs_meta.get("spatial_extent_m", 0.5))

        pred_movability_str = routed_result.get("prediction", "UNKNOWN")
        if pred_movability_str in ["IMMOVABLE", "MOVABLE", "MOBILE"]:
            new_movability = ContextualMovability(pred_movability_str)
        else:
            new_movability = ContextualMovability.UNKNOWN

        # Extract motion state
        obs_motion_str = routed_result.get("observed_motion", "STATIC")
        if obs_motion_str in ["STATIC", "DISPLACED", "DYNAMIC", "UNCERTAIN"]:
            new_motion = ObservedMotionState(obs_motion_str)
        else:
            new_motion = ObservedMotionState.STATIC

        if entity_id not in self.nodes:
            # Instantiate new node
            node = ContextualSceneGraphNode(
                entity_id=entity_id,
                semantic_class=semantic_class,
                contextual_movability=new_movability,
                observed_motion=new_motion,
                physical_support=routed_result.get("physical_support"),
                centroid_world=centroid.copy(),
                spatial_extent_m=extent,
                coordinate_frame=obs_meta.get("coordinate_frame", "floor_map"),
                semantic_confidence=0.90 if new_movability != ContextualMovability.UNKNOWN else 0.40,
                geometric_confidence=0.85 if obs_meta.get("depth_valid", True) else 0.30,
                evidence_provenance=routed_result.get("evidence_provenance", "MEASURED_FULL"),
                source_sessions=[session],
                contributing_observation_ids=[obs_id] if obs_id else [],
                first_seen=ts,
                last_seen=ts,
                total_observations_count=1,
                vlm_query_requested=bool(routed_result.get("vlm_required", False)),
                vlm_inference_executed=bool(routed_result.get("vlm_inference_executed", False)),
                vlm_decision=routed_result.get("decision", "RESOLVE_CHEAP"),
                vlm_reason_codes=list(routed_result.get("reason_codes", [])),
                vlm_model=routed_result.get("vlm_model"),
            )
            self.nodes[entity_id] = node
            return node

        # Update existing node
        node = self.nodes[entity_id]
        node.last_seen = ts
        node.total_observations_count += 1
        if session not in node.source_sessions:
            node.source_sessions.append(session)
        if obs_id and obs_id not in node.contributing_observation_ids:
            node.contributing_observation_ids.append(obs_id)

        # 1. Geometric position update (running average filter)
        n = min(node.total_observations_count, 50)
        if np.any(centroid != 0.0):
            node.centroid_world = node.centroid_world + (centroid - node.centroid_world) / n
        if extent > 0.0:
            node.spatial_extent_m = node.spatial_extent_m + (extent - node.spatial_extent_m) / n

        # 2. Observed motion update
        if new_motion == ObservedMotionState.DISPLACED:
            node.observed_motion = ObservedMotionState.DISPLACED
        elif new_motion == ObservedMotionState.DYNAMIC:
            node.observed_motion = ObservedMotionState.DYNAMIC

        # 3. Contextual movability update with conflict resolution
        if new_movability != ContextualMovability.UNKNOWN:
            if node.contextual_movability == ContextualMovability.UNKNOWN:
                node.contextual_movability = new_movability
                node.semantic_confidence = 0.85
            elif node.contextual_movability == new_movability:
                node.semantic_confidence = min(0.99, node.semantic_confidence + 0.05)
            else:
                # Contradiction between observations (e.g. MOVABLE vs IMMOVABLE)
                # Decay confidence rather than violently thrashing
                node.semantic_confidence = max(0.35, node.semantic_confidence - 0.20)
                if node.semantic_confidence < 0.40:
                    node.contextual_movability = ContextualMovability.UNKNOWN

        # Update VLM routing telemetry
        if routed_result.get("vlm_required"):
            node.vlm_query_requested = True
        if routed_result.get("vlm_inference_executed"):
            node.vlm_inference_executed = True
        node.vlm_decision = routed_result.get("decision", node.vlm_decision)
        for r_code in routed_result.get("reason_codes", []):
            if r_code not in node.vlm_reason_codes:
                node.vlm_reason_codes.append(r_code)

        return node
