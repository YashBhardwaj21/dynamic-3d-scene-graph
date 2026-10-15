"""Stage 6: Estimator-Aware & Baseline Router Policies.

Implements five distinct, reproducible routing policies for empirical comparison:
- R0: Semantic-class prior lookup only (zero VLM queries).
- R1: Geometry and observed motion only without VLM (zero VLM queries).
- R2: Non-estimator-aware selective routing (ambiguity-driven, blind to SLAM influence).
- R3: Estimator-aware selective routing (explicit SLAM loss reduction utility vs VLM cost).
- R4: All-eligible VLM reference (unguided contextual reasoning ceiling).

Core formulation for R3 (Estimator-Aware):
    query_utility = delta_L_SLAM - lambda_cost * C_vlm
where delta_L_SLAM quantifies the expected SLAM trajectory loss reduction if an ambiguous
or displaced object's true affordance is resolved to prevent map corruption and false loop closures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


DecisionType = Literal["RESOLVE_CHEAP", "ESCALATE", "DEFER", "ABSTAIN"]

CANONICAL_IMMOVABLE = {
    "door", "wall", "ceiling fan", "window", "floor", "ceiling", "pillar"
}

CANONICAL_MOBILE = {
    "person", "robot", "drone", "quadcopter drone", "human", "operator", "pet"
}

CANONICAL_MOVABLE = {
    "chair", "desk", "table", "laptop", "computer monitor", "monitor",
    "backpack", "handbag", "storage box", "keyboard", "mouse", "computer mouse",
    "bottle", "cup", "trash can", "toolbox"
}

# Ambiguous categories requiring visual affordance verification (mounted vs freestanding)
AMBIGUOUS_AFFORDANCE_CLASSES = {
    "pedestal fan", "fan", "whiteboard", "projector", "screen", "shelf", "cabinet"
}

MIN_SIGNIFICANT_DISPLACEMENT_M = 0.25
ENTROPY_AMBIGUITY_THRESHOLD = 0.35
DEFAULT_VLM_COST = 0.50


@dataclass
class RouterDecision:
    """Outcome contract for an evaluated evidence record."""
    entity_id: str
    decision: DecisionType
    reason_codes: List[str]
    vlm_required: bool
    cheap_label: Optional[str] = None
    confidence: float = 1.0
    query_utility: Optional[float] = None
    estimator_influence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "decision": self.decision,
            "reason_codes": self.reason_codes,
            "vlm_required": self.vlm_required,
            "cheap_label": self.cheap_label,
            "confidence": round(self.confidence, 4),
            "query_utility": round(self.query_utility, 4) if self.query_utility is not None else None,
            "estimator_influence": round(self.estimator_influence, 4) if self.estimator_influence is not None else None,
        }


def compute_estimator_influence(evidence: Dict[str, Any]) -> float:
    """Compute normalized SLAM landmark influence score in [0.0, 1.0].

    Derived from:
    - Total observation support (s1_obs_count + s2_obs_count or total_observations)
    - Geometric consistency score
    - Valid pose stability
    """
    has_valid_pose = evidence.get("has_valid_pose", True)
    if not has_valid_pose:
        return 0.0

    s1_cnt = int(evidence.get("s1_obs_count", 0))
    s2_cnt = int(evidence.get("s2_obs_count", 0))
    total_obs = s1_cnt + s2_cnt
    if total_obs == 0:
        total_obs = int(evidence.get("total_observations", evidence.get("views_count", 1)))

    # Observation support factor: saturation at 10 observations
    obs_factor = min(1.0, total_obs / 10.0)

    # Geometric consistency factor in [0.5, 1.0]
    geo_score = float(evidence.get("geometric_consistency_score", 0.90) or 0.90)
    geo_factor = max(0.2, min(1.0, geo_score))

    # Parallax / displacement presence increases state-estimation sensitivity
    world_disp = evidence.get("world_displacement_m")
    disp_factor = 1.2 if (world_disp is not None and world_disp > 0.25) else 1.0

    influence = min(1.0, obs_factor * geo_factor * disp_factor)
    return round(influence, 4)


def evaluate_policy_r0(evidence: Dict[str, Any]) -> RouterDecision:
    """R0: Semantic-Class Prior Lookup Only (Zero VLM queries)."""
    entity_id = str(evidence.get("entity_id", "unknown"))
    sem_class = str(evidence.get("semantic_class", "object")).lower().strip()
    prior = evidence.get("class_prior_mobility")

    if sem_class in CANONICAL_MOBILE or prior == "MOBILE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["R0_PRIOR_CANONICAL_MOBILE"],
            vlm_required=False,
            cheap_label="MOBILE",
            confidence=0.98,
        )
    if sem_class in CANONICAL_IMMOVABLE or prior == "IMMOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["R0_PRIOR_CANONICAL_IMMOVABLE"],
            vlm_required=False,
            cheap_label="IMMOVABLE",
            confidence=0.99,
        )
    if sem_class in CANONICAL_MOVABLE or prior == "MOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["R0_PRIOR_CANONICAL_MOVABLE"],
            vlm_required=False,
            cheap_label="MOVABLE",
            confidence=0.95,
        )
    return RouterDecision(
        entity_id=entity_id,
        decision="ABSTAIN",
        reason_codes=["R0_PRIOR_UNKNOWN_CATEGORY"],
        vlm_required=False,
        cheap_label="UNKNOWN",
        confidence=0.0,
    )


def evaluate_policy_r1(evidence: Dict[str, Any]) -> RouterDecision:
    """R1: Geometry & Temporal Motion Only (Zero VLM queries)."""
    entity_id = str(evidence.get("entity_id", "unknown"))
    sem_class = str(evidence.get("semantic_class", "object")).lower().strip()
    observed_motion = str(evidence.get("observed_motion", "STATIC")).upper()
    world_disp = evidence.get("world_displacement_m")

    # If observed moving, classify passively movable objects as MOVABLE and agents as MOBILE
    if observed_motion in ["MOVING", "DISPLACED"] or (world_disp and world_disp > 0.25):
        if sem_class in CANONICAL_MOBILE:
            label = "MOBILE"
        else:
            label = "MOVABLE"
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["R1_GEOMETRY_MOTION_DISPLACED"],
            vlm_required=False,
            cheap_label=label,
            confidence=0.90,
        )

    # Static: fallback to prior
    return evaluate_policy_r0(evidence)


def evaluate_policy_r2(evidence: Dict[str, Any]) -> RouterDecision:
    """R2: Non-Estimator-Aware Selective Routing (Ambiguity-Driven)."""
    entity_id = str(evidence.get("entity_id", "unknown"))
    sem_class = str(evidence.get("semantic_class", "object")).lower().strip()
    prior_mobility = evidence.get("class_prior_mobility")
    observed_motion = str(evidence.get("observed_motion", "STATIC")).upper()
    world_disp = evidence.get("world_displacement_m")
    disp_sigma = evidence.get("displacement_uncertainty_m", 0.04) or 0.04
    semantic_entropy = float(evidence.get("semantic_entropy", 0.0) or 0.0)
    has_valid_pose = evidence.get("has_valid_pose")
    has_valid_depth = evidence.get("depth_valid")
    has_images = bool(evidence.get("has_images", False))
    views_count = int(evidence.get("views_count", 0))

    # Gate 1: Estimator & Evidence Validity
    if has_valid_pose is False:
        return RouterDecision(
            entity_id=entity_id,
            decision="DEFER",
            reason_codes=["UNTRUSTWORTHY_ESTIMATOR_POSE"],
            vlm_required=False,
            cheap_label=None,
            confidence=0.0,
        )

    if has_valid_depth is False and observed_motion in ["MOVING", "DISPLACED"]:
        return RouterDecision(
            entity_id=entity_id,
            decision="DEFER",
            reason_codes=["INVALID_DEPTH_ON_MOVING_TARGET"],
            vlm_required=False,
            cheap_label=None,
            confidence=0.0,
        )

    # Mobile agents resolved cheaply
    if sem_class in CANONICAL_MOBILE or prior_mobility == "MOBILE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["CANONICAL_MOBILE_AGENT_PRIOR"],
            vlm_required=False,
            cheap_label="MOBILE",
            confidence=0.98,
        )

    reasons: List[str] = []

    # Significant displacement
    if world_disp is not None:
        is_displaced = (
            world_disp >= MIN_SIGNIFICANT_DISPLACEMENT_M
            and world_disp >= 3.0 * max(0.02, disp_sigma)
            and observed_motion in ["DISPLACED", "MOVING"]
        )
        if is_displaced:
            reasons.append("SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED")
    elif observed_motion in ["DISPLACED", "MOVING"]:
        reasons.append("SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED")

    # Ambiguous mounting class
    if sem_class in AMBIGUOUS_AFFORDANCE_CLASSES:
        reasons.append("AMBIGUOUS_PHYSICAL_SUPPORT_AFFORDANCE")

    # High semantic entropy
    if semantic_entropy >= ENTROPY_AMBIGUITY_THRESHOLD:
        reasons.append("HIGH_SEMANTIC_AMBIGUITY")

    if reasons:
        if not has_images or views_count <= 0:
            return RouterDecision(
                entity_id=entity_id,
                decision="DEFER",
                reason_codes=reasons + ["MISSING_REQUIRED_VISUAL_CONTEXT"],
                vlm_required=False,
                cheap_label=None,
                confidence=0.30,
            )
        return RouterDecision(
            entity_id=entity_id,
            decision="ESCALATE",
            reason_codes=reasons,
            vlm_required=True,
            cheap_label=None,
            confidence=0.90,
        )

    # Stationary immovable
    if sem_class in CANONICAL_IMMOVABLE or prior_mobility == "IMMOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["CANONICAL_IMMOVABLE_ARCHITECTURAL_PRIOR"],
            vlm_required=False,
            cheap_label="IMMOVABLE",
            confidence=0.99,
        )

    # Stationary movable
    if sem_class in CANONICAL_MOVABLE or prior_mobility == "MOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["STATIC_CANONICAL_MOVABLE_PRIOR"],
            vlm_required=False,
            cheap_label="MOVABLE",
            confidence=0.95,
        )

    return evaluate_policy_r0(evidence)


def evaluate_policy_r3(
    evidence: Dict[str, Any],
    lambda_cost: float = DEFAULT_VLM_COST,
) -> RouterDecision:
    """R3: Estimator-Aware Selective Routing.

    Evaluates:
        query_utility = delta_L_SLAM - lambda_cost * C_VLM
    where delta_L_SLAM is proportional to the object's landmark influence on the
    SLAM state estimator and the ambiguity of its physical relocation state.
    """
    entity_id = str(evidence.get("entity_id", "unknown"))
    sem_class = str(evidence.get("semantic_class", "object")).lower().strip()
    prior_mobility = evidence.get("class_prior_mobility")
    observed_motion = str(evidence.get("observed_motion", "STATIC")).upper()
    world_disp = evidence.get("world_displacement_m")
    disp_sigma = evidence.get("displacement_uncertainty_m", 0.04) or 0.04
    semantic_entropy = float(evidence.get("semantic_entropy", 0.0) or 0.0)

    has_valid_pose = evidence.get("has_valid_pose")
    has_valid_depth = evidence.get("depth_valid")
    has_images = bool(evidence.get("has_images", False))
    views_count = int(evidence.get("views_count", 0))

    # Gate 1: Estimator & Evidence Validity
    if has_valid_pose is False:
        return RouterDecision(
            entity_id=entity_id,
            decision="DEFER",
            reason_codes=["UNTRUSTWORTHY_ESTIMATOR_POSE"],
            vlm_required=False,
            cheap_label=None,
            confidence=0.0,
            query_utility=-1.0,
            estimator_influence=0.0,
        )

    if has_valid_depth is False and observed_motion in ["MOVING", "DISPLACED"]:
        return RouterDecision(
            entity_id=entity_id,
            decision="DEFER",
            reason_codes=["INVALID_DEPTH_ON_MOVING_TARGET"],
            vlm_required=False,
            cheap_label=None,
            confidence=0.0,
            query_utility=-1.0,
            estimator_influence=0.0,
        )

    # Calculate explicit estimator influence signal
    est_influence = compute_estimator_influence(evidence)

    # Canonical mobile agents resolved cheaply without VLM
    if sem_class in CANONICAL_MOBILE or prior_mobility == "MOBILE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["CANONICAL_MOBILE_AGENT_PRIOR"],
            vlm_required=False,
            cheap_label="MOBILE",
            confidence=0.98,
            query_utility=0.0,
            estimator_influence=est_influence,
        )

    # Determine relocation and affordance ambiguity factors
    ambiguity_factors = []
    is_displaced = False
    if world_disp is not None:
        is_displaced = (
            world_disp >= MIN_SIGNIFICANT_DISPLACEMENT_M
            and world_disp >= 3.0 * max(0.02, disp_sigma)
            and observed_motion in ["DISPLACED", "MOVING"]
        )
        if is_displaced:
            ambiguity_factors.append("SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED")
    elif observed_motion in ["DISPLACED", "MOVING"]:
        is_displaced = True
        ambiguity_factors.append("SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED")

    if sem_class in AMBIGUOUS_AFFORDANCE_CLASSES:
        ambiguity_factors.append("AMBIGUOUS_PHYSICAL_SUPPORT_AFFORDANCE")

    if semantic_entropy >= ENTROPY_AMBIGUITY_THRESHOLD:
        ambiguity_factors.append("HIGH_SEMANTIC_AMBIGUITY")

    # If evidence is ambiguous, calculate expected SLAM loss reduction:
    # A displaced object with high landmark influence poses severe loop-closure distortion risk.
    if ambiguity_factors:
        # Base expected loss reduction scaled by estimator influence
        # delta_L_SLAM = base_risk * estimator_influence
        base_risk = 1.0 if is_displaced else 0.8
        delta_L_slam = base_risk * est_influence

        # Query utility = delta_L_slam - lambda_cost * C_vlm (where C_vlm = 1.0 normalized)
        utility = delta_L_slam - (lambda_cost * 1.0)

        # Check visual context availability
        if not has_images or views_count <= 0:
            return RouterDecision(
                entity_id=entity_id,
                decision="DEFER",
                reason_codes=ambiguity_factors + ["MISSING_REQUIRED_VISUAL_CONTEXT"],
                vlm_required=False,
                cheap_label=None,
                confidence=0.30,
                query_utility=round(utility, 4),
                estimator_influence=est_influence,
            )

        # Estimator-aware gating:
        # If estimator influence is non-zero (landmarks actively inform SLAM trajectory),
        # escalate to protect SLAM consistency.
        # If estimator influence is zero (e.g. unreferenced background feature), avoid query.
        if est_influence > 0.0:
            return RouterDecision(
                entity_id=entity_id,
                decision="ESCALATE",
                reason_codes=ambiguity_factors + ["HIGH_ESTIMATOR_RELEVANCE"],
                vlm_required=True,
                cheap_label=None,
                confidence=0.90,
                query_utility=round(utility, 4),
                estimator_influence=est_influence,
            )
        else:
            return RouterDecision(
                entity_id=entity_id,
                decision="RESOLVE_CHEAP",
                reason_codes=ambiguity_factors + ["ZERO_ESTIMATOR_RELEVANCE_QUERY_AVOIDED"],
                vlm_required=False,
                cheap_label="MOVABLE" if is_displaced else "IMMOVABLE",
                confidence=0.60,
                query_utility=round(utility, 4),
                estimator_influence=est_influence,
            )

    # Stationary architectural elements
    if sem_class in CANONICAL_IMMOVABLE or prior_mobility == "IMMOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["CANONICAL_IMMOVABLE_ARCHITECTURAL_PRIOR"],
            vlm_required=False,
            cheap_label="IMMOVABLE",
            confidence=0.99,
            query_utility=0.0,
            estimator_influence=est_influence,
        )

    # Stationary movable furniture
    if sem_class in CANONICAL_MOVABLE or prior_mobility == "MOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["STATIC_CANONICAL_MOVABLE_PRIOR"],
            vlm_required=False,
            cheap_label="MOVABLE",
            confidence=0.95,
            query_utility=0.0,
            estimator_influence=est_influence,
        )

    # Unknown category: escalate if images available and influence exists
    if has_images and views_count > 0:
        return RouterDecision(
            entity_id=entity_id,
            decision="ESCALATE",
            reason_codes=["UNKNOWN_CATEGORY_ESCALATION"],
            vlm_required=True,
            cheap_label=None,
            confidence=0.50,
            query_utility=round(est_influence - lambda_cost, 4),
            estimator_influence=est_influence,
        )

    return RouterDecision(
        entity_id=entity_id,
        decision="ABSTAIN",
        reason_codes=["UNKNOWN_CATEGORY_NO_IMAGES"],
        vlm_required=False,
        cheap_label="UNKNOWN",
        confidence=0.0,
        query_utility=-1.0,
        estimator_influence=est_influence,
    )


def evaluate_policy_r4(evidence: Dict[str, Any]) -> RouterDecision:
    """R4: All-Eligible VLM Reference Policy."""
    entity_id = str(evidence.get("entity_id", "unknown"))
    has_images = bool(evidence.get("has_images", False))
    views_count = int(evidence.get("views_count", 0))

    if has_images and views_count > 0:
        return RouterDecision(
            entity_id=entity_id,
            decision="ESCALATE",
            reason_codes=["R4_ALL_ELIGIBLE_VLM"],
            vlm_required=True,
            cheap_label=None,
            confidence=0.90,
        )

    return RouterDecision(
        entity_id=entity_id,
        decision="DEFER",
        reason_codes=["R4_MISSING_IMAGES"],
        vlm_required=False,
        cheap_label="UNKNOWN",
        confidence=0.0,
    )


def evaluate_router_policy(
    evidence: Dict[str, Any],
    policy: Literal["R0", "R1", "R2", "R3", "R4"] = "R3",
    lambda_cost: float = DEFAULT_VLM_COST,
) -> RouterDecision:
    """Unified policy evaluator defaulting to R3 (Estimator-Aware)."""
    if policy == "R0":
        return evaluate_policy_r0(evidence)
    elif policy == "R1":
        return evaluate_policy_r1(evidence)
    elif policy == "R2":
        return evaluate_policy_r2(evidence)
    elif policy == "R4":
        return evaluate_policy_r4(evidence)
    return evaluate_policy_r3(evidence, lambda_cost=lambda_cost)
