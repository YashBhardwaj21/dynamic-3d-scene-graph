"""Stage 9A: Cross-Session Physical Identity Association and Candidate Matching.

Solves the persistent physical object identity problem across recording sessions:
1. Transforms Session 2 floor coordinates into Session 1 reference frame via verified SE(2) registration.
2. Applies strict spatial, semantic, and geometric extent gates.
3. Computes transparent association cost: normalized Euclidean / Mahalanobis distance + extent compatibility.
4. Performs globally consistent 1-to-1 bipartite matching (Hungarian assignment) with unassigned thresholds.
5. Rejects ambiguous matches where competing candidates have overlapping spatial uncertainties.
6. Does NOT use ground-truth labels, expected decisions, or relocation labels as matching features.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from vlm.src.stage5_adapter import RegistrationInfo


@dataclass
class EntityCluster:
    """Aggregated spatial and semantic cluster for an entity within a session."""
    entity_id: str
    semantic_class: str
    source_session: str
    centroid_floor: np.ndarray  # 2D [x, y] in session local floor frame
    centroid_std: float         # 1D dispersion metric in meters
    spatial_extent_m: float     # Bounding dimension (diameter/width)
    observations_count: int
    local_track_ids: List[str]
    observation_ids: List[str]
    has_valid_pose: bool = True
    depth_valid: bool = True

    # Aligned centroid in reference coordinate frame (populated after registration)
    centroid_aligned: Optional[np.ndarray] = None
    position_uncertainty_m: float = 0.05


@dataclass
class PhysicalInstanceMatch:
    """A verified or candidate cross-session physical instance association."""
    persistent_entity_id: str
    s1_entity_id: str
    s2_entity_id: str
    semantic_class: str
    distance_m: float
    position_uncertainty_m: float
    extent_diff_m: float
    match_score: float
    confidence: float
    status: str  # "MATCHED", "AMBIGUOUS", "DISPLACED_CANDIDATE", "GATED_OUT"
    s1_obs_count: int
    s2_obs_count: int
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class AssociationResult:
    """Complete cross-session physical association report."""
    is_valid: bool
    registration_status: str
    total_s1_entities: int
    total_s2_entities: int
    matched_pairs: List[PhysicalInstanceMatch]
    unmatched_s1: List[str]
    unmatched_s2: List[str]
    ambiguous_pairs: List[PhysicalInstanceMatch]
    diagnostics: List[str] = field(default_factory=list)


class CrossSessionCandidateMatcher:
    """Estimator-aware cross-session physical identity matcher."""

    def __init__(
        self,
        max_static_distance_m: float = 0.45,
        max_extent_relative_diff: float = 0.60,
        ambiguity_margin_m: float = 0.15,
        min_cluster_observations: int = 2,
    ):
        self.max_static_distance_m = max_static_distance_m
        self.max_extent_relative_diff = max_extent_relative_diff
        self.ambiguity_margin_m = ambiguity_margin_m
        self.min_cluster_observations = min_cluster_observations

    @staticmethod
    def extract_clusters_from_observations(
        observations: List[Dict[str, Any]],
        session_keyword: str,
        min_obs: int = 1,
    ) -> Dict[str, EntityCluster]:
        """Aggregate raw frame observations into robust entity clusters."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for o in observations:
            sess = o.get("source_session", "")
            if session_keyword not in sess:
                continue
            eid = o.get("global_entity_id", "")
            if not eid:
                continue
            cf = o.get("centroid_floor")
            if cf is not None and len(cf) >= 2 and all(x is not None for x in cf[:2]):
                grouped.setdefault(eid, []).append(o)

        clusters: Dict[str, EntityCluster] = {}
        for eid, obs_list in grouped.items():
            if len(obs_list) < min_obs:
                continue

            pts = np.array([o["centroid_floor"][:2] for o in obs_list], dtype=np.float64)
            c = np.median(pts, axis=0)
            dispersion = float(np.mean(np.std(pts, axis=0))) if len(pts) > 1 else 0.05
            sem_class = obs_list[0].get("class_name") or obs_list[0].get("semantic_class", "object")

            # Extract spatial extents if bounding boxes exist
            extents = []
            for o in obs_list:
                bmin = o.get("bbox_min_floor")
                bmax = o.get("bbox_max_floor")
                if bmin and bmax and len(bmin) >= 2 and len(bmax) >= 2:
                    extents.append(float(np.linalg.norm(np.array(bmax[:2]) - np.array(bmin[:2]))))
            avg_extent = float(np.mean(extents)) if extents else 0.50

            track_ids = list(set(o.get("local_track_id") for o in obs_list if o.get("local_track_id")))
            obs_ids = [o.get("obs_id") for o in obs_list if o.get("obs_id")]
            has_pose = all(o.get("has_valid_pose", False) for o in obs_list)
            depth_valid = any(o.get("depth_valid", True) for o in obs_list)

            # SEM of centroid: dispersion / sqrt(N)
            sem_pos = dispersion / math.sqrt(len(pts))

            clusters[eid] = EntityCluster(
                entity_id=eid,
                semantic_class=sem_class,
                source_session=session_keyword,
                centroid_floor=c,
                centroid_std=dispersion,
                spatial_extent_m=avg_extent,
                observations_count=len(obs_list),
                local_track_ids=track_ids,
                observation_ids=obs_ids,
                has_valid_pose=has_pose,
                depth_valid=depth_valid,
                position_uncertainty_m=sem_pos,
            )

        return clusters

    def match_entities(
        self,
        s1_clusters: Dict[str, EntityCluster],
        s2_clusters: Dict[str, EntityCluster],
        registration: RegistrationInfo,
    ) -> AssociationResult:
        """Globally match Session 1 and Session 2 entity clusters.

        Strict Gates:
        - Registration must be valid and verified.
        - Semantic classes must match.
        - Distance in common reference frame must satisfy gating.
        - Spatial extent change must be physically plausible.
        """
        diagnostics: List[str] = []
        if not registration.is_valid:
            diagnostics.append(f"Registration invalid or unverified: {registration.status}")
            return AssociationResult(
                is_valid=False,
                registration_status=registration.status,
                total_s1_entities=len(s1_clusters),
                total_s2_entities=len(s2_clusters),
                matched_pairs=[],
                unmatched_s1=list(s1_clusters.keys()),
                unmatched_s2=list(s2_clusters.keys()),
                ambiguous_pairs=[],
                diagnostics=diagnostics,
            )

        # 1. Transform Session 2 centroids into Session 1 reference frame
        # c1 = R_12 @ c2 + t_12
        reg_sem = registration.anchor_rms_residual_m / math.sqrt(max(1, registration.anchor_count))

        s2_aligned: Dict[str, EntityCluster] = {}
        for eid2, c2 in s2_clusters.items():
            c2_copy = copy.copy(c2)
            c2_copy.centroid_aligned = registration.R_12 @ c2.centroid_floor + registration.translation_m
            # Combined uncertainty: sqrt(sem_pos^2 + reg_sem^2)
            c2_copy.position_uncertainty_m = math.sqrt(c2.position_uncertainty_m**2 + reg_sem**2)
            s2_aligned[eid2] = c2_copy

        s1_keys = list(s1_clusters.keys())
        s2_keys = list(s2_aligned.keys())
        n1 = len(s1_keys)
        n2 = len(s2_keys)

        if n1 == 0 or n2 == 0:
            return AssociationResult(
                is_valid=True,
                registration_status=registration.status,
                total_s1_entities=n1,
                total_s2_entities=n2,
                matched_pairs=[],
                unmatched_s1=s1_keys,
                unmatched_s2=s2_keys,
                ambiguous_pairs=[],
                diagnostics=["One or both sessions contain zero clusters."],
            )

        # 2. Build cost matrix with infinite penalty for gated-out pairs
        INFTY_COST = 1e6
        cost_matrix = np.full((n1, n2), INFTY_COST, dtype=np.float64)
        distance_matrix = np.full((n1, n2), INFTY_COST, dtype=np.float64)

        for i, k1 in enumerate(s1_keys):
            c1 = s1_clusters[k1]
            for j, k2 in enumerate(s2_keys):
                c2 = s2_aligned[k2]

                # Gate A: Semantic class compatibility
                if c1.semantic_class.lower().strip() != c2.semantic_class.lower().strip():
                    continue

                # Gate B: Spatial distance in common reference frame
                dist = float(np.linalg.norm(c1.centroid_floor - c2.centroid_aligned))
                distance_matrix[i, j] = dist

                if dist > self.max_static_distance_m:
                    continue

                # Gate C: Relative spatial extent change
                max_ext = max(c1.spatial_extent_m, c2.spatial_extent_m, 0.10)
                ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m)
                rel_ext = ext_diff / max_ext
                if rel_ext > self.max_extent_relative_diff:
                    continue

                # Transparent cost: normalized distance + extent discrepancy
                combined_unc = math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2)
                norm_dist_cost = dist / max(0.05, combined_unc)
                cost = norm_dist_cost + 2.0 * rel_ext
                cost_matrix[i, j] = cost

        # 3. Optimal Global 1-to-1 Bipartite Matching (Hungarian Algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: List[PhysicalInstanceMatch] = []
        ambiguous_pairs: List[PhysicalInstanceMatch] = []
        matched_s1_idx: Set[int] = set()
        matched_s2_idx: Set[int] = set()

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost >= INFTY_COST:
                continue

            dist = distance_matrix[r, c]
            k1 = s1_keys[r]
            k2 = s2_keys[c]
            c1 = s1_clusters[k1]
            c2 = s2_aligned[k2]

            # Ambiguity Check: Does c2 have another competing candidate in S1 within ambiguity margin?
            competing_distances_for_c2 = [
                distance_matrix[other_r, c] for other_r in range(n1)
                if other_r != r and distance_matrix[other_r, c] < INFTY_COST
            ]
            is_ambiguous = False
            if competing_distances_for_c2:
                min_other = min(competing_distances_for_c2)
                if abs(min_other - dist) < self.ambiguity_margin_m:
                    is_ambiguous = True

            combined_unc = math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2)
            ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m)
            confidence = max(0.20, min(0.99, 1.0 - (dist / self.max_static_distance_m) * 0.5 - (cost / 10.0) * 0.3))

            match_obj = PhysicalInstanceMatch(
                persistent_entity_id=f"phys_{k1}",
                s1_entity_id=k1,
                s2_entity_id=k2,
                semantic_class=c1.semantic_class,
                distance_m=round(dist, 4),
                position_uncertainty_m=round(combined_unc, 4),
                extent_diff_m=round(ext_diff, 4),
                match_score=round(float(cost), 4),
                confidence=round(confidence, 4),
                status="AMBIGUOUS" if is_ambiguous else "MATCHED",
                s1_obs_count=c1.observations_count,
                s2_obs_count=c2.observations_count,
            )

            if is_ambiguous:
                match_obj.diagnostics.append(f"Ambiguous match: another S1 candidate exists within {self.ambiguity_margin_m}m margin.")
                ambiguous_pairs.append(match_obj)
            else:
                matched_pairs.append(match_obj)
                matched_s1_idx.add(r)
                matched_s2_idx.add(c)

        unmatched_s1 = [s1_keys[i] for i in range(n1) if i not in matched_s1_idx]
        unmatched_s2 = [s2_keys[j] for j in range(n2) if j not in matched_s2_idx]

        return AssociationResult(
            is_valid=True,
            registration_status=registration.status,
            total_s1_entities=n1,
            total_s2_entities=n2,
            matched_pairs=matched_pairs,
            unmatched_s1=unmatched_s1,
            unmatched_s2=unmatched_s2,
            ambiguous_pairs=ambiguous_pairs,
            diagnostics=diagnostics,
        )


def evaluate_matcher_against_ground_truth(
    association_result: AssociationResult,
    gt_path: Path,
) -> Dict[str, Any]:
    """Rigorously evaluate cross-session physical identity matches against benchmark ground truth."""
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Build GT cross-session physical pair lookup
    gt_objects = gt_data.get("physical_objects", [])
    gt_cross_session_pairs: Dict[Tuple[str, str], str] = {}
    gt_s1_to_inst: Dict[str, str] = {}
    gt_s2_to_inst: Dict[str, str] = {}

    for obj in gt_objects:
        inst_id = obj["object_instance_id"]
        tracks = obj.get("associated_tracks", [])
        s1_tracks = [t for t in tracks if not t.endswith("_s2")]
        s2_tracks = [t.replace("_s2", "") for t in tracks if t.endswith("_s2")]

        for t1 in s1_tracks:
            gt_s1_to_inst[t1] = inst_id
        for t2 in s2_tracks:
            gt_s2_to_inst[t2] = inst_id

        for t1 in s1_tracks:
            for t2 in s2_tracks:
                gt_cross_session_pairs[(t1, t2)] = inst_id

    # Evaluate matched pairs
    true_positive_matches = 0
    false_positive_matches = 0
    false_merges = 0

    evaluated_matches = []
    for m in association_result.matched_pairs:
        t1 = m.s1_entity_id
        t2 = m.s2_entity_id
        inst1 = gt_s1_to_inst.get(t1)
        inst2 = gt_s2_to_inst.get(t2)

        is_true_match = (inst1 is not None and inst2 is not None and inst1 == inst2)
        if is_true_match:
            true_positive_matches += 1
        else:
            false_positive_matches += 1
            if inst1 is not None and inst2 is not None and inst1 != inst2:
                false_merges += 1

        evaluated_matches.append({
            "s1_entity": t1,
            "s2_entity": t2,
            "s1_gt_instance": inst1,
            "s2_gt_instance": inst2,
            "distance_m": m.distance_m,
            "is_correct": is_true_match,
        })

    total_gt_pairs = len(gt_cross_session_pairs)
    precision = (true_positive_matches / max(1, len(association_result.matched_pairs))) * 100.0
    recall = (true_positive_matches / max(1, total_gt_pairs)) * 100.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "total_matched_pairs": len(association_result.matched_pairs),
        "true_positive_matches": true_positive_matches,
        "false_positive_matches": false_positive_matches,
        "false_merges": false_merges,
        "total_ground_truth_cross_session_pairs": total_gt_pairs,
        "association_precision_pct": round(precision, 2),
        "association_recall_pct": round(recall, 2),
        "association_f1_pct": round(f1, 2),
        "evaluated_matches": evaluated_matches,
    }
