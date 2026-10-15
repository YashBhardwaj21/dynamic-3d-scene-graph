"""Stage 6: Stage 5 to Stage 6 Evidence Adapter.

Constructs genuine, instance-level evidence records strictly from Stage 5 outputs
and recorded observations without touching ground-truth target annotations.

Ensures zero ground-truth leakage into the selective router.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class Stage5EvidenceAdapter:
    """Extracts instance-level estimated evidence from Stage 5 outputs for routing."""

    def __init__(self, repo_root: Optional[Path] = None):
        if repo_root is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
        self.repo_root = repo_root

        self.fused_records_by_entity: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.obs_by_id: Dict[str, Dict[str, Any]] = {}
        self.s1_points_by_entity: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.s2_points_by_entity: Dict[str, List[np.ndarray]] = defaultdict(list)

        # Load SE(2) cross-session registration
        audit_file = self.repo_root / "outputs" / "semantic_mapping_fused_desk_run" / "audit_summary.json"
        theta_deg = -16.13
        t_vec = np.array([-0.0815, 0.0930], dtype=np.float64)
        if audit_file.exists():
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    reg = json.load(f).get("registration", {})
                    theta_deg = reg.get("rotation_deg", -16.13)
                    t_vec = np.array(reg.get("translation_m", [-0.0815, 0.0930]), dtype=np.float64)
            except Exception:
                pass

        rad = math.radians(theta_deg)
        self.R_12 = np.array([
            [math.cos(rad), -math.sin(rad)],
            [math.sin(rad),  math.cos(rad)],
        ], dtype=np.float64)
        self.t_12 = t_vec

        self._load_stage5_data()

    def _load_stage5_data(self):
        fused_path = self.repo_root / "vlm" / "outputs" / "stage5_fused_evidence.jsonl"
        if fused_path.exists():
            with open(fused_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    eid = rec.get("entity_id", "")
                    if eid:
                        self.fused_records_by_entity[eid].append(rec)

        obs_path = self.repo_root / "vlm" / "outputs" / "object_observations.jsonl"
        if obs_path.exists():
            with open(obs_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    o = json.loads(line)
                    oid = o.get("obs_id", "")
                    if oid:
                        self.obs_by_id[oid] = o

                    eid = o.get("global_entity_id", "")
                    sess = o.get("source_session", "")
                    c_floor = o.get("centroid_floor")
                    if eid and c_floor and len(c_floor) >= 2:
                        pt_xy = np.array(c_floor[:2], dtype=np.float64)
                        if "recording_01" in sess:
                            self.s1_points_by_entity[eid].append(pt_xy)
                        elif "recording_02" in sess:
                            # Transform S2 points into reference S1 floor frame
                            pt_trans = self.R_12 @ pt_xy + self.t_12
                            self.s2_points_by_entity[eid].append(pt_trans)

    def build_instance_evidence(self, manifest_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Produce an instance-level evidence dictionary strictly from estimated features.

        IMPORTANT: Excludes ground-truth fields ('contextual_movability_gt',
        'observed_motion_gt', 'expected_decision', 'expected_reason') to eliminate leakage.
        """
        entity_id = manifest_entry["entity_id"]
        sem_class = manifest_entry.get("semantic_class", "object")
        s1_obs_id = manifest_entry.get("s1_obs_id")
        s2_obs_id = manifest_entry.get("s2_obs_id")

        # Gather relevant fused records
        fused_matches: List[Dict[str, Any]] = []
        for tid in [s1_obs_id, s2_obs_id]:
            if tid and tid in self.fused_records_by_entity:
                fused_matches.extend(self.fused_records_by_entity[tid])

        # 1. Coordinate & Pose Validity
        has_valid_pose = True
        depth_valid = True
        provenances = set()
        entropy_vals = []
        purity_vals = []
        geo_scores = []

        if fused_matches:
            has_valid_pose = any(
                m.get("router_summary", {}).get("evidence_complete", False)
                or m.get("has_valid_evidence", False)
                for m in fused_matches
            )
            depth_valid = any(m.get("geometry", {}).get("depth_valid", False) for m in fused_matches)
            for m in fused_matches:
                prov = m.get("geometry", {}).get("provenance")
                if prov:
                    provenances.add(prov)
                ent = m.get("semantics", {}).get("semantic_entropy")
                if ent is not None:
                    entropy_vals.append(ent)
                pur = m.get("semantics", {}).get("semantic_purity")
                if pur is not None:
                    purity_vals.append(pur)
                gs = m.get("geometry", {}).get("geometric_consistency_score")
                if gs is not None:
                    geo_scores.append(gs)
        else:
            # If no direct fused matches, check if observations exist
            has_obs = (s1_obs_id in self.obs_by_id) or (s2_obs_id in self.obs_by_id)
            if not has_obs and sem_class in ["person", "robot"]:
                has_valid_pose = True
                depth_valid = True
            elif not has_obs:
                has_valid_pose = False
                depth_valid = False

        # 2. Estimate Physical Displacement from actual observation clusters
        s1_pts = self.s1_points_by_entity.get(s1_obs_id, []) if s1_obs_id else []
        s2_pts = self.s2_points_by_entity.get(s2_obs_id, []) if s2_obs_id else []

        estimated_disp_m = 0.0
        disp_sigma_m = 0.04
        estimated_motion = "STATIC"

        if s1_pts and s2_pts:
            # Multi-session physical relocation estimate
            c1 = np.median(s1_pts, axis=0)
            c2 = np.median(s2_pts, axis=0)
            std1 = float(np.mean(np.std(s1_pts, axis=0))) if len(s1_pts) > 1 else 0.02
            std2 = float(np.mean(np.std(s2_pts, axis=0))) if len(s2_pts) > 1 else 0.02
            estimated_disp_m = float(np.linalg.norm(c2 - c1))
            disp_sigma_m = float(math.sqrt(std1**2 + std2**2))

            if estimated_disp_m >= max(0.25, 3.0 * disp_sigma_m):
                estimated_motion = "DISPLACED"
            else:
                estimated_motion = "STATIC"
        elif fused_matches:
            # Intra-session observed motion
            disps = [m["temporal"]["world_displacement_m"] for m in fused_matches if "temporal" in m]
            motions = [m["temporal"]["current_motion"] for m in fused_matches if "temporal" in m]
            estimated_disp_m = float(np.max(disps)) if disps else 0.0
            if "MOVING" in motions:
                estimated_motion = "MOVING"
            elif "DISPLACED" in motions:
                estimated_motion = "DISPLACED"
            else:
                estimated_motion = "STATIC"
        elif sem_class in ["person", "robot"]:
            estimated_motion = "DYNAMIC"

        mean_entropy = float(np.mean(entropy_vals)) if entropy_vals else 0.0
        mean_purity = float(np.mean(purity_vals)) if purity_vals else 1.0
        mean_geo_score = float(np.mean(geo_scores)) if geo_scores else 0.0
        primary_prov = list(provenances)[0] if provenances else ("MEASURED_FULL" if depth_valid else "MISSING")

        # Image availability
        has_images = bool(manifest_entry.get("has_images", False))
        views_count = int(manifest_entry.get("views_count", 0))
        crops_dir = manifest_entry.get("crops_dir")

        return {
            "entity_id": entity_id,
            "semantic_class": sem_class,
            "observed_motion": estimated_motion,
            "world_displacement_m": round(estimated_disp_m, 4),
            "displacement_uncertainty_m": round(disp_sigma_m, 4),
            "depth_valid": depth_valid,
            "has_valid_pose": has_valid_pose,
            "has_images": has_images,
            "views_count": views_count,
            "crops_dir": crops_dir,
            "semantic_entropy": round(mean_entropy, 4),
            "semantic_purity": round(mean_purity, 4),
            "geometric_consistency_score": round(mean_geo_score, 4),
            "provenance": primary_prov,
            "s1_obs_count": len(s1_pts),
            "s2_obs_count": len(s2_pts),
        }
