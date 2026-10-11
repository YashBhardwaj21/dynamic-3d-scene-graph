"""Natural Language Prompt to Navigation Goal Resolver.

Transforms instructions (e.g. "Go to the table with the drone and stop 0.5 metres away",
"Go to chair_003", "Approach the desk and stop 0.4 metres away") into validated,
collision-free robot goal poses in the floor-aligned coordinate frame.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GoalResolutionResult:
    """Structured resolution result containing target entity and approach pose."""
    success: bool
    prompt: str
    target_entity_id: Optional[str] = None
    target_class: Optional[str] = None
    standoff_distance_m: float = 0.5
    robot_radius_m: float = 0.18
    target_center_floor: Optional[Tuple[float, float, float]] = None
    approach_pose_2d: Optional[Tuple[float, float, float]] = None  # (x, y, yaw_rad)
    evidence: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "prompt": self.prompt,
            "target_entity_id": self.target_entity_id,
            "target_class": self.target_class,
            "standoff_distance_m": self.standoff_distance_m,
            "robot_radius_m": self.robot_radius_m,
            "target_center_floor": self.target_center_floor,
            "approach_pose_2d": {
                "x_m": self.approach_pose_2d[0] if self.approach_pose_2d else None,
                "y_m": self.approach_pose_2d[1] if self.approach_pose_2d else None,
                "yaw_rad": self.approach_pose_2d[2] if self.approach_pose_2d else None,
                "yaw_deg": math.degrees(self.approach_pose_2d[2]) if self.approach_pose_2d else None,
            } if self.approach_pose_2d else None,
            "evidence": self.evidence,
            "rejection_reason": self.rejection_reason,
        }


class GoalResolver:
    """Resolves natural language navigation prompts against a validated semantic map."""

    # Synonym mapping for common natural language terms to canonical perception classes
    CLASS_SYNONYMS = {
        "table": ["desk", "table", "dining table"],
        "desk": ["desk", "table"],
        "chair": ["chair", "armchair", "seat"],
        "drone": ["robot", "quadcopter drone", "airplane"],
        "robot": ["robot", "quadcopter drone"],
        "laptop": ["laptop", "computer"],
        "backpack": ["backpack", "bag", "handbag"],
        "bag": ["handbag", "backpack"],
        "handbag": ["handbag", "bag", "backpack"],
        "bottle": ["bottle", "cup"],
        "door": ["door"],
        "fan": ["ceiling fan"],
        "ceiling fan": ["ceiling fan"],
    }

    def __init__(
        self,
        semantic_map: Dict[str, Any],
        object_distances: Optional[Dict[str, Any]] = None,
        robot_radius_m: float = 0.18,
        camera_forward_offset_m: float = 0.12,
    ):
        self.semantic_map = semantic_map
        self.object_distances = object_distances or {}
        self.robot_radius_m = robot_radius_m
        self.camera_forward_offset_m = camera_forward_offset_m
        self.entities = semantic_map.get("entities", {})

    @classmethod
    def from_map_dir(cls, map_dir: Path, robot_radius_m: float = 0.18) -> "GoalResolver":
        map_dir = Path(map_dir)
        sem_path = map_dir / "semantic_map.json"
        if not sem_path.exists():
            raise FileNotFoundError(f"Missing semantic_map.json in {map_dir}")

        with open(sem_path, "r", encoding="utf-8") as f:
            sem_map = json.load(f)

        dist_path = map_dir / "object_distances.json"
        dist_data = None
        if dist_path.exists():
            with open(dist_path, "r", encoding="utf-8") as f:
                dist_data = json.load(f)

        return cls(sem_map, dist_data, robot_radius_m=robot_radius_m)

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Extract target class, explicit ID, standoff distance, and relational tokens."""
        p_lower = prompt.lower().strip()

        # 1. Parse standoff distance (e.g. "stop 0.5 metres away", "stop 0.4m away", "0.5m")
        standoff = 0.5  # default 0.5 meters
        m_dist = re.search(r"(\d+(?:\.\d+)?)\s*(?:metres?|meters?|m)\s*(?:away)?", p_lower)
        if m_dist:
            standoff = float(m_dist.group(1))

        # 2. Check for explicit entity ID (e.g. "chair_003", "desk_002", "robot_001")
        explicit_id = None
        m_id = re.search(r"\b([a-z_]+_\d{3})\b", p_lower)
        if m_id:
            candidate_id = m_id.group(1)
            if candidate_id in self.entities:
                explicit_id = candidate_id

        # 3. Target class matching
        target_class = None
        for canonical, syns in self.CLASS_SYNONYMS.items():
            for s in syns:
                if re.search(rf"\b{re.escape(s)}\b", p_lower):
                    target_class = canonical
                    break
            if target_class:
                break

        # 4. Relational context keywords ("with the drone", "near the window", etc.)
        relational_cue = None
        m_with = re.search(r"(?:with|near|holding|next to)\s+(?:the\s+)?([a-z_]+)", p_lower)
        if m_with:
            cue_word = m_with.group(1)
            # resolve cue word to class
            for canonical, syns in self.CLASS_SYNONYMS.items():
                if cue_word in syns or cue_word == canonical:
                    relational_cue = canonical
                    break
            if not relational_cue:
                relational_cue = cue_word

        return {
            "explicit_id": explicit_id,
            "target_class": target_class,
            "standoff_distance_m": standoff,
            "relational_cue": relational_cue,
            "raw_prompt": prompt,
        }

    def _is_entity_confirmed(self, entity: Dict[str, Any]) -> bool:
        if "is_confirmed" in entity:
            return bool(entity["is_confirmed"])
        status = str(entity.get("status", "")).upper()
        return status in ("CONFIRMED", "ACTIVE")

    def _get_entity_center(self, entity: Dict[str, Any]) -> Tuple[float, float, float]:
        if "centroid_floor" in entity:
            c = entity["centroid_floor"]
            return float(c[0]), float(c[1]), float(c[2])
        if "floor_coordinates" in entity and "center" in entity["floor_coordinates"]:
            c = entity["floor_coordinates"]["center"]
            return float(c[0]), float(c[1]), float(c[2])
        return (0.0, 0.0, 0.0)

    def _get_entity_radius(self, entity: Dict[str, Any]) -> float:
        if "bbox_min_floor" in entity and "bbox_max_floor" in entity:
            bmin = entity["bbox_min_floor"]
            bmax = entity["bbox_max_floor"]
            dx = float(bmax[0]) - float(bmin[0])
            dy = float(bmax[1]) - float(bmin[1])
            return max(0.2, 0.5 * max(dx, dy))
        if "floor_coordinates" in entity and "bounding_box_min_max" in entity["floor_coordinates"]:
            bb = entity["floor_coordinates"]["bounding_box_min_max"]
            dx = float(bb["max_x"]) - float(bb["min_x"])
            dy = float(bb["max_y"]) - float(bb["min_y"])
            return max(0.2, 0.5 * max(dx, dy))
        if "dimensions_m" in entity:
            dim = entity["dimensions_m"]
            if isinstance(dim, list) and len(dim) >= 2:
                return max(0.2, 0.5 * max(float(dim[0]), float(dim[1])))
            elif isinstance(dim, dict):
                return max(0.2, 0.5 * max(float(dim.get("width_x", 0.5)), float(dim.get("length_y", 0.5))))
        return 0.35

    def resolve(
        self,
        prompt: str,
        current_robot_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> GoalResolutionResult:
        """Resolve prompt to target entity and generate verified approach pose."""
        parsed = self.parse_prompt(prompt)
        standoff = parsed["standoff_distance_m"]
        explicit_id = parsed["explicit_id"]
        target_class = parsed["target_class"]
        rel_cue = parsed["relational_cue"]

        # Case 1: Explicit entity ID requested
        if explicit_id:
            entity = self.entities.get(explicit_id)
            if not entity:
                return GoalResolutionResult(
                    success=False,
                    prompt=prompt,
                    rejection_reason=f"Entity '{explicit_id}' not found in semantic map",
                )
            if not self._is_entity_confirmed(entity):
                q_reason = entity.get("quarantine_reason") or entity.get("status") or "unconfirmed"
                return GoalResolutionResult(
                    success=False,
                    prompt=prompt,
                    target_entity_id=explicit_id,
                    rejection_reason=f"Entity '{explicit_id}' is quarantined or unconfirmed ({q_reason})",
                )
            return self._build_approach_pose(explicit_id, entity, standoff, current_robot_pose, prompt, {"match_type": "explicit_id"})

        # Case 2: Target class with relational cue (e.g. "table with the drone")
        confirmed_candidates = []
        for eid, ent in self.entities.items():
            if not self._is_entity_confirmed(ent):
                continue
            e_cls = ent.get("class_name", "").lower()
            if target_class:
                allowed_classes = self.CLASS_SYNONYMS.get(target_class, [target_class])
                if e_cls in allowed_classes:
                    confirmed_candidates.append((eid, ent))

        if not confirmed_candidates:
            # Check if any candidate exists in unconfirmed/quarantine
            unconf = [eid for eid, ent in self.entities.items() if target_class and ent.get("class_name") in self.CLASS_SYNONYMS.get(target_class, [])]
            if unconf:
                return GoalResolutionResult(
                    success=False,
                    prompt=prompt,
                    rejection_reason=f"Found candidates {unconf} for '{target_class}', but they are UNCONFIRMED with insufficient multi-view evidence",
                )
            return GoalResolutionResult(
                success=False,
                prompt=prompt,
                rejection_reason=f"No confirmed entities of class '{target_class}' found in mapped environment",
            )

        # Disambiguate using relational cue if available
        if rel_cue:
            rel_cue_classes = self.CLASS_SYNONYMS.get(rel_cue, [rel_cue])
            cue_entities = [
                (eid, ent) for eid, ent in self.entities.items()
                if ent.get("class_name", "").lower() in rel_cue_classes
            ]

            scored_candidates = []
            for t_id, t_ent in confirmed_candidates:
                t_pos = np.array(self._get_entity_center(t_ent))
                # Find minimum distance to any cue entity
                min_cue_dist = float("inf")
                best_cue_id = None
                for c_id, c_ent in cue_entities:
                    c_pos = np.array(self._get_entity_center(c_ent))
                    dist = float(np.linalg.norm(t_pos[:2] - c_pos[:2]))
                    if dist < min_cue_dist:
                        min_cue_dist = dist
                        best_cue_id = c_id

                # If within 2.0 meters, plausible spatial relationship (e.g. drone on/near table)
                if min_cue_dist <= 2.0:
                    scored_candidates.append((t_id, t_ent, min_cue_dist, best_cue_id))

            if len(scored_candidates) == 1:
                t_id, t_ent, d_cue, c_id = scored_candidates[0]
                evidence = {
                    "match_type": "relational_association",
                    "related_entity": c_id,
                    "distance_to_relation_m": round(d_cue, 3),
                }
                return self._build_approach_pose(t_id, t_ent, standoff, current_robot_pose, prompt, evidence)
            elif len(scored_candidates) > 1:
                scored_candidates.sort(key=lambda x: x[2])
                t_id, t_ent, d_cue, c_id = scored_candidates[0]
                evidence = {
                    "match_type": "closest_to_relational_cue",
                    "related_entity": c_id,
                    "distance_to_relation_m": round(d_cue, 3),
                }
                return self._build_approach_pose(t_id, t_ent, standoff, current_robot_pose, prompt, evidence)

        # No relational cue or cue did not filter: if only 1 candidate, select it
        if len(confirmed_candidates) == 1:
            t_id, t_ent = confirmed_candidates[0]
            return self._build_approach_pose(t_id, t_ent, standoff, current_robot_pose, prompt, {"match_type": "unique_class_instance"})

        # Multiple confirmed candidates without distinguishing relation -> Ambiguity
        cand_ids = [c[0] for c in confirmed_candidates]
        cur_pos = np.array(current_robot_pose[:2])
        sorted_by_dist = sorted(confirmed_candidates, key=lambda x: np.linalg.norm(np.array(self._get_entity_center(x[1])[:2]) - cur_pos))
        best_id, best_ent = sorted_by_dist[0]
        evidence = {
            "match_type": "closest_instance_fallback",
            "all_candidates": cand_ids,
            "selected_proximity_rank": 1,
        }
        return self._build_approach_pose(best_id, best_ent, standoff, current_robot_pose, prompt, evidence)

    def _build_approach_pose(
        self,
        entity_id: str,
        entity: Dict[str, Any],
        standoff_m: float,
        current_robot_pose: Tuple[float, float, float],
        prompt: str,
        evidence: Dict[str, Any],
    ) -> GoalResolutionResult:
        """Calculate collision-free approach pose outside the object footprint."""
        cx, cy, cz = self._get_entity_center(entity)
        obj_radius = self._get_entity_radius(entity)

        rx, ry, _ = current_robot_pose
        # Direction vector from object center to current robot position
        dx = rx - cx
        dy = ry - cy
        dist_to_obj = math.hypot(dx, dy)

        if dist_to_obj < 1e-4:
            # Robot is inside object center: back up along -X
            dir_x, dir_y = -1.0, 0.0
        else:
            dir_x, dir_y = dx / dist_to_obj, dy / dist_to_obj

        # Total distance from object center to robot stopping base_link:
        # object surface radius + requested clearance + robot collision radius
        total_clearance = obj_radius + standoff_m + self.robot_radius_m
        goal_x = cx + dir_x * total_clearance
        goal_y = cy + dir_y * total_clearance

        # Desired robot yaw facing directly toward object center
        desired_yaw = math.atan2(cy - goal_y, cx - goal_x)

        evidence.update({
            "object_center_m": [cx, cy, cz],
            "estimated_object_radius_m": round(obj_radius, 3),
            "robot_radius_m": self.robot_radius_m,
            "camera_offset_m": self.camera_forward_offset_m,
            "calculated_standoff_m": standoff_m,
            "approach_vector": [round(dir_x, 3), round(dir_y, 3)],
        })

        return GoalResolutionResult(
            success=True,
            prompt=prompt,
            target_entity_id=entity_id,
            target_class=entity.get("class_name", "unknown"),
            standoff_distance_m=standoff_m,
            robot_radius_m=self.robot_radius_m,
            target_center_floor=(cx, cy, cz),
            approach_pose_2d=(goal_x, goal_y, desired_yaw),
            evidence=evidence,
        )
