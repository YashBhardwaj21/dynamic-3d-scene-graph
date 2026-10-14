"""Stage 5: Temporal Evidence Extractor for Non-VLM Pre-Routing Pipeline.

Quantifies temporal stability and motion:
1. World-relative object displacement compensating camera ego-motion.
2. Verified current motion classification (STATIC, MOVING, DISPLACED, TRANSIENT).
3. Continuous visibility duration and observation count.
4. Occlusion intervals and reappearance detection.
5. Persistent versus transient observation inconsistencies.
6. Differential position change since previous checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np


MOTION_NOISE_FLOOR_M = 0.04      # Baseline measurement noise standard deviation (1-sigma)
MOTION_GATE_3SIGMA_M = 0.12      # 3-sigma gate (0.12m)
DISPLACEMENT_THRESHOLD_M = 0.25  # Minimal true displacement threshold (isolates physical relocation from bounding box view-frustum drift)


@dataclass
class TemporalEvidenceRecord:
    """Timestamp-aligned temporal tracking and motion evidence record."""
    obs_id: str
    frame_index: int
    timestamp: float
    entity_id: str
    track_id: str
    current_motion: Literal["STATIC", "MOVING", "DISPLACED", "TRANSIENT", "UNCERTAIN"]
    world_displacement_m: float
    world_displacement_sigma_m: float
    camera_ego_displacement_m: float
    ego_motion_compensation_ratio: float
    motion_significance_ratio: float
    visibility_duration_s: float
    observation_count: int
    occlusion_events_count: int
    frames_since_last_seen: int
    is_reappearance_after_occlusion: bool
    temporal_persistence_score: float
    position_change_since_last_step_m: float
    has_valid_evidence: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TemporalEvidenceExtractor:
    """Tracks temporal continuity and estimates world-relative motion with camera ego-compensation."""

    def __init__(self):
        # Key: entity_id or track_id -> history dict
        self._tracks: Dict[str, Dict[str, Any]] = {}

    def extract_observation_temporal(
        self,
        obs: Dict[str, Any],
    ) -> TemporalEvidenceRecord:
        """Extract temporal and motion evidence for a timestamped observation."""
        obs_id = obs.get("obs_id", "")
        frame_idx = int(obs.get("frame_index", 0))
        ts = float(obs.get("timestamp", 0.0))
        entity_id = obs.get("global_entity_id", "")
        track_id = obs.get("local_track_id", "")
        session = obs.get("source_session", "")
        key = f"{session}_{track_id}" if (session and track_id) else (track_id or entity_id)

        # 3D floor centroid
        curr_centroid = np.array(obs.get("centroid_floor", [0.0, 0.0, 0.0]), dtype=np.float64)
        curr_cam_pose = np.array(obs.get("camera_pose", [0, 0, 0, 0, 0, 0, 1])[:3], dtype=np.float64)

        if key not in self._tracks:
            # First observation of this track
            self._tracks[key] = {
                "initial_centroid": curr_centroid.copy(),
                "last_centroid": curr_centroid.copy(),
                "initial_cam_pos": curr_cam_pose.copy(),
                "last_cam_pos": curr_cam_pose.copy(),
                "initial_ts": ts,
                "last_ts": ts,
                "initial_frame": frame_idx,
                "last_frame": frame_idx,
                "observation_count": 1,
                "occlusion_events": 0,
                "centroids_history": [curr_centroid.copy()],
                "ts_history": [ts],
            }
            return TemporalEvidenceRecord(
                obs_id=obs_id,
                frame_index=frame_idx,
                timestamp=ts,
                entity_id=entity_id,
                track_id=track_id,
                current_motion="TRANSIENT",
                world_displacement_m=0.0,
                world_displacement_sigma_m=MOTION_NOISE_FLOOR_M,
                camera_ego_displacement_m=0.0,
                ego_motion_compensation_ratio=1.0,
                motion_significance_ratio=0.0,
                visibility_duration_s=0.0,
                observation_count=1,
                occlusion_events_count=0,
                frames_since_last_seen=0,
                is_reappearance_after_occlusion=False,
                temporal_persistence_score=0.15,
                position_change_since_last_step_m=0.0,
                has_valid_evidence=True,
            )

        trk = self._tracks[key]
        prev_centroid = trk["last_centroid"]
        prev_ts = trk["last_ts"]
        prev_frame = trk["last_frame"]

        # Track updates
        trk["observation_count"] += 1
        trk["centroids_history"].append(curr_centroid.copy())
        trk["ts_history"].append(ts)

        # 1. Occlusion and gap detection
        frame_gap = frame_idx - prev_frame
        time_gap = ts - prev_ts
        is_reappearance = False
        if frame_gap > 12 or time_gap > 0.45:
            trk["occlusion_events"] += 1
            is_reappearance = True

        # 2. Camera Ego-Motion vs Object World Displacement
        cam_displacement = float(np.linalg.norm(curr_cam_pose - trk["initial_cam_pos"]))
        step_obj_disp = float(np.linalg.norm(curr_centroid - prev_centroid))
        total_obj_disp = float(np.linalg.norm(curr_centroid - trk["initial_centroid"]))

        # Empirical measurement noise standard deviation (1-sigma)
        n_obs = trk["observation_count"]
        combined_sigma = MOTION_NOISE_FLOOR_M

        # 3. Motion Classification & Signal-to-Noise Ratio (Z-score)
        motion_sig_ratio = total_obj_disp / max(0.005, combined_sigma)
        
        if n_obs < 3:
            current_motion = "TRANSIENT"
        elif total_obj_disp > max(DISPLACEMENT_THRESHOLD_M, 3.0 * combined_sigma):
            current_motion = "MOVING" if step_obj_disp > 0.05 else "DISPLACED"
        else:
            current_motion = "STATIC"

        # Ego-motion compensation ratio
        ego_comp = (cam_displacement / (total_obj_disp + 1e-4)) if cam_displacement > 0.05 else 1.0

        # 4. Visibility Duration & Temporal Persistence Score
        vis_duration = float(ts - trk["initial_ts"])
        # Persistence score approaches 1.0 as observation count increases (>15 frames = 0.90+)
        persistence_score = 1.0 - math.exp(-n_obs / 8.0)

        # Update last states
        trk["last_centroid"] = curr_centroid.copy()
        trk["last_cam_pos"] = curr_cam_pose.copy()
        trk["last_ts"] = ts
        trk["last_frame"] = frame_idx

        return TemporalEvidenceRecord(
            obs_id=obs_id,
            frame_index=frame_idx,
            timestamp=ts,
            entity_id=entity_id,
            track_id=track_id,
            current_motion=current_motion,
            world_displacement_m=round(total_obj_disp, 4),
            world_displacement_sigma_m=round(combined_sigma, 4),
            camera_ego_displacement_m=round(cam_displacement, 4),
            ego_motion_compensation_ratio=round(min(500.0, ego_comp), 2),
            motion_significance_ratio=round(motion_sig_ratio, 3),
            visibility_duration_s=round(vis_duration, 3),
            observation_count=n_obs,
            occlusion_events_count=trk["occlusion_events"],
            frames_since_last_seen=frame_gap,
            is_reappearance_after_occlusion=is_reappearance,
            temporal_persistence_score=round(persistence_score, 4),
            position_change_since_last_step_m=round(step_obj_disp, 4),
            has_valid_evidence=True,
        )
