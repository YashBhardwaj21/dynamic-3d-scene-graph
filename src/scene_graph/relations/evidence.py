from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional


class EvidenceResult(Enum):
    SUPPORTED = "supported"           # geometry confirms relation
    CONTRADICTED = "contradicted"     # geometry denies relation
    NOT_APPLICABLE = "not_applicable" # inadmissible pair
    INSUFFICIENT_DEPTH = "insufficient_depth"
    INSUFFICIENT_GEOMETRY = "insufficient_geometry"
    MISSING_MASK = "missing_mask"


@dataclass
class RelationEvidence:
    predicate: str              # "ON"
    subject_id: str             # "track_0003"
    object_id: str              # "track_0001"
    frame_index: int
    timestamp: float
    result: EvidenceResult      # e.g., SUPPORTED, CONTRADICTED
    value: Optional[float]      # 0.008 (plane distance)
    threshold: Optional[float]  # 0.02
    confidence: float           # [0,1] evidence strength, NOT calibrated probability
    reference_frame: str        # "world" or "camera"
    evidence_type: str          # "support_plane", "centroid_distance", etc.
    details: Dict[str, Any]     # {"support_overlap": 0.63, "plane_residual": 0.008, ...}
