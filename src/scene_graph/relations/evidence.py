from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class EvidenceResult(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_DEPTH = "insufficient_depth"
    INSUFFICIENT_GEOMETRY = "insufficient_geometry"
    MISSING_MASK = "missing_mask"


class ReferenceFrameType(Enum):
    WORLD = "world"
    CAMERA = "camera"


@dataclass(frozen=True)
class RelationEvidence:
    predicate: str
    subject_id: str
    object_id: str
    frame_index: int
    timestamp: float
    result: EvidenceResult
    value: Optional[float]
    threshold: Optional[float]
    confidence: float
    reference_frame: ReferenceFrameType
    evidence_type: str
    details: Dict[str, Any]