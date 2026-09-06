from enum import Enum
from typing import Dict, List, Tuple

from scene_graph.relations.evidence import RelationEvidence, EvidenceResult


class RelationState(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class RelationStateMachine:
    """Manages the semantic state of relations over time with hysteresis."""
    
    def __init__(self, confirmation_frames: int = 3, missing_frames: int = 2):
        self.confirmation_frames = confirmation_frames
        self.missing_frames = missing_frames
        
        # State: (subject_id, object_id, predicate) -> RelationState
        self.states: Dict[Tuple[str, str, str], RelationState] = {}
        
        # Counters: (subject_id, object_id, predicate) -> int
        self.support_counters: Dict[Tuple[str, str, str], int] = {}
        self.missing_counters: Dict[Tuple[str, str, str], int] = {}

    def update(self, evidences: List[RelationEvidence], frame_index: int) -> Dict[Tuple[str, str, str], RelationState]:
        """Update relation states based on new evidence."""
        
        # Build lookup of supported relations in current frame
        supported_keys = set()
        for ev in evidences:
            if ev.result == EvidenceResult.SUPPORTED:
                key = (ev.subject_id, ev.object_id, ev.predicate)
                supported_keys.add(key)
                
        # 1. Update existing relations
        for key in list(self.states.keys()):
            if key in supported_keys:
                self._update_supported(key)
            else:
                self._update_missing(key)
                
        # 2. Add new relations
        for key in supported_keys:
            if key not in self.states:
                self.states[key] = RelationState.UNKNOWN
                self.support_counters[key] = 1
                self.missing_counters[key] = 0
                if self.confirmation_frames <= 1:
                    self.states[key] = RelationState.SUPPORTED
                    
        return self.states

    def _update_supported(self, key: Tuple[str, str, str]):
        self.missing_counters[key] = 0
        if self.states[key] in (RelationState.UNKNOWN, RelationState.CONTRADICTED):
            self.support_counters[key] += 1
            if self.support_counters[key] >= self.confirmation_frames:
                self.states[key] = RelationState.SUPPORTED
        elif self.states[key] == RelationState.NOT_APPLICABLE:
            self.states[key] = RelationState.UNKNOWN
            self.support_counters[key] = 1

    def _update_missing(self, key: Tuple[str, str, str]):
        if self.states[key] == RelationState.SUPPORTED:
            self.missing_counters[key] += 1
            if self.missing_counters[key] > self.missing_frames:
                self.states[key] = RelationState.CONTRADICTED
        elif self.states[key] == RelationState.UNKNOWN:
            self.states[key] = RelationState.NOT_APPLICABLE
