from enum import Enum
from typing import Dict, List, Set, Tuple

from scene_graph.relations.evidence import RelationEvidence, EvidenceResult


class RelationState(Enum):
    """Strict 4-state lifecycle for every (subject, predicate, object) triple."""
    HYPOTHESIZED = "hypothesized"   # Seen but not yet confirmed
    SUPPORTED = "supported"         # Confirmed by temporal evidence
    CONTRADICTED = "contradicted"   # Actively denied by geometry
    UNKNOWN = "unknown"             # No recent evidence either way


# --- Transition table ---
# (current_state, evidence_this_frame) -> next_state
# This is the ONLY place state changes are defined.
_TRANSITIONS = {
    # From HYPOTHESIZED
    (RelationState.HYPOTHESIZED, EvidenceResult.SUPPORTED):    "increment_support",
    (RelationState.HYPOTHESIZED, EvidenceResult.CONTRADICTED): RelationState.CONTRADICTED,
    (RelationState.HYPOTHESIZED, None):                        "increment_missing",

    # From SUPPORTED
    (RelationState.SUPPORTED, EvidenceResult.SUPPORTED):       RelationState.SUPPORTED,
    (RelationState.SUPPORTED, EvidenceResult.CONTRADICTED):    RelationState.CONTRADICTED,
    (RelationState.SUPPORTED, None):                           "increment_missing",

    # From CONTRADICTED
    (RelationState.CONTRADICTED, EvidenceResult.SUPPORTED):    RelationState.HYPOTHESIZED,
    (RelationState.CONTRADICTED, EvidenceResult.CONTRADICTED): RelationState.CONTRADICTED,
    (RelationState.CONTRADICTED, None):                        "increment_missing",

    # From UNKNOWN
    (RelationState.UNKNOWN, EvidenceResult.SUPPORTED):         RelationState.HYPOTHESIZED,
    (RelationState.UNKNOWN, EvidenceResult.CONTRADICTED):      RelationState.CONTRADICTED,
    (RelationState.UNKNOWN, None):                             "gc_candidate",
}


class RelationStateMachine:
    """Manages the semantic state of relations over time with hysteresis.
    
    Key invariant: the state dict contains ONLY relations that have been 
    observed at least once. Dead relations (no evidence for gc_frames) 
    are garbage-collected and removed entirely, preventing accumulation.
    """
    
    def __init__(self, confirmation_frames: int = 3, missing_frames: int = 2):
        self.confirmation_frames = confirmation_frames
        self.missing_frames = missing_frames
        # Number of frames with no evidence before a key is garbage-collected
        self.gc_frames = missing_frames * 3
        
        # State: (subject_id, object_id, predicate) -> RelationState
        self.states: Dict[Tuple[str, str, str], RelationState] = {}
        
        # Counters
        self.support_counters: Dict[Tuple[str, str, str], int] = {}
        self.missing_counters: Dict[Tuple[str, str, str], int] = {}

    def update(self, evidences: List[RelationEvidence], frame_index: int,
               active_object_ids: Set[str] = None) -> Dict[Tuple[str, str, str], RelationState]:
        """Update relation states based on new evidence for a single frame.
        
        Args:
            evidences: All relation evidences computed for this frame.
            frame_index: Current frame index.
            active_object_ids: Set of currently active object IDs. If provided,
                relations involving non-active objects are garbage-collected.
        """
        
        # 1. Build per-key evidence lookup for THIS frame
        frame_evidence: Dict[Tuple[str, str, str], EvidenceResult] = {}
        for ev in evidences:
            key = (ev.subject_id, ev.object_id, ev.predicate)
            # If multiple evidences exist for same key, SUPPORTED wins over CONTRADICTED
            existing = frame_evidence.get(key)
            if existing is None or ev.result == EvidenceResult.SUPPORTED:
                frame_evidence[key] = ev.result
        
        # 2. Collect all keys that need processing (existing + new)
        all_keys = set(self.states.keys()) | set(frame_evidence.keys())
        
        # 3. Apply transitions
        keys_to_gc = []
        
        for key in all_keys:
            current_state = self.states.get(key, RelationState.UNKNOWN)
            evidence = frame_evidence.get(key)  # None if no evidence this frame
            
            # Garbage-collect relations involving non-active objects
            if active_object_ids is not None:
                subj_id, obj_id, _ = key
                if subj_id not in active_object_ids or obj_id not in active_object_ids:
                    keys_to_gc.append(key)
                    continue
            
            transition = _TRANSITIONS.get((current_state, evidence))
            
            if transition is None:
                # No evidence and key doesn't exist yet — skip
                continue
            elif isinstance(transition, RelationState):
                # Direct state change
                self.states[key] = transition
                self.missing_counters[key] = 0
                if transition == RelationState.HYPOTHESIZED:
                    self.support_counters[key] = 1
            elif transition == "increment_support":
                self.missing_counters[key] = 0
                self.support_counters[key] = self.support_counters.get(key, 0) + 1
                if self.support_counters[key] >= self.confirmation_frames:
                    self.states[key] = RelationState.SUPPORTED
            elif transition == "increment_missing":
                self.missing_counters[key] = self.missing_counters.get(key, 0) + 1
                if self.missing_counters[key] >= self.missing_frames:
                    self.states[key] = RelationState.UNKNOWN
            elif transition == "gc_candidate":
                self.missing_counters[key] = self.missing_counters.get(key, 0) + 1
                if self.missing_counters[key] >= self.gc_frames:
                    keys_to_gc.append(key)
        
        # 4. Garbage collect dead keys
        for key in keys_to_gc:
            self.states.pop(key, None)
            self.support_counters.pop(key, None)
            self.missing_counters.pop(key, None)
                
        return self.states
