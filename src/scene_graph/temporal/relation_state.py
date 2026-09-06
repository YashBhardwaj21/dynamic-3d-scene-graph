from enum import Enum
import numpy as np
from typing import Dict, List, Set, Tuple

from scene_graph.relations.evidence import RelationEvidence, EvidenceResult


from scene_graph.config import RelationTemporalConfig

class RelationState(Enum):
    """Strict 4-state lifecycle for every (subject, predicate, object) triple."""
    HYPOTHESIZED = "hypothesized"   # Seen but not yet confirmed
    SUPPORTED = "supported"         # Confirmed by temporal evidence
    CONTRADICTED = "contradicted"   # Actively denied by geometry
    UNKNOWN = "unknown"             # No recent evidence either way


class RelationStateMachine:
    """Manages the semantic state of relations over time with continuous belief.
    
    Belief is in [-1.0, 1.0]. 
    Positive values mean supported, negative mean contradicted.
    Belief decays towards 0 over time when unobserved.
    """
    
    def __init__(self, config: RelationTemporalConfig):
        self.config = config
        
        # State: (subject_id, object_id, predicate) -> RelationState
        self.states: Dict[Tuple[str, str, str], RelationState] = {}
        
        # Belief: (subject_id, object_id, predicate) -> float in [-1, 1]
        self.beliefs: Dict[Tuple[str, str, str], float] = {}
        
        # Last updated time: (subject_id, object_id, predicate) -> float
        self.last_update_times: Dict[Tuple[str, str, str], float] = {}

    def update(self, evidences: List[RelationEvidence], frame_index: int, timestamp: float,
               active_object_ids: Set[str] = None) -> Dict[Tuple[str, str, str], RelationState]:
        """Update relation states based on new evidence for a single frame.
        
        Args:
            evidences: All relation evidences computed for this frame.
            frame_index: Current frame index.
            timestamp: Current frame timestamp in seconds.
            active_object_ids: Set of currently active object IDs. If provided,
                relations involving non-active objects are garbage-collected.
        """
        
        # 1. Build per-key evidence lookup for THIS frame
        frame_evidence: Dict[Tuple[str, str, str], RelationEvidence] = {}
        for ev in evidences:
            key = (ev.subject_id, ev.object_id, ev.predicate)
            # If multiple evidences exist for same key, SUPPORTED wins over CONTRADICTED
            existing = frame_evidence.get(key)
            if existing is None or ev.result == EvidenceResult.SUPPORTED:
                frame_evidence[key] = ev
        
        # 2. Collect all keys that need processing (existing + new)
        all_keys = set(self.states.keys()) | set(frame_evidence.keys())
        
        # 3. Apply time decay and updates
        keys_to_gc = []
        
        for key in all_keys:
            # Garbage-collect relations involving non-active objects
            if active_object_ids is not None:
                subj_id, obj_id, _ = key
                if subj_id not in active_object_ids or obj_id not in active_object_ids:
                    keys_to_gc.append(key)
                    continue
            
            ev = frame_evidence.get(key)
            
            # Apply time decay
            if key in self.last_update_times:
                dt = timestamp - self.last_update_times[key]
                if dt > 0:
                    current_belief = self.beliefs.get(key, 0.0)
                    decay = self.config.decay_per_second * dt
                    # Decay towards zero
                    if current_belief > 0:
                        self.beliefs[key] = max(0.0, current_belief - decay)
                    elif current_belief < 0:
                        self.beliefs[key] = min(0.0, current_belief + decay)
            else:
                self.beliefs[key] = 0.0
                dt = 0
            
            # Apply evidence
            if ev is not None:
                # Assuming ev.confidence is the 'strength' of the evidence
                delta = ev.confidence if ev.result == EvidenceResult.SUPPORTED else -ev.confidence
                self.beliefs[key] = np.clip(self.beliefs[key] + delta, -1.0, 1.0)
                self.last_update_times[key] = timestamp
            
            belief = self.beliefs[key]
            
            # State determination
            if belief >= self.config.confirmation_threshold:
                self.states[key] = RelationState.SUPPORTED
            elif belief <= self.config.contradiction_threshold:
                self.states[key] = RelationState.CONTRADICTED
            elif belief > 0:
                self.states[key] = RelationState.HYPOTHESIZED
            else:
                self.states[key] = RelationState.UNKNOWN
                
            # Missing time check
            if ev is None:
                missing_time = timestamp - self.last_update_times.get(key, timestamp)
                if missing_time >= self.config.lost_after_seconds:
                    keys_to_gc.append(key)
                elif missing_time >= self.config.unknown_after_seconds:
                    self.states[key] = RelationState.UNKNOWN
                    
        # 4. Garbage collect dead keys
        for key in keys_to_gc:
            self.states.pop(key, None)
            self.beliefs.pop(key, None)
            self.last_update_times.pop(key, None)
                
        return self.states
