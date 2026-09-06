from enum import Enum
from typing import Dict, Any

from scene_graph.tracking.track import Track, TrackState


class ObjectState(Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"
    REMOVED = "removed"


class ObjectStateMachine:
    """Manages the semantic state of an object over time."""
    
    def __init__(self, hysteresis_frames: int = 5):
        self.hysteresis_frames = hysteresis_frames
        self.states: Dict[str, ObjectState] = {}
        self.missing_counters: Dict[str, int] = {}
        self.stable_counters: Dict[str, int] = {}

    def update(self, tracks: list[Track], frame_index: int) -> Dict[str, ObjectState]:
        """Update object states based on track updates."""
        
        # Build lookup for fast access
        track_lookup = {t.object_id: t for t in tracks}
        
        # 1. Update existing objects
        for obj_id in list(self.states.keys()):
            if obj_id in track_lookup:
                track = track_lookup[obj_id]
                self._update_existing(obj_id, track)
            else:
                self._handle_missing(obj_id)
                
        # 2. Add new objects
        for track in tracks:
            if track.object_id not in self.states:
                if track.state == TrackState.ACTIVE:
                    # New active tracks start as STABLE immediately or after stability wait.
                    self.states[track.object_id] = ObjectState.STABLE
                    self.missing_counters[track.object_id] = 0
                    self.stable_counters[track.object_id] = 0
                    
        return self.states

    def _update_existing(self, obj_id: str, track: Track):
        current_state = self.states[obj_id]
        
        if track.state == TrackState.ACTIVE:
            self.missing_counters[obj_id] = 0
            if current_state == ObjectState.UNSTABLE:
                self.stable_counters[obj_id] += 1
                if self.stable_counters[obj_id] >= self.hysteresis_frames:
                    self.states[obj_id] = ObjectState.STABLE
        elif track.state == TrackState.TEMPORARILY_UNOBSERVED:
            self.missing_counters[obj_id] += 1
            if current_state == ObjectState.STABLE and self.missing_counters[obj_id] > self.hysteresis_frames:
                self.states[obj_id] = ObjectState.UNSTABLE
        elif track.state == TrackState.LOST:
            self.states[obj_id] = ObjectState.REMOVED

    def _handle_missing(self, obj_id: str):
        # The tracker dropped the track entirely
        self.states[obj_id] = ObjectState.REMOVED
