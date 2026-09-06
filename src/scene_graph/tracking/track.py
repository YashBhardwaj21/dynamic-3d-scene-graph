from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from scene_graph.tracking.state import KalmanState


class TrackState(Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    TEMPORARILY_UNOBSERVED = "temporarily_unobserved"
    LOST = "lost"


@dataclass(eq=False)
class Track:
    object_id: str
    class_name: str
    state: TrackState
    _initial_centroid: np.ndarray
    last_observed_frame: int
    first_observed_frame: int
    observation_count: int
    missing_count: int
    detection_confidence: float
    track_observation_ratio: float
    last_timestamp: float
    recent_observations: deque
    kalman_state: KalmanState | None = None
    size_world: np.ndarray | None = None
    
    @property
    def centroid_world(self) -> np.ndarray:
        if self.kalman_state is not None:
            return self.kalman_state.position
        return self._initial_centroid
        
    @property
    def velocity_world(self) -> np.ndarray | None:
        if self.kalman_state is not None:
            return self.kalman_state.velocity
        return None
        
    @property
    def position_covariance_world(self) -> np.ndarray | None:
        if self.kalman_state is not None:
            return self.kalman_state.position_covariance
        return None
    
    # Store the actual object instances (e.g. Observation) over time if needed
    # internally by tracker, bounded by the deque.
    
    def __hash__(self):
        return hash(self.object_id)
        
    def __eq__(self, other):
        if not isinstance(other, Track):
            return False
        return self.object_id == other.object_id
