from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class TrackState(Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    TEMPORARILY_UNOBSERVED = "temporarily_unobserved"
    LOST = "lost"


@dataclass
class Track:
    object_id: str
    class_name: str
    state: TrackState
    centroid_world: np.ndarray
    last_observed_frame: int
    first_observed_frame: int
    observation_count: int
    missing_count: int
    detection_confidence: float
    track_observation_ratio: float
    recent_observations: deque
    position_uncertainty: float | None = None
    
    # Store the actual object instances (e.g. Observation) over time if needed
    # internally by tracker, bounded by the deque.
