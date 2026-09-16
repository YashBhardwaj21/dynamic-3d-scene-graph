from dataclasses import dataclass, field
from typing import Dict, Any

from scene_graph.tracking.track import Track, TrackState
from scene_graph.temporal.object_state import ObjectState
from scene_graph.graph.participation_state import GraphParticipationState


@dataclass
class GraphNode:
    """Represents an object node in the scene graph."""
    
    object_id: str
    class_name: str
    track: Track
    state: ObjectState
    participation: GraphParticipationState = GraphParticipationState.ACTIVE
    spatial_context_id: str = "world"
    is_spatial_anchor: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Returns True if the object is currently active in the scene graph world model."""
        return (
            self.participation == GraphParticipationState.ACTIVE
            and self.track.state in (
                TrackState.ACTIVE,
                TrackState.TEMPORARILY_UNOBSERVED,
            )
        )

    @property
    def is_observed(self) -> bool:
        """Returns True if the node is active and directly observed in the current frame."""
        return self.is_active and self.track.state == TrackState.ACTIVE

