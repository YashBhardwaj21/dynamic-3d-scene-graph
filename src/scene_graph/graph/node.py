from dataclasses import dataclass, field
from typing import Dict, Any

from scene_graph.tracking.track import Track
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
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Returns True if the object is currently stable and actively participating in the scene."""
        return self.participation == GraphParticipationState.ACTIVE and self.state == ObjectState.STABLE
