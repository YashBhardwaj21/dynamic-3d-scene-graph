from dataclasses import dataclass, field
from typing import Dict, Any

from scene_graph.tracking.track import Track
from scene_graph.temporal.object_state import ObjectState


@dataclass
class GraphNode:
    """Represents an object node in the scene graph."""
    
    object_id: str
    class_name: str
    track: Track
    state: ObjectState
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Returns True if the object is currently stable and actively participating in the scene."""
        return self.state == ObjectState.STABLE
