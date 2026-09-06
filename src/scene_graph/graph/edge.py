from dataclasses import dataclass, field
from typing import Dict, Any

from scene_graph.relations.evidence import RelationEvidence
from scene_graph.temporal.relation_state import RelationState


@dataclass
class GraphEdge:
    """Represents a relational edge between two nodes in the scene graph."""
    
    predicate: str
    subject_id: str
    object_id: str
    state: RelationState
    
    # Store the most recent evidence that supports this relation
    latest_evidence: RelationEvidence
    
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Returns True if the relation is currently confirmed."""
        return self.state == RelationState.CONFIRMED
