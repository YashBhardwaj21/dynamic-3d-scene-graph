from dataclasses import dataclass, field
from typing import Dict, Any

from scene_graph.relations.evidence import RelationEvidence
from scene_graph.temporal.relation_state import RelationState
from scene_graph.graph.participation_state import GraphParticipationState


@dataclass
class GraphEdge:
    """Represents a relational edge between two nodes in the scene graph."""
    
    predicate: str
    subject_id: str
    object_id: str
    state: RelationState
    # Store the most recent evidence that supports this relation
    latest_evidence: RelationEvidence
    
    participation: GraphParticipationState = GraphParticipationState.ACTIVE
    
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Returns True if the relation is currently active in the graph."""
        return self.participation == GraphParticipationState.ACTIVE and self.state == RelationState.SUPPORTED
