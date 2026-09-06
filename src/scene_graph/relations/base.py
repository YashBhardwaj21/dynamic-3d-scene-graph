from abc import ABC, abstractmethod
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence


class RelationModule(ABC):
    
    @abstractmethod
    def predicates(self) -> List[str]:
        """Return the list of predicate names this module computes (e.g., ['NEAR', 'FAR'])."""
        pass
        
    @abstractmethod
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        """Compute relation evidence for the given subject and object."""
        pass
