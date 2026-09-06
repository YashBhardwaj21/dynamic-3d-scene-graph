from typing import List, Tuple

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track, TrackState
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence
from scene_graph.relations.admissibility import AdmissibilityFilter


class RelationRegistry:
    """Config-driven declarative registry for relation modules."""
    
    def __init__(self, config: SceneGraphConfig):
        self.config = config
        self.admissibility_filter = AdmissibilityFilter(self.config)
        self._modules: List[RelationModule] = []
        
    def register(self, module: RelationModule):
        """Register a new relation computation module."""
        self._modules.append(module)
        
    def _generate_candidate_pairs(self, tracks: List[Track]) -> List[Tuple[Track, Track]]:
        """Generate all possible valid object pairs."""
        # Only evaluate pairs where both tracks are ACTIVE or TEMPORARILY_UNOBSERVED
        valid_states = {TrackState.ACTIVE, TrackState.TEMPORARILY_UNOBSERVED}
        valid_tracks = [t for t in tracks if t.state in valid_states]
        
        pairs = []
        for i, subj in enumerate(valid_tracks):
            for j, obj in enumerate(valid_tracks):
                if i != j:
                    pairs.append((subj, obj))
        return pairs

    def _filter_admissible(self, pairs: List[Tuple[Track, Track]], predicate: str) -> List[Tuple[Track, Track]]:
        """Filter pairs based on admissibility rules for a specific predicate."""
        return [
            (subj, obj) for subj, obj in pairs
            if self.admissibility_filter.is_admissible(predicate, subj, obj)
        ]

    def compute_all(self, tracks: List[Track], context: FrameContext) -> List[RelationEvidence]:
        """Compute all registered relations for all admissible pairs."""
        candidate_pairs = self._generate_candidate_pairs(tracks)
        all_evidences = []
        
        for module in self._modules:
            for predicate in module.predicates():
                admissible_pairs = self._filter_admissible(candidate_pairs, predicate)
                
                # Compute only for admissible pairs
                # (Some modules might compute multiple predicates at once, 
                #  so we pass the pair if it's admissible for ANY of its predicates)
                
            # Actually, the module itself might compute multiple predicates simultaneously.
            # We will pass all candidate pairs that are admissible for AT LEAST ONE 
            # predicate of this module.
            module_admissible_pairs = set()
            for predicate in module.predicates():
                module_admissible_pairs.update(self._filter_admissible(candidate_pairs, predicate))
                
            for subj, obj in module_admissible_pairs:
                evidences = module.compute(subj, obj, context)
                all_evidences.extend(evidences)
                
        return all_evidences
