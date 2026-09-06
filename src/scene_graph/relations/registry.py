from typing import List, Tuple

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track, TrackState
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence
from scene_graph.relations.admissibility import AdmissibilityFilter
from scene_graph.relations.inverse_algebra import INVERSE, SYMMETRIC


ALLOWED_PREDICATES = frozenset(INVERSE) | frozenset(SYMMETRIC)


class RelationRegistry:

    def __init__(self, config: SceneGraphConfig):
        self.config = config
        self.admissibility_filter = AdmissibilityFilter(config)
        self._modules: List[RelationModule] = []

    def register(self, module: RelationModule) -> None:
        self._modules.append(module)

    @staticmethod
    def _generate_candidate_pairs(
        tracks: List[Track],
    ) -> List[Tuple[Track, Track]]:
        active_tracks = [
            track
            for track in tracks
            if track.state == TrackState.ACTIVE
        ]

        active_tracks.sort(key=lambda track: track.object_id)

        return [
            (subject, object_)
            for subject in active_tracks
            for object_ in active_tracks
            if subject.object_id != object_.object_id
        ]

    def _filter_admissible(
        self,
        pairs: List[Tuple[Track, Track]],
        predicate: str,
    ) -> List[Tuple[Track, Track]]:
        return [
            (subject, object_)
            for subject, object_ in pairs
            if self.admissibility_filter.is_admissible(
                predicate,
                subject,
                object_,
            )
        ]

    def compute_all(
        self,
        tracks: List[Track],
        context: FrameContext,
    ) -> List[RelationEvidence]:
        candidate_pairs = self._generate_candidate_pairs(tracks)
        all_evidences: List[RelationEvidence] = []

        for module in self._modules:
            predicates = tuple(module.predicates())

            invalid_predicates = [
                predicate
                for predicate in predicates
                if predicate not in ALLOWED_PREDICATES
            ]

            if invalid_predicates:
                raise ValueError(
                    f"Module {module.__class__.__name__} provided "
                    f"invalid predicates: {invalid_predicates}"
                )

            admissible_pairs = set()

            for predicate in predicates:
                admissible_pairs.update(
                    self._filter_admissible(
                        candidate_pairs,
                        predicate,
                    )
                )

            sorted_pairs = sorted(
                admissible_pairs,
                key=lambda pair: (
                    pair[0].object_id,
                    pair[1].object_id,
                ),
            )

            evidences = module.compute_pairs(
                sorted_pairs,
                context,
            )

            for evidence in evidences:
                if evidence.predicate not in ALLOWED_PREDICATES:
                    raise ValueError(
                        f"Module {module.__class__.__name__} emitted "
                        f"invalid predicate '{evidence.predicate}'."
                    )

            all_evidences.extend(evidences)

        return all_evidences