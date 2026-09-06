from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import (
    RelationEvidence,
    EvidenceResult,
)


class DistanceRelationModule(RelationModule):

    def __init__(self, config):
        if config is None:
            raise ValueError("SceneGraphConfig is required.")

        if config.relations is None:
            raise ValueError("Relation configuration is required.")

        distance_config = config.relations.distance

        self.near_threshold = float(
            distance_config.near_threshold
        )
        self.far_threshold = float(
            distance_config.far_threshold
        )

        if self.near_threshold >= self.far_threshold:
            raise ValueError(
                "near_threshold must be smaller than far_threshold."
            )

    def predicates(self) -> List[str]:
        return ["NEAR", "FAR"]

    def compute_pairs(
        self,
        pairs: List[Tuple[Track, Track]],
        context: FrameContext,
    ) -> List[RelationEvidence]:

        canonical_pairs = [
            (subject, object_)
            for subject, object_ in pairs
            if subject.object_id < object_.object_id
        ]

        canonical_pairs.sort(
            key=lambda pair: (
                pair[0].object_id,
                pair[1].object_id,
            )
        )

        evidences: List[RelationEvidence] = []

        for subject, object_ in canonical_pairs:
            evidences.extend(
                self.compute(
                    subject,
                    object_,
                    context,
                )
            )

        return evidences

    def compute(
        self,
        subject: Track,
        object_: Track,
        context: FrameContext,
    ) -> List[RelationEvidence]:

        subject_geometry = context.observation_geometry.get(
            subject.object_id
        )
        object_geometry = context.observation_geometry.get(
            object_.object_id
        )

        if (
            subject_geometry is None
            or object_geometry is None
        ):
            return [
                self._insufficient_geometry(
                    subject,
                    object_,
                    context,
                )
            ]

        subject_points = subject_geometry.points_world_sampled
        object_points = object_geometry.points_world_sampled

        if (
            subject_points is None
            or object_points is None
            or len(subject_points) == 0
            or len(object_points) == 0
        ):
            return [
                self._insufficient_geometry(
                    subject,
                    object_,
                    context,
                )
            ]

        subject_points = np.asarray(
            subject_points,
            dtype=np.float64,
        )
        object_points = np.asarray(
            object_points,
            dtype=np.float64,
        )

        if (
            subject_points.ndim != 2
            or object_points.ndim != 2
            or subject_points.shape[1] != 3
            or object_points.shape[1] != 3
        ):
            return [
                self._insufficient_geometry(
                    subject,
                    object_,
                    context,
                )
            ]

        if (
            not np.isfinite(subject_points).all()
            or not np.isfinite(object_points).all()
        ):
            return [
                self._insufficient_geometry(
                    subject,
                    object_,
                    context,
                )
            ]

        tree = cKDTree(object_points)
        distances, _ = tree.query(
            subject_points,
            k=1,
        )

        distance_m = float(np.min(distances))

        if not np.isfinite(distance_m):
            return [
                self._insufficient_geometry(
                    subject,
                    object_,
                    context,
                )
            ]

        if distance_m < self.near_threshold:
            result = EvidenceResult.SUPPORTED
            predicate = "NEAR"
            threshold = self.near_threshold
            confidence = self._near_confidence(distance_m)

        elif distance_m > self.far_threshold:
            result = EvidenceResult.SUPPORTED
            predicate = "FAR"
            threshold = self.far_threshold
            confidence = self._far_confidence(distance_m)

        else:
            return []

        return [
            RelationEvidence(
                predicate=predicate,
                subject_id=subject.object_id,
                object_id=object_.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                result=result,
                value=distance_m,
                threshold=threshold,
                confidence=confidence,
                reference_frame="world",
                evidence_type="point_cloud_surface_distance",
                details={
                    "distance_m": distance_m,
                    "near_threshold_m": self.near_threshold,
                    "far_threshold_m": self.far_threshold,
                    "subject_point_count": int(
                        len(subject_points)
                    ),
                    "object_point_count": int(
                        len(object_points)
                    ),
                },
            )
        ]

    def _near_confidence(self, distance_m: float) -> float:
        if self.near_threshold <= 0.0:
            return 0.0

        return float(
            np.clip(
                1.0 - (
                    distance_m / self.near_threshold
                ),
                0.0,
                1.0,
            )
        )

    def _far_confidence(self, distance_m: float) -> float:
        separation = (
            distance_m - self.far_threshold
        )

        scale = self.far_threshold

        if scale <= 0.0:
            return 0.0

        return float(
            np.clip(
                separation / scale,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _insufficient_geometry(
        subject: Track,
        object_: Track,
        context: FrameContext,
    ) -> RelationEvidence:

        return RelationEvidence(
            predicate="NEAR",
            subject_id=subject.object_id,
            object_id=object_.object_id,
            frame_index=context.frame_index,
            timestamp=context.timestamp,
            result=EvidenceResult.INSUFFICIENT_GEOMETRY,
            value=None,
            threshold=None,
            confidence=0.0,
            reference_frame="world",
            evidence_type="missing_point_cloud_geometry",
            details={},
        )