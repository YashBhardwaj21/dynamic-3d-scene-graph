import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult


class DistanceRelationModule(RelationModule):
    """Computes NEAR and FAR relations based on centroid distance."""
    
    def __init__(self, config=None, near_threshold: float = 0.40, far_threshold: float = 1.50):
        if config is not None:
            self.near_threshold = config.get("relations.distance.near_threshold", near_threshold)
            self.far_threshold = config.get("relations.distance.far_threshold", far_threshold)
        else:
            self.near_threshold = near_threshold
            self.far_threshold = far_threshold
        
    def predicates(self) -> List[str]:
        return ["NEAR", "FAR"]
        
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        
        # Check if both have geometry
        if subject.centroid_world is None or object.centroid_world is None:
            return evidences
            
        dist = np.linalg.norm(subject.centroid_world - object.centroid_world)
        
        # NEAR
        if dist < self.near_threshold:
            confidence = max(0.0, 1.0 - (dist / self.near_threshold))
            evidences.append(RelationEvidence(
                predicate="NEAR",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=dist,
                result=EvidenceResult.SUPPORTED,
                threshold=self.near_threshold,
                confidence=confidence,
                reference_frame="world",
                evidence_type="centroid_distance",
                details={"distance_m": float(dist)}
            ))
            
        # FAR
        elif dist > self.far_threshold:
            confidence = min(1.0, (dist - self.far_threshold) / self.far_threshold)
            evidences.append(RelationEvidence(
                predicate="FAR",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=dist,
                result=EvidenceResult.SUPPORTED,
                threshold=self.far_threshold,
                confidence=confidence,
                reference_frame="world",
                evidence_type="centroid_distance",
                details={"distance_m": float(dist)}
            ))
            
        # If in deadband (between near and far), we return no evidence
            
        return evidences
