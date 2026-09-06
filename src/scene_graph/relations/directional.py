import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence


class DirectionalRelationModule(RelationModule):
    """Computes LEFT_OF and ABOVE directional relations."""
    
    def __init__(self, axis_margin_m: float = 0.05):
        self.axis_margin_m = axis_margin_m
        
    def predicates(self) -> List[str]:
        return ["LEFT_OF", "ABOVE"]
        
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        
        if subject.centroid_world is None or object.centroid_world is None:
            return evidences
            
        # Transform centroids into the relation reference frame
        # Actually, let's assume centroid_world is already aligned to the reference frame for simplicity
        # (This depends on where we applied the alignment transform. For now, assume it's aligned to world Z-up)
        
        # LEFT_OF
        # Assuming X is horizontal right (so subject X < object X means subject is left of object)
        dx = subject.centroid_world[0] - object.centroid_world[0]
        
        if dx < -self.axis_margin_m:
            confidence = min(1.0, (abs(dx) - self.axis_margin_m) / self.axis_margin_m)
            evidences.append(RelationEvidence(
                predicate="LEFT_OF",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=dx,
                threshold=self.axis_margin_m,
                confidence=confidence,
                reference_frame="world",
                evidence_type="centroid_directional",
                details={"dx": float(dx)}
            ))
            
        # ABOVE
        # Assuming Z is gravity up (so subject Z > object Z means subject is above object)
        dz = subject.centroid_world[2] - object.centroid_world[2]
        
        if dz > self.axis_margin_m:
            confidence = min(1.0, (abs(dz) - self.axis_margin_m) / self.axis_margin_m)
            evidences.append(RelationEvidence(
                predicate="ABOVE",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=dz,
                threshold=self.axis_margin_m,
                confidence=confidence,
                reference_frame="world",
                evidence_type="centroid_directional",
                details={"dz": float(dz)}
            ))
            
        return evidences
