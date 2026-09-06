import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult


class DirectionalRelationModule(RelationModule):
    """Computes LEFT_OF and ABOVE directional relations."""
    
    def __init__(self, config=None, margin_x: float = 0.05, margin_y: float = 0.05):
        if config is not None:
            self.margin_x = config.get("relations.directional.margin_x", margin_x)
            self.margin_y = config.get("relations.directional.margin_y", margin_y)
        else:
            self.margin_x = margin_x
            self.margin_y = margin_y
        
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
        abs_dx = abs(dx)
        
        if dx < -self.margin_x:
            confidence = min(1.0, (abs_dx - self.margin_x) / self.margin_x)
            evidences.append(RelationEvidence(
                predicate="LEFT_OF",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=abs_dx,
                result=EvidenceResult.SUPPORTED,
                threshold=self.margin_x,
                confidence=confidence,
                reference_frame="camera",
                evidence_type="camera_x_distance",
                details={"delta_x": float(dx)}
            ))
            
        # ABOVE
        # Assuming Z is gravity up (so subject Z > object Z means subject is above object)
        dz = subject.centroid_world[2] - object.centroid_world[2]
        
        if dz > self.margin_y:
            confidence = min(1.0, (dz - self.margin_y) / self.margin_y)
            evidences.append(RelationEvidence(
                predicate="ABOVE",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=dz,
                result=EvidenceResult.SUPPORTED,
                threshold=self.margin_y,
                confidence=confidence,
                reference_frame="world",
                evidence_type="centroid_directional",
                details={"dz": float(dz)}
            ))
            
        return evidences
