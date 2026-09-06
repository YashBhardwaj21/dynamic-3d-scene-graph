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
            self.margin_x = config.relations.directional.margin_x
            self.margin_y = config.relations.directional.margin_y
        else:
            self.margin_x = margin_x
            self.margin_y = margin_y
        
    def predicates(self) -> List[str]:
        return ["LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW"]
        
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        
        if subject.centroid_world is None or object.centroid_world is None:
            return evidences
            
        # Vector from object to subject
        delta = subject.centroid_world - object.centroid_world
        
        # Project onto reference frame axes
        dx = np.dot(delta, context.reference_frame.horizontal_axis_world)
        dy = np.dot(delta, context.reference_frame.up_axis_world)
        
        # Horizontal (LEFT_OF / RIGHT_OF)
        # dx > 0 means subject is RIGHT of object
        # dx < 0 means subject is LEFT of object
        abs_dx = abs(dx)
        if abs_dx > self.margin_x:
            confidence = min(1.0, (abs_dx - self.margin_x) / self.margin_x)
            predicate = "RIGHT_OF" if dx > 0 else "LEFT_OF"
            evidences.append(RelationEvidence(
                predicate=predicate,
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=abs_dx,
                result=EvidenceResult.SUPPORTED,
                threshold=self.margin_x,
                confidence=confidence,
                reference_frame="reference_frame",
                evidence_type="centroid_directional",
                details={"dx": float(dx)}
            ))
            
        # Vertical (ABOVE / BELOW)
        # dy > 0 means subject is ABOVE object
        # dy < 0 means subject is BELOW object
        abs_dy = abs(dy)
        if abs_dy > self.margin_y:
            confidence = min(1.0, (abs_dy - self.margin_y) / self.margin_y)
            predicate = "ABOVE" if dy > 0 else "BELOW"
            evidences.append(RelationEvidence(
                predicate=predicate,
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=abs_dy,
                result=EvidenceResult.SUPPORTED,
                threshold=self.margin_y,
                confidence=confidence,
                reference_frame="reference_frame",
                evidence_type="centroid_directional",
                details={"dy": float(dy)}
            ))
            
        return evidences
