import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult


class DepthOrderRelationModule(RelationModule):
    """Computes IN_FRONT_OF relation based on depth axis."""
    
    def __init__(self, config=None, depth_margin: float = 0.1):
        if config is not None:
            self.depth_margin = config.relations.depth_order.depth_margin
        else:
            self.depth_margin = depth_margin
        
    def predicates(self) -> List[str]:
        return ["IN_FRONT_OF", "BEHIND"]
        
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        
        if subject.centroid_world is None or object.centroid_world is None:
            return evidences
            
        # Vector from object to subject
        delta = subject.centroid_world - object.centroid_world
        
        # Project onto reference frame depth axis (forward is positive)
        # dz > 0 means subject is further away (BEHIND object)
        # dz < 0 means subject is closer (IN_FRONT_OF object)
        dz = np.dot(delta, context.reference_frame.depth_axis_world)
        abs_dz = abs(dz)
        
        if abs_dz > self.depth_margin:
            confidence = min(1.0, (abs_dz - self.depth_margin) / self.depth_margin)
            predicate = "BEHIND" if dz > 0 else "IN_FRONT_OF"
            evidences.append(RelationEvidence(
                predicate=predicate,
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=abs_dz,
                result=EvidenceResult.SUPPORTED,
                threshold=self.depth_margin,
                confidence=confidence,
                reference_frame="reference_frame",
                evidence_type="centroid_directional",
                details={"dz": float(dz)}
            ))
            
        return evidences
