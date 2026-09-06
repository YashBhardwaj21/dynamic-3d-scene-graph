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
            self.depth_margin = config.get("relations.depth_order.depth_margin", depth_margin)
        else:
            self.depth_margin = depth_margin
        
    def predicates(self) -> List[str]:
        return ["IN_FRONT_OF"]
        
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        
        if subject.centroid_world is None or object.centroid_world is None:
            return evidences
            
        # Assuming Y is the depth axis in the reference frame 
        # (Y > 0 is deeper into the scene, Y=0 is front)
        # So subject Y < object Y means subject is in front of object
        diff_z = subject.centroid_world[1] - object.centroid_world[1]
        
        if diff_z < -self.depth_margin:
            confidence = min(1.0, (abs(diff_z) - self.depth_margin) / self.depth_margin)
            evidences.append(RelationEvidence(
                predicate="IN_FRONT_OF",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=diff_z,
                result=EvidenceResult.SUPPORTED,
                threshold=self.depth_margin,
                confidence=confidence,
                reference_frame="camera",
                evidence_type="camera_z_distance",
                details={"delta_z": float(diff_z)}
            ))
            
        return evidences
