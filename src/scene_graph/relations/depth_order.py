import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence


class DepthOrderRelationModule(RelationModule):
    """Computes IN_FRONT_OF relation based on depth axis."""
    
    def __init__(self, axis_margin_m: float = 0.05):
        self.axis_margin_m = axis_margin_m
        
    def predicates(self) -> List[str]:
        return ["IN_FRONT_OF"]
        
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        
        if subject.centroid_world is None or object.centroid_world is None:
            return evidences
            
        # Assuming Y is the depth axis in the reference frame 
        # (Y > 0 is deeper into the scene, Y=0 is front)
        # So subject Y < object Y means subject is in front of object
        dy = subject.centroid_world[1] - object.centroid_world[1]
        
        if dy < -self.axis_margin_m:
            confidence = min(1.0, (abs(dy) - self.axis_margin_m) / self.axis_margin_m)
            evidences.append(RelationEvidence(
                predicate="IN_FRONT_OF",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=dy,
                threshold=self.axis_margin_m,
                confidence=confidence,
                reference_frame="world",
                evidence_type="centroid_directional",
                details={"dy": float(dy)}
            ))
            
        return evidences
