import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence


class ContainmentRelationModule(RelationModule):
    """Computes INSIDE relation based on point cloud containment."""
    
    def __init__(self, min_inside_ratio: float = 0.5):
        self.min_inside_ratio = min_inside_ratio
        
    def predicates(self) -> List[str]:
        return ["INSIDE"]
        
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        
        subj_geo = context.observation_geometry.get(subject.object_id)
        obj_geo = context.observation_geometry.get(object.object_id)
        
        if not subj_geo or not obj_geo:
            return evidences
            
        if subj_geo.points_world is None:
            return evidences
            
        # Simplified: Check if subject points are inside object bounding box
        pts = subj_geo.points_world
        
        in_x = (pts[:, 0] >= obj_geo.bbox_min_world[0]) & (pts[:, 0] <= obj_geo.bbox_max_world[0])
        in_y = (pts[:, 1] >= obj_geo.bbox_min_world[1]) & (pts[:, 1] <= obj_geo.bbox_max_world[1])
        in_z = (pts[:, 2] >= obj_geo.bbox_min_world[2]) & (pts[:, 2] <= obj_geo.bbox_max_world[2])
        
        inside_mask = in_x & in_y & in_z
        inside_count = np.sum(inside_mask)
        
        if len(pts) > 0:
            ratio = inside_count / len(pts)
            if ratio >= self.min_inside_ratio:
                evidences.append(RelationEvidence(
                    predicate="INSIDE",
                    subject_id=subject.object_id,
                    object_id=object.object_id,
                    frame_index=context.frame_index,
                    timestamp=context.timestamp,
                    value=ratio,
                    threshold=self.min_inside_ratio,
                    confidence=ratio,
                    reference_frame="world",
                    evidence_type="point_containment",
                    details={"inside_ratio": float(ratio)}
                ))
                
        return evidences
