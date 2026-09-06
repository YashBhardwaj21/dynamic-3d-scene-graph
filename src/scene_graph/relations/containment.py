import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult


class ContainmentRelationModule(RelationModule):
    """Computes INSIDE relation based on point cloud containment."""
    
    def __init__(self, config=None, min_containment_ratio: float = 0.5):
        if config is not None:
            self.min_containment_ratio = config.relations.containment.min_containment_ratio
        else:
            self.min_containment_ratio = min_containment_ratio
        
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
            overlap_ratio = inside_count / len(pts)
            confidence = overlap_ratio
            if overlap_ratio >= self.min_containment_ratio:
                evidences.append(RelationEvidence(
                    predicate="INSIDE",
                    subject_id=subject.object_id,
                    object_id=object.object_id,
                    frame_index=context.frame_index,
                    timestamp=context.timestamp,
                    value=overlap_ratio,
                    result=EvidenceResult.SUPPORTED,
                    threshold=self.min_containment_ratio,
                    confidence=confidence,
                    reference_frame="world",
                    evidence_type="volume_overlap",
                    details={"containment_ratio": float(overlap_ratio)}
                ))
            else:
                evidences.append(RelationEvidence(
                    predicate="INSIDE",
                    subject_id=subject.object_id,
                    object_id=object.object_id,
                    frame_index=context.frame_index,
                    timestamp=context.timestamp,
                    value=overlap_ratio,
                    result=EvidenceResult.CONTRADICTED,
                    threshold=self.min_containment_ratio,
                    confidence=1.0 - confidence,
                    reference_frame="world",
                    evidence_type="volume_overlap",
                    details={"containment_ratio": float(overlap_ratio)}
                ))
                
        return evidences
