import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult


class DepthOrderRelationModule(RelationModule):
    """Computes IN_FRONT_OF relation based on depth axis."""
    
    def __init__(self, config=None, depth_margin: float = 0.1):
        if config is not None and getattr(config, 'relations', None) and getattr(config.relations, 'depth_order', None):
            self.depth_margin = config.relations.depth_order.depth_margin
        else:
            self.depth_margin = depth_margin
        
    def predicates(self) -> List[str]:
        return ["IN_FRONT_OF"]
        
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        
        subj_geo = context.observation_geometry.get(subject.object_id)
        obj_geo = context.observation_geometry.get(object.object_id)
        
        if not subj_geo or not obj_geo:
            return evidences
            
        subj_pts = subj_geo.points_world_sampled if subj_geo.points_world_sampled is not None else subj_geo.points_world
        obj_pts = obj_geo.points_world_sampled if obj_geo.points_world_sampled is not None else obj_geo.points_world
        
        if subj_pts is None or obj_pts is None:
            return evidences
            
        # Project onto reference frame depth axis (forward is positive, so higher means further away)
        # A is IN_FRONT_OF B if A's furthest point is closer than B's closest point.
        subj_z = np.dot(subj_pts, context.reference_frame.depth_axis_world)
        obj_z = np.dot(obj_pts, context.reference_frame.depth_axis_world)
        
        gap_z = np.percentile(obj_z, 5) - np.percentile(subj_z, 95)
        
        if gap_z > self.depth_margin:
            confidence = min(1.0, (gap_z - self.depth_margin) / self.depth_margin)
            evidences.append(RelationEvidence(
                predicate="IN_FRONT_OF",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=gap_z,
                result=EvidenceResult.SUPPORTED,
                threshold=self.depth_margin,
                confidence=confidence,
                reference_frame="reference_frame",
                evidence_type="extent_directional",
                details={"gap_z": float(gap_z)}
            ))
        else:
            evidences.append(RelationEvidence(
                predicate="IN_FRONT_OF",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=gap_z,
                result=EvidenceResult.CONTRADICTED,
                threshold=self.depth_margin,
                confidence=1.0,
                reference_frame="reference_frame",
                evidence_type="extent_directional",
                details={"gap_z": float(gap_z)}
            ))
            
        return evidences
