import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult


class DirectionalRelationModule(RelationModule):
    """Computes LEFT_OF and ABOVE directional relations."""
    
    def __init__(self, config=None, margin_x: float = 0.05, margin_y: float = 0.05):
        if config is not None and getattr(config, 'relations', None) and getattr(config.relations, 'directional', None):
            self.margin_x = config.relations.directional.margin_x
            self.margin_y = config.relations.directional.margin_y
        else:
            self.margin_x = margin_x
            self.margin_y = margin_y
        
    def predicates(self) -> List[str]:
        return ["LEFT_OF", "ABOVE"]
        
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
            
        # Project points onto reference frame axes
        # Horizontal (LEFT_OF) -> looking from camera, horizontal axis points RIGHT
        # A is LEFT of B if A's max right extent is less than B's min right extent.
        subj_x = np.dot(subj_pts, context.reference_frame.horizontal_axis_world)
        obj_x = np.dot(obj_pts, context.reference_frame.horizontal_axis_world)
        
        gap_x = np.percentile(obj_x, 5) - np.percentile(subj_x, 95)
        
        if gap_x > self.margin_x:
            confidence = min(1.0, (gap_x - self.margin_x) / self.margin_x)
            evidences.append(RelationEvidence(
                predicate="LEFT_OF",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=gap_x,
                result=EvidenceResult.SUPPORTED,
                threshold=self.margin_x,
                confidence=confidence,
                reference_frame="reference_frame",
                evidence_type="extent_directional",
                details={"gap_x": float(gap_x)}
            ))
        else:
            evidences.append(RelationEvidence(
                predicate="LEFT_OF",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=gap_x,
                result=EvidenceResult.CONTRADICTED,
                threshold=self.margin_x,
                confidence=1.0,
                reference_frame="reference_frame",
                evidence_type="extent_directional",
                details={"gap_x": float(gap_x)}
            ))
            
        # Vertical (ABOVE)
        # A is ABOVE B if A's min y extent is greater than B's max y extent
        subj_y = np.dot(subj_pts, context.reference_frame.up_axis_world)
        obj_y = np.dot(obj_pts, context.reference_frame.up_axis_world)
        
        gap_y = np.percentile(subj_y, 5) - np.percentile(obj_y, 95)
        
        if gap_y > self.margin_y:
            confidence = min(1.0, (gap_y - self.margin_y) / self.margin_y)
            evidences.append(RelationEvidence(
                predicate="ABOVE",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=gap_y,
                result=EvidenceResult.SUPPORTED,
                threshold=self.margin_y,
                confidence=confidence,
                reference_frame="reference_frame",
                evidence_type="extent_directional",
                details={"gap_y": float(gap_y)}
            ))
        else:
            evidences.append(RelationEvidence(
                predicate="ABOVE",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=gap_y,
                result=EvidenceResult.CONTRADICTED,
                threshold=self.margin_y,
                confidence=1.0,
                reference_frame="reference_frame",
                evidence_type="extent_directional",
                details={"gap_y": float(gap_y)}
            ))
            
        return evidences
