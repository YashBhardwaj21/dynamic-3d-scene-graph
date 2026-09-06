import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence


class OcclusionRelationModule(RelationModule):
    """Computes OCCLUDING relation based on mask overlap and depth values."""
    
    def __init__(self, config=None, min_mask_overlap_ratio: float = 0.05, 
                 min_depth_order_ratio: float = 0.8,
                 min_valid_depth_samples: int = 10,
                 depth_margin: float = 0.05):
        if config is not None and getattr(config, 'relations', None) and getattr(config.relations, 'occlusion', None):
            self.min_mask_overlap_ratio = config.relations.occlusion.min_mask_overlap_ratio
            self.min_depth_order_ratio = config.relations.occlusion.min_depth_order_ratio
            self.min_valid_depth_samples = config.relations.occlusion.min_valid_depth_samples
            self.depth_margin = config.relations.occlusion.depth_margin
        else:
            self.min_mask_overlap_ratio = min_mask_overlap_ratio
            self.min_depth_order_ratio = min_depth_order_ratio
            self.min_valid_depth_samples = min_valid_depth_samples
            self.depth_margin = depth_margin
        
    def predicates(self) -> List[str]:
        return ["OCCLUDING"]
        
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        
        subj_geo = context.observation_geometry.get(subject.object_id)
        obj_geo = context.observation_geometry.get(object.object_id)
        
        if not subj_geo or not obj_geo:
            return evidences
            
        if subj_geo.mask is None or obj_geo.mask is None:
            return evidences
            
        # 1. Decode masks (already decoded in ObservationGeometry as boolean array)
        mask_A = subj_geo.mask
        mask_B = obj_geo.mask
        
        # Ensure masks are the same shape
        if mask_A.shape != mask_B.shape:
            return evidences
            
        # 2. Compute overlapping pixels
        mask_overlap = mask_A & mask_B
        overlap_count = np.sum(mask_overlap)
        
        size_A = np.sum(mask_A)
        size_B = np.sum(mask_B)
        
        if size_A == 0 or size_B == 0:
            return evidences
            
        # 3. If overlap ratio < threshold -> no evidence
        overlap_ratio = overlap_count / min(size_A, size_B)
        if overlap_ratio < self.min_mask_overlap_ratio:
            evidences.append(RelationEvidence(
                predicate="OCCLUDING",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=overlap_ratio,
                result=EvidenceResult.CONTRADICTED,
                threshold=self.min_mask_overlap_ratio,
                confidence=1.0,
                reference_frame="camera",
                evidence_type="mask_occlusion",
                details={"overlap_ratio": float(overlap_ratio)}
            ))
            return evidences
            
        # 4. Use depth statistics to determine order
        if not subj_geo.depth_stats or not obj_geo.depth_stats:
            return evidences
            
        depth_A = subj_geo.depth_stats.get("p95", subj_geo.depth_stats.get("median"))
        depth_B = obj_geo.depth_stats.get("p05", obj_geo.depth_stats.get("median"))
        
        if depth_A is None or depth_B is None:
            return evidences
        
        # If A is significantly closer than B, A is occluding B
        if depth_A < depth_B - self.depth_margin:
            confidence = min(1.0, (depth_B - depth_A) / (2 * self.depth_margin))
            evidences.append(RelationEvidence(
                predicate="OCCLUDING",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=overlap_ratio,
                result=EvidenceResult.SUPPORTED,
                threshold=self.min_mask_overlap_ratio,
                confidence=confidence,
                reference_frame="camera",
                evidence_type="mask_occlusion",
                details={"overlap_ratio": float(overlap_ratio), "depth_diff": float(depth_B - depth_A)}
            ))
        else:
            # If A is not strictly in front of B, it's contradicted
            evidences.append(RelationEvidence(
                predicate="OCCLUDING",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=overlap_ratio,
                result=EvidenceResult.CONTRADICTED,
                threshold=self.min_mask_overlap_ratio,
                confidence=0.5,
                reference_frame="camera",
                evidence_type="mask_occlusion",
                details={"overlap_ratio": float(overlap_ratio)}
            ))
            
        return evidences
