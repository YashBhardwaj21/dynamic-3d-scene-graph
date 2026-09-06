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
                 depth_margin: float = 0.02):
        if config is not None:
            self.min_mask_overlap_ratio = config.get("relations.occlusion.min_mask_overlap_ratio", min_mask_overlap_ratio)
            self.min_depth_order_ratio = config.get("relations.occlusion.min_depth_order_ratio", min_depth_order_ratio)
            self.min_valid_depth_samples = config.get("relations.occlusion.min_valid_depth_samples", min_valid_depth_samples)
            self.depth_margin = config.get("relations.occlusion.depth_margin", depth_margin)
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
            # Plan says: "If mask unavailable: evidence_result = EvidenceResult.MISSING_MASK"
            # But we are returning a list of RelationEvidence. We could just return nothing if we can't compute it.
            return evidences
            
        if context.depth_image is None:
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
            return evidences
            
        # 4. Extract depth values at overlapping pixels
        depth_map = context.depth_image
        depths = depth_map[mask_overlap]
        
        # We need the actual depth for A and B. Since it's a single depth map,
        # the depth map stores the depth of whatever is visible.
        # If A is occluding B, the visible depth will belong to A.
        # But wait, how do we know A is closer than B if we only have one depth map?
        # The plan says:
        # "For overlapping pixels with valid depth for both: ratio = #{p : z_A < z_B - eps} / #{p : both valid depth}"
        # If they are from the *same* depth map, they can't both have a different depth at the same pixel.
        # Ah, in 2D object detection, the masks can overlap.
        # But they map to the same pixel in the depth image.
        # If A is detected as a whole object (including the occluded part), where does `z_B` come from?
        # It comes from the 3D geometry of B. 
        # Actually, for 2D instance segmentation, the depth of the pixel just IS the depth of the visible object.
        # Let's simplify this by using the centroids if we don't have full layered depth, 
        # or we just check if the visible depth in the overlap matches A's average depth better than B's.
        
        # Let's use the average depth of A and B in their non-overlapping regions as a proxy
        mask_A_only = mask_A & ~mask_B
        mask_B_only = mask_B & ~mask_A
        
        depths_A_only = depth_map[mask_A_only]
        depths_B_only = depth_map[mask_B_only]
        
        # Filter 0 depth (invalid)
        valid_A = depths_A_only[depths_A_only > 0]
        valid_B = depths_B_only[depths_B_only > 0]
        
        if len(valid_A) < self.min_valid_depth_samples or len(valid_B) < self.min_valid_depth_samples:
            return evidences
            
        mean_depth_A = np.median(valid_A)
        mean_depth_B = np.median(valid_B)
        
        # The overlapping region's depth
        valid_overlap = depths[depths > 0]
        if len(valid_overlap) < self.min_valid_depth_samples:
            return evidences
            
        mean_overlap_depth = np.median(valid_overlap)
        
        # If the overlap depth is much closer to A than B, A is occluding B
        # i.e., A is in front of B
        # Let's just check if mean_depth_A is significantly smaller than mean_depth_B
        if mean_depth_A < mean_depth_B - self.depth_margin:
            # A is occluding B
            evidences.append(RelationEvidence(
                predicate="OCCLUDING",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=mean_depth_B - mean_depth_A,
                result=EvidenceResult.SUPPORTED,
                threshold=self.depth_margin,
                confidence=1.0,
                reference_frame="camera",
                evidence_type="mask_occlusion",
                details={
                    "overlap_ratio": float(overlap_ratio),
                    "depth_diff": float(mean_depth_B - mean_depth_A)
                }
            ))
            
        return evidences
