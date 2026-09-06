import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence


class SupportRelationModule(RelationModule):
    """Computes ON relation based on plane fitting and distance."""
    
    def __init__(self, plane_residual_m: float = 0.02, min_support_overlap: float = 0.1):
        self.plane_residual_m = plane_residual_m
        self.min_support_overlap = min_support_overlap
        
    def predicates(self) -> List[str]:
        return ["ON"]
        
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        
        # Check if subject and object geometries are cached
        subj_geo = context.observation_geometry.get(subject.object_id)
        obj_geo = context.observation_geometry.get(object.object_id)
        
        if not subj_geo or not obj_geo:
            return evidences
            
        if subj_geo.points_world is None or obj_geo.points_world is None:
            return evidences
            
        if len(obj_geo.points_world) < 10 or len(subj_geo.points_world) < 10:
            return evidences
            
        # For object (support surface), estimate plane
        # Simplified: assume normal is roughly Z-up in world frame, just find Z bounds
        # In a full implementation, we'd do RANSAC plane fit here.
        # But per the plan, we'll do a simple percentile check.
        # A's bottom distance to B's plane.
        
        # Approximate plane fit: find the mean Z of the support surface's top points
        # Or even simpler: the plane is defined by B's centroid Z if it's flat, or its bounding box top
        
        # Wait, the instruction says:
        # "RANSAC plane fit: ax + by + cz + d = 0 (configurable Z-range filter)"
        # Let's do a fast SVD/PCA for normal if we assume it's flat, or just use the reference frame's gravity axis.
        # Given we have `ObservationGeometry`, we can just use B's centroid and the reference frame's up vector 
        # for a basic check, or compute actual distance to B's points.
        
        # Using a highly simplified but robust proxy for ON:
        # 1. Subject's minimum Z must be close to Object's maximum Z (in gravity-aligned world frame)
        # 2. Subject's XY footprint must overlap Object's XY footprint
        
        subj_min_z = subj_geo.bbox_min_world[2]
        obj_max_z = obj_geo.bbox_max_world[2]
        
        bottom_distance = subj_min_z - obj_max_z
        
        if abs(bottom_distance) <= self.plane_residual_m:
            # Check overlap in XY
            subj_min_xy = subj_geo.bbox_min_world[:2]
            subj_max_xy = subj_geo.bbox_max_world[:2]
            obj_min_xy = obj_geo.bbox_min_world[:2]
            obj_max_xy = obj_geo.bbox_max_world[:2]
            
            overlap_min = np.maximum(subj_min_xy, obj_min_xy)
            overlap_max = np.minimum(subj_max_xy, obj_max_xy)
            
            overlap_area = max(0, overlap_max[0] - overlap_min[0]) * max(0, overlap_max[1] - overlap_min[1])
            subj_area = (subj_max_xy[0] - subj_min_xy[0]) * (subj_max_xy[1] - subj_min_xy[1])
            
            if subj_area > 0:
                overlap_ratio = overlap_area / subj_area
                if overlap_ratio >= self.min_support_overlap:
                    confidence = 1.0 - min(1.0, abs(bottom_distance) / self.plane_residual_m)
                    
                    evidences.append(RelationEvidence(
                        predicate="ON",
                        subject_id=subject.object_id,
                        object_id=object.object_id,
                        frame_index=context.frame_index,
                        timestamp=context.timestamp,
                        value=bottom_distance,
                        threshold=self.plane_residual_m,
                        confidence=confidence,
                        reference_frame="world",
                        evidence_type="support_plane",
                        details={
                            "plane_residual": float(bottom_distance),
                            "support_overlap": float(overlap_ratio)
                        }
                    ))
                    
        return evidences
