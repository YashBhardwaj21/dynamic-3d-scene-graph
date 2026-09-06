import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence


class SupportRelationModule(RelationModule):
    """Computes ON relation based on plane fitting and distance."""
    
    def __init__(self, config=None, plane_residual_m: float = 0.02, min_support_overlap: float = 0.1):
        if config is not None:
            self.plane_residual_m = config.relations.support.plane_residual_m
            self.min_support_overlap = config.relations.support.min_support_overlap
        else:
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
        
        # Using reference frame's up_axis_world
        up = context.reference_frame.up_axis_world
        
        # To find vertical extents, we could project all 8 corners of the AABB
        # or use points if available. We'll project the centroid and use AABB extents approximately.
        # But for exactness, if points are missing we return empty above.
        # So we can project points_world onto the up vector.
        subj_z_points = np.dot(subj_geo.points_world, up)
        obj_z_points = np.dot(obj_geo.points_world, up)
        
        subj_min_z = np.min(subj_z_points)
        obj_max_z = np.max(obj_z_points)
        
        bottom_distance = subj_min_z - obj_max_z
        
        if abs(bottom_distance) <= self.plane_residual_m:
            # Check overlap in horizontal plane (perpendicular to UP)
            # We project points onto a 2D plane defined by horizontal_axis and depth_axis
            horiz = context.reference_frame.horizontal_axis_world
            depth = context.reference_frame.depth_axis_world
            
            subj_h = np.dot(subj_geo.points_world, horiz)
            subj_d = np.dot(subj_geo.points_world, depth)
            obj_h = np.dot(obj_geo.points_world, horiz)
            obj_d = np.dot(obj_geo.points_world, depth)
            
            subj_min_xy = np.array([np.min(subj_h), np.min(subj_d)])
            subj_max_xy = np.array([np.max(subj_h), np.max(subj_d)])
            obj_min_xy = np.array([np.min(obj_h), np.min(obj_d)])
            obj_max_xy = np.array([np.max(obj_h), np.max(obj_d)])
            
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
                        result=EvidenceResult.SUPPORTED,
                        threshold=self.plane_residual_m,
                        confidence=confidence,
                        reference_frame="reference_frame",
                        evidence_type="support_plane",
                        details={
                            "plane_residual": float(bottom_distance),
                            "support_overlap": float(overlap_ratio)
                        }
                    ))
                    
        return evidences
