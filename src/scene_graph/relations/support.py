import numpy as np
from typing import List

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult
from scene_graph.geometry.plane import fit_plane_ransac


class SupportRelationModule(RelationModule):
    """Computes ON relation based on plane fitting and distance."""
    
    def __init__(self, config=None, plane_residual_m: float = 0.02, min_support_overlap: float = 0.1):
        if config is not None and getattr(config, 'relations', None) and getattr(config.relations, 'support', None):
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
            
        subj_pts = subj_geo.points_world_sampled if subj_geo.points_world_sampled is not None else subj_geo.points_world
        obj_pts = obj_geo.points_world_sampled if obj_geo.points_world_sampled is not None else obj_geo.points_world
        
        if subj_pts is None or obj_pts is None:
            return evidences
            
        if len(obj_pts) < 10 or len(subj_pts) < 10:
            return evidences
            
        union_pts = np.vstack((subj_pts, obj_pts))
        
        # Fit plane using robust RANSAC estimator
        plane = fit_plane_ransac(union_pts, distance_threshold=self.plane_residual_m)
        if plane is None:
            return evidences
            
        # Using reference frame's up_axis_world to check if plane is horizontal
        up = context.reference_frame.up_axis_world
        normal_dot = np.abs(np.dot(plane.normal, up))
        
        # We expect a support plane to be roughly horizontal
        if normal_dot < 0.8:
            return evidences
            
        # Enforce normal points "up"
        normal = plane.normal if np.dot(plane.normal, up) > 0 else -plane.normal
        distance = plane.distance if np.dot(plane.normal, up) > 0 else -plane.distance
        
        # Calculate subject and object minimum and maximum height over the plane
        subj_h = np.dot(subj_pts, normal) + distance
        obj_h = np.dot(obj_pts, normal) + distance
        
        subj_min_z = np.percentile(subj_h, 5)
        obj_max_z = np.percentile(obj_h, 95)
        
        bottom_distance = subj_min_z - obj_max_z
        
        # Check if subject is resting on the object's upper surface
        if abs(bottom_distance) <= self.plane_residual_m * 2:
            # For overlap, project points onto the support plane
            # create basis vectors on the plane
            v1 = np.cross(normal, np.array([1, 0, 0]))
            if np.linalg.norm(v1) < 0.1:
                v1 = np.cross(normal, np.array([0, 1, 0]))
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(normal, v1)
            
            subj_u = np.dot(subj_pts, v1)
            subj_v = np.dot(subj_pts, v2)
            obj_u = np.dot(obj_pts, v1)
            obj_v = np.dot(obj_pts, v2)
            
            subj_min_uv = np.array([np.min(subj_u), np.min(subj_v)])
            subj_max_uv = np.array([np.max(subj_u), np.max(subj_v)])
            obj_min_uv = np.array([np.min(obj_u), np.min(obj_v)])
            obj_max_uv = np.array([np.max(obj_u), np.max(obj_v)])
            
            overlap_min = np.maximum(subj_min_uv, obj_min_uv)
            overlap_max = np.minimum(subj_max_uv, obj_max_uv)
            
            overlap_area = max(0, overlap_max[0] - overlap_min[0]) * max(0, overlap_max[1] - overlap_min[1])
            subj_area = (subj_max_uv[0] - subj_min_uv[0]) * (subj_max_uv[1] - subj_min_uv[1])
            
            if subj_area > 0:
                overlap_ratio = overlap_area / subj_area
                if overlap_ratio >= self.min_support_overlap:
                    confidence = 1.0 - min(1.0, abs(bottom_distance) / (self.plane_residual_m * 2))
                    
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
                            "support_overlap": float(overlap_ratio),
                            "plane_normal": plane.normal.tolist(),
                            "plane_fit_residual": plane.residual_mean
                        }
                    ))
                    
        return evidences
