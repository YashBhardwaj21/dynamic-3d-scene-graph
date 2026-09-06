import numpy as np
from typing import List, Tuple
from scipy.spatial import cKDTree

from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult


class DistanceRelationModule(RelationModule):
    """Computes NEAR and FAR relations based on centroid distance."""
    
    def __init__(self, config=None, near_threshold: float = 0.40, far_threshold: float = 1.50):
        if config is not None and getattr(config, 'relations', None) and getattr(config.relations, 'distance', None):
            self.near_threshold = config.relations.distance.near_threshold
            self.far_threshold = config.relations.distance.far_threshold
        else:
            self.near_threshold = near_threshold
            self.far_threshold = far_threshold
        
    def predicates(self) -> List[str]:
        return ["NEAR", "FAR"]
        
    def compute_pairs(self, pairs: List[Tuple[Track, Track]], context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        # Filter for canonical pairs since distance is symmetric
        canonical_pairs = []
        for subj, obj in pairs:
            if subj.object_id < obj.object_id:
                canonical_pairs.append((subj, obj))
                
        for subj, obj in canonical_pairs:
            evidences.extend(self.compute(subj, obj, context))
            
        return evidences
        
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        evidences = []
        
        # Check if both have observation geometry for AABB
        if subject.object_id not in context.observation_geometry or object.object_id not in context.observation_geometry:
            # Fallback to centroids if full geometry is missing
            if subject.centroid_world is None or object.centroid_world is None:
                return evidences
            dist = np.linalg.norm(subject.centroid_world - object.centroid_world)
            evidence_type = "centroid_distance"
        else:
            subj_geo = context.observation_geometry[subject.object_id]
            obj_geo = context.observation_geometry[object.object_id]
            
            if subj_geo.points_world_sampled is not None and obj_geo.points_world_sampled is not None and len(subj_geo.points_world_sampled) > 0 and len(obj_geo.points_world_sampled) > 0:
                tree = cKDTree(subj_geo.points_world_sampled)
                distances, _ = tree.query(obj_geo.points_world_sampled, k=1)
                dist = np.min(distances)
                evidence_type = "point_cloud_distance"
            else:
                # Fallback to AABB
                min_diff = np.maximum(subj_geo.bbox_min_world - obj_geo.bbox_max_world, 0)
                max_diff = np.maximum(obj_geo.bbox_min_world - subj_geo.bbox_max_world, 0)
                dist_sq = np.sum(np.square(np.maximum(min_diff, max_diff)))
                dist = np.sqrt(dist_sq)
                evidence_type = "aabb_distance"
        
        # NEAR
        if dist < self.near_threshold:
            confidence = max(0.0, 1.0 - (dist / self.near_threshold))
            evidences.append(RelationEvidence(
                predicate="NEAR",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=dist,
                result=EvidenceResult.SUPPORTED,
                threshold=self.near_threshold,
                confidence=confidence,
                reference_frame="world",
                evidence_type=evidence_type,
                details={"distance_m": float(dist)}
            ))
            
            # Emitting CONTRADICTED for FAR ensures the state machine sees it's explicitly not FAR
            evidences.append(RelationEvidence(
                predicate="FAR",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=dist,
                result=EvidenceResult.CONTRADICTED,
                threshold=self.far_threshold,
                confidence=confidence,
                reference_frame="world",
                evidence_type=evidence_type,
                details={"distance_m": float(dist)}
            ))
            
        # FAR
        elif dist > self.far_threshold:
            confidence = min(1.0, (dist - self.far_threshold) / self.far_threshold)
            evidences.append(RelationEvidence(
                predicate="FAR",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=dist,
                result=EvidenceResult.SUPPORTED,
                threshold=self.far_threshold,
                confidence=confidence,
                reference_frame="world",
                evidence_type=evidence_type,
                details={"distance_m": float(dist)}
            ))
            
            evidences.append(RelationEvidence(
                predicate="NEAR",
                subject_id=subject.object_id,
                object_id=object.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                value=dist,
                result=EvidenceResult.CONTRADICTED,
                threshold=self.near_threshold,
                confidence=confidence,
                reference_frame="world",
                evidence_type=evidence_type,
                details={"distance_m": float(dist)}
            ))
            
        # If in deadband (between near and far), we return no evidence
            
        return evidences
