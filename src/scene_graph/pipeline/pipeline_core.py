import time
import logging
from typing import List
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.data.tum_source import FramePacket
from scene_graph.perception.observation import Observation
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.temporal.object_state import ObjectStateMachine
from scene_graph.temporal.relation_state import RelationStateMachine
from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.relations.registry import RelationRegistry
from scene_graph.relations.inverse_algebra import derive_inverse_evidence

# Core Relation Modules
from scene_graph.relations.distance import DistanceRelationModule
from scene_graph.relations.support import SupportRelationModule
from scene_graph.relations.directional import DirectionalRelationModule
from scene_graph.relations.containment import ContainmentRelationModule
from scene_graph.relations.depth_order import DepthOrderRelationModule
from scene_graph.relations.occlusion import OcclusionRelationModule

from scene_graph.geometry.reference_frame import RelationReferenceFrame, CameraFrame
from scene_graph.relations.context import FrameContext, ObservationGeometry


class SceneGraphPipeline:
    """Core update logic for both online and offline execution.
    
    Receives a single FramePacket and a list of observations, and updates 
    the causal scene graph state.
    """
    
    def __init__(self, config: SceneGraphConfig):
        self.config = config
        
        # 1. Tracking
        self.tracker = CausalTracker(config)
        
        # 2. Relations
        self.registry = RelationRegistry(config)
        self._register_modules()
        
        # Temporal state machines
        self.object_state_machine = ObjectStateMachine(
            hysteresis_frames=config.temporal.object.max_missing_frames
        )
        self.relation_state_machine = RelationStateMachine(
            confirmation_frames=config.temporal.relation.confirm_frames,
            missing_frames=config.temporal.relation.max_missing_frames
        )
        
        # 3. Graph
        self.graph = TemporalSceneGraph()
        
        self.global_relation_frame = None
        
    def _register_modules(self):
        # Register all core relation modules
        self.registry.register(DistanceRelationModule(self.config))
        self.registry.register(SupportRelationModule(self.config))
        self.registry.register(DirectionalRelationModule(self.config))
        self.registry.register(ContainmentRelationModule(self.config))
        self.registry.register(DepthOrderRelationModule(self.config))
        self.registry.register(OcclusionRelationModule(self.config))

    def update(self, packet: FramePacket, observations: List[Observation]) -> TemporalSceneGraph:
        """Process a single frame and update the scene graph."""
        
        start_time = time.time()
        
        # 1. Tracking
        tracks = self.tracker.update(
            observations=observations,
            frame_index=packet.frame_index,
            timestamp=packet.timestamp
        )
        
        # 2. Object States
        object_states = self.object_state_machine.update(
            tracks=tracks,
            frame_index=packet.frame_index
        )
        
        # Update graph nodes
        self.graph.update_nodes(tracks, object_states)
        
        # 3. Geometry computation (only for active tracks)
        active_nodes = self.graph.get_active_nodes()
        active_tracks = [n.track for n in active_nodes]
        
        observation_geometry = {}
        for track in active_tracks:
            if not track.recent_observations:
                continue
                
            obs = track.recent_observations[-1]
            if obs.frame_index == packet.frame_index and obs.geometry_status == "VALID":
                geo = ObservationGeometry(
                    obs_id=obs.obs_id,
                    track_id=track.object_id,
                    centroid_world=obs.centroid_world,
                    bbox_min_world=obs.bbox_min_world,
                    bbox_max_world=obs.bbox_max_world,
                    depth_stats=obs.depth_stats if hasattr(obs, 'depth_stats') else None,
                    points_world_sampled=obs.points_world_sampled if hasattr(obs, 'points_world_sampled') else None,
                    points_world=obs.object_geometry.points_world if obs.object_geometry else None,
                    points_camera=obs.object_geometry.points_camera if obs.object_geometry else None,
                    mask=obs.get_mask(),
                    valid_point_count=obs.valid_point_count
                )
                observation_geometry[track.object_id] = geo
                    
        # 4. Build FrameContext for relations
        if packet.world_T_camera is not None:
            try:
                camera_frame = CameraFrame.from_camera_pose(packet.world_T_camera)
                if self.global_relation_frame is None:
                    # Initialize using gravity and heading from first camera pose
                    self.global_relation_frame = RelationReferenceFrame.from_gravity_and_heading(
                        origin_world=camera_frame.origin_world,
                        up_axis_world=np.array([0.0, 0.0, 1.0]), 
                        heading_world=camera_frame.depth_axis_world
                    )
                
                from scene_graph.geometry.camera import CameraIntrinsics
                if packet.camera_model and "fx" in packet.camera_model:
                    intrinsics = CameraIntrinsics(**packet.camera_model)
                else:
                    raise ValueError(f"Frame {packet.frame_index} missing required 'camera_model' with intrinsics.")
                    
                context = FrameContext(
                    frame_index=packet.frame_index,
                    timestamp=packet.timestamp,
                    intrinsics=intrinsics,
                    world_T_camera=packet.world_T_camera,
                    reference_frame=self.global_relation_frame,
                    camera_frame=camera_frame,
                    depth_image=packet.depth,
                    observation_geometry=observation_geometry
                )
                
                # 5. Compute Evidences
                raw_evidences = self.registry.compute_all(active_tracks, context)
                
                # 6. Apply Inverse Algebra
                # For asymmetric predicates (ON->UNDER, LEFT_OF->RIGHT_OF, etc.),
                # derive the inverse. For symmetric predicates (NEAR, FAR),
                # the same evidence applies to the swapped pair directly.
                from scene_graph.relations.inverse_algebra import SYMMETRIC
                all_evidences = []
                for ev in raw_evidences:
                    all_evidences.append(ev)
                    if ev.predicate not in SYMMETRIC:
                        all_evidences.append(derive_inverse_evidence(ev))
                    
                # 7. Relation States
                active_ids = {t.object_id for t in active_tracks}
                relation_states = self.relation_state_machine.update(
                    evidences=all_evidences,
                    frame_index=packet.frame_index,
                    active_object_ids=active_ids
                )
                
                # Update graph edges
                self.graph.update_edges(all_evidences, relation_states)
            except ValueError as e:
                logging.warning(f"Frame {packet.frame_index} invalid: {e}")
        
        # Update graph metadata
        self.graph.current_frame_index = packet.frame_index
        self.graph.current_timestamp = packet.timestamp
        
        return self.graph
