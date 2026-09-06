import time
from typing import List, Dict, Optional

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.perception.yoloe_detector import YOLOEDetector
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.relations.context import FrameContext
from scene_graph.relations.registry import RelationRegistry
from scene_graph.geometry.reference_frame import RelationReferenceFrame
from scene_graph.relations.distance import DistanceRelationModule
from scene_graph.relations.support import SupportRelationModule
from scene_graph.relations.directional import DirectionalRelationModule
from scene_graph.relations.containment import ContainmentRelationModule
from scene_graph.relations.depth_order import DepthOrderRelationModule
from scene_graph.relations.occlusion import OcclusionRelationModule
from scene_graph.relations.inverse_algebra import derive_inverse_evidence

from scene_graph.temporal.object_state import ObjectStateMachine
from scene_graph.temporal.relation_state import RelationStateMachine
from scene_graph.graph.temporal_graph import TemporalSceneGraph


class OnlinePipeline:
    """
    Main online execution pipeline coordinating observation, tracking, 
    geometry, relations, and the temporal scene graph.
    """
    
    def __init__(self, config: SceneGraphConfig):
        self.config = config
        
        # 1. Perception
        # Set up detector
        model_path = config.perception.model_path
        
        intrinsics = None
        depth_model = None
        if config.camera:
            from scene_graph.geometry.camera import CameraIntrinsics
            intrinsics = CameraIntrinsics(
                fx=config.camera.fx, fy=config.camera.fy,
                cx=config.camera.cx, cy=config.camera.cy,
                width=config.camera.width, height=config.camera.height
            )
        if config.depth:
            from scene_graph.geometry.camera import DepthModel
            depth_model = DepthModel(scale=config.depth.scale)
            
        allowed_classes = None
        if config.perception.classes:
            allowed_classes = set(config.perception.classes)
        elif config.perception.vocabulary and config.vocabularies and config.perception.vocabulary in config.vocabularies:
            allowed_classes = set(config.vocabularies[config.perception.vocabulary].classes)
            
        if not allowed_classes:
            raise ValueError("No classes provided for open-vocabulary detector. Set perception.classes or a valid perception.vocabulary.")
            
        self.detector = YOLOEDetector(
            model_path=model_path,
            confidence_threshold=config.perception.confidence_threshold,
            allowed_classes=allowed_classes,
            intrinsics=intrinsics,
            depth_model=depth_model
        )
        
        # 2. Tracking
        self.tracker = CausalTracker(config)
        
        # 3. Geometry (stateless helper `compute_object_geometry` used inside update)
        
        # 4. Relations
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
        
        # 6. Graph
        self.graph = TemporalSceneGraph()
        
    def _register_modules(self):
        # Register all core relation modules
        self.registry.register(DistanceRelationModule(self.config))
        self.registry.register(SupportRelationModule(self.config))
        self.registry.register(DirectionalRelationModule(self.config))
        self.registry.register(ContainmentRelationModule(self.config))
        self.registry.register(DepthOrderRelationModule(self.config))
        self.registry.register(OcclusionRelationModule(self.config))

    def update(self, packet: FramePacket) -> TemporalSceneGraph:
        """Process a single frame and update the scene graph."""
        
        start_time = time.time()
        
        # 1. Observations
        observations = self.detector.detect(packet)
        
        # 2. Tracking
        tracks = self.tracker.update(
            observations=observations,
            frame_index=packet.frame_index,
            timestamp=packet.timestamp
        )
        
        # 3. Object States
        object_states = self.object_state_machine.update(
            tracks=tracks,
            frame_index=packet.frame_index
        )
        
        # Update graph nodes
        self.graph.update_nodes(tracks, object_states)
        
        # 4. Geometry computation (only for active tracks)
        active_nodes = self.graph.get_active_nodes()
        active_tracks = [n.track for n in active_nodes]
        
        # We need observations matched to these tracks. 
        # CausalTracker could be modified to return the matched observation for each track.
        # Alternatively, we just use the latest bounding box from the track itself!
        # But wait, we need masks for geometry. 
        # If ObservationSource produces masks, they are attached to observations.
        # We need to map tracks back to their latest observations to get masks.
        
        # To simplify, we will just pass the depth image and rely on the track's centroid
        # If geometry requires masks, we need `track.recent_observations[-1]`.
        
        # But wait, observation_geometry is now computed inside ObservationSource (YOLOEDetector)
        # and attached to Observation. We don't need to recompute it here!
        # `obs.centroid_world`, `obs.bbox_min_world`, etc. are already available.
        from scene_graph.relations.context import ObservationGeometry
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
                    points_world=None,
                    points_camera=None,
                    mask=None,
                    valid_point_count=obs.valid_point_count
                )
                observation_geometry[track.object_id] = geo
                    
        # 5. Build FrameContext for relations
        # If we don't have a valid camera pose, we cannot compute relations
        if packet.world_T_camera is None:
            # We can still track, but we abort relations and geometry requiring world coordinates
            # Wait, tracking already requires world coordinates in our causal_tracker.
            # So if pose is None, we actually should have skipped or just maintained state.
            pass
            
        if packet.world_T_camera is not None:
            try:
                reference_frame = RelationReferenceFrame.from_camera_pose(packet.world_T_camera)
                
                # Assuming intrinsics can be constructed from the camera_model dict
                # For this demo, let's mock it if it's a dict
                from scene_graph.geometry.camera import CameraIntrinsics
                if packet.camera_model and "fx" in packet.camera_model:
                    intrinsics = CameraIntrinsics(**packet.camera_model)
                else:
                    intrinsics = CameraIntrinsics(525.0, 525.0, 319.5, 239.5, 640, 480)
                    
                context = FrameContext(
                    frame_index=packet.frame_index,
                    timestamp=packet.timestamp,
                    intrinsics=intrinsics,
                    world_T_camera=packet.world_T_camera,
                    reference_frame=reference_frame,
                    depth_image=packet.depth,
                    observation_geometry=observation_geometry
                )
                
                # 6. Compute Evidences
                raw_evidences = self.registry.compute_all(active_tracks, context)
                
                # 7. Apply Inverse Algebra
                all_evidences = []
                for ev in raw_evidences:
                    all_evidences.append(ev)
                    all_evidences.append(derive_inverse_evidence(ev))
                    
                # 8. Relation States
                relation_states = self.relation_state_machine.update(
                    evidences=all_evidences,
                    frame_index=packet.frame_index
                )
                
                # Update graph edges
                self.graph.update_edges(all_evidences, relation_states)
            except ValueError as e:
                # Invalid transform
                pass
        
        # Update graph metadata
        self.graph.current_frame_index = packet.frame_index
        self.graph.current_timestamp = packet.timestamp
        
        end_time = time.time()
        # Optionally log timing
        
        return self.graph
