import time
from typing import List, Dict, Optional

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.perception.yoloe_detector import YOLOEDetector
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.geometry import compute_object_geometry
from scene_graph.relations.context import FrameContext
from scene_graph.relations.registry import RelationRegistry
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
        # Resolve path
        model_path = config.get("perception.model_path", "models/yolov8n-seg.pt")
        self.observer = YOLOEDetector(
            model_path=model_path,
            confidence_threshold=config.get("perception.confidence_threshold", 0.3),
            allowed_classes=set(config.get("perception.classes", []))
        )
        
        # 2. Tracking
        self.tracker = CausalTracker(config)
        
        # 3. Geometry (stateless helper `compute_object_geometry` used inside update)
        
        # 4. Relations
        self.registry = RelationRegistry(config)
        self._register_modules()
        
        # 5. Temporal States
        self.object_state_machine = ObjectStateMachine(
            hysteresis_frames=config.get("temporal.object.hysteresis_frames", 5)
        )
        self.relation_state_machine = RelationStateMachine(
            confirmation_frames=config.get("temporal.relation.confirmation_frames", 3),
            missing_frames=config.get("temporal.relation.missing_frames", 2)
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
        observations = self.observer.detect(packet)
        
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
        
        observation_geometry = {}
        for track in active_tracks:
            if not track.recent_observations:
                continue
                
            obs = track.recent_observations[-1]
            # Verify this observation is from current frame
            if obs.frame_index == packet.frame_index:
                geo = compute_object_geometry(
                    observation=obs,
                    depth_image=packet.depth,
                    intrinsics=packet.intrinsics,
                    world_T_camera=packet.world_T_camera,
                    reference_frame="world"
                )
                if geo:
                    observation_geometry[track.object_id] = geo
                    
        # 5. Relation Context
        context = FrameContext(
            frame_index=packet.frame_index,
            timestamp=packet.timestamp,
            intrinsics=packet.intrinsics,
            world_T_camera=packet.world_T_camera,
            reference_frame="world",
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
        
        # Update graph metadata
        self.graph.current_frame_index = packet.frame_index
        self.graph.current_timestamp = packet.timestamp
        
        end_time = time.time()
        # Optionally log timing
        
        return self.graph
