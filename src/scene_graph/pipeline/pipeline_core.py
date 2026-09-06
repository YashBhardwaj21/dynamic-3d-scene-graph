from typing import List

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.perception.observation import Observation
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.temporal.object_state import ObjectStateMachine
from scene_graph.temporal.relation_state import RelationStateMachine
from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.relations.registry import RelationRegistry
from scene_graph.relations.inverse_algebra import derive_inverse_evidence

from scene_graph.relations.distance import DistanceRelationModule
from scene_graph.relations.support import SupportRelationModule
from scene_graph.relations.directional import DirectionalRelationModule
from scene_graph.relations.containment import ContainmentRelationModule
from scene_graph.relations.depth_order import DepthOrderRelationModule
from scene_graph.relations.occlusion import OcclusionRelationModule

from scene_graph.geometry.point_cloud import (
    ObjectGeometry,
    GeometryStatus,
    compute_object_geometry,
)
from scene_graph.geometry.reference_frame import CameraFrame
from scene_graph.relations.context import FrameContext, ObservationGeometry


class SceneGraphPipeline:

    def __init__(self, config: SceneGraphConfig):
        self.config = config
        self.tracker = CausalTracker(config)

        self.registry = RelationRegistry(config)
        self._register_modules()

        self.object_state_machine = ObjectStateMachine()
        self.relation_state_machine = RelationStateMachine(
            config.temporal.relation
        )

        self.graph = TemporalSceneGraph()

    def _register_modules(self) -> None:
        self.registry.register(DistanceRelationModule(self.config))
        self.registry.register(SupportRelationModule(self.config))
        self.registry.register(DirectionalRelationModule(self.config))
        self.registry.register(ContainmentRelationModule(self.config))
        self.registry.register(DepthOrderRelationModule(self.config))
        self.registry.register(OcclusionRelationModule(self.config))

    def _compute_geometry(
        self,
        packet: FramePacket,
        observations: List[Observation],
        depth_m,
    ) -> None:
        if packet.depth is None:
            for obs in observations:
                obs.object_geometry = ObjectGeometry(
                    status=GeometryStatus.NO_DEPTH
                )
            return

        if packet.world_T_camera is None:
            for obs in observations:
                obs.object_geometry = ObjectGeometry(
                    status=GeometryStatus.NO_POSE
                )
            return

        geometry_config = self.config.geometry

        for obs in observations:
            mask = obs.get_mask()

            if mask is None:
                obs.object_geometry = ObjectGeometry(
                    status=GeometryStatus.INVALID_GEOMETRY
                )
                continue

            obs.object_geometry = compute_object_geometry(
                mask=mask,
                depth_m=depth_m,
                intrinsics=packet.camera_intrinsics,
                pose=packet.world_T_camera,
                min_valid_points=geometry_config.min_valid_points,
                mad_k=geometry_config.robust_depth.k,
                voxel_size_m=geometry_config.downsampling.voxel_size_m,
            )

    @staticmethod
    def _build_observation_geometry(
        tracks,
        frame_index: int,
    ) -> dict[str, ObservationGeometry]:
        geometry = {}

        for track in tracks:
            if not track.recent_observations:
                continue

            obs = track.recent_observations[-1]

            if obs.frame_index != frame_index:
                continue

            geo = obs.object_geometry

            if geo is None or geo.status != GeometryStatus.VALID:
                continue

            geometry[track.object_id] = ObservationGeometry(
                obs_id=obs.obs_id,
                track_id=track.object_id,
                centroid_world=geo.centroid_world,
                bbox_min_world=geo.bbox_min_world,
                bbox_max_world=geo.bbox_max_world,
                depth_stats=geo.depth_stats,
                points_world_sampled=geo.points_world_sampled,
                points_world=geo.points_world,
                points_camera=geo.points_camera,
                mask=obs.get_mask(),
                valid_point_count=geo.valid_point_count,
            )

        return geometry

    def update(
        self,
        packet: FramePacket,
        observations: List[Observation],
    ) -> TemporalSceneGraph:

        self.graph.current_frame_index = packet.frame_index
        self.graph.current_timestamp = packet.timestamp

        depth_m = None

        if packet.depth is not None:
            if packet.depth_model is None:
                raise ValueError(
                    f"Frame {packet.frame_index} has depth but no depth model."
                )

            depth_m = packet.depth_model.depth_to_meters(packet.depth)

        self._compute_geometry(
            packet=packet,
            observations=observations,
            depth_m=depth_m,
        )

        tracks = self.tracker.update(
            observations=observations,
            frame_index=packet.frame_index,
            timestamp=packet.timestamp,
        )

        object_states = self.object_state_machine.update(
            tracks=tracks,
            frame_index=packet.frame_index,
        )

        self.graph.update_nodes(tracks, object_states)

        active_tracks = [
            node.track
            for node in self.graph.get_active_nodes()
        ]

        if not active_tracks:
            return self.graph

        observation_geometry = self._build_observation_geometry(
            active_tracks,
            packet.frame_index,
        )

        if packet.world_T_camera is None:
            return self.graph

        if packet.relation_frame is None:
            raise ValueError(
                f"Frame {packet.frame_index} is missing relation reference frame."
            )

        context = FrameContext(
            frame_index=packet.frame_index,
            timestamp=packet.timestamp,
            intrinsics=packet.camera_intrinsics,
            world_T_camera=packet.world_T_camera,
            reference_frame=packet.relation_frame,
            camera_frame=CameraFrame.from_camera_pose(
                packet.world_T_camera
            ),
            depth_image=depth_m,
            observation_geometry=observation_geometry,
        )

        raw_evidences = self.registry.compute_all(
            active_tracks,
            context,
        )

        all_evidences = []

        for evidence in raw_evidences:
            all_evidences.append(evidence)
            all_evidences.append(
                derive_inverse_evidence(evidence)
            )

        active_object_ids = {
            track.object_id
            for track in active_tracks
        }

        relation_states = self.relation_state_machine.update(
            evidences=all_evidences,
            frame_index=packet.frame_index,
            timestamp=packet.timestamp,
            active_object_ids=active_object_ids,
        )

        self.graph.update_edges(
            evidences=all_evidences,
            relation_states=relation_states,
        )

        return self.graph