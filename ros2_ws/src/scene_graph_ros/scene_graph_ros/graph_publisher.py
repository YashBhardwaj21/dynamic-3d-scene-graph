"""Scene Graph ROS Publisher: JSON state, RViz MarkerArray, PointCloud2, and 2D Overlays."""

from __future__ import annotations

import json
from typing import List, Dict, Optional, Tuple, Sequence, Any
import numpy as np

from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, PointCloud2
from builtin_interfaces.msg import Duration

from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.graph.snapshot import SceneGraphSnapshot, create_snapshot
from scene_graph.geometry.point_cloud import GeometryStatus
from scene_graph.data.frame_packet import FramePacket

from scene_graph_ros.ros_conversions import (
    rotation_matrix_to_quaternion,
    numpy_to_ros_image,
    numpy_to_point_cloud2,
)
from scene_graph_ros.overlay_renderer import (
    render_detections_overlay,
    render_tracks_overlay,
    track_color,
    relation_color,
)


try:
    from std_srvs.srv import Trigger
except ImportError:
    Trigger = None


class GraphPublisher:
    """Publishes SceneGraph state to ROS topics and provides query services."""

    def __init__(
        self,
        node: Node,
        state_topic: str = "/scene_graph/state",
        markers_topic: str = "/scene_graph/markers",
        object_cloud_topic: str = "/scene_graph/object_cloud",
        overlay_detections_topic: str = "/scene_graph/overlay_detections",
        overlay_tracks_topic: str = "/scene_graph/overlay_tracks",
        world_frame: str = "world",
    ):
        self.node = node
        self.world_frame = world_frame

        self.state_pub = node.create_publisher(String, state_topic, 10)
        self.markers_pub = node.create_publisher(MarkerArray, markers_topic, 10)
        self.cloud_pub = node.create_publisher(PointCloud2, object_cloud_topic, 10)
        self.overlay_det_pub = node.create_publisher(Image, overlay_detections_topic, 10)
        self.overlay_track_pub = node.create_publisher(Image, overlay_tracks_topic, 10)

        self.trajectory_points: List[Point] = []
        self.max_trajectory_len = 1000
        self.latest_snapshot: Optional[SceneGraphSnapshot] = None

        if Trigger is not None:
            self.snapshot_srv = node.create_service(Trigger, "/scene_graph/get_snapshot", self._handle_get_snapshot)
            self.query_objects_srv = node.create_service(Trigger, "/scene_graph/query_objects", self._handle_query_objects)
            self.query_relations_srv = node.create_service(Trigger, "/scene_graph/query_relations", self._handle_query_relations)
        else:
            self.snapshot_srv = None
            self.query_objects_srv = None
            self.query_relations_srv = None

    def query_objects(
        self,
        class_name: Optional[str] = None,
        state: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list:
        if self.latest_snapshot is None:
            return []
        return self.latest_snapshot.query_objects(class_name=class_name, state=state, min_confidence=min_confidence)

    def query_relations(
        self,
        subject_id: Optional[str] = None,
        object_id: Optional[str] = None,
        predicate: Optional[str] = None,
        state: Optional[str] = None,
    ) -> list:
        if self.latest_snapshot is None:
            return []
        return self.latest_snapshot.query_relations(
            subject_id=subject_id, object_id=object_id, predicate=predicate, state=state
        )

    def _handle_get_snapshot(self, request, response):
        if self.latest_snapshot is None:
            response.success = False
            response.message = json.dumps({"error": "No snapshot available yet"})
            return response
        response.success = True
        response.message = json.dumps(self.latest_snapshot.to_dict())
        return response

    def _handle_query_objects(self, request, response):
        if self.latest_snapshot is None:
            response.success = False
            response.message = json.dumps([])
            return response
        response.success = True
        response.message = json.dumps([obj.to_dict() for obj in self.latest_snapshot.objects])
        return response

    def _handle_query_relations(self, request, response):
        if self.latest_snapshot is None:
            response.success = False
            response.message = json.dumps([])
            return response
        response.success = True
        response.message = json.dumps([rel.to_dict() for rel in self.latest_snapshot.relations])
        return response

    def publish(
        self,
        graph: TemporalSceneGraph,
        packet: FramePacket,
        snapshot: Optional[SceneGraphSnapshot] = None,
        observations: Optional[Sequence[Any]] = None,
    ):
        """Publish all topics for one frame."""
        if snapshot is None:
            snapshot = create_snapshot(graph, packet)

        self.latest_snapshot = snapshot

        self.publish_json_state(snapshot)
        self.publish_rviz_markers(snapshot, packet)
        self.publish_object_cloud(graph, packet)
        self.publish_overlays(graph, packet, observations)

    def publish_json_state(self, snapshot: SceneGraphSnapshot):
        """Publish JSON representation of the active scene graph."""
        payload = {
            "frame": snapshot.frame_index,
            "timestamp": snapshot.timestamp,
            "telemetry": {
                "input_fps": round(snapshot.telemetry.input_fps, 2),
                "processing_fps": round(snapshot.telemetry.processing_fps, 2),
                "total_latency_ms": round(snapshot.telemetry.total_latency_ms, 2),
                "queue_size": snapshot.telemetry.queue_size,
                "dropped_frames": snapshot.telemetry.dropped_frames,
                "tf_latency_ms": round(snapshot.telemetry.tf_latency_ms, 2),
                "inference_latency_ms": round(snapshot.telemetry.inference_latency_ms, 2),
            },
            "summary": {
                "active_objects": snapshot.telemetry.active_objects,
                "active_relations": snapshot.telemetry.active_relations,
            },
            "objects": [
                {
                    "id": str(obj.track_id),
                    "class": str(obj.class_name),
                    "state": str(obj.state),
                    "object_state": str(obj.object_state),
                    "confidence": round(float(obj.confidence), 3),
                    "observations": int(obj.observation_count),
                    "centroid": [round(x, 4) for x in obj.centroid_world] if obj.centroid_world else None,
                    "velocity": [round(x, 4) for x in obj.velocity_world] if obj.velocity_world else None,
                    "obb_center": [round(x, 4) for x in obj.obb_center_world] if obj.obb_center_world else None,
                    "obb_extents": [round(x, 4) for x in obj.obb_extents_world] if obj.obb_extents_world else None,
                    "label_belief": {k: round(float(v), 3) for k, v in obj.label_belief.items()} if obj.label_belief else {},
                }
                for obj in snapshot.objects
            ],
            "relations": [
                {
                    "subject": str(rel.subject_id),
                    "subject_class": str(rel.subject_class),
                    "predicate": str(rel.predicate),
                    "object": str(rel.object_id),
                    "object_class": str(rel.object_class),
                    "state": str(rel.state),
                    "confidence": round(float(rel.confidence), 3),
                }
                for rel in snapshot.relations
            ],
        }

        msg = String()
        msg.data = json.dumps(payload, indent=2)
        self.state_pub.publish(msg)

    def publish_rviz_markers(self, snapshot: SceneGraphSnapshot, packet: FramePacket):
        """Construct visualization_msgs/MarkerArray for objects, labels, relations, and trajectory."""
        marker_array = MarkerArray()
        stamp = self.node.get_clock().now().to_msg()
        marker_id = 0

        obj_positions: Dict[str, Tuple[float, float, float]] = {}

        # 1. Spatial Context Regions (Translucent bounding boxes)
        for cid, ctx in getattr(snapshot, "spatial_contexts", {}).items():
            if cid == "world":
                continue
            b_min = ctx.get("bbox_min")
            b_max = ctx.get("bbox_max")
            if b_min and b_max:
                ctx_marker = Marker()
                ctx_marker.header.frame_id = self.world_frame
                ctx_marker.header.stamp = stamp
                ctx_marker.ns = "spatial_contexts"
                ctx_marker.id = marker_id
                marker_id += 1
                ctx_marker.type = Marker.CUBE
                ctx_marker.action = Marker.ADD

                cx = (b_min[0] + b_max[0]) * 0.5
                cy = (b_min[1] + b_max[1]) * 0.5
                cz = (b_min[2] + b_max[2]) * 0.5
                sx = max(0.2, b_max[0] - b_min[0])
                sy = max(0.2, b_max[1] - b_min[1])
                sz = max(0.1, b_max[2] - b_min[2])

                ctx_marker.pose.position.x = float(cx)
                ctx_marker.pose.position.y = float(cy)
                ctx_marker.pose.position.z = float(cz)
                ctx_marker.pose.orientation.w = 1.0

                ctx_marker.scale.x = float(sx)
                ctx_marker.scale.y = float(sy)
                ctx_marker.scale.z = float(sz)

                ctx_marker.color.r = 0.35
                ctx_marker.color.g = 0.55
                ctx_marker.color.b = 0.75
                ctx_marker.color.a = 0.15
                ctx_marker.lifetime = Duration(sec=1, nanosec=0)
                marker_array.markers.append(ctx_marker)

        # 2. Objects and Labels
        for obj in snapshot.objects:
            pos = obj.obb_center_world or obj.centroid_world
            if pos is None:
                continue

            obj_positions[obj.track_id] = pos
            r_int, g_int, b_int = track_color(obj.track_id)
            r, g, b = r_int / 255.0, g_int / 255.0, b_int / 255.0

            obj_marker = Marker()
            obj_marker.header.frame_id = self.world_frame
            obj_marker.header.stamp = stamp
            obj_marker.ns = "scene_objects"
            obj_marker.id = marker_id
            marker_id += 1
            obj_marker.type = Marker.CUBE
            obj_marker.action = Marker.ADD

            obj_marker.pose.position.x = float(pos[0])
            obj_marker.pose.position.y = float(pos[1])
            obj_marker.pose.position.z = float(pos[2])

            if obj.obb_axes_world is not None:
                try:
                    axes = np.asarray(obj.obb_axes_world, dtype=np.float64)
                    qx, qy, qz, qw = rotation_matrix_to_quaternion(axes)
                    obj_marker.pose.orientation.x = float(qx)
                    obj_marker.pose.orientation.y = float(qy)
                    obj_marker.pose.orientation.z = float(qz)
                    obj_marker.pose.orientation.w = float(qw)
                except Exception:
                    obj_marker.pose.orientation.w = 1.0
            else:
                obj_marker.pose.orientation.w = 1.0

            if obj.obb_extents_world is not None:
                extents = [max(0.04, float(x)) for x in obj.obb_extents_world]
            else:
                extents = [0.15, 0.15, 0.15]

            obj_marker.scale.x = extents[0]
            obj_marker.scale.y = extents[1]
            obj_marker.scale.z = extents[2]

            obj_marker.color.r = float(r)
            obj_marker.color.g = float(g)
            obj_marker.color.b = float(b)
            # Anchors have subtle translucent appearance; child objects are solid
            is_anchor = getattr(obj, "is_spatial_anchor", False)
            obj_marker.color.a = 0.25 if is_anchor else 0.80
            obj_marker.lifetime = Duration(sec=1, nanosec=0)
            marker_array.markers.append(obj_marker)

            text_marker = Marker()
            text_marker.header.frame_id = self.world_frame
            text_marker.header.stamp = stamp
            text_marker.ns = "object_labels"
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(pos[0])
            text_marker.pose.position.y = float(pos[1])
            text_marker.pose.position.z = float(pos[2]) + extents[2] / 2.0 + 0.07
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.07
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            short_id = obj.track_id[-4:] if len(obj.track_id) >= 4 else obj.track_id
            if is_anchor:
                text_marker.text = f"{obj.class_name.upper()} #{short_id} [ANCHOR]"
            elif getattr(obj, "spatial_context_id", "world") != "world":
                text_marker.text = f"{obj.class_name.upper()} #{short_id} [on DESK]"
            else:
                text_marker.text = f"{obj.class_name.upper()} #{short_id} [{obj.object_state}]"
            text_marker.lifetime = Duration(sec=1, nanosec=0)
            marker_array.markers.append(text_marker)

        # 3. Relations: Vertical Structural Connectors & Sibling Directional Arrows
        priority = {
            "ON": 10, "SUPPORTED_BY": 10, "INSIDE": 9,
            "OCCLUDING": 8, "TOUCHING": 7, "NEAR": 6,
            "LEFT_OF": 5, "RIGHT_OF": 5, "ABOVE": 4, "IN_FRONT_OF": 4,
        }

        anchor_ids = {
            obj.track_id for obj in snapshot.objects if getattr(obj, "is_spatial_anchor", False)
        }

        structural_rels = [
            r for r in snapshot.relations if r.predicate in ("ON", "SUPPORTED_BY", "INSIDE")
        ]
        sibling_rels = [
            r for r in snapshot.relations
            if r.predicate not in ("ON", "SUPPORTED_BY", "INSIDE")
            and r.subject_id not in anchor_ids
            and r.object_id not in anchor_ids
        ]

        # 3a. Thin vertical connector lines for structural relations
        for rel in structural_rels:
            sub_id = rel.subject_id
            obj_id = rel.object_id
            if sub_id in obj_positions and obj_id in obj_positions:
                p_sub = obj_positions[sub_id]
                p_obj = obj_positions[obj_id]

                line_marker = Marker()
                line_marker.header.frame_id = self.world_frame
                line_marker.header.stamp = stamp
                line_marker.ns = "structural_connectors"
                line_marker.id = marker_id
                marker_id += 1
                line_marker.type = Marker.LINE_LIST
                line_marker.action = Marker.ADD

                line_marker.scale.x = 0.015  # Thin vertical line
                line_marker.color.r = 0.2
                line_marker.color.g = 0.8
                line_marker.color.b = 0.6
                line_marker.color.a = 0.85
                line_marker.lifetime = Duration(sec=1, nanosec=0)

                # From subject base to object top
                pt_top = Point(x=float(p_sub[0]), y=float(p_sub[1]), z=float(p_sub[2]))
                pt_bot = Point(x=float(p_sub[0]), y=float(p_sub[1]), z=float(p_obj[2]))
                line_marker.points.append(pt_top)
                line_marker.points.append(pt_bot)
                marker_array.markers.append(line_marker)

        # 3b. Sibling directional & proximity relations (ranked and capped to top 8)
        ranked_sibling_rels = sorted(
            sibling_rels,
            key=lambda r: (priority.get(r.predicate, 0), r.confidence),
            reverse=True,
        )[:8]

        for rel in ranked_sibling_rels:
            sub_id = rel.subject_id
            obj_id = rel.object_id

            if sub_id in obj_positions and obj_id in obj_positions:
                p1 = obj_positions[sub_id]
                p2 = obj_positions[obj_id]

                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                if dist < 0.04:
                    continue

                pt1 = Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))
                pt2 = Point(x=float(p2[0]), y=float(p2[1]), z=float(p2[2]))

                rel_col = relation_color(rel.predicate)
                cr, cg, cb = rel_col[0] / 255.0, rel_col[1] / 255.0, rel_col[2] / 255.0

                arrow_marker = Marker()
                arrow_marker.header.frame_id = self.world_frame
                arrow_marker.header.stamp = stamp
                arrow_marker.ns = "relation_arrows"
                arrow_marker.id = marker_id
                marker_id += 1
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD

                arrow_marker.points.append(pt1)
                arrow_marker.points.append(pt2)

                arrow_marker.scale.x = 0.012
                arrow_marker.scale.y = 0.024
                arrow_marker.scale.z = 0.035

                arrow_marker.color.r = float(cr)
                arrow_marker.color.g = float(cg)
                arrow_marker.color.b = float(cb)
                arrow_marker.color.a = 0.80
                arrow_marker.lifetime = Duration(sec=1, nanosec=0)
                marker_array.markers.append(arrow_marker)

                mid_x = (p1[0] + p2[0]) * 0.5
                mid_y = (p1[1] + p2[1]) * 0.5
                mid_z = (p1[2] + p2[2]) * 0.5

                edge_text_marker = Marker()
                edge_text_marker.header.frame_id = self.world_frame
                edge_text_marker.header.stamp = stamp
                edge_text_marker.ns = "relation_labels"
                edge_text_marker.id = marker_id
                marker_id += 1
                edge_text_marker.type = Marker.TEXT_VIEW_FACING
                edge_text_marker.action = Marker.ADD

                edge_text_marker.pose.position.x = float(mid_x)
                edge_text_marker.pose.position.y = float(mid_y)
                edge_text_marker.pose.position.z = float(mid_z) + 0.05
                edge_text_marker.pose.orientation.w = 1.0

                edge_text_marker.scale.z = 0.040
                edge_text_marker.color.r = float(cr)
                edge_text_marker.color.g = float(cg)
                edge_text_marker.color.b = float(cb)
                edge_text_marker.color.a = 0.95
                edge_text_marker.text = f"{rel.predicate}"
                edge_text_marker.lifetime = Duration(sec=1, nanosec=0)
                marker_array.markers.append(edge_text_marker)

        if packet.world_T_camera is not None:
            cam_pos = packet.world_T_camera[:3, 3]
            pt = Point(x=float(cam_pos[0]), y=float(cam_pos[1]), z=float(cam_pos[2]))
            self.trajectory_points.append(pt)
            if len(self.trajectory_points) > self.max_trajectory_len:
                self.trajectory_points.pop(0)

            cam_marker = Marker()
            cam_marker.header.frame_id = self.world_frame
            cam_marker.header.stamp = stamp
            cam_marker.ns = "camera_pose"
            cam_marker.id = marker_id
            marker_id += 1
            cam_marker.type = Marker.SPHERE
            cam_marker.action = Marker.ADD
            cam_marker.pose.position = pt
            cam_marker.pose.orientation.w = 1.0
            cam_marker.scale.x = 0.08
            cam_marker.scale.y = 0.08
            cam_marker.scale.z = 0.08
            cam_marker.color.r = 0.95
            cam_marker.color.g = 0.15
            cam_marker.color.b = 0.15
            cam_marker.color.a = 1.0
            cam_marker.lifetime = Duration(sec=1, nanosec=0)
            marker_array.markers.append(cam_marker)

            traj_marker = Marker()
            traj_marker.header.frame_id = self.world_frame
            traj_marker.header.stamp = stamp
            traj_marker.ns = "camera_trajectory"
            traj_marker.id = marker_id
            marker_id += 1
            traj_marker.type = Marker.LINE_STRIP
            traj_marker.action = Marker.ADD
            traj_marker.scale.x = 0.015
            traj_marker.color.r = 0.85
            traj_marker.color.g = 0.25
            traj_marker.color.b = 0.25
            traj_marker.color.a = 0.80
            traj_marker.points = list(self.trajectory_points)
            traj_marker.lifetime = Duration(sec=1, nanosec=0)
            marker_array.markers.append(traj_marker)

        self.markers_pub.publish(marker_array)

    def publish_object_cloud(self, graph: TemporalSceneGraph, packet: FramePacket):
        """Publish segmented 3D points per active tracked object on /scene_graph/object_cloud."""
        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []

        active_nodes = graph.get_active_nodes()
        for node in active_nodes:
            track = node.track
            if not track.recent_observations:
                continue

            latest_obs = track.recent_observations[-1]
            geom = getattr(latest_obs, "object_geometry", None)
            if geom is not None and geom.status == GeometryStatus.VALID and geom.points_world is not None:
                pts = geom.points_world
                n = pts.shape[0]
                if n > 0:
                    all_points.append(pts)
                    r, g, b = track_color(track.object_id)
                    col = np.tile(np.array([r, g, b], dtype=np.uint8), (n, 1))
                    all_colors.append(col)

        if all_points:
            pts_concat = np.vstack(all_points)
            cols_concat = np.vstack(all_colors)
        else:
            pts_concat = np.empty((0, 3), dtype=np.float32)
            cols_concat = np.empty((0, 3), dtype=np.uint8)

        cloud_msg = numpy_to_point_cloud2(
            points=pts_concat,
            frame_id=self.world_frame,
            timestamp=packet.timestamp,
            colors=cols_concat,
        )
        self.cloud_pub.publish(cloud_msg)

    def publish_overlays(
        self,
        graph: TemporalSceneGraph,
        packet: FramePacket,
        observations: Optional[Sequence[Any]] = None,
    ):
        """Publish 2D perception overlay images."""
        if packet.rgb is None or packet.rgb.size == 0:
            return

        frame_id = packet.metadata.get("frame_id", "camera_optical_frame")

        try:
            det_img = render_detections_overlay(packet.rgb, observations)
            det_msg = numpy_to_ros_image(
                det_img,
                encoding="rgb8",
                frame_id=frame_id,
                timestamp=packet.timestamp,
            )
            self.overlay_det_pub.publish(det_msg)
        except Exception as e:
            self.node.get_logger().warn(f"Failed to render detections overlay: {e}", throttle_duration_sec=2.0)

        try:
            track_img = render_tracks_overlay(packet.rgb, graph, packet.frame_index)
            track_msg = numpy_to_ros_image(
                track_img,
                encoding="rgb8",
                frame_id=frame_id,
                timestamp=packet.timestamp,
            )
            self.overlay_track_pub.publish(track_msg)
        except Exception as e:
            self.node.get_logger().warn(f"Failed to render tracks overlay: {e}", throttle_duration_sec=2.0)
