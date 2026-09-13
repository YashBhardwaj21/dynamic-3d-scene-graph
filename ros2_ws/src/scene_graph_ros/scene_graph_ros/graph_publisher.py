"""Scene Graph ROS Publisher: JSON state and RViz MarkerArray."""

import json
from typing import List, Dict, Optional, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.temporal.relation_state import RelationState
from scene_graph.graph.participation_state import GraphParticipationState
from scene_graph.data.frame_packet import FramePacket


class GraphPublisher:
    """Publishes TemporalSceneGraph state as structured JSON and RViz MarkerArray."""

    def __init__(
        self,
        node: Node,
        state_topic: str = "/scene_graph/state",
        markers_topic: str = "/scene_graph/markers",
        world_frame: str = "world",
    ):
        self.node = node
        self.world_frame = world_frame

        self.state_pub = node.create_publisher(String, state_topic, 10)
        self.markers_pub = node.create_publisher(MarkerArray, markers_topic, 10)

        self.trajectory_points: List[Point] = []
        self.max_trajectory_len = 1000

        # Colors for RViz
        self.color_palette = [
            (0.2, 0.7, 0.3),  # Green
            (0.2, 0.5, 0.9),  # Blue
            (0.9, 0.6, 0.1),  # Orange
            (0.8, 0.2, 0.8),  # Magenta
            (0.1, 0.8, 0.8),  # Cyan
            (0.9, 0.2, 0.2),  # Red
            (0.6, 0.4, 0.2),  # Brown
        ]

    def _get_class_color(self, class_name: str) -> Tuple[float, float, float]:
        idx = hash(class_name) % len(self.color_palette)
        return self.color_palette[idx]

    def publish(self, graph: TemporalSceneGraph, packet: FramePacket):
        """Publish both JSON state and RViz markers."""
        self.publish_json_state(graph, packet)
        self.publish_rviz_markers(graph, packet)

    def publish_json_state(self, graph: TemporalSceneGraph, packet: FramePacket):
        """Publish human-readable and machine-parseable JSON of the active scene graph."""
        active_nodes = graph.get_active_nodes()
        all_edges = list(graph.edges.values())
        active_edges = [e for e in all_edges if e.is_active]
        uncertain_edges = [
            e for e in all_edges
            if e.participation == GraphParticipationState.ACTIVE and e.state != RelationState.SUPPORTED
        ]

        objects_data = []
        node_map = {}
        for node in active_nodes:
            track = node.track
            node_map[track.object_id] = track.class_name

            centroid = None
            if track.smoothed_position is not None:
                centroid = [float(x) for x in track.smoothed_position]
            elif track.recent_observations and track.recent_observations[-1].object_geometry:
                geom = track.recent_observations[-1].object_geometry
                if geom.centroid_world is not None:
                    centroid = [float(x) for x in geom.centroid_world]

            objects_data.append({
                "id": str(track.object_id),
                "class": str(track.class_name),
                "state": str(node.state.value),
                "confidence": float(track.detection_confidence),
                "centroid": centroid,
            })

        relations_data = []
        for edge in active_edges:
            relations_data.append({
                "subject": str(edge.subject_id),
                "subject_class": node_map.get(edge.subject_id, "unknown"),
                "predicate": str(edge.predicate),
                "object": str(edge.object_id),
                "object_class": node_map.get(edge.object_id, "unknown"),
                "state": str(edge.state.value),
                "evidence": float(edge.evidence_weight) if hasattr(edge, "evidence_weight") else 1.0,
            })

        payload = {
            "frame": int(packet.frame_index),
            "timestamp": float(packet.timestamp),
            "summary": {
                "active_objects": len(active_nodes),
                "active_relations": len(active_edges),
                "uncertain_relations": len(uncertain_edges),
            },
            "objects": objects_data,
            "relations": relations_data,
        }

        msg = String()
        msg.data = json.dumps(payload, indent=2)
        self.state_pub.publish(msg)

    def publish_rviz_markers(self, graph: TemporalSceneGraph, packet: FramePacket):
        """Construct visualization_msgs/MarkerArray for objects, labels, relations, and trajectory."""
        marker_array = MarkerArray()
        stamp = self.node.get_clock().now().to_msg()

        # 1. Clear previous markers safely (using DELETEALL if needed, or by unique IDs)
        active_nodes = graph.get_active_nodes()
        centroids: Dict[str, np.ndarray] = {}

        marker_id = 0

        # Objects and Text Labels
        for node in active_nodes:
            track = node.track
            centroid = None

            if track.smoothed_position is not None:
                centroid = track.smoothed_position
            elif track.recent_observations and track.recent_observations[-1].object_geometry:
                geom = track.recent_observations[-1].object_geometry
                if geom.centroid_world is not None:
                    centroid = geom.centroid_world

            if centroid is None:
                continue

            centroids[track.object_id] = centroid
            r, g, b = self._get_class_color(track.class_name)

            # Object 3D marker (Sphere/Box)
            obj_marker = Marker()
            obj_marker.header.frame_id = self.world_frame
            obj_marker.header.stamp = stamp
            obj_marker.ns = "scene_objects"
            obj_marker.id = marker_id
            marker_id += 1
            obj_marker.type = Marker.CUBE
            obj_marker.action = Marker.ADD

            obj_marker.pose.position.x = float(centroid[0])
            obj_marker.pose.position.y = float(centroid[1])
            obj_marker.pose.position.z = float(centroid[2])
            obj_marker.pose.orientation.w = 1.0

            # Default size or from OBB extents
            extents = [0.15, 0.15, 0.15]
            if track.recent_observations and track.recent_observations[-1].object_geometry:
                geom = track.recent_observations[-1].object_geometry
                if geom.obb_extents_world is not None:
                    extents = [max(0.05, float(x)) for x in geom.obb_extents_world]

            obj_marker.scale.x = extents[0]
            obj_marker.scale.y = extents[1]
            obj_marker.scale.z = extents[2]

            obj_marker.color.r = float(r)
            obj_marker.color.g = float(g)
            obj_marker.color.b = float(b)
            obj_marker.color.a = 0.75
            obj_marker.lifetime = Duration(sec=1, nanosec=0)

            marker_array.markers.append(obj_marker)

            # Text label hovering above object
            text_marker = Marker()
            text_marker.header.frame_id = self.world_frame
            text_marker.header.stamp = stamp
            text_marker.ns = "object_labels"
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(centroid[0])
            text_marker.pose.position.y = float(centroid[1])
            text_marker.pose.position.z = float(centroid[2]) + extents[2] / 2.0 + 0.08
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.08  # Text height in meters
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"{track.class_name} [{track.object_id[-4:]}]"
            text_marker.lifetime = Duration(sec=1, nanosec=0)

            marker_array.markers.append(text_marker)

        # 2. Relation Lines & Relation Badges
        active_edges = [e for e in graph.edges.values() if e.is_active]
        rel_line_marker = Marker()
        rel_line_marker.header.frame_id = self.world_frame
        rel_line_marker.header.stamp = stamp
        rel_line_marker.ns = "relation_edges"
        rel_line_marker.id = marker_id
        marker_id += 1
        rel_line_marker.type = Marker.LINE_LIST
        rel_line_marker.action = Marker.ADD
        rel_line_marker.scale.x = 0.02  # Line width
        rel_line_marker.color.r = 1.0
        rel_line_marker.color.g = 0.8
        rel_line_marker.color.b = 0.2
        rel_line_marker.color.a = 0.85
        rel_line_marker.lifetime = Duration(sec=1, nanosec=0)

        for edge in active_edges:
            sub_id = edge.subject_id
            obj_id = edge.object_id

            if sub_id in centroids and obj_id in centroids:
                p1 = centroids[sub_id]
                p2 = centroids[obj_id]

                pt1 = Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))
                pt2 = Point(x=float(p2[0]), y=float(p2[1]), z=float(p2[2]))
                rel_line_marker.points.append(pt1)
                rel_line_marker.points.append(pt2)

                # Predicate text at midpoint
                mid = (p1 + p2) * 0.5
                edge_text_marker = Marker()
                edge_text_marker.header.frame_id = self.world_frame
                edge_text_marker.header.stamp = stamp
                edge_text_marker.ns = "relation_labels"
                edge_text_marker.id = marker_id
                marker_id += 1
                edge_text_marker.type = Marker.TEXT_VIEW_FACING
                edge_text_marker.action = Marker.ADD

                edge_text_marker.pose.position.x = float(mid[0])
                edge_text_marker.pose.position.y = float(mid[1])
                edge_text_marker.pose.position.z = float(mid[2]) + 0.05
                edge_text_marker.pose.orientation.w = 1.0

                edge_text_marker.scale.z = 0.06
                edge_text_marker.color.r = 1.0
                edge_text_marker.color.g = 0.9
                edge_text_marker.color.b = 0.4
                edge_text_marker.color.a = 0.95
                edge_text_marker.text = f"-- {edge.predicate} -->"
                edge_text_marker.lifetime = Duration(sec=1, nanosec=0)

                marker_array.markers.append(edge_text_marker)

        if rel_line_marker.points:
            marker_array.markers.append(rel_line_marker)

        # 3. Camera Pose & Trajectory
        if packet.world_T_camera is not None:
            cam_pos = packet.world_T_camera[:3, 3]
            pt = Point(x=float(cam_pos[0]), y=float(cam_pos[1]), z=float(cam_pos[2]))
            self.trajectory_points.append(pt)
            if len(self.trajectory_points) > self.max_trajectory_len:
                self.trajectory_points.pop(0)

            # Camera frustum / sphere marker
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
            cam_marker.scale.x = 0.10
            cam_marker.scale.y = 0.10
            cam_marker.scale.z = 0.10
            cam_marker.color.r = 0.9
            cam_marker.color.g = 0.1
            cam_marker.color.b = 0.1
            cam_marker.color.a = 1.0
            cam_marker.lifetime = Duration(sec=1, nanosec=0)
            marker_array.markers.append(cam_marker)

            # Trajectory line
            traj_marker = Marker()
            traj_marker.header.frame_id = self.world_frame
            traj_marker.header.stamp = stamp
            traj_marker.ns = "camera_trajectory"
            traj_marker.id = marker_id
            marker_id += 1
            traj_marker.type = Marker.LINE_STRIP
            traj_marker.action = Marker.ADD
            traj_marker.scale.x = 0.02
            traj_marker.color.r = 0.8
            traj_marker.color.g = 0.2
            traj_marker.color.b = 0.2
            traj_marker.color.a = 0.8
            traj_marker.points = list(self.trajectory_points)
            traj_marker.lifetime = Duration(sec=1, nanosec=0)
            marker_array.markers.append(traj_marker)

        self.markers_pub.publish(marker_array)
