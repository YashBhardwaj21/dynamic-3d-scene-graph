"""Immutable scene graph snapshot for visualization, logging, and transport.

SceneGraphSnapshot is the inspection boundary between core pipeline internals
and downstream consumers (GraphPublisher, live_2d_viewer, telemetry loggers).

Design rules:
  - Frozen dataclasses only (immutable after construction).
  - Fields are plain Python types (str, float, int, tuple) — no numpy arrays,
    no Track/SceneNode/SceneEdge references, no deques.
  - No point clouds — those are a separate visualization payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.point_cloud import GeometryStatus


# ---------------------------------------------------------------------------
# Snapshot data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectSnapshot:
    """Flat representation of one tracked object at a single frame."""
    track_id: str
    class_name: str
    state: str                                                  # TrackState.value
    object_state: str                                           # ObjectState.value
    confidence: float
    observation_count: int
    missing_count: int
    centroid_world: Optional[tuple[float, float, float]]        # Kalman-filtered position
    velocity_world: Optional[tuple[float, float, float]]
    obb_center_world: Optional[tuple[float, float, float]]
    obb_axes_world: Optional[tuple[tuple[float, ...], ...]]     # 3×3 rotation columns
    obb_extents_world: Optional[tuple[float, float, float]]
    bbox_min_world: Optional[tuple[float, float, float]]
    bbox_max_world: Optional[tuple[float, float, float]]


@dataclass(frozen=True)
class RelationSnapshot:
    """Flat representation of one active relation edge."""
    subject_id: str
    subject_class: str
    predicate: str
    object_id: str
    object_class: str
    state: str                                                  # RelationState.value
    confidence: float


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Per-frame timing and throughput metrics."""
    frame_index: int
    timestamp: float
    input_fps: float
    processing_fps: float
    total_latency_ms: float
    queue_size: int
    dropped_frames: int
    active_objects: int
    active_relations: int
    # Per-stage timings (populated when instrumentation is available)
    tf_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    geometry_latency_ms: float = 0.0
    tracking_latency_ms: float = 0.0
    relation_latency_ms: float = 0.0


@dataclass(frozen=True)
class SceneGraphSnapshot:
    """Complete immutable snapshot of one frame's scene graph state.

    This is the single interface that visualization, logging, and transport
    layers consume. It never holds references to mutable pipeline internals.
    """
    frame_index: int
    timestamp: float
    world_T_camera: Optional[tuple[tuple[float, ...], ...]]     # 4×4 as nested tuples
    objects: tuple[ObjectSnapshot, ...]
    relations: tuple[RelationSnapshot, ...]
    telemetry: TelemetrySnapshot


# ---------------------------------------------------------------------------
# Factory: extract snapshot from live pipeline state
# ---------------------------------------------------------------------------

def _ndarray_to_tuple3(arr) -> Optional[tuple[float, float, float]]:
    """Convert a 3-element array-like to a (float, float, float) tuple."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.shape[0] != 3 or not np.isfinite(a).all():
        return None
    return (float(a[0]), float(a[1]), float(a[2]))


def _ndarray_to_matrix(arr) -> Optional[tuple[tuple[float, ...], ...]]:
    """Convert a 2D array to nested tuples."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        return None
    return tuple(tuple(float(x) for x in row) for row in a)


def create_snapshot(
    graph: TemporalSceneGraph,
    packet: FramePacket,
    *,
    input_fps: float = 0.0,
    processing_fps: float = 0.0,
    total_latency_ms: float = 0.0,
    queue_size: int = 0,
    dropped_frames: int = 0,
    tf_latency_ms: float = 0.0,
    inference_latency_ms: float = 0.0,
    geometry_latency_ms: float = 0.0,
    tracking_latency_ms: float = 0.0,
    relation_latency_ms: float = 0.0,
) -> SceneGraphSnapshot:
    """Extract an immutable SceneGraphSnapshot from the live graph and packet.

    This is the only function that touches pipeline internals. Everything
    downstream works with the returned snapshot only.
    """
    active_nodes = graph.get_active_nodes()
    active_edges = graph.get_active_edges()

    # Build class name lookup for relation snapshots
    class_map: dict[str, str] = {}

    # --- Object snapshots ---
    objects: list[ObjectSnapshot] = []
    for node in active_nodes:
        track = node.track
        class_map[track.object_id] = track.class_name

        # Extract OBB from latest observation geometry
        obb_center = None
        obb_axes = None
        obb_extents = None
        bbox_min = None
        bbox_max = None

        if track.recent_observations:
            latest_obs = track.recent_observations[-1]
            geom = getattr(latest_obs, "object_geometry", None)
            if geom is not None and geom.status == GeometryStatus.VALID:
                obb_center = _ndarray_to_tuple3(geom.obb_center_world)
                obb_axes = _ndarray_to_matrix(geom.obb_axes_world)
                obb_extents = _ndarray_to_tuple3(geom.obb_extents_world)
                bbox_min = _ndarray_to_tuple3(geom.bbox_min_world)
                bbox_max = _ndarray_to_tuple3(geom.bbox_max_world)

        velocity = None
        if track.velocity_world is not None:
            velocity = _ndarray_to_tuple3(track.velocity_world)

        objects.append(ObjectSnapshot(
            track_id=track.object_id,
            class_name=track.class_name,
            state=track.state.value,
            object_state=node.state.value,
            confidence=float(track.detection_confidence),
            observation_count=track.observation_count,
            missing_count=track.missing_count,
            centroid_world=_ndarray_to_tuple3(track.centroid_world),
            velocity_world=velocity,
            obb_center_world=obb_center,
            obb_axes_world=obb_axes,
            obb_extents_world=obb_extents,
            bbox_min_world=bbox_min,
            bbox_max_world=bbox_max,
        ))

    # --- Relation snapshots ---
    relations: list[RelationSnapshot] = []
    for edge in active_edges:
        evidence_weight = 1.0
        if hasattr(edge, "latest_evidence") and edge.latest_evidence is not None:
            evidence_weight = float(getattr(edge.latest_evidence, "confidence", 1.0))

        relations.append(RelationSnapshot(
            subject_id=edge.subject_id,
            subject_class=class_map.get(edge.subject_id, "unknown"),
            predicate=edge.predicate,
            object_id=edge.object_id,
            object_class=class_map.get(edge.object_id, "unknown"),
            state=edge.state.value,
            confidence=evidence_weight,
        ))

    # --- Telemetry ---
    telemetry = TelemetrySnapshot(
        frame_index=packet.frame_index,
        timestamp=packet.timestamp,
        input_fps=input_fps,
        processing_fps=processing_fps,
        total_latency_ms=total_latency_ms,
        queue_size=queue_size,
        dropped_frames=dropped_frames,
        active_objects=len(objects),
        active_relations=len(relations),
        tf_latency_ms=tf_latency_ms,
        inference_latency_ms=inference_latency_ms,
        geometry_latency_ms=geometry_latency_ms,
        tracking_latency_ms=tracking_latency_ms,
        relation_latency_ms=relation_latency_ms,
    )

    # --- Camera pose ---
    world_T_camera = _ndarray_to_matrix(packet.world_T_camera)

    return SceneGraphSnapshot(
        frame_index=packet.frame_index,
        timestamp=packet.timestamp,
        world_T_camera=world_T_camera,
        objects=tuple(objects),
        relations=tuple(relations),
        telemetry=telemetry,
    )
