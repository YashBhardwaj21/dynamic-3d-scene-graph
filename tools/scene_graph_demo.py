"""Polished Streamlit demo for the Dynamic 3D Scene Graph."""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scene_graph.config import load_config
from scene_graph.data.tum_source import TUMReplaySource
from scene_graph.graph.query import QueryEngine
from scene_graph.pipeline.online_pipeline import OnlinePipeline
from scene_graph.relations.evidence import EvidenceResult

RELATION_FAMILIES = {
    "LEFT_OF": "horizontal", "RIGHT_OF": "horizontal",
    "ABOVE": "vertical", "BELOW": "vertical",
    "IN_FRONT_OF": "depth", "BEHIND": "depth",
    "ON": "support", "UNDER": "support",
    "INSIDE": "containment", "CONTAINING": "containment",
    "OCCLUDING": "occlusion", "OCCLUDED_BY": "occlusion",
    "NEAR": "near", "FAR": "far",
}

RELATION_COLORS = {
    "ON": (54, 207, 169), "UNDER": (54, 207, 169),
    "INSIDE": (109, 164, 240), "CONTAINING": (109, 164, 240),
    "NEAR": (188, 143, 246), "FAR": (150, 160, 175),
    "LEFT_OF": (248, 188, 90), "RIGHT_OF": (248, 188, 90),
    "ABOVE": (248, 188, 90), "BELOW": (248, 188, 90),
    "IN_FRONT_OF": (94, 211, 241), "BEHIND": (94, 211, 241),
    "OCCLUDING": (246, 105, 105), "OCCLUDED_BY": (246, 105, 105),
}

PRESENTATION_PREDICATES = {
    "horizontal": {"LEFT_OF", "RIGHT_OF"},
    "vertical": {"ABOVE", "BELOW"},
    "depth": {"IN_FRONT_OF", "BEHIND"},
    "support": {"ON", "UNDER"},
    "containment": {"INSIDE", "CONTAINING"},
    "occlusion": {"OCCLUDING", "OCCLUDED_BY"},
    "near": {"NEAR"},
    "far": {"FAR"},
}

KEY_PREDICATES = {
    "ON", "UNDER", "INSIDE", "CONTAINING", "LEFT_OF", "RIGHT_OF",
    "ABOVE", "BELOW", "IN_FRONT_OF", "BEHIND", "OCCLUDING", "OCCLUDED_BY",
    "NEAR",
}

ARROW_RELATIONS = {
    "LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "IN_FRONT_OF", "BEHIND",
    "OCCLUDING", "OCCLUDED_BY",
}

OBJECT_PALETTE = [
    (82, 177, 238), (89, 208, 164), (171, 133, 235), (239, 177, 87),
    (82, 193, 207), (229, 116, 150), (164, 181, 112), (133, 151, 226),
]

DISPLAY_FILTERS = ["KEY", "ALL"] + sorted(RELATION_COLORS)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp{background:#090d12;color:#e9eef3}
        .block-container{max-width:1480px;padding:1.0rem 1.8rem 1.5rem}
        .hero{display:flex;justify-content:space-between;align-items:end;gap:1rem;margin-bottom:1rem}
        .hero-title{font-size:2rem;font-weight:750;letter-spacing:-.04em;line-height:1.05}
        .hero-subtitle{color:#8996a4;font-size:.88rem;margin-top:.22rem}
        .hero-badge{color:#7dddb1;border:1px solid #254f40;background:#10241d;border-radius:999px;padding:.28rem .58rem;font-size:.68rem;font-weight:650;letter-spacing:.06em;white-space:nowrap}
        .metric-card{background:#11171f;border:1px solid #202a34;border-radius:12px;padding:.68rem .8rem;min-height:70px}
        .metric-label{color:#758290;font-size:.64rem;text-transform:uppercase;letter-spacing:.09em}
        .metric-value{color:#f4f7fa;font-size:1.32rem;font-weight:700;margin-top:.12rem}
        .panel-title{color:#8793a0;font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;margin:0 0 .55rem}
        .graph-object{background:#10161e;border:1px solid #232e39;border-radius:10px;padding:.62rem .7rem;margin-bottom:.48rem}
        .graph-object-title{font-size:.88rem;font-weight:700;color:#edf2f6}
        .graph-object-meta{font-size:.7rem;color:#74818f;margin-top:.1rem}
        .graph-relation{font-size:.75rem;padding:.32rem .42rem;margin-top:.32rem;background:#0b1117;border-radius:7px;color:#cbd3da}
        .relation-badge{font-size:.64rem;font-weight:750;letter-spacing:.04em;border-radius:5px;padding:.16rem .34rem;margin-right:.3rem}
        .empty-state{color:#778493;text-align:center;padding:1.5rem .6rem;border:1px dashed #27323e;border-radius:10px}
        .demo-note{color:#778492;font-size:.72rem;line-height:1.45}
        div[data-testid="stDataFrame"]{border:1px solid #202a34;border-radius:9px;overflow:hidden}
        button{border-radius:9px!important}
        </style>
        """,
        unsafe_allow_html=True,
    )


def track_color(track_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    return OBJECT_PALETTE[digest[0] % len(OBJECT_PALETTE)]


def relation_color(predicate: str) -> tuple[int, int, int]:
    return RELATION_COLORS.get(predicate, (215, 220, 225))


def graph_frame_index(graph: Any) -> int:
    return int(getattr(graph, "current_frame_index", -1))


def latest_observation(node: Any, frame_index: Optional[int] = None) -> Any | None:
    observations = getattr(node.track, "recent_observations", None)
    if not observations:
        return None
    observation = observations[-1]
    if frame_index is None:
        return observation
    return observation if observation.frame_index == frame_index else None


def current_frame_nodes(graph: Any, frame_index: int) -> list[Any]:
    return [node for node in graph.get_active_nodes() if latest_observation(node, frame_index) is not None]


def current_frame_edges(graph: Any, frame_index: int) -> list[Any]:
    return [
        edge for edge in graph.get_active_edges()
        if edge.latest_evidence is not None
        and edge.latest_evidence.frame_index == frame_index
        and edge.latest_evidence.result == EvidenceResult.SUPPORTED
    ]


def _edge_orientation(edge: Any, graph: Any, frame_index: int) -> Optional[str]:
    family = RELATION_FAMILIES.get(edge.predicate)
    if family not in {"horizontal", "vertical"}:
        return None

    subject = graph.nodes.get(edge.subject_id)
    object_ = graph.nodes.get(edge.object_id)
    if subject is None or object_ is None:
        return None

    subject_obs = latest_observation(subject, frame_index)
    object_obs = latest_observation(object_, frame_index)
    if subject_obs is None or object_obs is None:
        return None

    sx1, sy1, sx2, sy2 = np.asarray(subject_obs.bbox_xyxy, dtype=float)
    ox1, oy1, ox2, oy2 = np.asarray(object_obs.bbox_xyxy, dtype=float)
    sx, sy = (sx1 + sx2) * 0.5, (sy1 + sy2) * 0.5
    ox, oy = (ox1 + ox2) * 0.5, (oy1 + oy2) * 0.5

    if family == "horizontal":
        return "RIGHT_OF" if sx > ox else "LEFT_OF"
    return "ABOVE" if sy < oy else "BELOW"


def display_edges(graph: Any, frame_index: int) -> list[Any]:
    grouped: dict[tuple[frozenset[str], str], list[Any]] = {}

    for edge in current_frame_edges(graph, frame_index):
        family = RELATION_FAMILIES.get(edge.predicate, edge.predicate)
        key = (frozenset((edge.subject_id, edge.object_id)), family)
        grouped.setdefault(key, []).append(edge)

    selected = []
    for (_, family), edges in grouped.items():
        if family in {"horizontal", "vertical"}:
            desired = _edge_orientation(edges[0], graph, frame_index)
            match = next((edge for edge in edges if edge.predicate == desired), None)
            selected.append(match or edges[0])
        else:
            selected.append(edges[0])

    return sorted(selected, key=lambda edge: (edge.subject_id, edge.predicate, edge.object_id))


def node_rows(graph: Any, frame_index: Optional[int] = None) -> list[dict[str, Any]]:
    if frame_index is None:
        frame_index = graph_frame_index(graph)

    rows = []
    for node in current_frame_nodes(graph, frame_index):
        track = node.track
        rows.append({
            "track_id": node.object_id,
            "class": node.class_name,
            "state": node.state.value,
            "confidence": round(float(track.detection_confidence), 3),
            "observations": int(track.observation_count),
        })
    return sorted(rows, key=lambda row: row["track_id"])


def edge_rows(graph: Any, frame_index: Optional[int] = None) -> list[dict[str, Any]]:
    if frame_index is None:
        frame_index = graph_frame_index(graph)

    rows = []
    for edge in display_edges(graph, frame_index):
        subject = graph.nodes.get(edge.subject_id)
        object_ = graph.nodes.get(edge.object_id)
        if subject is None or object_ is None:
            continue
        evidence = edge.latest_evidence
        rows.append({
            "subject": f"{subject.class_name} · {subject.object_id}",
            "predicate": edge.predicate,
            "object": f"{object_.class_name} · {object_.object_id}",
            "confidence": round(float(evidence.confidence), 3),
            "frame": evidence.frame_index,
            "evidence": evidence.evidence_type,
        })
    return rows


def history_rows(graph: Any) -> list[dict[str, Any]]:
    return [{
        "frame": event.frame_index,
        "event": event.event_type.value.upper(),
        "subject": event.subject_id,
        "predicate": event.predicate or "",
        "object": event.object_id or "",
    } for event in graph.history.events]


def as_dataframe(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def clamp_bbox(bbox: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=float)
    return (
        int(np.clip(x1, 0, width - 1)), int(np.clip(y1, 0, height - 1)),
        int(np.clip(x2, 0, width - 1)), int(np.clip(y2, 0, height - 1)),
    )


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def boundary_point(bbox: tuple[int, int, int, int], target: tuple[float, float]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    cx, cy = bbox_center(bbox)
    dx, dy = target[0] - cx, target[1] - cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return int(cx), int(cy)

    candidates = []
    if dx > 0:
        candidates.append(((x2 - cx) / dx, x2, cy + (x2 - cx) * dy / dx))
    elif dx < 0:
        candidates.append(((x1 - cx) / dx, x1, cy + (x1 - cx) * dy / dx))
    if dy > 0:
        candidates.append(((y2 - cy) / dy, cx + (y2 - cy) * dx / dy, y2))
    elif dy < 0:
        candidates.append(((y1 - cy) / dy, cx + (y1 - cy) * dx / dy, y1))

    candidates = [item for item in candidates if item[0] > 0]
    if not candidates:
        return int(cx), int(cy)

    _, x, y = min(candidates, key=lambda item: item[0])
    return int(x), int(y)


def draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.10) -> np.ndarray:
    if mask.shape[:2] != image.shape[:2]:
        return image
    overlay = image.copy()
    overlay[mask.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], background: tuple[int, int, int], scale: float = 0.40) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x = max(3, min(origin[0], image.shape[1] - tw - 8))
    y = max(th + 8, min(origin[1], image.shape[0] - 3))
    cv2.rectangle(image, (x, y - th - 7), (x + tw + 7, y + 3), background, -1, cv2.LINE_AA)
    cv2.putText(image, text, (x + 3, y - 3), font, scale, (12, 16, 21), 1, cv2.LINE_AA)


def render_rgb_scene(rgb: np.ndarray, graph: Any, frame_index: Optional[int] = None, predicate_filter: str = "KEY") -> np.ndarray:
    if frame_index is None:
        frame_index = graph_frame_index(graph)

    canvas = rgb.copy()
    height, width = canvas.shape[:2]
    boxes: dict[str, tuple[int, int, int, int]] = {}
    nodes: dict[str, Any] = {}

    for node in current_frame_nodes(graph, frame_index):
        observation = latest_observation(node, frame_index)
        if observation is None:
            continue
        node_id = node.object_id
        nodes[node_id] = node
        boxes[node_id] = clamp_bbox(observation.bbox_xyxy, width, height)
        try:
            mask = observation.get_mask()
        except Exception:
            mask = None
        if mask is not None:
            canvas = draw_mask(canvas, mask, track_color(node_id))

    edges = display_edges(graph, frame_index)
    if predicate_filter == "KEY":
        edges = [edge for edge in edges if edge.predicate in KEY_PREDICATES]
    elif predicate_filter != "ALL":
        edges = [edge for edge in edges if edge.predicate == predicate_filter]

    for edge in edges:
        if edge.subject_id not in boxes or edge.object_id not in boxes:
            continue
        start = boundary_point(boxes[edge.subject_id], bbox_center(boxes[edge.object_id]))
        end = boundary_point(boxes[edge.object_id], bbox_center(boxes[edge.subject_id]))
        color = relation_color(edge.predicate)
        if edge.predicate in ARROW_RELATIONS:
            cv2.arrowedLine(canvas, start, end, color, 2, cv2.LINE_AA, tipLength=0.08)
        else:
            cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)
        mx, my = int((start[0] + end[0]) * 0.5), int((start[1] + end[1]) * 0.5 - 5)
        draw_label(canvas, edge.predicate, (mx, my), color, scale=0.37)

    for node_id, node in sorted(nodes.items()):
        x1, y1, x2, y2 = boxes[node_id]
        color = track_color(node_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        draw_label(canvas, f"{node.class_name.upper()} · #{node_id.split('_')[-1]}", (x1, max(20, y1 - 5)), color)

    return canvas


def render_current_graph(graph: Any, frame_index: int) -> None:
    nodes = {node.object_id: node for node in current_frame_nodes(graph, frame_index)}
    edges = display_edges(graph, frame_index)

    if not nodes:
        st.markdown('<div class="empty-state">No confirmed objects at this frame.</div>', unsafe_allow_html=True)
        return

    for node_id in sorted(nodes):
        node = nodes[node_id]
        outgoing = [edge for edge in edges if edge.subject_id == node_id]
        st.markdown(
            f"<div class='graph-object'><div class='graph-object-title'>{node.class_name.upper()} · #{node_id.split('_')[-1]}</div>"
            f"<div class='graph-object-meta'>{node_id} · {node.state.value} · confidence {node.track.detection_confidence:.2f}</div>",
            unsafe_allow_html=True,
        )
        if outgoing:
            for edge in outgoing:
                target = nodes.get(edge.object_id)
                if target is None:
                    continue
                confidence = float(edge.latest_evidence.confidence) if edge.latest_evidence else 0.0
                st.markdown(
                    f"<div class='graph-relation'><b>{edge.predicate}</b> → {target.class_name.upper()} · #{edge.object_id.split('_')[-1]} "
                    f"<span style='float:right;color:#788591'>{confidence:.2f}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div class='small-muted'>No supported outgoing relation.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)



def _rgb_depth_point_cloud(packet: Any, max_points: int = 6000, stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    depth = getattr(packet, "depth", None)
    rgb = getattr(packet, "rgb", None)
    intrinsics = getattr(packet, "camera_intrinsics", None)
    if depth is None or rgb is None or intrinsics is None:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    try:
        depth_m = packet.depth_model.depth_to_meters(depth) if getattr(packet, "depth_model", None) is not None else np.asarray(depth, dtype=np.float32)
    except Exception:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    h, w = depth_m.shape[:2]
    step = max(1, int(stride))
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth_m[::step, ::step]
    valid = np.isfinite(z) & (z > 0.15) & (z < 5.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z[valid]
    x = (u - float(intrinsics.cx)) * z / float(intrinsics.fx)
    y = (v - float(intrinsics.cy)) * z / float(intrinsics.fy)
    points = np.column_stack((x, y, z)).astype(np.float32)
    colors = np.asarray(rgb[::step, ::step], dtype=np.uint8)[valid]
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int32)
        points, colors = points[idx], colors[idx]
    pose = getattr(packet, "world_T_camera", None)
    if pose is not None:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            points = (points.astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    return points, colors


def _track_point_cloud(graph: Any, frame_index: int, max_points_per_object: int = 900) -> dict[str, np.ndarray]:
    clouds: dict[str, np.ndarray] = {}
    for node in current_frame_nodes(graph, frame_index):
        observation = latest_observation(node, frame_index)
        geometry = getattr(observation, "object_geometry", None) if observation is not None else None
        points = getattr(geometry, "points_world_sampled", None) if geometry is not None else None
        if points is None:
            continue
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
            continue
        if len(points) > max_points_per_object:
            idx = np.linspace(0, len(points) - 1, max_points_per_object).astype(np.int32)
            points = points[idx]
        clouds[node.object_id] = points
    return clouds


def _rgb_hex_colors(colors: np.ndarray) -> list[str]:
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors]


def render_point_cloud(graph: Any, packet: Any, frame_index: int, show_objects: bool = True, show_relations: bool = True) -> Any:
    import plotly.graph_objects as go

    scene_points, scene_colors = _rgb_depth_point_cloud(packet)
    fig = go.Figure()

    if len(scene_points):
        fig.add_trace(go.Scatter3d(
            x=scene_points[:, 0], y=scene_points[:, 1], z=scene_points[:, 2],
            mode="markers", name="Scene",
            marker=dict(size=1.8, color=_rgb_hex_colors(scene_colors), opacity=0.72),
            hovertemplate="X %{x:.2f}<br>Y %{y:.2f}<br>Z %{z:.2f}<extra>Scene</extra>",
        ))

    nodes = {node.object_id: node for node in current_frame_nodes(graph, frame_index)}
    object_clouds = _track_point_cloud(graph, frame_index)

    if show_objects:
        for object_id, points in object_clouds.items():
            node = nodes.get(object_id)
            if node is None:
                continue
            color = track_color(object_id)
            color_css = f"rgb({color[0]},{color[1]},{color[2]})"
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="markers", name=f"{node.class_name} #{object_id.split('_')[-1]}",
                marker=dict(size=2.8, color=color_css, opacity=0.9),
                hovertemplate=(f"{node.class_name.upper()} #{object_id.split('_')[-1]}"
                               "<br>X %{x:.2f}<br>Y %{y:.2f}<br>Z %{z:.2f}<extra></extra>"),
            ))

    centroids: dict[str, np.ndarray] = {}
    for object_id, node in nodes.items():
        observation = latest_observation(node, frame_index)
        geometry = getattr(observation, "object_geometry", None) if observation is not None else None
        centroid = getattr(geometry, "centroid_world", None) if geometry is not None else None
        if centroid is None:
            continue
        centroid = np.asarray(centroid, dtype=np.float32)
        if centroid.shape == (3,) and np.isfinite(centroid).all():
            centroids[object_id] = centroid

    if show_objects and centroids:
        ids = sorted(centroids)
        pts = np.vstack([centroids[i] for i in ids])
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers+text", name="Tracked objects",
            text=[f"{nodes[i].class_name} #{i.split('_')[-1]}" for i in ids], textposition="top center",
            marker=dict(size=6, color=[f"rgb({track_color(i)[0]},{track_color(i)[1]},{track_color(i)[2]})" for i in ids]),
            hovertemplate=[f"{nodes[i].class_name.upper()} #{i.split('_')[-1]}<br>X %{{x:.2f}}<br>Y %{{y:.2f}}<br>Z %{{z:.2f}}<extra></extra>" for i in ids],
        ))

    if show_relations:
        for edge in display_edges(graph, frame_index):
            if edge.subject_id not in centroids or edge.object_id not in centroids:
                continue
            a, b = centroids[edge.subject_id], centroids[edge.object_id]
            color = relation_color(edge.predicate)
            color_css = f"rgb({color[0]},{color[1]},{color[2]})"
            fig.add_trace(go.Scatter3d(
                x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines+text",
                text=["", edge.predicate], textposition="middle center", name=edge.predicate,
                showlegend=False, line=dict(color=color_css, width=5),
                hovertemplate=f"{edge.predicate}<extra></extra>",
            ))

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#090d12", plot_bgcolor="#090d12",
        font=dict(color="#d9e1e8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02, x=0),
        scene=dict(
            xaxis=dict(title="X", showbackground=False, gridcolor="#27313a", zerolinecolor="#39444e"),
            yaxis=dict(title="Y", showbackground=False, gridcolor="#27313a", zerolinecolor="#39444e"),
            zaxis=dict(title="Z", showbackground=False, gridcolor="#27313a", zerolinecolor="#39444e"),
            bgcolor="#090d12", aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.05)),
        ),
    )
    return fig


class CurrentFrameGraphView:
    def __init__(self, graph: Any, frame_index: int):
        self.graph = graph
        self.frame_index = frame_index
        self.nodes = graph.nodes
        self.history = graph.history

    def get_active_nodes(self) -> list[Any]:
        return current_frame_nodes(self.graph, self.frame_index)

    def get_active_edges(self) -> list[Any]:
        return current_frame_edges(self.graph, self.frame_index)


@dataclass
class DemoRuntime:
    config_path: str
    source: Any
    packets: Any
    pipeline: OnlinePipeline
    packet: Any | None = None
    graph: Any | None = None
    fps: float = 0.0
    started: bool = False

    @classmethod
    def create(cls, config_path: str) -> "DemoRuntime":
        config = load_config(config_path)
        source = TUMReplaySource(config)
        return cls(config_path, source, iter(source), OnlinePipeline(config))

    def step(self) -> bool:
        try:
            packet = next(self.packets)
        except StopIteration:
            return False

        started = time.perf_counter()
        graph = self.pipeline.update(packet)
        elapsed = time.perf_counter() - started
        self.packet = packet
        self.graph = graph
        self.fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self.started = True
        return True

    def start_to_useful_frame(self, max_steps: int = 12) -> bool:
        for _ in range(max_steps):
            if not self.step():
                return self.graph is not None
            if self.graph is not None and self.packet is not None:
                frame = self.packet.frame_index
                if current_frame_nodes(self.graph, frame):
                    return True
        return self.graph is not None

    def reset(self) -> None:
        config = load_config(self.config_path)
        self.source = TUMReplaySource(config)
        self.packets = iter(self.source)
        self.pipeline = OnlinePipeline(config)
        self.packet = None
        self.graph = None
        self.fps = 0.0
        self.started = False


def render_query_panel(graph: Any, frame_index: int) -> None:
    frame_graph = CurrentFrameGraphView(graph, frame_index)
    engine = QueryEngine(frame_graph)
    nodes = frame_graph.get_active_nodes()

    if not nodes:
        st.info("No confirmed objects are available for a query at this frame.")
        return

    lookup = {node.object_id: node for node in nodes}
    object_ids = sorted(lookup)
    predicates = sorted({edge.predicate for edge in frame_graph.get_active_edges()})

    query_type = st.radio("Query", ["What does…", "What is…", "Between two objects"], horizontal=True)

    if query_type == "What does…":
        subject_id = st.selectbox("Subject", object_ids, format_func=lambda value: f"{lookup[value].class_name.upper()} · {value}")
        predicate = st.selectbox("Relation", predicates or ["ON"])
        results = engine.what_does(subject_id, predicate)
        if results:
            for node in results:
                st.success(f"{subject_id} → {predicate} → {node.class_name.upper()} · {node.object_id}")
        else:
            st.info("No supported relation at this frame.")

    elif query_type == "What is…":
        predicate = st.selectbox("Relation", predicates or ["ON"])
        object_id = st.selectbox("Object", object_ids, format_func=lambda value: f"{lookup[value].class_name.upper()} · {value}")
        results = engine.what_is(predicate, object_id)
        if results:
            for node in results:
                st.success(f"{node.class_name.upper()} · {node.object_id} → {predicate} → {object_id}")
        else:
            st.info("No supported relation at this frame.")

    else:
        subject_id = st.selectbox("Subject", object_ids, key="query_subject", format_func=lambda value: f"{lookup[value].class_name.upper()} · {value}")
        object_options = [value for value in object_ids if value != subject_id]
        if not object_options:
            st.info("At least two active objects are required.")
            return
        object_id = st.selectbox("Object", object_options, format_func=lambda value: f"{lookup[value].class_name.upper()} · {value}")
        results = engine.get_relations_between(subject_id, object_id)
        if results:
            for edge in results:
                st.success(f"{subject_id} → {edge.predicate} → {object_id}")
        else:
            st.info("No supported relation at this frame.")


def main() -> None:
    st.set_page_config(page_title="Dynamic 3D Scene Graph", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    if "runtime" not in st.session_state:
        st.session_state.runtime = DemoRuntime.create("configs/tum_fr1_desk.yaml")

    runtime: DemoRuntime = st.session_state.runtime

    st.markdown(
        "<div class='hero'><div><div class='hero-title'>Dynamic 3D Scene Graph</div>"
        "<div class='hero-subtitle'>RGB-D perception · causal tracking · 3D spatial relations · temporal graph</div></div>"
        "<div class='hero-badge'>REPLAY DEMO</div></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Demo control")
        config_path = st.text_input("Configuration", runtime.config_path)

        if config_path != runtime.config_path:
            try:
                st.session_state.runtime = DemoRuntime.create(config_path)
                st.rerun()
            except Exception as exc:
                st.error(f"Configuration error: {exc}")

        start_col, next_col = st.columns(2)
        start_clicked = start_col.button("Start demo", use_container_width=True)
        next_clicked = next_col.button("Next frame", use_container_width=True)
        reset_clicked = st.button("Reset", use_container_width=True)

        overlay_filter = st.selectbox("RGB relation overlay", DISPLAY_FILTERS, index=0)

        st.markdown("---")
        st.markdown(
            "<div class='demo-note'>The scene view is strictly frame-local. "
            "Historical relations stay in History and are never mixed into the current RGB frame.</div>",
            unsafe_allow_html=True,
        )

        if reset_clicked:
            runtime.reset()
            st.rerun()

        if start_clicked:
            try:
                runtime.start_to_useful_frame()
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return

        elif next_clicked:
            try:
                if runtime.started and runtime.step():
                    pass
                elif not runtime.started:
                    runtime.start_to_useful_frame()
                else:
                    st.warning("End of configured replay.")
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return

    if runtime.packet is None or runtime.graph is None:
        st.info("Press **Start demo** to initialize the replay and skip the tracker warm-up frames.")
        return

    graph = runtime.graph
    frame_index = runtime.packet.frame_index
    objects = node_rows(graph, frame_index)
    visible_edges = display_edges(graph, frame_index)
    events = history_rows(graph)

    metrics = st.columns(5)
    for column, (label, value) in zip(metrics, [
        ("FRAME", frame_index),
        ("OBJECTS", len(objects)),
        ("RELATIONS", len(visible_edges)),
        ("GRAPH EVENTS", len(events)),
        ("PIPELINE FPS", f"{runtime.fps:.2f}"),
    ]):
        column.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    scene_tab, cloud_tab, query_tab, history_tab = st.tabs(["Scene", "3D Point Cloud", "Queries", "History"])

    with scene_tab:
        scene_col, graph_col = st.columns([1.75, 1.0], gap="large")

        with scene_col:
            st.markdown("<div class='panel-title'>Current RGB frame · frame-local graph projection</div>", unsafe_allow_html=True)
            st.image(render_rgb_scene(runtime.packet.rgb, graph, frame_index, overlay_filter), use_container_width=True)
            st.caption(f"Frame {frame_index} · {len(visible_edges)} current supported relation(s)")

        with graph_col:
            st.markdown("<div class='panel-title'>Current scene graph</div>", unsafe_allow_html=True)
            render_current_graph(graph, frame_index)

        st.write("")
        objects_col, relations_col = st.columns(2, gap="large")

        with objects_col:
            st.markdown("<div class='panel-title'>Tracked objects</div>", unsafe_allow_html=True)
            st.dataframe(
                as_dataframe(objects, ["track_id", "class", "state", "confidence", "observations"]),
                hide_index=True,
                use_container_width=True,
            )

        with relations_col:
            st.markdown("<div class='panel-title'>Current-frame relations</div>", unsafe_allow_html=True)
            st.dataframe(
                as_dataframe(edge_rows(graph, frame_index), ["subject", "predicate", "object", "confidence", "frame", "evidence"]),
                hide_index=True,
                use_container_width=True,
            )

    with cloud_tab:
        st.markdown("<div class='panel-title'>Current 3D point cloud · world coordinates</div>", unsafe_allow_html=True)
        control_a, control_b = st.columns(2)
        show_objects_3d = control_a.checkbox("Highlight tracked objects", value=True)
        show_relations_3d = control_b.checkbox("Show current-frame relations", value=True)
        try:
            import plotly  # noqa: F401
            fig = render_point_cloud(graph, runtime.packet, frame_index, show_objects_3d, show_relations_3d)
            if len(fig.data) == 0:
                st.info("No valid depth points are available for this frame.")
            else:
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
                scene_points, _ = _rgb_depth_point_cloud(runtime.packet)
                object_count = len(_track_point_cloud(graph, frame_index))
                st.caption(f"Frame {frame_index} · {len(scene_points):,} scene points · {object_count} tracked point clouds")
        except ImportError:
            st.error("Plotly is required for the interactive 3D point-cloud view. Install it with: pip install plotly")
        except Exception as exc:
            st.error(f"3D point-cloud rendering failed: {exc}")

    with query_tab:
        st.markdown("<div class='panel-title'>Query the current frame</div>", unsafe_allow_html=True)
        render_query_panel(graph, frame_index)

    with history_tab:
        history_col, evidence_col = st.columns([1.65, 1.0], gap="large")
        with history_col:
            st.markdown("<div class='panel-title'>Temporal graph history</div>", unsafe_allow_html=True)
            st.dataframe(
                as_dataframe(events[-50:], ["frame", "event", "subject", "predicate", "object"]),
                hide_index=True,
                use_container_width=True,
            )
        with evidence_col:
            st.markdown("<div class='panel-title'>Current-frame evidence</div>", unsafe_allow_html=True)
            current_edges = current_frame_edges(graph, frame_index)
            if not current_edges:
                st.info("No supported relation evidence at this frame.")
            else:
                labels = [f"{e.subject_id} → {e.predicate} → {e.object_id}" for e in current_edges]
                selected = st.selectbox("Relation", labels)
                edge = current_edges[labels.index(selected)]
                evidence = edge.latest_evidence
                st.json({
                    "predicate": evidence.predicate,
                    "subject_id": evidence.subject_id,
                    "object_id": evidence.object_id,
                    "frame_index": evidence.frame_index,
                    "result": evidence.result.value,
                    "confidence": evidence.confidence,
                    "evidence_type": evidence.evidence_type,
                    "details": evidence.details,
                })

    st.caption(
        f"TUM Freiburg1 Desk · frame {frame_index} · "
        f"{len(objects)} active objects · {len(visible_edges)} displayed current-frame relations · "
        f"{len(events)} historical graph events"
    )


if __name__ == "__main__":
    main()
