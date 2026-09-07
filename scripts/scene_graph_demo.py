from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    "ON": (62, 205, 165), "UNDER": (62, 205, 165),
    "INSIDE": (112, 170, 242), "CONTAINING": (112, 170, 242),
    "NEAR": (190, 145, 248), "FAR": (145, 155, 170),
    "LEFT_OF": (252, 188, 88), "RIGHT_OF": (252, 188, 88),
    "ABOVE": (252, 188, 88), "BELOW": (252, 188, 88),
    "IN_FRONT_OF": (98, 211, 242), "BEHIND": (98, 211, 242),
    "OCCLUDING": (248, 105, 105), "OCCLUDED_BY": (248, 105, 105),
}

ARROW_RELATIONS = {
    "LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW",
    "IN_FRONT_OF", "BEHIND", "OCCLUDING", "OCCLUDED_BY",
}

OBJECT_PALETTE = [
    (90, 190, 250), (100, 220, 175), (185, 145, 255), (255, 185, 95),
    (100, 215, 230), (245, 135, 165), (175, 190, 120), (150, 165, 245),
]

PREDICATES = ["ALL"] + list(RELATION_COLORS)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp{background:#0a0e13;color:#e9eef3}
        .block-container{max-width:1500px;padding-top:.55rem;padding-bottom:1.25rem}
        .hero-title{font-size:2rem;font-weight:720;letter-spacing:-.035em;line-height:1.05}
        .hero-subtitle{color:#8995a2;font-size:.88rem;margin:.2rem 0 .85rem}
        .metric-card{background:#111720;border:1px solid #202a35;border-radius:11px;padding:.65rem .8rem;min-height:68px}
        .metric-label{color:#778493;font-size:.66rem;text-transform:uppercase;letter-spacing:.09em}
        .metric-value{color:#f3f6f9;font-size:1.3rem;font-weight:680;margin-top:.12rem}
        .section-title{font-size:.74rem;color:#83909d;text-transform:uppercase;letter-spacing:.09em;margin:.15rem 0 .55rem}
        .object-card{background:#121922;border:1px solid #25313e;border-radius:10px;padding:.58rem .68rem;margin-bottom:.45rem}
        .relation-line{background:#0d131a;border:1px solid #202a34;border-radius:7px;padding:.34rem .45rem;margin-top:.3rem;font-size:.78rem}
        .small-muted{color:#768392;font-size:.72rem}
        div[data-testid="stDataFrame"]{border:1px solid #202a35;border-radius:9px;overflow:hidden}
        </style>
        """,
        unsafe_allow_html=True,
    )


def track_color(track_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    return OBJECT_PALETTE[digest[0] % len(OBJECT_PALETTE)]


def relation_color(predicate: str) -> tuple[int, int, int]:
    return RELATION_COLORS.get(predicate, (215, 220, 225))


def latest_observation(node: Any, frame_index: int | None = None) -> Any | None:
    observations = getattr(node.track, "recent_observations", None)
    if not observations:
        return None
    observation = observations[-1]
    if frame_index is None or observation.frame_index == frame_index:
        return observation
    return None


def current_frame_edges(graph: Any, frame_index: int | None = None) -> list[Any]:
    if frame_index is None:
        frame_index = getattr(graph, "current_frame_index", None)
    if frame_index is None:
        return []
    return [
        edge for edge in graph.get_active_edges()
        if edge.latest_evidence is not None
        and edge.latest_evidence.frame_index == frame_index
        and edge.latest_evidence.result == EvidenceResult.SUPPORTED
    ]


def _edge_pair_key(edge: Any) -> frozenset[str]:
    return frozenset((edge.subject_id, edge.object_id))


def _pick_display_edge(edges: list[Any], family: str, graph: Any, frame_index: int | None) -> Any:
    if len(edges) == 1:
        return edges[0]

    nodes = {node.object_id: node for node in graph.get_active_nodes()}
    subject = nodes.get(edges[0].subject_id)
    object_ = nodes.get(edges[0].object_id)

    if subject is not None and object_ is not None:
        subject_obs = latest_observation(subject, frame_index)
        object_obs = latest_observation(object_, frame_index)

        if subject_obs is not None and object_obs is not None:
            sb = np.asarray(subject_obs.bbox_xyxy, dtype=float)
            ob = np.asarray(object_obs.bbox_xyxy, dtype=float)
            sc = ((sb[0] + sb[2]) * .5, (sb[1] + sb[3]) * .5)
            oc = ((ob[0] + ob[2]) * .5, (ob[1] + ob[3]) * .5)

            if family == "horizontal":
                desired = "RIGHT_OF" if sc[0] > oc[0] else "LEFT_OF"
                matches = [edge for edge in edges if edge.predicate == desired]
                if matches:
                    return matches[0]

            if family == "vertical":
                desired = "ABOVE" if sc[1] < oc[1] else "BELOW"
                matches = [edge for edge in edges if edge.predicate == desired]
                if matches:
                    return matches[0]

    preference = {
        "support": ["ON", "UNDER"],
        "containment": ["INSIDE", "CONTAINING"],
        "depth": ["IN_FRONT_OF", "BEHIND"],
        "occlusion": ["OCCLUDING", "OCCLUDED_BY"],
    }

    for predicate in preference.get(family, []):
        matches = [edge for edge in edges if edge.predicate == predicate]
        if matches:
            return matches[0]

    return max(edges, key=lambda edge: float(edge.latest_evidence.confidence))


def display_edges(graph: Any, frame_index: int | None = None) -> list[Any]:
    grouped: dict[tuple[frozenset[str], str], list[Any]] = {}
    for edge in current_frame_edges(graph, frame_index):
        family = RELATION_FAMILIES.get(edge.predicate, edge.predicate)
        grouped.setdefault((_edge_pair_key(edge), family), []).append(edge)

    selected = [
        _pick_display_edge(edges, family, graph, frame_index)
        for (_, family), edges in grouped.items()
    ]
    return sorted(selected, key=lambda edge: (edge.subject_id, edge.predicate, edge.object_id))


def node_rows(graph: Any, frame_index: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for node in graph.get_active_nodes():
        observation = latest_observation(node, frame_index)
        if observation is None:
            continue
        track = node.track
        rows.append({
            "track_id": node.object_id,
            "class": node.class_name,
            "state": node.state.value,
            "confidence": round(float(track.detection_confidence), 3),
            "observations": int(track.observation_count),
        })
    return sorted(rows, key=lambda row: row["track_id"])


def edge_rows(graph: Any, frame_index: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for edge in display_edges(graph, frame_index):
        subject = graph.nodes.get(edge.subject_id)
        object_ = graph.nodes.get(edge.object_id)
        evidence = edge.latest_evidence
        if subject is None or object_ is None or evidence is None:
            continue
        rows.append({
            "subject_id": edge.subject_id,
            "subject": subject.class_name,
            "predicate": edge.predicate,
            "object_id": edge.object_id,
            "object": object_.class_name,
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
        int(np.clip(x1, 0, width - 1)),
        int(np.clip(y1, 0, height - 1)),
        int(np.clip(x2, 0, width - 1)),
        int(np.clip(y2, 0, height - 1)),
    )


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * .5, (y1 + y2) * .5)


def bbox_boundary_point(bbox: tuple[int, int, int, int], target: tuple[float, float]) -> tuple[int, int]:
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


def draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = .12) -> np.ndarray:
    if mask.shape[:2] != image.shape[:2]:
        return image
    overlay = image.copy()
    overlay[mask.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], background: tuple[int, int, int], scale: float = .40) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x = max(3, min(origin[0], image.shape[1] - tw - 8))
    y = max(th + 8, min(origin[1], image.shape[0] - 3))
    cv2.rectangle(image, (x, y - th - 7), (x + tw + 7, y + 3), background, -1, cv2.LINE_AA)
    cv2.putText(image, text, (x + 3, y - 3), font, scale, (10, 14, 19), 1, cv2.LINE_AA)


def render_rgb_scene(rgb: np.ndarray, graph: Any, frame_index: int | None = None, predicate_filter: str = "ALL") -> np.ndarray:
    if frame_index is None:
        frame_index = getattr(graph, "current_frame_index", None)

    canvas = rgb.copy()
    height, width = canvas.shape[:2]
    nodes = {}
    boxes: dict[str, tuple[int, int, int, int]] = {}

    for node in graph.get_active_nodes():
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
    if predicate_filter != "ALL":
        edges = [edge for edge in edges if edge.predicate == predicate_filter]

    for edge in edges:
        if edge.subject_id not in boxes or edge.object_id not in boxes:
            continue

        sb = boxes[edge.subject_id]
        ob = boxes[edge.object_id]
        start = bbox_boundary_point(sb, bbox_center(ob))
        end = bbox_boundary_point(ob, bbox_center(sb))
        color = relation_color(edge.predicate)

        if edge.predicate in ARROW_RELATIONS:
            cv2.arrowedLine(canvas, start, end, color, 2, cv2.LINE_AA, tipLength=.075)
        else:
            cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)

        midpoint = (int((start[0] + end[0]) * .5), int((start[1] + end[1]) * .5) - 4)
        draw_label(canvas, edge.predicate, midpoint, color, .37)

    for node_id, node in sorted(nodes.items()):
        x1, y1, x2, y2 = boxes[node_id]
        color = track_color(node_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        draw_label(canvas, f"{node.class_name.upper()} · #{node_id.split('_')[-1]}", (x1, max(20, y1 - 5)), color)

    return canvas


def render_current_graph(graph: Any, frame_index: int) -> None:
    nodes = {
        node.object_id: node
        for node in graph.get_active_nodes()
        if latest_observation(node, frame_index) is not None
    }
    edges = display_edges(graph, frame_index)

    if not nodes:
        st.info("No stable tracked objects at this frame.")
        return

    for node_id in sorted(nodes):
        node = nodes[node_id]
        st.markdown(f"**{node.class_name.upper()} · #{node_id.split('_')[-1]}**")
        st.caption(f"{node_id} · {node.state.value} · confidence {node.track.detection_confidence:.2f}")
        outgoing = [edge for edge in edges if edge.subject_id == node_id]
        for edge in outgoing:
            target = nodes.get(edge.object_id)
            if target is None:
                continue
            confidence = float(edge.latest_evidence.confidence) if edge.latest_evidence else 0.0
            st.write(f"↳ {edge.predicate} → {target.class_name.upper()} · #{edge.object_id.split('_')[-1]} · {confidence:.2f}")


class CurrentFrameGraphView:
    def __init__(self, graph: Any, frame_index: int):
        self.graph = graph
        self.frame_index = frame_index
        self.nodes = graph.nodes
        self.history = graph.history

    def get_active_nodes(self) -> list[Any]:
        return [
            node for node in self.graph.get_active_nodes()
            if latest_observation(node, self.frame_index) is not None
        ]

    def get_active_edges(self) -> list[Any]:
        return display_edges(self.graph, self.frame_index)



class CurrentFrameGraphView:
    def __init__(self, graph: Any, frame_index: int):
        self.graph = graph
        self.frame_index = frame_index
        self.nodes = graph.nodes
        self.history = graph.history

    def get_active_nodes(self) -> list[Any]:
        return [node for node in self.graph.get_active_nodes() if latest_observation(node, self.frame_index) is not None]

    def get_active_edges(self) -> list[Any]:
        return display_edges(self.graph, self.frame_index)

def render_query_panel(graph: Any, frame_index: int) -> None:
    frame_graph = CurrentFrameGraphView(graph, frame_index)
    nodes = frame_graph.get_active_nodes()

    if not nodes:
        st.info("No stable objects are available for current-frame queries.")
        return

    engine = QueryEngine(frame_graph)
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
        subject_id = st.selectbox("Subject", object_ids, key="between_subject", format_func=lambda value: f"{lookup[value].class_name.upper()} · {value}")
        object_options = [value for value in object_ids if value != subject_id]
        if not object_options:
            st.info("At least two objects are required.")
            return
        object_id = st.selectbox("Object", object_options, format_func=lambda value: f"{lookup[value].class_name.upper()} · {value}")
        results = engine.get_relations_between(subject_id, object_id)
        if results:
            for edge in results:
                confidence = float(edge.latest_evidence.confidence) if edge.latest_evidence else 0.0
                st.success(f"{subject_id} → {edge.predicate} → {object_id} · {confidence:.2f}")
        else:
            st.info("No supported relation at this frame.")


@dataclass
class DemoRuntime:
    config_path: str
    source: Any
    packets: Any
    pipeline: OnlinePipeline
    packet: Any | None = None
    graph: Any | None = None
    fps: float = 0.0
    frames_processed: int = 0
    warmup_frames: int = 0

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
        self.frames_processed += 1
        return True

    def warm_start(self, max_frames: int = 8) -> bool:
        for _ in range(max_frames):
            if not self.step():
                return self.graph is not None
            if self.packet is not None and self.graph is not None:
                if node_rows(self.graph, self.packet.frame_index) and current_frame_edges(self.graph, self.packet.frame_index):
                    self.warmup_frames += self.frames_processed
                    return True
        self.warmup_frames += self.frames_processed
        return self.graph is not None

    def reset(self) -> None:
        config = load_config(self.config_path)
        self.source = TUMReplaySource(config)
        self.packets = iter(self.source)
        self.pipeline = OnlinePipeline(config)
        self.packet = None
        self.graph = None
        self.fps = 0.0
        self.frames_processed = 0
        self.warmup_frames = 0


def main() -> None:
    st.set_page_config(page_title="Dynamic 3D Scene Graph", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    if "runtime" not in st.session_state:
        st.session_state.runtime = DemoRuntime.create("configs/tum_fr1_desk.yaml")
        st.session_state.started = False

    runtime: DemoRuntime = st.session_state.runtime

    st.markdown(
        "<div class='hero-title'>Dynamic 3D Scene Graph</div>"
        "<div class='hero-subtitle'>RGB-D scene understanding · causal tracking · spatial relations · temporal graph</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Demo")
        config_path = st.text_input("Configuration", runtime.config_path)

        if config_path != runtime.config_path:
            st.session_state.runtime = DemoRuntime.create(config_path)
            st.session_state.started = False
            st.rerun()

        reset_clicked, next_clicked = st.columns(2)
        reset = reset_clicked.button("Reset", use_container_width=True)
        advance = next_clicked.button("Next frame", use_container_width=True)

        relation_filter = st.selectbox("RGB relation overlay", PREDICATES, index=0)

        if reset:
            runtime.reset()
            st.session_state.started = False
            st.rerun()

    if not st.session_state.started:
        with st.spinner("Initializing scene graph and finding first useful frame…"):
            runtime.warm_start(max_frames=8)
        st.session_state.started = True

    if advance:
        try:
            if not runtime.step():
                st.warning("End of configured replay.")
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            return

    if runtime.packet is None or runtime.graph is None:
        st.info("No frame available from the configured replay.")
        return

    frame_index = runtime.packet.frame_index
    graph = runtime.graph
    objects = node_rows(graph, frame_index)
    visible_edges = display_edges(graph, frame_index)
    events = history_rows(graph)

    metric_values = [
        ("FRAME", frame_index),
        ("OBJECTS", len(objects)),
        ("RELATIONS", len(visible_edges)),
        ("EVENTS", len(events)),
        ("FPS", f"{runtime.fps:.2f}"),
    ]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, metric_values):
        col.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    scene_tab, query_tab, history_tab = st.tabs(["Scene", "Queries", "History"])

    with scene_tab:
        scene_col, graph_col = st.columns([1.8, 1.0], gap="large")

        with scene_col:
            st.markdown("<div class='section-title'>RGB scene · current-frame graph</div>", unsafe_allow_html=True)
            st.image(
                render_rgb_scene(runtime.packet.rgb, graph, frame_index, relation_filter),
                use_container_width=True,
            )
            st.caption(f"Frame {frame_index} · {len(visible_edges)} current-frame supported relation(s)")

        with graph_col:
            st.markdown("<div class='section-title'>Current semantic graph</div>", unsafe_allow_html=True)
            render_current_graph(graph, frame_index)

        objects_col, relations_col = st.columns(2, gap="large")

        with objects_col:
            st.markdown("<div class='section-title'>Tracked objects</div>", unsafe_allow_html=True)
            st.dataframe(
                as_dataframe(objects, ["track_id", "class", "state", "confidence", "observations"]),
                hide_index=True,
                use_container_width=True,
            )

        with relations_col:
            st.markdown("<div class='section-title'>Current-frame relations</div>", unsafe_allow_html=True)
            st.dataframe(
                as_dataframe(
                    edge_rows(graph, frame_index),
                    ["subject_id", "subject", "predicate", "object_id", "object", "confidence", "frame", "evidence"],
                ),
                hide_index=True,
                use_container_width=True,
            )

    with query_tab:
        st.markdown("<div class='section-title'>Query the current-frame graph</div>", unsafe_allow_html=True)
        render_query_panel(graph, frame_index)

        st.markdown("<div class='section-title'>Selected relation evidence</div>", unsafe_allow_html=True)
        if visible_edges:
            labels = [f"{edge.subject_id} → {edge.predicate} → {edge.object_id}" for edge in visible_edges]
            selected = st.selectbox("Relation", labels)
            edge = visible_edges[labels.index(selected)]
            evidence = edge.latest_evidence
            if evidence is not None:
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
        else:
            st.info("No current-frame supported relation evidence.")

    with history_tab:
        history_col, recent_col = st.columns([1.7, 1.0], gap="large")

        with history_col:
            st.markdown("<div class='section-title'>Graph event history</div>", unsafe_allow_html=True)
            st.dataframe(
                as_dataframe(events[-75:], ["frame", "event", "subject", "predicate", "object"]),
                hide_index=True,
                use_container_width=True,
            )

        with recent_col:
            st.markdown("<div class='section-title'>Recent changes</div>", unsafe_allow_html=True)
            for row in reversed(events[-12:]):
                st.write(
                    f"#{row['frame']} · {row['event']} · "
                    f"{row['subject']} {row['predicate']} {row['object']}".strip()
                )

    st.caption(
        f"TUM Freiburg1 Desk · frame {frame_index} · "
        f"{len(objects)} visible tracked object(s) · "
        f"{len(visible_edges)} current-frame relation(s) · "
        f"{len(events)} total graph events"
    )


if __name__ == "__main__":
    main()
