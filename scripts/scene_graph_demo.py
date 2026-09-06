"""Interactive, evidence-backed viewer for the Dynamic 3D Scene Graph.

Launch from the repository root with:
    streamlit run scripts/scene_graph_demo.py

The viewer processes the configured TUM replay sequentially.  It never loads
stored observations or invents graph output: every item shown is extracted from
the current ``TemporalSceneGraph`` and its ``GraphHistory``.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scene_graph.config import load_config
from scene_graph.data.tum_source import TUMReplaySource
from scene_graph.graph.query import QueryEngine
from scene_graph.pipeline.online_pipeline import OnlinePipeline


PREDICATES = [
    "ALL", "ON", "NEAR", "FAR", "LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW",
    "IN_FRONT_OF", "BEHIND", "OCCLUDING",
]
ARROW_PREDICATES = {"LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "IN_FRONT_OF", "BEHIND", "OCCLUDING"}


def track_color(track_id: str) -> tuple[int, int, int]:
    """Return a stable, bright RGB colour for a track id."""
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    hsv = np.uint8([[[digest[0] % 180, 180, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(value) for value in bgr)


def latest_observation(node: Any) -> Any | None:
    observations = node.track.recent_observations
    return observations[-1] if observations else None


def node_rows(graph: Any) -> list[dict[str, Any]]:
    rows = []
    for node in graph.get_active_nodes():
        track = node.track
        rows.append({
            "track_id": node.object_id,
            "class_name": node.class_name,
            "state": node.state.value,
            "detection_confidence": round(float(track.detection_confidence), 3),
            "observation_count": track.observation_count,
            "track_observation_ratio": round(float(track.track_observation_ratio), 3),
        })
    return sorted(rows, key=lambda row: row["track_id"])


def edge_rows(graph: Any, supported_only: bool = True) -> list[dict[str, Any]]:
    edges = graph.get_active_edges() if supported_only else graph.edges.values()
    rows = []
    for edge in edges:
        subject = graph.nodes.get(edge.subject_id)
        object_ = graph.nodes.get(edge.object_id)
        if subject is None or object_ is None:
            continue
        evidence = edge.latest_evidence
        rows.append({
            "subject_id": edge.subject_id,
            "subject_class": subject.class_name,
            "predicate": edge.predicate,
            "object_id": edge.object_id,
            "object_class": object_.class_name,
            "state": edge.state.value,
            "confidence": round(float(evidence.confidence), 3) if evidence else None,
            "frame": evidence.frame_index if evidence else graph.current_frame_index,
            "evidence_type": evidence.evidence_type if evidence else None,
        })
    return sorted(rows, key=lambda row: (row["subject_id"], row["predicate"], row["object_id"]))


def history_rows(graph: Any) -> list[dict[str, Any]]:
    return [{
        "frame": event.frame_index,
        "event": event.event_type.value.upper(),
        "subject": event.subject_id,
        "predicate": event.predicate or "",
        "object": event.object_id or "",
    } for event in graph.history.events]


def as_dataframe(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Build a compact table that retains its columns even with no graph data."""
    return pd.DataFrame(rows, columns=columns)


def _anchor_for(node: Any, image_shape: tuple[int, ...]) -> tuple[int, int] | None:
    observation = latest_observation(node)
    if observation is None:
        return None
    x1, y1, x2, y2 = np.asarray(observation.bbox_xyxy, dtype=float)
    height, width = image_shape[:2]
    return (int(np.clip((x1 + x2) / 2, 0, width - 1)), int(np.clip((y1 + y2) / 2, 0, height - 1)))


def render_rgb_scene(rgb: np.ndarray, graph: Any, predicate_filter: str = "ALL") -> np.ndarray:
    """Overlay real tracks and currently supported graph edges on an RGB frame."""
    canvas = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    nodes = {node.object_id: node for node in graph.get_active_nodes()}
    # Relations first so object labels always remain readable.
    pair_offsets: dict[tuple[str, str], int] = {}
    for edge in graph.get_active_edges():
        if predicate_filter != "ALL" and edge.predicate != predicate_filter:
            continue
        subject, object_ = nodes.get(edge.subject_id), nodes.get(edge.object_id)
        if subject is None or object_ is None:
            continue
        start, end = _anchor_for(subject, canvas.shape), _anchor_for(object_, canvas.shape)
        if start is None or end is None:
            continue
        colour = track_color(edge.subject_id)
        if edge.predicate in ARROW_PREDICATES:
            cv2.arrowedLine(canvas, start, end, colour, 2, cv2.LINE_AA, tipLength=0.04)
        else:
            cv2.line(canvas, start, end, colour, 1 if edge.predicate == "NEAR" else 2, cv2.LINE_AA)
        pair = (edge.subject_id, edge.object_id)
        offset = pair_offsets.get(pair, 0)
        pair_offsets[pair] = offset + 1
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 14 * offset)
        cv2.putText(canvas, edge.predicate, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, edge.predicate, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)

    for node in nodes.values():
        observation = latest_observation(node)
        if observation is None:
            continue
        x1, y1, x2, y2 = np.asarray(observation.bbox_xyxy, dtype=int)
        colour = track_color(node.object_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        label = f"{node.class_name.upper()}  {node.object_id}  {node.track.detection_confidence:.2f}"
        label_y = max(18, y1 - 7)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
        cv2.rectangle(canvas, (x1, label_y - text_height - 6), (x1 + text_width + 6, label_y + 3), colour, -1)
        cv2.putText(canvas, label, (x1 + 3, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (15, 15, 15), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def graph_markdown(graph: Any) -> str:
    nodes = {node.object_id: node for node in graph.get_active_nodes()}
    if not nodes:
        return "No stable object tracks yet. Step through frames to build temporal evidence."
    lines = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        lines.append(f"**{node.class_name.upper()}** · `{node.object_id}` · {node.state.value}")
        outgoing = [edge for edge in graph.get_active_edges() if edge.subject_id == node_id]
        for edge in sorted(outgoing, key=lambda item: (item.predicate, item.object_id)):
            target = nodes.get(edge.object_id)
            target_text = f"{target.class_name.upper()} · `{edge.object_id}`" if target else f"`{edge.object_id}`"
            confidence = edge.latest_evidence.confidence if edge.latest_evidence else 0.0
            lines.append(f"&nbsp;&nbsp;&nbsp;└─ **{edge.predicate}** ({confidence:.2f}) → {target_text}")
    return "<br>".join(lines)


@dataclass
class DemoRuntime:
    config_path: str
    source: Any
    packets: Any
    pipeline: OnlinePipeline
    packet: Any | None = None
    graph: Any | None = None
    fps: float = 0.0

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
        self.graph = self.pipeline.update(packet)
        elapsed = time.perf_counter() - started
        self.packet = packet
        self.fps = 1.0 / elapsed if elapsed else 0.0
        return True


def render_query_panel(graph: Any, st: Any) -> None:
    engine = QueryEngine(graph)
    nodes = graph.get_active_nodes()
    if not nodes:
        st.info("Queries become available after stable tracks are created.")
        return
    object_ids = [node.object_id for node in nodes]
    predicates = sorted({edge.predicate for edge in graph.get_active_edges()}) or ["ON"]
    query_type = st.selectbox("Query", ["What does…", "What is…", "Relations between…"])
    if query_type == "What does…":
        subject = st.selectbox("Subject", object_ids)
        predicate = st.selectbox("Relation", predicates)
        results = engine.what_does(subject, predicate)
        st.write([f"{subject} → {predicate} → {node.object_id}" for node in results] or "No current supported result.")
    elif query_type == "What is…":
        predicate = st.selectbox("Relation", predicates)
        object_id = st.selectbox("Object", object_ids)
        results = engine.what_is(predicate, object_id)
        st.write([f"{node.object_id} → {predicate} → {object_id}" for node in results] or "No current supported result.")
    else:
        subject = st.selectbox("Subject", object_ids)
        object_id = st.selectbox("Object", object_ids, index=min(1, len(object_ids) - 1))
        results = engine.get_relations_between(subject, object_id)
        st.write([edge.predicate for edge in results] or "No current supported relation.")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Dynamic 3D Scene Graph", layout="wide")
    st.title("DYNAMIC 3D SCENE GRAPH")
    st.caption("RGB Scene with 3D-derived Scene-Graph Relations · TUM Freiburg1 Desk")
    config_path = st.sidebar.text_input("Configuration", "configs/tum_fr1_desk.yaml")
    st.sidebar.selectbox("Mode", ["Replay", "Live"], disabled=True, help="Replay is the deterministic demo mode; camera capture is intentionally not added.")
    if "demo_runtime" not in st.session_state or st.session_state.demo_runtime.config_path != config_path:
        try:
            st.session_state.demo_runtime = DemoRuntime.create(config_path)
        except Exception as exc:
            st.error(f"Could not initialize the TUM pipeline: {exc}")
            return
    runtime: DemoRuntime = st.session_state.demo_runtime
    if "demo_playing" not in st.session_state:
        st.session_state.demo_playing = False
    controls = st.columns(4)
    if controls[0].button("Reset replay"):
        st.session_state.demo_runtime = DemoRuntime.create(config_path)
        st.rerun()
    advance = controls[1].button("Next frame")
    if controls[2].button("Pause" if st.session_state.demo_playing else "Play"):
        st.session_state.demo_playing = not st.session_state.demo_playing
    relation_filter = controls[3].selectbox("Relation overlay", PREDICATES)
    if advance or st.session_state.demo_playing:
        if not runtime.step():
            st.session_state.demo_playing = False
            st.warning("End of the configured replay window. Reset to run it again.")
        elif st.session_state.demo_playing:
            time.sleep(0.05)

    if runtime.packet is None or runtime.graph is None:
        st.info("Ready. Select **Next frame** to begin deterministic TUM replay.")
        return
    graph = runtime.graph
    node_data, relation_data, event_data = node_rows(graph), edge_rows(graph), history_rows(graph)
    st.caption(f"DATASET  TUM Freiburg1 Desk   |   FRAME  {runtime.packet.frame_index}   |   OBJECTS  {len(node_data)}   |   RELATIONS  {len(relation_data)}   |   TRACKS  {len(graph.nodes)}   |   EVENTS  {len(event_data)}   |   PIPELINE FPS  {runtime.fps:.2f}")
    left, right = st.columns((3, 2))
    with left:
        st.subheader("RGB Scene + 3D Scene-Graph Relations")
        st.image(render_rgb_scene(runtime.packet.rgb, graph, relation_filter), use_container_width=True)
    with right:
        st.subheader("Current Scene Graph")
        st.markdown(graph_markdown(graph), unsafe_allow_html=True)
    objects_col, relations_col = st.columns(2)
    with objects_col:
        st.subheader("Objects")
        st.dataframe(as_dataframe(node_data, ["track_id", "class_name", "state", "detection_confidence", "observation_count", "track_observation_ratio"]), hide_index=True, use_container_width=True)
    with relations_col:
        st.subheader("Supported Relations")
        st.dataframe(as_dataframe(relation_data, ["subject_id", "subject_class", "predicate", "object_id", "object_class", "state", "confidence", "frame"]), hide_index=True, use_container_width=True)
    st.subheader("Query")
    render_query_panel(graph, st)
    st.subheader("Temporal History")
    st.dataframe(as_dataframe(event_data[-30:], ["frame", "event", "subject", "predicate", "object"]), hide_index=True, use_container_width=True)
    with st.expander("Relation Evidence"):
        if relation_data:
            selected = st.selectbox("Active relation", range(len(relation_data)), format_func=lambda index: f"{relation_data[index]['subject_id']} → {relation_data[index]['predicate']} → {relation_data[index]['object_id']}")
            evidence = relation_data[selected]
            st.json({key: evidence[key] for key in ("predicate", "subject_id", "object_id", "confidence", "frame", "evidence_type")})
        else:
            st.info("No supported relation evidence at this frame.")
    # A rerun advances exactly one real packet, keeping the UI responsive even
    # when inference is slower than the desired presentation rate.
    if st.session_state.demo_playing:
        st.rerun()


if __name__ == "__main__":
    main()
