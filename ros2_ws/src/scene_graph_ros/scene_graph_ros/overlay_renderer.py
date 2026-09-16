"""2D perception overlay rendering for ROS image publishers."""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Tuple, Sequence
import cv2
import numpy as np

OBJECT_PALETTE = [
    (82, 177, 238),
    (89, 208, 164),
    (171, 133, 235),
    (239, 177, 87),
    (82, 193, 207),
    (229, 116, 150),
    (164, 181, 112),
    (133, 151, 226),
]

RELATION_COLORS = {
    "ON": (54, 207, 169),
    "UNDER": (54, 207, 169),
    "INSIDE": (109, 164, 240),
    "CONTAINING": (109, 164, 240),
    "NEAR": (188, 143, 246),
    "FAR": (150, 160, 175),
    "LEFT_OF": (248, 188, 90),
    "RIGHT_OF": (248, 188, 90),
    "ABOVE": (248, 188, 90),
    "BELOW": (248, 188, 90),
    "IN_FRONT_OF": (94, 211, 241),
    "BEHIND": (94, 211, 241),
    "OCCLUDING": (246, 105, 105),
    "OCCLUDED_BY": (246, 105, 105),
}


def track_color(track_id: str) -> Tuple[int, int, int]:
    """Deterministically map a track ID string to a consistent RGB color."""
    digest = hashlib.sha256(str(track_id).encode("utf-8")).digest()
    return OBJECT_PALETTE[digest[0] % len(OBJECT_PALETTE)]


def relation_color(predicate: str) -> Tuple[int, int, int]:
    """Map relation predicate string to an RGB color."""
    return RELATION_COLORS.get(predicate, (215, 220, 225))


def clamp_bbox(bbox: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp [x1, y1, x2, y2] to image bounds."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return (
        int(np.clip(x1, 0, width - 1)),
        int(np.clip(y1, 0, height - 1)),
        int(np.clip(x2, 0, width - 1)),
        int(np.clip(y2, 0, height - 1)),
    )


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Return center point (cx, cy) of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def boundary_point(bbox: Tuple[int, int, int, int], target: Tuple[float, float]) -> Tuple[int, int]:
    """Calculate point on bounding box border facing towards target."""
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


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.20,
) -> np.ndarray:
    """Alpha-blend a boolean mask with the given color onto image."""
    if mask.shape[:2] != image.shape[:2]:
        return image
    overlay = image.copy()
    overlay[mask.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def draw_label(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    background: Tuple[int, int, int],
    scale: float = 0.40,
    text_color: Tuple[int, int, int] = (12, 16, 21),
) -> None:
    """Draw a text label badge with a colored background box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x = max(3, min(origin[0], image.shape[1] - tw - 8))
    y = max(th + 8, min(origin[1], image.shape[0] - 3))
    cv2.rectangle(image, (x, y - th - 7), (x + tw + 7, y + 3), background, -1, cv2.LINE_AA)
    cv2.putText(image, text, (x + 3, y - 3), font, scale, text_color, 1, cv2.LINE_AA)


def render_detections_overlay(
    rgb: np.ndarray,
    observations: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """Render 2D detections overlay: RGB + segmentation masks + boxes + confidence."""
    if rgb is None or rgb.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    canvas = rgb.copy()
    h, w = canvas.shape[:2]

    if not observations:
        return canvas

    for i, obs in enumerate(observations):
        color = OBJECT_PALETTE[i % len(OBJECT_PALETTE)]

        try:
            mask = obs.get_mask() if hasattr(obs, "get_mask") else getattr(obs, "mask", None)
        except Exception:
            mask = None

        if mask is not None:
            canvas = draw_mask(canvas, mask, color, alpha=0.25)

        bbox_xyxy = getattr(obs, "bbox_xyxy", None)
        if bbox_xyxy is not None:
            bx1, by1, bx2, by2 = clamp_bbox(bbox_xyxy, w, h)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)

            cls_name = getattr(obs, "class_name", "object")
            conf = float(getattr(obs, "confidence", 1.0))
            label = f"{cls_name.upper()} {conf:.2f}"
            draw_label(canvas, label, (bx1, max(18, by1 - 4)), color)

    return canvas


def render_tracks_overlay(
    rgb: np.ndarray,
    graph: Any,
    frame_index: Optional[int] = None,
) -> np.ndarray:
    """Render 2D tracks and relation overlay: RGB + context hulls + badges + sparse sibling relations."""
    if rgb is None or rgb.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    canvas = rgb.copy()
    h, w = canvas.shape[:2]

    if graph is None:
        return canvas

    active_nodes = graph.get_active_nodes()
    boxes: dict[str, Tuple[int, int, int, int]] = {}
    node_map: dict[str, Any] = {}

    for node in active_nodes:
        track = node.track
        node_id = str(track.object_id)
        node_map[node_id] = node

        obs = track.recent_observations[-1] if track.recent_observations else None
        if obs is None:
            continue

        bbox_xyxy = getattr(obs, "bbox_xyxy", None)
        if bbox_xyxy is None:
            continue

        box = clamp_bbox(bbox_xyxy, w, h)
        boxes[node_id] = box
        col = track_color(node_id)

        try:
            mask = obs.get_mask() if hasattr(obs, "get_mask") else getattr(obs, "mask", None)
        except Exception:
            mask = None

        if mask is not None:
            canvas = draw_mask(canvas, mask, col, alpha=0.20)

    spatial_contexts = getattr(graph, "spatial_contexts", {})

    # 1. Render Translucent Context Hulls for Discovered Spatial Contexts
    for cid, ctx in spatial_contexts.items():
        if cid == "world" or not getattr(ctx, "member_track_ids", None):
            continue

        ctx_boxes = [boxes[tid] for tid in ctx.member_track_ids if tid in boxes]
        if ctx.anchor_track_id and ctx.anchor_track_id in boxes:
            ctx_boxes.append(boxes[ctx.anchor_track_id])

        if ctx_boxes:
            min_x = max(0, min(b[0] for b in ctx_boxes) - 10)
            min_y = max(0, min(b[1] for b in ctx_boxes) - 10)
            max_x = min(w - 1, max(b[2] for b in ctx_boxes) + 10)
            max_y = min(h - 1, max(b[3] for b in ctx_boxes) + 10)

            overlay = canvas.copy()
            cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), (70, 95, 120), -1)
            canvas = cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0.0)
            cv2.rectangle(canvas, (min_x, min_y), (max_x, max_y), (130, 170, 210), 1, cv2.LINE_AA)

            anchor_node = graph.nodes.get(ctx.anchor_track_id)
            anchor_name = anchor_node.track.class_name.upper() if anchor_node else "ANCHOR"
            short_id = ctx.anchor_track_id[-4:] if ctx.anchor_track_id else ""
            draw_label(
                canvas,
                f"{anchor_name} #{short_id} [SPATIAL CONTEXT]",
                (min_x + 6, min_y + 16),
                (55, 80, 105),
                scale=0.38,
                text_color=(235, 245, 255),
            )

    # 2. Render Structural Connectors and Sibling Relations (No Giant All-Pairs Web)
    active_edges = [e for e in graph.edges.values() if e.is_active]
    priority = {
        "ON": 10, "SUPPORTED_BY": 10, "INSIDE": 9,
        "OCCLUDING": 8, "TOUCHING": 7, "NEAR": 6,
        "LEFT_OF": 5, "RIGHT_OF": 5, "ABOVE": 4, "IN_FRONT_OF": 4,
    }

    # Separate structural relations from sibling relations
    structural_edges = [e for e in active_edges if e.predicate in ("ON", "SUPPORTED_BY", "INSIDE")]
    sibling_edges = [e for e in active_edges if e.predicate not in ("ON", "SUPPORTED_BY", "INSIDE")]

    # Render small vertical connectors for structural support
    for edge in structural_edges:
        sub_id = str(edge.subject_id)
        obj_id = str(edge.object_id)
        if sub_id in boxes and obj_id in boxes:
            sub_box = boxes[sub_id]
            obj_box = boxes[obj_id]
            # Connect bottom of subject to top of anchor
            p_sub = (int((sub_box[0] + sub_box[2]) * 0.5), sub_box[3])
            p_obj = (int((sub_box[0] + sub_box[2]) * 0.5), max(obj_box[1], sub_box[3] + 4))
            col = relation_color(edge.predicate)
            cv2.line(canvas, p_sub, p_obj, col, 2, cv2.LINE_AA)

    # Sibling relations: rank by priority and cap rendered count to preserve readability
    ranked_sibling_edges = sorted(
        sibling_edges,
        key=lambda e: (
            priority.get(str(e.predicate), 0),
            float(getattr(e.latest_evidence, "confidence", 1.0) if hasattr(e, "latest_evidence") and e.latest_evidence else 1.0)
        ),
        reverse=True,
    )[:8]  # Visual cap: top 8 most salient sibling edges

    for edge in ranked_sibling_edges:
        sub_id = str(edge.subject_id)
        obj_id = str(edge.object_id)
        if sub_id not in boxes or obj_id not in boxes:
            continue

        start = boundary_point(boxes[sub_id], bbox_center(boxes[obj_id]))
        end = boundary_point(boxes[obj_id], bbox_center(boxes[sub_id]))
        col = relation_color(edge.predicate)

        cv2.arrowedLine(canvas, start, end, col, 2, cv2.LINE_AA, tipLength=0.08)
        mx, my = int((start[0] + end[0]) * 0.5), int((start[1] + end[1]) * 0.5 - 4)
        draw_label(canvas, edge.predicate, (mx, my), col, scale=0.35)

    # 3. Render Object Badges with Context Annotation
    for node_id, box in boxes.items():
        node = node_map[node_id]
        track = node.track
        col = track_color(node_id)
        bx1, by1, bx2, by2 = box

        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), col, 2, cv2.LINE_AA)
        short_id = node_id[-4:] if len(node_id) >= 4 else node_id

        # Context-aware badge
        if getattr(node, "is_spatial_anchor", False):
            badge = f"{track.class_name.upper()} #{short_id} [ANCHOR]"
        elif getattr(node, "spatial_context_id", "world") != "world":
            parent_cid = node.spatial_context_id
            parent_ctx = spatial_contexts.get(parent_cid)
            parent_anchor = graph.nodes.get(parent_ctx.anchor_track_id) if parent_ctx and parent_ctx.anchor_track_id else None
            p_name = parent_anchor.track.class_name.upper() if parent_anchor else "DESK"
            badge = f"{track.class_name.upper()} #{short_id} [on {p_name}]"
        else:
            badge = f"{track.class_name.upper()} #{short_id} [{node.state.value}]"

        draw_label(canvas, badge, (bx1, max(18, by1 - 4)), col)

    # 4. Status Bar
    n_objs = len(boxes)
    n_ctxs = sum(1 for c in spatial_contexts.values() if getattr(c, "context_id", "") != "world")
    n_rels = len(active_edges)
    status_text = f"Objects: {n_objs} | Contexts: {n_ctxs} | Confirmed Relations: {n_rels}"
    draw_label(canvas, status_text, (10, h - 10), (20, 24, 30), scale=0.42, text_color=(230, 235, 240))

    return canvas
