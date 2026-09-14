"""2D perception overlay rendering for ROS image publishers.

Renders high-quality 2D perception overlays:
  1. Detection overlay: RGB + raw detection masks + bounding boxes + class labels + confidence
  2. Tracking overlay: RGB + tracked object masks + track IDs + state badges + relation arrows
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Tuple, Sequence
import cv2
import numpy as np

# Harmonious palettes matching demo
OBJECT_PALETTE = [
    (82, 177, 238),   # Cyan
    (89, 208, 164),   # Mint
    (171, 133, 235),  # Violet
    (239, 177, 87),   # Amber
    (82, 193, 207),   # Teal
    (229, 116, 150),  # Coral
    (164, 181, 112),  # Olive
    (133, 151, 226),  # Lavender
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

        # Mask
        try:
            mask = obs.get_mask() if hasattr(obs, "get_mask") else getattr(obs, "mask", None)
        except Exception:
            mask = None

        if mask is not None:
            canvas = draw_mask(canvas, mask, color, alpha=0.25)

        # Bbox
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
    """Render 2D tracks and relation overlay: RGB + track badges + state + relation arrows."""
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

        # Latest observation bbox
        obs = track.recent_observations[-1] if track.recent_observations else None
        if obs is None:
            continue

        bbox_xyxy = getattr(obs, "bbox_xyxy", None)
        if bbox_xyxy is None:
            continue

        box = clamp_bbox(bbox_xyxy, w, h)
        boxes[node_id] = box
        col = track_color(node_id)

        # Draw mask if present
        try:
            mask = obs.get_mask() if hasattr(obs, "get_mask") else getattr(obs, "mask", None)
        except Exception:
            mask = None

        if mask is not None:
            canvas = draw_mask(canvas, mask, col, alpha=0.20)

    # Filter relations to at most 1 primary relation per object-pair to prevent clutter
    active_edges = [e for e in graph.edges.values() if e.is_active]
    priority = {
        "ON": 10, "UNDER": 10, "INSIDE": 10, "CONTAINING": 10,
        "ABOVE": 7, "BELOW": 7, "LEFT_OF": 6, "RIGHT_OF": 6,
        "IN_FRONT_OF": 5, "BEHIND": 5,
        "NEAR": 2, "FAR": 1,
    }
    grouped: dict[frozenset[str], list[Any]] = {}
    for edge in active_edges:
        key = frozenset((str(edge.subject_id), str(edge.object_id)))
        grouped.setdefault(key, []).append(edge)

    display_edges = [
        max(p_edges, key=lambda e: (
            priority.get(str(e.predicate), 0),
            float(getattr(e.latest_evidence, "confidence", 1.0) if hasattr(e, "latest_evidence") and e.latest_evidence else 1.0)
        ))
        for p_edges in grouped.values()
    ]

    for edge in display_edges:
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

    # Draw track boxes and badges on top
    for node_id, box in boxes.items():
        node = node_map[node_id]
        track = node.track
        col = track_color(node_id)
        bx1, by1, bx2, by2 = box

        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), col, 2, cv2.LINE_AA)
        short_id = node_id[-4:] if len(node_id) >= 4 else node_id
        label = f"{track.class_name.upper()} #{short_id} [{node.state.value}]"
        draw_label(canvas, label, (bx1, max(18, by1 - 4)), col)

    return canvas
