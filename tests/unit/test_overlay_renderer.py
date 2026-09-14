"""Unit tests for 2D perception overlay renderer."""

import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

WS_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WS_ROOT / "ros2_ws" / "src" / "scene_graph_ros"))

from scene_graph_ros.overlay_renderer import (
    track_color,
    relation_color,
    clamp_bbox,
    bbox_center,
    boundary_point,
    draw_mask,
    draw_label,
    render_detections_overlay,
    render_tracks_overlay,
)


def test_track_color_deterministic():
    """Verify that the same track ID consistently maps to the exact same color."""
    c1 = track_color("track_cup_1")
    c2 = track_color("track_cup_1")
    c3 = track_color("track_cup_2")

    assert c1 == c2
    assert isinstance(c1, tuple) and len(c1) == 3
    # Different track ID should likely produce a different color or palette entry
    assert isinstance(c3, tuple) and len(c3) == 3


def test_relation_color():
    """Verify predefined relation colors and fallback."""
    assert relation_color("ON") == (54, 207, 169)
    assert relation_color("LEFT_OF") == (248, 188, 90)
    assert relation_color("UNKNOWN_PRED") == (215, 220, 225)


def test_clamp_bbox():
    """Verify bbox clamping within image boundaries."""
    bbox = [-10.0, -5.0, 700.0, 500.0]
    bx1, by1, bx2, by2 = clamp_bbox(bbox, width=640, height=480)
    assert bx1 == 0
    assert by1 == 0
    assert bx2 == 639
    assert by2 == 479


def test_bbox_center_and_boundary_point():
    """Verify bbox center and boundary intersection point computation."""
    box = (100, 100, 200, 200)
    cx, cy = bbox_center(box)
    assert cx == 150.0
    assert cy == 150.0

    # Target is strictly to the right
    bp_right = boundary_point(box, (300.0, 150.0))
    assert bp_right[0] == 200
    assert bp_right[1] == 150

    # Target is strictly above
    bp_above = boundary_point(box, (150.0, 50.0))
    assert bp_above[0] == 150
    assert bp_above[1] == 100


def test_draw_mask():
    """Verify alpha mask blending."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True

    blended = draw_mask(img, mask, (200, 100, 50), alpha=0.5)
    assert blended.shape == (50, 50, 3)
    # Masked region should now have non-zero color
    assert np.all(blended[15, 15] > 0)
    # Non-masked region remains 0
    assert np.all(blended[0, 0] == 0)


def test_render_detections_overlay():
    """Verify rendering of detection boxes and masks."""
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 20:40] = True

    mock_obs = SimpleNamespace(
        bbox_xyxy=np.array([20.0, 20.0, 40.0, 40.0]),
        class_name="cup",
        confidence=0.92,
        get_mask=lambda: mask,
    )

    out = render_detections_overlay(rgb, [mock_obs])
    assert out.shape == (100, 100, 3)
    assert out.dtype == np.uint8
    # Should contain rendered pixels
    assert np.any(out > 0)


def test_render_tracks_overlay():
    """Verify rendering of track boxes, state badges, and relations."""
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_obs1 = SimpleNamespace(
        bbox_xyxy=np.array([10.0, 10.0, 30.0, 30.0]),
        get_mask=lambda: None,
    )
    mock_obs2 = SimpleNamespace(
        bbox_xyxy=np.array([60.0, 60.0, 80.0, 80.0]),
        get_mask=lambda: None,
    )

    mock_track1 = SimpleNamespace(
        object_id="track_0001",
        class_name="cup",
        recent_observations=[mock_obs1],
    )
    mock_track2 = SimpleNamespace(
        object_id="track_0002",
        class_name="table",
        recent_observations=[mock_obs2],
    )

    mock_node1 = SimpleNamespace(
        object_id="track_0001",
        class_name="cup",
        track=mock_track1,
        state=SimpleNamespace(value="CONFIRMED"),
    )
    mock_node2 = SimpleNamespace(
        object_id="track_0002",
        class_name="table",
        track=mock_track2,
        state=SimpleNamespace(value="CONFIRMED"),
    )

    mock_edge = SimpleNamespace(
        subject_id="track_0001",
        object_id="track_0002",
        predicate="ON",
        is_active=True,
    )

    mock_graph = SimpleNamespace(
        get_active_nodes=lambda: [mock_node1, mock_node2],
        edges={("track_0001", "ON", "track_0002"): mock_edge},
    )

    out = render_tracks_overlay(rgb, mock_graph)
    assert out.shape == (100, 100, 3)
    assert out.dtype == np.uint8
    assert np.any(out > 0)
