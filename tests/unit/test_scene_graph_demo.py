import importlib.util
from pathlib import Path
import sys

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "scene_graph_demo", Path(__file__).parents[2] / "scripts" / "scene_graph_demo.py"
)
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


class FakeTrack:
    object_id = "cup_001"
    class_name = "cup"
    detection_confidence = 0.9
    observation_count = 4
    track_observation_ratio = 1.0
    recent_observations = []


class FakeNode:
    object_id = "cup_001"
    class_name = "cup"
    state = type("State", (), {"value": "stable"})()
    track = FakeTrack()


class EmptyGraph:
    def get_active_nodes(self):
        return []

    def get_active_edges(self):
        return []


def test_demo_node_table_is_empty_safe():
    assert demo.node_rows(EmptyGraph()) == []
    frame = demo.as_dataframe([], ["track_id", "class_name"])
    assert list(frame.columns) == ["track_id", "class_name"]


def test_rgb_overlay_handles_empty_graph():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    rendered = demo.render_rgb_scene(image, EmptyGraph())
    assert np.array_equal(rendered, image)


def test_track_colours_are_deterministic():
    assert demo.track_color("cup_001") == demo.track_color("cup_001")
