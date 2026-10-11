"""Unit tests for Dashboard modules (recording_manager, dataset_browser, pipeline_runner, simulator)."""

import json
from pathlib import Path
import pytest

from tools.dashboard.recording_manager import RecordingManager
from tools.dashboard.dataset_browser import DatasetBrowser
from tools.dashboard.pipeline_runner import PipelineRunner
from tools.dashboard.simulator import VehicleSimulator


def test_dataset_browser_discovery():
    browser = DatasetBrowser()
    raw = browser.list_raw_datasets()
    assert len(raw) >= 1
    assert "desk_run_20261009_161552" in raw

    runs = browser.list_mapping_runs()
    assert len(runs) >= 1
    assert "semantic_mapping_desk_run_161552" in runs

    summary = browser.get_dataset_summary("desk_run_20261009_161552")
    assert summary["exists"] is True
    assert summary["total_files"] > 10

    files = browser.get_recursive_file_list(browser.raw_root / "desk_run_20261009_161552")
    assert len(files) > 10


def test_dataset_browser_file_preview():
    browser = DatasetBrowser()
    run_dir = browser.outputs_root / "semantic_mapping_desk_run_161552"
    
    # JSON preview
    prev_json = browser.read_file_preview(run_dir / "occupancy_metadata.json")
    assert prev_json["type"] == "json"
    assert "map_frame" in prev_json["data"]

    # PGM preview
    prev_pgm = browser.read_file_preview(run_dir / "map.pgm")
    assert prev_pgm["type"] == "numpy_image"
    assert prev_pgm["array"].shape[2] == 3


def test_pipeline_runner_status():
    runner = PipelineRunner()
    status = runner.get_status()
    assert status["status"] == "IDLE"
    assert status["is_running"] is False

    orb = runner.check_orbslam3_status()
    assert "is_ready" in orb


def test_vehicle_simulator_mode1():
    run_dir = Path("outputs/semantic_mapping_desk_run_161552")
    sim = VehicleSimulator(run_dir)
    entities = sim.get_confirmed_entity_names()
    assert len(entities) > 0

    # Simulate goal navigation
    res = sim.simulate_navigation_goal(
        "go to desk_003 and stop 0.5 metres away",
        start_pose=(0.15, -0.12, 0.0),
    )
    assert res.planned_path is not None
    assert len(res.states) > 5
    assert res.goal_reached is True

    # Check Plotly render
    fig = sim.render_topdown_plotly(sim_result=res)
    assert len(fig.data) >= 3


def test_vehicle_simulator_mode2_cmd_vel():
    run_dir = Path("outputs/semantic_mapping_desk_run_161552")
    sim = VehicleSimulator(run_dir)

    # In-place rotation at safe pose
    cmd_seq = [
        {"linear_x": 0.00, "angular_z": 0.50, "duration_s": 0.5},
    ]
    res_safe = sim.simulate_velocity_commands(cmd_seq, start_pose=(0.15, -0.12, 0.0))
    assert res_safe.success is True
    assert len(res_safe.states) > 0
    assert res_safe.has_collision is False

    # Driving into obstacle must be detected and flagged
    cmd_collide = [
        {"linear_x": 0.30, "angular_z": 0.00, "duration_s": 1.0},
    ]
    res_col = sim.simulate_velocity_commands(cmd_collide, start_pose=(0.15, -0.12, 0.0))
    assert res_col.has_collision is True
    assert res_col.collision_time_s is not None
