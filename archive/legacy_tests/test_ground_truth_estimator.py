"""Tests for archived GroundTruthEstimator."""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "archive/tum/replay") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "archive/tum/replay"))
if str(REPO_ROOT / "archive/tum/loaders") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "archive/tum/loaders"))

from ground_truth import GroundTruthEstimator
from scene_graph.data.pose_estimate import TrackingState
from scene_graph.data.sensor_frame import SensorFrame
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.geometry.camera import CameraIntrinsics


def create_dummy_sensor_frame(timestamp=10.0, sequence_number=0):
    return SensorFrame(
        session_id="test_session",
        sequence_number=sequence_number,
        timestamp=Timestamp(value=timestamp, domain=TimestampDomain.SYSTEM_TIME),
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        camera_intrinsics=CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480),
    )


def test_gt_trajectory_association(tmp_path):
    gt_file = tmp_path / "groundtruth.txt"
    gt_file.write_text(
        "# TUM ground truth\n"
        "10.0000 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n"
        "10.0333 1.1 2.1 3.1 0.0 0.0 0.0 1.0\n"
        "10.0666 1.2 2.2 3.2 0.0 0.0 0.0 1.0\n"
    )

    estimator = GroundTruthEstimator(trajectory_file=gt_file, max_dt=0.02)
    sf = create_dummy_sensor_frame(timestamp=10.035)
    pose, state = estimator.estimate(sf)

    assert pose.valid is True
    assert np.allclose(pose.translation, [1.1, 2.1, 3.1])
    assert pose.source == "ground_truth"
    assert state.tracking_state == TrackingState.TRACKING


def test_gt_out_of_sync_rejection(tmp_path):
    gt_file = tmp_path / "groundtruth.txt"
    gt_file.write_text("10.0 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n")

    estimator = GroundTruthEstimator(trajectory_file=gt_file, max_dt=0.02)
    sf = create_dummy_sensor_frame(timestamp=10.5)
    pose, state = estimator.estimate(sf)

    assert pose.valid is False
    assert state.tracking_state == TrackingState.LOST
