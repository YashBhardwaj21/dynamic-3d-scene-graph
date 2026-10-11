"""Unit tests for Stage 3 Trajectory Ingestion, Provenance, 1-to-1 Matching, and Quality Gates."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from scene_graph.pipeline.trajectory_manager import (
    load_and_validate_trajectory,
    load_tum_trajectory,
    compute_file_sha256,
    TrajectoryValidationError,
    TrajectoryResult,
)


def create_tum_file(path: Path, rows: list) -> Path:
    """Helper to write TUM trajectory lines."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r[0]:.4f} {r[1]:.4f} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} {r[5]:.4f} {r[6]:.4f} {r[7]:.4f}\n")
    return path


def test_trajectory_valid_ingest():
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_p = Path(tmpdir) / "CameraTrajectory.txt"
        # 10 frames from t=0.0 to 0.9s
        rows = [
            (i * 0.1, 0.1 * i, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            for i in range(10)
        ]
        create_tum_file(traj_p, rows)
        rgb_timestamps = [i * 0.1 for i in range(10)]

        res = load_and_validate_trajectory(traj_p, rgb_timestamps, max_dt=0.05, min_coverage_pct=50.0)
        assert res.is_valid is True
        assert res.status == "VALID"
        assert res.unique_matched_frames == 10
        assert res.coverage_percentage == 100.0
        assert len(res.file_sha256) == 64


def test_trajectory_mismatched_recording_rejection():
    """Trajectories from a different recording (different timestamps) must be rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_p = Path(tmpdir) / "CameraTrajectory.txt"
        # Timestamps around t=1000s
        rows = [
            (1000.0 + i * 0.1, 0.1 * i, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            for i in range(10)
        ]
        create_tum_file(traj_p, rows)
        # Recording timestamps around t=0.0s
        rgb_timestamps = [i * 0.1 for i in range(10)]

        res = load_and_validate_trajectory(traj_p, rgb_timestamps)
        assert res.is_valid is False
        assert res.status == "REJECTED_MISMATCH"
        assert "Cannot reuse trajectory from a different dataset" in res.rejection_reason


def test_trajectory_truncated_low_coverage_rejection():
    """Trajectories that cover too few frames must fail the coverage gate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_p = Path(tmpdir) / "CameraTrajectory.txt"
        # Trajectory only covers 2 frames out of 20
        rows = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ]
        create_tum_file(traj_p, rows)
        rgb_timestamps = [i * 0.1 for i in range(20)]

        res = load_and_validate_trajectory(traj_p, rgb_timestamps, min_coverage_pct=50.0)
        assert res.is_valid is False
        assert res.status == "REJECTED_LOW_COVERAGE"


def test_trajectory_non_monotonic_duplicate_timestamps():
    """Duplicate or backwards timestamps must fail immediately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_p = Path(tmpdir) / "CameraTrajectory.txt"
        rows = [
            (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),  # Duplicate timestamp 0.1
        ]
        create_tum_file(traj_p, rows)
        rgb_timestamps = [0.1, 0.2]

        res = load_and_validate_trajectory(traj_p, rgb_timestamps)
        assert res.is_valid is False
        assert res.status == "REJECTED_INVALID_VALUES"
        assert "not strictly monotonic" in res.rejection_reason


def test_trajectory_non_finite_values():
    """NaN or Inf values in poses must be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_p = Path(tmpdir) / "CameraTrajectory.txt"
        with open(traj_p, "w", encoding="utf-8") as f:
            f.write("0.1 0.0 nan 0.0 0.0 0.0 0.0 1.0\n")
        rgb_timestamps = [0.1]

        res = load_and_validate_trajectory(traj_p, rgb_timestamps)
        assert res.is_valid is False
        assert res.status == "REJECTED_INVALID_VALUES"


def test_trajectory_implausible_discontinuity():
    """Implausible translation jumps must be rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_p = Path(tmpdir) / "CameraTrajectory.txt"
        rows = [
            (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.2, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),  # 10m jump in 0.1s!
        ]
        create_tum_file(traj_p, rows)
        rgb_timestamps = [0.1, 0.2]

        res = load_and_validate_trajectory(traj_p, rgb_timestamps, max_translation_jump_m=3.0)
        assert res.is_valid is False
        assert res.status == "REJECTED_DISCONTINUITY"
        assert "Implausible translation jump" in res.rejection_reason


def test_trajectory_one_to_one_matching_no_reuse():
    """One trajectory pose must not be counted multiple times for different frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_p = Path(tmpdir) / "CameraTrajectory.txt"
        # 2 poses at t=0.10s and 0.50s
        rows = [
            (0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ]
        create_tum_file(traj_p, rows)
        # 3 RGB frames close to 0.1s (0.09s, 0.10s, 0.11s) and 1 frame at 0.50s
        rgb_timestamps = [0.09, 0.10, 0.11, 0.50]

        res = load_and_validate_trajectory(traj_p, rgb_timestamps, max_dt=0.05, min_coverage_pct=10.0)
        assert res.is_valid is True
        # Pose at 0.10s matches at most ONE of the 3 candidate frames, and pose at 0.50s matches 1 frame
        assert res.unique_matched_frames == 2
        assert res.unmatched_frames_count == 2
