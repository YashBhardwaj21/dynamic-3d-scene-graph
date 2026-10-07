"""Ground truth trajectory pose estimator for evaluation and replay.

Ingests reference poses (e.g. from TUM groundtruth.txt) and associates them
with incoming SensorFrames based on timestamp matching.
Used strictly for evaluation benchmarks, never as a production SLAM source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from scene_graph.data.pose_estimate import EstimatorState, PoseEstimate, TrackingState
from scene_graph.data.sensor_frame import SensorFrame
from scene_graph.data.tum_loader import PoseEntry, parse_file_list
from scene_graph.estimation.base import BasePoseEstimator
from scene_graph.geometry.transforms import pose_to_transform


class GroundTruthEstimator(BasePoseEstimator):
    """Associates SensorFrames with offline ground-truth trajectory poses."""

    def __init__(
        self,
        trajectory_file: str | Path | None = None,
        pose_entries: Sequence[PoseEntry] | None = None,
        max_dt: float = 0.05,
        world_frame: str = "world",
    ):
        self.max_dt = max_dt
        self.world_frame = world_frame
        self.pose_entries: list[PoseEntry] = []

        if pose_entries is not None:
            self.pose_entries = list(pose_entries)
        elif trajectory_file is not None:
            self.load_trajectory(trajectory_file)

        self._timestamps = np.array([e.timestamp for e in self.pose_entries], dtype=np.float64)

    def load_trajectory(self, trajectory_file: str | Path) -> None:
        """Parse TUM groundtruth.txt file (timestamp tx ty tz qx qy qz qw)."""
        rows = parse_file_list(trajectory_file)
        entries: list[PoseEntry] = []
        for row in rows:
            if len(row) < 8:
                continue
            try:
                entries.append(
                    PoseEntry(
                        timestamp=float(row[0]),
                        tx=float(row[1]),
                        ty=float(row[2]),
                        tz=float(row[3]),
                        qx=float(row[4]),
                        qy=float(row[5]),
                        qz=float(row[6]),
                        qw=float(row[7]),
                    )
                )
            except ValueError:
                continue

        self.pose_entries = entries
        self._timestamps = np.array([e.timestamp for e in self.pose_entries], dtype=np.float64)

    def estimate(self, sensor_frame: SensorFrame) -> tuple[PoseEstimate, EstimatorState]:
        ts = sensor_frame.timestamp.value

        if len(self._timestamps) == 0:
            return (
                PoseEstimate.invalid(
                    timestamp=ts,
                    source="ground_truth_empty",
                    transform_source="no_gt_trajectory",
                    frame_id=self.world_frame,
                ),
                EstimatorState.lost(backend_name="ground_truth", odometry_lost=True),
            )

        # Find closest timestamp
        idx = int(np.argmin(np.abs(self._timestamps - ts)))
        best_dt = abs(self._timestamps[idx] - ts)
        gt_entry = self.pose_entries[idx]

        if best_dt > self.max_dt:
            return (
                PoseEstimate.invalid(
                    timestamp=ts,
                    source="ground_truth_out_of_sync",
                    target_timestamp=ts,
                    transform_source=f"gt_skew_{best_dt*1000.0:.1f}ms",
                    frame_id=self.world_frame,
                ),
                EstimatorState(
                    backend_name="ground_truth",
                    tracking_state=TrackingState.LOST,
                    odometry_lost=True,
                    telemetry_timestamp=gt_entry.timestamp,
                    telemetry_age=best_dt,
                ),
            )

        T = pose_to_transform(
            gt_entry.tx, gt_entry.ty, gt_entry.tz,
            gt_entry.qx, gt_entry.qy, gt_entry.qz, gt_entry.qw
        )

        pose = PoseEstimate(
            world_T_camera=T,
            timestamp=ts,
            valid=True,
            age=best_dt,
            source="ground_truth",
            covariance=None,
            frame_id=self.world_frame,
            pose_timestamp=gt_entry.timestamp,
            target_timestamp=ts,
            lookup_mode="nearest_offline",
            transform_source="tum_groundtruth",
        )

        state = EstimatorState(
            backend_name="ground_truth",
            tracking_state=TrackingState.TRACKING,
            matches=0,
            inliers=0,
            inlier_ratio=1.0,
            features=0,
            telemetry_timestamp=gt_entry.timestamp,
            telemetry_age=best_dt,
            covariance_available=False,
        )

        return pose, state

    def reset(self) -> None:
        pass
