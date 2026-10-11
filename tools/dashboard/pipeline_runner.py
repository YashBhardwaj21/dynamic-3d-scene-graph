"""Mapping Pipeline Subprocess Runner for Dashboard.

Executes tools/mapping/run_mapping_pipeline.py with validated parameters,
streams real-time logs, verifies ORB-SLAM3 availability, and discovers artifacts upon completion.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Manages offline mapping jobs with live log capture and artifact indexing."""

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent.parent
        self.process: Optional[subprocess.Popen] = None
        self.status: str = "IDLE"  # IDLE, RUNNING, COMPLETED, FAILED
        self.active_dataset: Optional[str] = None
        self.active_run_id: Optional[str] = None
        self.active_out_dir: Optional[Path] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.return_code: Optional[int] = None
        self._logs: List[str] = []
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None

    def check_orbslam3_status(self) -> Dict[str, Any]:
        """Check ORB-SLAM3 binary and vocabulary existence in known locations."""
        candidates_bin = [
            Path("/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum"),
            self.repo_root.parent / "ORB_SLAM3" / "Examples" / "RGB-D" / "rgbd_tum",
            Path("C:/ORB_SLAM3/Examples/RGB-D/rgbd_tum.exe"),
        ]
        candidates_voc = [
            Path("/home/saturn/ORB_SLAM3/Vocabulary/ORBvoc.txt"),
            self.repo_root.parent / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt",
            Path("C:/ORB_SLAM3/Vocabulary/ORBvoc.txt"),
        ]

        bin_found = None
        for b in candidates_bin:
            if b.exists():
                bin_found = str(b)
                break

        voc_found = None
        for v in candidates_voc:
            if v.exists():
                voc_found = str(v)
                break

        is_ready = (bin_found is not None) and (voc_found is not None)
        return {
            "is_ready": is_ready,
            "binary_path": bin_found,
            "vocabulary_path": voc_found,
            "message": "ORB-SLAM3 is configured and available for headless execution." if is_ready else "ORB-SLAM3 binary or vocabulary not found. If running on Windows without WSL, provide pre-computed CameraTrajectory.txt.",
        }

    def start_pipeline(
        self,
        dataset_name: str,
        sample_step: int = 10,
        run_id: Optional[str] = None,
        trajectory_file: Optional[str] = None,
        robot_radius: float = 0.18,
        safety_margin: float = 0.05,
    ) -> bool:
        """Launch run_mapping_pipeline.py asynchronously with unique output directory."""
        if self.is_running():
            return False

        dataset_path = (self.repo_root / "data" / "raw" / dataset_name).resolve()
        if not dataset_path.exists():
            with self._lock:
                self.status = "FAILED"
                self._logs.append(f"[Error] Dataset directory does not exist: {dataset_path}")
            return False

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.active_dataset = dataset_name
        self.active_run_id = run_id.strip() if run_id and run_id.strip() else f"run_{dataset_name}_{timestamp_str}"
        self.active_out_dir = (self.repo_root / "outputs" / self.active_run_id).resolve()
        self.start_time = time.time()
        self.end_time = None
        self.return_code = None

        with self._lock:
            self.status = "RUNNING"
            self._logs = [
                f"[PipelineRunner] Initiating offline mapping for dataset: {dataset_name}",
                f"[PipelineRunner] Output Run ID: {self.active_run_id}",
                f"[PipelineRunner] Sampling step: {sample_step} | Robot radius: {robot_radius}m",
            ]

        pipeline_script = self.repo_root / "tools" / "mapping" / "run_mapping_pipeline.py"
        if not pipeline_script.exists():
            with self._lock:
                self.status = "FAILED"
                self._logs.append(f"[Error] Pipeline script not found: {pipeline_script}")
            return False

        cmd = [
            sys.executable,
            str(pipeline_script),
            str(dataset_path),
            "--output-root", str(self.repo_root / "outputs"),
            "--run-id", self.active_run_id,
            "--sample-step", str(sample_step),
            "--robot-radius", str(robot_radius),
            "--safety-margin", str(safety_margin),
        ]
        if trajectory_file and Path(trajectory_file).exists():
            cmd.extend(["--trajectory-file", str(trajectory_file)])

        try:
            # Set unbuffered python output
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            self._monitor_thread = threading.Thread(target=self._read_pipeline_output, daemon=True)
            self._monitor_thread.start()
            return True
        except Exception as e:
            with self._lock:
                self.status = "FAILED"
                self._logs.append(f"[Error] Failed to spawn mapping pipeline: {e}")
            return False

    def _read_pipeline_output(self):
        """Continuously collect output lines."""
        if not self.process or not self.process.stdout:
            return

        for line in self.process.stdout:
            clean = line.rstrip()
            if clean:
                with self._lock:
                    self._logs.append(clean)
                    if len(self._logs) > 500:
                        self._logs.pop(0)

        ret = self.process.wait()
        self.end_time = time.time()
        self.return_code = ret
        with self._lock:
            if ret == 0:
                self.status = "COMPLETED"
                self._logs.append(f"[PipelineRunner] Mapping completed successfully in {self.end_time - self.start_time:.1f}s.")
            else:
                self.status = "FAILED"
                self._logs.append(f"[PipelineRunner] Mapping failed with exit code {ret}.")

    def is_running(self) -> bool:
        if self.process and self.process.poll() is None:
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Return state, elapsed time, and latest logs."""
        with self._lock:
            logs_copy = list(self._logs)

        elapsed = 0.0
        if self.start_time:
            end = self.end_time if self.end_time else time.time()
            elapsed = max(0.0, end - self.start_time)

        return {
            "status": self.status,
            "is_running": self.is_running(),
            "dataset_name": self.active_dataset,
            "run_id": self.active_run_id,
            "output_dir": str(self.active_out_dir) if self.active_out_dir else None,
            "elapsed_seconds": round(elapsed, 1),
            "return_code": self.return_code,
            "recent_logs": logs_copy[-30:],
            "all_logs_count": len(logs_copy),
        }

    def get_discovered_artifacts(self) -> Dict[str, Any]:
        """Index all key deliverables in the active output directory."""
        if not self.active_out_dir or not self.active_out_dir.exists():
            return {}

        out = self.active_out_dir
        artifacts: Dict[str, Any] = {}

        # 1. Nav2 Occupancy Grid & Costmap
        if (out / "map.pgm").exists():
            artifacts["nav2_map"] = str(out / "map.pgm")
        if (out / "navigation_costmap.pgm").exists():
            artifacts["navigation_costmap"] = str(out / "navigation_costmap.pgm")
        if (out / "occupancy_metadata.json").exists():
            artifacts["occupancy_metadata"] = str(out / "occupancy_metadata.json")

        # 2. Semantic Map & Aliases
        if (out / "semantic_map.json").exists():
            artifacts["semantic_map"] = str(out / "semantic_map.json")
        if (out / "entity_aliases.json").exists():
            artifacts["entity_aliases"] = str(out / "entity_aliases.json")
        if (out / "floor_plane.json").exists():
            artifacts["floor_plane"] = str(out / "floor_plane.json")

        # 3. Visualizations
        vis_dir = out / "visualizations"
        for v in ["planned_goal_route.png", "semantic_floor_plan.png", "mapping_visualization.png"]:
            if (vis_dir / v).exists():
                artifacts[f"vis_{v.replace('.png', '')}"] = str(vis_dir / v)
            elif (out / v).exists():
                artifacts[f"vis_{v.replace('.png', '')}"] = str(out / v)

        # 4. Reports & Trajectories
        if (out / "audit_report.md").exists():
            artifacts["audit_report"] = str(out / "audit_report.md")
        if (out / "trajectory" / "CameraTrajectory.txt").exists():
            artifacts["camera_trajectory"] = str(out / "trajectory" / "CameraTrajectory.txt")
        if (out / "world_objects.ply").exists():
            artifacts["world_point_cloud"] = str(out / "world_objects.ply")

        return artifacts
