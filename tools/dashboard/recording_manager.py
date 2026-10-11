"""RealSense D455 Recording Manager for Dashboard.

Handles subprocess management, graceful shutdown (SIGINT flush),
live preview frame buffering, and recording status tracking.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RecordingManager:
    """Manages the background D455 capture process with graceful flush."""

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent.parent
        self.process: Optional[subprocess.Popen] = None
        self.active_dataset_name: Optional[str] = None
        self.active_save_dir: Optional[Path] = None
        self.start_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self.status: str = "IDLE"  # IDLE, STARTING, RECORDING, STOPPING, ERROR
        self.error_message: Optional[str] = None
        self.preview_file: Path = self.repo_root / "data" / ".dashboard_preview.jpg"
        self._log_lines: list[str] = []
        self._monitor_thread: Optional[threading.Thread] = None

    def check_realsense_available(self) -> Tuple[bool, str]:
        """Check if pyrealsense2 is installed and a physical D455 camera is detected."""
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                return False, "No RealSense device connected. Please connect your Intel RealSense D455 via USB 3.0."
            dev_name = devices[0].get_info(rs.camera_info.name)
            return True, f"Found {dev_name} (USB {devices[0].get_info(rs.camera_info.usb_type_descriptor) if devices[0].supports(rs.camera_info.usb_type_descriptor) else 'unknown'})"
        except ImportError:
            return False, "pyrealsense2 library is not installed in the active Python environment."
        except Exception as e:
            return False, f"RealSense hardware check error: {e}"

    def start_recording(
        self,
        dataset_name: Optional[str] = None,
        output_root: Optional[Path] = None,
        enable_imu: bool = False,
    ) -> bool:
        """Launch tools/recording/record_d455.py as a managed background subprocess."""
        if self.is_recording():
            self.error_message = "A recording is already active."
            return False

        out_root = (output_root or (self.repo_root / "data" / "raw")).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.active_dataset_name = dataset_name.strip() if dataset_name and dataset_name.strip() else f"d455_rgbd_{timestamp_str}"
        self.active_save_dir = out_root / self.active_dataset_name
        self.start_time = time.time()
        self.stop_time = None
        self.status = "STARTING"
        self.error_message = None
        self._log_lines = []

        # Ensure preview file parent directory exists
        self.preview_file.parent.mkdir(parents=True, exist_ok=True)
        if self.preview_file.exists():
            try:
                self.preview_file.unlink()
            except Exception:
                pass

        recorder_script = self.repo_root / "tools" / "recording" / "record_d455.py"
        if not recorder_script.exists():
            self.status = "ERROR"
            self.error_message = f"Recorder script not found at {recorder_script}"
            return False

        # Build command
        cmd = [
            sys.executable,
            str(recorder_script),
            "--name", self.active_dataset_name,
            "--output-dir", str(out_root),
            "--no-preview",
            "--preview-file", str(self.preview_file),
        ]
        if enable_imu:
            cmd.append("--enable-imu")

        try:
            # Launch with pipe for stdout to capture runtime status and errors
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.status = "RECORDING"

            # Start background reader thread for logs
            self._monitor_thread = threading.Thread(target=self._read_logs, daemon=True)
            self._monitor_thread.start()
            return True
        except Exception as e:
            self.status = "ERROR"
            self.error_message = f"Failed to launch recorder process: {e}"
            return False

    def _read_logs(self):
        """Continuously capture logs from subprocess."""
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            clean_line = line.strip()
            if clean_line:
                self._log_lines.append(clean_line)
                if len(self._log_lines) > 200:
                    self._log_lines.pop(0)
                if "[Error]" in clean_line or "Failed to start RealSense camera" in clean_line:
                    self.error_message = clean_line

        # When process exits
        ret = self.process.poll()
        if ret is not None and ret != 0:
            if not self.error_message:
                self.error_message = f"Recorder exited unexpectedly with returncode {ret}"
            self.status = "ERROR"
        elif self.status == "STOPPING":
            self.status = "IDLE"

    def stop_recording(self, timeout_s: float = 12.0) -> bool:
        """Send graceful interrupt (SIGINT/Ctrl+C) allowing frame writer to flush to disk."""
        if not self.is_recording():
            return True

        self.status = "STOPPING"
        self.stop_time = time.time()

        if self.process:
            try:
                # Send SIGINT to allow recorder finally block to flush queue
                if sys.platform == "win32":
                    # On Windows, try terminate or CTRL_BREAK if created in separate group
                    import signal
                    try:
                        self.process.send_signal(signal.CTRL_C_EVENT)
                    except Exception:
                        self.process.terminate()
                else:
                    self.process.send_signal(signal.SIGINT)

                # Wait gracefully for disk queue join and metadata save
                try:
                    self.process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    logger.warning("Recorder graceful flush timed out; terminating.")
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                logger.error("Error stopping recorder process: %s", e)
                try:
                    self.process.kill()
                except Exception:
                    pass

        self.status = "IDLE"
        return True

    def is_recording(self) -> bool:
        if self.process and self.process.poll() is None:
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Return full recording telemetry."""
        is_rec = self.is_recording()
        if not is_rec and self.status == "RECORDING":
            self.status = "IDLE" if not self.error_message else "ERROR"

        elapsed = 0.0
        if self.start_time:
            end = self.stop_time if self.stop_time else time.time()
            elapsed = max(0.0, end - self.start_time)

        # Count frames from rgb.txt if dataset directory exists
        frame_count = 0
        if self.active_save_dir and (self.active_save_dir / "rgb.txt").exists():
            try:
                with open(self.active_save_dir / "rgb.txt", "r", encoding="utf-8") as f:
                    lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
                    frame_count = len(lines)
            except Exception:
                pass

        return {
            "status": self.status,
            "is_recording": is_rec,
            "dataset_name": self.active_dataset_name,
            "save_location": str(self.active_save_dir) if self.active_save_dir else None,
            "elapsed_seconds": round(elapsed, 1),
            "frame_count": frame_count,
            "error_message": self.error_message,
            "recent_logs": self._log_lines[-8:],
        }

    def get_preview_image_bytes(self) -> Optional[bytes]:
        """Fetch the latest exported preview image frame bytes."""
        if self.preview_file.exists():
            try:
                with open(self.preview_file, "rb") as f:
                    return f.read()
            except Exception:
                return None
        return None
