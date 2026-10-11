"""Dataset and Output Artifact Browser for Dashboard.

Provides automatic discovery, recursive inspection, paginated tables,
and specialized file viewers (images, PGM maps, video, text, JSON, YAML, 3D point clouds).
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


class DatasetBrowser:
    """Discovers, indexes, and renders files across data/raw/ and outputs/."""

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent.parent
        self.raw_root = self.repo_root / "data" / "raw"
        self.outputs_root = self.repo_root / "outputs"

    def list_raw_datasets(self) -> List[str]:
        """List all discovered valid raw recording directories."""
        if not self.raw_root.exists():
            return []
        datasets = []
        for p in sorted(self.raw_root.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                datasets.append(p.name)
        return datasets

    def list_mapping_runs(self) -> List[str]:
        """List all discovered mapping run output directories."""
        if not self.outputs_root.exists():
            return []
        runs = []
        for p in sorted(self.outputs_root.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                runs.append(p.name)
        return runs

    def get_dataset_summary(self, dataset_name: str, is_output: bool = False) -> Dict[str, Any]:
        """Summarize directory content, metadata, and key metrics."""
        base_dir = (self.outputs_root if is_output else self.raw_root) / dataset_name
        if not base_dir.exists():
            return {"exists": False}

        total_files = 0
        for _, _, files in os.walk(base_dir):
            total_files += len(files)

        # Fast approximate size from top-level files
        total_size_bytes = 0
        for entry in base_dir.iterdir():
            if entry.is_file():
                try:
                    total_size_bytes += entry.stat().st_size
                except Exception:
                    pass

        summary: Dict[str, Any] = {
            "exists": True,
            "path": str(base_dir),
            "total_files": total_files,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "is_output": is_output,
        }

        # Check for metadata files
        meta_file = base_dir / "dataset_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    summary["meta"] = json.load(f)
            except Exception:
                pass

        val_file = base_dir / "recording_validation.json"
        if val_file.exists():
            try:
                with open(val_file, "r", encoding="utf-8") as f:
                    summary["validation"] = json.load(f)
            except Exception:
                pass

        audit_file = base_dir / "audit_summary.json"
        if audit_file.exists():
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    summary["audit_summary"] = json.load(f)
            except Exception:
                pass

        return summary

    def get_recursive_file_list(
        self,
        target_dir: Path,
        filter_ext: Optional[str] = None,
        max_files: int = 300,
    ) -> List[Dict[str, Any]]:
        """Recursively scan directory and return structured metadata per file."""
        if not target_dir.exists():
            return []

        entries = []
        for root, _, files in os.walk(target_dir):
            for f in sorted(files):
                if len(entries) >= max_files:
                    break
                full_p = Path(root) / f
                rel_p = full_p.relative_to(target_dir)
                ext = full_p.suffix.lower()

                if filter_ext and ext != filter_ext.lower():
                    continue

                try:
                    sz = full_p.stat().st_size
                    mtime = full_p.stat().st_mtime
                except Exception:
                    sz = 0
                    mtime = 0

                cat = self._categorize_extension(ext)
                entries.append({
                    "relative_path": str(rel_p).replace("\\", "/"),
                    "absolute_path": str(full_p),
                    "filename": f,
                    "extension": ext,
                    "category": cat,
                    "size_bytes": sz,
                    "size_str": self._format_size(sz),
                    "modified_time": mtime,
                })
        return sorted(entries, key=lambda x: x["relative_path"])

    @staticmethod
    def _categorize_extension(ext: str) -> str:
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".pgm"):
            return "Image"
        if ext in (".mp4", ".avi", ".mov", ".mkv"):
            return "Video"
        if ext in (".json", ".jsonl"):
            return "JSON"
        if ext in (".yaml", ".yml"):
            return "YAML"
        if ext in (".txt", ".log", ".md", ".sh"):
            return "Text"
        if ext in (".csv",):
            return "Table"
        if ext in (".ply", ".pcd"):
            return "3D Model / PointCloud"
        if ext in (".bag", ".osa"):
            return "SLAM Data / Bag"
        return "Binary / Other"

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    def read_file_preview(self, file_path: Path) -> Dict[str, Any]:
        """Read content formatted for appropriate UI viewer component."""
        file_path = Path(file_path)
        if not file_path.exists():
            return {"type": "error", "message": f"File does not exist: {file_path}"}

        ext = file_path.suffix.lower()
        sz = file_path.stat().st_size

        # 1. Images
        if ext in (".png", ".jpg", ".jpeg", ".bmp"):
            return {"type": "image", "path": str(file_path)}

        # 2. PGM ROS maps (convert to RGB for display)
        if ext == ".pgm":
            try:
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # In standard ROS PGM: 0 is occupied (black), 254 is free (white), 205 is unknown (grey)
                    rgb_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    return {"type": "numpy_image", "array": rgb_vis, "shape": img.shape}
            except Exception as e:
                return {"type": "error", "message": f"Failed to decode PGM: {e}"}

        # 3. Video
        if ext in (".mp4", ".mov", ".avi"):
            return {"type": "video", "path": str(file_path)}

        # 4. JSON / JSONL
        if ext in (".json", ".jsonl"):
            if sz > 5 * 1024 * 1024:
                return {"type": "text", "content": f"[File too large to render interactively: {self._format_size(sz)}. Use Download below.]"}
            try:
                if ext == ".json":
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return {"type": "json", "data": data}
                else:  # JSONL: parse top 50 lines
                    lines_data = []
                    with open(file_path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i >= 50:
                                break
                            if line.strip():
                                lines_data.append(json.loads(line))
                    return {"type": "jsonl", "data": lines_data, "total_lines": self._count_lines(file_path)}
            except Exception as e:
                return {"type": "error", "message": f"JSON parse error: {e}"}

        # 5. YAML
        if ext in (".yaml", ".yml"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    ydata = yaml.safe_load(f)
                return {"type": "yaml", "data": ydata}
            except Exception as e:
                return {"type": "error", "message": f"YAML parse error: {e}"}

        # 6. Text / Log / Markdown / CSV
        if ext in (".txt", ".log", ".md", ".sh", ".csv"):
            if sz > 3 * 1024 * 1024:
                # Read head
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    head_lines = [f.readline() for _ in range(200)]
                return {"type": "text", "content": "".join(head_lines) + f"\n\n... [Truncated: File is {self._format_size(sz)}]"}
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                return {"type": "text", "content": content}
            except Exception as e:
                return {"type": "error", "message": f"Read error: {e}"}

        # 7. 3D PLY PointCloud Summary
        if ext == ".ply":
            return {"type": "ply_info", "path": str(file_path), "size_str": self._format_size(sz)}

        # Default: Binary download
        return {"type": "binary", "path": str(file_path), "size_str": self._format_size(sz)}

    @staticmethod
    def _count_lines(p: Path) -> int:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
