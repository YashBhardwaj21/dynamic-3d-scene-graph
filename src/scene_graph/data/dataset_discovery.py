"""Dynamic Recording Discovery, Camera Calibration Ingestion, and RGB-D Association."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import yaml

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.data.synchronization import associate


@dataclass
class RecordingManifest:
    """Complete manifest of discovered sequence inputs and calibrated parameters."""
    dataset_dir: Path
    sequence_name: str
    rgb_frames: List[Tuple[float, str]]          # List of (timestamp, relative_path)
    depth_frames: List[Tuple[float, str]]        # List of (timestamp, relative_path)
    associated_pairs: List[Tuple[float, str, float, str]]  # List of (t_rgb, rgb_rel, t_depth, depth_rel)
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float = 0.001        # Default 1mm per unit for uint16 depth
    depth_units_per_meter: float = 1000.0
    fps_estimate: float = 30.0
    duration_s: float = 0.0
    metadata_source: str = "auto"
    has_precomputed_associations: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def total_rgb_count(self) -> int:
        return len(self.rgb_frames)

    @property
    def total_depth_count(self) -> int:
        return len(self.depth_frames)

    @property
    def total_associated_count(self) -> int:
        return len(self.associated_pairs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "sequence_name": self.sequence_name,
            "total_rgb_frames": self.total_rgb_count,
            "total_depth_frames": self.total_depth_count,
            "total_associated_pairs": self.total_associated_count,
            "fps_estimate": round(self.fps_estimate, 2),
            "duration_s": round(self.duration_s, 2),
            "depth_scale_m_per_unit": self.depth_scale_m_per_unit,
            "depth_units_per_meter": self.depth_units_per_meter,
            "intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
            },
            "metadata_source": self.metadata_source,
            "has_precomputed_associations": self.has_precomputed_associations,
            "notes": self.notes,
        }


def _read_tum_list(file_path: Path) -> List[Tuple[float, str]]:
    """Read a TUM-style timestamp + relative path index file (e.g. rgb.txt or depth.txt)."""
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    ts = float(parts[0])
                    rel_path = parts[1]
                    items.append((ts, rel_path))
                except ValueError:
                    continue
    items.sort(key=lambda x: x[0])
    return items


def _scan_image_dir(dir_path: Path, rel_prefix: str) -> List[Tuple[float, str]]:
    """Scan directory of image files, parsing timestamps from filenames."""
    items = []
    valid_exts = {".png", ".jpg", ".jpeg", ".tiff"}
    if not dir_path.exists():
        return items

    for f in dir_path.iterdir():
        if f.suffix.lower() in valid_exts:
            # Parse timestamp from filename stem
            stem = f.stem
            try:
                ts = float(stem)
            except ValueError:
                # If stem isn't a float, use file modification time or sequential index
                ts = float(f.stat().st_mtime)
            rel_path = f"{rel_prefix}/{f.name}"
            items.append((ts, rel_path))
    items.sort(key=lambda x: x[0])
    return items


def load_camera_calibration(
    dataset_path: Path,
    intrinsics_file: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[CameraIntrinsics, float, float, str]:
    """Dynamically load or discover camera calibration and depth scale.

    Returns:
        (intrinsics, depth_scale_m_per_unit, depth_units_per_meter, source_description)
    """
    overrides = overrides or {}

    # Defaults (standard 640x480 RealSense D455 profile)
    fx = overrides.get("fx")
    fy = overrides.get("fy")
    cx = overrides.get("cx")
    cy = overrides.get("cy")
    width = overrides.get("width")
    height = overrides.get("height")
    depth_scale = overrides.get("depth_scale")

    source_desc = "defaults"

    # 1. Custom file provided
    candidate_files = []
    if intrinsics_file and intrinsics_file.exists():
        candidate_files.append((intrinsics_file, "custom_cli_file"))

    # 2. Look inside recording directory
    candidate_files.extend([
        (dataset_path / "camera_intrinsics.json", "camera_intrinsics.json"),
        (dataset_path / "metadata.json", "metadata.json"),
        (dataset_path / "config.yaml", "config.yaml"),
        (dataset_path / "d455_rgbd.yaml", "d455_rgbd.yaml"),
    ])

    for c_path, label in candidate_files:
        if not c_path.exists():
            continue

        try:
            if c_path.suffix.lower() == ".json":
                with open(c_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Format A: metadata.json with color_intrinsics
                if "color_intrinsics" in data:
                    ci = data["color_intrinsics"]
                    fx = fx or float(ci.get("fx", 384.13))
                    fy = fy or float(ci.get("fy", 383.69))
                    cx = cx or float(ci.get("cx", 320.27))
                    cy = cy or float(ci.get("cy", 244.57))
                    width = width or int(ci.get("width", 640))
                    height = height or int(ci.get("height", 480))
                    depth_scale = depth_scale or float(data.get("depth_scale_m_per_unit", 0.001))
                    source_desc = f"{label} (color_intrinsics)"
                    break

                # Format B: flat camera_intrinsics.json
                if "fx" in data:
                    fx = fx or float(data["fx"])
                    fy = fy or float(data.get("fy", data["fx"]))
                    cx = cx or float(data["cx"])
                    cy = cy or float(data["cy"])
                    width = width or int(data.get("width", 640))
                    height = height or int(data.get("height", 480))
                    d_scale_val = data.get("depth_unit_meters", data.get("depth_scale_m_per_unit"))
                    if d_scale_val is not None:
                        depth_scale = depth_scale or float(d_scale_val)
                    elif "depth_scale" in data and float(data["depth_scale"]) > 10.0:
                        # depth_scale given as units per meter (e.g. 1000 or 5000)
                        depth_scale = depth_scale or (1.0 / float(data["depth_scale"]))
                    source_desc = f"{label} (flat_json)"
                    break

            elif c_path.suffix.lower() in {".yaml", ".yml"}:
                with open(c_path, "r", encoding="utf-8") as f:
                    ydata = yaml.safe_load(f)
                if ydata and ("Camera1.fx" in ydata or "Camera.fx" in ydata):
                    pfx = "Camera1." if "Camera1.fx" in ydata else "Camera."
                    fx = fx or float(ydata[f"{pfx}fx"])
                    fy = fy or float(ydata.get(f"{pfx}fy", fx))
                    cx = cx or float(ydata[f"{pfx}cx"])
                    cy = cy or float(ydata[f"{pfx}cy"])
                    width = width or int(ydata.get("Camera.width", 640))
                    height = height or int(ydata.get("Camera.height", 480))
                    if "RGBD.DepthMapFactor" in ydata:
                        depth_scale = depth_scale or (1.0 / float(ydata["RGBD.DepthMapFactor"]))
                    source_desc = f"{label} (yaml)"
                    break
        except Exception:
            continue

    # Final fallback resolution
    fx = fx if fx is not None else 384.13
    fy = fy if fy is not None else 383.69
    cx = cx if cx is not None else 320.27
    cy = cy if cy is not None else 244.57
    width = width if width is not None else 640
    height = height if height is not None else 480
    depth_scale = depth_scale if depth_scale is not None else 0.001

    depth_units_per_m = 1.0 / depth_scale if depth_scale > 0 else 1000.0

    intrinsics = CameraIntrinsics(
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        width=int(width), height=int(height)
    )

    return intrinsics, float(depth_scale), float(depth_units_per_m), source_desc


def discover_recording(
    dataset_dir: str | Path,
    intrinsics_file: Optional[str | Path] = None,
    max_sync_dt: float = 0.05,
    custom_depth_scale: Optional[float] = None,
    custom_associations_file: Optional[str | Path] = None,
) -> RecordingManifest:
    """Dynamically discover frames, calibration, and timestamp synchronization for an arbitrary recording."""
    ds_path = Path(dataset_dir).resolve()
    if not ds_path.exists() or not ds_path.is_dir():
        raise FileNotFoundError(f"Recording directory does not exist or is not a directory: {ds_path}")

    seq_name = ds_path.name
    notes = []

    # 1. Discover RGB frames
    rgb_file = ds_path / "rgb.txt"
    if rgb_file.exists():
        rgb_frames = _read_tum_list(rgb_file)
        notes.append(f"Discovered {len(rgb_frames)} RGB frames from rgb.txt")
    else:
        rgb_frames = _scan_image_dir(ds_path / "rgb", "rgb")
        notes.append(f"Discovered {len(rgb_frames)} RGB frames via directory scan (rgb/)")

    if not rgb_frames:
        raise ValueError(f"No RGB images found in {ds_path} (checked rgb.txt and rgb/ directory)!")

    # 2. Discover Depth frames
    depth_file = ds_path / "depth.txt"
    if depth_file.exists():
        depth_frames = _read_tum_list(depth_file)
        notes.append(f"Discovered {len(depth_frames)} Depth frames from depth.txt")
    else:
        depth_frames = _scan_image_dir(ds_path / "depth", "depth")
        notes.append(f"Discovered {len(depth_frames)} Depth frames via directory scan (depth/)")

    if not depth_frames:
        raise ValueError(f"No Depth images found in {ds_path} (checked depth.txt and depth/ directory)!")

    # 3. Discover or Generate RGB-D Associations
    has_precomputed = False
    associated_pairs = []

    # Check custom associations file
    if custom_associations_file:
        c_assoc_path = Path(custom_associations_file)
        if c_assoc_path.exists():
            with open(c_assoc_path, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) >= 4 and not p[0].startswith("#"):
                        associated_pairs.append((float(p[0]), p[1], float(p[2]), p[3]))
            has_precomputed = True
            notes.append(f"Loaded {len(associated_pairs)} precomputed associations from {custom_associations_file}")

    # Check dataset directory associations.txt
    if not associated_pairs and (ds_path / "associations.txt").exists():
        with open(ds_path / "associations.txt", "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 4 and not p[0].startswith("#"):
                    associated_pairs.append((float(p[0]), p[1], float(p[2]), p[3]))
        if associated_pairs:
            has_precomputed = True
            notes.append(f"Loaded {len(associated_pairs)} associations from dataset associations.txt")

    # If no associations or custom requested, generate dynamically via optimal monotonic DP
    if not associated_pairs:
        rgb_ts = [t for t, _ in rgb_frames]
        depth_ts = [t for t, _ in depth_frames]
        matches = associate(rgb_ts, depth_ts, max_dt=max_sync_dt)
        for i_rgb, i_depth in matches:
            associated_pairs.append((
                rgb_frames[i_rgb][0],
                rgb_frames[i_rgb][1],
                depth_frames[i_depth][0],
                depth_frames[i_depth][1],
            ))
        notes.append(f"Dynamically generated {len(associated_pairs)} optimal monotonic RGB-D associations (max_dt={max_sync_dt}s)")

    if not associated_pairs:
        raise RuntimeError(f"Failed to associate any RGB and Depth frames within max_dt={max_sync_dt}s in {ds_path}!")

    # 4. Discover Camera Calibration & Depth Scale
    overrides = {}
    if custom_depth_scale is not None:
        overrides["depth_scale"] = custom_depth_scale

    intr_path = Path(intrinsics_file) if intrinsics_file else None
    intrinsics, depth_scale, depth_units_per_m, calib_src = load_camera_calibration(
        ds_path, intr_path, overrides
    )
    notes.append(f"Calibration loaded from: {calib_src} (fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, scale={depth_scale:.6f}m/unit)")

    # 5. Estimate sequence duration and FPS
    t_start = associated_pairs[0][0]
    t_end = associated_pairs[-1][0]
    duration_s = max(0.001, t_end - t_start)
    fps_est = len(associated_pairs) / duration_s if duration_s > 0 else 30.0

    return RecordingManifest(
        dataset_dir=ds_path,
        sequence_name=seq_name,
        rgb_frames=rgb_frames,
        depth_frames=depth_frames,
        associated_pairs=associated_pairs,
        intrinsics=intrinsics,
        depth_scale_m_per_unit=depth_scale,
        depth_units_per_meter=depth_units_per_m,
        fps_estimate=fps_est,
        duration_s=duration_s,
        metadata_source=calib_src,
        has_precomputed_associations=has_precomputed,
        notes=notes,
    )
