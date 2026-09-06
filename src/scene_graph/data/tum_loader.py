"""TUM RGB-D dataset file parsers.

Parses rgb.txt, depth.txt, and groundtruth.txt files following the TUM RGB-D format:
- Comments starting with '#' are ignored.
- Blank/whitespace-only lines are ignored.
- Timestamps are parsed as floating-point values.
- Ground truth poses are parsed into position (tx, ty, tz) and orientation quaternion (qx, qy, qz, qw).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Union
import numpy as np


@dataclass(frozen=True)
class RGBEntry:
    """An entry from rgb.txt.

    Attributes:
        timestamp: Capture timestamp in seconds.
        file_path: Relative path to the RGB image (e.g. 'rgb/1305031452.791720.png').
    """

    timestamp: float
    file_path: str


@dataclass(frozen=True)
class DepthEntry:
    """An entry from depth.txt.

    Attributes:
        timestamp: Capture timestamp in seconds.
        file_path: Relative path to the depth image (e.g. 'depth/1305031453.374112.png').
    """

    timestamp: float
    file_path: str


@dataclass(frozen=True)
class PoseEntry:
    """An entry from groundtruth.txt representing camera pose in world coordinates.

    Attributes:
        timestamp: Trajectory timestamp in seconds.
        tx: Translation along X axis in meters.
        ty: Translation along Y axis in meters.
        tz: Translation along Z axis in meters.
        qx: Quaternion x component.
        qy: Quaternion y component.
        qz: Quaternion z component.
        qw: Quaternion w component.
    """

    timestamp: float
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float

    @property
    def translation(self) -> np.ndarray:
        """3D translation vector [tx, ty, tz] as float64 ndarray."""
        return np.array([self.tx, self.ty, self.tz], dtype=np.float64)

    @property
    def quaternion(self) -> np.ndarray:
        """Quaternion [qx, qy, qz, qw] as float64 ndarray."""
        return np.array([self.qx, self.qy, self.qz, self.qw], dtype=np.float64)

    def as_transform_matrix(self) -> np.ndarray:
        """Convert translation and quaternion to a 4x4 SE(3) transformation matrix."""
        from scene_graph.geometry.transforms import pose_to_transform
        return pose_to_transform(self.tx, self.ty, self.tz, self.qx, self.qy, self.qz, self.qw)


def parse_file_list(file_path: Union[str, Path]) -> List[List[str]]:
    """Parse a whitespace-delimited TUM text file, skipping comments and empty lines.

    Args:
        file_path: Path to the text file.

    Returns:
        List of parsed token rows (each row is a list of string tokens).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"TUM file not found: {path.resolve()}")

    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            # Skip empty lines and comment lines
            if not line_str or line_str.startswith("#"):
                continue
            tokens = line_str.split()
            if tokens:
                rows.append(tokens)
    return rows


def load_tum_rgb(rgb_txt_path: Union[str, Path]) -> List[RGBEntry]:
    """Load RGB entries from a TUM rgb.txt file.

    Format per line: <timestamp> <rgb_relative_filepath>

    Args:
        rgb_txt_path: Path to rgb.txt.

    Returns:
        List of RGBEntry instances.
    """
    rows = parse_file_list(rgb_txt_path)
    entries: List[RGBEntry] = []
    for row_idx, tokens in enumerate(rows):
        if len(tokens) < 2:
            raise ValueError(
                f"Malformed line {row_idx} in {rgb_txt_path}: expected timestamp and filename, got: {tokens}"
            )
        timestamp = float(tokens[0])
        filename = tokens[1]
        entries.append(RGBEntry(timestamp=timestamp, file_path=filename))
    return entries


def load_tum_depth(depth_txt_path: Union[str, Path]) -> List[DepthEntry]:
    """Load Depth entries from a TUM depth.txt file.

    Format per line: <timestamp> <depth_relative_filepath>

    Args:
        depth_txt_path: Path to depth.txt.

    Returns:
        List of DepthEntry instances.
    """
    rows = parse_file_list(depth_txt_path)
    entries: List[DepthEntry] = []
    for row_idx, tokens in enumerate(rows):
        if len(tokens) < 2:
            raise ValueError(
                f"Malformed line {row_idx} in {depth_txt_path}: expected timestamp and filename, got: {tokens}"
            )
        timestamp = float(tokens[0])
        filename = tokens[1]
        entries.append(DepthEntry(timestamp=timestamp, file_path=filename))
    return entries


def load_tum_groundtruth(gt_txt_path: Union[str, Path]) -> List[PoseEntry]:
    """Load groundtruth trajectory poses from a TUM groundtruth.txt file.

    Format per line: <timestamp> <tx> <ty> <tz> <qx> <qy> <qz> <qw>

    Args:
        gt_txt_path: Path to groundtruth.txt.

    Returns:
        List of PoseEntry instances.
    """
    rows = parse_file_list(gt_txt_path)
    entries: List[PoseEntry] = []
    for row_idx, tokens in enumerate(rows):
        if len(tokens) < 8:
            raise ValueError(
                f"Malformed line {row_idx} in {gt_txt_path}: expected 8 elements, got {len(tokens)}: {tokens}"
            )
        timestamp = float(tokens[0])
        tx, ty, tz = float(tokens[1]), float(tokens[2]), float(tokens[3])
        qx, qy, qz, qw = (
            float(tokens[4]),
            float(tokens[5]),
            float(tokens[6]),
            float(tokens[7]),
        )
        entries.append(
            PoseEntry(
                timestamp=timestamp,
                tx=tx,
                ty=ty,
                tz=tz,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
            )
        )
    return entries


class TUMLoader:
    """Convenience loader for a full TUM RGB-D sequence directory."""

    def __init__(self, sequence_dir: Union[str, Path]) -> None:
        """Initialize TUMLoader with sequence directory path.

        Args:
            sequence_dir: Root directory of the sequence containing rgb.txt, depth.txt, and groundtruth.txt.
        """
        self.sequence_dir = Path(sequence_dir)
        if not self.sequence_dir.is_dir():
            raise NotADirectoryError(f"Sequence directory not found: {self.sequence_dir.resolve()}")

        self.rgb_txt_path = self.sequence_dir / "rgb.txt"
        self.depth_txt_path = self.sequence_dir / "depth.txt"
        self.gt_txt_path = self.sequence_dir / "groundtruth.txt"

    def load_rgb(self) -> List[RGBEntry]:
        """Load all RGB entries."""
        return load_tum_rgb(self.rgb_txt_path)

    def load_depth(self) -> List[DepthEntry]:
        """Load all depth entries."""
        return load_tum_depth(self.depth_txt_path)

    def load_groundtruth(self) -> List[PoseEntry]:
        """Load all ground truth trajectory poses."""
        return load_tum_groundtruth(self.gt_txt_path)

    def resolve_rgb_path(self, entry: RGBEntry) -> Path:
        """Resolve full filesystem path for an RGB entry."""
        return self.sequence_dir / entry.file_path

    def resolve_depth_path(self, entry: DepthEntry) -> Path:
        """Resolve full filesystem path for a depth entry."""
        return self.sequence_dir / entry.file_path
