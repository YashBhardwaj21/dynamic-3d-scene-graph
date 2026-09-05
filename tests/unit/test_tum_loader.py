"""Unit tests for TUM file parsers (Substage 1.1).

Verifies:
1. Comment and blank line skipping logic.
2. Timestamp and field parsing precision.
3. Quaternion to SE(3) transformation matrix properties.
4. Exact entry counts on TUM Freiburg1 Desk dataset:
   - 613 RGB entries
   - 595 Depth entries
   - 2335 Ground truth pose entries
5. Timestamp monotonicity on real dataset files.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from scene_graph.data.tum_loader import (
    DepthEntry,
    PoseEntry,
    RGBEntry,
    TUMLoader,
    load_tum_depth,
    load_tum_groundtruth,
    load_tum_rgb,
    parse_file_list,
)


WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
TUM_DATASET_DIR = WORKSPACE_DIR / "rgbd_dataset_freiburg1_desk"


def test_parse_file_list_synthetic():
    """Verify comment skipping and whitespace handling."""
    content = """# Header comment
# timestamp filename
  
1305031452.791720   rgb/1.png  
# Intermediate comment

1305031452.823674 rgb/2.png
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        rows = parse_file_list(temp_path)
        assert len(rows) == 2
        assert rows[0] == ["1305031452.791720", "rgb/1.png"]
        assert rows[1] == ["1305031452.823674", "rgb/2.png"]
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_parse_file_list_missing_file():
    """Verify FileNotFoundError on missing file."""
    with pytest.raises(FileNotFoundError):
        parse_file_list(Path("non_existent_file_path.txt"))


def test_load_tum_rgb_synthetic():
    """Verify RGBEntry construction from text file."""
    content = "# color images\n1305031452.791720 rgb/1305031452.791720.png\n"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        entries = load_tum_rgb(temp_path)
        assert len(entries) == 1
        assert isinstance(entries[0], RGBEntry)
        assert entries[0].timestamp == pytest.approx(1305031452.791720)
        assert entries[0].file_path == "rgb/1305031452.791720.png"
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_load_tum_depth_synthetic():
    """Verify DepthEntry construction from text file."""
    content = "# depth maps\n1305031453.374112 depth/1305031453.374112.png\n"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        entries = load_tum_depth(temp_path)
        assert len(entries) == 1
        assert isinstance(entries[0], DepthEntry)
        assert entries[0].timestamp == pytest.approx(1305031453.374112)
        assert entries[0].file_path == "depth/1305031453.374112.png"
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_load_tum_groundtruth_synthetic():
    """Verify PoseEntry parsing and SE(3) matrix generation."""
    content = "# ground truth trajectory\n1305031449.7996 1.2334 -0.0113 1.6941 0.7907 0.4393 -0.1770 -0.3879\n"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        entries = load_tum_groundtruth(temp_path)
        assert len(entries) == 1
        pose = entries[0]
        assert isinstance(pose, PoseEntry)
        assert pose.timestamp == pytest.approx(1305031449.7996)
        assert pose.tx == pytest.approx(1.2334)
        assert pose.ty == pytest.approx(-0.0113)
        assert pose.tz == pytest.approx(1.6941)
        assert pose.qx == pytest.approx(0.7907)
        assert pose.qy == pytest.approx(0.4393)
        assert pose.qz == pytest.approx(-0.1770)
        assert pose.qw == pytest.approx(-0.3879)

        # Check translation property
        np.testing.assert_allclose(pose.translation, [1.2334, -0.0113, 1.6941])

        # Check SE(3) transformation matrix properties
        T = pose.as_transform_matrix()
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[:3, 3], [1.2334, -0.0113, 1.6941])
        # R must be orthogonal: R @ R.T == I
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        # Det(R) must be +1 (proper rotation)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_real_dataset_counts_and_properties():
    """Verify exact counts required by Substage 1.1:

    - 613 RGB
    - 595 depth
    - 2335 GT
    And verify timestamp monotonicity.
    """
    if not TUM_DATASET_DIR.is_dir():
        pytest.skip(f"Dataset directory not found at {TUM_DATASET_DIR}")

    loader = TUMLoader(TUM_DATASET_DIR)

    # 1. RGB
    rgb_entries = loader.load_rgb()
    assert len(rgb_entries) == 613, f"Expected 613 RGB entries, found {len(rgb_entries)}"
    rgb_timestamps = [e.timestamp for e in rgb_entries]
    assert all(
        t2 >= t1 for t1, t2 in zip(rgb_timestamps[:-1], rgb_timestamps[1:])
    ), "RGB timestamps are not monotonically increasing"
    assert rgb_entries[0].timestamp == pytest.approx(1305031452.791720)
    assert rgb_entries[-1].timestamp == pytest.approx(1305031473.196069)

    # 2. Depth
    depth_entries = loader.load_depth()
    assert len(depth_entries) == 595, f"Expected 595 Depth entries, found {len(depth_entries)}"
    depth_timestamps = [e.timestamp for e in depth_entries]
    assert all(
        t2 >= t1 for t1, t2 in zip(depth_timestamps[:-1], depth_timestamps[1:])
    ), "Depth timestamps are not monotonically increasing"
    assert depth_entries[0].timestamp == pytest.approx(1305031453.374112)
    assert depth_entries[-1].timestamp == pytest.approx(1305031473.190828)

    # 3. Ground truth
    gt_entries = loader.load_groundtruth()
    assert len(gt_entries) == 2335, f"Expected 2335 GT entries, found {len(gt_entries)}"
    gt_timestamps = [e.timestamp for e in gt_entries]
    assert all(
        t2 >= t1 for t1, t2 in zip(gt_timestamps[:-1], gt_timestamps[1:])
    ), "GT timestamps are not monotonically increasing"
    assert gt_entries[0].timestamp == pytest.approx(1305031449.7996)
    assert gt_entries[-1].timestamp == pytest.approx(1305031473.1991)
