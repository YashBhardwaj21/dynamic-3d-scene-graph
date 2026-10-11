"""Unit tests for rigid coordinate frame contract and mathematical quality gates."""

import numpy as np
import pytest

from scene_graph.geometry.coordinate_frames import (
    CoordinateFrameContract,
    RobotExtrinsics,
    verify_transform_matrix,
    verify_round_trip,
    FRAME_FLOOR_MAP,
    FRAME_BASE_LINK,
)
from scene_graph.geometry.floor_alignment import construct_floor_transform


def test_verify_transform_matrix_identity():
    """Identity transform must pass all quality gates with zero error."""
    T = np.eye(4, dtype=np.float64)
    res = verify_transform_matrix(T, "identity")
    assert res["verified"] is True
    assert res["orthogonality_error"] < 1e-12
    assert abs(res["det"] - 1.0) < 1e-12


def test_verify_transform_matrix_rejection_bad_bottom_row():
    """Invalid bottom row must raise ValueError."""
    T = np.eye(4, dtype=np.float64)
    T[3, 0] = 0.1
    with pytest.raises(ValueError, match="bottom row violation"):
        verify_transform_matrix(T)


def test_verify_transform_matrix_rejection_non_orthogonal():
    """Non-orthogonal matrix must raise ValueError."""
    T = np.eye(4, dtype=np.float64)
    T[0, 1] = 0.5  # Shear
    with pytest.raises(ValueError, match="orthogonality violation"):
        verify_transform_matrix(T)


def test_verify_transform_matrix_rejection_left_handed():
    """Left-handed reflection (det = -1) must raise ValueError."""
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = -1.0  # Reflection
    with pytest.raises(ValueError, match="determinant violation"):
        verify_transform_matrix(T)


def test_robot_extrinsics_transforms():
    """Camera-to-base extrinsics must pass all mathematical quality gates."""
    ext = RobotExtrinsics(x_offset_m=0.15, y_offset_m=0.0, z_offset_m=0.18, pitch_deg=5.0)
    T_base_to_opt = ext.to_transform_base_to_camera_optical()
    T_opt_to_base = ext.to_transform_camera_optical_to_base()

    verify_transform_matrix(T_base_to_opt, "base_to_opt")
    verify_transform_matrix(T_opt_to_base, "opt_to_base")

    # Inverse consistency
    inv_err = np.max(np.abs(T_base_to_opt @ T_opt_to_base - np.eye(4)))
    assert inv_err < 1e-10

    # Point round-trip check
    pts = np.random.uniform(-5.0, 5.0, size=(100, 3))
    round_trip_err = verify_round_trip(pts, T_base_to_opt, T_opt_to_base)
    assert round_trip_err < 1e-10


def test_coordinate_frame_contract_with_real_plane():
    """Contract initialized with arbitrary valid floor plane must satisfy all invariants."""
    # Example upward normal from real sequence
    normal = np.array([-0.0582, -0.9006, 0.4307])
    normal /= np.linalg.norm(normal)
    d = 0.8886

    T_floor_world, T_world_floor = construct_floor_transform(normal, d)
    contract = CoordinateFrameContract(
        T_floor_world=T_floor_world,
        T_world_floor=T_world_floor,
        extrinsics=RobotExtrinsics(),
    )

    # Floor plane point transformed to floor_map must have Z = 0
    p_floor_in_world = -d * normal
    p_floor_homogeneous = np.array([p_floor_in_world[0], p_floor_in_world[1], p_floor_in_world[2], 1.0])
    p_in_floor_frame = T_floor_world @ p_floor_homogeneous
    assert abs(p_in_floor_frame[2]) < 1e-6, f"Floor point Z must be 0, got {p_in_floor_frame[2]}"

    # Verify serialization dictionary
    data = contract.to_dict()
    assert data["source_frames"]["floor_map"] == FRAME_FLOOR_MAP
    assert data["source_frames"]["base_link"] == FRAME_BASE_LINK
    assert len(data["T_floor_world"]) == 4
