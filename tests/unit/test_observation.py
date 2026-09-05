"""Unit tests for Observation dataclass and RLE serialization (Substage 2.1 & 2.2)."""

import numpy as np
import pytest

from scene_graph.perception.observation import Observation, encode_mask_rle, decode_mask_rle


def test_encode_decode_mask_rle():
    """Verify that RLE encoding and decoding are lossless."""
    # Create a 10x10 dummy mask
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    mask[8, 8] = True
    
    rle = encode_mask_rle(mask)
    
    assert "counts" in rle
    assert "size" in rle
    assert rle["size"] == [10, 10]
    
    decoded_mask = decode_mask_rle(rle)
    
    assert decoded_mask.shape == (10, 10)
    assert decoded_mask.dtype == bool
    np.testing.assert_array_equal(mask, decoded_mask)


def test_encode_mask_rle_all_background():
    """Verify RLE works for completely empty mask."""
    mask = np.zeros((10, 10), dtype=bool)
    rle = encode_mask_rle(mask)
    
    # Should be a single count of 100 for background
    assert rle["counts"] == [100]
    
    decoded = decode_mask_rle(rle)
    np.testing.assert_array_equal(mask, decoded)


def test_encode_mask_rle_all_foreground():
    """Verify RLE works for completely full mask."""
    mask = np.ones((10, 10), dtype=bool)
    rle = encode_mask_rle(mask)
    
    # RLE always starts with background count, so it should be [0, 100]
    assert rle["counts"] == [0, 100]
    
    decoded = decode_mask_rle(rle)
    np.testing.assert_array_equal(mask, decoded)


def test_encode_mask_rle_invalid_shape():
    """Verify RLE rejects non-2D arrays."""
    mask = np.zeros((10, 10, 3), dtype=bool)
    with pytest.raises(ValueError, match="must be 2D"):
        encode_mask_rle(mask)


def test_observation_dataclass():
    """Verify Observation dataclass instantiation and mask decoding."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True
    rle = encode_mask_rle(mask)
    
    obs = Observation(
        obs_id="obs_test_123",
        frame_index=150,
        timestamp=1.5,
        class_name="cup",
        confidence=0.85,
        bbox_xyxy=np.array([2.0, 2.0, 4.0, 4.0]),
        mask_rle=rle,
        centroid_world=np.array([1.0, 2.0, 3.0]),
        valid_point_count=4
    )
    
    assert obs.obs_id == "obs_test_123"
    assert obs.class_name == "cup"
    
    decoded = obs.get_mask()
    assert decoded is not None
    np.testing.assert_array_equal(decoded, mask)
    
    # Test observation with no mask
    obs_no_mask = Observation(
        obs_id="obs_test_456",
        frame_index=151,
        timestamp=1.6,
        class_name="book",
        confidence=0.9,
        bbox_xyxy=np.array([1.0, 1.0, 5.0, 5.0]),
        mask_rle=None
    )
    
    assert obs_no_mask.get_mask() is None
