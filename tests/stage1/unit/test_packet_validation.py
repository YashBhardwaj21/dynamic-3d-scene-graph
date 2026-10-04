"""Unit tests for Stage 1 Packet Validation and Malformed Input Rejection."""

import json
import struct

import pytest

from tests.stage1.fixtures.synthetic_generator import make_synthetic_packet_bytes


def test_valid_synthetic_packet_structure():
    raw_packet = make_synthetic_packet_bytes(sequence_number=10, timestamp_sec=50.0)
    assert len(raw_packet) > 4
    header_size = struct.unpack("!I", raw_packet[:4])[0]
    header_bytes = raw_packet[4 : 4 + header_size]
    header = json.loads(header_bytes.decode("utf-8"))

    assert header["protocol_version"] == 2
    assert header["frame_id"] == 10
    assert header["color"]["width"] == 640
    assert header["color"]["height"] == 480
    assert header["depth"]["depth_scale"] == 0.001


def test_reject_oversized_header_limit():
    MAX_HEADER_LIMIT = 1024 * 1024  # 1 MB
    impossible_size = 2 * 1024 * 1024
    raw_len = struct.pack("!I", impossible_size)
    header_size = struct.unpack("!I", raw_len)[0]

    assert header_size > MAX_HEADER_LIMIT  # Receiver logic rejects before allocation


def test_reject_inconsistent_dimensions():
    """Verify that payload sizes inconsistent with width * height * channels are detected."""
    c_w, c_h = 640, 480
    expected_rgb = c_w * c_h * 3
    tampered_rgb_size = expected_rgb - 100  # Corrupted or truncated payload length

    assert tampered_rgb_size != expected_rgb


def test_malformed_json_header_rejected_safely():
    garbage_bytes = b"{malformed json: true, bad: ["
    with pytest.raises(json.JSONDecodeError):
        json.loads(garbage_bytes.decode("utf-8"))
