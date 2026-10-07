"""Unit tests for Stage 1 Transport Framing and Protocol Invariants."""

import json
import struct

import numpy as np

from tests.stage1.fixtures.synthetic_generator import make_synthetic_packet_bytes


def test_transport_framing_integrity():
    """Verify that packet bytes can be partitioned into header, color, and depth streams."""
    raw = make_synthetic_packet_bytes(sequence_number=5, timestamp_sec=100.0, width=640, height=480)

    header_len = struct.unpack("!I", raw[:4])[0]
    header_json = raw[4 : 4 + header_len]
    header = json.loads(header_json.decode("utf-8"))

    color_offset = 4 + header_len
    color_len = header["color"]["payload_size"]
    color_bytes = raw[color_offset : color_offset + color_len]

    depth_offset = color_offset + color_len
    depth_len = header["depth"]["payload_size"]
    depth_bytes = raw[depth_offset : depth_offset + depth_len]

    assert len(raw) == depth_offset + depth_len
    assert len(color_bytes) == 640 * 480 * 3
    assert len(depth_bytes) == 640 * 480 * 2

    # Reconstruction check
    color_arr = np.frombuffer(color_bytes, dtype=np.uint8).reshape((480, 640, 3))
    depth_arr = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((480, 640))
    assert color_arr.shape == (480, 640, 3)
    assert depth_arr.shape == (480, 640)
