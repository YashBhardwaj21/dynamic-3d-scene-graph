"""Integration test: Localhost TCP streaming from Stage 1 Sender to Receiver."""

import json
import socket
import struct
import time

import numpy as np

from ros2_ws.src.d455_bridge.d455_bridge.d455_receiver import D455Receiver


def test_sender_receiver_socket_loopback():
    """Verify that a TCP client sending a Stage 1 packet is successfully received by D455Receiver."""
    # Find free port
    temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    temp_sock.bind(("127.0.0.1", 0))
    port = temp_sock.getsockname()[1]
    temp_sock.close()

    receiver = D455Receiver(host="127.0.0.1", port=port, queue_size=4, max_skew_ms=50.0)

    try:
        # Give receiver thread a moment to bind and listen
        time.sleep(0.1)

        # Connect sender client
        client = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Build synthetic packet
        rgb_data = np.full((480, 640, 3), 100, dtype=np.uint8).tobytes()
        depth_data = np.full((480, 640), 1200, dtype=np.uint16).tobytes()

        header = {
            "protocol_version": 2,
            "session_id": "test_loopback",
            "frame_id": 1,
            "color": {
                "timestamp": 10.0,
                "timestamp_domain": "hardware_clock",
                "width": 640,
                "height": 480,
                "format": "bgr8",
                "channels": 3,
                "bytes_per_channel": 1,
                "payload_size": len(rgb_data),
                "fx": 385.0,
                "fy": 385.0,
                "ppx": 320.0,
                "ppy": 240.0,
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "distortion_model": "plumb_bob",
            },
            "depth": {
                "timestamp": 10.005,
                "timestamp_domain": "hardware_clock",
                "width": 640,
                "height": 480,
                "format": "z16",
                "channels": 1,
                "bytes_per_channel": 2,
                "payload_size": len(depth_data),
                "depth_scale": 0.001,
            },
            "imu": [],
            "rgb_depth_dt_ms": 5.0,
        }

        h_bytes = json.dumps(header).encode("utf-8")
        payload = struct.pack("!I", len(h_bytes)) + h_bytes + rgb_data + depth_data

        client.sendall(payload)

        # Wait for receiver queue to capture the frame
        packet = None
        for _ in range(20):
            if not receiver.frame_queue.empty():
                packet = receiver.frame_queue.get_nowait()
                break
            time.sleep(0.05)

        assert packet is not None
        assert packet["header"]["frame_id"] == 1
        assert len(packet["rgb_bytes"]) == len(rgb_data)
        assert len(packet["depth_bytes"]) == len(depth_data)

        client.close()

    finally:
        receiver.destroy_node()
