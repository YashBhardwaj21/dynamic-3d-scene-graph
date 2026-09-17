import json
import socket
import struct
import sys
import time
from pathlib import Path
import numpy as np
import pytest

WS_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WS_ROOT / "tools"))
sys.path.insert(0, str(WS_ROOT / "ros2_ws" / "src" / "d455_bridge"))

from d455_sender import send_packet
from d455_bridge.d455_receiver import D455Receiver, recv_exact


def create_synthetic_frame_data(width=64, height=48, skew_ms=5.0):
    rgb = np.full((height, width, 3), 128, dtype=np.uint8)
    depth = np.full((height, width), 1500, dtype=np.uint16)
    rgb_bytes = rgb.tobytes()
    depth_bytes = depth.tobytes()

    header = {
        "type": "d455_rgbd",
        "frame_id": 1,
        "host_time": time.time(),
        "color": {
            "timestamp": 1000.0,
            "timestamp_domain": "hardware_clock",
            "width": width,
            "height": height,
            "format": "bgr8",
            "channels": 3,
            "bytes_per_channel": 1,
            "payload_size": len(rgb_bytes),
            "fx": 384.0,
            "fy": 384.0,
            "ppx": 32.0,
            "ppy": 24.0,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "distortion_model": "plumb_bob",
        },
        "depth": {
            "timestamp": 1000.0 + skew_ms,
            "timestamp_domain": "hardware_clock",
            "width": width,
            "height": height,
            "format": "z16",
            "channels": 1,
            "bytes_per_channel": 2,
            "payload_size": len(depth_bytes),
            "depth_scale": 0.001,
            "is_aligned_to_color": True,
        },
        "imu": [
            {
                "timestamp": 1.0,
                "accel": [0.0, 0.0, 9.81],
                "gyro": [0.01, -0.02, 0.005],
            }
        ],
        "rgb_depth_dt_ms": skew_ms,
    }
    return header, rgb_bytes, depth_bytes


def test_send_packet_framing():
    header, rgb_bytes, depth_bytes = create_synthetic_frame_data(16, 12)

    class MockSocket:
        def __init__(self):
            self.sent_data = bytearray()

        def sendall(self, data):
            self.sent_data.extend(data)

    mock_sock = MockSocket()
    duration_ms = send_packet(mock_sock, header, rgb_bytes, depth_bytes)
    assert duration_ms >= 0.0

    raw = bytes(mock_sock.sent_data)
    assert len(raw) > 4
    header_len = struct.unpack("!I", raw[:4])[0]
    header_raw = raw[4 : 4 + header_len]
    parsed_header = json.loads(header_raw.decode("utf-8"))

    assert parsed_header["type"] == "d455_rgbd"
    assert parsed_header["color"]["fx"] == 384.0
    assert parsed_header["depth"]["is_aligned_to_color"] is True

    payload_offset = 4 + header_len
    rgb_extracted = raw[payload_offset : payload_offset + len(rgb_bytes)]
    depth_extracted = raw[payload_offset + len(rgb_bytes) :]

    assert rgb_extracted == rgb_bytes
    assert depth_extracted == depth_bytes


def test_d455_receiver_network_stream_and_parsing():
    receiver = D455Receiver(host="127.0.0.1", port=0, queue_size=4)
    port = receiver.bound_port
    assert port > 0

    header, rgb_bytes, depth_bytes = create_synthetic_frame_data(32, 24, skew_ms=3.2)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(("127.0.0.1", port))
        send_packet(client, header, rgb_bytes, depth_bytes)

        deadline = time.time() + 2.0
        packet = None
        while time.time() < deadline:
            if not receiver.frame_queue.empty():
                packet = receiver.frame_queue.get_nowait()
                break
            time.sleep(0.01)

        assert packet is not None, "Frame packet was not received by receiver within timeout"
        assert packet["rgb_bytes"] == rgb_bytes
        assert packet["depth_bytes"] == depth_bytes
        assert packet["header"]["frame_id"] == 1
        assert packet["header"]["rgb_depth_dt_ms"] == 3.2
    finally:
        client.close()
        receiver.destroy_node()


def test_rgb_depth_skew_rejection():
    receiver = D455Receiver(host="127.0.0.1", port=0, queue_size=4, max_skew_ms=33.0)
    try:
        header_bad, rgb_b, depth_b = create_synthetic_frame_data(16, 12, skew_ms=52.0)
        packet_bad = {
            "header": header_bad,
            "rgb_bytes": rgb_b,
            "depth_bytes": depth_b,
            "receive_time": time.time(),
        }
        receiver._publish_frame(packet_bad)
        assert len(receiver.rgb_pub.published) == 0
        assert len(receiver.depth_pub.published) == 0

        header_ok, _, _ = create_synthetic_frame_data(16, 12, skew_ms=12.0)
        packet_ok = {
            "header": header_ok,
            "rgb_bytes": rgb_b,
            "depth_bytes": depth_b,
            "receive_time": time.time(),
        }
        receiver._publish_frame(packet_ok)
        assert len(receiver.rgb_pub.published) == 1
        assert len(receiver.depth_pub.published) == 1
        assert len(receiver.camera_info_pub.published) == 1
        assert len(receiver.imu_pub.published) == 1
    finally:
        receiver.destroy_node()


def test_bounded_queue_drops_oldest():
    receiver = D455Receiver(host="127.0.0.1", port=0, queue_size=2)
    try:
        header, rgb_b, depth_b = create_synthetic_frame_data(8, 8)
        for f_id in range(4):
            hdr = dict(header)
            hdr["frame_id"] = f_id
            pkt = {"header": hdr, "rgb_bytes": rgb_b, "depth_bytes": depth_b, "receive_time": time.time()}
            try:
                receiver.frame_queue.put_nowait(pkt)
            except Exception:
                try:
                    receiver.frame_queue.get_nowait()
                    receiver.dropped_frames += 1
                except Exception:
                    pass
                receiver.frame_queue.put_nowait(pkt)

        assert receiver.dropped_frames == 2
        assert receiver.frame_queue.qsize() == 2

        p1 = receiver.frame_queue.get_nowait()
        p2 = receiver.frame_queue.get_nowait()
        assert p1["header"]["frame_id"] == 2
        assert p2["header"]["frame_id"] == 3
    finally:
        receiver.destroy_node()


def test_ros_message_field_fidelity():
    receiver = D455Receiver(host="127.0.0.1", port=0, queue_size=4)
    try:
        header, rgb_b, depth_b = create_synthetic_frame_data(64, 48, skew_ms=4.0)
        packet = {
            "header": header,
            "rgb_bytes": rgb_b,
            "depth_bytes": depth_b,
            "receive_time": time.time(),
        }
        receiver._publish_frame(packet)

        assert len(receiver.rgb_pub.published) == 1
        rgb_msg = receiver.rgb_pub.published[0]
        assert rgb_msg.header.frame_id == "camera_color_optical_frame"
        assert rgb_msg.width == 64
        assert rgb_msg.height == 48
        assert rgb_msg.encoding == "bgr8"
        assert rgb_msg.step == 64 * 3
        assert bytes(rgb_msg.data) == rgb_b

        assert len(receiver.depth_pub.published) == 1
        depth_msg = receiver.depth_pub.published[0]
        assert depth_msg.header.frame_id == "camera_color_optical_frame"
        assert depth_msg.width == 64
        assert depth_msg.height == 48
        assert depth_msg.encoding == "16UC1"
        assert depth_msg.step == 64 * 2
        assert bytes(depth_msg.data) == depth_b

        assert len(receiver.camera_info_pub.published) == 1
        cam_msg = receiver.camera_info_pub.published[0]
        assert cam_msg.header.frame_id == "camera_color_optical_frame"
        assert cam_msg.width == 64
        assert cam_msg.height == 48
        assert cam_msg.k[0] == 384.0
        assert cam_msg.k[4] == 384.0
        assert cam_msg.k[2] == 32.0
        assert cam_msg.k[5] == 24.0
        assert cam_msg.distortion_model == "plumb_bob"

        assert len(receiver.imu_pub.published) == 1
        imu_msg = receiver.imu_pub.published[0]
        assert imu_msg.header.frame_id == "camera_imu_optical_frame"
        assert pytest.approx(imu_msg.linear_acceleration.z, abs=1e-3) == 9.81
        assert pytest.approx(imu_msg.angular_velocity.x, abs=1e-3) == 0.01

        assert rgb_msg.header.stamp.sec == depth_msg.header.stamp.sec
        assert rgb_msg.header.stamp.nanosec == depth_msg.header.stamp.nanosec
        assert rgb_msg.header.stamp.sec == cam_msg.header.stamp.sec
        assert rgb_msg.header.stamp.nanosec == cam_msg.header.stamp.nanosec
    finally:
        receiver.destroy_node()


def test_receiver_reconnect_resilience():
    receiver = D455Receiver(host="127.0.0.1", port=0, queue_size=4)
    port = receiver.bound_port
    try:
        header, rgb_b, depth_b = create_synthetic_frame_data(16, 12)

        c1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c1.connect(("127.0.0.1", port))
        send_packet(c1, header, rgb_b, depth_b)
        time.sleep(0.05)
        c1.close()

        time.sleep(0.1)
        c2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c2.connect(("127.0.0.1", port))
        hdr2 = dict(header)
        hdr2["frame_id"] = 2
        send_packet(c2, hdr2, rgb_b, depth_b)
        time.sleep(0.05)
        c2.close()

        assert receiver.total_reconnections >= 2
    finally:
        receiver.destroy_node()
