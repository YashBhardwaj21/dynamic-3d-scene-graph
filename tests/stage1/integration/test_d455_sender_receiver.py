"""Integration test: Localhost TCP streaming from Stage 1 Sender to Receiver and ROS topic publication."""

import json
import socket
import struct
import time

import numpy as np

from ros2_ws.src.d455_bridge.d455_bridge.d455_receiver import HAS_RCLPY, D455Receiver


def _send_packet(client: socket.socket, header: dict, rgb_data: bytes, depth_data: bytes):
    h_bytes = json.dumps(header).encode("utf-8")
    payload = struct.pack("!I", len(h_bytes)) + h_bytes + rgb_data + depth_data
    client.sendall(payload)


def test_sender_receiver_socket_loopback_and_ros_publication():
    """Verify that a TCP client streaming D455 packets results in valid ROS topic publications."""
    temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    temp_sock.bind(("127.0.0.1", 0))
    port = temp_sock.getsockname()[1]
    temp_sock.close()

    receiver = D455Receiver(host="127.0.0.1", port=port, queue_size=4, max_skew_ms=50.0)

    try:
        time.sleep(0.1)

        client = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        rgb_data = np.full((480, 640, 3), 150, dtype=np.uint8).tobytes()
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
                "distortion": [0.01, -0.02, 0.001, -0.001, 0.0],
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
            "imu": [
                {
                    "timestamp": 10.002,
                    "domain": "hardware_clock",
                    "accel": [0.1, 9.81, -0.2],
                    "gyro": [0.01, -0.02, 0.03],
                }
            ],
            "rgb_depth_dt_ms": 5.0,
        }

        _send_packet(client, header, rgb_data, depth_data)

        # Wait for receiver socket thread to ingest packet into queue
        for _ in range(40):
            if not receiver.frame_queue.empty():
                break
            time.sleep(0.05)

        assert not receiver.frame_queue.empty()
        assert receiver.total_frames_received == 1

        if not HAS_RCLPY:
            # Standalone unit test environment: verify receiver dispatch and topic publications on mock publishers
            receiver._dispatch_queued_frames()
            assert receiver.total_frames_published == 1
            assert len(receiver.rgb_pub.published) == 1
            assert len(receiver.depth_pub.published) == 1
            assert len(receiver.camera_info_pub.published) == 1
            assert len(receiver.metadata_pub.published) == 1

            meta_raw = receiver.metadata_pub.published[-1]
            meta = json.loads(meta_raw.data)
            assert meta["session_id"] == "test_loopback"
            assert meta["sequence_number"] == 1
            assert meta["depth_scale"] == pytest.approx(0.001)
            assert meta["rgb_depth_dt_ms"] == pytest.approx(5.0)
            assert meta["source_timestamp"] == pytest.approx(10.0)
            assert "network_arrival_timestamp" in meta
            assert "mapped_ros_timestamp" in meta
            assert "host_capture_timestamp" in meta
        else:
            # Create dedicated subscriber node to verify actual ROS topic publication
            import rclpy
            from sensor_msgs.msg import CameraInfo, Image, Imu
            from std_msgs.msg import String as StringMsg

            sub_node = rclpy.create_node("test_topic_subscriber")
            received_rgb = []
            received_depth = []
            received_info = []
            received_imu = []
            received_meta = []

            sub_node.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: received_rgb.append(m), 10)
            sub_node.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", lambda m: received_depth.append(m), 10)
            sub_node.create_subscription(CameraInfo, "/camera/camera/color/camera_info", lambda m: received_info.append(m), 10)
            sub_node.create_subscription(Imu, "/camera/camera/imu", lambda m: received_imu.append(m), 10)
            sub_node.create_subscription(StringMsg, "/camera/camera/metadata", lambda m: received_meta.append(m), 10)

            executor = rclpy.executors.SingleThreadedExecutor()
            executor.add_node(receiver)
            executor.add_node(sub_node)

            # Wait for DDS subscription discovery
            for _ in range(50):
                executor.spin_once(timeout_sec=0.02)
                if receiver.rgb_pub.get_subscription_count() > 0:
                    break

            # Dispatch frame to ROS publishers
            receiver._dispatch_queued_frames()

            assert receiver.total_frames_published == 1

            # Spin executor to process delivered ROS messages
            for _ in range(20):
                executor.spin_once(timeout_sec=0.05)
                if received_rgb and received_depth and received_info and received_imu and received_meta:
                    break

            # Verify actual ROS topic publications
            assert len(received_rgb) >= 1
            rgb_msg = received_rgb[-1]
            assert rgb_msg.header.frame_id == "camera_color_optical_frame"
            assert rgb_msg.width == 640
            assert rgb_msg.height == 480
            assert rgb_msg.encoding == "bgr8"

            assert len(received_depth) >= 1
            depth_msg = received_depth[-1]
            assert depth_msg.header.frame_id == "camera_color_optical_frame"
            assert depth_msg.width == 640
            assert depth_msg.height == 480
            assert depth_msg.encoding == "16UC1"

            assert len(received_info) >= 1
            info_msg = received_info[-1]
            assert info_msg.width == 640
            assert info_msg.height == 480
            assert info_msg.k[0] == pytest.approx(385.0)
            assert info_msg.distortion_model == "plumb_bob"
            assert list(info_msg.d) == pytest.approx([0.01, -0.02, 0.001, -0.001, 0.0])

            assert len(received_imu) >= 1
            imu_msg = received_imu[-1]
            assert imu_msg.header.frame_id == "camera_imu_optical_frame"
            assert imu_msg.linear_acceleration.y == pytest.approx(9.81)
            assert imu_msg.angular_velocity.z == pytest.approx(0.03)

            assert len(received_meta) >= 1
            meta_record = json.loads(received_meta[-1].data)
            assert meta_record["session_id"] == "test_loopback"
            assert meta_record["sequence_number"] == 1
            assert meta_record["depth_scale"] == pytest.approx(0.001)
            assert meta_record["rgb_depth_dt_ms"] == pytest.approx(5.0)
            assert meta_record["source_timestamp"] == pytest.approx(10.0)
            assert "network_arrival_timestamp" in meta_record
            assert "mapped_ros_timestamp" in meta_record
            assert "host_capture_timestamp" in meta_record

        client.close()

    finally:
        if "sub_node" in locals():
            sub_node.destroy_node()
        receiver.destroy_node()


def test_d455_receiver_sequence_gap_and_session_reset():
    """Verify that forward sequence gaps are tracked and session change clears stale queue."""
    temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    temp_sock.bind(("127.0.0.1", 0))
    port = temp_sock.getsockname()[1]
    temp_sock.close()

    receiver = D455Receiver(host="127.0.0.1", port=port, queue_size=5)

    try:
        time.sleep(0.1)
        client = socket.create_connection(("127.0.0.1", port), timeout=2.0)

        rgb_bytes = np.zeros((480, 640, 3), dtype=np.uint8).tobytes()
        depth_bytes = np.zeros((480, 640), dtype=np.uint16).tobytes()

        def make_h(session_id, f_id):
            return {
                "protocol_version": 2,
                "session_id": session_id,
                "frame_id": f_id,
                "color": {
                    "timestamp": 10.0 + f_id * 0.033,
                    "width": 640, "height": 480, "channels": 3, "bytes_per_channel": 1,
                    "payload_size": len(rgb_bytes), "fx": 385.0, "fy": 385.0, "ppx": 320.0, "ppy": 240.0,
                },
                "depth": {
                    "timestamp": 10.0 + f_id * 0.033,
                    "width": 640, "height": 480, "channels": 1, "bytes_per_channel": 2,
                    "payload_size": len(depth_bytes),
                },
                "imu": [],
            }

        # 1. Send frame 1
        _send_packet(client, make_h("session_A", 1), rgb_bytes, depth_bytes)
        time.sleep(0.1)

        # 2. Send frame 5 (gap of 3 frames: 2, 3, 4 missing)
        _send_packet(client, make_h("session_A", 5), rgb_bytes, depth_bytes)
        time.sleep(0.1)

        assert receiver.sequence_gap_count == 1
        assert receiver.missing_frames_count == 3

        # 3. New session B arrives: should purge existing frames from session A
        _send_packet(client, make_h("session_B", 1), rgb_bytes, depth_bytes)
        for _ in range(50):
            if receiver.current_session_id == "session_B":
                break
            time.sleep(0.02)

        assert receiver.current_session_id == "session_B"
        assert receiver.last_sequence_number == 1
        # Dropped frames should reflect the purged session A frames
        assert receiver.dropped_frames >= 2

        client.close()

    finally:
        receiver.destroy_node()
