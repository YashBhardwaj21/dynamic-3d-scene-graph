"""Unit tests for Stage 1 Bounded Buffer Policy (Latest-Frame / Drop-Oldest).

Directly validates D455Receiver:
1. When multiple video frames are queued, only the newest frame is published.
2. Older video frames are discarded and counted as dropped_frames.
3. High-rate IMU telemetry from all frames is preserved and published without loss.
"""

import numpy as np

from ros2_ws.src.d455_bridge.d455_bridge.d455_receiver import D455Receiver


def _create_synthetic_packet(frame_id: int, timestamp: float, num_imu_samples: int = 2):
    rgb = np.full((480, 640, 3), frame_id, dtype=np.uint8).tobytes()
    depth = np.full((480, 640), 1000 + frame_id, dtype=np.uint16).tobytes()
    imu_samples = [
        {
            "timestamp": timestamp + i * 0.005,
            "domain": "hardware_clock",
            "accel": [0.0, 9.81, 0.0],
            "gyro": [0.0, 0.0, 0.0],
        }
        for i in range(num_imu_samples)
    ]
    header = {
        "protocol_version": 2,
        "session_id": "test_buffer_session",
        "frame_id": frame_id,
        "color": {
            "timestamp": timestamp,
            "timestamp_domain": "hardware_clock",
            "width": 640,
            "height": 480,
            "format": "bgr8",
            "channels": 3,
            "bytes_per_channel": 1,
            "payload_size": len(rgb),
            "fx": 385.0,
            "fy": 385.0,
            "ppx": 320.0,
            "ppy": 240.0,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "distortion_model": "plumb_bob",
        },
        "depth": {
            "timestamp": timestamp,
            "timestamp_domain": "hardware_clock",
            "width": 640,
            "height": 480,
            "format": "z16",
            "channels": 1,
            "bytes_per_channel": 2,
            "payload_size": len(depth),
            "depth_scale": 0.001,
        },
        "imu": imu_samples,
    }
    return {
        "header": header,
        "rgb_bytes": rgb,
        "depth_bytes": depth,
        "arrival_ros_sec": timestamp,
        "receive_time_wall": timestamp,
        "receive_time_mono": timestamp,
    }


def test_d455_receiver_latest_frame_dispatch():
    """Verify that D455Receiver drains accumulated queue and publishes ONLY the newest RGB-D frame."""
    receiver = D455Receiver(host="127.0.0.1", port=0, queue_size=5)

    try:
        # Pre-seed the clock mapping with a known baseline
        receiver.clock_mapping.update(source_time=10.0, target_time=100.0)

        # Enqueue 5 frames (IDs 101, 102, 103, 104, 105)
        for f_id in range(101, 106):
            pkt = _create_synthetic_packet(frame_id=f_id, timestamp=10.0 + (f_id - 101) * 0.033, num_imu_samples=2)
            receiver.frame_queue.put_nowait(pkt)

        assert receiver.frame_queue.qsize() == 5

        # Execute dispatch
        receiver._dispatch_queued_frames()

        # Invariant: exactly 1 RGB-D frame was published (the newest, ID 105)
        assert receiver.total_frames_published == 1
        assert receiver.dropped_frames == 4

        # Invariant: ALL 10 IMU samples (5 frames x 2 samples) were published
        assert receiver.imu_count == 10

        # Queue must now be empty
        assert receiver.frame_queue.empty()

        # Check published image content corresponds to the newest frame (frame 105)
        if hasattr(receiver.rgb_pub, "published") and receiver.rgb_pub.published:
            last_img = receiver.rgb_pub.published[-1]
            img_data = bytes(last_img.data)
            assert img_data[0] == 105

    finally:
        receiver.destroy_node()


def test_d455_receiver_overflow_preserves_imu():
    """Verify that when frame_queue overflows, the oldest video frame is dropped but its IMU is preserved."""
    # Capacity 2 queue
    receiver = D455Receiver(host="127.0.0.1", port=0, queue_size=2)

    try:
        receiver.clock_mapping.update(source_time=1.0, target_time=10.0)

        # Push 4 frames into a capacity-2 queue
        for f_id in range(1, 5):
            pkt = _create_synthetic_packet(frame_id=f_id, timestamp=1.0 + f_id * 0.033, num_imu_samples=3)
            # Simulate worker put_nowait with overflow drop
            if receiver.frame_queue.full():
                stale = receiver.frame_queue.get_nowait()
                receiver.dropped_frames += 1
                receiver._publish_imu_only(stale)
            receiver.frame_queue.put_nowait(pkt)

        # 2 frames dropped on overflow
        assert receiver.dropped_frames == 2
        # IMU from the 2 overflowed frames was published
        assert receiver.imu_count == 6

        # Dispatch the 2 remaining queued frames
        receiver._dispatch_queued_frames()

        # Only the newest frame (ID 4) published its video; frame 3 video was dropped during drain
        assert receiver.total_frames_published == 1
        assert receiver.dropped_frames == 3  # 2 on overflow + 1 on drain

        # All 12 IMU samples from all 4 frames were successfully published
        assert receiver.imu_count == 12

    finally:
        receiver.destroy_node()
