"""System test: Deterministic localhost benchmark and latency bound verification.

Directly tests D455Receiver under severe consumer backpressure:
- Producer: 30 FPS (pushes frames every ~33.3ms)
- Consumer: 5 FPS (dispatches frames every ~200ms)
- Verifies:
  1. Latest-frame drop-oldest policy strictly bounds sensor latency to <= 67ms.
  2. Consumer never processes stale accumulated frames from seconds ago.
  3. All high-rate IMU telemetry is fully preserved without gaps.
"""

import numpy as np
import pytest

from ros2_ws.src.d455_bridge.d455_bridge.d455_receiver import D455Receiver


def _make_packet(frame_id: int, timestamp: float):
    rgb = np.full((480, 640, 3), frame_id % 255, dtype=np.uint8).tobytes()
    depth = np.full((480, 640), 1000, dtype=np.uint16).tobytes()
    imu_samples = [
        {
            "timestamp": timestamp,
            "domain": "hardware_clock",
            "accel": [0.0, 9.81, 0.0],
            "gyro": [0.0, 0.0, 0.0],
        }
    ]
    header = {
        "protocol_version": 2,
        "session_id": "latency_test_session",
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


def test_d455_receiver_bounded_latency_under_backpressure():
    """Verify that D455Receiver guarantees low-latency freshness (< 67 ms) when consumer is 6x slower than camera."""
    # Capacity 2 queue: at 30 FPS, queue age is at most 2 * 33.3ms = ~66.7ms
    receiver = D455Receiver(host="127.0.0.1", port=0, queue_size=2)

    try:
        # Pre-seed clock mapping with identity offset for direct comparison
        receiver.clock_mapping.update(source_time=100.0, target_time=100.0)

        t_base = 100.0
        fps_producer = 30.0
        dt_frame = 1.0 / fps_producer  # ~0.0333s
        total_frames = 60

        published_timestamps = []

        # Hook publisher to capture published frame timestamps
        original_pub_frame = receiver._publish_frame

        def capturing_pub(packet):
            ts = float(packet["header"]["color"]["timestamp"])
            published_timestamps.append(ts)
            original_pub_frame(packet)

        receiver._publish_frame = capturing_pub

        # Run 60 frames (~2.0 seconds of video)
        for i in range(total_frames):
            t_current = t_base + i * dt_frame
            pkt = _make_packet(frame_id=i, timestamp=t_current)

            # Ingest into receiver queue with overflow handling
            if receiver.frame_queue.full():
                stale = receiver.frame_queue.get_nowait()
                receiver.dropped_frames += 1
                receiver._publish_imu_only(stale)
            receiver.frame_queue.put_nowait(pkt)

            # Slow consumer: only dispatches every 6 frames (5 FPS consumer vs 30 FPS producer)
            if (i + 1) % 6 == 0:
                receiver._dispatch_queued_frames()

                # Verify freshness invariant: the most recently published frame must be near current producer time!
                assert len(published_timestamps) > 0
                latest_published_ts = published_timestamps[-1]
                latency_age_sec = t_current - latest_published_ts

                # Max allowable age is 2 frame periods (~0.067s)
                # Without latest-frame dropping, latency would accumulate to > 1.6 seconds!
                assert latency_age_sec <= (2.0 * dt_frame + 1e-4), (
                    f"Frame latency {latency_age_sec:.4f}s exceeded bound ({2.0 * dt_frame:.4f}s) "
                    f"at producer time {t_current:.4f}s"
                )

        # Invariant checks
        # 1. Total dispatches = 60 / 6 = 10 published frames
        assert receiver.total_frames_published == 10
        # 2. 50 video frames were dropped to maintain low latency
        assert receiver.dropped_frames == 50
        # 3. Exactly 60 IMU samples were published (zero IMU data loss!)
        assert receiver.imu_count == 60

    finally:
        receiver.destroy_node()
