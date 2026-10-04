"""Unit tests for Stage 1 Bounded Buffer Policy (Latest-Frame / Drop-Oldest)."""

import queue

import pytest


def test_latest_frame_drop_oldest_queue_behavior():
    """Verify that a capacity=1 queue always holds the newest frame and drops stale frames."""
    q: queue.Queue = queue.Queue(maxsize=1)
    dropped = 0

    def push_latest(item):
        nonlocal dropped
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                _ = q.get_nowait()
                dropped += 1
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                dropped += 1

    # Simulate fast producer (e.g. 30 FPS camera) vs slow consumer
    for frame_id in range(10):
        push_latest(f"frame_{frame_id}")

    assert q.qsize() == 1
    assert dropped == 9
    assert q.get_nowait() == "frame_9"


def test_maximum_queue_age_bound():
    """Verify latency bound: 4 frames at 30 FPS is at most ~133 ms."""
    fps = 30.0
    frame_interval_sec = 1.0 / fps

    q_capacity = 4
    max_latency_sec = q_capacity * frame_interval_sec
    assert max_latency_sec == pytest.approx(0.1333, rel=1e-2)

    # In contrast, an unbounded 64-frame queue would accumulate >2.1 seconds of stale video
    stale_latency_sec = 64 * frame_interval_sec
    assert stale_latency_sec > 2.0
