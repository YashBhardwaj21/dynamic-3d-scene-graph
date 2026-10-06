# Stage 1: Latency Budget & Bounded Buffering

## 1. Latency Objective: Freshness over Throughput

In robotic perception and dynamic scene graph generation, **low and bounded sensor age** is paramount over maximum throughput. Processing a 3-second-old frame causes tracking failure, spatial misregistration, and causal relation corruption.

## 2. Queue Audit and Latency Bounds

| Queue Location | Capacity | Type | Drop Policy | Max Age (@ 30 FPS) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sender Transport Queue** | 1 frame | `queue.Queue` | Drop oldest | $\approx 33\text{ ms}$ | Decouples USB capture from TCP send; network hiccups cannot block camera. |
| **Receiver Socket Queue** | 2 frames | `queue.Queue` | Drop oldest | $\approx 66\text{ ms}$ | Buffers between network receive thread and ROS publication timer with minimal latency accumulation. |
| **ROS Message Synchronizer** | 4 frames | `ApproximateTimeSynchronizer` | Drop oldest | $\approx 133\text{ ms}$ | Replaces previous 30-frame synchronizer that accumulated 1.0s of stale backlog. |
| **Ingestion Worker Queue** | 4 frames | `queue.Queue` | Drop oldest (`drop_old_frames=True`) | $\approx 133\text{ ms}$ | Replaces previous 64-frame queue that accumulated >2.1s of backlog behind YOLO. |
| **IMU History Buffer** | 500 samples | `collections.deque` | FIFO (automatic) | $\approx 2.5\text{ s}$ (@ 200 Hz) | Windowed extraction for frame intervals; $O(1)$ append without memory copies. |

## 3. Measured Stage 1 Latency Budget (Monotonic Clocks)

Stage 1 instrumentation uses `time.perf_counter()` to record percentiles:
- **`wait_for_frames` latency (P50/P95):** RealSense USB acquisition block time ($\approx 33\text{ ms}$ between frames).
- **`align.process` latency (P50/P95):** Hardware-accelerated depth-to-color alignment ($\approx 3\text{ - }6\text{ ms}$).
- **`sendall` latency (P50/P95):** TCP transmission across Hyper-V vSwitch ($\approx 1\text{ - }3\text{ ms}$ with `TCP_NODELAY`).
- **`decode` latency (P50/P95):** Header parsing and payload extraction ($\approx 0.5\text{ ms}$).
