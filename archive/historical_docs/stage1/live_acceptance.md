# Stage 1: Live Intel RealSense D455 Ingestion Acceptance Criteria

## 1. Acceptance Thresholds & Protected Invariants

To declare Stage 1 live ingestion trustworthy and ready for downstream RTAB-Map and scene-graph processing, the live stream must satisfy all criteria below:

| Metric | Acceptance Threshold | Measurement Source | Rationale |
| :--- | :--- | :--- | :--- |
| **RGB Frame Rate** | $\ge 28.0\text{ FPS}$ (Target: 30 FPS) | `/camera/camera/color/image_raw` | Ensures real-time visual tracking and timely keyframe creation. |
| **Depth Frame Rate** | $\ge 28.0\text{ FPS}$ (Target: 30 FPS) | `/camera/camera/aligned_depth_to_color/image_raw` | Guarantees synchronous metric depth for every processed color frame. |
| **Hardware RGB-Depth Skew** | P50 $< 5\text{ ms}$, P95 $< 15\text{ ms}$, P99 $< 33\text{ ms}$ | `rgb_depth_dt_ms` in `/camera/camera/metadata` | Optical alignment validity; reject any frame exceeding 50 ms tolerance. |
| **Frame Age at Ingestion** | P95 $< 45\text{ ms}$ | Network arrival vs ROS dispatch | Prevents stale frame accumulation in bounded queues (queue=1 sender, queue=2 receiver). |
| **Sequence Continuity** | 0 duplicates, 0 out-of-order, 0 unexpected drops | Sequence counter in metadata | Network protocol framing and queue bounded drop policy integrity. |
| **Runtime Depth Scale** | Authoritative hardware scale ($\approx 0.001\text{ m/unit}$) | `depth_sensor.get_depth_scale()` | Eliminates arbitrary hardcoded 1000.0 assumptions; validated against $0.001 \pm 0.0001\text{ m/unit}$. |
| **Session ID Continuity** | Real session ID preserved end-to-end | `SensorFrame.session_id` | Prevents cross-session contamination and invalid stale temporal state. |
| **Timing Provenance** | Host capture, network arrival, mapped ROS time | `SensorFrame` timestamps | Preserves full audit trail from hardware sensor clock to ROS/TF clock. |
| **Clock Synchronization** | State: HEALTHY, drift $< 100\text{ ppm}$, no jumps | `ClockMapping` in receiver | Sub-millisecond continuous mapping from Windows sensor clock to ROS clock. |

---

## 2. Verification Procedure

Execute the live diagnostic tool:
```bash
python3 tools/check_live_d455.py --duration 10.0
```
or with JSON summary export:
```bash
python3 tools/check_live_d455.py --duration 10.0 --output_json results/stage1_live_report.json
```
