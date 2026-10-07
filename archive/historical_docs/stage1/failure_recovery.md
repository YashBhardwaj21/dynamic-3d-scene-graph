# Stage 1: Failure Modes and Recovery

## 1. Failure Modes & Mitigations

### 1. Camera Disconnect / USB Timeout
- **Detection:** `pipeline.wait_for_frames(timeout_ms=5000)` throws `rs.error` or timeout exception.
- **Recovery:** Pipeline cleanly stops and releases USB context. Sender enters a bounded 2.0s retry loop waiting for camera reconnection. Upon reconnect, a new `session_id` is created, frame counter resets, and calibration is re-queried. Stale queues are purged.

### 2. TCP Bridge Disconnection / WSL2 Restart
- **Detection:** `sendall` raises `BrokenPipeError` or `ConnectionResetError`; receiver socket reads 0 bytes.
- **Recovery:** Existing socket is closed cleanly. Sender enters a 1.0s backoff loop attempting reconnection via direct Hyper-V vSwitch IP. Receiver remains in non-blocking `accept()` loop. When reconnected, `ClockMapping` resets to prevent stale timing pollution.

### 3. Clock Offset Jump / NTP Synchronization
- **Detection:** `ClockMapping` detects step change $> 0.5\text{ s}$ between source and target time.
- **Recovery:** Filter marks status as **UNHEALTHY** and resets baseline. Downstream continues safely using local arrival timestamps without process crash.

### 4. Malformed / Oversized Packet
- **Detection:** Header length $> 1\text{ MB}$, payload size $> 10\text{ MB}$, or payload size mismatch with image dimensions.
- **Recovery:** Receiver logs structured error, terminates the corrupted connection, and immediately returns to the `accept()` loop without throwing an unhandled exception.

### 5. Downstream Perception Backlog
- **Detection:** Ingestion queue full (`qsize == maxsize`).
- **Recovery:** Drop-oldest policy immediately evicts stale pending frames, ensuring fresh sensor data is handed to perception.
