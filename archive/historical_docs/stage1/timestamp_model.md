# Stage 1: Timestamp Model & Clock Mapping

## 1. Explicit Timestamp Domains

The Stage 1 architecture distinguishes 4 authoritative clock domains (`src/scene_graph/data/timestamp.py`):
1. **`HARDWARE_CLOCK`**: RealSense onboard ASIC / oscillator clock. High relative accuracy, but starts from zero on camera boot and is uncalibrated to host wall time.
2. **`SYSTEM_TIME`**: Operating system host clock (e.g. Windows `time.time()` or Linux `CLOCK_REALTIME`). Subject to NTP adjustments and virtualization drift.
3. **`GLOBAL_TIME`**: RealSense SDK global time protocol estimating host-camera synchronization.
4. **`SIMULATED_TIME`**: ROS `/clock` or dataset replay time (e.g. TUM 2011 timestamps).

## 2. Preventing Silent Domain Mixing

The `Timestamp` class enforces domain compatibility:
- Comparisons (`<`, `<=`, `>`, `>=`) and differences (`delta_to`) between incompatible domains raise `IncompatibleTimestampDomainError`.
- Raw floats cannot be compared without explicit domain provenance.

## 3. Windows -> WSL2 Clock Mapping (`ClockMapping`)

### Problem Definition
In our Windows $\rightarrow$ WSL2 bridge deployment, the Windows RealSense hardware clock runs in a different time domain from the WSL2 Linux kernel time. Measurements showed an empirical offset of approximately $\approx 0.916\text{ s}$ plus linear drift.
Directly using `self.get_clock().now()` destroyed hardware frame timing and caused jitter. Directly using the raw Windows sensor timestamp caused TF lookups in ROS to fail because $t_{\text{tf}} - t_{\text{sensor}} > \text{max\_age}$.

### Solution Architecture
`ClockMapping` implements an online exponential moving average filter tracking:
$$\Delta_{\text{offset}} = t_{\text{ros\_arrival}} - t_{\text{sensor\_timestamp}}$$
$$\Delta_{\text{smoothed}} = (1 - \alpha) \Delta_{\text{smoothed}} + \alpha \Delta_{\text{raw}}$$
$$\text{Mapped ROS Timestamp} = t_{\text{sensor}} + \Delta_{\text{smoothed}}$$

### Health & Discontinuity Handling
- **Jump Detection:** If $|\Delta_{\text{raw}} - \Delta_{\text{smoothed}}| > 0.5\text{ s}$ (e.g. clock step, camera reset), the filter is marked **UNHEALTHY** and resets.
- **Stale Detection:** If no packets arrive for $> 2.0\text{ s}$, the mapping becomes stale and degraded.
- **Degraded Fallback:** While unhealthy, ingestion continues safely using local arrival timestamps without process termination.
