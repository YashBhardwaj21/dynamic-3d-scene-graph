# Stage 1: Testing & Verification Strategy

## 1. Test Organization

Tests are strictly partitioned by test level:
```
tests/stage1/
├── unit/                       # Deterministic in-memory unit tests (no hardware or network)
│   ├── test_sensor_contract.py
│   ├── test_timestamp_contract.py
│   ├── test_timestamp_domains.py
│   ├── test_depth_scale.py
│   ├── test_calibration.py
│   ├── test_imu_contract.py
│   ├── test_synchronization.py
│   ├── test_buffer_policy.py
│   ├── test_packet_validation.py
│   ├── test_transport_protocol.py
│   ├── test_clock_mapping.py
│   └── test_regression_previous_failures.py
├── integration/                # Multi-component and localhost network tests
│   ├── test_d455_sender_receiver.py
│   ├── test_rgb_depth_ingestion.py
│   ├── test_imu_ingestion.py
│   ├── test_reconnect.py
│   ├── test_record_replay_parity.py
│   └── test_ros_ingestion.py
├── system/                     # End-to-end benchmarks and failure recovery
│   ├── test_latency_budget.py
│   ├── test_sensor_failure_recovery.py
│   ├── test_transport_failure_recovery.py
│   ├── test_live_pipeline_startup.py
│   └── hardware/               # Real hardware-in-the-loop tests (D455 required)
│       ├── test_d455_startup.py
│       ├── test_d455_profiles.py
│       ├── test_d455_timestamps.py
│       ├── test_d455_depth_scale.py
│       ├── test_d455_calibration.py
│       └── test_d455_imu.py
└── fixtures/                   # Synthetic deterministic frame & packet generators
    └── synthetic_generator.py
```

## 2. Hardware Test Isolation Rule

Hardware-in-the-loop tests in `tests/stage1/system/hardware/` are strictly isolated:
- They check for `pyrealsense2` and physical USB device presence.
- If hardware is missing, they invoke `pytest.skip()` cleanly with an informative message.
- They **never** fail CI or deterministic test runs due to absent hardware.

## 3. How to Run the Tests

### Stage 1 Unit & Integration Tests:
```bash
pytest tests/stage1/unit tests/stage1/integration tests/stage1/system -v
```

### Full Repository Test Suite:
```bash
pytest tests/ -v
```

### Hardware Validation Run (When D455 is plugged in via USB):
```bash
pytest tests/stage1/system/hardware -v -s
```
