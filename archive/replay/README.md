# Archived Replay & Stored Observation Infrastructure

## Overview
This directory contains historical batch processing and cached observation replay utilities:
- `stored_loader.py`: Pre-computed JSON chunk observation loader (`StoredObservationLoader`).
- `offline_pipeline.py`: Batch iterator wrapping `SceneGraphPipeline`.

## When and Why It Existed
Created during early development of relation and temporal state machines to avoid re-running deep learning GPU detection (YOLOE) on every test run.

## Why It Is Archived
In the live system, perception observations are computed dynamically by `YOLOEDetector` on incoming `FramePacket` instances. Stored observation loading must not masquerade as an active sensor source in production.
