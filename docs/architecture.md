# Dynamic Scene Graph Architecture

## Overview

The Dynamic Scene Graph is built on a strict, feed-forward, dataset-independent architecture. It processes sequential geometric observations to build a spatiotemporal graph of objects and their relations.

## Invariants

The pipeline must follow this exact sequence:

1. **Dataset-Specific Code** -> `FrameSource`
2. **Generic Container** -> `FramePacket`
3. **Perception** -> `ObservationProducer` -> `Observation`
4. **Geometry** -> `GeometryProcessor` -> `ObservationGeometry`
5. **Tracking** -> `CausalTracker` -> `Track`
6. **Relation Inference** -> `RelationRegistry` -> `RelationEvidence`
7. **Temporal Logic** -> `TemporalState`
8. **Final Structure** -> `DynamicSceneGraph`

## Strict Boundaries

1. **No Downstream Dataset Knowledge**: No component below `FrameSource` may import dataset-specific loaders or configs.
2. **No Reverse Dependencies**: The tracker cannot access future frames. The relations cannot directly access raw RGB-D sensors.
3. **No Mocks for Relations**: Test environments must supply synthetic geometry, never mock implementations of the mathematical relations themselves.
