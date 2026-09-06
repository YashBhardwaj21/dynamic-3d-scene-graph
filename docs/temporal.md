# Temporal Architecture

## Causal Tracking
The tracking subsystem is time-aware, updating predictions based on actual 	imestamp deltas rather than sequential frame indices.
Uncertainty and kinematic states are modeled across time. The Hungarian matching algorithm is used for robust temporal association of bounding boxes across frames.

## Relation State Transitions
Relation evaluations return one of four explicit evidence states:
- SUPPORTED
- CONTRADICTED
- UNKNOWN
- NOT_APPLICABLE

Missing geometry (UNKNOWN) does not immediately terminate an active relation. The temporal state machine gracefully transitions between CANDIDATE, ACTIVE, TEMPORARILY_UNOBSERVED, and LOST based on cumulative evidence counts over time.

## Graph Semantics
The logical graph node state (GraphParticipationState) is decoupled from the underlying causal tracker state. Graph edges retain full temporal history, preserving the number of consecutive supporting or contradicting frames.
