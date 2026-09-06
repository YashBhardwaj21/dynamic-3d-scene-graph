# Evaluation Protocol

## Testing Rules
- No downstream component may use mock implementations of relations.
- Synthetic 3D geometric fixtures must be used to test relations deterministically.

## Tracker Validation
Must prove identity maintenance across:
- Identical objects crossing.
- Variable time deltas.
- Missing frames / temporary disappearance.

## Causality Check
The pipeline must prove that it leaks no future data:
`state_prefix(frame=100) == state_full(frame=100)`

## Dataset Generalization
1. Must successfully run on `configs/tum_fr1_desk.yaml`
2. Must successfully run on `configs/tum_validation.yaml` with ZERO algorithm code changes.
3. Must eventually run on an entirely foreign RGB-D dataset using ONLY a new data loader and calibration adapter.
