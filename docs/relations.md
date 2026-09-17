# Relation Architecture

## Canonical Predicates & Categories
The scene graph adheres to an authoritative 12-predicate ontology across 3 physical categories:

### 1. STRUCTURAL
- `SUPPORTED_BY` (Query-time inverse: `SUPPORTS`)
- `INSIDE` (Query-time inverse: `CONTAINS`)
- `ATTACHED_TO` (Symmetric)

### 2. SPATIAL
- `NEAR` (Symmetric)
- `TOUCHING` (Symmetric)
- `LEFT_OF` (Inverse: `RIGHT_OF`)
- `RIGHT_OF` (Inverse: `LEFT_OF`)
- `ABOVE` (Inverse: `BELOW`)
- `BELOW` (Inverse: `ABOVE`)
- `FRONT_OF` (Inverse: `BEHIND`)
- `BEHIND` (Inverse: `FRONT_OF`)

### 3. VISIBILITY
- `OCCLUDES` (Inverse: `OCCLUDED_BY`)

Legacy aliases (`ON`, `UNDER`, `CONTAINING`, `IN_FRONT_OF`, `OCCLUDING`) are transparently normalized to their canonical counterparts.


## Canonical Estimators
Primary relations are evaluated by dedicated estimators:
- DistanceEstimator
- SupportEstimator
- ContainmentEstimator
- DirectionalEstimator
- DepthOrderEstimator
- OcclusionEstimator

The remaining relations are seamlessly derived via inverse_algebra.py, ensuring logical consistency (e.g., inverse(inverse(R)) == R).

## Reference Frames
Spatial and directional relationships are evaluated using a RelationReferenceFrame. This abstracts the coordinate system into semantic directions: origin, up_axis, horizontal_axis, and depth_axis.

## Admissibility
Relations maintain strict semantic admissibility checks to prevent illogical evaluations (e.g., checking if a person is "inside" a cup). Admissibility is decoupled from geometric evidence.
