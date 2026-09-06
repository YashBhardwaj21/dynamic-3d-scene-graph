# Relation Architecture

## Supported Predicates
The relation registry currently supports 14 core spatial predicates:
1. NEAR
2. FAR
3. ON
4. UNDER
5. INSIDE
6. CONTAINING
7. LEFT_OF
8. RIGHT_OF
9. ABOVE
10. BELOW
11. IN_FRONT_OF
12. BEHIND
13. OCCLUDING
14. OCCLUDED_BY

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
