# Geometry Architecture

## Transforms
SE(3) transforms and point conversions follow consistent shape conventions:
- Matrices are 4x4.
- Points are Nx3.
- Quaternions are converted to matrices using 	ransforms.quaternion_to_matrix.

## Camera and Depth
Camera intrinsics and depth scales are loaded via the configuration schema (e.g., SceneGraphConfig), avoiding hardcoded values.
Invalid depths (<=0, NaN, Inf) are filtered out during processing.

## Object Geometry
Geometry is computed per observation and stored in ObjectGeometry. The generation process performs filtering (e.g., median depth) to reduce background contamination.

## Explicit Status
Geometry generation yields a GeometryStatus (e.g. VALID, NO_DEPTH, INSUFFICIENT_DEPTH), preventing algorithms from making assumptions about missing 3D coordinates when depth evidence is insufficient.
