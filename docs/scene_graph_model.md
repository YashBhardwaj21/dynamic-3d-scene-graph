# Canonical Scene Graph Semantic Model & Ontology

## 1. Core Abstractions: Distinction of Roles

To prevent identity collapse and track-instability, the world model distinguishes three temporal concepts:

1. **Observation**: A single, instantaneous sensor/detector measurement in one frame (e.g. 2D bounding box, mask RLE, raw detector confidence, open-vocabulary label hypothesis). Observations are transient and stateless.
2. **Track**: A short-term causal identity estimate produced by recursive filtering (e.g. Kalman filter state, 3D velocity, bounding box extent, covariance, hits/misses). A track is tied to an observation window; a detector dropout or occlusion causes track misses.
3. **PersistentEntity**: A long-lived world-model identity that persists across detector dropouts, track drops, occlusions, and viewpoints. An entity accumulates semantic hypotheses over time, retains historical context relations, and transitions through lifecycle states.

```
Observation (Frame t) ──> Track (Causal Filter) ──> PersistentEntity (World Model)
```

---

## 2. Entity Schema

Every entity in the scene graph adheres to the canonical schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `entity_id` | `str` | Unique persistent identifier (e.g. `entity_000042`, `desk_0004`). |
| `entity_type` | `EntityType` | Coarse semantic category: `WORLD`, `PLACE`, `SURFACE`, `CONTAINER`, `OBJECT`, `AGENT`, `ROBOT`. |
| `semantic_hypotheses` | `List[SemanticHypothesis]` | Ranked open-vocabulary label hypotheses with confidence and detector provenance. |
| `pose` | `np.ndarray` (4x4) | 3D homogeneous world transformation matrix $[R \mid t]$. |
| `geometry` | `Optional[ObjectGeometry]` | Detailed 3D geometric representation (OBB, AABB, point cloud, planar hull). |
| `velocity` | `np.ndarray` (3,) | 3D translational velocity in world frame [m/s]. |
| `uncertainty` | `np.ndarray` (3x3) | Spatial position covariance in world coordinates [m²]. |
| `first_seen` | `float` | Timestamp of first initial confirmation in the world model [s]. |
| `last_seen` | `float` | Timestamp when the entity was last confirmed or updated [s]. |
| `last_observed` | `float` | Timestamp when a direct sensor observation was associated [s]. |
| `visibility_state` | `VisibilityState` | Sensor visibility status: `VISIBLE`, `OCCLUDED`, `OUT_OF_VIEW`, `UNKNOWN`. |
| `lifecycle_state` | `EntityLifecycleState` | Persistence status: `TENTATIVE`, `CONFIRMED`, `VISIBLE`, `OCCLUDED`, `STALE`, `LOST`, `ARCHIVED`. |
| `source_track_ids` | `List[str]` | History of sensor track IDs associated with this persistent entity. |
| `parent_context_id` | `Optional[str]` | ID of the enclosing or supporting context entity in the hierarchy. |
| `attributes` | `Dict[str, Any]` | Extensible metadata (color, material, affordance, physical properties). |

### Entity Types
- **WORLD**: Global reference origin (Level 0 context).
- **PLACE**: Spatial region, room, or functional zone (e.g. `office_1`, `hallway`).
- **SURFACE**: Planar horizontal support anchor (e.g. `desk_4`, `countertop`).
- **CONTAINER**: 3D bounding receptacle enclosing items (e.g. `drawer_1`, `bin_3`).
- **OBJECT**: Physical manipulable or discrete item (e.g. `mug_17`, `laptop_2`).
- **AGENT**: Human or other dynamic agent in the environment.
- **ROBOT**: The perceiving robotic platform and kinematic base.

---

## 3. Relation Edge Schema

Every relational edge between two entities adheres to the canonical schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `relation_id` | `str` | Unique relation identifier (e.g. `entity_17_SUPPORTED_BY_desk_4`). |
| `subject_entity_id` | `str` | Entity ID of the subject entity $A$. |
| `predicate` | `str` | Authoritative canonical predicate name. |
| `object_entity_id` | `str` | Entity ID of the target/object entity $B$. |
| `state` | `RelationState` | Temporal belief state: `CANDIDATE`, `CONFIRMED`, `WEAKENING`, `OCCLUDED`, `UNKNOWN`, `CONTRADICTED`, `ENDED`. |
| `confidence` | `float` | Continuous confidence score $\in [0, 1]$. |
| `uncertainty` | `float` | Metric uncertainty score $\in [0, 1]$. |
| `first_confirmed` | `float` | Timestamp when the relation first achieved `CONFIRMED` state [s]. |
| `last_confirmed` | `float` | Timestamp when evidence was last confirmed [s]. |
| `last_evidence` | `Optional[RelationEvidence]` | Most recent geometric evidence packet supporting or evaluating this edge. |
| `source_estimators` | `List[str]` | Names of geometric estimators that evaluated this relation. |
| `evidence_summary` | `Dict[str, Any]` | Quantitative metrics (contact distance, overlap ratio, normal cosine). |
| `frame_id` | `int` | Frame index of the latest update. |
| `world_timestamp` | `float` | Timestamp of the latest update [s]. |
| `attributes` | `Dict[str, Any]` | Extensible relation metadata. |

---

## 4. Authoritative Relation Categories & Predicates

The repository defines exactly 12 canonical predicates organized into 3 physical categories:

### A. STRUCTURAL (Physical Support & Containment)
1. **`SUPPORTED_BY`**: Subject $A$ is physically supported by surface/object $B$ (gravity aligned, contact plane, footprint overlap).
2. **`INSIDE`**: Subject $A$ is spatially contained within the 3D volume or interior bounds of container $B$.
3. **`ATTACHED_TO`**: Subject $A$ is physically affixed or mechanically coupled to entity $B$.

### B. SPATIAL (Metric & Directional Proximity)
4. **`NEAR`**: Subject $A$ is within scale-aware metric proximity of object $B$ (bounding surface distance $\le d_{\text{near}}$).
5. **`TOUCHING`**: Subject $A$ and object $B$ are in physical geometric contact without supporting relationship.
6. **`LEFT_OF`**: Subject $A$ is positioned to the left of object $B$ in the active reference frame.
7. **`RIGHT_OF`**: Subject $A$ is positioned to the right of object $B$ in the active reference frame.
8. **`ABOVE`**: Subject $A$ is positioned vertically higher than object $B$ along the reference up-axis.
9. **`BELOW`**: Subject $A$ is positioned vertically lower than object $B$ along the reference up-axis.
10. **`FRONT_OF`**: Subject $A$ is positioned ahead of object $B$ along the reference heading/depth axis.
11. **`BEHIND`**: Subject $A$ is positioned behind object $B$ along the reference heading/depth axis.

### C. VISIBILITY (Sensor Raycast & Frustum)
12. **`OCCLUDES`**: Subject $A$ intercepts the camera line-of-sight towards target $B$, partially or fully occluding it.

---

## 5. Canonical Direction & Query-Time Inverse Rules

The graph stores **only canonical relations** to guarantee graph sparsity and prevent contradictory edge mutations. Inverse relations are **derived at query time**:

| Stored Canonical Relation | Query-Time Inverse | Algebra Rule |
| :--- | :--- | :--- |
| `SUPPORTED_BY(A, B)` | `SUPPORTS(B, A)` | $B \text{ supports } A \iff A \text{ supported by } B$ |
| `INSIDE(A, B)` | `CONTAINS(B, A)` | $B \text{ contains } A \iff A \text{ inside } B$ |
| `ATTACHED_TO(A, B)` | `ATTACHED_TO(B, A)` | Symmetric physical attachment |
| `NEAR(A, B)` | `NEAR(B, A)` | Symmetric metric proximity |
| `TOUCHING(A, B)` | `TOUCHING(B, A)` | Symmetric surface contact |
| `LEFT_OF(A, B)` | `RIGHT_OF(B, A)` | Directional horizontal complement |
| `RIGHT_OF(A, B)` | `LEFT_OF(B, A)` | Directional horizontal complement |
| `ABOVE(A, B)` | `BELOW(B, A)` | Vertical axial complement |
| `BELOW(A, B)` | `ABOVE(B, A)` | Vertical axial complement |
| `FRONT_OF(A, B)` | `BEHIND(B, A)` | Depth axial complement |
| `BEHIND(A, B)` | `FRONT_OF(B, A)` | Depth axial complement |
| `OCCLUDES(A, B)` | `OCCLUDED_BY(B, A)` | Line-of-sight sensor complement |

Double-inverse property holds universally:
$$\text{inverse}(\text{inverse}(P)) = P$$

---

## 6. Legacy Alias Normalization

For backwards compatibility with existing evaluation modules and tests, the ontology transparently normalizes legacy predicate names:
- `"ON"` $\rightarrow$ `SUPPORTED_BY`
- `"UNDER"` $\rightarrow$ `SUPPORTS`
- `"CONTAINING"` $\rightarrow$ `CONTAINS`
- `"IN_FRONT_OF"` $\rightarrow$ `FRONT_OF`
- `"OCCLUDING"` $\rightarrow$ `OCCLUDES`
