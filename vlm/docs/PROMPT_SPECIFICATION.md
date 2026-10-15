# Structured Affordance Prompt & Inference Contract Specification

**Document Version:** `2.0.0`  
**Template Identifier:** `v2_structured_ontology_schema`  
**Associated Module:** `vlm/src/selective_router.py`  
**Governing Standard:** Rüdt et al., *Contextual Movability & 3D Scene Graphs*

---

## 1. Problem Formulation & Design Principles

The goal of selective VLM inference is **not** to ask the model to re-detect 2D bounding boxes or estimate 3D trajectories. The vision-language model is invoked strictly to resolve **physical affordance ambiguity** when sensory evidence indicates that an object's structural integration with the environment is uncertain (e.g., distinguishing a wall-mounted pedestal fan from a freestanding one, or verifying whether a displaced chair is transportable).

### Core Principles
1. **No Fixed Demonstrations:** Demonstrations with a hardcoded target answer (e.g., `'movability': 'MOVABLE'`) systematically anchor autoregressive decoders toward that class. Prompt template `v2` uses a pure schema and definition specification with zero biasing demonstrations.
2. **Disentanglement of Movability and Motion:**
   - **Inherent Movability (`movability`):** What an object *is* (`IMMOVABLE`, `MOVABLE`, `MOBILE`).
   - **Observed Motion (`observed_motion`):** What an object *did* (`STATIC`, `MOVING`, `DISPLACED`).
   - A stationary chair is still `MOVABLE`. A moving robot is `MOBILE`. A structural door is `IMMOVABLE` regardless of whether it was opened.
3. **Mandatory Identity Validation:** The model must echo and confirm the specific `object_id` and `object_class` under evaluation. If the model hallucinates a different object ID, the prediction is rejected as a mismatch (`validation_error = "object_id_mismatch"`).
4. **Fail-Safe Abstention:** When visual evidence is occluded, cropped out, or degraded, the prompt instructs the model to set `"abstain": true` rather than hallucinating an unsupported label.

---

## 2. Ontology Definitions

| Class | Definition | Inclusion Criteria | Canonical Examples |
| :--- | :--- | :--- | :--- |
| **`IMMOVABLE`** | Fixed architectural infrastructure | Built-in, bolted, structurally hung, or physically integrated into room architecture. | Structural walls, framed passage doors, ceiling fans, structural pillars. |
| **`MOVABLE`** | Passively transportable objects | Objects lacking internal actuation that can be picked up, rolled, or pushed by humans or robot manipulators. | Caster chairs, desks, storage boxes, laptops, backpacks, pedestal fans on bases. |
| **`MOBILE`** | Actively self-moving or transported entities | Entities with intrinsic locomotion or objects mounted on an active mobile robot/carrier. | Mobile robots, autonomous RC cars, humans, drones, objects on mobile robot bases. |

### Permitted Aliases & Normalization
The output parser (`parse_vlm_output`) maps permitted synonyms:
- `IMMOVABLE`: `["FIXED", "STRUCTURAL", "ARCHITECTURAL", "WALL_MOUNTED", "CEILING_MOUNTED", "MOUNTED"]`
- `MOVABLE`: `["PORTABLE", "RELOCATABLE", "FURNITURE", "OBJECT", "FREESTANDING", "FREESTANDING_ON_SURFACE"]`
- `MOBILE`: `["DYNAMIC", "SELF_PROPELLED", "LOCOMOTIVE", "CARRIER", "ROBOT", "AGENT", "MOVING_AGENT"]`
- Any unmapped label (e.g. `FRUIT`, `UNKNOWN`) normalizes to `UNKNOWN` with `abstain = True`.

---

## 3. Neutral Prompt Template Specification

```text
You are an expert robotics perception system evaluating physical affordance and contextual movability for indoor scene understanding.
Target Instance ID: {entity_id}
Expected Semantic Class: {semantic_class}

ONTOLOGY DEFINITIONS (Rüdt et al.):
- IMMOVABLE: Fixed architectural infrastructure (e.g., walls, structural doors, ceiling fans, built-in cabinets). Cannot be relocated without deconstruction.
- MOVABLE: Passively transportable objects lacking intrinsic locomotion (e.g., chairs, desks, boxes, backpacks, laptops). May be currently stationary or displaced.
- MOBILE: Self-propelled agents (e.g., robots, humans) or objects actively carried by a moving platform in the scene.

EVALUATION INSTRUCTIONS:
1. Inspect the visual views of the target instance.
2. Identify physical_support: 'mounted_to_structure', 'freestanding_on_surface', 'wheeled_base', 'carried_by_agent', or 'unknown'.
3. Identify transport_context: 'active_carrier', 'stationary_environment', or 'unknown'.
4. Assign movability strictly as IMMOVABLE, MOVABLE, or MOBILE based on physical affordance, not temporary motion.
5. If visual evidence is severely occluded, corrupted, or missing, set abstain to true.
6. Provide a concise short_explanation citing visual cues.

OUTPUT SCHEMA (Respond ONLY with a valid JSON object matching these fields):
{
  "object_id": "<must match Target Instance ID: {entity_id}>",
  "object_class": "<must match Expected Semantic Class: {semantic_class}>",
  "movability": "<IMMOVABLE | MOVABLE | MOBILE | UNKNOWN>",
  "physical_support": "<mounted_to_structure | freestanding_on_surface | wheeled_base | carried_by_agent | unknown>",
  "transport_context": "<stationary_environment | active_carrier | unknown>",
  "confidence": <float between 0.0 and 1.0>,
  "abstain": <true | false>,
  "short_explanation": "<concise justification>"
}
```

---

## 4. Visual Input Assembly & Decoding Validation

The input validator (`validate_image_inputs`) guarantees:
1. **Configurable Multi-View Inputs:** Supports 1, 2, or 3 composite observations (`crop_view_i` + `scene_view_i`).
2. **Decoding Verification:** Every image file must be successfully verified by PIL before inclusion in base64 payloads. Corrupt files populate `decoding_failures`.
3. **No Text-Only Requests:** If zero usable views are present (`usable_views == 0`), the router immediately aborts inference, returning `DEFER` with `MISSING_REQUIRED_VISUAL_CONTEXT` and `prediction = UNKNOWN`.
4. **Deterministic Ordering:** Views are ordered strictly by camera angle index (`0, 1, 2`).

---

## 5. Cache Fingerprinting & Execution Modes

### Cryptographic Fingerprint
Cache entries are indexed by a 24-character SHA-256 digest:
```python
cache_data = {
    "entity_id": entity_id,
    "semantic_class": sem_class,
    "model_name": self.model_name,
    "prompt_version": PROMPT_TEMPLATE_VERSION,
    "views_to_use": img_telemetry["usable_views"],
    "img_hashes": img_telemetry["image_hashes"],
}
cache_key = hashlib.sha256(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()[:24]
```
Modifying any input image, the prompt version, or the target model automatically produces a distinct key, preventing stale predictions from being reused.

### Replay Mode vs Live Mode
- **`mode = "replay"`:** Evaluates against frozen, pre-recorded predictions. If a cache miss occurs, the system records `vlm_cache_hit = False`, `vlm_inference_executed = False`, `prediction = "UNKNOWN"`, and raises no live HTTP calls.
- **`mode = "live"`:** Invokes the local inference server (`llama-server`), records true wall-clock execution time (`lat_s`), measures token usage, and persists results.
