from scene_graph.relations.evidence import RelationEvidence


INVERSE = {
    "ON": "UNDER",       "UNDER": "ON",
    "INSIDE": "CONTAINING", "CONTAINING": "INSIDE",
    "LEFT_OF": "RIGHT_OF",  "RIGHT_OF": "LEFT_OF",
    "ABOVE": "BELOW",       "BELOW": "ABOVE",
    "IN_FRONT_OF": "BEHIND", "BEHIND": "IN_FRONT_OF",
    "OCCLUDING": "OCCLUDED_BY", "OCCLUDED_BY": "OCCLUDING",
}

SYMMETRIC = {"NEAR", "FAR"}


def derive_inverse_evidence(evidence: RelationEvidence) -> RelationEvidence:
    """Derive the inverse relation evidence (e.g. A ON B -> B UNDER A)."""
    if evidence.predicate in INVERSE:
        inv_predicate = INVERSE[evidence.predicate]
        inverse_pred = INVERSE[evidence.predicate]
    elif evidence.predicate in SYMMETRIC:
        inverse_pred = evidence.predicate
    else:
        raise ValueError(f"Unknown inverse for predicate: {evidence.predicate}")
        
    return RelationEvidence(
        predicate=inverse_pred,
        subject_id=evidence.object_id,
        object_id=evidence.subject_id,
        frame_index=evidence.frame_index,
        timestamp=evidence.timestamp,
        result=evidence.result,
        value=evidence.value,
        threshold=evidence.threshold,
        confidence=evidence.confidence,
        reference_frame=evidence.reference_frame,
        evidence_type=f"inverse_{evidence.evidence_type}",
        details=evidence.details.copy()
    )


def canonical_pair(id_a: str, id_b: str) -> tuple[str, str]:
    """Return a sorted tuple for symmetric relation cache keys."""
    return (id_a, id_b) if id_a < id_b else (id_b, id_a)
