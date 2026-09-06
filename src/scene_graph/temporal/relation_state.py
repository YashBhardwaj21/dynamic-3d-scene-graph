from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from scene_graph.config import RelationTemporalConfig
from scene_graph.relations.evidence import EvidenceResult, RelationEvidence


RelationKey = Tuple[str, str, str]


class RelationState(Enum):
    HYPOTHESIZED = "hypothesized"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class RelationStateMachine:
    def __init__(self, config: RelationTemporalConfig):
        self.config = config

        self.states: Dict[RelationKey, RelationState] = {}
        self.beliefs: Dict[RelationKey, float] = {}
        self.last_update_times: Dict[RelationKey, float] = {}

    @staticmethod
    def _select_evidence(
        evidences: List[RelationEvidence],
    ) -> Dict[RelationKey, RelationEvidence]:
        selected: Dict[RelationKey, RelationEvidence] = {}

        for evidence in evidences:
            key = (
                evidence.subject_id,
                evidence.object_id,
                evidence.predicate,
            )

            current = selected.get(key)

            if current is None:
                selected[key] = evidence
                continue

            current_strength = abs(float(current.confidence))
            candidate_strength = abs(float(evidence.confidence))

            if candidate_strength > current_strength:
                selected[key] = evidence
            elif (
                candidate_strength == current_strength
                and evidence.result == EvidenceResult.SUPPORTED
            ):
                selected[key] = evidence

        return selected

    def _apply_decay(
        self,
        key: RelationKey,
        timestamp: float,
    ) -> None:
        last_update = self.last_update_times.get(key)

        if last_update is None:
            self.beliefs.setdefault(key, 0.0)
            return

        dt = timestamp - last_update

        if dt < 0:
            raise ValueError(
                f"Non-monotonic timestamp for relation {key}: "
                f"{timestamp} < {last_update}"
            )

        if dt == 0:
            return

        belief = self.beliefs.get(key, 0.0)
        decay = self.config.decay_per_second * dt

        if belief > 0.0:
            belief = max(0.0, belief - decay)
        elif belief < 0.0:
            belief = min(0.0, belief + decay)

        self.beliefs[key] = belief

    @staticmethod
    def _evidence_delta(evidence: RelationEvidence) -> float:
        confidence = float(evidence.confidence)

        if not np.isfinite(confidence):
            raise ValueError(
                f"Non-finite evidence confidence for relation "
                f"({evidence.subject_id}, "
                f"{evidence.predicate}, "
                f"{evidence.object_id})"
            )

        confidence = max(0.0, min(1.0, confidence))

        if evidence.result == EvidenceResult.SUPPORTED:
            return confidence

        if evidence.result == EvidenceResult.CONTRADICTED:
            return -confidence

        return 0.0

    def _update_state(
        self,
        key: RelationKey,
    ) -> RelationState:
        belief = self.beliefs.get(key, 0.0)

        if belief >= self.config.confirmation_threshold:
            return RelationState.SUPPORTED

        if belief <= self.config.contradiction_threshold:
            return RelationState.CONTRADICTED

        if belief > 0.0:
            return RelationState.HYPOTHESIZED

        return RelationState.UNKNOWN

    def update(
        self,
        evidences: List[RelationEvidence],
        frame_index: int,
        timestamp: float,
        active_object_ids: Optional[Set[str]] = None,
    ) -> Dict[RelationKey, RelationState]:

        del frame_index

        frame_evidence = self._select_evidence(evidences)

        existing_keys = set(self.states)
        observed_keys = set(frame_evidence)

        all_keys = sorted(
            existing_keys | observed_keys,
            key=lambda key: (key[0], key[1], key[2]),
        )

        keys_to_remove: List[RelationKey] = []

        for key in all_keys:
            subject_id, object_id, _ = key

            if active_object_ids is not None:
                if (
                    subject_id not in active_object_ids
                    or object_id not in active_object_ids
                ):
                    keys_to_remove.append(key)
                    continue

            self._apply_decay(key, timestamp)

            evidence = frame_evidence.get(key)

            if evidence is not None:
                delta = self._evidence_delta(evidence)

                belief = self.beliefs.get(key, 0.0)
                self.beliefs[key] = float(
                    np.clip(
                        belief + delta,
                        -1.0,
                        1.0,
                    )
                )

                self.last_update_times[key] = timestamp

            state = self._update_state(key)

            if evidence is None:
                last_update = self.last_update_times.get(key)

                if last_update is not None:
                    missing_time = timestamp - last_update

                    if missing_time >= self.config.lost_after_seconds:
                        keys_to_remove.append(key)
                        continue

                    if missing_time >= self.config.unknown_after_seconds:
                        state = RelationState.UNKNOWN

            self.states[key] = state

        for key in keys_to_remove:
            self.states.pop(key, None)
            self.beliefs.pop(key, None)
            self.last_update_times.pop(key, None)

        return dict(self.states)