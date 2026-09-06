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
        if config is None:
            raise ValueError("RelationTemporalConfig is required.")

        if config.confirmation_threshold <= 0.0:
            raise ValueError("confirmation_threshold must be positive.")

        if config.contradiction_threshold >= 0.0:
            raise ValueError("contradiction_threshold must be negative.")

        if config.decay_per_second < 0.0:
            raise ValueError("decay_per_second must be non-negative.")

        if config.unknown_after_seconds < 0.0:
            raise ValueError("unknown_after_seconds must be non-negative.")

        if config.lost_after_seconds < 0.0:
            raise ValueError("lost_after_seconds must be non-negative.")

        if config.lost_after_seconds < config.unknown_after_seconds:
            raise ValueError("lost_after_seconds must be greater than or equal to unknown_after_seconds.")

        self.config = config
        self.states: Dict[RelationKey, RelationState] = {}
        self.beliefs: Dict[RelationKey, float] = {}
        self.last_update_times: Dict[RelationKey, float] = {}
        self._last_frame_index: Optional[int] = None
        self._last_timestamp: Optional[float] = None

    @staticmethod
    def _select_evidence(evidences: List[RelationEvidence]) -> Dict[RelationKey, RelationEvidence]:
        selected: Dict[RelationKey, RelationEvidence] = {}

        for evidence in evidences:
            if evidence.result not in (EvidenceResult.SUPPORTED, EvidenceResult.CONTRADICTED):
                continue

            key = (evidence.subject_id, evidence.object_id, evidence.predicate)
            current = selected.get(key)

            if current is None:
                selected[key] = evidence
                continue

            current_confidence = float(current.confidence)
            candidate_confidence = float(evidence.confidence)

            if not np.isfinite(current_confidence) or not np.isfinite(candidate_confidence):
                raise ValueError(
                    f"Non-finite evidence confidence for relation {key}"
                )

            if candidate_confidence > current_confidence:
                selected[key] = evidence
            elif candidate_confidence == current_confidence and evidence.result == EvidenceResult.SUPPORTED:
                selected[key] = evidence

        return selected

    def _validate_frame(self, frame_index: int, timestamp: float) -> None:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative.")

        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite.")

        if self._last_frame_index is not None and frame_index < self._last_frame_index:
            raise ValueError(
                f"Non-monotonic frame index: {frame_index} < {self._last_frame_index}"
            )

        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError(
                f"Non-monotonic timestamp: {timestamp} < {self._last_timestamp}"
            )

    def _apply_decay(self, key: RelationKey, timestamp: float) -> None:
        last_update = self.last_update_times.get(key)

        if last_update is None:
            self.beliefs.setdefault(key, 0.0)
            return

        dt = timestamp - last_update

        if dt < 0.0:
            raise ValueError(
                f"Non-monotonic timestamp for relation {key}: {timestamp} < {last_update}"
            )

        if dt == 0.0:
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
                f"({evidence.subject_id}, {evidence.predicate}, {evidence.object_id})"
            )

        confidence = float(np.clip(confidence, 0.0, 1.0))

        if evidence.result == EvidenceResult.SUPPORTED:
            return confidence

        if evidence.result == EvidenceResult.CONTRADICTED:
            return -confidence

        return 0.0

    def _update_state(self, key: RelationKey) -> RelationState:
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

        self._validate_frame(frame_index, timestamp)

        frame_evidence = self._select_evidence(evidences)

        all_keys = sorted(
            set(self.states) | set(frame_evidence),
            key=lambda key: (key[0], key[1], key[2]),
        )

        keys_to_remove: List[RelationKey] = []

        for key in all_keys:
            subject_id, object_id, _ = key

            if active_object_ids is not None and (
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

        self._last_frame_index = frame_index
        self._last_timestamp = timestamp

        return dict(self.states)