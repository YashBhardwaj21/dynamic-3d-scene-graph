from collections import deque
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from scene_graph.config import SceneGraphConfig
from scene_graph.perception.observation import Observation
from scene_graph.tracking.track import Track, TrackState
from scene_graph.tracking.tracker_base import TrackerInterface
from scene_graph.tracking.track_history import TrackHistory
from scene_graph.tracking.state import KalmanState


Prediction = Tuple[np.ndarray, np.ndarray]


class CausalTracker(TrackerInterface):

    POSITION_DIMENSION = 3

    def __init__(self, config: SceneGraphConfig, history: Optional[TrackHistory] = None):
        if config is None:
            raise ValueError("SceneGraphConfig is required.")

        if config.tracking is None:
            raise ValueError("Tracking configuration is required.")

        self.config = config
        tracking_config = config.tracking

        self.association_threshold_m = float(tracking_config.association.max_distance_m)
        self.size_weight = float(tracking_config.association.size_weight)
        self.max_missing_seconds = float(tracking_config.occlusion.max_missing_seconds)
        self.min_hits_to_confirm = int(tracking_config.confirmation.min_hits)
        self.q_std = float(tracking_config.process.acceleration_std_mps2)
        self.measurement_std_m = float(tracking_config.measurement.position_std_m)
        self.initial_position_variance = float(tracking_config.initialization.position_variance_m2)
        self.initial_velocity_variance = float(tracking_config.initialization.velocity_variance_m2s2)
        self.chi2_threshold = float(chi2.ppf(tracking_config.gating.chi2_probability, df=self.POSITION_DIMENSION))
        self.history = history

        if self.association_threshold_m <= 0.0:
            raise ValueError("tracking.association.max_distance_m must be positive.")

        if self.size_weight < 0.0:
            raise ValueError("tracking.association.size_weight must be non-negative.")

        if self.max_missing_seconds < 0.0:
            raise ValueError("tracking.occlusion.max_missing_seconds must be non-negative.")

        if self.min_hits_to_confirm < 1:
            raise ValueError("tracking.confirmation.min_hits must be positive.")

        if self.q_std <= 0.0:
            raise ValueError("tracking.process.acceleration_std_mps2 must be positive.")

        if self.measurement_std_m <= 0.0:
            raise ValueError("tracking.measurement.position_std_m must be positive.")

        if self.initial_position_variance <= 0.0:
            raise ValueError("tracking.initialization.position_variance_m2 must be positive.")

        if self.initial_velocity_variance <= 0.0:
            raise ValueError("tracking.initialization.velocity_variance_m2s2 must be positive.")

        if not np.isfinite(self.chi2_threshold) or self.chi2_threshold <= 0.0:
            raise ValueError("tracking.gating.chi2_probability produced an invalid threshold.")

        self.measurement_covariance = np.eye(self.POSITION_DIMENSION, dtype=np.float64) * self.measurement_std_m ** 2
        self.tracks: Dict[str, Track] = {}
        self.next_track_id = 1
        self._last_frame_index: Optional[int] = None
        self._last_timestamp: Optional[float] = None

    def update(self, observations: List[Observation], frame_index: int, timestamp: float) -> List[Track]:
        self._validate_frame(frame_index, timestamp)

        predictions = self._predict_tracks(timestamp)
        matched, unmatched_obs, unmatched_tracks = self._associate(observations, predictions)

        for obs_idx, track_id in matched:
            self._update_track(track_id, observations[obs_idx], frame_index, timestamp)

        for track_id in unmatched_tracks:
            self._mark_track_missing(track_id, frame_index, timestamp)

        for obs_idx in unmatched_obs:
            self._create_track(observations[obs_idx], frame_index, timestamp)

        self._cleanup_lost_tracks()

        self._last_frame_index = frame_index
        self._last_timestamp = timestamp

        return [self.tracks[track_id] for track_id in sorted(self.tracks)]

    def _validate_frame(self, frame_index: int, timestamp: float) -> None:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative.")

        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite.")

        if self._last_frame_index is not None and frame_index < self._last_frame_index:
            raise ValueError(f"Non-monotonic frame index: {frame_index} < {self._last_frame_index}")

        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError(f"Non-monotonic timestamp: {timestamp} < {self._last_timestamp}")

    def _predict_tracks(self, timestamp: float) -> Dict[str, Prediction]:
        predictions: Dict[str, Prediction] = {}

        for track_id, track in self.tracks.items():
            if track.kalman_state is None:
                position = np.asarray(track.centroid_world, dtype=np.float64)
                covariance = track.position_covariance_world

                if covariance is None:
                    covariance = self.measurement_covariance.copy()
                else:
                    covariance = np.asarray(covariance, dtype=np.float64)

                predictions[track_id] = (position.copy(), covariance.copy())
                continue

            dt = timestamp - track.last_timestamp

            if dt < 0.0:
                raise ValueError(f"Non-monotonic timestamp for track {track_id}: {timestamp} < {track.last_timestamp}")

            predicted_state = copy.deepcopy(track.kalman_state)
            predicted_state.predict(dt, self.q_std)

            predictions[track_id] = (
                predicted_state.position,
                predicted_state.position_covariance,
            )

        return predictions

    def _associate(self, observations: List[Observation], predictions: Dict[str, Prediction]) -> Tuple[List[Tuple[int, str]], List[int], List[str]]:
        track_ids = sorted(predictions)

        if not observations or not track_ids:
            return [], list(range(len(observations))), track_ids

        cost_matrix = np.full((len(observations), len(track_ids)), np.inf, dtype=np.float64)

        for obs_idx, observation in enumerate(observations):
            centroid = self._get_observation_centroid(observation)

            if centroid is None:
                continue

            observation_size = self._get_observation_size(observation)
            measurement_covariance = self._get_measurement_covariance(observation)

            for track_idx, track_id in enumerate(track_ids):
                track = self.tracks[track_id]

                if observation.class_name != track.class_name:
                    continue

                predicted_position, predicted_covariance = predictions[track_id]
                innovation = centroid - predicted_position
                distance = float(np.linalg.norm(innovation))

                if not np.isfinite(distance) or distance > self.association_threshold_m:
                    continue

                innovation_covariance = predicted_covariance + measurement_covariance

                if not self._valid_covariance(innovation_covariance):
                    continue

                try:
                    mahalanobis_sq = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
                except np.linalg.LinAlgError:
                    continue

                if not np.isfinite(mahalanobis_sq) or mahalanobis_sq > self.chi2_threshold:
                    continue

                cost = mahalanobis_sq

                if observation_size is not None and track.size_world is not None and self.size_weight > 0.0:
                    size_scale = max(float(np.linalg.norm(track.size_world)), np.finfo(float).eps)
                    normalized_size_error = float(np.linalg.norm(observation_size - track.size_world) / size_scale)
                    cost += self.size_weight * normalized_size_error

                if np.isfinite(cost):
                    cost_matrix[obs_idx, track_idx] = cost

        finite_mask = np.isfinite(cost_matrix)

        if not finite_mask.any():
            return [], list(range(len(observations))), track_ids

        feasible_observations = np.flatnonzero(finite_mask.any(axis=1))
        feasible_tracks = np.flatnonzero(finite_mask.any(axis=0))

        feasible_cost_matrix = cost_matrix[np.ix_(feasible_observations, feasible_tracks)]

        row_indices, col_indices = linear_sum_assignment(feasible_cost_matrix)

        matched: List[Tuple[int, str]] = []
        unmatched_obs = set(range(len(observations)))
        unmatched_tracks = set(track_ids)

        for local_row, local_col in zip(row_indices, col_indices):
            row = int(feasible_observations[local_row])
            col = int(feasible_tracks[local_col])

            if not np.isfinite(cost_matrix[row, col]):
                continue

            track_id = track_ids[col]
            matched.append((row, track_id))
            unmatched_obs.remove(row)
            unmatched_tracks.remove(track_id)

        return matched, sorted(unmatched_obs), sorted(unmatched_tracks)

    def _update_track(self, track_id: str, observation: Observation, frame_index: int, timestamp: float) -> None:
        track = self.tracks[track_id]
        old_state = track.state
        measurement = self._get_observation_centroid(observation)

        if measurement is None:
            return

        measurement_covariance = self._get_measurement_covariance(observation)

        if track.kalman_state is None:
            track.kalman_state = KalmanState(
                measurement,
                initial_cov_pos=self.initial_position_variance,
                initial_cov_vel=self.initial_velocity_variance,
            )
        else:
            dt = timestamp - track.last_timestamp

            if dt < 0.0:
                raise ValueError(f"Non-monotonic timestamp for track {track_id}: {timestamp} < {track.last_timestamp}")

            updated_state = copy.deepcopy(track.kalman_state)
            updated_state.predict(dt, self.q_std)
            updated_state.update(measurement, measurement_covariance)
            track.kalman_state = updated_state

        observation_size = self._get_observation_size(observation)

        if observation_size is not None:
            if track.size_world is None or track.observation_count <= 0:
                track.size_world = observation_size.copy()
            else:
                count = track.observation_count
                track.size_world = ((count * track.size_world) + observation_size) / (count + 1)

        track.last_observed_frame = frame_index
        track.last_timestamp = timestamp
        track.observation_count += 1
        track.missing_count = 0
        track.detection_confidence = float(observation.confidence)

        age = frame_index - track.first_observed_frame + 1
        track.track_observation_ratio = track.observation_count / max(age, 1)
        track.recent_observations.append(observation)

        if track.state == TrackState.CANDIDATE and track.observation_count >= self.min_hits_to_confirm:
            track.state = TrackState.ACTIVE
        elif track.state == TrackState.TEMPORARILY_UNOBSERVED:
            track.state = TrackState.ACTIVE

        if self.history is not None:
            if old_state != track.state:
                self.history.record_transition(track_id, frame_index, old_state.value, track.state.value, "observation_matched")

            self.history.record_observation(track_id, observation)

    def _mark_track_missing(self, track_id: str, frame_index: int, timestamp: float) -> None:
        track = self.tracks[track_id]
        old_state = track.state

        track.missing_count += 1

        age = frame_index - track.first_observed_frame + 1
        track.track_observation_ratio = track.observation_count / max(age, 1)

        missing_seconds = timestamp - track.last_timestamp

        if missing_seconds < 0.0:
            raise ValueError(f"Non-monotonic timestamp for track {track_id}: {timestamp} < {track.last_timestamp}")

        if missing_seconds >= self.max_missing_seconds:
            track.state = TrackState.LOST
        elif track.state in (TrackState.ACTIVE, TrackState.CANDIDATE):
            track.state = TrackState.TEMPORARILY_UNOBSERVED

        if self.history is not None and old_state != track.state:
            self.history.record_transition(track_id, frame_index, old_state.value, track.state.value, "missing_threshold")

    def _create_track(self, observation: Observation, frame_index: int, timestamp: float) -> None:
        measurement = self._get_observation_centroid(observation)

        if measurement is None:
            return

        observation_buffer_size = self.config.tracking.history.observation_buffer_size
        size_world = self._get_observation_size(observation)
        track_id = f"track_{self.next_track_id:04d}"
        self.next_track_id += 1

        track = Track(
            object_id=track_id,
            class_name=observation.class_name,
            state=TrackState.CANDIDATE,
            _initial_centroid=np.copy(measurement),
            last_observed_frame=frame_index,
            first_observed_frame=frame_index,
            observation_count=1,
            missing_count=0,
            detection_confidence=float(observation.confidence),
            track_observation_ratio=1.0,
            last_timestamp=timestamp,
            recent_observations=deque([observation], maxlen=observation_buffer_size),
            kalman_state=KalmanState(
                measurement,
                initial_cov_pos=self.initial_position_variance,
                initial_cov_vel=self.initial_velocity_variance,
            ),
            size_world=size_world,
        )

        if self.min_hits_to_confirm <= 1:
            track.state = TrackState.ACTIVE

        self.tracks[track_id] = track

        if self.history is not None:
            self.history.record_transition(track_id, frame_index, "none", track.state.value, "new_track")
            self.history.record_observation(track_id, observation)

    def _cleanup_lost_tracks(self) -> None:
        lost_track_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.state == TrackState.LOST
        ]

        for track_id in lost_track_ids:
            del self.tracks[track_id]

    def _get_measurement_covariance(self, observation: Observation) -> np.ndarray:
        geometry = getattr(observation, "object_geometry", None)

        if geometry is not None:
            covariance = getattr(geometry, "position_covariance_world", None)

            if covariance is not None:
                covariance = np.asarray(covariance, dtype=np.float64)

                if covariance.shape == (3, 3) and self._valid_covariance(covariance):
                    return self._symmetrize(covariance)

        return self.measurement_covariance.copy()

    @staticmethod
    def _get_observation_centroid(observation: Observation) -> np.ndarray | None:
        geometry = getattr(observation, "object_geometry", None)

        if geometry is None:
            return None

        centroid = getattr(geometry, "centroid_world", None)

        if centroid is None:
            return None

        centroid = np.asarray(centroid, dtype=np.float64)

        if centroid.shape != (3,) or not np.isfinite(centroid).all():
            return None

        return centroid

    @staticmethod
    def _get_observation_size(observation: Observation) -> np.ndarray | None:
        geometry = getattr(observation, "object_geometry", None)

        if geometry is None:
            return None

        minimum = getattr(geometry, "bbox_min_world", None)
        maximum = getattr(geometry, "bbox_max_world", None)

        if minimum is None or maximum is None:
            return None

        minimum = np.asarray(minimum, dtype=np.float64)
        maximum = np.asarray(maximum, dtype=np.float64)

        if minimum.shape != (3,) or maximum.shape != (3,):
            return None

        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            return None

        size = maximum - minimum

        if not np.isfinite(size).all() or np.any(size < 0.0):
            return None

        return size

    @staticmethod
    def _symmetrize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.T)

    @classmethod
    def _valid_covariance(cls, covariance: np.ndarray) -> bool:
        covariance = np.asarray(covariance, dtype=np.float64)

        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            return False

        if not np.isfinite(covariance).all():
            return False

        covariance = cls._symmetrize(covariance)
        eigenvalues = np.linalg.eigvalsh(covariance)

        return np.isfinite(eigenvalues).all() and np.min(eigenvalues) >= -1e-10