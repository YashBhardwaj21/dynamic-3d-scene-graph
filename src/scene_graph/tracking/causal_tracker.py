import collections
from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from scene_graph.config import SceneGraphConfig
from scene_graph.perception.observation import Observation
from scene_graph.tracking.track import Track, TrackState
from scene_graph.tracking.tracker_base import TrackerInterface
from scene_graph.tracking.track_history import TrackHistory
from scene_graph.tracking.state import KalmanState


class CausalTracker(TrackerInterface):
    """Causal, global nearest-neighbor tracker using Hungarian assignment."""
    
    def __init__(self, config: SceneGraphConfig):
        self.config = config
        
        self.association_threshold_m = self.config.tracking.association.max_distance_m
        self.size_weight = self.config.tracking.association.size_weight
        
        self.max_missing_seconds = self.config.tracking.occlusion.max_missing_seconds
        self.min_hits_to_confirm = self.config.tracking.confirmation.min_hits
        
        self.q_std = self.config.tracking.process.acceleration_std_mps2
        self.r_cov = np.eye(3) * (self.config.tracking.measurement.position_std_m ** 2)
        
        # 99.9% chi-square for 3 DOF is 16.27
        from scipy.stats import chi2
        self.chi2_threshold = chi2.ppf(self.config.tracking.gating.chi2_probability, df=3)
        
        self.tracks: Dict[str, Track] = {}
        self.next_track_id = 1
        
        # External history storage
        self.history = None
        # We don't have experiment.save_intermediates in strict config.
        # To avoid breaking if missing, we check if it was loaded.
        # But for the core pipeline, we can just skip it unless passed explicitly.
        pass

    def update(self, observations: List[Observation], frame_index: int, timestamp: float) -> List[Track]:
        """Update tracks with new observations for this frame."""
        
        # 1. Predict track positions (velocity or constant position)
        predictions = self._predict_tracks(timestamp)
        
        # 2. Hungarian Association
        matched, unmatched_obs, unmatched_tracks = self._associate(
            observations, predictions, timestamp
        )
        
        # 3. Update matched tracks
        for obs_idx, track_id in matched:
            self._update_track(track_id, observations[obs_idx], frame_index, timestamp)
            
        # 4. Handle unmatched tracks (missing)
        for track_id in unmatched_tracks:
            self._mark_track_missing(track_id, frame_index, timestamp)
            
        # 5. Handle unmatched observations (new tracks)
        for obs_idx in unmatched_obs:
            self._create_track(observations[obs_idx], frame_index, timestamp)
            
        # 6. Delete LOST tracks
        self._cleanup_lost_tracks(frame_index)
        
        # Return non-lost tracks
        return list(self.tracks.values())

    def _predict_tracks(self, timestamp: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Predict the location and covariance of tracks at the current frame."""
        predictions = {}
        for track_id, track in self.tracks.items():
            dt = timestamp - track.last_timestamp
            
            # Predict Position and Covariance
            if track.kalman_state is not None:
                track.kalman_state.predict(dt, self.q_std)
                pred_pos = track.kalman_state.position
                pred_cov = track.kalman_state.position_covariance
            else:
                pred_pos = track._initial_centroid
                pred_cov = np.eye(3) * 0.1
                
            predictions[track_id] = (pred_pos, pred_cov)
                
        return predictions

    def _associate(self, observations: List[Observation], predictions: Dict[str, Tuple[np.ndarray, np.ndarray]], timestamp: float) -> Tuple[List[Tuple[int, str]], List[int], List[str]]:
        """Associate observations to predicted track positions."""
        track_ids = list(self.tracks.keys())
        
        if not observations or not track_ids:
            return [], list(range(len(observations))), track_ids
            
        # Cost matrix: rows = observations, cols = tracks
        cost_matrix = np.full((len(observations), len(track_ids)), 1e9)
        
        for i, obs in enumerate(observations):
            if not getattr(obs, 'object_geometry', None) or obs.object_geometry.centroid_world is None:
                continue # Cannot associate observations without 3D geometry
                
            # Estimate size
            obs_size = None
            if obs.object_geometry.bbox_max_world is not None and obs.object_geometry.bbox_min_world is not None:
                obs_size = obs.object_geometry.bbox_max_world - obs.object_geometry.bbox_min_world
                
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                
                # Semantic class MUST match
                if obs.class_name != track.class_name:
                    continue
                    
                pred_pos, pred_cov = predictions[track_id]
                diff = obs.object_geometry.centroid_world - pred_pos
                dist = np.linalg.norm(diff)
                
                try:
                    cov_inv = np.linalg.inv(pred_cov)
                    mahalanobis_sq = diff.T @ cov_inv @ diff
                except np.linalg.LinAlgError:
                    mahalanobis_sq = dist * dist / 0.1
                
                # Mahalanobis Chi-Sq gating
                if mahalanobis_sq <= self.chi2_threshold or dist <= self.association_threshold_m:
                    cost = dist  # base cost is euclidean distance
                    
                    if self.size_weight > 0 and obs_size is not None and track.size_world is not None:
                        size_diff = np.linalg.norm(obs_size - track.size_world)
                        cost += self.size_weight * size_diff
                        
                    cost_matrix[i, j] = cost
                    
        # Apply Hungarian algorithm
        # Linear sum assignment minimizes the total cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched = []
        unmatched_obs = set(range(len(observations)))
        unmatched_tracks = set(track_ids)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e9:
                track_id = track_ids[c]
                matched.append((r, track_id))
                unmatched_obs.remove(r)
                unmatched_tracks.remove(track_id)
                
        return matched, list(unmatched_obs), list(unmatched_tracks)

    def _update_track(self, track_id: str, obs: Observation, frame_index: int, timestamp: float):
        track = self.tracks[track_id]
        old_state = track.state
        
        # Update Kalman Filter
        if track.kalman_state is not None:
            if getattr(obs, 'object_geometry', None) and obs.object_geometry.centroid_world is not None:
                track.kalman_state.update(obs.object_geometry.centroid_world, self.r_cov)
        else:
            # Initialize Kalman State if it wasn't already (e.g. if this is the first real update)
            if getattr(obs, 'object_geometry', None) and obs.object_geometry.centroid_world is not None:
                track.kalman_state = KalmanState(obs.object_geometry.centroid_world)
            
        # Update Size
        obs_size = None
        if getattr(obs, 'object_geometry', None) and obs.object_geometry.bbox_max_world is not None and obs.object_geometry.bbox_min_world is not None:
            obs_size = obs.object_geometry.bbox_max_world - obs.object_geometry.bbox_min_world
            
        if obs_size is not None:
            if track.size_world is None:
                track.size_world = obs_size
            else:
                track.size_world = 0.8 * track.size_world + 0.2 * obs_size


        track.last_observed_frame = frame_index
        track.last_timestamp = timestamp
        track.observation_count += 1
        track.missing_count = 0
        track.detection_confidence = obs.confidence
        
        age = frame_index - track.first_observed_frame + 1
        track.track_observation_ratio = track.observation_count / age
        track.recent_observations.append(obs)
        
        # State machine transition
        if track.state == TrackState.CANDIDATE:
            if track.observation_count >= self.min_hits_to_confirm:
                track.state = TrackState.ACTIVE
        elif track.state == TrackState.TEMPORARILY_UNOBSERVED:
            track.state = TrackState.ACTIVE
            
        if self.history and old_state != track.state:
            self.history.record_transition(track_id, frame_index, old_state.value, track.state.value, "observation_matched")
            
        if self.history:
            self.history.record_observation(track_id, obs)

    def _mark_track_missing(self, track_id: str, frame_index: int, timestamp: float):
        track = self.tracks[track_id]
        old_state = track.state
        
        track.missing_count += 1
        age = frame_index - track.first_observed_frame + 1
        track.track_observation_ratio = track.observation_count / age
        
        missing_seconds = timestamp - track.last_timestamp
        
        if missing_seconds >= self.max_missing_seconds:
            track.state = TrackState.LOST
        elif track.state == TrackState.ACTIVE:
            track.state = TrackState.TEMPORARILY_UNOBSERVED
            
        if self.history and old_state != track.state:
            self.history.record_transition(track_id, frame_index, old_state.value, track.state.value, "missing_threshold")

    def _create_track(self, obs: Observation, frame_index: int, timestamp: float):
        if not getattr(obs, 'object_geometry', None) or obs.object_geometry.centroid_world is None:
            return
            
        track_id = f"track_{self.next_track_id:04d}"
        self.next_track_id += 1
        
        track = Track(
            object_id=track_id,
            class_name=obs.class_name,
            state=TrackState.CANDIDATE,
            _initial_centroid=np.copy(obs.object_geometry.centroid_world),
            last_observed_frame=frame_index,
            first_observed_frame=frame_index,
            observation_count=1,
            missing_count=0,
            detection_confidence=obs.confidence,
            track_observation_ratio=1.0,
            last_timestamp=timestamp,
            recent_observations=collections.deque([obs], maxlen=self.config.tracking.history.observation_buffer_size),
            kalman_state=KalmanState(obs.object_geometry.centroid_world)
        )
        
        # Immediate confirmation if min_hits_to_confirm == 1
        if self.min_hits_to_confirm <= 1:
            track.state = TrackState.ACTIVE
            
        self.tracks[track_id] = track
        
        if self.history:
            self.history.record_transition(track_id, frame_index, "none", track.state.value, "new_track")
            self.history.record_observation(track_id, obs)

    def _cleanup_lost_tracks(self, frame_index: int):
        # We don't remove LOST tracks from self.tracks immediately during the loop
        # so we build a list of IDs to delete.
        to_delete = [t_id for t_id, t in self.tracks.items() if t.state == TrackState.LOST]
        for t_id in to_delete:
            del self.tracks[t_id]
