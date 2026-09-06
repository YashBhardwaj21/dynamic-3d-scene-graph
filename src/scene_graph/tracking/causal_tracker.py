import collections
from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from scene_graph.config import SceneGraphConfig
from scene_graph.perception.observation import Observation
from scene_graph.tracking.track import Track, TrackState
from scene_graph.tracking.tracker_base import TrackerInterface
from scene_graph.tracking.track_history import TrackHistory


class CausalTracker(TrackerInterface):
    """Causal, global nearest-neighbor tracker using Hungarian assignment."""
    
    def __init__(self, config: SceneGraphConfig):
        self.config = config
        
        self.association_threshold_m = self.config.get("tracking.association_threshold_m", 0.25)
        self.max_missing_frames = self.config.get("tracking.max_missing_frames", 5)
        self.min_hits_to_confirm = self.config.get("tracking.min_hits_to_confirm", 3)
        self.velocity_history_min = self.config.get("tracking.velocity_history_min", 3)
        
        self.tracks: Dict[str, Track] = {}
        self.next_track_id = 1
        
        # External history storage
        self.history = None
        if self.config.get("experiment.save_intermediates", False):
            self.history = TrackHistory()

    def update(self, observations: List[Observation], frame_index: int) -> List[Track]:
        """Update tracks with new observations for this frame."""
        
        # 1. Predict track positions (velocity or constant position)
        predictions = self._predict_tracks(frame_index)
        
        # 2. Hungarian Association
        matched, unmatched_obs, unmatched_tracks = self._associate(
            observations, predictions
        )
        
        # 3. Update matched tracks
        for obs_idx, track_id in matched:
            self._update_track(track_id, observations[obs_idx], frame_index)
            
        # 4. Handle unmatched tracks (missing)
        for track_id in unmatched_tracks:
            self._mark_track_missing(track_id, frame_index)
            
        # 5. Handle unmatched observations (new tracks)
        for obs_idx in unmatched_obs:
            self._create_track(observations[obs_idx], frame_index)
            
        # 6. Delete LOST tracks
        self._cleanup_lost_tracks(frame_index)
        
        # Return non-lost tracks
        return list(self.tracks.values())

    def _predict_tracks(self, frame_index: int) -> Dict[str, np.ndarray]:
        """Predict the location of tracks at the current frame."""
        predictions = {}
        for track_id, track in self.tracks.items():
            # For now, constant position model unless we have enough history for velocity
            if len(track.recent_observations) >= self.velocity_history_min:
                # Basic velocity estimation (last - oldest) / dt
                # Using simple position differences if timestamps are uniform
                # To keep it causal and simple, we'll stick to a basic EMA or constant position
                pass 
            
            predictions[track_id] = np.copy(track.centroid_world)
            
        return predictions

    def _associate(self, observations: List[Observation], predictions: Dict[str, np.ndarray]) -> Tuple[List[Tuple[int, str]], List[int], List[str]]:
        """Associate observations to predicted track positions."""
        track_ids = list(self.tracks.keys())
        
        if not observations or not track_ids:
            return [], list(range(len(observations))), track_ids
            
        # Cost matrix: rows = observations, cols = tracks
        cost_matrix = np.full((len(observations), len(track_ids)), np.inf)
        
        for i, obs in enumerate(observations):
            if obs.centroid_world is None:
                continue # Cannot associate observations without 3D geometry
                
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                
                # Semantic class MUST match
                if obs.class_name != track.class_name:
                    continue
                    
                predicted_pos = predictions[track_id]
                dist = np.linalg.norm(obs.centroid_world - predicted_pos)
                
                # Distance must be within threshold
                if dist <= self.association_threshold_m:
                    cost_matrix[i, j] = dist
                    
        # Apply Hungarian algorithm
        # Linear sum assignment minimizes the total cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched = []
        unmatched_obs = set(range(len(observations)))
        unmatched_tracks = set(track_ids)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] != np.inf:
                track_id = track_ids[c]
                matched.append((r, track_id))
                unmatched_obs.remove(r)
                unmatched_tracks.remove(track_id)
                
        return matched, list(unmatched_obs), list(unmatched_tracks)

    def _update_track(self, track_id: str, obs: Observation, frame_index: int):
        track = self.tracks[track_id]
        old_state = track.state
        
        track.centroid_world = np.copy(obs.centroid_world)
        track.last_observed_frame = frame_index
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

    def _mark_track_missing(self, track_id: str, frame_index: int):
        track = self.tracks[track_id]
        old_state = track.state
        
        track.missing_count += 1
        age = frame_index - track.first_observed_frame + 1
        track.track_observation_ratio = track.observation_count / age
        
        if track.missing_count >= self.max_missing_frames:
            track.state = TrackState.LOST
        elif track.state == TrackState.ACTIVE:
            track.state = TrackState.TEMPORARILY_UNOBSERVED
            
        if self.history and old_state != track.state:
            self.history.record_transition(track_id, frame_index, old_state.value, track.state.value, "missing_threshold")

    def _create_track(self, obs: Observation, frame_index: int):
        if obs.centroid_world is None:
            return
            
        track_id = f"track_{self.next_track_id:04d}"
        self.next_track_id += 1
        
        track = Track(
            object_id=track_id,
            class_name=obs.class_name,
            state=TrackState.CANDIDATE,
            centroid_world=np.copy(obs.centroid_world),
            last_observed_frame=frame_index,
            first_observed_frame=frame_index,
            observation_count=1,
            missing_count=0,
            detection_confidence=obs.confidence,
            track_observation_ratio=1.0,
            recent_observations=collections.deque([obs], maxlen=10)
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
