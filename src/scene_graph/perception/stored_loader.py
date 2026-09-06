"""Stored Observation Loader (Substage 2.4)."""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from scene_graph.data.frame_packet import FramePacket
from scene_graph.perception.observation_source import ObservationSource
from scene_graph.perception.observation import Observation


class StoredObservationLoader(ObservationSource):
    """Loads pre-computed observations from chunked JSON files in a directory.
    
    This avoids running the GPU detector during relation/temporal development.
    Uses lazy chunk loading to maintain low memory overhead.
    """
    
    def __init__(self, data_dir: str, stream_name: str = "observations_controlled"):
        """Initialize with the path to the processed dataset directory.
        
        Args:
            data_dir: Path to directory containing metadata.json and observations/
            stream_name: The subdirectory containing the observation chunks.
        """
        self.data_dir = Path(data_dir)
        self.obs_dir = self.data_dir / stream_name
        self._frames_cache = {}
        self._loaded_chunks = set()
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load metadata from {metadata_path}: {e}")
            
        self.chunk_files = self.metadata.get("chunks", [])
        
    def _load_chunk(self, chunk_filename: str):
        """Load a specific chunk file into memory."""
        chunk_path = self.obs_dir / chunk_filename
        if not chunk_path.exists():
            print(f"Warning: Chunk file {chunk_path} not found.")
            return
            
        try:
            with open(chunk_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading chunk {chunk_path}: {e}")
            return
            
        for frame_data in data.get("frames", []):
            frame_idx = frame_data["frame_index"]
            obs_list = []
            
            for obs_dict in frame_data.get("observations", []):
                # Convert list back to numpy arrays where necessary
                obs = Observation(
                    obs_id=obs_dict["obs_id"],
                    frame_index=obs_dict["frame_index"],
                    timestamp=obs_dict["timestamp"],
                    class_name=obs_dict["class_name"],
                    confidence=obs_dict["confidence"],
                    bbox_xyxy=np.array(obs_dict["bbox_xyxy"], dtype=np.float32),
                    mask_rle=obs_dict.get("mask_rle"),
                    centroid_camera=np.array(obs_dict["centroid_camera"], dtype=np.float32) if obs_dict.get("centroid_camera") else None,
                    centroid_world=np.array(obs_dict["centroid_world"], dtype=np.float32) if obs_dict.get("centroid_world") else None,
                    bbox_min_world=np.array(obs_dict["bbox_min_world"], dtype=np.float32) if obs_dict.get("bbox_min_world") else None,
                    bbox_max_world=np.array(obs_dict["bbox_max_world"], dtype=np.float32) if obs_dict.get("bbox_max_world") else None,
                    depth_stats=obs_dict.get("depth_stats"),
                    points_world_sampled=np.array(obs_dict["points_world_sampled"], dtype=np.float32) if obs_dict.get("points_world_sampled") else None,
                    valid_point_count=obs_dict.get("valid_point_count", 0),
                    point_cloud_ref=obs_dict.get("point_cloud_ref")
                )
                obs_list.append(obs)
                
            self._frames_cache[frame_idx] = obs_list
            
        self._loaded_chunks.add(chunk_filename)
        
    def detect(self, packet: FramePacket) -> List[Observation]:
        """Match the ObservationProducer API."""
        return self.observations_for_frame(packet.frame_index)
        
    def observations_for_frame(self, frame_index: int) -> List[Observation]:
        """Return the pre-computed observations for the given frame index."""
        
        # If frame is already in cache, return it
        if frame_index in self._frames_cache:
            return self._frames_cache[frame_index]
            
        # Try to find the chunk containing this frame by loading unloaded chunks
        # In a real sequential streaming scenario, it's usually the next chunk
        for chunk_file in self.chunk_files:
            if chunk_file not in self._loaded_chunks:
                self._load_chunk(chunk_file)
                if frame_index in self._frames_cache:
                    return self._frames_cache[frame_index]
                    
        # If we loaded everything and still don't have it, return empty
        return []
        
    def get_model_info(self) -> dict:
        """Return the metadata of the model that generated these observations."""
        return self.metadata.get("detector", {})
