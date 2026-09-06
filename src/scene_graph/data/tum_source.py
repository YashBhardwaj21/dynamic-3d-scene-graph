from typing import Iterator, Optional, Union
from pathlib import Path
import cv2

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.data.frame_source import FrameSource
from scene_graph.data.tum_loader import TUMLoader
from scene_graph.data.synchronization import associate


class TUMReplaySource(FrameSource):
    """A causal source for TUM RGB-D sequences.
    
    Yields FramePackets one by one, ensuring the downstream pipeline
    can only process frames sequentially, matching real-time execution.
    """
    
    def __init__(
        self,
        config: SceneGraphConfig
    ):
        self.config = config
        
        dataset_root = self.config.get("dataset.root")
        if not dataset_root:
            raise ValueError("Config missing dataset.root")
            
        self.loader = TUMLoader(dataset_root)
        
        self.rgb_entries = self.loader.load_rgb()
        self.depth_entries = self.loader.load_depth()
        self.pose_entries = self.loader.load_groundtruth()
        
        # Extract timestamps
        rgb_timestamps = [e.timestamp for e in self.rgb_entries]
        depth_timestamps = [e.timestamp for e in self.depth_entries]
        pose_timestamps = [e.timestamp for e in self.pose_entries]
        
        rgb_depth_max_dt = self.config.get("sync.rgb_depth_max_dt", 0.02)
        rgb_pose_max_dt = self.config.get("sync.rgb_pose_max_dt", 0.02)
        
        # Associate
        self.rgb_to_depth = dict(associate(rgb_timestamps, depth_timestamps, rgb_depth_max_dt))
        self.rgb_to_pose = dict(associate(rgb_timestamps, pose_timestamps, rgb_pose_max_dt))
        
        start_frame = self.config.get("sequence.start_frame")
        end_frame = self.config.get("sequence.end_frame")
        
        self.start_idx = 0 if start_frame is None else start_frame
        self._end_idx = len(self.rgb_entries) - 1 if end_frame is None else end_frame
        self.end_idx = min(self._end_idx, len(self.rgb_entries) - 1)

        if not (0 <= self.start_idx <= self.end_idx < len(self.rgb_entries)):
            raise ValueError(
                f"Invalid sequence bounds: start_frame={self.start_idx}, "
                f"end_frame={self.end_idx}, total_rgb_frames={len(self.rgb_entries)}"
            )
        
    def __iter__(self) -> Iterator[FramePacket]:
        """Yield frame packets one at a time."""
        for frame_idx in range(self.start_idx, self.end_idx + 1):
            rgb_entry = self.rgb_entries[frame_idx]
            
            # Load RGB Image (Mandatory)
            rgb_path = str(self.loader.resolve_rgb_path(rgb_entry))
            rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise FileNotFoundError(f"Failed to load RGB image at {rgb_path}")
            rgb_np = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            
            # Load Depth Image if matched (Optional)
            depth_np = None
            has_depth = False
            if frame_idx in self.rgb_to_depth:
                d_idx = self.rgb_to_depth[frame_idx]
                depth_path = str(self.loader.resolve_depth_path(self.depth_entries[d_idx]))
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_raw is not None:
                    depth_np = depth_raw
                    has_depth = True
                
            # Get Pose Matrix if matched (Optional)
            pose_np = None
            has_pose = False
            if frame_idx in self.rgb_to_pose:
                p_idx = self.rgb_to_pose[frame_idx]
                pose_np = self.pose_entries[p_idx].as_transform_matrix()
                has_pose = True
                
            packet = FramePacket(
                frame_index=frame_idx,
                timestamp=rgb_entry.timestamp,
                rgb=rgb_np,
                depth=depth_np,
                pose=pose_np,
                has_depth=has_depth,
                has_pose=has_pose
            )
            
            yield packet

    def __len__(self) -> int:
        return max(0, self.end_idx - self.start_idx + 1)
