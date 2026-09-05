"""FramePacket dataclass and stream builder."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np

from scene_graph.data.synchronization import associate
from scene_graph.data.tum_loader import TUMLoader


@dataclass
class FramePacket:
    """A synchronized, per-frame container for downstream scene graph tasks."""
    frame_index: int            # sequential within range (matches RGB stream index)
    timestamp: float            # RGB timestamp
    rgb: np.ndarray             # (480, 640, 3) uint8 RGB format
    depth: Optional[np.ndarray] # (480, 640) uint16
    pose: Optional[np.ndarray]  # (4, 4) world_T_camera
    has_depth: bool
    has_pose: bool


def build_frame_packets(
    sequence_dir: Union[str, Path],
    start_frame: int = 100,
    end_frame: int = 300,
    max_dt: float = 0.02
) -> List[FramePacket]:
    """Build FramePackets for a specific range of RGB frames.

    Iterates over the RGB stream from `start_frame` to `end_frame` (inclusive),
    syncs depth and pose data, loads the images from disk, and constructs FramePackets.
    Emits packets even if depth or pose are missing.

    Args:
        sequence_dir: Directory containing TUM dataset.
        start_frame: Inclusive start index for RGB stream.
        end_frame: Inclusive end index for RGB stream.
        max_dt: Maximum timestamp difference for association.

    Returns:
        List of FramePacket objects.
    """
    loader = TUMLoader(sequence_dir)
    rgb_entries = loader.load_rgb()
    depth_entries = loader.load_depth()
    pose_entries = loader.load_groundtruth()
    
    # Extract timestamps for matching
    rgb_timestamps = [e.timestamp for e in rgb_entries]
    depth_timestamps = [e.timestamp for e in depth_entries]
    pose_timestamps = [e.timestamp for e in pose_entries]
    
    # Run nearest-neighbour timestamp association against the RGB anchor stream
    rgb_to_depth = dict(associate(rgb_timestamps, depth_timestamps, max_dt))
    rgb_to_pose = dict(associate(rgb_timestamps, pose_timestamps, max_dt))
    
    packets: List[FramePacket] = []
    
    # End frame is inclusive, so we need + 1, bounded by stream length
    safe_end = min(end_frame + 1, len(rgb_entries))
    
    for frame_idx in range(start_frame, safe_end):
        rgb_entry = rgb_entries[frame_idx]
        
        # Load RGB Image
        rgb_path = str(loader.resolve_rgb_path(rgb_entry))
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Failed to load RGB image at {rgb_path}")
        rgb_np = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        
        # Load Depth Image if matched
        depth_np = None
        has_depth = False
        if frame_idx in rgb_to_depth:
            d_idx = rgb_to_depth[frame_idx]
            depth_path = str(loader.resolve_depth_path(depth_entries[d_idx]))
            # IMREAD_ANYDEPTH preserves the 16-bit depth values
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_raw is not None:
                depth_np = depth_raw
                has_depth = True
            
        # Get Pose Matrix if matched
        pose_np = None
        has_pose = False
        if frame_idx in rgb_to_pose:
            p_idx = rgb_to_pose[frame_idx]
            pose_np = pose_entries[p_idx].as_transform_matrix()
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
        packets.append(packet)
        
    return packets
