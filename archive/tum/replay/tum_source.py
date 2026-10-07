from typing import Iterator, Optional, Union
from pathlib import Path
import cv2
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.data.frame_source import FrameSource
from scene_graph.data.sensor_frame import SensorFrame, StreamStatus
from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.estimation.base import BasePoseEstimator
from scene_graph.estimation.ground_truth import GroundTruthEstimator
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.reference_frame import RelationReferenceFrame
from scene_graph.geometry.reference_frame_provider import ReferenceFrameProvider
from scene_graph.data.tum_loader import TUMLoader
from scene_graph.data.synchronization import associate


class TUMReplaySource(FrameSource):
    """A causal source for TUM RGB-D sequences.
    
    Yields FramePackets one by one, ensuring the downstream pipeline
    can only process frames sequentially, matching real-time execution.
    """
    
    def __init__(
        self,
        config: SceneGraphConfig,
        pose_estimator: Optional[BasePoseEstimator] = None,
    ):
        self.config = config
        
        if self.config.dataset is None:
            raise ValueError("Config missing dataset")
            
        dataset_root = self.config.dataset.root
        self.loader = TUMLoader(dataset_root)
        
        self.rgb_entries = self.loader.load_rgb()
        self.depth_entries = self.loader.load_depth()
        self.pose_entries = self.loader.load_groundtruth()
        
        # Extract timestamps
        rgb_timestamps = [e.timestamp for e in self.rgb_entries]
        depth_timestamps = [e.timestamp for e in self.depth_entries]
        pose_timestamps = [e.timestamp for e in self.pose_entries]
        
        rgb_depth_max_dt = self.config.sync.rgb_depth_max_dt if self.config.sync else 0.02
        rgb_pose_max_dt = self.config.sync.rgb_pose_max_dt if self.config.sync else 0.05
        
        # Associate depth
        self.rgb_to_depth = dict(associate(rgb_timestamps, depth_timestamps, rgb_depth_max_dt))
        self.rgb_to_pose = dict(associate(rgb_timestamps, pose_timestamps, rgb_pose_max_dt))
        
        # Estimator setup (defaults to GroundTruthEstimator for offline replay if none provided)
        self.pose_estimator = pose_estimator
        if self.pose_estimator is None:
            self.pose_estimator = GroundTruthEstimator(
                pose_entries=self.pose_entries,
                max_dt=rgb_pose_max_dt,
            )

        self.ref_frame_provider = ReferenceFrameProvider(dataset_type="tum")

        if self.config.sequence is None:
            self.start_idx = 0
            self._end_idx = len(self.rgb_entries) - 1
        else:
            self.start_idx = self.config.sequence.start_frame
            self._end_idx = len(self.rgb_entries) - 1 if self.config.sequence.end_frame is None else self.config.sequence.end_frame
            
        self.end_idx = min(self._end_idx, len(self.rgb_entries) - 1)

        if not (0 <= self.start_idx <= self.end_idx < len(self.rgb_entries)):
            raise ValueError(
                f"Invalid sequence bounds: start_frame={self.start_idx}, "
                f"end_frame={self.end_idx}, total_rgb_frames={len(self.rgb_entries)}"
            )
        
    def __iter__(self) -> Iterator[FramePacket]:
        """Yield frame packets one at a time."""
        
        camera_intrinsics = None
        if self.config.camera is not None:
            camera_intrinsics = CameraIntrinsics(
                fx=self.config.camera.fx,
                fy=self.config.camera.fy,
                cx=self.config.camera.cx,
                cy=self.config.camera.cy,
                width=self.config.camera.width,
                height=self.config.camera.height
            )
            
        depth_model = None
        if self.config.depth is not None:
            depth_model = DepthModel(scale=self.config.depth.scale)
            
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
                    if depth_model is not None:
                        depth_np = depth_model.depth_to_meters(depth_raw)
                    else:
                        scale = self.config.depth.scale if self.config.depth is not None else 5000.0
                        depth_np = depth_raw.astype(np.float32) / scale
                    has_depth = True

            # Construct Stage 1 SensorFrame (acquisition-only)
            sensor_frame = SensorFrame(
                session_id=getattr(self.config.dataset, "name", "tum_replay") or "tum_replay",
                sequence_number=frame_idx,
                timestamp=Timestamp(value=rgb_entry.timestamp, domain=TimestampDomain.SIMULATED_TIME, source="tum_rgb"),
                rgb=rgb_np,
                camera_intrinsics=camera_intrinsics,
                depth=depth_np,
                depth_scale=1.0 / self.config.depth.scale if (self.config.depth and self.config.depth.scale > 0) else 0.0002,
                frame_id="camera_color_optical_frame",
                optical_frame_id="camera_depth_optical_frame",
                status=StreamStatus.OK if has_depth else StreamStatus.DEPTH_DROPPED,
            )

            # Stage 3: Estimate pose via BasePoseEstimator
            pose_estimate, estimator_state = self.pose_estimator.estimate(sensor_frame)

            # Construct relation reference frame
            origin = pose_estimate.world_T_camera[:3, 3] if pose_estimate.valid else None
            relation_frame = self.ref_frame_provider.get_frame(origin_world=origin)

            # Construct FramePacket
            packet = FramePacket.from_sensor_frame(
                sensor_frame=sensor_frame,
                pose_estimate=pose_estimate,
                estimator_state=estimator_state,
                relation_frame=relation_frame,
                world_frame="world",
            )
            
            yield packet

    def __len__(self) -> int:
        return max(0, self.end_idx - self.start_idx + 1)
