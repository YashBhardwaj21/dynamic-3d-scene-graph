"""YOLOE Detector wrapper (Substage 2.3)."""

import uuid
from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from scene_graph.data.frame_packet import FramePacket
from scene_graph.perception.observation_source import ObservationProducer
from scene_graph.perception.observation import Observation, encode_mask_rle

class YOLOEDetector(ObservationProducer):
    """Wrapper around Ultralytics YOLOE/YOLOv8 model for observation extraction."""
    
    def __init__(self, 
                 model_path: str, 
                 confidence_threshold: float = 0.40,
                 allowed_classes: Optional[Tuple[str, ...]] = None):
        """Initialize the detector.
        
        Args:
            model_path: Path to the Ultralytics model weights. Must exist locally.
            confidence_threshold: Minimum confidence score [0, 1].
            allowed_classes: Set of allowed class names. If None, allows all.
        """
        if YOLO is None:
            raise ImportError("ultralytics package is required for YOLOEDetector.")
            
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}. Automatic downloads are disabled.")
            
        self.model_path = str(model_path_obj)
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = allowed_classes
        self.model = YOLO(self.model_path)
        
        # Configure open-vocabulary prompting if the model supports it
        if self.allowed_classes is not None:
            # We assume Ultralytics finds its assets without copying to the root.
            # Fail hard if initialization fails.
            try:
                self.model.set_classes(list(self.allowed_classes))
            except (AttributeError, AssertionError) as e:
                raise RuntimeError(
                    f"Model {self.model_path} does not support set_classes() or has wrong architecture. "
                    f"Cannot filter by requested vocabulary. Error: {e}"
                ) from e
                
        # Internal sequential ID counter
        self._obs_counter = 1
        
    def _generate_obs_id(self) -> str:
        """Generate a sequential observation ID (e.g., obs_000001)."""
        obs_id = f"obs_{self._obs_counter:06d}"
        self._obs_counter += 1
        return obs_id
        
    def detect(self, packet: FramePacket) -> List[Observation]:
        """Run YOLOE and extract observations with 3D geometry."""
        # Run inference
        results = self.model(packet.rgb, conf=self.confidence_threshold, verbose=False)
        
        observations = []
        if not results or len(results) == 0:
            return observations
            
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return observations
            
        for i, box in enumerate(result.boxes):
            # 1. Parse Box
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # Original raw class name from detector
            class_name = self.model.names[cls_id].lower()
            
            # Filter if vocabulary is provided
            if self.allowed_classes is not None and class_name not in self.allowed_classes:
                continue
                
            xyxy = box.xyxy[0].cpu().numpy()
            
            # 2. Parse Mask
            mask_rle = None
            mask_np = None
            if result.masks is not None and len(result.masks) > i:
                mask_data = result.masks.data[i].cpu().numpy()
                import cv2
                if mask_data.shape != (packet.rgb.shape[0], packet.rgb.shape[1]):
                    mask_np = cv2.resize(mask_data, (packet.rgb.shape[1], packet.rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_np = mask_data
                    
                mask_np = (mask_np > 0)
                mask_rle = encode_mask_rle(mask_np)
                
            obs = Observation(
                obs_id=self._generate_obs_id(),
                frame_index=packet.frame_index,
                timestamp=packet.timestamp,
                class_name=class_name,
                confidence=conf,
                bbox_xyxy=xyxy,
                mask_rle=mask_rle
            )
            observations.append(obs)
            
        return observations
        
    def get_model_info(self) -> dict:
        import ultralytics
        import torch
        import platform
        import hashlib
        
        # Calculate SHA256 of model
        sha256 = "unknown"
        try:
            with open(self.model_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass
            
        return {
            "model_path": self.model_path,
            "model_sha256": sha256,
            "ultralytics_version": ultralytics.__version__,
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
            "device": str(self.model.device),
            "confidence_threshold": self.confidence_threshold,
            "vocabulary_size": len(self.allowed_classes) if self.allowed_classes else "all"
        }
