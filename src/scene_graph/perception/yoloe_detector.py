"""YOLOE Detector wrapper supporting prompt-free, text-prompt, and visual-prompt modes."""

from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np

try:
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
except ImportError:
    YOLOE = None
    YOLOEVPSegPredictor = None

from scene_graph.data.frame_packet import FramePacket
from scene_graph.perception.observation_source import ObservationProducer
from scene_graph.perception.observation import Observation, encode_mask_rle


class PerceptionMode(Enum):
    """Supported perception operating modes for YOLOE."""
    PROMPT_FREE = "prompt_free"
    TEXT_PROMPT = "text"
    VISUAL_PROMPT = "visual"

    @classmethod
    def from_str(cls, val: str) -> "PerceptionMode":
        v = val.lower().strip()
        if v in ("prompt_free", "promptfree", "pf"):
            return cls.PROMPT_FREE
        elif v in ("text", "text_prompt", "prompt"):
            return cls.TEXT_PROMPT
        elif v in ("visual", "visual_prompt", "exemplar"):
            return cls.VISUAL_PROMPT
        raise ValueError(
            f"Unsupported perception mode: {val!r}. "
            "Expected 'prompt_free', 'text', or 'visual'."
        )


class YOLOEDetector(ObservationProducer):
    """Open-vocabulary detector supporting prompt-free, text, and visual prompting.
    
    Observation.class_name is treated as detector metadata, NOT physical object identity.
    """
    
    def __init__(
        self, 
        model_path: str = "models/yoloe-26m-seg-pf.pt", 
        confidence_threshold: float = 0.40,
        mode: Union[PerceptionMode, str] = PerceptionMode.PROMPT_FREE,
        text_prompts: Optional[List[str]] = None,
        visual_prompt: Optional[Any] = None,
        device: str = "auto",
        allowed_classes: Optional[Tuple[str, ...]] = None,
        image_size: Optional[int] = 480,
    ):
        """Initialize the open-vocabulary detector.
        
        Args:
            model_path: Path to Ultralytics model weights. Must exist locally.
            confidence_threshold: Minimum detection confidence score [0, 1].
            mode: Operating mode (PROMPT_FREE, TEXT_PROMPT, or VISUAL_PROMPT).
            text_prompts: Optional runtime text prompts for TEXT_PROMPT mode.
            visual_prompt: Optional exemplar for VISUAL_PROMPT mode.
            device: Compute device ('auto', 'cpu', 'cuda').
            allowed_classes: Deprecated alias for text_prompts (backward compatibility).
            image_size: Network input resolution for accelerated CPU inference (e.g. 480).
        """
        if YOLOE is None:
            raise ImportError("ultralytics package is required for YOLOEDetector.")
            
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            repo_root = Path(__file__).resolve().parents[3]
            candidate = repo_root / model_path_obj
            if candidate.exists():
                model_path_obj = candidate
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {model_path}. "
                    "Automatic downloads are disabled."
                )
            
        self.model_path = str(model_path_obj)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.image_size = image_size
        self.mode = PerceptionMode.from_str(mode) if isinstance(mode, str) else mode
        
        # Backward compatibility for allowed_classes -> text_prompts
        if text_prompts is None and allowed_classes is not None:
            text_prompts = list(allowed_classes)
            self.mode = PerceptionMode.TEXT_PROMPT

        if text_prompts and self.mode == PerceptionMode.PROMPT_FREE:
            self.mode = PerceptionMode.TEXT_PROMPT

        if self.mode in (
            PerceptionMode.TEXT_PROMPT,
            PerceptionMode.VISUAL_PROMPT,
        ) and self.model_path.endswith("-pf.pt"):
            raise ValueError(
                f"{self.mode.value} mode requires a prompted YOLOE checkpoint "
                "(*-seg.pt), not a prompt-free checkpoint (*-seg-pf.pt)."
            )

        self.current_text_prompts: Optional[List[str]] = None
        self.current_visual_prompt: Optional[Any] = None
        self.model = YOLOE(self.model_path)

        # In PROMPT_FREE mode: do NOT call set_classes().
        # Prompt-free checkpoints reject set_classes() and use their built-in 4,585-name vocabulary.
        if self.mode == PerceptionMode.TEXT_PROMPT and text_prompts:
            self.set_text_prompts(text_prompts)
        elif self.mode == PerceptionMode.VISUAL_PROMPT and visual_prompt is not None:
            self.set_visual_prompt(visual_prompt)
                
        # Internal sequential ID counter
        self._obs_counter = 1

    def set_text_prompts(self, prompts: List[str]) -> None:
        """Dynamically update recognized vocabulary at runtime without source modification."""
        if not prompts:
            return
        self.mode = PerceptionMode.TEXT_PROMPT
        self.current_text_prompts = list(prompts)
        try:
            self.model.set_classes(self.current_text_prompts)
        except (AttributeError, AssertionError, RuntimeError) as e:
            raise RuntimeError(
                f"Model {self.model_path} failed to set runtime text prompts: {e}"
            ) from e

    def set_visual_prompt(self, visual_prompt: Any) -> None:
        """Set the visual prompt used by YOLOE visual-prompt inference."""
        if not isinstance(visual_prompt, dict):
            raise TypeError(
                "visual_prompt must be a dict containing 'bboxes' and 'cls'."
            )
        if "bboxes" not in visual_prompt or "cls" not in visual_prompt:
            raise ValueError(
                "visual_prompt must contain 'bboxes' and 'cls'."
            )
        bboxes = np.asarray(visual_prompt["bboxes"], dtype=np.float32)
        cls = np.asarray(visual_prompt["cls"], dtype=np.int64)
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError(
                "visual_prompt['bboxes'] must have shape (N, 4)."
            )
        if cls.ndim != 1:
            raise ValueError(
                "visual_prompt['cls'] must have shape (N,)."
            )
        if len(bboxes) != len(cls):
            raise ValueError(
                "visual_prompt['bboxes'] and visual_prompt['cls'] "
                "must contain the same number of entries."
            )
        if len(bboxes) == 0:
            raise ValueError("visual_prompt must contain at least one example.")
        if np.any(cls < 0):
            raise ValueError("visual_prompt class indices must be non-negative.")
        self.mode = PerceptionMode.VISUAL_PROMPT
        self.current_visual_prompt = {
            "bboxes": bboxes,
            "cls": cls,
        }
        
    def _generate_obs_id(self) -> str:
        """Generate a sequential observation ID (e.g., obs_000001)."""
        obs_id = f"obs_{self._obs_counter:06d}"
        self._obs_counter += 1
        return obs_id
        
    def detect(
        self, 
        packet: FramePacket,
        mode: Optional[Union[PerceptionMode, str]] = None,
        text_prompts: Optional[List[str]] = None,
        visual_prompt: Optional[Any] = None,
    ) -> List[Observation]:
        """Run open-vocabulary inference and extract observations."""
        if mode is not None:
            target_mode = PerceptionMode.from_str(mode) if isinstance(mode, str) else mode
            if target_mode != self.mode:
                self.mode = target_mode

        if text_prompts is not None:
            self.set_text_prompts(text_prompts)

        if visual_prompt is not None:
            self.set_visual_prompt(visual_prompt)

        # Run model inference with tuned resolution
        imgsz = self.image_size if self.image_size is not None else 640
        if self.mode == PerceptionMode.VISUAL_PROMPT:
            if self.current_visual_prompt is None:
                raise RuntimeError(
                    "VISUAL_PROMPT mode requires a visual prompt."
                )
            results = self.model.predict(
                source=packet.rgb,
                imgsz=imgsz,
                conf=self.confidence_threshold,
                visual_prompts=self.current_visual_prompt,
                predictor=YOLOEVPSegPredictor,
                verbose=False,
            )
        else:
            results = self.model.predict(
                source=packet.rgb,
                imgsz=imgsz,
                conf=self.confidence_threshold,
                verbose=False,
            )
        
        observations = []
        if not results or len(results) == 0:
            return observations
            
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return observations
            
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # Semantic label is metadata from detector output
            if hasattr(self.model, "names") and cls_id in self.model.names:
                class_name = self.model.names[cls_id].lower()
            else:
                class_name = f"concept_{cls_id}"
                
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Parse binary mask if available
            mask_rle = None
            if result.masks is not None and len(result.masks) > i:
                mask_data = result.masks.data[i].cpu().numpy()
                import cv2
                if mask_data.shape != (packet.rgb.shape[0], packet.rgb.shape[1]):
                    mask_np = cv2.resize(
                        mask_data, 
                        (packet.rgb.shape[1], packet.rgb.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
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
            "device": str(getattr(self.model, "device", self.device)),
            "confidence_threshold": self.confidence_threshold,
            "mode": self.mode.value,
            "current_text_prompts": self.current_text_prompts,
        }

