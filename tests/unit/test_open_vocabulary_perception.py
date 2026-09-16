from pathlib import Path
from unittest.mock import MagicMock, patch
import yaml
import pytest
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.perception.yoloe_detector import YOLOEDetector, PerceptionMode
from scene_graph.perception.observation import Observation
from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.camera import CameraIntrinsics


def test_default_config_has_no_classes_or_vocabulary():
    """Verify that default.yaml contains no hardcoded classes or vocabulary blocks."""
    config_path = Path("configs/default.yaml")
    assert config_path.exists(), "configs/default.yaml must exist"

    with open(config_path, "r") as f:
        raw_yaml = yaml.safe_load(f)

    # Core perception must be prompt_free by default
    assert "perception" in raw_yaml
    perception_cfg = raw_yaml["perception"]
    assert perception_cfg.get("mode") == "prompt_free"
    assert "classes" not in perception_cfg, "Default perception must not contain hardcoded classes"
    assert "vocabulary" not in perception_cfg, "Default perception must not reference hardcoded vocabulary"
    assert "vocabularies" not in raw_yaml, "Default config must not declare vocabulary tables"

    # Verify Pydantic loading
    loaded_config = SceneGraphConfig.from_files(config_path)
    assert loaded_config.perception.mode == "prompt_free"
    assert len(loaded_config.perception.classes) == 0


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_prompt_free_initialization_without_set_classes(mock_yolo_cls, mock_exists):
    """Verify that PROMPT_FREE mode never calls set_classes() on the model checkpoint."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    detector = YOLOEDetector(
        model_path="models/yoloe-26m-seg-pf.pt",
        mode=PerceptionMode.PROMPT_FREE,
    )

    assert detector.mode == PerceptionMode.PROMPT_FREE
    # Prompt-free checkpoints reject set_classes(); verify it was never called
    mock_model.set_classes.assert_not_called()


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_runtime_text_prompt_adaptation(mock_yolo_cls, mock_exists):
    """Verify that runtime text prompts configure set_classes() dynamically without file edits."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    detector = YOLOEDetector(
        model_path="models/yoloe-26m-seg-pf.pt",
        mode=PerceptionMode.PROMPT_FREE,
    )

    # Dynamically inject runtime text prompt
    runtime_prompts = ["laptop", "coffee mug", "water bottle", "person"]
    detector.set_text_prompts(runtime_prompts)

    assert detector.mode == PerceptionMode.TEXT_PROMPT
    assert detector.current_text_prompts == runtime_prompts
    mock_model.set_classes.assert_called_once_with(runtime_prompts)


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_visual_prompt_mode_setting(mock_yolo_cls, mock_exists):
    """Verify setting visual exemplar prompt at runtime with valid bboxes and cls."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    detector = YOLOEDetector(
        model_path="models/yoloe-26m-seg-pf.pt",
        mode=PerceptionMode.PROMPT_FREE,
    )

    dummy_box_exemplar = {
        "bboxes": np.array([[10.0, 20.0, 50.0, 60.0]], dtype=np.float32),
        "cls": np.array([0], dtype=np.int64),
    }
    detector.set_visual_prompt(dummy_box_exemplar)

    assert detector.mode == PerceptionMode.VISUAL_PROMPT
    assert np.array_equal(detector.current_visual_prompt["bboxes"], dummy_box_exemplar["bboxes"])
    assert np.array_equal(detector.current_visual_prompt["cls"], dummy_box_exemplar["cls"])


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_pf_checkpoint_rejected_for_prompted_modes(mock_yolo_cls, mock_exists):
    """Verify that -pf.pt checkpoint raises ValueError in TEXT_PROMPT or VISUAL_PROMPT mode."""
    with pytest.raises(ValueError, match="requires a prompted YOLOE checkpoint"):
        YOLOEDetector(
            model_path="models/yoloe-26m-seg-pf.pt",
            mode=PerceptionMode.TEXT_PROMPT,
        )

    with pytest.raises(ValueError, match="requires a prompted YOLOE checkpoint"):
        YOLOEDetector(
            model_path="models/yoloe-26m-seg-pf.pt",
            mode=PerceptionMode.VISUAL_PROMPT,
        )


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOEVPSegPredictor")
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_visual_prompt_predict_invocation(mock_yolo_cls, mock_predictor_cls, mock_exists):
    """Verify that detect() passes visual_prompts and predictor to predict()."""
    mock_model = MagicMock()
    mock_model.predict.return_value = []
    mock_yolo_cls.return_value = mock_model

    detector = YOLOEDetector(
        model_path="models/yoloe-26m-seg.pt",
        mode=PerceptionMode.VISUAL_PROMPT,
        visual_prompt={
            "bboxes": np.array([[5.0, 10.0, 50.0, 60.0]], dtype=np.float32),
            "cls": np.array([0], dtype=np.int64),
        },
    )

    packet = FramePacket(
        frame_index=1,
        timestamp=0.033,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=None,
        world_T_camera=None,
        camera_intrinsics=CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480),
    )

    detector.detect(packet)
    mock_model.predict.assert_called_once()
    _, kwargs = mock_model.predict.call_args
    assert "visual_prompts" in kwargs
    assert kwargs["predictor"] == mock_predictor_cls


def test_observation_class_is_metadata():
    """Verify Observation stores class_name as metadata, decoupled from physical identity."""
    obs = Observation(
        obs_id="obs_000042",
        frame_index=1,
        timestamp=0.033,
        class_name="coffee_mug",
        confidence=0.88,
        bbox_xyxy=np.array([10.0, 20.0, 50.0, 60.0]),
        mask_rle=None,
    )

    assert obs.obs_id == "obs_000042"
    assert obs.class_name == "coffee_mug"
    # Metadata attributes
    assert obs.confidence == 0.88
