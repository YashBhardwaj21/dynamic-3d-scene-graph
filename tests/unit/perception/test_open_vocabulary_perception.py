from pathlib import Path
from unittest.mock import MagicMock, patch
import yaml
import pytest
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.perception.yoloe_detector import (
    YOLOEDetector,
    PerceptionMode,
    CANONICAL_23_CLASSES,
    CANONICAL_SYNONYMS,
)
from scene_graph.perception.observation import Observation
from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.camera import CameraIntrinsics


def test_runtime_configs_have_text_mode_and_canonical_prompts():
    """Verify runtime YAML configurations specify text mode and canonical 23 classes."""
    for config_rel in ["configs/default.yaml", "configs/runtime/live_d455.yaml"]:
        config_path = Path(config_rel)
        assert config_path.exists(), f"{config_rel} must exist"

        with open(config_path, "r") as f:
            raw_yaml = yaml.safe_load(f)

        assert "perception" in raw_yaml
        p_cfg = raw_yaml["perception"]
        assert p_cfg.get("mode") == "text"
        assert p_cfg.get("model") == "models/yoloe-26s-seg.pt"
        assert "prompts" in p_cfg
        assert len(p_cfg["prompts"]) == 23
        assert "chair" in p_cfg["prompts"]
        assert "quadcopter drone" in p_cfg["prompts"]

        loaded = SceneGraphConfig.from_files(config_path)
        assert loaded.perception.mode == "text"
        assert len(loaded.perception.prompts) == 23


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_prompt_free_initialization_without_set_classes(mock_yolo_cls, mock_exists):
    """Verify that PROMPT_FREE mode never calls set_classes() on the model checkpoint."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    detector = YOLOEDetector(
        model_path="models/yoloe-26s-seg-pf.pt",
        mode=PerceptionMode.PROMPT_FREE,
    )

    assert detector.mode == PerceptionMode.PROMPT_FREE
    mock_model.set_classes.assert_not_called()


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_text_prompt_configuration_on_prompted_checkpoint(mock_yolo_cls, mock_exists):
    """Verify that runtime text prompts configure set_classes() on prompted checkpoints."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    runtime_prompts = ["person", "chair", "desk", "quadcopter drone"]
    detector = YOLOEDetector(
        model_path="models/yoloe-26s-seg.pt",
        mode=PerceptionMode.TEXT_PROMPT,
        text_prompts=runtime_prompts,
    )

    assert detector.mode == PerceptionMode.TEXT_PROMPT
    assert detector.current_text_prompts == runtime_prompts
    mock_model.set_classes.assert_called_once_with(runtime_prompts)


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_pf_checkpoint_rejected_for_prompted_modes(mock_yolo_cls, mock_exists):
    """Verify that -pf.pt checkpoint raises ValueError in TEXT_PROMPT or VISUAL_PROMPT mode."""
    with pytest.raises(ValueError, match="requires a prompted YOLOE checkpoint"):
        YOLOEDetector(
            model_path="models/yoloe-26s-seg-pf.pt",
            mode=PerceptionMode.TEXT_PROMPT,
            text_prompts=["chair"],
        )

    with pytest.raises(ValueError, match="requires a prompted YOLOE checkpoint"):
        YOLOEDetector(
            model_path="models/yoloe-26s-seg-pf.pt",
            mode=PerceptionMode.VISUAL_PROMPT,
        )


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_set_text_prompts_on_pf_checkpoint_raises(mock_yolo_cls, mock_exists):
    """Verify that dynamically calling set_text_prompts on a -pf.pt model fails explicitly."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    detector = YOLOEDetector(
        model_path="models/yoloe-26s-seg-pf.pt",
        mode=PerceptionMode.PROMPT_FREE,
    )

    with pytest.raises(ValueError, match="Cannot configure text prompts on a prompt-free checkpoint"):
        detector.set_text_prompts(["person", "chair"])


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_allowlist_filtering_and_canonical_normalization(mock_yolo_cls, mock_exists):
    """Verify out-of-vocabulary model labels are dropped and synonyms normalized."""
    mock_model = MagicMock()
    mock_model.names = {0: "barber shop", 1: "monitor", 2: "chair", 3: "lock chamber"}
    mock_yolo_cls.return_value = mock_model

    detector = YOLOEDetector(
        model_path="models/yoloe-26s-seg.pt",
        mode=PerceptionMode.TEXT_PROMPT,
        text_prompts=["computer monitor", "chair"],
        confidence_threshold=0.20,
    )

    # Construct mock prediction results
    box_0 = MagicMock()
    box_0.cls = [MagicMock(item=lambda: 0)]
    box_0.conf = [MagicMock(item=lambda: 0.85)]
    box_0.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([0, 0, 50, 50])))]

    box_1 = MagicMock()  # monitor -> normalized to computer monitor (allowed)
    box_1.cls = [MagicMock(item=lambda: 1)]
    box_1.conf = [MagicMock(item=lambda: 0.90)]
    box_1.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([50, 50, 100, 100])))]

    box_2 = MagicMock()  # chair (allowed)
    box_2.cls = [MagicMock(item=lambda: 2)]
    box_2.conf = [MagicMock(item=lambda: 0.80)]
    box_2.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([120, 120, 200, 200])))]

    box_3 = MagicMock()  # lock chamber (not allowed)
    box_3.cls = [MagicMock(item=lambda: 3)]
    box_3.conf = [MagicMock(item=lambda: 0.95)]
    box_3.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([210, 210, 300, 300])))]

    mock_result = MagicMock()
    mock_result.boxes = [box_0, box_1, box_2, box_3]
    mock_result.masks = None
    mock_model.predict.return_value = [mock_result]

    packet = FramePacket(
        frame_index=1,
        timestamp=0.033,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=None,
        world_T_camera=None,
        camera_intrinsics=CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480),
    )

    observations = detector.detect(packet)

    # Exactly 2 observations must pass allowlist
    assert len(observations) == 2
    classes = [o.class_name for o in observations]
    assert "barber shop" not in classes
    assert "lock chamber" not in classes
    assert "computer monitor" in classes  # Normalized from monitor
    assert "chair" in classes


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_overlapping_class_suppression(mock_yolo_cls, mock_exists):
    """Verify overlapping ambiguous classes (desk vs table) are suppressed by confidence."""
    mock_model = MagicMock()
    mock_model.names = {0: "desk", 1: "table"}
    mock_yolo_cls.return_value = mock_model

    detector = YOLOEDetector(
        model_path="models/yoloe-26s-seg.pt",
        mode=PerceptionMode.TEXT_PROMPT,
        text_prompts=["desk", "table"],
        confidence_threshold=0.20,
    )

    # Two heavily overlapping bounding boxes (IoU > 0.8)
    box_desk = MagicMock()
    box_desk.cls = [MagicMock(item=lambda: 0)]
    box_desk.conf = [MagicMock(item=lambda: 0.75)]
    box_desk.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([10, 10, 100, 100])))]

    box_table = MagicMock()
    box_table.cls = [MagicMock(item=lambda: 1)]
    box_table.conf = [MagicMock(item=lambda: 0.45)]
    box_table.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([12, 12, 102, 102])))]

    mock_result = MagicMock()
    mock_result.boxes = [box_desk, box_table]
    mock_result.masks = None
    mock_model.predict.return_value = [mock_result]

    packet = FramePacket(
        frame_index=1,
        timestamp=0.033,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=None,
        world_T_camera=None,
        camera_intrinsics=CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480),
    )

    observations = detector.detect(packet)
    assert len(observations) == 1
    assert observations[0].class_name == "desk"
    assert observations[0].confidence == pytest.approx(0.75)


@patch("scene_graph.perception.yoloe_detector.Path.exists", return_value=True)
@patch("scene_graph.perception.yoloe_detector.YOLOE")
def test_empty_detections_handled_safely(mock_yolo_cls, mock_exists):
    """Verify empty detections produce empty observations without errors."""
    mock_model = MagicMock()
    mock_model.names = {0: "chair"}
    mock_model.predict.return_value = []
    mock_yolo_cls.return_value = mock_model

    detector = YOLOEDetector(
        model_path="models/yoloe-26s-seg.pt",
        mode=PerceptionMode.TEXT_PROMPT,
        text_prompts=["chair"],
    )

    packet = FramePacket(
        frame_index=1,
        timestamp=0.033,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=None,
        world_T_camera=None,
        camera_intrinsics=CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480),
    )

    assert detector.detect(packet) == []


def test_real_inference_on_recorded_sequence():
    """Verify real inference with models/yoloe-26s-seg.pt on recorded image data/raw/.../000000.png."""
    sample_img = Path("data/raw/d455_rgbd_20261010_123443/rgb/000000.png")
    model_path = Path("models/yoloe-26s-seg.pt")
    if not sample_img.exists() or not model_path.exists():
        pytest.skip("Recorded sequence or model weights not present on disk")

    import cv2
    img = cv2.imread(str(sample_img))
    assert img is not None

    detector = YOLOEDetector(
        model_path=str(model_path),
        confidence_threshold=0.25,
        mode=PerceptionMode.TEXT_PROMPT,
        text_prompts=list(CANONICAL_23_CLASSES),
        device="cpu",
        image_size=480,
    )

    packet = FramePacket(
        frame_index=0,
        timestamp=0.0,
        rgb=img,
        depth=None,
        world_T_camera=None,
        camera_intrinsics=CameraIntrinsics(fx=386, fy=386, cx=320, cy=240, width=640, height=480),
    )

    observations = detector.detect(packet)
    assert len(observations) > 0

    # Ensure every single observation belongs strictly to the allowlisted 23 classes
    allowed_set = set(CANONICAL_23_CLASSES)
    for obs in observations:
        assert obs.class_name in allowed_set, f"Detected out-of-vocabulary class: {obs.class_name}"
        assert obs.confidence >= 0.25
        assert obs.bbox_xyxy is not None
