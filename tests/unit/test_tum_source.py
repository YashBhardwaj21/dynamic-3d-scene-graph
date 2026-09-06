import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from scene_graph.config import SceneGraphConfig
from scene_graph.data.tum_source import TUMReplaySource


@pytest.fixture
def mock_config(tmp_path):
    dataset_dir = tmp_path / "fake_dataset_dir"
    dataset_dir.mkdir()
    config_dict = {
        "dataset": {
            "root": str(dataset_dir),
            "type": "tum_rgbd"
        },
        "sequence": {
            "start_frame": 0,
            "end_frame": 10
        },
        "sync": {
            "rgb_depth_max_dt": 0.02,
            "rgb_pose_max_dt": 0.02
        }
    }
    return SceneGraphConfig.model_validate(config_dict)


@patch("scene_graph.data.tum_source.TUMLoader")
@patch("scene_graph.data.tum_source.cv2")
def test_tum_source_iter(mock_cv2, mock_loader_cls, mock_config):
    # Setup mocks
    mock_loader = MagicMock()
    mock_loader_cls.return_value = mock_loader
    
    # Fake RGB entries
    mock_rgb_entry = MagicMock()
    mock_rgb_entry.timestamp = 1.0
    mock_loader.load_rgb.return_value = [mock_rgb_entry] * 15
    
    # Fake empty depth and pose to avoid matching logic complexity
    mock_loader.load_depth.return_value = []
    mock_loader.load_groundtruth.return_value = []
    
    # Mock cv2 to return a dummy image
    mock_cv2.imread.return_value = "fake_bgr_image"
    mock_cv2.cvtColor.return_value = "fake_rgb_image"
    
    source = TUMReplaySource(mock_config)
    
    assert len(source) == 11  # from start_frame 0 to end_frame 10 inclusive
    
    packets = list(source)
    assert len(packets) == 11
    
    assert packets[0].frame_index == 0
    assert packets[0].rgb == "fake_rgb_image"
    assert packets[0].has_depth is False
    assert packets[0].has_pose is False
    
    # Check that it yielded exactly what we asked for
    assert packets[-1].frame_index == 10
