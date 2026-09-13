"""Deterministic configuration loader bridging ROS parameters to SceneGraphConfig."""

from pathlib import Path
from typing import Optional, Union

from scene_graph.config import SceneGraphConfig


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve relative paths with respect to repository root if not found locally."""
    p = Path(path)
    if p.exists():
        return p.resolve()
    
    # Check parent paths for repo root containing configs/
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / p
        if candidate.exists():
            return candidate.resolve()
        current = current.parent
        
    return p.resolve()


def load_scene_graph_config(
    config_path: Optional[Union[str, Path]] = None,
    base_path: Union[str, Path] = "configs/default.yaml",
) -> SceneGraphConfig:
    """Load SceneGraphConfig deterministically from base and optional override YAML.
    
    Args:
        config_path: Path to sequence or experiment override YAML.
                     If None or pointing to default.yaml, only base is loaded.
        base_path: Path to default baseline YAML.
        
    Returns:
        Validated SceneGraphConfig instance.
    """
    resolved_base = resolve_path(base_path)
    if not resolved_base.exists():
        raise FileNotFoundError(f"Base configuration not found at {resolved_base}")

    if not config_path:
        return SceneGraphConfig.from_files(base_path=resolved_base)

    resolved_override = resolve_path(config_path)
    if not resolved_override.exists():
        raise FileNotFoundError(f"Override configuration not found at {resolved_override}")

    # If the user passed default.yaml as override, load base alone
    if resolved_override.resolve() == resolved_base.resolve():
        return SceneGraphConfig.from_files(base_path=resolved_base)

    return SceneGraphConfig.from_files(
        base_path=resolved_base,
        override_path=resolved_override,
    )
