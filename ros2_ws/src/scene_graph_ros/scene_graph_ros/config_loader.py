"""Deterministic configuration loader bridging ROS parameters to SceneGraphConfig."""

from pathlib import Path
from typing import Optional, Union

from scene_graph.config import SceneGraphConfig


import os


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve relative or absolute paths with respect to repository root if not found locally."""
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    p = Path(expanded)
    if p.is_absolute() and p.exists():
        return p.resolve()

    # 1. Check relative to current working directory
    if p.exists():
        return p.resolve()

    # 2. Check relative to scene_graph package repository root
    try:
        import scene_graph
        sg_root = Path(scene_graph.__file__).resolve().parent.parent.parent
        candidate = sg_root / p
        if candidate.exists():
            return candidate.resolve()
    except Exception:
        pass

    # 3. Walk up from __file__ looking for target path or repository root with configs/
    current = Path(__file__).resolve().parent
    while current != current.parent:
        candidate = current / p
        if candidate.exists():
            return candidate.resolve()
        if (current / "configs" / "default.yaml").exists():
            candidate = current / p
            if candidate.exists():
                return candidate.resolve()
        current = current.parent

    # 4. Walk up from current working directory
    current = Path.cwd().resolve()
    while current != current.parent:
        candidate = current / p
        if candidate.exists():
            return candidate.resolve()
        if (current / "configs" / "default.yaml").exists():
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
