"""Configuration loader bridging ROS parameters to SceneGraphConfig."""

import os
from pathlib import Path
from typing import Optional, Union

from scene_graph.config import SceneGraphConfig


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve relative or absolute paths with respect to repository root if not found locally."""
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    p = Path(expanded)
    if p.is_absolute() and p.exists():
        return p.resolve()

    if p.exists():
        return p.resolve()

    try:
        import scene_graph
        sg_root = Path(scene_graph.__file__).resolve().parent.parent.parent
        candidate = sg_root / p
        if candidate.exists():
            return candidate.resolve()
    except Exception:
        pass

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
    resolved_base = resolve_path(base_path)
    if not resolved_base.exists():
        raise FileNotFoundError(f"Base configuration not found at {resolved_base}")

    if not config_path:
        return SceneGraphConfig.from_files(base_path=resolved_base)

    resolved_override = resolve_path(config_path)
    if not resolved_override.exists():
        raise FileNotFoundError(f"Override configuration not found at {resolved_override}")

    if resolved_override.resolve() == resolved_base.resolve():
        return SceneGraphConfig.from_files(base_path=resolved_base)

    return SceneGraphConfig.from_files(
        base_path=resolved_base,
        override_path=resolved_override,
    )
