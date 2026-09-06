import yaml
from pathlib import Path

def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries."""
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

class SceneGraphConfig:
    def __init__(self, config_dict: dict):
        self._config = config_dict
        
    @classmethod
    def from_files(cls, base_path: str | Path, override_path: str | Path | None = None) -> "SceneGraphConfig":
        config = load_config(base_path)
        if override_path:
            override_config = load_config(override_path)
            config = merge_configs(config, override_config)
        return cls(config)

    def get(self, key: str, default=None):
        keys = key.split('.')
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def raw(self) -> dict:
        return self._config
