"""Configuration loading utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import yaml


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary.
    
    Args:
        base: Base dictionary.
        override: Override dictionary whose values take precedence.

    Returns:
        Merged dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: Union[str, Path],
    base_config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load configuration dictionary from a JSON or YAML file.

    Args:
        config_path: Path to the JSON or YAML configuration file.
        base_config_path: Optional path to a base configuration file to merge with.

    Returns:
        Dict containing configuration parameters.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        ValueError: If file cannot be parsed as JSON or YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    base_config: Dict[str, Any] = {}
    if base_config_path is not None:
        base_config = load_config(base_config_path)

    with open(path, "r", encoding="utf-8") as f:
        ext = path.suffix.lower()
        if ext in (".yaml", ".yml"):
            try:
                config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse YAML configuration file at {path}: {e}")
        elif ext == ".json":
            try:
                config = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON configuration file at {path}: {e}")
        else:
            # Fallback: attempt JSON first, then YAML
            content = f.read()
            try:
                config = json.loads(content)
            except json.JSONDecodeError:
                try:
                    config = yaml.safe_load(content) or {}
                except yaml.YAMLError as e:
                    raise ValueError(f"Failed to parse configuration file at {path}: {e}")

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file at {path} must contain a top-level dictionary/mapping.")

    if base_config:
        return _deep_merge_dicts(base_config, config)

    return config
