"""YAML configuration loader."""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path] = "configs/data.yaml") -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dict containing parsed configuration parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
