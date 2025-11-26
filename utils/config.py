"""Helper utilities for working with YAML-based configuration files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = {"plate_image_send_interval": 20, "video_paths": []}


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        LOGGER.warning("Config %s not found. Falling back to defaults.", path)
        return DEFAULT_CONFIG.copy()

    with path.open("r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file) or {}
        except yaml.YAMLError as exc:  # noqa: BLE001
            LOGGER.error("Failed to parse config %s: %s", path, exc)
            return DEFAULT_CONFIG.copy()
    return {**DEFAULT_CONFIG, **config}


def save_config(config_path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
