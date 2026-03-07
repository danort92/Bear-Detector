"""Configuration loading and merging utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg or {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*.

    Values in *override* take precedence. Nested dicts are merged rather than
    replaced entirely.

    Parameters
    ----------
    base:
        Base configuration dictionary.
    override:
        Override configuration dictionary.

    Returns
    -------
    dict
        Merged configuration.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_merged_config(
    default_path: str | Path = "config/default.yaml",
    override_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the default config and optionally merge an override config.

    Parameters
    ----------
    default_path:
        Path to the base YAML configuration.
    override_path:
        Optional path to an experiment-specific YAML configuration.

    Returns
    -------
    dict
        Final merged configuration dictionary.
    """
    cfg = load_config(default_path)
    if override_path is not None:
        override = load_config(override_path)
        cfg = merge_configs(cfg, override)
    return cfg
