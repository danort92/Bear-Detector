"""Tests for configuration loading and merging."""

from __future__ import annotations

import pytest
from pathlib import Path

from src.utils.config import load_config, merge_configs, load_merged_config


PROJECT_ROOT = Path(__file__).parent.parent


class TestLoadConfig:
    def test_loads_default_yaml(self):
        cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
        assert isinstance(cfg, dict)
        assert "experiment" in cfg
        assert "data" in cfg
        assert "model" in cfg

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_returns_dict_for_empty_yaml(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        cfg = load_config(empty)
        assert cfg == {}


class TestMergeConfigs:
    def test_shallow_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        merged = merge_configs(base, override)
        assert merged["a"] == 1
        assert merged["b"] == 99

    def test_nested_merge(self):
        base = {"training": {"epochs": 10, "lr": 0.001}}
        override = {"training": {"epochs": 100}}
        merged = merge_configs(base, override)
        assert merged["training"]["epochs"] == 100
        assert merged["training"]["lr"] == 0.001  # preserved

    def test_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        merge_configs(base, override)
        assert base["a"]["b"] == 1  # original unchanged

    def test_adds_new_keys(self):
        base = {"a": 1}
        override = {"b": 2}
        merged = merge_configs(base, override)
        assert merged["b"] == 2


class TestLoadMergedConfig:
    def test_no_override(self):
        cfg = load_merged_config(PROJECT_ROOT / "config" / "default.yaml")
        assert "experiment" in cfg

    def test_with_override(self, tmp_path):
        override_yaml = tmp_path / "override.yaml"
        override_yaml.write_text("experiment:\n  seed: 999\n")
        cfg = load_merged_config(
            PROJECT_ROOT / "config" / "default.yaml",
            override_yaml,
        )
        assert cfg["experiment"]["seed"] == 999
        # Other keys preserved
        assert "data" in cfg
