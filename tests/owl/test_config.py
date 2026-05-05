import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from owl.config import BaseConfig
from pydantic import Field, ValidationError


class InnerConfig(BaseConfig):
    name: str
    size: int = Field(ge=1)


class LeafConfig(BaseConfig):
    name: str
    size: int = Field(ge=1)


class InnerConfigWithSubconfigRef(BaseConfig):
    name: str
    leaf: LeafConfig

    @classmethod
    def subconfig_dirs(cls) -> set[str]:
        return {"leaf"}


class RootConfig(BaseConfig):
    inner: InnerConfig
    seed: int = Field(ge=0)


class RootConfigWithSubconfigRef(BaseConfig):
    inner: InnerConfig

    @classmethod
    def subconfig_dirs(cls) -> set[str]:
        return {"inner"}


class RootConfigWithNestedSubconfigRef(BaseConfig):
    inner: InnerConfigWithSubconfigRef

    @classmethod
    def subconfig_dirs(cls) -> set[str]:
        return {"inner"}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def test_base_config_can_apply_nested_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(
        config_path,
        {
            "inner": {"name": "base", "size": 2},
            "seed": 42,
        },
    )

    cfg = RootConfig.from_file(
        config_path,
        overrides={"inner.name": "overridden", "seed": 7},
    )

    assert cfg.inner.name == "overridden"
    assert cfg.seed == 7


def test_base_config_rejects_unknown_override_field(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(
        config_path,
        {
            "inner": {"name": "base", "size": 2},
            "seed": 42,
        },
    )

    with pytest.raises(ValidationError, match="missing_field"):
        RootConfig.from_file(config_path, overrides={"inner.missing_field": 1})


def test_base_config_validates_overridden_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(
        config_path,
        {
            "inner": {"name": "base", "size": 2},
            "seed": 42,
        },
    )

    with pytest.raises(ValidationError, match="size"):
        RootConfig.from_file(config_path, overrides={"inner.size": 0})


def test_base_config_can_override_subconfig_reference_and_nested_value(
    tmp_path: Path,
) -> None:
    (tmp_path / "inner").mkdir()
    _write_yaml(
        tmp_path / "inner" / "preset_a.yaml",
        {"name": "preset-a", "size": 2},
    )
    _write_yaml(
        tmp_path / "inner" / "preset_b.yaml",
        {"name": "preset-b", "size": 3},
    )
    _write_yaml(tmp_path / "config.yaml", {"inner": "preset_a"})

    cfg = RootConfigWithSubconfigRef.from_file(
        tmp_path / "config.yaml",
        overrides={"inner": "preset_b", "inner.size": 10},
    )

    assert cfg.inner.name == "preset-b"
    assert cfg.inner.size == 10


def test_base_config_can_override_nested_subconfig_reference(
    tmp_path: Path,
) -> None:
    (tmp_path / "inner").mkdir()
    (tmp_path / "inner" / "leaf").mkdir()
    _write_yaml(
        tmp_path / "inner" / "preset.yaml",
        {"name": "inner", "leaf": "leaf_a"},
    )
    _write_yaml(
        tmp_path / "inner" / "leaf" / "leaf_a.yaml",
        {"name": "leaf-a", "size": 2},
    )
    _write_yaml(
        tmp_path / "inner" / "leaf" / "leaf_b.yaml",
        {"name": "leaf-b", "size": 3},
    )
    _write_yaml(tmp_path / "config.yaml", {"inner": "preset"})

    cfg = RootConfigWithNestedSubconfigRef.from_file(
        tmp_path / "config.yaml",
        overrides={"inner.leaf": "leaf_b"},
    )

    assert cfg.inner.leaf.name == "leaf-b"
    assert cfg.inner.leaf.size == 3


def test_base_config_can_override_nested_subconfig_reference_and_nested_value(
    tmp_path: Path,
) -> None:
    (tmp_path / "inner").mkdir()
    (tmp_path / "inner" / "leaf").mkdir()
    _write_yaml(
        tmp_path / "inner" / "preset.yaml",
        {"name": "inner", "leaf": "leaf_a"},
    )
    _write_yaml(
        tmp_path / "inner" / "leaf" / "leaf_a.yaml",
        {"name": "leaf-a", "size": 2},
    )
    _write_yaml(
        tmp_path / "inner" / "leaf" / "leaf_b.yaml",
        {"name": "leaf-b", "size": 3},
    )
    _write_yaml(tmp_path / "config.yaml", {"inner": "preset"})

    cfg = RootConfigWithNestedSubconfigRef.from_file(
        tmp_path / "config.yaml",
        overrides={"inner.leaf": "leaf_b", "inner.leaf.size": 10},
    )

    assert cfg.inner.leaf.name == "leaf-b"
    assert cfg.inner.leaf.size == 10


def test_base_config_from_file_does_not_mutate_input_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(
        config_path,
        {
            "inner": {"name": "base", "size": 2},
            "seed": 42,
        },
    )
    overrides: dict[str, Any] = {
        "inner": {"name": "override", "size": 3},
        "inner.size": 4,
    }

    _ = RootConfig.from_file(config_path, overrides=overrides)

    assert overrides["inner"] == {"name": "override", "size": 3}


def test_base_config_raises_informative_error_for_invalid_override_path(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(
        config_path,
        {
            "inner": {"name": "base", "size": 2},
            "seed": 42,
        },
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid override path 'seed\.value': 'seed' is not a mapping",
    ):
        RootConfig.from_file(config_path, overrides={"seed.value": 1})


def test_base_config_warns_when_override_value_matches_existing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(
        config_path,
        {
            "inner": {"name": "base", "size": 2},
            "seed": 42,
        },
    )

    with caplog.at_level(logging.WARNING):
        cfg = RootConfig.from_file(config_path, overrides={"seed": 42})

    assert cfg.seed == 42
    assert (
        "Override 'seed' has the same value (42) as the existing config" in caplog.text
    )
