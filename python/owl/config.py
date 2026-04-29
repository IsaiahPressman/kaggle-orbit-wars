import copy
import logging
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict

_YAML_SUFFIXES = (".yaml", ".yml")
_logger = logging.getLogger(__name__)


def _split_override_field_path(field_path: str) -> list[str]:
    parts = field_path.split(".")
    if not field_path or any(not part for part in parts):
        raise ValueError(
            f"Invalid override field path '{field_path}'. "
            "Expected dot-separated field names"
        )

    return parts


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def subconfig_dirs(cls) -> set[str]:
        return set()

    @classmethod
    def from_file(cls, path: Path, overrides: dict[str, Any] | None = None) -> Self:
        config_data = cls._load_yaml_mapping(path)
        overrides_map = copy.deepcopy(overrides or {})
        top_level_overrides: dict[str, Any] = {}
        nested_overrides: dict[str, Any] = {}
        for field_path, value in overrides_map.items():
            path_parts = _split_override_field_path(field_path)
            if len(path_parts) == 1:
                top_level_overrides[field_path] = value
            else:
                nested_overrides[field_path] = value

        cls._apply_overrides(config_data, top_level_overrides)
        cls._resolve_all_subconfig_references(
            config_data, root_dir=path.resolve().parent
        )
        cls._apply_overrides(config_data, nested_overrides)
        return cls.model_validate(config_data)

    def to_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix not in _YAML_SUFFIXES:
            raise ValueError(f"Unsupported config file extension '{suffix}'")

        data = self.model_dump(mode="json", round_trip=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

    @classmethod
    def _apply_overrides(
        cls, config_data: dict[str, Any], overrides: dict[str, Any]
    ) -> None:
        for field_path, value in overrides.items():
            path = _split_override_field_path(field_path)

            current_data = config_data
            for field_name in path[:-1]:
                next_data = current_data.get(field_name)
                if not isinstance(next_data, dict):
                    if field_name in current_data:
                        raise ValueError(
                            f"Invalid override path '{field_path}': "
                            f"'{field_name}' is not a mapping "
                            f"(got {type(next_data).__name__})"
                        )

                    next_data = {}
                    current_data[field_name] = next_data

                current_data = next_data

            if path[-1] in current_data and current_data[path[-1]] == value:
                _logger.warning(
                    "Override '%s' has the same value (%s) as the existing config",
                    field_path,
                    value,
                )

            current_data[path[-1]] = value

    @classmethod
    def _resolve_subconfig_reference(
        cls, root_dir: Path, field_name: str, value: str
    ) -> dict[str, Any]:
        candidate = root_dir / field_name / value
        if candidate.suffix:
            candidates = [candidate]
        else:
            candidates = [candidate.with_suffix(suffix) for suffix in _YAML_SUFFIXES]

        for candidate_path in candidates:
            if candidate_path.is_file():
                return cls._load_yaml_mapping(candidate_path)

        attempted = ", ".join(str(path) for path in candidates)
        raise ValueError(
            f"Could not resolve '{field_name}: {value}'. Tried: {attempted}"
        )

    @classmethod
    def _resolve_all_subconfig_references(
        cls, config_data: dict[str, Any], root_dir: Path
    ) -> None:
        for field_name in cls.subconfig_dirs():
            if field_name not in cls.model_fields:
                raise ValueError(
                    f"Unknown subconfig field '{field_name}' on {cls.__name__}"
                )

            field_value = config_data.get(field_name)
            if not isinstance(field_value, str):
                continue

            config_data[field_name] = cls._resolve_subconfig_reference(
                root_dir=root_dir,
                field_name=field_name,
                value=field_value,
            )

    @classmethod
    def _load_yaml_mapping(cls, path: Path) -> dict[str, Any]:
        suffix = path.suffix.lower()
        if suffix not in _YAML_SUFFIXES:
            raise ValueError(f"Unsupported config file extension '{suffix}'")

        with path.open(encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            raise ValueError(
                f"Config file must contain a top-level mapping. "
                f"Got: {type(config_data)}"
            )

        return config_data
