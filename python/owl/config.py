import copy
import logging
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Literal, Self, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict

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
        overrides_by_depth: dict[int, dict[str, Any]] = {}
        for field_path, value in overrides_map.items():
            path_parts = _split_override_field_path(field_path)
            overrides_by_depth.setdefault(len(path_parts), {})[field_path] = value

        cls._apply_overrides(config_data, overrides_by_depth.pop(1, {}))
        cls._resolve_all_subconfig_references(
            config_data, root_dir=path.resolve().parent
        )
        for depth in sorted(overrides_by_depth):
            cls._apply_overrides(config_data, overrides_by_depth[depth])
            cls._resolve_all_subconfig_references(
                config_data, root_dir=path.resolve().parent
            )
        return cls.model_validate(config_data)

    def to_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix != ".yaml":
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
    ) -> tuple[dict[str, Any], Path]:
        candidate = root_dir / field_name / value
        if not candidate.suffix:
            candidate = candidate.with_suffix(".yaml")

        if candidate.is_file():
            return cls._load_yaml_mapping(candidate), candidate.parent

        raise ValueError(f"Could not resolve '{field_name}: {value}'")

    @classmethod
    def _iter_base_config_types(cls, annotation: Any) -> list[type["BaseConfig"]]:
        if hasattr(annotation, "__value__"):
            return cls._iter_base_config_types(annotation.__value__)

        origin = get_origin(annotation)
        if origin is Annotated:
            return cls._iter_base_config_types(get_args(annotation)[0])

        if origin in (Union, UnionType):
            config_types: list[type[BaseConfig]] = []
            for arg in get_args(annotation):
                config_types.extend(cls._iter_base_config_types(arg))
            return config_types

        if isinstance(annotation, type) and issubclass(annotation, BaseConfig):
            return [annotation]

        return []

    @classmethod
    def _subconfig_type_for_data(
        cls, field_name: str, config_data: dict[str, Any]
    ) -> type["BaseConfig"] | None:
        config_types = cls._iter_base_config_types(
            cls.model_fields[field_name].annotation
        )
        if len(config_types) == 1:
            return config_types[0]

        for config_type in config_types:
            for tag_name, tag_field in config_type.model_fields.items():
                if tag_name not in config_data:
                    continue

                if get_origin(tag_field.annotation) is not Literal:
                    continue

                if config_data[tag_name] in get_args(tag_field.annotation):
                    return config_type

        return None

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
            if isinstance(field_value, str):
                subconfig_data, subconfig_root_dir = cls._resolve_subconfig_reference(
                    root_dir=root_dir,
                    field_name=field_name,
                    value=field_value,
                )
                config_data[field_name] = subconfig_data
            elif isinstance(field_value, dict):
                subconfig_data = field_value
                subconfig_root_dir = root_dir / field_name
            else:
                continue

            subconfig_type = cls._subconfig_type_for_data(field_name, subconfig_data)
            if subconfig_type is not None:
                subconfig_type._resolve_all_subconfig_references(
                    subconfig_data, root_dir=subconfig_root_dir
                )

    @classmethod
    def _load_yaml_mapping(cls, path: Path) -> dict[str, Any]:
        suffix = path.suffix.lower()
        if suffix != ".yaml":
            raise ValueError(f"Unsupported config file extension '{suffix}'")

        with path.open(encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            raise ValueError(
                f"Config file must contain a top-level mapping. "
                f"Got: {type(config_data)}"
            )

        return config_data
