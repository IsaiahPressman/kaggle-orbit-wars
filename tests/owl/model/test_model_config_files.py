from pathlib import Path

import pytest
import yaml
from owl.model import (
    ActorPureConfig,
    ModelConfig,
    RecurrentTransformerV1Config,
    StatelessTransformerV1Config,
    create_model,
)
from owl.model.actor import ActorConfig
from owl.rl import (
    ActionConfig,
    ActionDiscreteTargetsConfig,
    ActionPureConfig,
    EntityBasedConfig,
)
from pydantic import TypeAdapter

_REPO_ROOT = Path(__file__).parents[3]


def _load_model_config_file(config_path: Path) -> ModelConfig:
    with config_path.open(encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    match config_data["model_arch"]:
        case "stateless_transformer_v1":
            return StatelessTransformerV1Config.from_file(config_path)
        case "recurrent_transformer_v1":
            return RecurrentTransformerV1Config.from_file(config_path)
        case model_arch:
            raise ValueError(f"Unknown model_arch: {model_arch}")


@pytest.mark.parametrize(
    "config_path",
    sorted((_REPO_ROOT / "configs" / "model").glob("*.yaml")),
)
def test_model_config_files_load(config_path: Path) -> None:
    _ = _load_model_config_file(config_path)


@pytest.mark.parametrize(
    "config_path",
    sorted((_REPO_ROOT / "configs" / "model" / "actor").glob("*.yaml")),
)
def test_actor_config_files_load(config_path: Path) -> None:
    with config_path.open(encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    _ = TypeAdapter(ActorConfig).validate_python(config_data)


@pytest.mark.parametrize(
    ("filename", "expected_params"),
    [
        ("stateless_transformer_tiny.yaml", 1_207_182),
        ("stateless_transformer_5m_gelu.yaml", 5_532_942),
        ("stateless_transformer_5m_pure.yaml", 5_804_862),
        ("stateless_transformer_20m_gelu.yaml", 20_093_402),
        ("stateless_transformer_20m_swiglu.yaml", 20_914_202),
        ("stateless_transformer_28m.yaml", 27_785_738),
        ("stateless_transformer_152m.yaml", 151_666_970),
        ("recurrent_transformer_5m_gelu.yaml", 5_270_286),
    ],
)
def test_model_config_file_parameter_count(
    filename: str,
    expected_params: int,
) -> None:
    config = _load_model_config_file(_REPO_ROOT / "configs" / "model" / filename)
    action_spec: ActionConfig
    if isinstance(config.actor, ActorPureConfig):
        action_spec = ActionPureConfig()
    else:
        action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    model = create_model(
        config,
        obs_spec=EntityBasedConfig(),
        action_spec=action_spec,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == expected_params
