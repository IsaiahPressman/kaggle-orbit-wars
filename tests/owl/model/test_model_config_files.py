from pathlib import Path

import pytest
import torch
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
    EntityBasedCrossAttnV1Config,
    EntityBasedExtV2Config,
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
    ("filename", "obs_spec", "expected_params"),
    [
        ("stateless_transformer_2m.yaml", EntityBasedExtV2Config(), 1_810_318),
        ("stateless_transformer_6m.yaml", EntityBasedExtV2Config(), 5_679_118),
        ("stateless_transformer_6m_deep.yaml", EntityBasedExtV2Config(), 5_833_118),
        (
            "stateless_transformer_6m_cross_attn.yaml",
            EntityBasedCrossAttnV1Config(),
            5_968_014,
        ),
        ("stateless_transformer_6m_pure.yaml", EntityBasedExtV2Config(), 5_951_038),
        ("stateless_transformer_11m.yaml", EntityBasedExtV2Config(), 10_950_158),
        ("stateless_transformer_21m_gelu.yaml", EntityBasedExtV2Config(), 20_529_914),
        (
            "stateless_transformer_21m_swiglu.yaml",
            EntityBasedExtV2Config(),
            21_231_770,
        ),
        ("stateless_transformer_28m.yaml", EntityBasedExtV2Config(), 28_123_130),
        ("stateless_transformer_153m.yaml", EntityBasedExtV2Config(), 152_891_930),
        (
            "stateless_transformer_200m_d38.yaml",
            EntityBasedExtV2Config(),
            200_162_330,
        ),
        (
            "stateless_transformer_200m_d60.yaml",
            EntityBasedExtV2Config(),
            200_861_338,
        ),
        (
            "stateless_transformer_1B.yaml",
            EntityBasedExtV2Config(),
            1_002_055_706,
        ),
        ("recurrent_transformer_5m.yaml", EntityBasedExtV2Config(), 5_416_462),
    ],
)
def test_model_config_file_parameter_count(
    filename: str,
    obs_spec: EntityBasedExtV2Config | EntityBasedCrossAttnV1Config,
    expected_params: int,
) -> None:
    config = _load_model_config_file(_REPO_ROOT / "configs" / "model" / filename)
    action_spec: ActionConfig
    if isinstance(config.actor, ActorPureConfig):
        action_spec = ActionPureConfig()
    else:
        action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    with torch.device("meta"):
        model = create_model(
            config,
            obs_spec=obs_spec,
            action_spec=action_spec,
        )

    assert sum(parameter.numel() for parameter in model.parameters()) == expected_params


@pytest.mark.parametrize(
    ("filename", "expected_depth"),
    [
        ("stateless_transformer_200m_d38.yaml", 38),
        ("stateless_transformer_200m_d60.yaml", 60),
    ],
)
def test_200m_model_config_depths(filename: str, expected_depth: int) -> None:
    config = _load_model_config_file(_REPO_ROOT / "configs" / "model" / filename)

    assert isinstance(config, StatelessTransformerV1Config)
    assert config.depth == expected_depth
