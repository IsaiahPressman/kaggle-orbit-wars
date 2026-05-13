from pathlib import Path

import pytest
import torch
from owl.train import FullConfig, PPOConfig
from owl.train.utils import autocast_context

_REPO_ROOT = Path(__file__).parents[3]


def test_ppo_config_validates_with_pydantic() -> None:
    config = PPOConfig.model_validate(
        {
            "horizon": 4,
            "segments_per_minibatch": 2,
            "ppo_epochs": 3,
            "gamma": 0.9,
        }
    )

    assert config.horizon == 4
    assert config.segments_per_minibatch == 2
    assert config.ppo_epochs == 3
    assert config.gamma == pytest.approx(0.9)
    assert config.checkpoint_freq is None
    assert config.eval_replay_games == 0

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        PPOConfig(gamma=-0.1)
    with pytest.raises(ValueError, match="greater than or equal to 1000"):
        PPOConfig(checkpoint_freq=999)
    with pytest.raises(ValueError, match="eval_replay_games must be even"):
        PPOConfig(eval_replay_games=1)
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        PPOConfig(ppo_epochs=0)

    assert PPOConfig(dtype="bfloat16").dtype == "bfloat16"

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PPOConfig(removed_field=True)


def test_full_config_accepts_nested_discriminated_configs() -> None:
    config = FullConfig.model_validate(
        {
            "env": {
                "n_envs": 2,
                "obs_spec": {"obs_spec": "entity_based", "max_entities": 45},
                "action_spec": {
                    "action_spec": "pure",
                    "max_per_planet_launches": 1,
                    "min_fleet_size": 4,
                },
            },
            "model": {
                "model_arch": "stateless_transformer_v1",
                "embed_dim": 32,
                "depth": 1,
                "n_heads": 4,
            },
            "optimizer": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "lr_schedule": {
                    "warmup_steps": 2,
                    "decay_steps": 10,
                    "lr_min_ratio": 0.1,
                },
            },
            "rl": {
                "horizon": 4,
                "segments_per_minibatch": 2,
            },
        }
    )

    assert config.env.n_envs == 2
    assert config.env.action_spec.min_fleet_size == 4
    assert config.optimizer.learning_rate == pytest.approx(0.001)
    assert config.optimizer.lr_schedule is not None
    assert config.optimizer.lr_schedule.warmup_steps == 2
    assert config.rl.segments_per_minibatch == 2
    assert config.runtime.n_runtime_gpus == 1


def test_full_config_defaults_to_single_launch_actions() -> None:
    config = FullConfig.model_validate(
        {
            "env": {
                "n_envs": 2,
            },
            "model": {
                "model_arch": "stateless_transformer_v1",
                "embed_dim": 32,
                "depth": 1,
                "n_heads": 4,
            },
            "optimizer": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
            },
            "rl": {
                "horizon": 4,
            },
        }
    )

    assert config.env.action_spec.max_per_planet_launches == 1
    assert config.env.action_spec.min_fleet_size == 6


def test_full_config_accepts_single_launch_training_actions() -> None:
    config = FullConfig.model_validate(
        {
            "env": {
                "n_envs": 2,
                "action_spec": {
                    "action_spec": "pure",
                    "max_per_planet_launches": 1,
                },
            },
            "model": {
                "model_arch": "stateless_transformer_v1",
                "embed_dim": 32,
                "depth": 1,
                "n_heads": 4,
            },
            "optimizer": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
            },
            "rl": {
                "horizon": 4,
            },
        }
    )

    assert config.env.action_spec.max_per_planet_launches == 1


def test_full_config_rejects_model_owned_obs_spec() -> None:
    with pytest.raises(
        ValueError,
        match=r"Extra inputs are not permitted",
    ):
        FullConfig.model_validate(
            {
                "env": {
                    "n_envs": 2,
                    "action_spec": {
                        "action_spec": "pure",
                        "max_per_planet_launches": 1,
                    },
                },
                "model": {
                    "model_arch": "stateless_transformer_v1",
                    "obs_spec": {},
                    "embed_dim": 32,
                    "depth": 1,
                    "n_heads": 4,
                },
                "optimizer": {
                    "optimizer": "adamw",
                    "learning_rate": 0.001,
                },
                "rl": {
                    "horizon": 4,
                },
            }
        )


def test_full_config_accepts_discrete_target_model_action_spec() -> None:
    config = FullConfig.model_validate(
        {
            "env": {
                "n_envs": 2,
                "action_spec": {
                    "action_spec": "discrete_targets",
                    "max_per_planet_launches": 1,
                },
            },
            "model": {
                "model_arch": "stateless_transformer_v1",
                "actor": {"action_spec": "discrete_targets"},
                "embed_dim": 32,
                "depth": 1,
                "n_heads": 4,
            },
            "optimizer": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
            },
            "rl": {
                "horizon": 4,
            },
        }
    )

    assert config.env.action_spec.action_spec == "discrete_targets"
    assert config.model.actor.action_spec == "discrete_targets"


def test_full_config_accepts_discrete_target_bins_model_action_spec() -> None:
    config = FullConfig.model_validate(
        {
            "env": {
                "n_envs": 2,
                "action_spec": {
                    "action_spec": "discrete_target_bins",
                    "n_bins": 11,
                },
            },
            "model": {
                "model_arch": "stateless_transformer_v1",
                "actor": {"action_spec": "discrete_target_bins", "n_bins": 11},
                "embed_dim": 32,
                "depth": 1,
                "n_heads": 4,
            },
            "optimizer": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
            },
            "rl": {
                "horizon": 4,
            },
        }
    )

    assert config.env.action_spec.action_spec == "discrete_target_bins"
    assert config.model.actor.action_spec == "discrete_target_bins"


def test_full_config_rejects_discrete_target_bins_n_bins_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="model actor n_bins must match env action_spec n_bins",
    ):
        FullConfig.model_validate(
            {
                "env": {
                    "n_envs": 2,
                    "action_spec": {
                        "action_spec": "discrete_target_bins",
                        "n_bins": 11,
                    },
                },
                "model": {
                    "model_arch": "stateless_transformer_v1",
                    "actor": {"action_spec": "discrete_target_bins", "n_bins": 7},
                    "embed_dim": 32,
                    "depth": 1,
                    "n_heads": 4,
                },
                "optimizer": {
                    "optimizer": "adamw",
                    "learning_rate": 0.001,
                },
                "rl": {
                    "horizon": 4,
                },
            }
        )


def test_full_config_rejects_rl_env_count() -> None:
    with pytest.raises(
        ValueError,
        match=r"Extra inputs are not permitted",
    ):
        FullConfig.model_validate(
            {
                "env": {
                    "n_envs": 2,
                },
                "model": {
                    "model_arch": "stateless_transformer_v1",
                    "embed_dim": 32,
                    "depth": 1,
                    "n_heads": 4,
                },
                "optimizer": {
                    "optimizer": "adamw",
                    "learning_rate": 0.001,
                },
                "rl": {
                    "horizon": 4,
                    "n_envs": 3,
                },
            }
        )


@pytest.mark.parametrize("config_path", sorted((_REPO_ROOT / "configs").glob("*.yaml")))
def test_training_config_files_load(config_path: Path) -> None:
    _ = FullConfig.from_file(config_path)


def test_autocast_context_respects_dtype_config() -> None:
    with autocast_context(PPOConfig(dtype="float32"), torch.device("cpu")):
        assert not torch.is_autocast_enabled("cpu")

    with autocast_context(PPOConfig(dtype="bfloat16"), torch.device("cpu")):
        assert torch.is_autocast_enabled("cpu")
