from pathlib import Path

import pytest
import torch
from owl.train import FullConfig, PPOConfig, ppo

_REPO_ROOT = Path(__file__).parents[3]


def test_ppo_config_validates_with_pydantic() -> None:
    config = PPOConfig.model_validate(
        {
            "horizon": 4,
            "n_envs": 2,
            "segment_sampling": {"segments_per_minibatch": 2},
            "gamma": 0.9,
        }
    )

    assert config.horizon == 4
    assert config.segment_sampling.segments_per_minibatch == 2
    assert config.gamma == pytest.approx(0.9)

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        PPOConfig(gamma=-0.1)
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PPOConfig(segments_per_minibatch=2)

    assert PPOConfig(dtype="bfloat16").dtype == "bfloat16"
    assert PPOConfig(
        recompute_advantages_each_epoch=True
    ).recompute_advantages_each_epoch


def test_ppo_config_accepts_vtrace_mode() -> None:
    assert PPOConfig(advantage_mode="gae_vtrace").advantage_mode == "gae_vtrace"


def test_full_config_accepts_nested_discriminated_configs() -> None:
    config = FullConfig.model_validate(
        {
            "env": {
                "n_envs": 2,
                "obs_spec": {"obs_spec": "obs_v1", "max_entities": 45},
                "action_spec": {
                    "action_spec": "pure",
                    "max_per_planet_launches": 2,
                },
            },
            "model": {
                "model_arch": "stateless_transformer_v1",
                "action_spec": {
                    "action_spec": "pure",
                    "max_per_planet_launches": 2,
                },
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
                "n_envs": 2,
                "segment_sampling": {
                    "sampling": "advantage_priority",
                    "segments_per_minibatch": 2,
                    "prio_alpha": 0.5,
                },
            },
        }
    )

    assert config.env.n_envs == 2
    assert config.optimizer.learning_rate == pytest.approx(0.001)
    assert config.optimizer.lr_schedule is not None
    assert config.optimizer.lr_schedule.warmup_steps == 2
    assert config.rl.segment_sampling.sampling == "advantage_priority"


def test_full_config_defaults_to_multi_launch_actions() -> None:
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
                "n_envs": 2,
            },
        }
    )

    assert config.env.action_spec.max_per_planet_launches == 3
    assert config.model.action_spec.max_per_planet_launches == 3


def test_full_config_rejects_single_launch_training_actions() -> None:
    with pytest.raises(
        ValueError,
        match=r"training requires env\.action_spec\.max_per_planet_launches > 1",
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
                    "action_spec": {
                        "action_spec": "pure",
                        "max_per_planet_launches": 1,
                    },
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
                    "n_envs": 2,
                },
            }
        )


def test_full_config_rejects_mismatched_model_and_env_action_specs() -> None:
    with pytest.raises(
        ValueError,
        match=r"model\.action_spec\.max_per_planet_launches must match "
        r"env\.action_spec\.max_per_planet_launches",
    ):
        FullConfig.model_validate(
            {
                "env": {
                    "n_envs": 2,
                    "action_spec": {
                        "action_spec": "pure",
                        "max_per_planet_launches": 2,
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
                    "n_envs": 2,
                },
            }
        )


def test_full_config_rejects_mismatched_rl_and_env_counts() -> None:
    with pytest.raises(
        ValueError,
        match=r"rl\.n_envs must match env\.n_envs \(2\), got 3",
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


@pytest.mark.parametrize("preset", ["debug", "baseline", "pufferish"])
def test_training_presets_load(preset: str) -> None:
    config = FullConfig.from_file(_REPO_ROOT / "configs" / "train" / f"{preset}.yaml")

    assert config.env.n_envs == config.rl.n_envs
    assert config.env.action_spec.max_per_planet_launches > 1
    assert config.model.action_spec.max_per_planet_launches == (
        config.env.action_spec.max_per_planet_launches
    )


def test_training_presets_make_debug_baseline_and_pufferish_modes_explicit() -> None:
    debug = FullConfig.from_file(_REPO_ROOT / "configs" / "train" / "debug.yaml")
    baseline = FullConfig.from_file(_REPO_ROOT / "configs" / "train" / "baseline.yaml")
    pufferish = FullConfig.from_file(
        _REPO_ROOT / "configs" / "train" / "pufferish.yaml"
    )

    assert debug.env.n_envs == 1
    assert debug.rl.horizon == 16
    assert debug.rl.update_epochs == 1
    assert debug.rl.segment_sampling.segments_per_minibatch == 1

    assert baseline.rl.advantage_mode == "gae"
    assert not baseline.rl.recompute_advantages_each_epoch
    assert baseline.rl.segment_sampling.sampling == "uniform"
    assert baseline.rl.segment_sampling.segments_per_minibatch == 64

    assert pufferish.rl.advantage_mode == "gae_vtrace"
    assert pufferish.rl.recompute_advantages_each_epoch
    assert pufferish.rl.segment_sampling.sampling == "advantage_priority"
    assert pufferish.rl.segment_sampling.prio_alpha == pytest.approx(0.5)
    assert pufferish.rl.segment_sampling.prio_beta == pytest.approx(0.2)


def test_autocast_context_respects_dtype_config() -> None:
    with ppo.autocast_context(PPOConfig(dtype="float32"), torch.device("cpu")):
        assert not torch.is_autocast_enabled("cpu")

    with ppo.autocast_context(PPOConfig(dtype="bfloat16"), torch.device("cpu")):
        assert torch.is_autocast_enabled("cpu")
