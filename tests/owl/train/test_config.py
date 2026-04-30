from pathlib import Path

import pytest
import torch
from owl.train import FullConfig, PPOConfig, ppo

_REPO_ROOT = Path(__file__).parents[3]


def test_ppo_config_validates_with_pydantic() -> None:
    config = PPOConfig.model_validate(
        {
            "horizon": 4,
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
    assert (
        PPOConfig(
            recompute_advantages_each_minibatch=False
        ).recompute_advantages_each_minibatch
        is False
    )


def test_ppo_config_accepts_puffer_vtrace_mode() -> None:
    assert PPOConfig(advantage_mode="puffer_vtrace").advantage_mode == "puffer_vtrace"


def test_ppo_config_rejects_removed_vtrace_mode() -> None:
    old_mode = "gae" + "_vtrace"

    with pytest.raises(ValueError, match="Input should be 'gae' or 'puffer_vtrace'"):
        PPOConfig(advantage_mode=old_mode)


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
            },
        }
    )

    assert config.env.action_spec.max_per_planet_launches == 3
    assert config.model.action_spec.max_per_planet_launches == 3


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
            },
        }
    )

    assert config.env.action_spec.max_per_planet_launches == 1
    assert config.model.action_spec.max_per_planet_launches == 1


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


@pytest.mark.parametrize("preset", ["baseline", "pufferish"])
def test_training_presets_load(preset: str) -> None:
    config = FullConfig.from_file(_REPO_ROOT / "configs" / f"{preset}.yaml")

    assert config.model.action_spec.max_per_planet_launches == (
        config.env.action_spec.max_per_planet_launches
    )
    assert config.model.embed_dim == 256
    assert config.model.depth == 6
    assert config.model.n_heads == 8
    assert config.optimizer.optimizer == "muon"
    assert config.optimizer.adamw_lr == pytest.approx(3e-4)
    assert config.optimizer.muon_lr == pytest.approx(0.015)


def test_training_presets_make_baseline_and_pufferish_modes_explicit() -> None:
    baseline = FullConfig.from_file(_REPO_ROOT / "configs" / "baseline.yaml")
    pufferish = FullConfig.from_file(_REPO_ROOT / "configs" / "pufferish.yaml")

    assert baseline.env.n_envs == 2048
    assert baseline.rl.advantage_mode == "gae"
    assert not baseline.rl.recompute_advantages_each_minibatch
    assert baseline.rl.segment_sampling.sampling == "uniform"
    assert baseline.rl.segment_sampling.segments_per_minibatch == 16

    assert pufferish.env.n_envs == 2048
    assert pufferish.rl.advantage_mode == "puffer_vtrace"
    assert pufferish.rl.recompute_advantages_each_minibatch
    assert pufferish.rl.segment_sampling.sampling == "advantage_priority"
    assert pufferish.rl.segment_sampling.prio_alpha == pytest.approx(0.5)
    assert pufferish.rl.segment_sampling.prio_beta == pytest.approx(0.2)


def test_autocast_context_respects_dtype_config() -> None:
    with ppo.autocast_context(PPOConfig(dtype="float32"), torch.device("cpu")):
        assert not torch.is_autocast_enabled("cpu")

    with ppo.autocast_context(PPOConfig(dtype="bfloat16"), torch.device("cpu")):
        assert torch.is_autocast_enabled("cpu")
