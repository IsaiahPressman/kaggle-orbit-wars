import pytest
import torch
from owl.train import FullConfig, PPOConfig, ppo


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
    assert config.rl.segment_sampling.sampling == "advantage_priority"


def test_autocast_context_respects_dtype_config() -> None:
    with ppo.autocast_context(PPOConfig(dtype="float32"), torch.device("cpu")):
        assert not torch.is_autocast_enabled("cpu")

    with ppo.autocast_context(PPOConfig(dtype="bfloat16"), torch.device("cpu")):
        assert torch.is_autocast_enabled("cpu")
