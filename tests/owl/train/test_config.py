import pytest
import torch
from owl.train import PPOConfig, ppo


def test_ppo_config_validates_with_pydantic() -> None:
    config = PPOConfig.model_validate(
        {
            "horizon": 4,
            "n_envs": 2,
            "segments_per_minibatch": 2,
            "gamma": 0.9,
        }
    )

    assert config.horizon == 4
    assert config.gamma == pytest.approx(0.9)

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        PPOConfig(gamma=-0.1)

    assert PPOConfig(dtype="bfloat16").dtype == "bfloat16"
    assert PPOConfig(
        recompute_advantages_each_epoch=True
    ).recompute_advantages_each_epoch


def test_ppo_config_accepts_vtrace_mode() -> None:
    assert PPOConfig(advantage_mode="gae_vtrace").advantage_mode == "gae_vtrace"


def test_autocast_context_respects_dtype_config() -> None:
    with ppo.autocast_context(PPOConfig(dtype="float32"), torch.device("cpu")):
        assert not torch.is_autocast_enabled("cpu")

    with ppo.autocast_context(PPOConfig(dtype="bfloat16"), torch.device("cpu")):
        assert torch.is_autocast_enabled("cpu")
