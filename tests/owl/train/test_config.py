from pathlib import Path

import pytest
import torch
from owl.checkpoint_quantization import NF4_G128_LSQ
from owl.model import (
    RecurrentTransformerV1,
    RecurrentTransformerV1Config,
    StatelessTransformerV1,
)
from owl.model.stateless_transformer_v1 import StatelessTransformerV1Config
from owl.rl import ActionDiscreteTargetsConfig, ActionPureConfig, EntityBasedConfig
from owl.train import FullConfig, PPOConfig
from owl.train.optimizer import AdamWConfig
from owl.train.utils import (
    autocast_context,
    configure_model_compile,
    configure_torch,
)

_REPO_ROOT = Path(__file__).parents[3]


def test_ppo_config_validates_with_pydantic() -> None:
    config = PPOConfig.model_validate(
        {
            "horizon": 4,
            "segments_per_minibatch": 2,
            "gradient_accumulation_steps": 2,
            "ppo_epochs": 3,
            "gamma": 0.9,
        }
    )

    assert config.horizon == 4
    assert config.segments_per_minibatch == 2
    assert config.gradient_accumulation_steps == 2
    assert config.ppo_epochs == 3
    assert config.gamma == pytest.approx(0.9)
    assert config.checkpoint_freq is None
    assert config.eval_replay_games == 0
    assert config.teacher_mode is None
    assert config.teacher_init is None
    assert config.teacher_kl_coef == pytest.approx(0.001)
    assert config.teacher_value_coef == pytest.approx(0.001)
    assert config.teacher_schedule.mode == "none"
    assert config.teacher_segments_per_minibatch == 32
    assert config.ppo_clip_mode == "per_player"
    assert config.model_compile == "trunk"
    assert config.model_compile_mode == "max-autotune-no-cudagraphs"

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        PPOConfig(gamma=-0.1)
    with pytest.raises(ValueError, match="greater than or equal to 1000"):
        PPOConfig(checkpoint_freq=999)
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        PPOConfig(ppo_epochs=0)
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        PPOConfig(gradient_accumulation_steps=0)

    assert PPOConfig(dtype="bfloat16").dtype == "bfloat16"
    assert PPOConfig(model_compile="none").model_compile == "none"
    assert PPOConfig(model_compile="trunk").model_compile == "trunk"
    assert PPOConfig(model_compile_mode="default").model_compile_mode == "default"
    assert PPOConfig(eval_replay_games=1).eval_replay_games == 1
    assert PPOConfig(teacher_mode="last_best").teacher_mode == "last_best"
    assert PPOConfig(
        teacher_mode="fixed",
        teacher_init=Path("teacher/checkpoint.pt"),
    ).teacher_init == Path("teacher/checkpoint.pt")
    teacher_schedule = PPOConfig(
        teacher_schedule={
            "mode": "linear_decay",
            "decay_steps": 100,
            "decay_min_ratio": 0.25,
        }
    ).teacher_schedule
    assert teacher_schedule.mode == "linear_decay"
    assert teacher_schedule.decay_steps == 100
    assert teacher_schedule.decay_min_ratio == pytest.approx(0.25)
    assert PPOConfig(ppo_clip_mode="per_entity").ppo_clip_mode == "per_entity"

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PPOConfig(removed_field=True)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        PPOConfig(teacher_kl_coef=-0.1)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        PPOConfig(teacher_value_coef=-0.1)
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PPOConfig(teacher_schedule={"mode": "none", "decay_steps": 100})
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        PPOConfig(
            teacher_schedule={
                "mode": "linear_decay",
                "decay_steps": 0,
                "decay_min_ratio": 0.25,
            }
        )
    with pytest.raises(ValueError, match="less than 1"):
        PPOConfig(
            teacher_schedule={
                "mode": "linear_decay",
                "decay_steps": 100,
                "decay_min_ratio": 1.0,
            }
        )
    assert (
        PPOConfig(teacher_segments_per_minibatch=16).teacher_segments_per_minibatch
        == 16
    )
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        PPOConfig(teacher_segments_per_minibatch=0)
    with pytest.raises(ValueError, match="Input should be 'last_best' or 'fixed'"):
        PPOConfig(teacher_mode="latest")
    with pytest.raises(ValueError, match="teacher_init is required"):
        PPOConfig(teacher_mode="fixed")
    with pytest.raises(
        ValueError,
        match="Input should be 'per_player' or 'per_entity'",
    ):
        PPOConfig(ppo_clip_mode="per_source")
    with pytest.raises(ValueError, match="Input should be 'none', 'mlp' or 'trunk'"):
        PPOConfig(model_compile="attention")
    with pytest.raises(
        ValueError,
        match=(
            "Input should be 'default', 'reduce-overhead', 'max-autotune' or "
            "'max-autotune-no-cudagraphs'"
        ),
    ):
        PPOConfig(model_compile_mode="fast")


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
                "lora": {
                    "rank": 8,
                    "alpha_scale": 2.0,
                    "target_modules": ["q", "v"],
                    "target_block_count": 1,
                    "target_value_head": True,
                    "target_policy_head": True,
                    "roundtrip_quantization": NF4_G128_LSQ,
                },
            },
            "optimizer": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "lr_schedule": {
                    "schedule": "linear_warmup_cosine_decay",
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
    assert isinstance(config.model, StatelessTransformerV1Config)
    assert config.model.lora is not None
    assert config.model.lora.roundtrip_quantization == NF4_G128_LSQ
    assert config.model.lora.rank == 8
    assert config.model.lora.alpha_scale == pytest.approx(2.0)
    assert config.model.lora.target_value_head is True
    assert config.model.lora.target_policy_head is True
    assert config.rl.segments_per_minibatch == 2
    assert config.runtime.n_runtime_gpus == 1


def test_full_config_accepts_win_only_reward_with_matching_value_mode() -> None:
    config = FullConfig.model_validate(
        {
            "env": {
                "n_envs": 2,
                "reward_mode": "win_only",
            },
            "model": {
                "model_arch": "stateless_transformer_v1",
                "embed_dim": 32,
                "depth": 1,
                "n_heads": 4,
                "value_mode": "win_only",
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

    assert config.env.reward_mode == "win_only"
    assert config.model.value_mode == "win_only"


def test_full_config_rejects_win_only_reward_without_matching_value_mode() -> None:
    with pytest.raises(ValueError, match=r"requires model\.value_mode='win_only'"):
        FullConfig.model_validate(
            {
                "env": {
                    "n_envs": 2,
                    "reward_mode": "win_only",
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


def test_full_config_rejects_win_only_value_mode_for_other_rewards() -> None:
    with pytest.raises(ValueError, match=r"requires env\.reward_mode='win_only'"):
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
                    "value_mode": "win_only",
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


def test_full_config_resolves_lora_subconfig_override() -> None:
    config = FullConfig.from_file(
        _REPO_ROOT / "configs/stateless_200m.yaml",
        overrides={
            "model.lora": "2p_200m_qv_r16",
            "env.two_player_weight": 1.0,
        },
    )

    assert isinstance(config.model, StatelessTransformerV1Config)
    assert config.model.lora is not None
    assert config.model.lora.rank == 16
    assert config.model.lora.alpha_scale == pytest.approx(1.0)
    assert config.model.lora.target_modules == ("q", "v")
    assert config.model.lora.target_value_head is True
    assert config.model.lora.target_policy_head is True
    assert config.env.two_player_weight == pytest.approx(1.0)


def test_stateless_200m_2p_lora_config_uses_lora_finetuning_settings() -> None:
    config = FullConfig.from_file(_REPO_ROOT / "configs/stateless_200m_2p_lora.yaml")

    assert config.env.two_player_weight == pytest.approx(1.0)
    assert isinstance(config.model, StatelessTransformerV1Config)
    assert config.model.embed_dim == 768
    assert config.model.depth == 38
    assert config.model.lora is not None
    assert config.model.lora.rank == 16
    assert config.model.lora.target_value_head is True
    assert config.model.lora.target_policy_head is True
    assert isinstance(config.optimizer, AdamWConfig)
    assert config.optimizer.learning_rate == pytest.approx(2.0e-4)
    assert config.optimizer.weight_decay == pytest.approx(0.0)
    assert config.rl.teacher_mode == "last_best"


def test_full_config_rejects_lora_without_any_target() -> None:
    with pytest.raises(ValueError, match="lora must target at least one"):
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
                    "lora": {"rank": 2, "target_modules": []},
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


def test_full_config_rejects_unknown_lora_roundtrip_quantization() -> None:
    with pytest.raises(ValueError, match="Input should be"):
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
                    "lora": {
                        "rank": 2,
                        "roundtrip_quantization": "nf2_g128_lsq",
                    },
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


def test_full_config_rejects_lora_on_recurrent_model() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        FullConfig.model_validate(
            {
                "env": {
                    "n_envs": 2,
                    "action_spec": {
                        "action_spec": "discrete_targets",
                        "max_per_planet_launches": 1,
                    },
                },
                "model": {
                    "model_arch": "recurrent_transformer_v1",
                    "embed_dim": 32,
                    "depth": 1,
                    "n_heads": 4,
                    "actor": {"action_spec": "discrete_targets"},
                    "lora": {"rank": 2},
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


def test_full_config_rejects_more_eval_replays_than_envs() -> None:
    with pytest.raises(ValueError, match=r"eval_replay_games must be <= env\.n_envs"):
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
                    "eval_replay_games": 3,
                },
            }
        )


def test_full_config_accepts_adam_optimizer_config() -> None:
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
                "optimizer": "adam",
                "learning_rate": 0.0001,
                "betas": [0.9, 0.999],
                "eps": 1.0e-5,
                "weight_decay": 0.0,
                "lr_schedule": {
                    "schedule": "linear_warmup_cosine_decay",
                    "warmup_steps": 2,
                    "decay_steps": 10,
                    "lr_min_ratio": 0.02,
                },
            },
            "rl": {
                "horizon": 4,
            },
        }
    )

    assert config.optimizer.optimizer == "adam"
    assert config.optimizer.learning_rate == pytest.approx(0.0001)
    assert config.optimizer.betas == pytest.approx((0.9, 0.999))
    assert config.optimizer.lr_schedule is not None
    assert config.optimizer.lr_schedule.lr_min_ratio == pytest.approx(0.02)


def test_full_config_accepts_recurrent_transformer_discrete_targets() -> None:
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
                "model_arch": "recurrent_transformer_v1",
                "embed_dim": 32,
                "depth": 1,
                "n_heads": 4,
                "recurrence_mode": "include_planets",
                "actor": {"action_spec": "discrete_targets"},
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

    assert config.model.model_arch == "recurrent_transformer_v1"
    assert isinstance(config.model, RecurrentTransformerV1Config)
    assert config.model.recurrence_mode == "include_planets"
    assert config.model.actor.launch_mode == "binary"


def test_full_config_rejects_recurrent_transformer_non_binary_launch_mode() -> None:
    with pytest.raises(ValueError, match="binary launch mode"):
        FullConfig.model_validate(
            {
                "env": {
                    "n_envs": 2,
                    "action_spec": {
                        "action_spec": "discrete_targets",
                        "max_per_planet_launches": 1,
                    },
                },
                "model": {
                    "model_arch": "recurrent_transformer_v1",
                    "embed_dim": 32,
                    "depth": 1,
                    "n_heads": 4,
                    "actor": {
                        "action_spec": "discrete_targets",
                        "launch_mode": "target_token",
                    },
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


def test_configure_torch_uses_legacy_tf32_flags_for_inductor() -> None:
    configure_torch()

    assert torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32
    assert torch.backends.cudnn.benchmark


def test_configure_model_compile_compiles_only_transformer_mlps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[object, ...], dict[str, object]]] = []

    def fake_compile(
        module: torch.nn.Module,
        *args: object,
        **kwargs: object,
    ) -> None:
        calls.append((id(module), args, kwargs))

    monkeypatch.setattr(torch.nn.Module, "compile", fake_compile)
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=32, depth=2, n_heads=4, mlp_ratio=1.0),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    state_keys = set(model.state_dict())
    expected_module_ids = {id(block.mlp) for block in model.blocks}

    compiled = configure_model_compile(model, PPOConfig(model_compile="mlp"))

    assert compiled == 2
    assert set(model.state_dict()) == state_keys
    assert {call[0] for call in calls} == expected_module_ids
    assert all(call[1] == () for call in calls)
    assert all(
        call[2] == {"mode": "max-autotune-no-cudagraphs", "dynamic": True}
        for call in calls
    )


def test_configure_model_compile_includes_player_count_adapter_mlps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[object, ...], dict[str, object]]] = []

    def fake_compile(
        module: torch.nn.Module,
        *args: object,
        **kwargs: object,
    ) -> None:
        calls.append((id(module), args, kwargs))

    monkeypatch.setattr(torch.nn.Module, "compile", fake_compile)
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(
            embed_dim=32,
            depth=2,
            n_heads=4,
            mlp_ratio=1.0,
            player_count_adapters_enabled=True,
            player_count_adapter_blocks=2,
        ),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    expected_module_ids = {
        id(block.mlp)
        for adapter in model.player_count_adapters.values()
        for block in adapter.blocks
    }

    compiled = configure_model_compile(model, PPOConfig(model_compile="mlp"))

    assert compiled == len(expected_module_ids)
    assert {call[0] for call in calls} == expected_module_ids


def test_configure_model_compile_compiles_stateless_transformer_trunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, str]] = []

    def fake_compile_transformer_trunk(
        self: StatelessTransformerV1,
        *,
        mode: str,
    ) -> int:
        calls.append((id(self), mode))
        return 1

    monkeypatch.setattr(
        StatelessTransformerV1,
        "compile_transformer_trunk",
        fake_compile_transformer_trunk,
        raising=False,
    )
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=32, depth=2, n_heads=4, mlp_ratio=1.0),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    state_keys = set(model.state_dict())

    compiled = configure_model_compile(
        model,
        PPOConfig(model_compile="trunk", model_compile_mode="default"),
    )

    assert compiled == 1
    assert set(model.state_dict()) == state_keys
    assert calls == [(id(model), "default")]


def test_configure_model_compile_rejects_recurrent_trunk() -> None:
    model = RecurrentTransformerV1(
        RecurrentTransformerV1Config(embed_dim=32, depth=2, n_heads=4, mlp_ratio=1.0),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
    )

    with pytest.raises(RuntimeError, match="does not support recurrent_transformer_v1"):
        configure_model_compile(model, PPOConfig(model_compile="trunk"))


def test_configure_model_compile_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[torch.nn.Module] = []

    def fake_compile(
        module: torch.nn.Module,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        calls.append(module)

    monkeypatch.setattr(torch.nn.Module, "compile", fake_compile)
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=32, depth=1, n_heads=4, mlp_ratio=1.0),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )

    compiled = configure_model_compile(model, PPOConfig(model_compile="none"))

    assert compiled == 0
    assert calls == []


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


def test_full_config_rejects_minibatch_accumulation_that_does_not_divide_envs() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "env.n_envs must be divisible by rl.segments_per_minibatch "
            r"\* rl.gradient_accumulation_steps"
        ),
    ):
        FullConfig.model_validate(
            {
                "env": {
                    "n_envs": 4,
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
                    "segments_per_minibatch": 2,
                    "gradient_accumulation_steps": 3,
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
