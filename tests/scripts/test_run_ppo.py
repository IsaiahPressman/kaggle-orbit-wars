from __future__ import annotations

import importlib.util
import sys
import time
from argparse import Namespace
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from owl.checkpoint_quantization import (
    NF4_G128_LSQ,
    dequantize_model_state_dict,
    quantize_model_state_dict,
)
from owl.model import LoRALinear
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    MAX_COMETS,
    MAX_PLANETS,
    ActionPureConfig,
    EntityBasedConfig,
    EntityBasedExtV1Config,
    ObsBatch,
    PureActionMask,
)
from owl.train import FullConfig, PPOTrainer
from owl.train.distributed import DistributedContext
from owl.train.logging import LogMode
from owl.train.optimizer import CompositeOptimizer

_RUN_PPO_PATH = Path(__file__).parents[2] / "scripts" / "run_ppo.py"
_RUN_PPO_SPEC = importlib.util.spec_from_file_location("run_ppo", _RUN_PPO_PATH)
assert _RUN_PPO_SPEC is not None
assert _RUN_PPO_SPEC.loader is not None
run_ppo = importlib.util.module_from_spec(_RUN_PPO_SPEC)
sys.modules["run_ppo"] = run_ppo
_RUN_PPO_SPEC.loader.exec_module(run_ppo)


def _full_config(*, checkpoint_freq: int | None = None) -> FullConfig:
    return FullConfig.model_validate(
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
                "lr_schedule": {
                    "schedule": "linear_warmup_cosine_decay",
                    "warmup_steps": 1,
                    "decay_steps": 4,
                    "lr_min_ratio": 0.1,
                },
            },
            "rl": {
                "horizon": 4,
                "checkpoint_freq": checkpoint_freq,
            },
        }
    )


def _config_with_envs(n_envs: int) -> FullConfig:
    cfg = _full_config()
    return cfg.model_copy(
        update={
            "env": cfg.env.model_copy(update={"n_envs": n_envs}),
        }
    )


def _config_with_resume_shape(
    *,
    n_envs: int,
    segments_per_minibatch: int,
    gradient_accumulation_steps: int,
    runtime_gpus: int,
    eval_replay_games: int = 0,
) -> FullConfig:
    cfg = _full_config()
    return FullConfig.model_validate(
        {
            **cfg.model_dump(mode="python"),
            "env": {
                **cfg.env.model_dump(mode="python"),
                "n_envs": n_envs,
            },
            "rl": {
                **cfg.rl.model_dump(mode="python"),
                "segments_per_minibatch": segments_per_minibatch,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "eval_replay_games": eval_replay_games,
            },
            "runtime": {
                **cfg.runtime.model_dump(mode="python"),
                "n_runtime_gpus": runtime_gpus,
            },
        }
    )


def _distributed_context(world_size: int) -> run_ppo.DistributedContext:
    return run_ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=world_size,
        initialized=False,
    )


def _is_lora_adapter_key(key: str) -> bool:
    return key.endswith((".lora_down", ".lora_up"))


def _base_lora_state(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value for key, value in state_dict.items() if not _is_lora_adapter_key(key)
    }


def _assert_quantized_state_equal(left: object, right: object) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_quantized_state_equal(left[key], right[key])
        return
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
        return
    assert left == right


def test_apply_lora_for_config_wraps_stateless_training_model() -> None:
    cfg = FullConfig.model_validate(
        {
            "env": {
                "n_envs": 2,
            },
            "model": {
                "model_arch": "stateless_transformer_v1",
                "embed_dim": 32,
                "depth": 2,
                "n_heads": 4,
                "lora": {
                    "rank": 4,
                    "target_modules": ["q", "v"],
                    "target_block_count": 1,
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
    model = run_ppo._create_model(
        cfg.model,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
    )

    application = run_ppo._apply_lora_for_config(model, cfg.model)

    assert application is not None
    assert application.module_count == 2
    assert application.trainable_parameters == 512
    assert not isinstance(model.blocks[0].attn.q, LoRALinear)
    assert isinstance(model.blocks[1].attn.q, LoRALinear)
    assert isinstance(model.blocks[1].attn.v, LoRALinear)
    assert all(
        name.endswith((".lora_down", ".lora_up")) == parameter.requires_grad
        for name, parameter in model.named_parameters()
    )


def test_create_training_model_roundtrip_quantizes_lora_base_model() -> None:
    base_cfg = _full_config()
    cfg = FullConfig.model_validate(
        {
            **base_cfg.model_dump(mode="python"),
            "model": {
                **base_cfg.model.model_dump(mode="python"),
                "lora": {
                    "rank": 2,
                    "target_modules": ["q"],
                    "roundtrip_quantization": NF4_G128_LSQ,
                },
            },
        }
    )
    torch.manual_seed(123)
    reference_model = run_ppo._create_model(
        cfg.model,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
    )
    reference_model.reset_parameters()
    assert run_ppo._apply_lora_for_config(reference_model, cfg.model) is not None
    reference_state = reference_model.state_dict()
    reference_base_state = _base_lora_state(reference_state)
    expected_quantized = quantize_model_state_dict(
        reference_base_state,
        NF4_G128_LSQ,
    )
    expected_base_state = dequantize_model_state_dict(expected_quantized)

    torch.manual_seed(123)
    model, application = run_ppo._create_training_model_for_config(
        cfg,
        device=torch.device("cpu"),
        reset_parameters=True,
    )

    assert application is not None
    actual_state = model.state_dict()
    for key, expected_tensor in expected_base_state.items():
        assert torch.equal(actual_state[key], expected_tensor)
    for key, expected_tensor in reference_state.items():
        if _is_lora_adapter_key(key):
            assert torch.equal(actual_state[key], expected_tensor)
    actual_quantized = quantize_model_state_dict(
        _base_lora_state(actual_state),
        NF4_G128_LSQ,
    )
    _assert_quantized_state_equal(actual_quantized, expected_quantized)


def test_create_eval_model_can_skip_lora_base_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_cfg = _full_config()
    cfg = FullConfig.model_validate(
        {
            **base_cfg.model_dump(mode="python"),
            "model": {
                **base_cfg.model.model_dump(mode="python"),
                "lora": {
                    "rank": 2,
                    "target_modules": ["q"],
                    "roundtrip_quantization": NF4_G128_LSQ,
                },
            },
        }
    )

    def fail_roundtrip(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("LoRA base roundtrip should be skipped")

    monkeypatch.setattr(
        run_ppo,
        "_roundtrip_lora_base_quantization_for_config",
        fail_roundtrip,
    )

    model = run_ppo._create_eval_model_for_config(
        cfg,
        device=torch.device("cpu"),
        roundtrip_lora_base=False,
    )

    assert isinstance(model.blocks[0].attn.q, LoRALinear)
    assert not model.training


def test_load_model_weights_allows_base_checkpoint_for_lora_model(
    tmp_path: Path,
) -> None:
    base_cfg = _full_config()
    lora_cfg = FullConfig.model_validate(
        {
            **base_cfg.model_dump(mode="python"),
            "model": {
                **base_cfg.model.model_dump(mode="python"),
                "lora": {
                    "rank": 2,
                    "target_modules": ["q", "v"],
                },
            },
        }
    )
    base_model = run_ppo._create_model(
        base_cfg.model,
        obs_spec=base_cfg.env.obs_spec,
        action_spec=base_cfg.env.action_spec,
    )
    lora_model = run_ppo._create_model(
        lora_cfg.model,
        obs_spec=lora_cfg.env.obs_spec,
        action_spec=lora_cfg.env.action_spec,
    )
    assert run_ppo._apply_lora_for_config(lora_model, lora_cfg.model) is not None
    path = tmp_path / "base_checkpoint.pt"
    torch.save(
        {
            "model": base_model.state_dict(),
            "env_steps": 64,
            "player_step_total": 5,
            "total_games_played": 7,
            "total_active_entities": 11,
            "wandb_run_id": "run-abc",
        },
        path,
    )
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = lora_model
    trainer.device = torch.device("cpu")
    trainer.player_step_total = 0
    trainer.total_games_played = 0
    trainer.total_active_entities = 0

    metadata = trainer.load_model_weights(path)

    assert metadata.env_steps == 64
    assert trainer.player_step_total == 5
    assert trainer.total_games_played == 7
    assert trainer.total_active_entities == 11
    lora_state = lora_model.state_dict()
    for key, value in base_model.state_dict().items():
        assert torch.equal(lora_state[key], value)


def test_lora_fresh_launch_rejects_loading_optimizer_state() -> None:
    base_cfg = _full_config()
    cfg = FullConfig.model_validate(
        {
            **base_cfg.model_dump(mode="python"),
            "model": {
                **base_cfg.model.model_dump(mode="python"),
                "lora": {"rank": 2},
            },
        }
    )
    launch = run_ppo.FreshLaunch(
        config_path=Path("config.yaml"),
        output_dir=Path("runs"),
        overrides={},
        load_model_weights_path=Path("checkpoint.pt"),
        load_model_weights_mode="model_and_optimizer",
    )

    with pytest.raises(ValueError, match="model_and_optimizer is not supported"):
        run_ppo._validate_lora_launch_config(cfg, launch)


def test_initial_last_best_model_wraps_lora_and_supports_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: teacher_mode=last_best fine-tuning seeded from a non-LoRA base.
    # last_best must share the LoRA-wrapped student architecture so that a later
    # _refresh_eval_model_from_weights can copy the student's adapter-bearing
    # state dict into it instead of crashing on unexpected LoRA keys.
    monkeypatch.setattr(run_ppo, "_compile_eval_model", lambda *_a, **_k: None)
    base_cfg = _full_config()
    base_model = run_ppo._create_model(
        base_cfg.model,
        obs_spec=base_cfg.env.obs_spec,
        action_spec=base_cfg.env.action_spec,
    )
    base_model.reset_parameters()
    base_checkpoint = tmp_path / "base_checkpoint.pt"
    torch.save({"model": base_model.state_dict()}, base_checkpoint)
    # last_best validation reads the teacher checkpoint's sibling config.yaml.
    base_cfg.to_file(tmp_path / "config.yaml")

    student_cfg = FullConfig.model_validate(
        {
            **base_cfg.model_dump(mode="python"),
            "model": {
                **base_cfg.model.model_dump(mode="python"),
                "lora": {"rank": 2, "target_modules": ["q", "v"]},
            },
            "rl": {
                **base_cfg.rl.model_dump(mode="python"),
                "teacher_mode": "last_best",
                "teacher_init": str(base_checkpoint),
            },
        }
    )

    last_best = run_ppo._initial_last_best_model(
        student_cfg, device=torch.device("cpu")
    )

    assert last_best is not None
    # last_best is LoRA-wrapped like the student, and its frozen base weights are
    # seeded from the non-LoRA base checkpoint (adapters stay at config init).
    assert isinstance(last_best.blocks[0].attn.q, LoRALinear)
    last_best_state = last_best.state_dict()
    for key, value in base_model.state_dict().items():
        assert torch.equal(last_best_state[key], value)

    # Winning triggers refreshing last_best from the adapter-bearing student; this
    # is the path that previously raised on unexpected LoRA keys.
    student = run_ppo._create_model(
        student_cfg.model,
        obs_spec=student_cfg.env.obs_spec,
        action_spec=student_cfg.env.action_spec,
    )
    student.reset_parameters()
    assert run_ppo._apply_lora_for_config(student, student_cfg.model) is not None
    run_ppo._refresh_eval_model_from_weights(last_best, student)

    refreshed_state = last_best.state_dict()
    for key, value in student.state_dict().items():
        assert torch.equal(refreshed_state[key], value)


class _FakeLogger:
    def __init__(self, *, run_id: str | None = "run-123") -> None:
        self.closed = False
        self.logged: list[tuple[dict[str, float], int]] = []
        self.summary: dict[str, int | float] = {}
        self._run_id = run_id

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self.logged.append((metrics, step))

    def set_summary(self, key: str, value: int | float) -> None:
        self.summary[key] = value

    def close(self) -> None:
        self.closed = True


class _FakeTrainer:
    def __init__(
        self,
        *,
        fail: bool = False,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.fail = fail
        self.metrics = {"loss": 1.0} if metrics is None else metrics
        self.checkpoints: list[tuple[Path, int, str | None]] = []
        self.iterations = 0
        self.model = torch.nn.Linear(1, 1)
        self.device = torch.device("cpu")
        self.teacher_updates: list[tuple[torch.nn.Module | None, bool]] = []
        self.checkpoint_models: dict[Path, torch.nn.Module | None] = {}

    def train_iteration(self) -> dict[str, float]:
        self.iterations += 1
        if self.fail:
            raise RuntimeError("training failed")
        return dict(self.metrics)

    def write_checkpoint(
        self,
        path: Path,
        *,
        env_steps: int,
        wandb_run_id: str | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.checkpoints.append((path, env_steps, wandb_run_id))
        self.checkpoint_models[path] = model

    def set_teacher_model(
        self,
        teacher_model: torch.nn.Module | None,
        *,
        active: bool,
    ) -> None:
        self.teacher_updates.append((teacher_model, active))


def _patch_eval_model_from_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> list[torch.nn.Module]:
    created_models: list[torch.nn.Module] = []

    def fake_create_eval_model_from_weights(
        source_model: torch.nn.Module,
        _cfg: FullConfig,
        *,
        device: torch.device,
    ) -> torch.nn.Module:
        model = torch.nn.Linear(1, 1).to(device)
        model.load_state_dict(source_model.state_dict())
        model.eval()
        created_models.append(model)
        return model

    monkeypatch.setattr(
        run_ppo,
        "_create_eval_model_from_weights",
        fake_create_eval_model_from_weights,
    )
    return created_models


def _write_checkpoint_metadata(path: Path, *, env_steps: int) -> None:
    torch.save(
        {
            "model": {},
            "optimizer": {},
            "lr_scheduler": None,
            "env_steps": env_steps,
            "optimizer_steps": 0,
            "player_step_total": 0,
            "total_games_played": 0,
            "total_active_entities": 0,
            "target_kl_exceeded_total": 0,
            "wandb_run_id": "run-123",
        },
        path,
    )


def test_next_periodic_checkpoint_step_handles_crossed_cadence() -> None:
    assert run_ppo._next_periodic_checkpoint_step(checkpoint_freq=None) is None
    assert run_ppo._next_periodic_checkpoint_step(checkpoint_freq=1000) == 1000
    assert (
        run_ppo._next_periodic_checkpoint_step(
            checkpoint_freq=1000,
            env_steps=1256,
        )
        == 2000
    )


def test_format_checkpoint_step_zero_pads_grouped_digits() -> None:
    assert run_ppo._format_checkpoint_step(1_000_000_000) == "01_000_000_000"
    assert run_ppo._format_checkpoint_step(22_000_000) == "00_022_000_000"


def test_should_stop_training_checks_step_and_runtime_limits() -> None:
    assert run_ppo._should_stop_training(
        env_steps=128,
        started_at=time.monotonic(),
        max_env_steps=128,
        max_runtime_seconds=None,
    )
    assert run_ppo._should_stop_training(
        env_steps=1,
        started_at=time.monotonic() - 2.0,
        max_env_steps=None,
        max_runtime_seconds=1.0,
    )
    assert not run_ppo._should_stop_training(
        env_steps=1,
        started_at=time.monotonic(),
        max_env_steps=10,
        max_runtime_seconds=10.0,
    )


def test_should_stop_training_reduces_distributed_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = run_ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=1,
        local_rank=1,
        world_size=2,
        initialized=True,
    )
    calls: list[bool] = []

    def fake_all_reduce_any(
        value: bool,
        _context: run_ppo.DistributedContext,
    ) -> bool:
        assert _context is context
        calls.append(value)
        return True

    monkeypatch.setattr(run_ppo, "all_reduce_any", fake_all_reduce_any)

    assert run_ppo._should_stop_training(
        env_steps=1,
        started_at=time.monotonic(),
        max_env_steps=10,
        max_runtime_seconds=10.0,
        distributed=context,
    )
    assert calls == [False]


def test_max_runtime_hours_converts_to_seconds() -> None:
    assert run_ppo._max_runtime_seconds(None) is None
    assert run_ppo._max_runtime_seconds(1.5) == 5400.0


def test_validate_args_rejects_non_positive_runtime_hours() -> None:
    with pytest.raises(ValueError, match="--max-runtime-hours must be positive"):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=0.0,
                output_dir=Path("runs"),
                overrides=None,
                load_model_weights=None,
                load_model_weights_mode="model_only",
                log_mode=LogMode.WANDB,
            )
        )


def test_validate_args_rejects_debug_resume() -> None:
    with pytest.raises(ValueError, match="resume launches require wandb logging"):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=None,
                overrides=None,
                load_model_weights=None,
                load_model_weights_mode="model_only",
                log_mode=LogMode.DEBUG,
            )
        )


def test_validate_args_rejects_resume_overrides() -> None:
    with pytest.raises(ValueError, match="resume launches cannot use config overrides"):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=None,
                overrides=[["rl.horizon=8"]],
                load_model_weights=None,
                load_model_weights_mode="model_only",
                log_mode=LogMode.WANDB,
            )
        )


def test_validate_args_rejects_resume_load_model_weights() -> None:
    with pytest.raises(
        ValueError,
        match="resume launches cannot use --load-model-weights",
    ):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=None,
                overrides=None,
                load_model_weights=Path("checkpoint.pt"),
                load_model_weights_mode="model_only",
                log_mode=LogMode.WANDB,
            )
        )


def test_validate_args_rejects_load_model_weights_mode_without_checkpoint() -> None:
    with pytest.raises(
        ValueError,
        match="--load-model-weights-mode requires --load-model-weights",
    ):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=Path("runs"),
                overrides=None,
                load_model_weights=None,
                load_model_weights_mode="model_and_optimizer",
                log_mode=LogMode.WANDB,
            )
        )


def test_parse_cli_overrides_flattens_repeated_flags() -> None:
    assert run_ppo._parse_cli_overrides(
        [["rl.horizon=8"], ["env.n_envs=4", "model.depth=2"]]
    ) == {
        "rl.horizon": 8,
        "env.n_envs": 4,
        "model.depth": 2,
    }


def test_resolve_fresh_launch_accepts_load_model_weights(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    output_dir = tmp_path / "runs"

    launch = run_ppo._resolve_launch(
        Namespace(
            target=config_path,
            output_dir=output_dir,
            overrides=[["rl.horizon=8"]],
            load_model_weights=checkpoint_path,
            load_model_weights_mode="model_and_optimizer",
        )
    )

    assert launch == run_ppo.FreshLaunch(
        config_path=config_path,
        output_dir=output_dir,
        overrides={"rl.horizon": 8},
        load_model_weights_path=checkpoint_path,
        load_model_weights_mode="model_and_optimizer",
    )


def test_resolve_teacher_init_path_uses_config_directory(tmp_path: Path) -> None:
    cfg = _full_config()
    cfg = cfg.model_copy(
        update={
            "rl": cfg.rl.model_copy(
                update={"teacher_init": Path("teachers/checkpoint.pt")}
            )
        }
    )

    resolved = run_ppo._resolve_teacher_init_path(cfg, tmp_path / "config.yaml")

    assert resolved.rl.teacher_init == (tmp_path / "teachers/checkpoint.pt").resolve()


def test_teacher_obs_spec_for_student_allows_max_entities_mismatch() -> None:
    student_obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    teacher_obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 5)

    resolved = run_ppo._teacher_obs_spec_for_student(
        teacher_obs_spec,
        student_obs_spec=student_obs_spec,
        checkpoint_path=Path("teacher.pt"),
    )

    assert resolved == student_obs_spec
    assert teacher_obs_spec.max_entities == MAX_PLANETS + MAX_COMETS + 5


def test_teacher_obs_spec_for_student_rejects_other_obs_mismatches() -> None:
    student_obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    teacher_obs_spec = EntityBasedExtV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 5)

    with pytest.raises(ValueError, match="except max_entities"):
        run_ppo._teacher_obs_spec_for_student(
            teacher_obs_spec,
            student_obs_spec=student_obs_spec,
            checkpoint_path=Path("teacher.pt"),
        )


def test_load_teacher_init_model_uses_student_max_entities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student_obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    teacher_obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 5)
    base_cfg = _full_config()
    student_cfg = base_cfg.model_copy(
        update={"env": base_cfg.env.model_copy(update={"obs_spec": student_obs_spec})}
    )
    teacher_cfg = base_cfg.model_copy(
        update={"env": base_cfg.env.model_copy(update={"obs_spec": teacher_obs_spec})}
    )
    checkpoint_path = tmp_path / "teacher" / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.touch()
    teacher_cfg.to_file(checkpoint_path.parent / "config.yaml")
    teacher_model = torch.nn.Linear(1, 1)
    created_obs_specs: list[object] = []

    def fake_create_model(
        _config: object,
        *,
        obs_spec: object,
        action_spec: object,
    ) -> torch.nn.Module:
        del action_spec
        created_obs_specs.append(obs_spec)
        return teacher_model

    monkeypatch.setattr(run_ppo, "_create_model", fake_create_model)
    monkeypatch.setattr(run_ppo, "_load_model_weights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_ppo, "_compile_eval_model", lambda *_args, **_kwargs: None)

    loaded = run_ppo._load_teacher_init_model(
        checkpoint_path,
        student_cfg=student_cfg,
        device=torch.device("cpu"),
    )

    assert loaded is teacher_model
    assert created_obs_specs == [student_obs_spec]


def test_fixed_teacher_fresh_launch_leaves_last_best_unseeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    cfg = cfg.model_copy(
        update={
            "rl": cfg.rl.model_copy(
                update={
                    "teacher_mode": "fixed",
                    "teacher_init": Path("teacher/checkpoint.pt"),
                }
            )
        }
    )
    config_path = tmp_path / "config.yaml"
    cfg.to_file(config_path)
    output_dir = tmp_path / "runs"
    run_dir = output_dir / "run"
    teacher_init_path = (config_path.parent / "teacher/checkpoint.pt").resolve()
    student_model = torch.nn.Linear(1, 1)
    fixed_teacher_model = torch.nn.Linear(1, 1)
    trainer_ref: dict[str, object] = {}
    session_ref: dict[str, object] = {}
    teacher_loads: list[Path] = []

    class FakeEnv:
        def __init__(
            self,
            *,
            n_envs: int,
            obs_spec: object,
            action_spec: object,
            two_player_weight: float,
            pin_memory: bool,
        ) -> None:
            del two_player_weight, pin_memory
            self.n_envs = n_envs
            self.obs_spec = obs_spec
            self.action_spec = action_spec

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.model = kwargs["model"]
            self.teacher_model = kwargs["teacher_model"]
            self.teacher_active = kwargs["teacher_active"]
            self.teacher_updates: list[tuple[torch.nn.Module | None, bool]] = []
            trainer_ref["trainer"] = self

        def set_teacher_model(
            self,
            teacher_model: torch.nn.Module | None,
            *,
            active: bool,
        ) -> None:
            self.teacher_updates.append((teacher_model, active))

    def fake_create_run_dir(output: Path) -> Path:
        assert output == output_dir
        run_dir.mkdir(parents=True)
        return run_dir

    def fake_load_teacher_init_model(
        checkpoint_path: Path,
        *,
        student_cfg: FullConfig,
        device: torch.device,
    ) -> torch.nn.Linear:
        assert student_cfg == cfg.model_copy(
            update={"rl": cfg.rl.model_copy(update={"teacher_init": teacher_init_path})}
        )
        assert device == torch.device("cpu")
        teacher_loads.append(checkpoint_path)
        return fixed_teacher_model

    def fake_run_training_session(**kwargs: object) -> None:
        session_ref.update(kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ppo.py",
            str(config_path),
            str(output_dir),
            "--log-mode",
            "debug",
        ],
    )
    monkeypatch.setattr(run_ppo, "assert_release_build", lambda: None)
    monkeypatch.setattr(run_ppo, "configure_torch", lambda: None)
    monkeypatch.setattr(
        run_ppo,
        "distributed_session",
        lambda: nullcontext(DistributedContext.single_process_cpu()),
    )
    monkeypatch.setattr(run_ppo, "_create_run_dir", fake_create_run_dir)
    monkeypatch.setattr(run_ppo, "VectorizedEnv", FakeEnv)
    monkeypatch.setattr(
        run_ppo,
        "_create_model",
        lambda *_args, **_kwargs: student_model,
    )
    monkeypatch.setattr(run_ppo, "configure_model_compile", lambda *_args: 0)
    monkeypatch.setattr(
        run_ppo,
        "create_optimizer",
        lambda trainer_model, _cfg: torch.optim.SGD(
            trainer_model.parameters(),
            lr=0.1,
        ),
    )
    monkeypatch.setattr(run_ppo, "create_lr_scheduler", lambda *_args: None)
    monkeypatch.setattr(
        run_ppo,
        "_load_teacher_init_model",
        fake_load_teacher_init_model,
    )
    monkeypatch.setattr(run_ppo, "PPOTrainer", FakeTrainer)
    monkeypatch.setattr(run_ppo, "_run_training_session", fake_run_training_session)

    run_ppo.main()

    trainer = trainer_ref["trainer"]
    assert isinstance(trainer, FakeTrainer)
    assert teacher_loads == [teacher_init_path]
    assert trainer.teacher_model is fixed_teacher_model
    assert trainer.teacher_active
    assert trainer.teacher_updates == []
    assert session_ref["last_best_model"] is None


def test_fresh_launch_from_checkpoint_uses_starting_checkpoint_as_teacher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config()
    cfg = cfg.model_copy(
        update={"rl": cfg.rl.model_copy(update={"teacher_mode": "last_best"})}
    )
    config_path = tmp_path / "config.yaml"
    cfg.to_file(config_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    output_dir = tmp_path / "runs"
    run_dir = output_dir / "run"
    student_model = torch.nn.Linear(1, 1)
    student_model.weight.data.fill_(1.0)
    student_model.bias.data.fill_(1.0)
    teacher_model = torch.nn.Linear(1, 1)
    trainer_ref: dict[str, object] = {}
    session_ref: dict[str, object] = {}
    models = iter((student_model, teacher_model))
    compiled_models: list[torch.nn.Module] = []

    class FakeEnv:
        def __init__(
            self,
            *,
            n_envs: int,
            obs_spec: object,
            action_spec: object,
            two_player_weight: float,
            pin_memory: bool,
        ) -> None:
            del two_player_weight, pin_memory
            self.n_envs = n_envs
            self.obs_spec = obs_spec
            self.action_spec = action_spec

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.model = kwargs["model"]
            self.teacher_updates: list[tuple[torch.nn.Module, bool]] = []
            trainer_ref["trainer"] = self

        def load_model_weights(
            self,
            path: Path,
            *,
            load_optimizer: bool = False,
        ) -> run_ppo.PPOCheckpointMetadata:
            assert path == checkpoint_path
            assert not load_optimizer
            loaded_model = self.model
            assert isinstance(loaded_model, torch.nn.Linear)
            loaded_model.weight.data.fill_(7.0)
            loaded_model.bias.data.fill_(7.0)
            return run_ppo.PPOCheckpointMetadata(env_steps=123)

        def set_teacher_model(
            self,
            teacher_model: torch.nn.Module | None,
            *,
            active: bool,
        ) -> None:
            assert teacher_model is not None
            self.teacher_updates.append((teacher_model, active))

    def fake_create_run_dir(output: Path) -> Path:
        assert output == output_dir
        run_dir.mkdir(parents=True)
        return run_dir

    def fake_run_training_session(**kwargs: object) -> None:
        session_ref.update(kwargs)

    def fake_create_model(*_args: object, **_kwargs: object) -> torch.nn.Linear:
        return next(models)

    def fake_configure_model_compile(
        model_arg: torch.nn.Module,
        _cfg: object,
    ) -> int:
        compiled_models.append(model_arg)
        return 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ppo.py",
            str(config_path),
            str(output_dir),
            "--load-model-weights",
            str(checkpoint_path),
            "--log-mode",
            "debug",
        ],
    )
    monkeypatch.setattr(run_ppo, "assert_release_build", lambda: None)
    monkeypatch.setattr(run_ppo, "configure_torch", lambda: None)
    monkeypatch.setattr(
        run_ppo,
        "distributed_session",
        lambda: nullcontext(DistributedContext.single_process_cpu()),
    )
    monkeypatch.setattr(run_ppo, "_create_run_dir", fake_create_run_dir)
    monkeypatch.setattr(run_ppo, "VectorizedEnv", FakeEnv)
    monkeypatch.setattr(run_ppo, "_create_model", fake_create_model)
    monkeypatch.setattr(
        run_ppo, "configure_model_compile", fake_configure_model_compile
    )
    monkeypatch.setattr(
        run_ppo,
        "create_optimizer",
        lambda trainer_model, _cfg: torch.optim.SGD(trainer_model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(run_ppo, "create_lr_scheduler", lambda *_args: None)
    monkeypatch.setattr(run_ppo, "PPOTrainer", FakeTrainer)
    monkeypatch.setattr(run_ppo, "_run_training_session", fake_run_training_session)

    run_ppo.main()

    trainer = trainer_ref["trainer"]
    assert isinstance(trainer, FakeTrainer)
    assert len(trainer.teacher_updates) == 1
    active_teacher_model, active = trainer.teacher_updates[0]
    assert active
    assert active_teacher_model is teacher_model
    assert active_teacher_model is not student_model
    assert active_teacher_model.weight.item() == pytest.approx(7.0)
    assert compiled_models == [student_model, teacher_model]
    assert session_ref["start_env_steps"] == 123
    last_best_model = session_ref["last_best_model"]
    assert isinstance(last_best_model, torch.nn.Linear)
    assert last_best_model is teacher_model
    assert last_best_model.weight.item() == pytest.approx(7.0)


def test_resolve_resume_launch_prefers_final_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    final_checkpoint = run_dir / "checkpoint_final.pt"
    _write_checkpoint_metadata(final_checkpoint, env_steps=20_000)
    _write_checkpoint_metadata(
        run_dir / "checkpoint_00_000_010_000.pt",
        env_steps=10_000,
    )
    (run_dir / "checkpoint_last_best.pt").touch()

    launch = run_ppo._resolve_resume_launch(run_dir)

    assert launch.config_path == run_dir / "config.yaml"
    assert launch.checkpoint_path == final_checkpoint
    assert launch.last_best_checkpoint_path == run_dir / "checkpoint_last_best.pt"


def test_resolve_resume_launch_uses_numbered_when_newer_than_final(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    final_checkpoint = run_dir / "checkpoint_final.pt"
    _write_checkpoint_metadata(final_checkpoint, env_steps=10_000)
    latest_checkpoint = run_dir / "checkpoint_00_000_020_000.pt"
    _write_checkpoint_metadata(latest_checkpoint, env_steps=20_000)
    (run_dir / "checkpoint_last_best.pt").touch()

    launch = run_ppo._resolve_resume_launch(run_dir)

    assert launch.checkpoint_path == latest_checkpoint


def test_resolve_resume_launch_uses_latest_numbered_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    (run_dir / "checkpoint_00_000_010_000.pt").touch()
    latest_checkpoint = run_dir / "checkpoint_00_000_020_000.pt"
    latest_checkpoint.touch()
    (run_dir / "checkpoint_last_best.pt").touch()

    launch = run_ppo._resolve_resume_launch(run_dir)

    assert launch.checkpoint_path == latest_checkpoint


def test_resolve_resume_launch_rejects_missing_last_best(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    (run_dir / "checkpoint_final.pt").touch()

    with pytest.raises(ValueError, match="expected last-best checkpoint"):
        run_ppo._resolve_resume_launch(run_dir)


def test_resolve_resume_launch_uses_adjacent_config_for_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    checkpoint = run_dir / "checkpoint_00_000_020_000.pt"
    checkpoint.touch()
    (run_dir / "checkpoint_last_best.pt").touch()

    launch = run_ppo._resolve_resume_launch(checkpoint)

    assert launch.config_path == run_dir / "config.yaml"
    assert launch.checkpoint_path == checkpoint


def test_resume_wandb_run_id_requires_checkpoint_run_id() -> None:
    metadata = run_ppo.PPOCheckpointMetadata(
        env_steps=1,
        wandb_run_id=None,
    )

    with pytest.raises(ValueError, match="missing wandb_run_id"):
        run_ppo._resume_wandb_run_id(metadata, LogMode.WANDB)


def test_checkpoint_metadata_rejects_positional_fields() -> None:
    with pytest.raises(TypeError):
        run_ppo.PPOCheckpointMetadata(1, "run-abc")


def test_checkpoint_metadata_accepts_current_trainer_checkpoint_schema(
    tmp_path: Path,
) -> None:
    checkpoint = {
        "model": {},
        "optimizer": {},
        "lr_scheduler": None,
        "env_steps": 123,
        "optimizer_steps": 7,
        "player_step_total": 19,
        "total_games_played": 23,
        "total_active_entities": 29,
        "target_kl_exceeded_total": 3,
        "wandb_run_id": "run-abc",
    }

    metadata = run_ppo._checkpoint_metadata(
        checkpoint,
        path=tmp_path / "checkpoint.pt",
    )

    assert metadata.env_steps == 123
    assert metadata.player_step_total == 19
    assert metadata.total_games_played == 23
    assert metadata.total_active_entities == 29
    assert metadata.wandb_run_id == "run-abc"


def test_checkpoint_metadata_defaults_missing_total_active_entities(
    tmp_path: Path,
) -> None:
    checkpoint = {
        "model": {},
        "optimizer": {},
        "lr_scheduler": None,
        "env_steps": 123,
        "optimizer_steps": 7,
        "player_step_total": 19,
        "total_games_played": 23,
        "target_kl_exceeded_total": 3,
        "wandb_run_id": "run-abc",
    }

    metadata = run_ppo._checkpoint_metadata(
        checkpoint,
        path=tmp_path / "checkpoint.pt",
    )

    assert metadata.total_active_entities == 0


def test_evaluate_against_last_best_uses_eval_mode_no_grad_and_eval_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config_with_envs(4)
    current_model = torch.nn.Linear(1, 1)
    last_best_model = torch.nn.Linear(1, 1)
    current_model.train()
    last_best_model.eval()
    seen_eval_sizes: list[tuple[int, int, int]] = []
    perf_times = iter([10.0, 14.0])

    def fake_evaluate_games(
        **kwargs: object,
    ) -> tuple[object, dict[int, object], dict[str, list[float]], int]:
        assert kwargs["current_model"] is current_model
        assert kwargs["last_best_model"] is last_best_model
        assert not current_model.training
        assert not last_best_model.training
        assert not torch.is_grad_enabled()
        seen_eval_sizes.append(
            (kwargs["n_games"], kwargs["n_envs"], kwargs["replay_games"])
        )
        stats = run_ppo._EvalStats.empty()
        stats.add_game_result(run_ppo.MODEL_CURRENT)
        stats.add_game_result(run_ppo.MODEL_LAST_BEST)
        stats_by_count = {
            2: run_ppo._EvalStats.empty(),
            4: run_ppo._EvalStats.empty(),
        }
        stats_by_count[2].add_game_result(run_ppo.MODEL_CURRENT)
        stats_by_count[4].add_game_result(run_ppo.MODEL_LAST_BEST)
        return (
            stats,
            stats_by_count,
            {
                "game_length_mean": [12.0],
                "_neutral_planets_captured_per_game": [1.0],
                "_neutral_comets_captured_per_game": [2.0],
                "_neutral_planet_undershots_per_game": [3.0],
                "_neutral_comet_undershots_per_game": [4.0],
            },
            6,
        )

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_games",
        fake_evaluate_games,
    )
    monkeypatch.setattr(run_ppo.time, "perf_counter", lambda: next(perf_times))

    metrics = run_ppo._evaluate_against_last_best(
        current_model=current_model,
        last_best_model=last_best_model,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert metrics["eval/win_rate_against_last_best"] == pytest.approx(0.5)
    assert metrics["eval/win_rate_against_last_best_2p"] == pytest.approx(1.0)
    assert metrics["eval/win_rate_against_last_best_4p"] == pytest.approx(0.0)
    assert metrics["eval/game_length_mean"] == pytest.approx(12.0)
    assert metrics["eval/neutral_planet_undershot_rate"] == pytest.approx(0.75)
    assert metrics["eval/neutral_comet_undershot_rate"] == pytest.approx(2.0 / 3.0)
    assert "eval/_neutral_planets_captured_per_game" not in metrics
    assert metrics["time/eval_seconds"] == pytest.approx(4.0)
    assert metrics["perf/eval_sps"] == pytest.approx(1.5)
    assert seen_eval_sizes == [(4, 4, 0)]
    assert current_model.training
    assert not last_best_model.training


def test_evaluate_against_last_best_records_weighted_eval_replay_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config_with_envs(4)
    cfg = cfg.model_copy(
        update={"rl": cfg.rl.model_copy(update={"eval_replay_games": 2})}
    )
    seen_replays: list[tuple[int, Path | None]] = []

    def fake_evaluate_games(
        **kwargs: object,
    ) -> tuple[object, dict[int, object], dict[str, list[float]], int]:
        seen_replays.append(
            (
                kwargs["replay_games"],
                kwargs["replay_output_path"],
            )
        )
        stats = run_ppo._EvalStats.empty()
        stats.add_game_result(run_ppo.MODEL_CURRENT)
        stats_by_count = {2: stats, 4: run_ppo._EvalStats.empty()}
        return stats, stats_by_count, {}, 1

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_games",
        fake_evaluate_games,
    )

    run_ppo._evaluate_against_last_best(
        current_model=torch.nn.Linear(1, 1),
        last_best_model=torch.nn.Linear(1, 1),
        cfg=cfg,
        device=torch.device("cpu"),
        replay_dir=tmp_path,
    )

    assert seen_replays == [(2, tmp_path / "eval.jsonl")]


def test_evaluate_against_last_best_omits_empty_player_count_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_evaluate_games(
        **_kwargs: object,
    ) -> tuple[object, dict[int, object], dict[str, list[float]], int]:
        stats = run_ppo._EvalStats.empty()
        stats.add_game_result(run_ppo.MODEL_CURRENT)
        stats_by_count = {
            2: run_ppo._EvalStats.empty(),
            4: run_ppo._EvalStats.empty(),
        }
        stats_by_count[2].add_game_result(run_ppo.MODEL_CURRENT)
        return stats, stats_by_count, {}, 1

    monkeypatch.setattr(run_ppo, "_evaluate_games", fake_evaluate_games)

    metrics = run_ppo._evaluate_against_last_best(
        current_model=torch.nn.Linear(1, 1),
        last_best_model=torch.nn.Linear(1, 1),
        cfg=_config_with_envs(2),
        device=torch.device("cpu"),
    )

    assert metrics["eval/win_rate_against_last_best_2p"] == pytest.approx(1.0)
    assert "eval/win_rate_against_last_best_4p" not in metrics


def test_record_eval_terminal_result_counts_team_ties_as_half_win() -> None:
    stats = run_ppo._EvalStats.empty()

    run_ppo._record_eval_terminal_result(
        stats,
        assignment=torch.tensor([0, 1, 1, 0]),
        start_mask=torch.tensor([True, True, True, True]),
        returns=torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )

    assert stats.model_games == [1, 1]
    assert stats.wins == [0.5, 0.5]
    assert stats.win_rate(run_ppo.MODEL_CURRENT) == pytest.approx(0.5)


def test_assign_eval_models_randomizes_active_player_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignments = torch.full((2, 4), -1, dtype=torch.int64)
    permutations = iter(
        [
            torch.tensor([1, 0]),
            torch.tensor([3, 0, 2, 1]),
        ]
    )

    monkeypatch.setattr(run_ppo.torch, "randperm", lambda _n: next(permutations))

    run_ppo._assign_eval_models(
        assignments,
        0,
        active_slots=torch.tensor([True, True, False, False]),
        player_count=2,
    )
    run_ppo._assign_eval_models(
        assignments,
        1,
        active_slots=torch.tensor([True, True, True, True]),
        player_count=4,
    )

    assert assignments[0].tolist() == [
        run_ppo.MODEL_LAST_BEST,
        run_ppo.MODEL_CURRENT,
        -1,
        -1,
    ]
    assert assignments[1].tolist() == [
        run_ppo.MODEL_CURRENT,
        run_ppo.MODEL_CURRENT,
        run_ppo.MODEL_LAST_BEST,
        run_ppo.MODEL_LAST_BEST,
    ]


def test_eval_actions_for_assignments_uses_stochastic_model_outputs() -> None:
    class FakeModel:
        def __init__(self, *, launch_value: bool, ship_value: int) -> None:
            self.launch_value = launch_value
            self.ship_value = ship_value

        def __call__(
            self,
            obs: ObsBatch,  # noqa: ARG002
            *,
            deterministic: bool = False,
        ) -> SimpleNamespace:
            assert not deterministic
            shape = (1, 4, ACTION_ENTITY_SLOTS, 1)
            actions = run_ppo.PureActions(
                launch=torch.full(shape, self.launch_value, dtype=torch.bool),
                ships=torch.full(shape, self.ship_value, dtype=torch.int64),
                angle=torch.zeros(shape, dtype=torch.float32),
            )
            return SimpleNamespace(actions=actions, next_hidden_state=None)

    obs = ObsBatch(
        planets=torch.zeros((1, 1, 1)),
        orbiting_planets=torch.zeros((1, 1), dtype=torch.bool),
        fleets=torch.zeros((1, 1, 1)),
        comets=torch.zeros((1, 1, 1)),
        entity_mask=torch.zeros((1, 1), dtype=torch.bool),
        still_playing=torch.ones((1, 4), dtype=torch.bool),
        global_features=torch.zeros((1, 1)),
        action_mask=PureActionMask(
            can_act=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
            max_launch=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64),
        ),
    )

    actions = run_ppo._eval_actions_for_assignments(
        obs,
        torch.tensor([[0, 1, 0, 1]]),
        current_model=FakeModel(launch_value=True, ship_value=3),
        last_best_model=FakeModel(launch_value=False, ship_value=7),
        config=Namespace(dtype="float32"),
        device=torch.device("cpu"),
    )

    assert actions.launch[0, :, 0, 0].tolist() == [True, False, True, False]
    assert actions.ships[0, :, 0, 0].tolist() == [3, 7, 3, 7]


def test_evaluate_games_carries_recurrent_hidden_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def obs_batch(n_envs: int) -> ObsBatch:
        return ObsBatch(
            planets=torch.zeros((n_envs, 1, 1)),
            orbiting_planets=torch.zeros((n_envs, 1), dtype=torch.bool),
            fleets=torch.zeros((n_envs, 1, 1)),
            comets=torch.zeros((n_envs, 1, 1)),
            entity_mask=torch.zeros((n_envs, 1), dtype=torch.bool),
            still_playing=torch.tensor(
                [[True, True, False, False] for _ in range(n_envs)],
                dtype=torch.bool,
            ),
            global_features=torch.zeros((n_envs, 1)),
            action_mask=PureActionMask(
                can_act=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
                max_launch=torch.zeros(
                    (n_envs, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64
                ),
            ),
        )

    class FakeEnv:
        def __init__(
            self,
            *,
            n_envs: int,
            two_player_weight: float,
            **_kwargs: object,
        ) -> None:
            assert two_player_weight == pytest.approx(0.25)
            self.n_envs = n_envs
            self.steps = 0

        def reset(self) -> ObsBatch:
            return obs_batch(self.n_envs)

        def step(
            self,
            actions: run_ppo.ActionBundle,  # noqa: ARG002
        ) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
            self.steps += 1
            rewards = torch.zeros((self.n_envs, 4), dtype=torch.float32)
            dones = torch.zeros((self.n_envs, 4), dtype=torch.bool)
            if self.steps == 2:
                dones.fill_(True)
            return obs_batch(self.n_envs), rewards, dones, {}

        def terminal_metrics(self, env_index: int) -> dict[str, float]:  # noqa: ARG002
            return {"game_length_mean": 2.0}

    class RecordingRecurrentModel:
        def __init__(self, *, initial: float, launch_value: bool) -> None:
            self.initial = initial
            self.launch_value = launch_value
            self.seen_hidden: list[torch.Tensor] = []
            self.reset_dones: list[torch.Tensor] = []

        def initial_hidden_state(
            self,
            batch_size: int,
            *,
            device: torch.device,
        ) -> torch.Tensor:
            return torch.full((batch_size,), self.initial, device=device)

        def __call__(
            self,
            obs: ObsBatch,
            *,
            deterministic: bool = False,
            hidden_state: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            assert not deterministic
            assert hidden_state is not None
            self.seen_hidden.append(hidden_state.detach().cpu().clone())
            n_envs = obs.global_features.shape[0]
            shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
            actions = run_ppo.PureActions(
                launch=torch.full(shape, self.launch_value, dtype=torch.bool),
                angle=torch.zeros(shape, dtype=torch.float32),
                ships=torch.ones(shape, dtype=torch.int64),
            )
            return SimpleNamespace(
                actions=actions,
                next_hidden_state=hidden_state + 1.0,
            )

        def reset_hidden_state(
            self,
            hidden_state: torch.Tensor | None,
            dones: torch.Tensor,
        ) -> torch.Tensor | None:
            assert hidden_state is not None
            self.reset_dones.append(dones.detach().cpu().clone())
            keep = ~dones.all(dim=1).to(device=hidden_state.device)
            return hidden_state * keep.to(dtype=hidden_state.dtype)

    monkeypatch.setattr(run_ppo, "VectorizedEnv", FakeEnv)
    current_model = RecordingRecurrentModel(initial=0.0, launch_value=True)
    last_best_model = RecordingRecurrentModel(initial=10.0, launch_value=False)
    base_cfg = _config_with_envs(2)
    cfg = base_cfg.model_copy(
        update={
            "env": base_cfg.env.model_copy(update={"two_player_weight": 0.25}),
        }
    )

    stats, stats_by_player_count, env_metrics, steps = run_ppo._evaluate_games(
        current_model=current_model,
        last_best_model=last_best_model,
        cfg=cfg,
        n_games=2,
        n_envs=2,
        device=torch.device("cpu"),
    )

    assert steps == 4
    assert env_metrics == {"game_length_mean": [2.0, 2.0]}
    assert stats.model_games == [2, 2]
    assert stats_by_player_count[2].model_games == [2, 2]
    assert stats_by_player_count[4].model_games == [0, 0]
    assert [hidden.tolist() for hidden in current_model.seen_hidden] == [
        [0.0, 0.0],
        [1.0, 1.0],
    ]
    assert [hidden.tolist() for hidden in last_best_model.seen_hidden] == [
        [10.0, 10.0],
        [11.0, 11.0],
    ]
    assert [dones.any().item() for dones in current_model.reset_dones] == [
        False,
        True,
    ]


def test_select_actions_handles_discrete_target_bundles() -> None:
    shape = (1, 4, ACTION_ENTITY_SLOTS, 1)
    actions_a = run_ppo.DiscreteTargetActions(
        launch=torch.full(shape, True, dtype=torch.bool),
        target=torch.full(shape, 3, dtype=torch.int64),
        ships=torch.full(shape, 5, dtype=torch.int64),
    )
    actions_b = run_ppo.DiscreteTargetActions(
        launch=torch.full(shape, False, dtype=torch.bool),
        target=torch.full(shape, 7, dtype=torch.int64),
        ships=torch.full(shape, 11, dtype=torch.int64),
    )

    selected = run_ppo._select_actions(
        actions_a,
        actions_b,
        torch.tensor([[True, False, True, False]]),
    )

    assert isinstance(selected, run_ppo.DiscreteTargetActions)
    assert selected.target[0, :, 0, 0].tolist() == [3, 7, 3, 7]
    assert selected.ships[0, :, 0, 0].tolist() == [5, 11, 5, 11]


def test_select_actions_handles_discrete_target_bin_bundles() -> None:
    shape = (1, 4, ACTION_ENTITY_SLOTS)
    actions_a = run_ppo.DiscreteTargetBinActions(
        target=torch.full(shape, 2, dtype=torch.int64),
        fleet_bin=torch.full(shape, 4, dtype=torch.int64),
    )
    actions_b = run_ppo.DiscreteTargetBinActions(
        target=torch.full(shape, 6, dtype=torch.int64),
        fleet_bin=torch.full(shape, 8, dtype=torch.int64),
    )

    selected = run_ppo._select_actions(
        actions_a,
        actions_b,
        torch.tensor([[True, False, True, False]]),
    )

    assert isinstance(selected, run_ppo.DiscreteTargetBinActions)
    assert selected.target[0, :, 0].tolist() == [2, 6, 2, 6]
    assert selected.fleet_bin[0, :, 0].tolist() == [4, 8, 4, 8]


def test_create_model_uses_env_owned_specs() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = run_ppo._create_model(
        _full_config().model,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )

    assert model.obs_spec == obs_spec
    assert model.action_spec == action_spec
    assert model.fleet_proj.in_features == obs_spec.fleet_channels
    assert model.actor.max_per_planet_launches == 1


def test_create_eval_model_from_weights_builds_fresh_compiled_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config()
    source_model = run_ppo._create_model(
        cfg.model,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
    )
    with torch.no_grad():
        for index, parameter in enumerate(source_model.parameters()):
            parameter.fill_(float(index + 1))
    compiled_models: list[torch.nn.Module] = []

    def fake_configure_model_compile(
        model: torch.nn.Module,
        compile_cfg: object,
    ) -> int:
        assert compile_cfg is cfg.rl
        compiled_models.append(model)
        return 0

    monkeypatch.setattr(
        run_ppo, "configure_model_compile", fake_configure_model_compile
    )

    teacher_model = run_ppo._create_eval_model_from_weights(
        source_model,
        cfg,
        device=torch.device("cpu"),
    )

    assert teacher_model is not source_model
    assert not teacher_model.training
    assert compiled_models == [teacher_model]
    for key, source_tensor in source_model.state_dict().items():
        assert torch.equal(teacher_model.state_dict()[key], source_tensor)
    source_param = next(source_model.parameters())
    teacher_param = next(teacher_model.parameters())
    assert teacher_param.data_ptr() != source_param.data_ptr()


def test_refresh_eval_model_from_weights_updates_existing_model() -> None:
    source_model = torch.nn.Linear(1, 1)
    target_model = torch.nn.Linear(1, 1)
    source_model.weight.data.fill_(5.0)
    source_model.bias.data.fill_(7.0)
    target_model.train()

    run_ppo._refresh_eval_model_from_weights(target_model, source_model)

    assert target_model is not source_model
    assert target_model.weight.item() == pytest.approx(5.0)
    assert target_model.bias.item() == pytest.approx(7.0)
    assert not target_model.training


def test_trainable_parameter_count_ignores_frozen_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[1].weight.requires_grad = False

    assert run_ppo._trainable_parameter_count(model) == 10


def test_with_runtime_gpus_records_world_size() -> None:
    cfg = _full_config()

    updated = run_ppo._with_runtime_gpus(cfg, 4)

    assert updated.runtime.n_runtime_gpus == 4
    assert cfg.runtime.n_runtime_gpus == 1


def test_adapt_resume_config_returns_unchanged_config_when_gpu_count_matches() -> None:
    cfg = run_ppo._with_runtime_gpus(_full_config(), 4)

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(4),
    )

    assert updated == cfg


def test_adapt_resume_config_halves_envs_and_accumulation_for_more_gpus() -> None:
    cfg = _config_with_resume_shape(
        n_envs=64,
        segments_per_minibatch=8,
        gradient_accumulation_steps=2,
        runtime_gpus=2,
    )

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(4),
    )

    assert updated.env.n_envs == 32
    assert updated.rl.segments_per_minibatch == 8
    assert updated.rl.gradient_accumulation_steps == 1
    assert updated.runtime.n_runtime_gpus == 4
    assert cfg.env.n_envs == 64
    assert cfg.rl.gradient_accumulation_steps == 2


def test_adapt_resume_config_reduces_minibatch_segments_for_more_gpus() -> None:
    cfg = _config_with_resume_shape(
        n_envs=64,
        segments_per_minibatch=16,
        gradient_accumulation_steps=1,
        runtime_gpus=2,
    )

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(4),
    )

    assert updated.env.n_envs == 32
    assert updated.rl.segments_per_minibatch == 8
    assert updated.rl.gradient_accumulation_steps == 1
    assert updated.runtime.n_runtime_gpus == 4


def test_adapt_resume_config_doubles_envs_and_accumulation_for_fewer_gpus() -> None:
    cfg = _config_with_resume_shape(
        n_envs=32,
        segments_per_minibatch=8,
        gradient_accumulation_steps=1,
        runtime_gpus=4,
    )

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(2),
    )

    assert updated.env.n_envs == 64
    assert updated.rl.segments_per_minibatch == 8
    assert updated.rl.gradient_accumulation_steps == 2
    assert updated.runtime.n_runtime_gpus == 2


def test_adapt_resume_config_reduces_segments_for_fewer_gpus_when_needed() -> None:
    cfg = _config_with_resume_shape(
        n_envs=48,
        segments_per_minibatch=16,
        gradient_accumulation_steps=1,
        runtime_gpus=3,
    )

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(2),
    )

    assert updated.env.n_envs == 72
    assert updated.rl.segments_per_minibatch == 12
    assert updated.rl.gradient_accumulation_steps == 2
    assert updated.runtime.n_runtime_gpus == 2


def test_adapt_resume_config_reduces_segments_when_accumulation_not_exact() -> None:
    cfg = _config_with_resume_shape(
        n_envs=12,
        segments_per_minibatch=3,
        gradient_accumulation_steps=2,
        runtime_gpus=2,
    )

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(3),
    )

    assert updated.env.n_envs == 8
    assert updated.rl.segments_per_minibatch == 2
    assert updated.rl.gradient_accumulation_steps == 2
    assert updated.runtime.n_runtime_gpus == 3


def test_adapt_resume_config_rejects_fractional_env_count() -> None:
    cfg = _config_with_resume_shape(
        n_envs=8,
        segments_per_minibatch=2,
        gradient_accumulation_steps=2,
        runtime_gpus=2,
    )

    with pytest.raises(ValueError, match=r"env.n_envs=.*does not scale evenly"):
        run_ppo._adapt_resume_config_for_runtime_gpus(
            cfg,
            _distributed_context(3),
        )


def test_adapt_resume_config_rejects_odd_derived_env_count() -> None:
    cfg = _config_with_resume_shape(
        n_envs=6,
        segments_per_minibatch=2,
        gradient_accumulation_steps=1,
        runtime_gpus=2,
    )

    with pytest.raises(ValueError, match="n_envs must be even"):
        run_ppo._adapt_resume_config_for_runtime_gpus(
            cfg,
            _distributed_context(4),
        )


def test_adapt_resume_config_rejects_fractional_train_batch() -> None:
    cfg = _config_with_resume_shape(
        n_envs=12,
        segments_per_minibatch=4,
        gradient_accumulation_steps=1,
        runtime_gpus=2,
    )

    with pytest.raises(
        ValueError,
        match=r"rl.segments_per_minibatch .*does not scale evenly",
    ):
        run_ppo._adapt_resume_config_for_runtime_gpus(
            cfg,
            _distributed_context(3),
        )


def test_adapt_resume_config_rejects_eval_replay_games_above_derived_envs() -> None:
    cfg = _config_with_resume_shape(
        n_envs=8,
        segments_per_minibatch=4,
        gradient_accumulation_steps=1,
        runtime_gpus=2,
        eval_replay_games=8,
    )

    with pytest.raises(
        ValueError,
        match=r"rl.eval_replay_games must be <= env.n_envs",
    ):
        run_ppo._adapt_resume_config_for_runtime_gpus(
            cfg,
            _distributed_context(4),
        )


def test_run_training_loop_writes_periodic_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    cfg = cfg.model_copy(
        update={
            "env": cfg.env.model_copy(
                update={
                    "obs_spec": EntityBasedConfig(
                        max_entities=MAX_PLANETS + MAX_COMETS + 3
                    ),
                },
            ),
        },
    )
    trainer = _FakeTrainer(metrics={"loss": 1.0, "train/max_entities": 17.0})
    logger = _FakeLogger()
    eval_calls = 0
    _patch_eval_model_from_weights(monkeypatch)

    def fake_evaluate_against_last_best(**_kwargs: object) -> dict[str, float]:
        nonlocal eval_calls
        eval_calls += 1
        return {"eval/win_rate_against_last_best": 0.25}

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_against_last_best",
        fake_evaluate_against_last_best,
    )

    env_steps = run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        env_steps_per_iteration=800,
        max_env_steps=1600,
        max_runtime_seconds=None,
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    assert env_steps == 1600
    assert trainer.checkpoints == [
        (tmp_path / "checkpoint_00_000_001_600.pt", 1600, None),
        (tmp_path / "checkpoint_last_best.pt", 0, None),
    ]
    assert trainer.checkpoint_models[tmp_path / "checkpoint_last_best.pt"] is not None
    assert [step for _metrics, step in logger.logged] == [800, 1600, 1600]
    assert logger.logged[0][0]["train/max_entities"] == pytest.approx(17.0)
    assert logger.logged[1][0]["train/max_entities"] == pytest.approx(17.0)
    assert logger.logged[-1][0] == {"eval/win_rate_against_last_best": 0.25}
    assert eval_calls == 1
    assert "model/trainable_parameters" not in logger.logged[0][0]
    assert "trainable_parameters" not in logger.logged[0][0]


def test_run_training_loop_resumes_checkpoint_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = _FakeTrainer()
    logger = _FakeLogger()
    _patch_eval_model_from_weights(monkeypatch)
    (tmp_path / "checkpoint_last_best.pt").touch()

    def fake_evaluate_against_last_best(**_kwargs: object) -> dict[str, float]:
        return {"eval/win_rate_against_last_best": 0.25}

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_against_last_best",
        fake_evaluate_against_last_best,
    )

    env_steps = run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        env_steps_per_iteration=800,
        max_env_steps=2000,
        max_runtime_seconds=None,
        start_env_steps=1200,
        wandb_run_id="run-123",
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    assert env_steps == 2000
    assert trainer.checkpoints == [
        (tmp_path / "checkpoint_00_000_002_000.pt", 2000, "run-123")
    ]
    assert [step for _metrics, step in logger.logged] == [2000, 2000]


def test_run_training_loop_returns_immediately_when_resume_reached_step_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = _FakeTrainer()
    logger = _FakeLogger()
    _patch_eval_model_from_weights(monkeypatch)

    env_steps = run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        env_steps_per_iteration=800,
        max_env_steps=1200,
        max_runtime_seconds=None,
        start_env_steps=1200,
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    assert env_steps == 1200
    assert trainer.iterations == 0
    assert trainer.checkpoints == []
    assert logger.logged == []


def test_run_training_loop_saves_last_best_when_eval_clears_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = _FakeTrainer()
    logger = _FakeLogger()
    _patch_eval_model_from_weights(monkeypatch)

    def fake_evaluate_against_last_best(**_kwargs: object) -> dict[str, float]:
        return {
            "eval/win_rate_against_last_best": 0.7,
            "eval/game_length_mean": 12.0,
        }

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_against_last_best",
        fake_evaluate_against_last_best,
    )

    env_steps = run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        env_steps_per_iteration=1000,
        max_env_steps=1000,
        max_runtime_seconds=None,
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    assert env_steps == 1000
    assert trainer.checkpoints == [
        (tmp_path / "checkpoint_00_000_001_000.pt", 1000, None),
        (tmp_path / "checkpoint_last_best.pt", 0, None),
        (tmp_path / "checkpoint_last_best.pt", 1000, None),
    ]
    assert logger.logged[-1][0]["eval/game_length_mean"] == 12.0


def test_run_training_loop_activates_last_best_teacher_after_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    cfg = cfg.model_copy(
        update={"rl": cfg.rl.model_copy(update={"teacher_mode": "last_best"})}
    )
    trainer = _FakeTrainer()
    logger = _FakeLogger()
    created_models = _patch_eval_model_from_weights(monkeypatch)

    def fake_evaluate_against_last_best(**_kwargs: object) -> dict[str, float]:
        return {"eval/win_rate_against_last_best": 0.7}

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_against_last_best",
        fake_evaluate_against_last_best,
    )

    run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        env_steps_per_iteration=1000,
        max_env_steps=1000,
        max_runtime_seconds=None,
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    assert len(trainer.teacher_updates) == 1
    teacher_model, active = trainer.teacher_updates[0]
    assert teacher_model is not None
    assert teacher_model is created_models[0]
    assert len(created_models) == 1
    assert active


def test_run_training_session_sets_trainable_parameter_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _full_config()
    trainer = _FakeTrainer()
    logger = _FakeLogger()

    def create_fake_logger(*_args: object, **_kwargs: object) -> _FakeLogger:
        return logger

    monkeypatch.setattr(run_ppo, "create_logger", create_fake_logger)

    run_ppo._run_training_session(
        trainer=trainer,
        run_dir=tmp_path,
        cfg=cfg,
        log_mode=LogMode.DEBUG,
        env_steps_per_iteration=8,
        max_env_steps=8,
        max_runtime_seconds=None,
        distributed=DistributedContext.single_process_cpu(),
        start_env_steps=16,
        trainable_parameters=123,
        compiled_model_modules=4,
    )

    assert logger.summary == {
        "compiled_model_modules": 4,
        "trainable_parameters": 123,
    }
    assert logger.closed


def test_run_training_session_worker_skips_logger_and_final_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config()
    trainer = _FakeTrainer()
    distributed = run_ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=1,
        local_rank=1,
        world_size=2,
        initialized=False,
    )

    def create_fake_logger(*_args: object, **_kwargs: object) -> _FakeLogger:
        raise AssertionError("worker rank must not create a logger")

    monkeypatch.setattr(run_ppo, "create_logger", create_fake_logger)

    run_ppo._run_training_session(
        trainer=trainer,
        run_dir=tmp_path,
        cfg=cfg,
        log_mode=LogMode.DEBUG,
        env_steps_per_iteration=8,
        max_env_steps=8,
        max_runtime_seconds=None,
        distributed=distributed,
    )

    assert trainer.iterations == 1
    assert trainer.checkpoints == []


def test_run_training_session_closes_logger_and_skips_final_checkpoint_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _full_config()
    trainer = _FakeTrainer()
    logger = _FakeLogger()

    def raise_from_loop(**_kwargs: object) -> int:
        raise RuntimeError("training failed")

    def create_fake_logger(*_args: object, **_kwargs: object) -> _FakeLogger:
        return logger

    monkeypatch.setattr(run_ppo, "create_logger", create_fake_logger)
    monkeypatch.setattr(run_ppo, "_run_training_loop", raise_from_loop)

    with pytest.raises(RuntimeError, match="training failed"):
        run_ppo._run_training_session(
            trainer=trainer,
            run_dir=tmp_path,
            cfg=cfg,
            log_mode=LogMode.DEBUG,
            env_steps_per_iteration=8,
            max_env_steps=8,
            max_runtime_seconds=None,
            distributed=DistributedContext.single_process_cpu(),
        )

    assert logger.closed
    assert trainer.checkpoints == []


def test_ppo_trainer_write_checkpoint_includes_training_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.optimizer_steps = 7
    trainer.player_step_total = 19
    trainer.total_games_played = 23
    trainer.total_active_entities = 29
    trainer.target_kl_exceeded_total = 3
    path = tmp_path / "checkpoint.pt"

    trainer.write_checkpoint(
        path,
        env_steps=512,
        wandb_run_id="run-abc",
    )

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["env_steps"] == 512
    assert checkpoint["optimizer_steps"] == 7
    assert checkpoint["player_step_total"] == 19
    assert checkpoint["total_games_played"] == 23
    assert checkpoint["total_active_entities"] == 29
    assert checkpoint["target_kl_exceeded_total"] == 3
    assert checkpoint["wandb_run_id"] == "run-abc"
    assert checkpoint["model"].keys() == model.state_dict().keys()
    assert "state" in checkpoint["optimizer"]
    assert checkpoint["lr_scheduler"] == scheduler.state_dict()
    assert set(checkpoint) == {
        "model",
        "optimizer",
        "lr_scheduler",
        "env_steps",
        "optimizer_steps",
        "player_step_total",
        "total_games_played",
        "total_active_entities",
        "target_kl_exceeded_total",
        "wandb_run_id",
    }
    metadata = run_ppo._checkpoint_metadata(checkpoint, path=path)
    assert metadata.env_steps == 512
    assert metadata.total_active_entities == 29
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()


def test_ppo_trainer_write_checkpoint_can_save_explicit_model(
    tmp_path: Path,
) -> None:
    trainer_model = torch.nn.Linear(2, 1)
    checkpoint_model = torch.nn.Linear(2, 1)
    for param in trainer_model.parameters():
        param.data.fill_(1.0)
    for param in checkpoint_model.parameters():
        param.data.fill_(3.0)
    optimizer = torch.optim.AdamW(trainer_model.parameters(), lr=0.001)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = trainer_model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = None
    trainer.optimizer_steps = 7
    trainer.player_step_total = 19
    trainer.total_games_played = 23
    trainer.total_active_entities = 29
    trainer.target_kl_exceeded_total = 3
    path = tmp_path / "checkpoint.pt"

    trainer.write_checkpoint(
        path,
        env_steps=0,
        wandb_run_id="run-abc",
        model=checkpoint_model,
    )

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["env_steps"] == 0
    assert torch.equal(
        checkpoint["model"]["weight"],
        checkpoint_model.state_dict()["weight"],
    )
    assert not torch.equal(
        checkpoint["model"]["weight"],
        trainer_model.state_dict()["weight"],
    )


def test_ppo_trainer_load_checkpoint_restores_training_state(tmp_path: Path) -> None:
    src_model = torch.nn.Linear(2, 1)
    dst_model = torch.nn.Linear(2, 1)
    src_optimizer = torch.optim.AdamW(src_model.parameters(), lr=0.001)
    dst_optimizer = torch.optim.AdamW(dst_model.parameters(), lr=0.001)
    src_scheduler = torch.optim.lr_scheduler.LambdaLR(
        src_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    dst_scheduler = torch.optim.lr_scheduler.LambdaLR(
        dst_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    for param in src_model.parameters():
        param.data.fill_(3.0)
    src_optimizer.zero_grad()
    src_model(torch.ones(1, 2)).sum().backward()
    src_optimizer.step()
    src_scheduler.step()
    src_trainer = PPOTrainer.__new__(PPOTrainer)
    src_trainer.model = src_model
    src_trainer.optimizer = src_optimizer
    src_trainer.lr_scheduler = src_scheduler
    src_trainer.optimizer_steps = 11
    src_trainer.player_step_total = 37
    src_trainer.total_games_played = 41
    src_trainer.total_active_entities = 43
    src_trainer.target_kl_exceeded_total = 5
    path = tmp_path / "checkpoint.pt"
    src_trainer.write_checkpoint(path, env_steps=2048, wandb_run_id="run-abc")

    dst_trainer = PPOTrainer.__new__(PPOTrainer)
    dst_trainer.model = dst_model
    dst_trainer.optimizer = dst_optimizer
    dst_trainer.lr_scheduler = dst_scheduler
    dst_trainer.optimizer_steps = 0
    dst_trainer.player_step_total = 0
    dst_trainer.total_games_played = 0
    dst_trainer.total_active_entities = 0
    dst_trainer.target_kl_exceeded_total = 0
    dst_trainer.device = torch.device("cpu")

    metadata = dst_trainer.load_checkpoint(path)

    assert metadata.env_steps == 2048
    assert metadata.player_step_total == 37
    assert metadata.total_games_played == 41
    assert metadata.total_active_entities == 43
    assert metadata.wandb_run_id == "run-abc"
    assert dst_trainer.optimizer_steps == 11
    assert dst_trainer.player_step_total == 37
    assert dst_trainer.total_games_played == 41
    assert dst_trainer.total_active_entities == 43
    assert dst_trainer.target_kl_exceeded_total == 5
    for src_param, dst_param in zip(
        src_model.parameters(),
        dst_model.parameters(),
        strict=True,
    ):
        assert torch.equal(src_param, dst_param)
    assert dst_optimizer.state_dict()["state"]
    assert dst_scheduler.state_dict() == src_scheduler.state_dict()


def test_ppo_trainer_load_checkpoint_defaults_missing_total_active_entities(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = None
    trainer.optimizer_steps = 0
    trainer.player_step_total = 0
    trainer.total_games_played = 0
    trainer.total_active_entities = 17
    trainer.target_kl_exceeded_total = 0
    trainer.device = torch.device("cpu")
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": None,
            "env_steps": 1,
            "optimizer_steps": 0,
            "player_step_total": 5,
            "total_games_played": 7,
            "target_kl_exceeded_total": 0,
            "wandb_run_id": "run-abc",
        },
        path,
    )

    metadata = trainer.load_checkpoint(path)

    assert metadata.total_active_entities == 0
    assert trainer.total_active_entities == 0


def test_ppo_trainer_load_model_weights_keeps_only_logging_counters(
    tmp_path: Path,
) -> None:
    src_model = torch.nn.Linear(2, 1)
    dst_model = torch.nn.Linear(2, 1)
    src_optimizer = torch.optim.AdamW(src_model.parameters(), lr=0.001)
    dst_optimizer = torch.optim.AdamW(dst_model.parameters(), lr=0.001)
    src_scheduler = torch.optim.lr_scheduler.LambdaLR(
        src_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    dst_scheduler = torch.optim.lr_scheduler.LambdaLR(
        dst_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    for param in src_model.parameters():
        param.data.fill_(3.0)
    src_optimizer.zero_grad()
    src_model(torch.ones(1, 2)).sum().backward()
    src_optimizer.step()
    src_scheduler.step()
    src_trainer = PPOTrainer.__new__(PPOTrainer)
    src_trainer.model = src_model
    src_trainer.optimizer = src_optimizer
    src_trainer.lr_scheduler = src_scheduler
    src_trainer.optimizer_steps = 11
    src_trainer.player_step_total = 37
    src_trainer.total_games_played = 41
    src_trainer.total_active_entities = 43
    src_trainer.target_kl_exceeded_total = 5
    path = tmp_path / "checkpoint.pt"
    src_trainer.write_checkpoint(path, env_steps=2048, wandb_run_id="run-abc")

    dst_trainer = PPOTrainer.__new__(PPOTrainer)
    dst_trainer.model = dst_model
    dst_trainer.optimizer = dst_optimizer
    dst_trainer.lr_scheduler = dst_scheduler
    dst_trainer.optimizer_steps = 0
    dst_trainer.player_step_total = 0
    dst_trainer.total_games_played = 0
    dst_trainer.total_active_entities = 0
    dst_trainer.target_kl_exceeded_total = 0
    dst_trainer.device = torch.device("cpu")
    scheduler_state_before = dst_scheduler.state_dict()

    metadata = dst_trainer.load_model_weights(path)

    assert metadata.env_steps == 2048
    assert metadata.player_step_total == 37
    assert metadata.total_games_played == 41
    assert metadata.total_active_entities == 43
    assert metadata.wandb_run_id == "run-abc"
    assert dst_trainer.optimizer_steps == 0
    assert dst_trainer.player_step_total == 37
    assert dst_trainer.total_games_played == 41
    assert dst_trainer.total_active_entities == 43
    assert dst_trainer.target_kl_exceeded_total == 0
    assert not dst_optimizer.state_dict()["state"]
    assert dst_scheduler.state_dict() == scheduler_state_before
    for src_param, dst_param in zip(
        src_model.parameters(),
        dst_model.parameters(),
        strict=True,
    ):
        assert torch.equal(src_param, dst_param)


def test_ppo_trainer_load_model_weights_can_load_optimizer_without_scheduler(
    tmp_path: Path,
) -> None:
    src_model = torch.nn.Linear(2, 1)
    dst_model = torch.nn.Linear(2, 1)
    src_optimizer = torch.optim.AdamW(src_model.parameters(), lr=0.001)
    dst_optimizer = torch.optim.AdamW(dst_model.parameters(), lr=0.01)
    src_scheduler = torch.optim.lr_scheduler.LambdaLR(
        src_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    dst_scheduler = torch.optim.lr_scheduler.LambdaLR(
        dst_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    for param in src_model.parameters():
        param.data.fill_(3.0)
    src_optimizer.zero_grad()
    src_model(torch.ones(1, 2)).sum().backward()
    src_optimizer.step()
    src_scheduler.step()
    src_trainer = PPOTrainer.__new__(PPOTrainer)
    src_trainer.model = src_model
    src_trainer.optimizer = src_optimizer
    src_trainer.lr_scheduler = src_scheduler
    src_trainer.optimizer_steps = 11
    src_trainer.player_step_total = 37
    src_trainer.total_games_played = 41
    src_trainer.total_active_entities = 43
    src_trainer.target_kl_exceeded_total = 5
    path = tmp_path / "checkpoint.pt"
    src_trainer.write_checkpoint(path, env_steps=2048, wandb_run_id="run-abc")

    dst_trainer = PPOTrainer.__new__(PPOTrainer)
    dst_trainer.model = dst_model
    dst_trainer.optimizer = dst_optimizer
    dst_trainer.lr_scheduler = dst_scheduler
    dst_trainer.optimizer_steps = 0
    dst_trainer.player_step_total = 0
    dst_trainer.total_games_played = 0
    dst_trainer.total_active_entities = 0
    dst_trainer.target_kl_exceeded_total = 0
    dst_trainer.device = torch.device("cpu")
    scheduler_state_before = dst_scheduler.state_dict()

    metadata = dst_trainer.load_model_weights(path, load_optimizer=True)

    assert metadata.env_steps == 2048
    assert metadata.player_step_total == 37
    assert metadata.total_games_played == 41
    assert metadata.total_active_entities == 43
    assert metadata.wandb_run_id == "run-abc"
    assert dst_trainer.optimizer_steps == 0
    assert dst_trainer.player_step_total == 37
    assert dst_trainer.total_games_played == 41
    assert dst_trainer.total_active_entities == 43
    assert dst_trainer.target_kl_exceeded_total == 0
    assert dst_optimizer.state_dict()["state"]
    assert dst_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert dst_scheduler.state_dict() == scheduler_state_before
    for src_param, dst_param in zip(
        src_model.parameters(),
        dst_model.parameters(),
        strict=True,
    ):
        assert torch.equal(src_param, dst_param)


def test_ppo_trainer_load_model_weights_rejects_optimizer_state_shape_mismatch(
    tmp_path: Path,
) -> None:
    src_model = torch.nn.Linear(2, 1)
    dst_model = torch.nn.Linear(2, 1)
    src_optimizer = torch.optim.AdamW(src_model.parameters(), lr=0.001)
    dst_optimizer = torch.optim.AdamW(dst_model.parameters(), lr=0.001)
    src_optimizer.zero_grad()
    src_model(torch.ones(1, 2)).sum().backward()
    src_optimizer.step()
    src_trainer = PPOTrainer.__new__(PPOTrainer)
    src_trainer.model = src_model
    src_trainer.optimizer = src_optimizer
    src_trainer.lr_scheduler = None
    src_trainer.optimizer_steps = 11
    src_trainer.player_step_total = 37
    src_trainer.total_games_played = 41
    src_trainer.total_active_entities = 43
    src_trainer.target_kl_exceeded_total = 5
    path = tmp_path / "checkpoint.pt"
    src_trainer.write_checkpoint(path, env_steps=2048, wandb_run_id="run-abc")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(checkpoint, dict)
    optimizer_state = checkpoint["optimizer"]
    assert isinstance(optimizer_state, dict)
    state = optimizer_state["state"]
    assert isinstance(state, dict)
    param_state = next(iter(state.values()))
    assert isinstance(param_state, dict)
    param_state["exp_avg"] = torch.ones(3, 3)
    torch.save(checkpoint, path)

    dst_trainer = PPOTrainer.__new__(PPOTrainer)
    dst_trainer.model = dst_model
    dst_trainer.optimizer = dst_optimizer
    dst_trainer.lr_scheduler = None
    dst_trainer.optimizer_steps = 0
    dst_trainer.player_step_total = 0
    dst_trainer.total_games_played = 0
    dst_trainer.total_active_entities = 0
    dst_trainer.target_kl_exceeded_total = 0
    dst_trainer.device = torch.device("cpu")

    with pytest.raises(ValueError, match="optimizer state tensor 'exp_avg' shape"):
        dst_trainer.load_model_weights(path, load_optimizer=True)


def test_ppo_trainer_load_model_weights_can_load_composite_optimizer(
    tmp_path: Path,
) -> None:
    src_model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    dst_model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    src_optimizer = CompositeOptimizer(
        [
            torch.optim.AdamW(src_model[0].parameters(), lr=0.001),
            torch.optim.AdamW(src_model[1].parameters(), lr=0.002),
        ]
    )
    dst_optimizer = CompositeOptimizer(
        [
            torch.optim.AdamW(dst_model[0].parameters(), lr=0.01),
            torch.optim.AdamW(dst_model[1].parameters(), lr=0.02),
        ]
    )
    for param in src_model.parameters():
        param.data.fill_(3.0)
    src_optimizer.zero_grad()
    src_model(torch.ones(1, 2)).sum().backward()
    src_optimizer.step()
    src_trainer = PPOTrainer.__new__(PPOTrainer)
    src_trainer.model = src_model
    src_trainer.optimizer = src_optimizer
    src_trainer.lr_scheduler = None
    src_trainer.optimizer_steps = 11
    src_trainer.player_step_total = 37
    src_trainer.total_games_played = 41
    src_trainer.total_active_entities = 43
    src_trainer.target_kl_exceeded_total = 5
    path = tmp_path / "checkpoint.pt"
    src_trainer.write_checkpoint(path, env_steps=2048, wandb_run_id="run-abc")

    dst_trainer = PPOTrainer.__new__(PPOTrainer)
    dst_trainer.model = dst_model
    dst_trainer.optimizer = dst_optimizer
    dst_trainer.lr_scheduler = None
    dst_trainer.optimizer_steps = 0
    dst_trainer.player_step_total = 0
    dst_trainer.total_games_played = 0
    dst_trainer.total_active_entities = 0
    dst_trainer.target_kl_exceeded_total = 0
    dst_trainer.device = torch.device("cpu")

    metadata = dst_trainer.load_model_weights(path, load_optimizer=True)

    assert metadata.env_steps == 2048
    assert metadata.total_active_entities == 43
    assert dst_trainer.optimizer_steps == 0
    assert dst_trainer.total_active_entities == 43
    assert dst_trainer.target_kl_exceeded_total == 0
    assert dst_optimizer.optimizers[0].state_dict()["state"]
    assert dst_optimizer.optimizers[1].state_dict()["state"]
    assert dst_optimizer.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.01)
    assert dst_optimizer.optimizers[1].param_groups[0]["lr"] == pytest.approx(0.02)
    for src_param, dst_param in zip(
        src_model.parameters(),
        dst_model.parameters(),
        strict=True,
    ):
        assert torch.equal(src_param, dst_param)


def test_ppo_trainer_load_checkpoint_rejects_scheduler_mismatch(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.optimizer_steps = 0
    trainer.player_step_total = 0
    trainer.total_games_played = 0
    trainer.total_active_entities = 0
    trainer.target_kl_exceeded_total = 0
    trainer.device = torch.device("cpu")
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": None,
            "env_steps": 1,
            "optimizer_steps": 0,
            "player_step_total": 0,
            "total_games_played": 0,
            "target_kl_exceeded_total": 0,
            "wandb_run_id": "run-abc",
        },
        path,
    )

    with pytest.raises(ValueError, match="missing lr_scheduler state"):
        trainer.load_checkpoint(path)
