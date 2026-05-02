from __future__ import annotations

import importlib.util
import time
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from owl.rl import MAX_COMETS, MAX_PLANETS, ActionPureConfig, ObsV1Config
from owl.train import FullConfig, PPOTrainer
from owl.train.logging import LogMode

_RUN_PPO_PATH = Path(__file__).parents[2] / "scripts" / "run_ppo.py"
_RUN_PPO_SPEC = importlib.util.spec_from_file_location("run_ppo", _RUN_PPO_PATH)
assert _RUN_PPO_SPEC is not None
assert _RUN_PPO_SPEC.loader is not None
run_ppo = importlib.util.module_from_spec(_RUN_PPO_SPEC)
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


class _FakeLogger:
    def __init__(self) -> None:
        self.closed = False
        self.logged: list[tuple[dict[str, float], int]] = []
        self.summary: dict[str, int | float] = {}

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self.logged.append((metrics, step))

    def set_summary(self, key: str, value: int | float) -> None:
        self.summary[key] = value

    def close(self) -> None:
        self.closed = True


class _FakeTrainer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.checkpoints: list[tuple[Path, int]] = []
        self.iterations = 0

    def train_iteration(self) -> dict[str, float]:
        self.iterations += 1
        if self.fail:
            raise RuntimeError("training failed")
        return {"loss": 1.0}

    def write_checkpoint(
        self,
        path: Path,
        *,
        config: FullConfig,
        config_path: Path,
        env_steps: int,
    ) -> None:
        del config, config_path
        self.checkpoints.append((path, env_steps))


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


def test_max_runtime_hours_converts_to_seconds() -> None:
    assert run_ppo._max_runtime_seconds(None) is None
    assert run_ppo._max_runtime_seconds(1.5) == 5400.0


def test_validate_args_rejects_non_positive_runtime_hours() -> None:
    with pytest.raises(ValueError, match="--max-runtime-hours must be positive"):
        run_ppo._validate_args(Namespace(max_env_steps=None, max_runtime_hours=0.0))


def test_create_model_uses_env_owned_specs() -> None:
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=2)
    model = run_ppo._create_model(
        _full_config().model,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )

    assert model.obs_spec == obs_spec
    assert model.action_spec == action_spec
    assert model.fleet_proj.in_features == obs_spec.fleet_channels
    assert model.actor.launch_slot_tokens.num_embeddings == 2


def test_trainable_parameter_count_ignores_frozen_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[1].weight.requires_grad = False

    assert run_ppo._trainable_parameter_count(model) == 10


def test_run_training_loop_writes_periodic_checkpoints(tmp_path: Path) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = _FakeTrainer()
    logger = _FakeLogger()

    env_steps = run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        config_path=Path("config.yaml"),
        env_steps_per_iteration=800,
        max_env_steps=1600,
        max_runtime_seconds=None,
    )

    assert env_steps == 1600
    assert trainer.checkpoints == [(tmp_path / "checkpoint-1600.pt", 1600)]
    assert [step for _metrics, step in logger.logged] == [800, 1600]
    assert "model/trainable_parameters" not in logger.logged[0][0]
    assert "trainable_parameters" not in logger.logged[0][0]


def test_run_training_session_sets_trainable_parameter_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _full_config()
    trainer = _FakeTrainer()
    logger = _FakeLogger()

    def create_fake_logger(*_args: object) -> _FakeLogger:
        return logger

    monkeypatch.setattr(run_ppo, "create_logger", create_fake_logger)

    run_ppo._run_training_session(
        trainer=trainer,
        run_dir=tmp_path,
        cfg=cfg,
        config_path=Path("config.yaml"),
        log_mode=LogMode.DEBUG,
        env_steps_per_iteration=8,
        max_env_steps=8,
        max_runtime_seconds=None,
        trainable_parameters=123,
    )

    assert logger.summary == {"trainable_parameters": 123}
    assert logger.closed


def test_run_training_session_closes_logger_and_skips_final_checkpoint_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _full_config()
    trainer = _FakeTrainer()
    logger = _FakeLogger()

    def raise_from_loop(**_kwargs: object) -> int:
        raise RuntimeError("training failed")

    def create_fake_logger(*_args: object) -> _FakeLogger:
        return logger

    monkeypatch.setattr(run_ppo, "create_logger", create_fake_logger)
    monkeypatch.setattr(run_ppo, "_run_training_loop", raise_from_loop)

    with pytest.raises(RuntimeError, match="training failed"):
        run_ppo._run_training_session(
            trainer=trainer,
            run_dir=tmp_path,
            cfg=cfg,
            config_path=Path("config.yaml"),
            log_mode=LogMode.DEBUG,
            env_steps_per_iteration=8,
            max_env_steps=8,
            max_runtime_seconds=None,
        )

    assert logger.closed
    assert trainer.checkpoints == []


def test_ppo_trainer_write_checkpoint_includes_training_state(tmp_path: Path) -> None:
    cfg = _full_config()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    path = tmp_path / "checkpoint.pt"

    trainer.write_checkpoint(
        path,
        config=cfg,
        config_path=Path("config.yaml"),
        env_steps=512,
    )

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["env_steps"] == 512
    assert checkpoint["config"]["env"]["n_envs"] == 2
    assert checkpoint["config_path"] == "config.yaml"
    assert checkpoint["model"].keys() == model.state_dict().keys()
    assert "state" in checkpoint["optimizer"]
    assert checkpoint["lr_scheduler"] == scheduler.state_dict()
    assert set(checkpoint) == {
        "model",
        "optimizer",
        "lr_scheduler",
        "config",
        "config_path",
        "env_steps",
    }
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()
