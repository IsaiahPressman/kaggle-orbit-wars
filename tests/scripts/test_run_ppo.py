from __future__ import annotations

import importlib.util
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    MAX_COMETS,
    MAX_PLANETS,
    ActionPureConfig,
    ObsBatch,
    ObsV1Config,
)
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


def _config_with_envs(n_envs: int) -> FullConfig:
    cfg = _full_config()
    return cfg.model_copy(
        update={
            "env": cfg.env.model_copy(update={"n_envs": n_envs}),
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
        self.model = torch.nn.Linear(1, 1)
        self.device = torch.device("cpu")

    def train_iteration(self) -> dict[str, float]:
        self.iterations += 1
        if self.fail:
            raise RuntimeError("training failed")
        return {"loss": 1.0}

    def write_checkpoint(
        self,
        path: Path,
        *,
        env_steps: int,
    ) -> None:
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


def test_max_runtime_hours_converts_to_seconds() -> None:
    assert run_ppo._max_runtime_seconds(None) is None
    assert run_ppo._max_runtime_seconds(1.5) == 5400.0


def test_validate_args_rejects_non_positive_runtime_hours() -> None:
    with pytest.raises(ValueError, match="--max-runtime-hours must be positive"):
        run_ppo._validate_args(Namespace(max_env_steps=None, max_runtime_hours=0.0))


def test_evaluate_against_last_best_uses_eval_mode_no_grad_and_eval_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config_with_envs(4)
    current_model = torch.nn.Linear(1, 1)
    last_best_model = torch.nn.Linear(1, 1)
    current_model.train()
    last_best_model.eval()
    seen_eval_sizes: list[tuple[int, int]] = []
    perf_times = iter([10.0, 14.0])

    def fake_evaluate_player_count(
        **kwargs: object,
    ) -> tuple[object, dict[str, list[float]], int]:
        assert kwargs["current_model"] is current_model
        assert kwargs["last_best_model"] is last_best_model
        assert not current_model.training
        assert not last_best_model.training
        assert not torch.is_grad_enabled()
        seen_eval_sizes.append((kwargs["n_games"], kwargs["n_envs"]))
        stats = run_ppo._EvalStats.empty()
        stats.add_game_result(run_ppo.MODEL_CURRENT)
        return (
            stats,
            {
                "game_length_mean": [12.0],
                "terminal_episodes_2p": [1.0],
            },
            6,
        )

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_player_count",
        fake_evaluate_player_count,
    )
    monkeypatch.setattr(run_ppo.time, "perf_counter", lambda: next(perf_times))

    metrics = run_ppo._evaluate_against_last_best(
        current_model=current_model,
        last_best_model=last_best_model,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert metrics["eval/win_rate_against_last_best"] == pytest.approx(1.0)
    assert metrics["eval/game_length_mean"] == pytest.approx(12.0)
    assert metrics["eval/terminal_episodes_2p"] == pytest.approx(2.0)
    assert metrics["eval/terminal_episodes"] == pytest.approx(2.0)
    assert metrics["time/eval_seconds"] == pytest.approx(4.0)
    assert metrics["perf/eval_sps"] == pytest.approx(3.0)
    assert seen_eval_sizes == [(2, 2), (2, 2)]
    assert current_model.training
    assert not last_best_model.training


def test_evaluate_against_last_best_splits_eval_replay_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config_with_envs(4)
    cfg = cfg.model_copy(
        update={"rl": cfg.rl.model_copy(update={"eval_replay_games": 2})}
    )
    seen_replays: list[tuple[int, Path | None]] = []

    def fake_evaluate_player_count(
        **kwargs: object,
    ) -> tuple[object, dict[str, list[float]], int]:
        seen_replays.append(
            (
                kwargs["replay_games"],
                kwargs["replay_output_path"],
            )
        )
        stats = run_ppo._EvalStats.empty()
        stats.add_game_result(run_ppo.MODEL_CURRENT)
        return stats, {}, 1

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_player_count",
        fake_evaluate_player_count,
    )

    run_ppo._evaluate_against_last_best(
        current_model=torch.nn.Linear(1, 1),
        last_best_model=torch.nn.Linear(1, 1),
        cfg=cfg,
        device=torch.device("cpu"),
        replay_dir=tmp_path,
    )

    assert seen_replays == [
        (1, tmp_path / "eval_2p.jsonl"),
        (1, tmp_path / "eval_4p.jsonl"),
    ]


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
            actions = run_ppo.ModelActions(
                launch=torch.full(shape, self.launch_value, dtype=torch.bool),
                ships=torch.full(shape, self.ship_value, dtype=torch.int64),
                angle=torch.zeros(shape, dtype=torch.float32),
            )
            return SimpleNamespace(actions=actions)

    obs = ObsBatch(
        planets=torch.zeros((1, 1, 1)),
        orbiting_planets=torch.zeros((1, 1), dtype=torch.bool),
        fleets=torch.zeros((1, 1, 1)),
        comets=torch.zeros((1, 1, 1)),
        entity_mask=torch.zeros((1, 1), dtype=torch.bool),
        still_playing=torch.ones((1, 4), dtype=torch.bool),
        global_features=torch.zeros((1, 1)),
        can_act=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
        max_launch=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64),
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


def test_run_training_loop_writes_periodic_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = _FakeTrainer()
    logger = _FakeLogger()
    eval_calls = 0

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
    )

    assert env_steps == 1600
    assert trainer.checkpoints == [(tmp_path / "checkpoint_00_000_001_600.pt", 1600)]
    assert [step for _metrics, step in logger.logged] == [800, 1600, 1600]
    assert logger.logged[-1][0] == {"eval/win_rate_against_last_best": 0.25}
    assert eval_calls == 1
    assert "model/trainable_parameters" not in logger.logged[0][0]
    assert "trainable_parameters" not in logger.logged[0][0]


def test_run_training_loop_saves_last_best_when_eval_clears_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = _FakeTrainer()
    logger = _FakeLogger()

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
    )

    assert env_steps == 1000
    assert trainer.checkpoints == [
        (tmp_path / "checkpoint_00_000_001_000.pt", 1000),
        (tmp_path / "checkpoint_last_best.pt", 1000),
    ]
    assert logger.logged[-1][0]["eval/game_length_mean"] == 12.0


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
            log_mode=LogMode.DEBUG,
            env_steps_per_iteration=8,
            max_env_steps=8,
            max_runtime_seconds=None,
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
    path = tmp_path / "checkpoint.pt"

    trainer.write_checkpoint(
        path,
        env_steps=512,
    )

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["env_steps"] == 512
    assert checkpoint["model"].keys() == model.state_dict().keys()
    assert "state" in checkpoint["optimizer"]
    assert checkpoint["lr_scheduler"] == scheduler.state_dict()
    assert set(checkpoint) == {
        "model",
        "optimizer",
        "lr_scheduler",
        "env_steps",
    }
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()
