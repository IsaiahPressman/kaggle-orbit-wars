from __future__ import annotations

import importlib.util
import sys
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
    EntityBasedConfig,
    ObsBatch,
)
from owl.train import FullConfig, PPOTrainer
from owl.train.distributed import DistributedContext
from owl.train.logging import LogMode

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
    ) -> None:
        self.checkpoints.append((path, env_steps, wandb_run_id))


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
                log_mode=LogMode.WANDB,
            )
        )


def test_resolve_resume_launch_prefers_final_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    final_checkpoint = run_dir / "checkpoint_final.pt"
    final_checkpoint.touch()
    (run_dir / "checkpoint_00_000_010_000.pt").touch()
    (run_dir / "checkpoint_last_best.pt").touch()

    launch = run_ppo._resolve_resume_launch(run_dir)

    assert launch.config_path == run_dir / "config.yaml"
    assert launch.checkpoint_path == final_checkpoint
    assert launch.last_best_checkpoint_path == run_dir / "checkpoint_last_best.pt"


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
        optimizer_steps=1,
        player_step_total=1,
        total_games_played=1,
        target_kl_exceeded_total=0,
        wandb_run_id=None,
    )

    with pytest.raises(ValueError, match="missing wandb_run_id"):
        run_ppo._resume_wandb_run_id(metadata, LogMode.WANDB)


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
        if kwargs["player_count"] == 2:
            stats.add_game_result(run_ppo.MODEL_CURRENT)
        else:
            stats.add_game_result(run_ppo.MODEL_LAST_BEST)
        return (
            stats,
            {
                "game_length_mean": [12.0],
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

    assert metrics["eval/win_rate_against_last_best"] == pytest.approx(0.5)
    assert metrics["eval/win_rate_against_last_best_2p"] == pytest.approx(1.0)
    assert metrics["eval/win_rate_against_last_best_4p"] == pytest.approx(0.0)
    assert metrics["eval/game_length_mean"] == pytest.approx(12.0)
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
    assert model.actor.launch_slot_tokens.shape[0] == 1


def test_trainable_parameter_count_ignores_frozen_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[1].weight.requires_grad = False

    assert run_ppo._trainable_parameter_count(model) == 10


def test_with_runtime_gpus_records_world_size() -> None:
    cfg = _full_config()

    updated = run_ppo._with_runtime_gpus(cfg, 4)

    assert updated.runtime.n_runtime_gpus == 4
    assert cfg.runtime.n_runtime_gpus == 1


def test_validate_runtime_gpus_rejects_resume_mismatch() -> None:
    cfg = run_ppo._with_runtime_gpus(_full_config(), 4)
    distributed = run_ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=2,
        initialized=False,
    )

    with pytest.raises(ValueError, match="resume runtime GPU count mismatch"):
        run_ppo._validate_runtime_gpus(cfg, distributed)


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
        (tmp_path / "checkpoint_00_000_001_600.pt", 1600, None)
    ]
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
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = _FakeTrainer()
    logger = _FakeLogger()

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
        (tmp_path / "checkpoint_last_best.pt", 1000, None),
    ]
    assert logger.logged[-1][0]["eval/game_length_mean"] == 12.0


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
        start_env_steps=16,
        trainable_parameters=123,
    )

    assert logger.summary == {"trainable_parameters": 123}
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
        "target_kl_exceeded_total",
        "wandb_run_id",
    }
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()


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
    dst_trainer.target_kl_exceeded_total = 0
    dst_trainer.device = torch.device("cpu")

    metadata = dst_trainer.load_checkpoint(path)

    assert metadata.env_steps == 2048
    assert metadata.optimizer_steps == 11
    assert metadata.player_step_total == 37
    assert metadata.total_games_played == 41
    assert metadata.target_kl_exceeded_total == 5
    assert metadata.wandb_run_id == "run-abc"
    assert dst_trainer.optimizer_steps == 11
    assert dst_trainer.player_step_total == 37
    assert dst_trainer.total_games_played == 41
    assert dst_trainer.target_kl_exceeded_total == 5
    for src_param, dst_param in zip(
        src_model.parameters(),
        dst_model.parameters(),
        strict=True,
    ):
        assert torch.equal(src_param, dst_param)
    assert dst_optimizer.state_dict()["state"]
    assert dst_scheduler.state_dict() == src_scheduler.state_dict()


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
