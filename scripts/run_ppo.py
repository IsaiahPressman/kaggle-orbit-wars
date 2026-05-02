from __future__ import annotations

import argparse
import copy
import itertools
import random
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, assert_never

import torch
import yaml
from owl.model import BaseModelAPI, ModelActions, ModelConfig, StatelessTransformerV1
from owl.replay import ReplayRecorder
from owl.rl import ActionConfig, ObsBatch, ObsConfig, VectorizedEnv
from owl.rs import assert_release_build
from owl.train import FullConfig, PPOTrainer, configure_torch
from owl.train.logging import LogMode, create_logger
from owl.train.optimizer import (
    create_lr_scheduler,
    create_optimizer,
)
from owl.train.ppo import _mean_env_metrics
from owl.train.utils import DTypeConfig, autocast_context
from tqdm import tqdm

MODEL_CURRENT = 0
MODEL_LAST_BEST = 1
PLAYER_COUNTS = (2, 4)
LAST_BEST_WIN_RATE_THRESHOLD = 0.7


def main() -> None:
    args = _parse_args()
    assert_release_build()
    configure_torch()
    overrides = _parse_cli_overrides(args.overrides)
    cfg = FullConfig.from_file(args.config, overrides=overrides)

    run_dir = _create_run_dir(args.output_dir)
    cfg.to_file(run_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedEnv(
        n_envs=cfg.env.n_envs,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
        two_player_weight=cfg.env.two_player_weight,
        pin_memory=cfg.env.pin_memory,
    )
    model = _create_model(
        cfg.model,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
    ).to(device)
    trainable_parameters = _trainable_parameter_count(model)
    optimizer = create_optimizer(model, cfg.optimizer)
    lr_scheduler = create_lr_scheduler(optimizer, cfg.optimizer.lr_schedule)
    trainer = PPOTrainer(
        config=cfg.rl,
        env=env,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
    )

    env_steps_per_iteration = cfg.rl.horizon * env.n_envs
    max_runtime_seconds = _max_runtime_seconds(args.max_runtime_hours)
    _run_training_session(
        trainer=trainer,
        run_dir=run_dir,
        cfg=cfg,
        log_mode=args.log_mode,
        env_steps_per_iteration=env_steps_per_iteration,
        max_env_steps=args.max_env_steps,
        max_runtime_seconds=max_runtime_seconds,
        trainable_parameters=trainable_parameters,
    )


def _run_training_session(
    *,
    trainer: PPOTrainer,
    run_dir: Path,
    cfg: FullConfig,
    log_mode: LogMode,
    env_steps_per_iteration: int,
    max_env_steps: int | None,
    max_runtime_seconds: float | None,
    trainable_parameters: int | None = None,
) -> None:
    with closing(create_logger(log_mode, run_dir, cfg)) as logger:
        if trainable_parameters is not None:
            logger.set_summary("trainable_parameters", trainable_parameters)
        env_steps = _run_training_loop(
            trainer=trainer,
            logger=logger,
            run_dir=run_dir,
            cfg=cfg,
            env_steps_per_iteration=env_steps_per_iteration,
            max_env_steps=max_env_steps,
            max_runtime_seconds=max_runtime_seconds,
        )
        trainer.write_checkpoint(
            run_dir / "checkpoint_final.pt",
            env_steps=env_steps,
        )


def _run_training_loop(
    *,
    trainer: PPOTrainer,
    logger: Any,
    run_dir: Path,
    cfg: FullConfig,
    env_steps_per_iteration: int,
    max_env_steps: int | None,
    max_runtime_seconds: float | None,
) -> int:
    env_steps = 0
    started_at = time.monotonic()
    next_checkpoint_env_steps = _next_periodic_checkpoint_step(
        checkpoint_freq=cfg.rl.checkpoint_freq,
    )
    last_best_model = (
        _clone_eval_model(trainer.model)
        if next_checkpoint_env_steps is not None
        else None
    )
    with tqdm(unit="env steps", dynamic_ncols=True) as progress:
        while True:
            metrics = trainer.train_iteration()
            env_steps += env_steps_per_iteration
            progress.update(env_steps_per_iteration)
            logger.log({**metrics, "train/env_steps": float(env_steps)}, step=env_steps)
            if (
                next_checkpoint_env_steps is not None
                and env_steps >= next_checkpoint_env_steps
            ):
                checkpoint_path = (
                    run_dir / f"checkpoint_{_format_checkpoint_step(env_steps)}.pt"
                )
                trainer.write_checkpoint(
                    checkpoint_path,
                    env_steps=env_steps,
                )
                if last_best_model is None:
                    raise RuntimeError("last_best_model must exist for checkpoints")
                eval_metrics = _evaluate_against_last_best(
                    current_model=trainer.model,
                    last_best_model=last_best_model,
                    cfg=cfg,
                    device=trainer.device,
                    replay_dir=(
                        run_dir / "eval_replays" / checkpoint_path.stem
                        if cfg.rl.eval_replay_games > 0
                        else None
                    ),
                )
                logger.log(eval_metrics, step=env_steps)
                if (
                    eval_metrics["eval/win_rate_against_last_best"]
                    >= LAST_BEST_WIN_RATE_THRESHOLD
                ):
                    _copy_model_state(last_best_model, trainer.model)
                    trainer.write_checkpoint(
                        run_dir / "checkpoint_last_best.pt",
                        env_steps=env_steps,
                    )
                next_checkpoint_env_steps = _next_periodic_checkpoint_step(
                    checkpoint_freq=cfg.rl.checkpoint_freq,
                    env_steps=env_steps,
                )
            if _should_stop_training(
                env_steps=env_steps,
                started_at=started_at,
                max_env_steps=max_env_steps,
                max_runtime_seconds=max_runtime_seconds,
            ):
                break
    return env_steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO training for Orbit Wars.")
    parser.add_argument("config", type=Path, help="Top-level config YAML file")
    parser.add_argument("output_dir", type=Path, help="Directory for run artifacts")
    parser.add_argument(
        "--log-mode",
        type=LogMode,
        choices=list(LogMode),
        default=LogMode.WANDB,
        help="Metric logging backend",
    )
    parser.add_argument(
        "-o",
        "--overrides",
        nargs="+",
        action="append",
        default=None,
        metavar="field.path=value",
        help="Optional overrides in the format field.path=value",
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=None,
        help="Stop after at least this many environment steps",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=None,
        help="Stop after at least this many wall-clock hours",
    )
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_env_steps is not None and args.max_env_steps <= 0:
        raise ValueError("--max-env-steps must be positive")
    if args.max_runtime_hours is not None and args.max_runtime_hours <= 0.0:
        raise ValueError("--max-runtime-hours must be positive")


def _parse_cli_overrides(raw_overrides: list[list[str]] | None) -> dict[str, Any]:
    if raw_overrides is None:
        return {}

    parsed_overrides: dict[str, Any] = {}
    for override in itertools.chain.from_iterable(raw_overrides):
        field_path, separator, raw_value = override.partition("=")
        if not separator:
            raise ValueError(
                f"Invalid override '{override}'. Expected format 'field.path=value'"
            )

        parts = field_path.split(".")
        if not field_path or any(not part for part in parts):
            raise ValueError(
                f"Invalid override field path '{field_path}'. "
                "Expected dot-separated field names"
            )

        if field_path in parsed_overrides:
            raise ValueError(f"Duplicate override field '{field_path}'")

        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML value for override '{field_path}': {raw_value}"
            ) from exc

        parsed_overrides[field_path] = value

    return parsed_overrides


def _create_run_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = output_dir / timestamp
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            time.sleep(1.0)

    return run_dir


def _create_model(
    config: ModelConfig,
    *,
    obs_spec: ObsConfig,
    action_spec: ActionConfig,
) -> StatelessTransformerV1:
    match config.model_arch:
        case "stateless_transformer_v1":
            return StatelessTransformerV1(
                config,
                obs_spec=obs_spec,
                action_spec=action_spec,
            )
        case _:
            assert_never(config)


def _trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _clone_eval_model(model: BaseModelAPI) -> BaseModelAPI:
    last_best_model = copy.deepcopy(model)
    last_best_model.eval()
    return last_best_model


def _copy_model_state(dst: BaseModelAPI, src: BaseModelAPI) -> None:
    dst.load_state_dict(src.state_dict())
    dst.eval()


def _next_periodic_checkpoint_step(
    *, checkpoint_freq: int | None, env_steps: int = 0
) -> int | None:
    if checkpoint_freq is None:
        return None
    return (env_steps // checkpoint_freq + 1) * checkpoint_freq


def _format_checkpoint_step(env_steps: int) -> str:
    if env_steps < 0:
        raise ValueError("env_steps must be non-negative")
    digits = f"{env_steps:011d}"
    return f"{digits[:2]}_{digits[2:5]}_{digits[5:8]}_{digits[8:11]}"


def _evaluate_against_last_best(
    *,
    current_model: BaseModelAPI,
    last_best_model: BaseModelAPI,
    cfg: FullConfig,
    device: torch.device,
    replay_dir: Path | None = None,
) -> dict[str, float]:
    started_at = time.perf_counter()
    eval_steps = 0
    current_was_training = current_model.training
    last_best_was_training = last_best_model.training
    current_model.eval()
    last_best_model.eval()
    stats = _EvalStats.empty()
    env_metrics: dict[str, list[float]] = {}
    try:
        with torch.no_grad():
            per_player_count_games = cfg.env.n_envs // len(PLAYER_COUNTS)
            eval_n_envs = cfg.env.n_envs // 2
            if cfg.rl.eval_replay_games > cfg.env.n_envs:
                raise ValueError("rl.eval_replay_games must be <= env.n_envs")
            for player_count in PLAYER_COUNTS:
                player_stats, player_env_metrics, player_steps = _evaluate_player_count(
                    current_model=current_model,
                    last_best_model=last_best_model,
                    cfg=cfg,
                    player_count=player_count,
                    n_games=per_player_count_games,
                    n_envs=eval_n_envs,
                    device=device,
                    replay_games=cfg.rl.eval_replay_games // len(PLAYER_COUNTS),
                    replay_output_path=(
                        replay_dir / f"eval_{player_count}p.jsonl"
                        if replay_dir is not None
                        else None
                    ),
                )
                stats.merge(player_stats)
                _extend_env_metrics(env_metrics, player_env_metrics)
                eval_steps += player_steps
    finally:
        current_model.train(current_was_training)
        last_best_model.train(last_best_was_training)

    elapsed = max(time.perf_counter() - started_at, 1e-12)
    metrics = _eval_env_metrics(env_metrics)
    metrics["eval/win_rate_against_last_best"] = stats.win_rate(MODEL_CURRENT)
    metrics["time/eval_seconds"] = float(elapsed)
    metrics["perf/eval_sps"] = float(eval_steps / elapsed)
    return metrics


def _evaluate_player_count(
    *,
    current_model: BaseModelAPI,
    last_best_model: BaseModelAPI,
    cfg: FullConfig,
    player_count: int,
    n_games: int,
    n_envs: int,
    device: torch.device,
    replay_games: int = 0,
    replay_output_path: Path | None = None,
) -> tuple[_EvalStats, dict[str, list[float]], int]:
    env = VectorizedEnv(
        n_envs=n_envs,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
        two_player_weight=1.0 if player_count == 2 else 0.0,
        pin_memory=device.type == "cuda",
    )
    obs = env.reset()
    assignments = torch.full((n_envs, 4), -1, dtype=torch.int64)
    start_masks = obs.still_playing.clone()
    returns = torch.zeros((n_envs, 4), dtype=torch.float32)
    recorder = (
        ReplayRecorder(
            output_path=replay_output_path,
            source="run_ppo_eval",
            player_count=player_count,
            total_games=n_games,
            sample_games=replay_games,
            metadata={
                "eval_opponent": "last_best",
            },
            rng=random.Random(),
        )
        if replay_output_path is not None and replay_games > 0
        else None
    )
    current_game_ordinals: list[int | None] = [None] * n_envs
    started_games = 0
    for env_index in range(n_envs):
        _assign_eval_models(
            assignments,
            env_index,
            active_slots=start_masks[env_index],
            player_count=player_count,
        )
        if started_games < n_games:
            current_game_ordinals[env_index] = started_games
            if recorder is not None:
                recorder.start_episode(
                    env,
                    env_index,
                    game_ordinal=started_games,
                    assignments=assignments[env_index],
                    start_mask=start_masks[env_index],
                )
            started_games += 1

    games = 0
    steps = 0
    stats = _EvalStats.empty()
    env_metrics: dict[str, list[float]] = {}
    while games < n_games:
        actions = _eval_actions_for_assignments(
            obs,
            assignments,
            current_model=current_model,
            last_best_model=last_best_model,
            config=cfg.rl,
            device=device,
        )
        obs, rewards, dones, _step_env_metrics = env.step(
            actions.launch,
            actions.action_value(),
            actions.ships,
        )
        steps += n_envs
        terminal_envs = torch.nonzero(dones.all(dim=1), as_tuple=False).flatten()
        terminal_env_set = {int(env_index.item()) for env_index in terminal_envs}
        if recorder is not None:
            recorder.record_step(
                env,
                terminal_envs=terminal_env_set,
                rewards=rewards,
                dones=dones,
            )
        returns += rewards
        for env_index_tensor in terminal_envs:
            if games >= n_games:
                break
            env_index = int(env_index_tensor.item())
            if current_game_ordinals[env_index] is not None:
                terminal_metrics = env.terminal_metrics(env_index)
                if terminal_metrics is None:
                    raise RuntimeError(f"missing terminal metrics for env {env_index}")
                _extend_single_env_metrics(env_metrics, terminal_metrics)
                _record_eval_terminal_result(
                    stats,
                    assignments[env_index],
                    start_masks[env_index],
                    returns[env_index],
                )
                games += 1

            returns[env_index].zero_()
            start_masks[env_index].copy_(obs.still_playing[env_index])
            _assign_eval_models(
                assignments,
                env_index,
                active_slots=start_masks[env_index],
                player_count=player_count,
            )
            if started_games < n_games:
                current_game_ordinals[env_index] = started_games
                if recorder is not None:
                    recorder.start_episode(
                        env,
                        env_index,
                        game_ordinal=started_games,
                        assignments=assignments[env_index],
                        start_mask=start_masks[env_index],
                    )
                started_games += 1
            else:
                current_game_ordinals[env_index] = None
    return stats, env_metrics, steps


def _extend_single_env_metrics(
    totals: dict[str, list[float]],
    step_metrics: dict[str, float],
) -> None:
    for key, value in step_metrics.items():
        totals.setdefault(key, []).append(value)


class _EvalStats:
    def __init__(self, *, model_games: list[int], wins: list[float]) -> None:
        self.model_games = model_games
        self.wins = wins

    @classmethod
    def empty(cls) -> _EvalStats:
        return cls(model_games=[0, 0], wins=[0.0, 0.0])

    def add_game_result(self, winner: int | None) -> None:
        for model_index in range(2):
            self.model_games[model_index] += 1
        if winner is None:
            for model_index in range(2):
                self.wins[model_index] += 0.5
        else:
            self.wins[winner] += 1.0

    def merge(self, other: _EvalStats) -> None:
        for model_index in range(2):
            self.model_games[model_index] += other.model_games[model_index]
            self.wins[model_index] += other.wins[model_index]

    def win_rate(self, model_index: int) -> float:
        games = self.model_games[model_index]
        if games == 0:
            return 0.0
        return self.wins[model_index] / games


def _assign_eval_models(
    assignments: torch.Tensor,
    env_index: int,
    *,
    active_slots: torch.Tensor,
    player_count: int,
) -> None:
    active = torch.nonzero(active_slots, as_tuple=False).flatten().tolist()
    if len(active) != player_count:
        raise ValueError(
            f"expected {player_count} active players, got {len(active)} "
            f"for env {env_index}"
        )

    pattern = _eval_assignment_pattern(player_count)
    assignments[env_index].fill_(-1)
    for slot, model_index in zip(active, pattern, strict=True):
        assignments[env_index, slot] = model_index


def _eval_assignment_pattern(player_count: int) -> tuple[int, ...]:
    if player_count == 2:
        return (MODEL_CURRENT, MODEL_LAST_BEST)
    if player_count == 4:
        return (MODEL_CURRENT, MODEL_LAST_BEST, MODEL_LAST_BEST, MODEL_CURRENT)
    raise ValueError(f"player_count must be 2 or 4, got {player_count}")


def _record_eval_terminal_result(
    stats: _EvalStats,
    assignment: torch.Tensor,
    start_mask: torch.Tensor,
    returns: torch.Tensor,
) -> None:
    active_returns = returns[start_mask]
    if active_returns.numel() == 0:
        raise ValueError("cannot record a terminal result without starting players")
    model_returns = [0.0, 0.0]
    for player in torch.nonzero(start_mask, as_tuple=False).flatten().tolist():
        model_index = int(assignment[player].item())
        if model_index not in (MODEL_CURRENT, MODEL_LAST_BEST):
            raise ValueError(f"missing model assignment for player slot {player}")
        model_returns[model_index] += float(returns[player].item())
    if model_returns[MODEL_CURRENT] > model_returns[MODEL_LAST_BEST]:
        stats.add_game_result(MODEL_CURRENT)
    elif model_returns[MODEL_LAST_BEST] > model_returns[MODEL_CURRENT]:
        stats.add_game_result(MODEL_LAST_BEST)
    else:
        stats.add_game_result(None)


def _eval_actions_for_assignments(
    obs: ObsBatch,
    assignments: torch.Tensor,
    *,
    current_model: BaseModelAPI,
    last_best_model: BaseModelAPI,
    config: DTypeConfig,
    device: torch.device,
) -> ModelActions:
    device_obs = _obs_to_device(obs, device)
    with autocast_context(config, device):
        current_output = current_model(device_obs)
        last_best_output = last_best_model(device_obs)
    use_current = assignments.to(device=device).eq(MODEL_CURRENT)
    return _select_actions(
        current_output.actions, last_best_output.actions, use_current
    )


def _obs_to_device(obs: ObsBatch, device: torch.device) -> ObsBatch:
    return ObsBatch(
        **{
            field: getattr(obs, field).to(
                device=device,
                non_blocking=device.type == "cuda",
            )
            for field in ObsBatch.model_fields
        }
    )


def _select_actions(
    actions_a: ModelActions,
    actions_b: ModelActions,
    use_a: torch.Tensor,
) -> ModelActions:
    action_mask = use_a[:, :, None, None]
    return ModelActions(
        launch=torch.where(action_mask, actions_a.launch, actions_b.launch).cpu(),
        ships=torch.where(action_mask, actions_a.ships, actions_b.ships).cpu(),
        angle=_select_optional_action(actions_a.angle, actions_b.angle, action_mask),
        target=_select_optional_action(actions_a.target, actions_b.target, action_mask),
    )


def _select_optional_action(
    tensor_a: torch.Tensor | None,
    tensor_b: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if tensor_a is None and tensor_b is None:
        return None
    if tensor_a is None or tensor_b is None:
        raise ValueError("checkpoint action value tensors must have matching kinds")
    return torch.where(mask, tensor_a, tensor_b).cpu()


def _extend_env_metrics(
    totals: dict[str, list[float]],
    step_metrics: dict[str, list[float]],
) -> None:
    for key, values in step_metrics.items():
        totals.setdefault(key, []).extend(values)


def _eval_env_metrics(metrics: dict[str, list[float]]) -> dict[str, float]:
    return {
        f"eval/{key.removeprefix('train/')}": value
        for key, value in _mean_env_metrics(metrics).items()
    }


def _max_runtime_seconds(max_runtime_hours: float | None) -> float | None:
    if max_runtime_hours is None:
        return None
    return max_runtime_hours * 60.0 * 60.0


def _should_stop_training(
    *,
    env_steps: int,
    started_at: float,
    max_env_steps: int | None,
    max_runtime_seconds: float | None,
) -> bool:
    if max_env_steps is not None and env_steps >= max_env_steps:
        return True
    return (
        max_runtime_seconds is not None
        and time.monotonic() - started_at >= max_runtime_seconds
    )


if __name__ == "__main__":
    main()
