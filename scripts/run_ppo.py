from __future__ import annotations

import argparse
import copy
import itertools
import random
import re
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from owl.model import (
    BaseModelAPI,
    ModelConfig,
    ModelHiddenState,
    ModelOutput,
    create_model,
)
from owl.replay import ReplayRecorder
from owl.rl import (
    ActionBundle,
    ActionConfig,
    ActionMask,
    DiscreteTargetActionMask,
    DiscreteTargetActions,
    DiscreteTargetBinActionMask,
    DiscreteTargetBinActions,
    ObsBatch,
    ObsConfig,
    PureActionMask,
    PureActions,
    VectorizedEnv,
)
from owl.rs import assert_release_build
from owl.train import FullConfig, PPOTrainer, configure_torch
from owl.train.distributed import (
    DistributedContext,
    all_reduce_any,
    broadcast_object,
    distributed_session,
    unwrap_model,
    wrap_model_for_distributed,
)
from owl.train.logging import LogMode, create_logger
from owl.train.optimizer import (
    create_lr_scheduler,
    create_optimizer,
)
from owl.train.ppo import PPOCheckpointMetadata, _mean_env_metrics
from owl.train.utils import (
    DTypeConfig,
    autocast_context,
    configure_model_compile,
)
from tqdm import tqdm

MODEL_CURRENT = 0
MODEL_LAST_BEST = 1
PLAYER_COUNTS = (2, 4)
LAST_BEST_WIN_RATE_THRESHOLD = 0.7
CHECKPOINT_FINAL = "checkpoint_final.pt"
CHECKPOINT_LAST_BEST = "checkpoint_last_best.pt"
_NUMBERED_CHECKPOINT_RE = re.compile(
    r"^checkpoint_(\d{2})_(\d{3})_(\d{3})_(\d{3})\.pt$"
)


@dataclass(frozen=True)
class FreshLaunch:
    config_path: Path
    output_dir: Path
    overrides: dict[str, Any]
    load_model_weights_path: Path | None = None


@dataclass(frozen=True)
class ResumeLaunch:
    config_path: Path
    run_dir: Path
    checkpoint_path: Path
    last_best_checkpoint_path: Path


Launch = FreshLaunch | ResumeLaunch


class _NoopLogger:
    @property
    def run_id(self) -> None:
        return None

    def log(self, metrics: dict[str, float], *, step: int) -> None:  # noqa: ARG002
        return None

    def set_summary(
        self,
        key: str,  # noqa: ARG002
        value: int | float,  # noqa: ARG002
    ) -> None:
        return None

    def close(self) -> None:
        return None


def main() -> None:
    args = _parse_args()
    assert_release_build()
    configure_torch()
    launch = _resolve_launch(args)
    with distributed_session() as distributed:
        _log_cli_overrides(args.overrides, distributed)
        cfg = FullConfig.from_file(
            launch.config_path,
            overrides=launch.overrides if isinstance(launch, FreshLaunch) else None,
        )
        cfg = _resolve_teacher_init_path(cfg, launch.config_path)

        if isinstance(launch, FreshLaunch):
            cfg = _with_runtime_gpus(cfg, distributed.world_size)
            run_dir = (
                _create_run_dir(launch.output_dir)
                if distributed.is_main_process
                else None
            )
            if distributed.is_main_process:
                if run_dir is None:
                    raise RuntimeError("main process failed to create run dir")
                cfg.to_file(run_dir / "config.yaml")
            run_dir = broadcast_object(run_dir, distributed)
        else:
            cfg = _adapt_resume_config_for_runtime_gpus(cfg, distributed)
            run_dir = launch.run_dir

        device = distributed.device
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
        if isinstance(launch, FreshLaunch):
            model.reset_parameters()
        compiled_model_modules = configure_model_compile(model, cfg.rl)
        teacher_init_model = (
            None
            if cfg.rl.teacher_init is None
            else _load_teacher_init_model(
                cfg.rl.teacher_init,
                student_cfg=cfg,
                device=device,
            )
        )
        fixed_teacher_model = (
            teacher_init_model if cfg.rl.teacher_mode == "fixed" else None
        )
        last_best_model = _initial_last_best_model(cfg, teacher_init_model)

        trainable_parameters = _trainable_parameter_count(model)
        optimizer = create_optimizer(model, cfg.optimizer)
        lr_scheduler = create_lr_scheduler(optimizer, cfg.optimizer.lr_schedule)
        model = wrap_model_for_distributed(model, distributed)
        trainer = PPOTrainer(
            config=cfg.rl,
            env=env,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            teacher_model=fixed_teacher_model,
            teacher_active=fixed_teacher_model is not None,
            distributed_context=distributed,
        )
        start_env_steps = 0
        resume_run_id: str | None = None
        if isinstance(launch, ResumeLaunch):
            checkpoint_metadata = trainer.load_checkpoint(launch.checkpoint_path)
            resume_run_id = _resume_wandb_run_id(checkpoint_metadata, args.log_mode)
            start_env_steps = checkpoint_metadata.env_steps
            last_best_model = _clone_eval_model(unwrap_model(model))
            last_best_metadata = _load_model_from_checkpoint(
                last_best_model,
                path=launch.last_best_checkpoint_path,
                device=device,
            )
            _validate_last_best_run_id(
                last_best_metadata,
                resume_run_id=resume_run_id,
                checkpoint_path=launch.last_best_checkpoint_path,
            )
            if cfg.rl.teacher_mode == "last_best":
                trainer.set_teacher_model(last_best_model, active=True)
        elif launch.load_model_weights_path is not None:
            checkpoint_metadata = trainer.load_model_weights(
                launch.load_model_weights_path
            )
            start_env_steps = checkpoint_metadata.env_steps
            if cfg.rl.teacher_mode == "last_best":
                last_best_model = _clone_eval_model(unwrap_model(model))
        if (
            isinstance(launch, FreshLaunch)
            and cfg.rl.teacher_mode == "last_best"
            and last_best_model is not None
        ):
            trainer.set_teacher_model(last_best_model, active=True)

        env_steps_per_iteration = cfg.rl.horizon * env.n_envs * distributed.world_size
        max_runtime_seconds = _max_runtime_seconds(args.max_runtime_hours)
        _run_training_session(
            trainer=trainer,
            run_dir=run_dir,
            cfg=cfg,
            log_mode=args.log_mode,
            env_steps_per_iteration=env_steps_per_iteration,
            max_env_steps=args.max_env_steps,
            max_runtime_seconds=max_runtime_seconds,
            distributed=distributed,
            start_env_steps=start_env_steps,
            resume_run_id=resume_run_id,
            last_best_model=last_best_model,
            trainable_parameters=trainable_parameters,
            compiled_model_modules=compiled_model_modules,
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
    distributed: DistributedContext,
    start_env_steps: int = 0,
    resume_run_id: str | None = None,
    last_best_model: BaseModelAPI | None = None,
    trainable_parameters: int | None = None,
    compiled_model_modules: int = 0,
) -> None:
    if not distributed.is_main_process:
        _run_training_session_worker(
            trainer=trainer,
            run_dir=run_dir,
            cfg=cfg,
            env_steps_per_iteration=env_steps_per_iteration,
            max_env_steps=max_env_steps,
            max_runtime_seconds=max_runtime_seconds,
            distributed=distributed,
            start_env_steps=start_env_steps,
            last_best_model=last_best_model,
        )
        return

    with closing(
        create_logger(log_mode, run_dir, cfg, resume_run_id=resume_run_id)
    ) as logger:
        if trainable_parameters is not None:
            logger.set_summary("trainable_parameters", trainable_parameters)
        if compiled_model_modules > 0:
            logger.set_summary("compiled_model_modules", compiled_model_modules)
        env_steps = _run_training_loop(
            trainer=trainer,
            logger=logger,
            run_dir=run_dir,
            cfg=cfg,
            env_steps_per_iteration=env_steps_per_iteration,
            max_env_steps=max_env_steps,
            max_runtime_seconds=max_runtime_seconds,
            dist_ctx=distributed,
            start_env_steps=start_env_steps,
            wandb_run_id=logger.run_id,
            last_best_model=last_best_model,
        )
        trainer.write_checkpoint(
            run_dir / "checkpoint_final.pt",
            env_steps=env_steps,
            wandb_run_id=logger.run_id,
        )


def _run_training_session_worker(
    *,
    trainer: PPOTrainer,
    run_dir: Path,
    cfg: FullConfig,
    env_steps_per_iteration: int,
    max_env_steps: int | None,
    max_runtime_seconds: float | None,
    distributed: DistributedContext,
    start_env_steps: int = 0,
    last_best_model: BaseModelAPI | None = None,
) -> None:
    _run_training_loop(
        trainer=trainer,
        logger=_NoopLogger(),
        run_dir=run_dir,
        cfg=cfg,
        env_steps_per_iteration=env_steps_per_iteration,
        max_env_steps=max_env_steps,
        max_runtime_seconds=max_runtime_seconds,
        dist_ctx=distributed,
        start_env_steps=start_env_steps,
        last_best_model=last_best_model,
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
    dist_ctx: DistributedContext,
    start_env_steps: int = 0,
    wandb_run_id: str | None = None,
    last_best_model: BaseModelAPI | None = None,
) -> int:
    env_steps = start_env_steps
    started_at = time.monotonic()
    next_checkpoint_env_steps = _next_periodic_checkpoint_step(
        checkpoint_freq=cfg.rl.checkpoint_freq,
        env_steps=env_steps,
    )
    if next_checkpoint_env_steps is not None and last_best_model is None:
        last_best_model = _clone_eval_model(unwrap_model(trainer.model))
    if _should_stop_training(
        env_steps=env_steps,
        started_at=started_at,
        max_env_steps=max_env_steps,
        max_runtime_seconds=max_runtime_seconds,
        distributed=dist_ctx,
    ):
        return env_steps
    with tqdm(
        unit="env steps",
        dynamic_ncols=True,
        initial=env_steps,
        disable=not dist_ctx.is_main_process,
    ) as progress:
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
                if dist_ctx.is_main_process:
                    trainer.write_checkpoint(
                        checkpoint_path,
                        env_steps=env_steps,
                        wandb_run_id=wandb_run_id,
                    )
                if last_best_model is None:
                    raise RuntimeError("last_best_model must exist for checkpoints")
                dist_ctx.barrier()
                eval_metrics: dict[str, float] | None = None
                if dist_ctx.is_main_process:
                    eval_metrics = _evaluate_against_last_best(
                        current_model=unwrap_model(trainer.model),
                        last_best_model=last_best_model,
                        cfg=cfg,
                        device=trainer.device,
                        replay_dir=(
                            run_dir / "eval_replays" / checkpoint_path.stem
                            if cfg.rl.eval_replay_games > 0
                            else None
                        ),
                    )
                eval_metrics = broadcast_object(eval_metrics, dist_ctx)
                if eval_metrics is None:
                    raise RuntimeError("missing broadcast eval metrics")
                logger.log(eval_metrics, step=env_steps)
                replace_last_best = (
                    eval_metrics["eval/win_rate_against_last_best"]
                    >= LAST_BEST_WIN_RATE_THRESHOLD
                )
                if replace_last_best:
                    last_best_model = _clone_eval_model(unwrap_model(trainer.model))
                    if cfg.rl.teacher_mode == "last_best":
                        trainer.set_teacher_model(
                            last_best_model,
                            active=True,
                        )
                    if dist_ctx.is_main_process:
                        trainer.write_checkpoint(
                            run_dir / "checkpoint_last_best.pt",
                            env_steps=env_steps,
                            wandb_run_id=wandb_run_id,
                        )
                dist_ctx.barrier()
                next_checkpoint_env_steps = _next_periodic_checkpoint_step(
                    checkpoint_freq=cfg.rl.checkpoint_freq,
                    env_steps=env_steps,
                )
            if _should_stop_training(
                env_steps=env_steps,
                started_at=started_at,
                max_env_steps=max_env_steps,
                max_runtime_seconds=max_runtime_seconds,
                distributed=dist_ctx,
            ):
                break
    return env_steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO training for Orbit Wars.")
    parser.add_argument(
        "target",
        type=Path,
        help=(
            "Config YAML for a fresh run, or a run directory/checkpoint file to resume"
        ),
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="Directory for fresh run artifacts",
    )
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
        "--load-model-weights",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "For fresh launches, initialize model weights from a checkpoint while "
            "keeping only env_steps, player_step_total, and total_games_played"
        ),
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
    if args.output_dir is None and args.overrides is not None:
        raise ValueError("resume launches cannot use config overrides")
    if args.output_dir is None and args.load_model_weights is not None:
        raise ValueError("resume launches cannot use --load-model-weights")
    if args.output_dir is None and args.log_mode == LogMode.DEBUG:
        raise ValueError("resume launches require wandb logging")


def _resolve_launch(args: argparse.Namespace) -> Launch:
    if args.output_dir is not None:
        if not args.target.is_file():
            raise ValueError(f"fresh run config does not exist: {args.target}")
        if (
            args.load_model_weights is not None
            and not args.load_model_weights.is_file()
        ):
            raise ValueError(
                f"model-weights checkpoint does not exist: {args.load_model_weights}"
            )
        return FreshLaunch(
            config_path=args.target,
            output_dir=args.output_dir,
            overrides=_parse_cli_overrides(args.overrides),
            load_model_weights_path=args.load_model_weights,
        )
    return _resolve_resume_launch(args.target)


def _resolve_resume_launch(target: Path) -> ResumeLaunch:
    if target.is_dir():
        run_dir = target
        checkpoint_path = _latest_resume_checkpoint(run_dir)
    elif target.is_file():
        if target.name == CHECKPOINT_LAST_BEST:
            raise ValueError(
                "checkpoint_last_best.pt cannot be used as a resume target"
            )
        run_dir = target.parent
        checkpoint_path = target
    else:
        raise ValueError(f"resume target does not exist: {target}")

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"expected resume config at {config_path}")
    last_best_checkpoint_path = run_dir / CHECKPOINT_LAST_BEST
    if not last_best_checkpoint_path.is_file():
        raise ValueError(
            f"expected last-best checkpoint at {last_best_checkpoint_path}"
        )
    return ResumeLaunch(
        config_path=config_path,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        last_best_checkpoint_path=last_best_checkpoint_path,
    )


def _latest_resume_checkpoint(run_dir: Path) -> Path:
    final_checkpoint = run_dir / CHECKPOINT_FINAL
    if final_checkpoint.is_file():
        return final_checkpoint

    numbered_checkpoints: list[tuple[int, Path]] = []
    for checkpoint_path in run_dir.glob("checkpoint_*.pt"):
        step = _parse_numbered_checkpoint_step(checkpoint_path.name)
        if step is not None:
            numbered_checkpoints.append((step, checkpoint_path))
    if not numbered_checkpoints:
        raise ValueError(f"no resume checkpoint found in {run_dir}")
    return max(numbered_checkpoints, key=lambda item: item[0])[1]


def _parse_numbered_checkpoint_step(name: str) -> int | None:
    match = _NUMBERED_CHECKPOINT_RE.fullmatch(name)
    if match is None:
        return None
    return int("".join(match.groups()))


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


def _log_cli_overrides(
    raw_overrides: list[list[str]] | None,
    distributed: DistributedContext,
) -> None:
    if not raw_overrides or not distributed.is_main_process:
        return

    overrides_flat = list(itertools.chain.from_iterable(raw_overrides))
    print(f"Launched with the following raw manual overrides: '{overrides_flat}'")


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
) -> BaseModelAPI:
    return create_model(config, obs_spec=obs_spec, action_spec=action_spec)


def _resolve_teacher_init_path(cfg: FullConfig, config_path: Path) -> FullConfig:
    teacher_init = cfg.rl.teacher_init
    if teacher_init is None or teacher_init.is_absolute():
        return cfg
    return cfg.model_copy(
        update={
            "rl": cfg.rl.model_copy(
                update={"teacher_init": (config_path.parent / teacher_init).resolve()},
            ),
        },
    )


def _load_teacher_init_model(
    checkpoint_path: Path,
    *,
    student_cfg: FullConfig,
    device: torch.device,
) -> BaseModelAPI:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f"teacher_init checkpoint does not exist: {checkpoint_path}")
    teacher_cfg = FullConfig.from_file(_checkpoint_config_path(checkpoint_path))
    _validate_teacher_specs(
        teacher_cfg,
        student_cfg=student_cfg,
        checkpoint_path=checkpoint_path,
    )
    teacher_model = _create_model(
        teacher_cfg.model,
        obs_spec=teacher_cfg.env.obs_spec,
        action_spec=teacher_cfg.env.action_spec,
    ).to(device)
    _load_model_weights(teacher_model, path=checkpoint_path, device=device)
    teacher_model.eval()
    return teacher_model


def _initial_last_best_model(
    cfg: FullConfig,
    teacher_init_model: BaseModelAPI | None,
) -> BaseModelAPI | None:
    if teacher_init_model is None:
        return None
    if cfg.rl.checkpoint_freq is None and cfg.rl.teacher_mode != "last_best":
        return None
    return _clone_eval_model(teacher_init_model)


def _checkpoint_config_path(checkpoint_path: Path) -> Path:
    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"expected checkpoint config at {config_path}")
    return config_path


def _validate_teacher_specs(
    teacher_cfg: FullConfig,
    *,
    student_cfg: FullConfig,
    checkpoint_path: Path,
) -> None:
    if teacher_cfg.env.obs_spec != student_cfg.env.obs_spec:
        raise ValueError(f"teacher obs_spec must match student: {checkpoint_path}")
    if teacher_cfg.env.action_spec != student_cfg.env.action_spec:
        raise ValueError(f"teacher action_spec must match student: {checkpoint_path}")


def _load_model_weights(
    model: BaseModelAPI,
    *,
    path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must be a dictionary: {path}")
    if "model" not in checkpoint:
        raise ValueError(f"checkpoint is missing model weights: {path}")
    model.load_state_dict(checkpoint["model"])
    model.eval()


def _with_runtime_gpus(cfg: FullConfig, world_size: int) -> FullConfig:
    return cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(
                update={"n_runtime_gpus": world_size},
            ),
        },
    )


def _adapt_resume_config_for_runtime_gpus(
    cfg: FullConfig,
    distributed: DistributedContext,
) -> FullConfig:
    saved_gpus = cfg.runtime.n_runtime_gpus
    current_gpus = distributed.world_size
    if saved_gpus == current_gpus:
        return cfg

    n_envs = _scale_resume_value_for_runtime_gpus(
        cfg.env.n_envs,
        saved_gpus=saved_gpus,
        current_gpus=current_gpus,
        name="env.n_envs",
    )
    train_segments = _scale_resume_value_for_runtime_gpus(
        cfg.rl.segments_per_minibatch * cfg.rl.gradient_accumulation_steps,
        saved_gpus=saved_gpus,
        current_gpus=current_gpus,
        name=("rl.segments_per_minibatch * rl.gradient_accumulation_steps"),
    )
    segments_per_minibatch, gradient_accumulation_steps = (
        _derive_resume_minibatch_shape(
            train_segments=train_segments,
            segments_per_minibatch=cfg.rl.segments_per_minibatch,
        )
    )

    updated = cfg.model_copy(
        update={
            "env": cfg.env.model_copy(update={"n_envs": n_envs}),
            "rl": cfg.rl.model_copy(
                update={
                    "segments_per_minibatch": segments_per_minibatch,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                },
            ),
            "runtime": cfg.runtime.model_copy(
                update={"n_runtime_gpus": current_gpus},
            ),
        },
    )
    adapted = FullConfig.model_validate(updated.model_dump(mode="python"))
    if adapted.rl.eval_replay_games > adapted.env.n_envs:
        raise ValueError(
            "cannot derive resume config for runtime GPU count change: "
            "rl.eval_replay_games must be <= env.n_envs"
        )
    return adapted


def _scale_resume_value_for_runtime_gpus(
    value: int,
    *,
    saved_gpus: int,
    current_gpus: int,
    name: str,
) -> int:
    numerator = value * saved_gpus
    if numerator % current_gpus != 0:
        raise ValueError(
            "cannot derive resume config for runtime GPU count change: "
            f"{name}={value} from runtime.n_runtime_gpus={saved_gpus} "
            f"does not scale evenly to {current_gpus} GPUs"
        )
    return numerator // current_gpus


def _derive_resume_minibatch_shape(
    *,
    train_segments: int,
    segments_per_minibatch: int,
) -> tuple[int, int]:
    if train_segments % segments_per_minibatch == 0:
        return segments_per_minibatch, train_segments // segments_per_minibatch

    for candidate_segments in range(
        min(train_segments, segments_per_minibatch),
        0,
        -1,
    ):
        if train_segments % candidate_segments == 0:
            return candidate_segments, train_segments // candidate_segments

    raise RuntimeError("positive integers should always have a divisor")


def _trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _clone_eval_model(model: BaseModelAPI) -> BaseModelAPI:
    last_best_model = copy.deepcopy(model)
    last_best_model.eval()
    return last_best_model


def _resume_wandb_run_id(
    metadata: PPOCheckpointMetadata,
    log_mode: LogMode,
) -> str:
    if log_mode == LogMode.DEBUG:
        raise ValueError("resume launches require wandb logging")
    if metadata.wandb_run_id is None:
        raise ValueError("resume checkpoint is missing wandb_run_id")
    return metadata.wandb_run_id


def _load_model_from_checkpoint(
    model: BaseModelAPI,
    *,
    path: Path,
    device: torch.device,
) -> PPOCheckpointMetadata:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must be a dictionary: {path}")
    metadata = _checkpoint_metadata(checkpoint, path=path)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return metadata


def _checkpoint_metadata(
    checkpoint: object,
    *,
    path: Path,
) -> PPOCheckpointMetadata:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must be a dictionary: {path}")
    expected_keys = {
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
    if set(checkpoint) != expected_keys:
        raise ValueError(
            f"checkpoint keys must be {sorted(expected_keys)}, "
            f"got {sorted(checkpoint)}: {path}"
        )
    return PPOCheckpointMetadata(
        env_steps=_checkpoint_nonnegative_int(
            checkpoint["env_steps"],
            name="env_steps",
            path=path,
        ),
        player_step_total=_checkpoint_nonnegative_int(
            checkpoint["player_step_total"],
            name="player_step_total",
            path=path,
        ),
        total_games_played=_checkpoint_nonnegative_int(
            checkpoint["total_games_played"],
            name="total_games_played",
            path=path,
        ),
        wandb_run_id=_checkpoint_optional_str(
            checkpoint["wandb_run_id"],
            name="wandb_run_id",
            path=path,
        ),
    )


def _checkpoint_nonnegative_int(value: object, *, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint {name} must be an integer: {path}")
    if value < 0:
        raise ValueError(f"checkpoint {name} must be non-negative: {path}")
    return value


def _checkpoint_optional_str(value: object, *, name: str, path: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"checkpoint {name} must be a non-empty string or None: {path}"
        )
    return value


def _validate_last_best_run_id(
    metadata: PPOCheckpointMetadata,
    *,
    resume_run_id: str,
    checkpoint_path: Path,
) -> None:
    if metadata.wandb_run_id is None:
        raise ValueError(
            f"last-best checkpoint is missing wandb_run_id: {checkpoint_path}"
        )
    if metadata.wandb_run_id != resume_run_id:
        raise ValueError(
            f"last-best checkpoint wandb_run_id does not match resume checkpoint: "
            f"{checkpoint_path}"
        )


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
    stats_by_player_count: dict[int, _EvalStats] = {}
    env_metrics: dict[str, list[float]] = {}
    try:
        with torch.no_grad():
            if cfg.rl.eval_replay_games > cfg.env.n_envs:
                raise ValueError("rl.eval_replay_games must be <= env.n_envs")
            player_stats, player_stats_by_count, player_env_metrics, player_steps = (
                _evaluate_games(
                    current_model=current_model,
                    last_best_model=last_best_model,
                    cfg=cfg,
                    n_games=cfg.env.n_envs,
                    n_envs=cfg.env.n_envs,
                    device=device,
                    replay_games=cfg.rl.eval_replay_games,
                    replay_output_path=(
                        replay_dir / "eval.jsonl" if replay_dir is not None else None
                    ),
                )
            )
            stats_by_player_count.update(player_stats_by_count)
            stats.merge(player_stats)
            _extend_env_metrics(env_metrics, player_env_metrics)
            eval_steps += player_steps
    finally:
        current_model.train(current_was_training)
        last_best_model.train(last_best_was_training)

    elapsed = max(time.perf_counter() - started_at, 1e-12)
    metrics = _eval_env_metrics(env_metrics)
    metrics["eval/win_rate_against_last_best"] = stats.win_rate(MODEL_CURRENT)
    for player_count, player_stats in stats_by_player_count.items():
        if player_stats.model_games[MODEL_CURRENT] == 0:
            continue
        metrics[f"eval/win_rate_against_last_best_{player_count}p"] = (
            player_stats.win_rate(MODEL_CURRENT)
        )
    metrics["time/eval_seconds"] = float(elapsed)
    metrics["perf/eval_sps"] = float(eval_steps / elapsed)
    return metrics


def _evaluate_games(
    *,
    current_model: BaseModelAPI,
    last_best_model: BaseModelAPI,
    cfg: FullConfig,
    n_games: int,
    n_envs: int,
    device: torch.device,
    replay_games: int = 0,
    replay_output_path: Path | None = None,
) -> tuple[_EvalStats, dict[int, _EvalStats], dict[str, list[float]], int]:
    env = VectorizedEnv(
        n_envs=n_envs,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
        two_player_weight=cfg.env.two_player_weight,
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
            player_count=None,
            total_games=n_games,
            sample_games=replay_games,
            metadata={
                "eval_opponent": "last_best",
            },
            rng=random.Random(),
            split_files=True,
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
            player_count=_player_count_for_eval(start_masks[env_index], env_index),
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
    stats_by_player_count = {
        player_count: _EvalStats.empty() for player_count in PLAYER_COUNTS
    }
    env_metrics: dict[str, list[float]] = {}
    hidden_current = current_model.initial_hidden_state(n_envs, device=device)
    hidden_last_best = last_best_model.initial_hidden_state(n_envs, device=device)
    while games < n_games:
        actions, hidden_current, hidden_last_best = (
            _eval_actions_for_assignments_and_hidden(
                obs,
                assignments,
                current_model=current_model,
                last_best_model=last_best_model,
                hidden_current=hidden_current,
                hidden_last_best=hidden_last_best,
                config=cfg.rl,
                device=device,
            )
        )
        obs, rewards, dones, _step_env_metrics = env.step(actions)
        hidden_current = current_model.reset_hidden_state(hidden_current, dones)
        hidden_last_best = last_best_model.reset_hidden_state(hidden_last_best, dones)
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
                player_count = _player_count_for_eval(start_masks[env_index], env_index)
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
                _record_eval_terminal_result(
                    stats_by_player_count[player_count],
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
                player_count=_player_count_for_eval(start_masks[env_index], env_index),
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
    return stats, stats_by_player_count, env_metrics, steps


def _player_count_for_eval(active_slots: torch.Tensor, env_index: int) -> int:
    player_count = int(active_slots.sum().item())
    if player_count not in PLAYER_COUNTS:
        raise ValueError(
            f"expected 2 or 4 active players, got {player_count} for env {env_index}"
        )
    return player_count


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

    pattern = torch.tensor(_eval_assignment_pattern(player_count), dtype=torch.int64)
    pattern = pattern[torch.randperm(pattern.numel())]
    assignments[env_index].fill_(-1)
    for slot, model_index in zip(active, pattern, strict=True):
        assignments[env_index, slot] = int(model_index.item())


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
) -> ActionBundle:
    actions, _hidden_current, _hidden_last_best = (
        _eval_actions_for_assignments_and_hidden(
            obs,
            assignments,
            current_model=current_model,
            last_best_model=last_best_model,
            hidden_current=None,
            hidden_last_best=None,
            config=config,
            device=device,
        )
    )
    return actions


def _eval_actions_for_assignments_and_hidden(
    obs: ObsBatch,
    assignments: torch.Tensor,
    *,
    current_model: BaseModelAPI,
    last_best_model: BaseModelAPI,
    hidden_current: ModelHiddenState | None,
    hidden_last_best: ModelHiddenState | None,
    config: DTypeConfig,
    device: torch.device,
) -> tuple[ActionBundle, ModelHiddenState | None, ModelHiddenState | None]:
    device_obs = _obs_to_device(obs, device)
    with autocast_context(config, device):
        current_output = _model_output_for_eval(
            current_model,
            device_obs,
            deterministic=False,
            hidden_state=hidden_current,
        )
        last_best_output = _model_output_for_eval(
            last_best_model,
            device_obs,
            deterministic=False,
            hidden_state=hidden_last_best,
        )
    use_current = assignments.to(device=device).eq(MODEL_CURRENT)
    return (
        _select_actions(current_output.actions, last_best_output.actions, use_current),
        current_output.next_hidden_state,
        last_best_output.next_hidden_state,
    )


def _model_output_for_eval(
    model: BaseModelAPI,
    obs: ObsBatch,
    *,
    deterministic: bool,
    hidden_state: ModelHiddenState | None,
) -> ModelOutput:
    if hidden_state is None:
        return model(obs, deterministic=deterministic)
    return model(obs, deterministic=deterministic, hidden_state=hidden_state)


def _obs_to_device(obs: ObsBatch, device: torch.device) -> ObsBatch:
    return ObsBatch(
        **{
            field: getattr(obs, field).to(
                device=device,
                non_blocking=device.type == "cuda",
            )
            for field in ObsBatch.model_fields
            if field != "action_mask"
        },
        action_mask=_action_mask_to_device(obs, device),
    )


def _action_mask_to_device(obs: ObsBatch, device: torch.device) -> ActionMask:
    if isinstance(obs.action_mask, PureActionMask):
        return PureActionMask(
            can_act=obs.action_mask.can_act.to(
                device=device,
                non_blocking=device.type == "cuda",
            ),
            max_launch=obs.action_mask.max_launch.to(
                device=device,
                non_blocking=device.type == "cuda",
            ),
        )
    if isinstance(obs.action_mask, DiscreteTargetActionMask):
        return DiscreteTargetActionMask(
            can_act=obs.action_mask.can_act.to(
                device=device,
                non_blocking=device.type == "cuda",
            ),
            max_launch=obs.action_mask.max_launch.to(
                device=device,
                non_blocking=device.type == "cuda",
            ),
        )
    return DiscreteTargetBinActionMask(
        can_act=obs.action_mask.can_act.to(
            device=device,
            non_blocking=device.type == "cuda",
        )
    )


def _select_actions(
    actions_a: ActionBundle,
    actions_b: ActionBundle,
    use_a: torch.Tensor,
) -> ActionBundle:
    if type(actions_a) is not type(actions_b):
        raise ValueError("checkpoint action bundles must have matching kinds")
    if isinstance(actions_a, PureActions) and isinstance(actions_b, PureActions):
        action_mask = use_a[:, :, None, None]
        return PureActions(
            launch=_select_action(actions_a.launch, actions_b.launch, action_mask),
            angle=_select_action(actions_a.angle, actions_b.angle, action_mask),
            ships=_select_action(actions_a.ships, actions_b.ships, action_mask),
        )
    if isinstance(actions_a, DiscreteTargetActions) and isinstance(
        actions_b,
        DiscreteTargetActions,
    ):
        action_mask = use_a[:, :, None, None]
        return DiscreteTargetActions(
            launch=_select_action(actions_a.launch, actions_b.launch, action_mask),
            target=_select_action(actions_a.target, actions_b.target, action_mask),
            ships=_select_action(actions_a.ships, actions_b.ships, action_mask),
        )
    if isinstance(actions_a, DiscreteTargetBinActions) and isinstance(
        actions_b,
        DiscreteTargetBinActions,
    ):
        action_mask = use_a[:, :, None]
        return DiscreteTargetBinActions(
            target=_select_action(actions_a.target, actions_b.target, action_mask),
            fleet_bin=_select_action(
                actions_a.fleet_bin,
                actions_b.fleet_bin,
                action_mask,
            ),
        )
    raise ValueError("unsupported checkpoint action bundle kind")


def _select_action(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    while mask.ndim < tensor_a.ndim:
        mask = mask.unsqueeze(-1)
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
    distributed: DistributedContext | None = None,
) -> bool:
    should_stop = max_env_steps is not None and env_steps >= max_env_steps
    should_stop = should_stop or (
        max_runtime_seconds is not None
        and time.monotonic() - started_at >= max_runtime_seconds
    )
    if distributed is not None and distributed.initialized:
        return all_reduce_any(should_stop, distributed)
    return should_stop


if __name__ == "__main__":
    main()
