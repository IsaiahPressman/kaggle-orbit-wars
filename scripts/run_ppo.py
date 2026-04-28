from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, assert_never

import numpy as np
import torch
import yaml
from owl.model import ModelConfig, StatelessTransformerV1
from owl.rl import VectorizedEnv
from owl.train import FullConfig, PPOTrainer
from owl.train.logging import LogMode, create_logger
from owl.train.optimizer import (
    LRScheduler,
    Optimizer,
    create_lr_scheduler,
    create_optimizer,
)
from tqdm import tqdm


def main() -> None:
    args = _parse_args()
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
    model = _create_model(cfg.model).to(device)
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

    logger = create_logger(args.log_mode, run_dir, cfg)
    env_steps_per_iteration = cfg.rl.horizon * env.n_envs
    env_steps = 0
    started_at = time.monotonic()
    next_checkpoint_env_steps = _next_periodic_checkpoint_step(
        checkpoint_every_env_steps=args.checkpoint_every_env_steps,
    )

    try:
        with tqdm(unit="env steps", dynamic_ncols=True) as progress:
            while True:
                metrics = trainer.train_iteration()
                env_steps += env_steps_per_iteration
                progress.update(env_steps_per_iteration)
                logger.log({**metrics, "env_steps": float(env_steps)}, step=env_steps)
                if (
                    next_checkpoint_env_steps is not None
                    and env_steps >= next_checkpoint_env_steps
                ):
                    _write_checkpoint(
                        run_dir / f"checkpoint-{env_steps}.pt",
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        cfg=cfg,
                        config_path=args.config,
                        env_steps=env_steps,
                    )
                    next_checkpoint_env_steps = _next_periodic_checkpoint_step(
                        checkpoint_every_env_steps=args.checkpoint_every_env_steps,
                        env_steps=env_steps,
                    )
                if _should_stop_training(
                    env_steps=env_steps,
                    started_at=started_at,
                    max_env_steps=args.max_env_steps,
                    max_runtime_seconds=args.max_runtime_seconds,
                ):
                    break
    finally:
        active_exception = sys.exception()
        cleanup_error: Exception | None = None
        if (
            "model" in locals()
            and "optimizer" in locals()
            and "lr_scheduler" in locals()
        ):
            try:
                _write_checkpoint(
                    run_dir / "checkpoint-final.pt",
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    cfg=cfg,
                    config_path=args.config,
                    env_steps=env_steps,
                )
            except Exception as exc:
                if active_exception is None:
                    cleanup_error = exc
                else:
                    traceback.print_exc()
        try:
            logger.close()
        except Exception as exc:
            if active_exception is None and cleanup_error is None:
                cleanup_error = exc
            else:
                traceback.print_exc()
        if cleanup_error is not None:
            raise cleanup_error


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
        "--checkpoint-every-env-steps",
        type=int,
        default=0,
        help="Write periodic checkpoints every N environment steps; 0 disables them",
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=None,
        help="Stop after at least this many environment steps",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=None,
        help="Stop after at least this many wall-clock seconds",
    )
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.checkpoint_every_env_steps < 0:
        raise ValueError("--checkpoint-every-env-steps must be non-negative")
    if args.max_env_steps is not None and args.max_env_steps <= 0:
        raise ValueError("--max-env-steps must be positive")
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0.0:
        raise ValueError("--max-runtime-seconds must be positive")


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


def _create_model(config: ModelConfig) -> StatelessTransformerV1:
    match config.model_arch:
        case "stateless_transformer_v1":
            return StatelessTransformerV1(config)
        case _:
            assert_never(config)


def _next_periodic_checkpoint_step(
    *, checkpoint_every_env_steps: int, env_steps: int = 0
) -> int | None:
    if checkpoint_every_env_steps <= 0:
        return None
    return (env_steps // checkpoint_every_env_steps + 1) * checkpoint_every_env_steps


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


def _write_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler | None,
    cfg: FullConfig,
    config_path: Path,
    env_steps: int,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": (None if lr_scheduler is None else lr_scheduler.state_dict()),
        "config": cfg.model_dump(mode="json", round_trip=True),
        "config_path": str(config_path),
        "rng_state": _rng_state(),
        "env_steps": env_steps,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


if __name__ == "__main__":
    main()
