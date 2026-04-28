from __future__ import annotations

import argparse
import itertools
import time
from datetime import datetime
from pathlib import Path
from typing import Any, assert_never

import torch
import yaml
from owl.model import ModelConfig, StatelessTransformerV1
from owl.rl import VectorizedEnv
from owl.train import FullConfig, PPOTrainer
from owl.train.logging import LogMode, create_logger
from owl.train.optimizer import create_lr_scheduler, create_optimizer
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
    env_steps_per_iteration = cfg.rl.horizon * cfg.rl.n_envs
    env_steps = 0

    try:
        with tqdm(unit="env steps", dynamic_ncols=True) as progress:
            while True:
                metrics = trainer.train_iteration()
                env_steps += env_steps_per_iteration
                progress.update(env_steps_per_iteration)
                logger.log({**metrics, "env_steps": float(env_steps)}, step=env_steps)
    finally:
        logger.close()


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
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
