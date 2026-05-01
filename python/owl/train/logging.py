from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import assert_never

from owl.train import FullConfig


class LogMode(StrEnum):
    DEBUG = auto()
    WANDB = auto()


class MetricLogger:
    def log(self, metrics: dict[str, float], *, step: int) -> None:
        raise NotImplementedError

    def set_summary(self, key: str, value: int | float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class DebugLogger(MetricLogger):
    def log(self, metrics: dict[str, float], *, step: int) -> None:
        del step
        print(metrics)

    def set_summary(self, key: str, value: int | float) -> None:
        del key, value

    def close(self) -> None:
        return None


class WandbLogger(MetricLogger):
    def __init__(self, run_dir: Path, cfg: FullConfig) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project="orbit-wars",
            dir=run_dir,
            name=run_dir.name,
            config=cfg.model_dump(mode="json"),
        )

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self._wandb.log(metrics, step=step)

    def set_summary(self, key: str, value: int | float) -> None:
        run = self._wandb.run
        if run is None:
            raise RuntimeError("wandb run is not initialized")
        run.summary[key] = value

    def close(self) -> None:
        self._run.finish()


def create_logger(log_mode: LogMode, run_dir: Path, cfg: FullConfig) -> MetricLogger:
    match log_mode:
        case LogMode.DEBUG:
            return DebugLogger()
        case LogMode.WANDB:
            return WandbLogger(run_dir, cfg)
        case _:
            assert_never(log_mode)
