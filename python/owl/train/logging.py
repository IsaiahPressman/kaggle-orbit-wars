from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Any, assert_never

from owl.train import FullConfig


class LogMode(StrEnum):
    DEBUG = auto()
    WANDB = auto()


class MetricLogger:
    @property
    def run_id(self) -> str | None:
        raise NotImplementedError

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        raise NotImplementedError

    def set_summary(self, key: str, value: int | float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class DebugLogger(MetricLogger):
    @property
    def run_id(self) -> str | None:
        return None

    def log(self, metrics: dict[str, float], *, step: int) -> None:  # noqa: ARG002
        print(metrics)

    def set_summary(
        self,
        key: str,  # noqa: ARG002
        value: int | float,  # noqa: ARG002
    ) -> None:
        return None

    def close(self) -> None:
        return None


class WandbLogger(MetricLogger):
    def __init__(
        self,
        run_dir: Path,
        cfg: FullConfig,
        *,
        resume_run_id: str | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        init_kwargs: dict[str, Any] = {}
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"
        self._run = wandb.init(
            project="orbit-wars",
            dir=run_dir,
            name=run_dir.name,
            config=cfg.model_dump(mode="json"),
            **init_kwargs,
        )

    @property
    def run_id(self) -> str | None:
        return self._run.id

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self._wandb.log(metrics, step=step)

    def set_summary(self, key: str, value: int | float) -> None:
        run = self._wandb.run
        if run is None:
            raise RuntimeError("wandb run is not initialized")
        run.summary[key] = value

    def close(self) -> None:
        self._run.finish()


def create_logger(
    log_mode: LogMode,
    run_dir: Path,
    cfg: FullConfig,
    *,
    resume_run_id: str | None = None,
) -> MetricLogger:
    match log_mode:
        case LogMode.DEBUG:
            return DebugLogger()
        case LogMode.WANDB:
            return WandbLogger(run_dir, cfg, resume_run_id=resume_run_id)
        case _:
            assert_never(log_mode)
