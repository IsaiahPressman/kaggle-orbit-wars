from __future__ import annotations

from collections.abc import Iterable
from math import cos, pi
from typing import Annotated, Any, Literal, Protocol, TypeAlias, assert_never

import torch
from pydantic import Field
from torch import nn

from owl.config import BaseConfig
from owl.model import BaseModelAPI, InputLayer

StateDict: TypeAlias = dict[str, Any]
AdamBeta: TypeAlias = Annotated[float, Field(ge=0.0, lt=1.0)]


class LinearWarmupCosineDecayLRScheduleConfig(BaseConfig):
    schedule: Literal["linear_warmup_cosine_decay"] = "linear_warmup_cosine_decay"
    warmup_steps: int = Field(default=0, ge=0)
    decay_steps: int = Field(default=1, ge=1)
    lr_min_ratio: float = Field(default=1e-3, ge=1e-3, le=0.1)


class CosineLRScheduleConfig(BaseConfig):
    schedule: Literal["cosine"] = "cosine"
    phase_steps: int = Field(default=1, ge=1)
    lr_min_ratio: float = Field(default=1e-3, ge=0.0, lt=1.0)


LRScheduleConfig: TypeAlias = Annotated[
    LinearWarmupCosineDecayLRScheduleConfig | CosineLRScheduleConfig,
    Field(discriminator="schedule"),
]


class Optimizer(Protocol):
    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def step(self) -> None: ...

    def state_dict(self) -> StateDict: ...

    def load_state_dict(self, state_dict: StateDict) -> None: ...


class LRScheduler(Protocol):
    def step(self) -> None: ...

    def get_last_lr(self) -> list[float]: ...

    def state_dict(self) -> StateDict: ...

    def load_state_dict(self, state_dict: StateDict) -> None: ...


class AdamConfig(BaseConfig):
    optimizer: Literal["adam"] = "adam"
    learning_rate: float = Field(default=3e-4, gt=0.0)
    betas: tuple[AdamBeta, AdamBeta] = (0.9, 0.999)
    eps: float = Field(default=1e-5, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    lr_schedule: LRScheduleConfig | None = None


class AdamWConfig(BaseConfig):
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=3e-4, gt=0.0)
    betas: tuple[AdamBeta, AdamBeta] = (0.9, 0.999)
    adamw_eps: float = Field(default=1e-5, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    lr_schedule: LRScheduleConfig | None = None


class MuonConfig(BaseConfig):
    optimizer: Literal["muon"] = "muon"
    adamw_lr: float = Field(default=3e-4, gt=0.0)
    adamw_eps: float = Field(default=1e-5, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    muon_lr: float = Field(default=1e-2, gt=0.0)
    muon_weight_decay: float = Field(default=0.1, ge=0.0)
    muon_momentum: float = Field(default=0.95, ge=0.0, lt=1.0)
    lr_schedule: LRScheduleConfig | None = None


OptimizerConfig: TypeAlias = Annotated[
    AdamConfig | AdamWConfig | MuonConfig, Field(discriminator="optimizer")
]


class CompositeOptimizer:
    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]) -> None:
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("CompositeOptimizer requires at least one optimizer")

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> StateDict:
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        optimizer_states = state_dict["optimizers"]
        if not isinstance(optimizer_states, list):
            raise ValueError("CompositeOptimizer state must contain optimizer states")
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                "CompositeOptimizer state optimizer count must match current "
                f"optimizer count {len(self.optimizers)}, got {len(optimizer_states)}"
            )
        for optimizer, optimizer_state in zip(
            self.optimizers,
            optimizer_states,
            strict=True,
        ):
            optimizer.load_state_dict(optimizer_state)


class CompositeLRScheduler:
    def __init__(self, schedulers: Iterable[torch.optim.lr_scheduler.LambdaLR]) -> None:
        self.schedulers = list(schedulers)
        if not self.schedulers:
            raise ValueError("CompositeLRScheduler requires at least one scheduler")

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def get_last_lr(self) -> list[float]:
        return [lr for scheduler in self.schedulers for lr in scheduler.get_last_lr()]

    def state_dict(self) -> StateDict:
        return {
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        scheduler_states = state_dict["schedulers"]
        if not isinstance(scheduler_states, list):
            raise ValueError("CompositeLRScheduler state must contain scheduler states")
        if len(scheduler_states) != len(self.schedulers):
            raise ValueError(
                "CompositeLRScheduler state scheduler count must match current "
                f"scheduler count {len(self.schedulers)}, got {len(scheduler_states)}"
            )
        for scheduler, scheduler_state in zip(
            self.schedulers,
            scheduler_states,
            strict=True,
        ):
            scheduler.load_state_dict(scheduler_state)


def create_lr_scheduler(
    optimizer: Optimizer,
    config: LRScheduleConfig | None,
) -> LRScheduler | None:
    if config is None:
        return None
    if isinstance(optimizer, CompositeOptimizer):
        return CompositeLRScheduler(
            torch.optim.lr_scheduler.LambdaLR(
                inner_optimizer,
                lr_lambda=lambda step: lr_multiplier(config, step),
            )
            for inner_optimizer in optimizer.optimizers
        )
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer or CompositeOptimizer")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(config, step),
    )


def lr_multiplier(config: LRScheduleConfig, step: int) -> float:
    if step < 0:
        raise ValueError("step must be non-negative")

    match config.schedule:
        case "linear_warmup_cosine_decay":
            if config.warmup_steps > 0 and step < config.warmup_steps:
                return step / config.warmup_steps

            decay_step = min(
                max(step - config.warmup_steps, 0),
                config.decay_steps,
            )
            progress = decay_step / config.decay_steps
            cosine = 0.5 * (1.0 + cos(pi * progress))
            return config.lr_min_ratio + (1.0 - config.lr_min_ratio) * cosine
        case "cosine":
            phase_progress = (step % (2 * config.phase_steps)) / config.phase_steps
            cosine = 0.5 * (1.0 + cos(pi * phase_progress))
            return config.lr_min_ratio + (1.0 - config.lr_min_ratio) * cosine
        case _:
            assert_never(config.schedule)


def create_optimizer(model: BaseModelAPI, config: OptimizerConfig) -> Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            fused=True,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.adamw_eps,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "muon":
        muon_params: list[nn.Parameter] = []
        adamw_params: list[nn.Parameter] = []
        excluded_param_ids = _excluded_from_muon_param_ids(model)
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 2 and id(param) not in excluded_param_ids:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(
                torch.optim.Muon(
                    muon_params,
                    lr=config.muon_lr,
                    weight_decay=config.muon_weight_decay,
                    momentum=config.muon_momentum,
                )
            )
        if adamw_params:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_params,
                    lr=config.adamw_lr,
                    eps=config.adamw_eps,
                    weight_decay=config.weight_decay,
                )
            )
        return CompositeOptimizer(optimizers)
    assert_never(config.optimizer)


def _excluded_from_muon_param_ids(model: BaseModelAPI) -> set[int]:
    return {
        id(param)
        for layer in (*model.get_input_layers(), *model.get_output_layers())
        for param in _layer_parameters(layer)
    }


def _layer_parameters(layer: InputLayer) -> tuple[nn.Parameter, ...]:
    if isinstance(layer, nn.Parameter):
        return (layer,)
    return tuple(layer.parameters())
