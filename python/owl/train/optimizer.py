from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Annotated, Literal, assert_never, overload

import torch
from pydantic import Field
from torch import nn

from owl.config import BaseConfig
from owl.model import BaseModelAPI

OptimizerName = Literal["adamw", "muon"]


class AdamWConfig(BaseConfig):
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=3e-4, gt=0.0)
    adamw_eps: float = Field(default=1e-5, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)


class MuonConfig(BaseConfig):
    optimizer: Literal["muon"] = "muon"
    learning_rate: float = Field(default=3e-4, gt=0.0)
    adamw_eps: float = Field(default=1e-5, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    muon_lr: float | None = Field(default=None, gt=0.0)
    muon_weight_decay: float = Field(default=0.1, ge=0.0)
    muon_momentum: float = Field(default=0.95, ge=0.0, lt=1.0)


type OptimizerConfig = Annotated[
    AdamWConfig | MuonConfig, Field(discriminator="optimizer")
]


class CompositeOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]) -> None:
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("CompositeOptimizer requires at least one optimizer")
        params = [
            param
            for optimizer in self.optimizers
            for group in optimizer.param_groups
            for param in group["params"]
        ]
        super().__init__(params, defaults={})

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is not None:
            raise ValueError("CompositeOptimizer does not support closures")
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    def state_dict(self) -> dict[str, object]:
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
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


def create_optimizer(
    model: BaseModelAPI, config: OptimizerConfig
) -> torch.optim.Optimizer | CompositeOptimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
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
                    lr=config.muon_lr or config.learning_rate,
                    weight_decay=config.muon_weight_decay,
                    momentum=config.muon_momentum,
                )
            )
        if adamw_params:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_params,
                    lr=config.learning_rate,
                    eps=config.adamw_eps,
                    weight_decay=config.weight_decay,
                )
            )
        return CompositeOptimizer(optimizers)
    assert_never(config.optimizer)


def _excluded_from_muon_param_ids(model: BaseModelAPI) -> set[int]:
    excluded_modules = (*model.get_input_layers(), *model.get_output_layers())
    return {id(param) for module in excluded_modules for param in module.parameters()}
