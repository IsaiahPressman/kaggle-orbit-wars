from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from owl.rl import ObsBatch


@dataclass
class ModelActions:
    launch: torch.Tensor
    ships: torch.Tensor
    angle: torch.Tensor | None = None
    target: torch.Tensor | None = None

    def action_value(self) -> torch.Tensor:
        if self.angle is not None and self.target is None:
            return self.angle
        if self.target is not None and self.angle is None:
            return self.target
        raise ValueError("exactly one of actions.angle or actions.target must be set")


@dataclass
class ModelActionLogProbs:
    launch: torch.Tensor
    angle_and_size: torch.Tensor
    per_player_entity: torch.Tensor
    target: torch.Tensor | None = None


@dataclass
class ModelActionEntropies:
    launch: torch.Tensor
    angle_and_size: torch.Tensor
    per_player_entity: torch.Tensor
    target: torch.Tensor | None = None


@dataclass
class ModelOutput:
    actions: ModelActions
    log_probs: ModelActionLogProbs
    entropies: ModelActionEntropies
    values: torch.Tensor
    winner_probabilities: torch.Tensor


@dataclass
class ModelEvaluation:
    log_probs: ModelActionLogProbs
    entropies: ModelActionEntropies
    values: torch.Tensor
    winner_probabilities: torch.Tensor


class BaseModelAPI(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
    ) -> ModelOutput: ...

    @abstractmethod
    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ModelActions,
    ) -> ModelEvaluation: ...

    def compute_value(self, obs: ObsBatch) -> torch.Tensor:
        return self(obs, deterministic=True).values

    @abstractmethod
    def get_input_layers(self) -> tuple[nn.Module, ...]: ...

    @abstractmethod
    def get_output_layers(self) -> tuple[nn.Module, ...]: ...
