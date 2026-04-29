from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from owl.rl import ObsBatch


@dataclass
class ModelActions:
    launch: torch.Tensor
    angle: torch.Tensor
    ships: torch.Tensor


@dataclass
class ModelActionLogProbs:
    launch: torch.Tensor
    angle_and_size: torch.Tensor
    per_player_entity: torch.Tensor


@dataclass
class ModelActionEntropies:
    launch: torch.Tensor
    angle_and_size: torch.Tensor
    per_player_entity: torch.Tensor


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

    @abstractmethod
    def get_input_layers(self) -> tuple[nn.Module, ...]: ...

    @abstractmethod
    def get_output_layers(self) -> tuple[nn.Module, ...]: ...
