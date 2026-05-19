from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeAlias

import torch
from torch import nn

from owl.rl import ActionBundle, ObsBatch

InputLayer = nn.Module | nn.Parameter
ModelActions: TypeAlias = ActionBundle
ModelHiddenState: TypeAlias = object


@dataclass
class ModelActionLogProbs:
    launch: torch.Tensor
    event: torch.Tensor
    per_player_entity: torch.Tensor
    target: torch.Tensor | None = None


@dataclass
class ModelActionEntropies:
    launch: torch.Tensor
    event: torch.Tensor
    per_player_entity: torch.Tensor
    target: torch.Tensor | None = None
    components: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ModelOutput:
    actions: ModelActions
    log_probs: ModelActionLogProbs
    entropies: ModelActionEntropies
    values: torch.Tensor
    winner_probabilities: torch.Tensor
    next_hidden_state: ModelHiddenState | None = None


@dataclass
class ModelEvaluation:
    log_probs: ModelActionLogProbs
    entropies: ModelActionEntropies
    values: torch.Tensor
    winner_probabilities: torch.Tensor
    next_hidden_state: ModelHiddenState | None = None


class BaseModelAPI(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
        hidden_state: ModelHiddenState | None = None,
    ) -> ModelOutput: ...

    @abstractmethod
    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ModelActions,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
    ) -> ModelEvaluation: ...

    @abstractmethod
    def compute_value(
        self,
        obs: ObsBatch,
        *,
        hidden_state: ModelHiddenState | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def reset_parameters(self) -> None: ...

    @abstractmethod
    def get_input_layers(self) -> tuple[InputLayer, ...]: ...

    @abstractmethod
    def get_output_layers(self) -> tuple[nn.Module, ...]: ...

    def initial_hidden_state(
        self,
        batch_size: int,  # noqa: ARG002
        *,
        device: torch.device,  # noqa: ARG002
    ) -> ModelHiddenState | None:
        return None

    def detach_hidden_state(
        self,
        hidden_state: ModelHiddenState | None,
    ) -> ModelHiddenState | None:
        return hidden_state

    def index_hidden_state(
        self,
        hidden_state: ModelHiddenState | None,
        indices: torch.Tensor,  # noqa: ARG002
    ) -> ModelHiddenState | None:
        return hidden_state

    def reset_hidden_state(
        self,
        hidden_state: ModelHiddenState | None,
        dones: torch.Tensor,  # noqa: ARG002
    ) -> ModelHiddenState | None:
        return hidden_state
