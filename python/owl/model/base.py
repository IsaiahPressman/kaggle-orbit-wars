from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import nn

from owl.rl import ActionBundle, ActionConfig, ObsBatch

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
    components: dict[str, torch.Tensor]
    target: torch.Tensor | None = None


@dataclass
class ModelActionKLDivergences:
    launch: torch.Tensor
    event: torch.Tensor
    per_player_entity: torch.Tensor
    components: dict[str, torch.Tensor]
    target: torch.Tensor | None = None


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


@dataclass
class ModelTeacherEvaluation:
    student: ModelEvaluation
    action_kl: ModelActionKLDivergences | None
    teacher_winner_probabilities: torch.Tensor | None


@dataclass
class ModelServingOutput:
    actions: ModelActions
    values: torch.Tensor
    winner_probabilities: torch.Tensor
    next_hidden_state: ModelHiddenState | None = None


class BaseModelAPI(nn.Module, ABC):
    @property
    def action_spec(self) -> ActionConfig:
        return self._action_spec

    @action_spec.setter
    def action_spec(self, value: ActionConfig) -> None:
        self._action_spec = value

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

    def evaluate_action_kl(
        self,
        obs: ObsBatch,
        teacher: BaseModelAPI,
        actions: ModelActions,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
    ) -> ModelActionKLDivergences:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement action KL evaluation"
        )

    def evaluate_actions_with_teacher(
        self,
        obs: ObsBatch,
        actions: ModelActions,
        teacher: BaseModelAPI,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
        compute_teacher_action_kl: bool = True,
        compute_teacher_value: bool = True,
    ) -> ModelTeacherEvaluation:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement teacher evaluation"
        )

    def serve(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
        hidden_state: ModelHiddenState | None = None,
    ) -> ModelServingOutput:
        output = self.forward(
            obs,
            deterministic=deterministic,
            hidden_state=hidden_state,
        )
        return ModelServingOutput(
            actions=output.actions,
            values=output.values,
            winner_probabilities=output.winner_probabilities,
            next_hidden_state=output.next_hidden_state,
        )

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
