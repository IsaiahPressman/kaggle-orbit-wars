from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from owl.model.actor.common import FeedForward, OutputProjectionMLP
from owl.model.actor.config import ActorDiscreteTargetBinsConfig
from owl.model.actor.discrete_targets import (
    DiscreteActorInputs,
    DiscreteTargetSelectionParams,
    gather_selected_target_values,
)
from owl.model.base import (
    InputLayer,
    ModelActionEntropies,
    ModelActionLogProbs,
)
from owl.rl import ACTION_ENTITY_SLOTS, OUTER_PLAYER_SLOTS, DiscreteTargetBinActions


@dataclass(frozen=True)
class DiscreteTargetBinsPolicyParams:
    target_logits: torch.Tensor
    fleet_bin_logits: torch.Tensor


class DiscreteTargetBinsActor(nn.Module):
    def __init__(
        self,
        config: ActorDiscreteTargetBinsConfig,
        *,
        transformer_config: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.head_dim = transformer_config.embed_dim

        self.source_role = nn.Parameter(torch.empty(1, transformer_config.embed_dim))
        self.target_role = nn.Parameter(torch.empty(1, transformer_config.embed_dim))
        _init_token_parameter(self.source_role)
        _init_token_parameter(self.target_role)
        self.source_norm = nn.LayerNorm(transformer_config.embed_dim)
        self.target_norm = nn.LayerNorm(transformer_config.embed_dim)
        self.q = nn.Linear(transformer_config.embed_dim, transformer_config.embed_dim)
        self.k = nn.Linear(transformer_config.embed_dim, transformer_config.embed_dim)
        self.v = nn.Linear(transformer_config.embed_dim, transformer_config.embed_dim)
        self.out = nn.Linear(transformer_config.embed_dim, transformer_config.embed_dim)
        self.norm2 = nn.LayerNorm(transformer_config.embed_dim)
        self.mlp = FeedForward(transformer_config)
        self.bin_pair_proj = nn.Linear(
            transformer_config.embed_dim,
            transformer_config.embed_dim,
        )
        self.fleet_bin_head = OutputProjectionMLP(
            transformer_config,
            self.config.n_bins,
        )

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        return (self.source_role, self.target_role)

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        return (self.fleet_bin_head.out,)

    def forward(
        self,
        actor_inputs: DiscreteActorInputs,
        can_act: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[DiscreteTargetBinActions, ModelActionLogProbs, ModelActionEntropies]:
        selection = self._selection_params(actor_inputs, can_act)
        source_active = can_act.flatten(start_dim=-2).any(dim=-1)
        if deterministic:
            target = selection.target_logits.argmax(dim=-1)
        else:
            target = Categorical(logits=selection.target_logits.float()).sample()
        target = torch.where(source_active, target, torch.zeros_like(target))

        params = self._policy_params_for_selected_target(
            selection,
            actor_inputs.source,
            can_act,
            target,
        )
        if deterministic:
            fleet_bin = params.fleet_bin_logits.argmax(dim=-1)
        else:
            fleet_bin = Categorical(logits=params.fleet_bin_logits.float()).sample()
        fleet_bin = torch.where(source_active, fleet_bin, torch.zeros_like(fleet_bin))

        target_log_prob, fleet_bin_log_prob = discrete_target_bin_log_probs(
            params,
            target,
            fleet_bin,
            source_active,
        )
        target_entropy, fleet_bin_entropy = self._entropy(
            selection,
            actor_inputs.source,
            can_act,
            source_active,
        )
        per_player_entity_log_prob = target_log_prob + fleet_bin_log_prob
        per_player_entity_entropy = target_entropy + fleet_bin_entropy

        zeros = torch.zeros_like(per_player_entity_log_prob)
        return (
            DiscreteTargetBinActions(target=target, fleet_bin=fleet_bin),
            ModelActionLogProbs(
                launch=zeros.unsqueeze(-1),
                target=target_log_prob.unsqueeze(-1),
                event=fleet_bin_log_prob.unsqueeze(-1),
                per_player_entity=per_player_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=zeros.unsqueeze(-1),
                target=target_entropy.unsqueeze(-1),
                event=fleet_bin_entropy.unsqueeze(-1),
                per_player_entity=per_player_entity_entropy,
                components={
                    "target": target_entropy,
                    "fleet_bin": fleet_bin_entropy,
                },
            ),
        )

    def log_prob(
        self,
        actor_inputs: DiscreteActorInputs,
        can_act: torch.Tensor,
        actions: DiscreteTargetBinActions,
    ) -> tuple[ModelActionLogProbs, ModelActionEntropies]:
        _require_discrete_target_bin_actions_shape(
            actions,
            (
                actor_inputs.source.shape[0],
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
            ),
        )
        selection = self._selection_params(actor_inputs, can_act)
        source_active = can_act.flatten(start_dim=-2).any(dim=-1)
        _require_valid_discrete_target_bin_action(
            actions.target,
            actions.fleet_bin,
            can_act,
            source_active,
        )
        params = self._policy_params_for_selected_target(
            selection,
            actor_inputs.source,
            can_act,
            actions.target.clamp(0, ACTION_ENTITY_SLOTS - 1),
        )
        target_log_prob, fleet_bin_log_prob = discrete_target_bin_log_probs(
            params,
            actions.target,
            actions.fleet_bin,
            source_active,
        )
        target_entropy, fleet_bin_entropy = self._entropy(
            selection,
            actor_inputs.source,
            can_act,
            source_active,
        )
        per_player_entity_log_prob = target_log_prob + fleet_bin_log_prob
        per_player_entity_entropy = target_entropy + fleet_bin_entropy

        zeros = torch.zeros_like(per_player_entity_log_prob)
        return (
            ModelActionLogProbs(
                launch=zeros.unsqueeze(-1),
                target=target_log_prob.unsqueeze(-1),
                event=fleet_bin_log_prob.unsqueeze(-1),
                per_player_entity=per_player_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=zeros.unsqueeze(-1),
                target=target_entropy.unsqueeze(-1),
                event=fleet_bin_entropy.unsqueeze(-1),
                per_player_entity=per_player_entity_entropy,
                components={
                    "target": target_entropy,
                    "fleet_bin": fleet_bin_entropy,
                },
            ),
        )

    def _selection_params(
        self,
        actor_inputs: DiscreteActorInputs,
        can_act: torch.Tensor,
    ) -> DiscreteTargetSelectionParams:
        source_input = actor_inputs.source
        target_input = actor_inputs.target
        expected_input_shape = (
            source_input.shape[0],
            OUTER_PLAYER_SLOTS,
            ACTION_ENTITY_SLOTS,
            self.head_dim,
        )
        if source_input.shape != expected_input_shape:
            raise ValueError(
                "discrete target-bin source input must have shape "
                f"{expected_input_shape}, got {tuple(source_input.shape)}"
            )
        if target_input.shape != expected_input_shape:
            raise ValueError(
                "discrete target-bin target input must have shape "
                f"{expected_input_shape}, got {tuple(target_input.shape)}"
            )
        expected_can_act_shape = (
            source_input.shape[0],
            OUTER_PLAYER_SLOTS,
            ACTION_ENTITY_SLOTS,
            ACTION_ENTITY_SLOTS,
            self.config.n_bins,
        )
        if can_act.shape != expected_can_act_shape:
            raise ValueError(
                "discrete target-bin can_act must have shape "
                f"{expected_can_act_shape}, got {tuple(can_act.shape)}"
            )

        source_role = self.source_role.to(dtype=source_input.dtype)
        target_role = self.target_role.to(dtype=target_input.dtype)
        source_x = self.source_norm(source_input + source_role)
        target_x = self.target_norm(target_input + target_role)
        q = self.q(source_x)
        k = self.k(target_x)
        v = self.v(target_x)
        target_logits = torch.einsum("bpsd,bptd->bpst", q, k)
        target_logits = target_logits / math.sqrt(self.head_dim)
        target_valid = can_act.any(dim=-1)
        target_logits = target_logits.masked_fill(
            ~target_valid,
            torch.finfo(target_logits.dtype).min,
        )
        safe_target_logits = torch.where(
            target_valid.any(dim=-1, keepdim=True),
            target_logits,
            torch.zeros_like(target_logits),
        )
        return DiscreteTargetSelectionParams(
            continue_logits=torch.zeros_like(safe_target_logits[..., 0]),
            target_logits=safe_target_logits,
            target_values=v,
        )

    def _policy_params_for_selected_target(
        self,
        selection: DiscreteTargetSelectionParams,
        source_input: torch.Tensor,
        can_act: torch.Tensor,
        target_index: torch.Tensor,
    ) -> DiscreteTargetBinsPolicyParams:
        selected_target_values = gather_selected_target_values(
            selection.target_values,
            target_index,
        )
        fleet_bin_logits = self._fleet_bin_logits_from_target_values(
            source_input,
            selected_target_values,
        )
        selected_bin_mask = gather_selected_bin_mask(can_act, target_index)
        fleet_bin_logits = masked_safe_logits(fleet_bin_logits, selected_bin_mask)
        return DiscreteTargetBinsPolicyParams(
            target_logits=selection.target_logits,
            fleet_bin_logits=fleet_bin_logits,
        )

    def _fleet_bin_logits_from_target_values(
        self,
        source_input: torch.Tensor,
        target_values: torch.Tensor,
    ) -> torch.Tensor:
        selected_v = self.out(target_values)
        enriched = source_input + selected_v
        enriched = enriched + self.mlp(self.norm2(enriched))
        source_hidden = self.bin_pair_proj(enriched)
        return self.fleet_bin_head(source_hidden)

    def _entropy(
        self,
        selection: DiscreteTargetSelectionParams,
        source_input: torch.Tensor,
        can_act: torch.Tensor,
        source_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_entropy = categorical_entropy(
            selection.target_logits,
            can_act.any(dim=-1),
        )
        target_index = selection.target_logits.argmax(dim=-1)
        params = self._policy_params_for_selected_target(
            selection,
            source_input,
            can_act,
            target_index,
        )
        selected_bin_mask = gather_selected_bin_mask(can_act, target_index)
        fleet_bin_entropy = categorical_entropy(
            params.fleet_bin_logits,
            selected_bin_mask,
        )
        return (
            torch.where(
                source_active,
                target_entropy,
                torch.zeros_like(target_entropy),
            ),
            torch.where(
                source_active,
                fleet_bin_entropy,
                torch.zeros_like(fleet_bin_entropy),
            ),
        )


def _init_token_parameter(parameter: nn.Parameter) -> None:
    nn.init.normal_(parameter, mean=0.0, std=parameter.shape[-1] ** -0.5)


def gather_selected_bin_mask(
    can_act: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    gather_index = target_index[..., None, None].expand(
        *target_index.shape,
        1,
        can_act.shape[-1],
    )
    return can_act.gather(dim=3, index=gather_index).squeeze(3)


def masked_safe_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.where(
        mask.any(dim=-1, keepdim=True),
        masked_logits,
        torch.zeros_like(masked_logits),
    )


def discrete_target_bin_log_probs(
    params: DiscreteTargetBinsPolicyParams,
    target: torch.Tensor,
    fleet_bin: torch.Tensor,
    source_active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_target = target.clamp(0, ACTION_ENTITY_SLOTS - 1)
    safe_fleet_bin = fleet_bin.clamp(0, params.fleet_bin_logits.shape[-1] - 1)
    target_log_all = F.log_softmax(params.target_logits.float(), dim=-1)
    target_log_prob = target_log_all.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    fleet_bin_log_all = F.log_softmax(params.fleet_bin_logits.float(), dim=-1)
    fleet_bin_log_prob = fleet_bin_log_all.gather(
        -1,
        safe_fleet_bin.unsqueeze(-1),
    ).squeeze(-1)
    return (
        torch.where(source_active, target_log_prob, torch.zeros_like(target_log_prob)),
        torch.where(
            source_active,
            fleet_bin_log_prob,
            torch.zeros_like(fleet_bin_log_prob),
        ),
    )


def categorical_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits.float(), dim=-1)
    prob = log_prob.exp()
    return -(prob * log_prob).masked_fill(~mask, 0.0).sum(dim=-1)


def _require_discrete_target_bin_actions_shape(
    actions: DiscreteTargetBinActions,
    expected_shape: tuple[int, int, int],
) -> None:
    for name, tensor in (
        ("target", actions.target),
        ("fleet_bin", actions.fleet_bin),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"actions.{name} must have shape {expected_shape}, got {tensor.shape}"
            )
        if tensor.dtype != torch.int64:
            raise ValueError(
                f"actions.{name} must have dtype torch.int64, got {tensor.dtype}"
            )

def _require_valid_discrete_target_bin_action(
    target: torch.Tensor,
    fleet_bin: torch.Tensor,
    can_act: torch.Tensor,
    source_active: torch.Tensor,
) -> None:
    target_in_range = target.ge(0) & target.lt(ACTION_ENTITY_SLOTS)
    fleet_bin_in_range = fleet_bin.ge(0) & fleet_bin.lt(can_act.shape[-1])
    safe_target = target.clamp(0, ACTION_ENTITY_SLOTS - 1)
    safe_fleet_bin = fleet_bin.clamp(0, can_act.shape[-1] - 1)
    target_index = safe_target[..., None, None].expand(
        *safe_target.shape,
        1,
        can_act.shape[-1],
    )
    selected_target_mask = can_act.gather(dim=3, index=target_index).squeeze(3)
    selected_valid = selected_target_mask.gather(
        -1,
        safe_fleet_bin.unsqueeze(-1),
    ).squeeze(-1)
    invalid_active = source_active & (
        ~target_in_range | ~fleet_bin_in_range | ~selected_valid
    )
    if invalid_active.any().item():
        raise ValueError(
            "actions.target and actions.fleet_bin must select a valid target-bin "
            "pair for active source slots"
        )
