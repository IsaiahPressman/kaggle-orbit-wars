from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from owl.model.actor.common import (
    FeedForward,
    OutputProjectionMLP,
    binary_entropy_from_logits,
    sample_launch,
)
from owl.model.actor.config import ActorDiscreteTargetsConfig
from owl.model.actor.logistic_mixture import (
    discretized_logistic_mixture_log_prob,
    log_interpolate,
    logistic_cdf_diff_logprob,
    logsubexp,
    sample_discretized_logistic_mixture,
    ship_support,
    truncated_logistic_mixture_entropy,
)
from owl.model.base import (
    InputLayer,
    ModelActionEntropies,
    ModelActionLogProbs,
)
from owl.rl import OUTER_PLAYER_SLOTS, DiscreteTargetActions

__all__ = [
    "DiscreteActorInputs",
    "DiscreteTargetPolicyParams",
    "DiscreteTargetSelectionParams",
    "DiscreteTargetSizeParams",
    "DiscreteTargetsActor",
    "discretized_logistic_mixture_log_prob",
    "log_interpolate",
    "logistic_cdf_diff_logprob",
    "logsubexp",
    "sample_discretized_logistic_mixture",
    "ship_support",
    "truncated_logistic_mixture_entropy",
]


@dataclass(frozen=True)
class DiscreteActorInputs:
    source: torch.Tensor
    target: torch.Tensor
    pairwise_bias: torch.Tensor | None = None


@dataclass(frozen=True)
class DiscreteTargetSelectionParams:
    target_logits: torch.Tensor
    target_values: torch.Tensor
    continue_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class DiscreteTargetSizeParams:
    size_mix_logits: torch.Tensor
    size_mu: torch.Tensor
    size_scale: torch.Tensor
    continue_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class DiscreteTargetPolicyParams:
    target_logits: torch.Tensor
    size_mix_logits: torch.Tensor
    size_mu: torch.Tensor
    size_scale: torch.Tensor
    continue_logits: torch.Tensor | None = None


class DiscreteTargetsActor(nn.Module):
    def __init__(
        self,
        config: ActorDiscreteTargetsConfig,
        *,
        transformer_config: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_heads = 1
        self.head_dim = transformer_config.embed_dim
        mixtures = config.n_action_mixtures

        self.source_role = nn.Parameter(torch.empty(1, transformer_config.embed_dim))
        self.target_role = nn.Parameter(torch.empty(1, transformer_config.embed_dim))
        if config.launch_mode == "target_token":
            self.no_launch_target = nn.Parameter(
                torch.empty(1, transformer_config.embed_dim)
            )
            _init_token_parameter(self.no_launch_target)
        else:
            self.register_parameter("no_launch_target", None)

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
        self.continue_source_proj: nn.Linear | None
        self.continue_head: OutputProjectionMLP | None
        if config.launch_mode == "binary":
            self.continue_source_proj = nn.Linear(
                transformer_config.embed_dim,
                transformer_config.embed_dim,
            )
            self.continue_head = OutputProjectionMLP(transformer_config, 1)
        elif config.launch_mode == "binary_after":
            self.continue_source_proj = None
            self.continue_head = OutputProjectionMLP(transformer_config, 1)
        else:
            self.continue_source_proj = None
            self.continue_head = None
        self.size_pair_proj = nn.Linear(
            transformer_config.embed_dim,
            transformer_config.embed_dim,
        )
        self.mix_head = OutputProjectionMLP(transformer_config, mixtures)
        self.mean_head = OutputProjectionMLP(transformer_config, mixtures)
        self.scale_head = OutputProjectionMLP(transformer_config, mixtures)

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        input_layers: tuple[InputLayer, ...] = (self.source_role, self.target_role)
        if self.no_launch_target is not None:
            input_layers = (*input_layers, self.no_launch_target)
        return input_layers

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        output_layers: tuple[nn.Linear, ...] = (
            self.mix_head.out,
            self.mean_head.out,
            self.scale_head.out,
        )
        if self.continue_head is not None:
            output_layers = (self.continue_head.out, *output_layers)
        return output_layers

    def forward(
        self,
        actor_inputs: DiscreteActorInputs,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        *,
        min_fleet_size: int,
        deterministic: bool,
    ) -> tuple[DiscreteTargetActions, ModelActionLogProbs, ModelActionEntropies]:
        actions, selection, params, source_active = self._sample_actions(
            actor_inputs,
            can_act,
            max_launch,
            min_fleet_size=min_fleet_size,
            deterministic=deterministic,
        )

        entropy_params = self._policy_params_for_entropy(
            selection,
            actor_inputs.source,
            max_launch,
            min_fleet_size=min_fleet_size,
        )
        launch_log_prob, target_log_prob, size_log_prob = discrete_action_log_probs(
            params,
            actions.launch[..., 0],
            actions.target[..., 0],
            actions.ships[..., 0],
            max_launch,
            source_active,
            self.config.launch_mode,
            min_fleet_size=min_fleet_size,
        )
        (
            launch_entropy,
            target_entropy,
            size_entropy,
            size_mixture_entropy,
            size_logistic_entropy,
        ) = discrete_action_entropy(
            entropy_params,
            max_launch,
            source_active,
            can_act,
            self.config.launch_mode,
            min_fleet_size=min_fleet_size,
            entropy_ship_quantiles=self.config.entropy_ship_quantiles,
        )
        per_player_entity_log_prob = launch_log_prob + target_log_prob + size_log_prob
        per_player_entity_entropy = launch_entropy + target_entropy + size_entropy

        return (
            actions,
            ModelActionLogProbs(
                launch=launch_log_prob.unsqueeze(-1),
                target=target_log_prob.unsqueeze(-1),
                event=size_log_prob.unsqueeze(-1),
                per_player_entity=per_player_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=launch_entropy.unsqueeze(-1),
                target=target_entropy.unsqueeze(-1),
                event=size_entropy.unsqueeze(-1),
                per_player_entity=per_player_entity_entropy,
                components={
                    "launch": launch_entropy,
                    "target": target_entropy,
                    "fleet_size_full": size_entropy,
                    "fleet_size_mixture": size_mixture_entropy,
                    "fleet_size_logistic": size_logistic_entropy,
                },
            ),
        )

    def sample_actions(
        self,
        actor_inputs: DiscreteActorInputs,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        *,
        min_fleet_size: int,
        deterministic: bool,
    ) -> DiscreteTargetActions:
        actions, _, _, _ = self._sample_actions(
            actor_inputs,
            can_act,
            max_launch,
            min_fleet_size=min_fleet_size,
            deterministic=deterministic,
        )
        return actions

    def log_prob(
        self,
        actor_inputs: DiscreteActorInputs,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        actions: DiscreteTargetActions,
        *,
        min_fleet_size: int,
    ) -> tuple[ModelActionLogProbs, ModelActionEntropies]:
        _require_discrete_actions_shape(
            actions,
            (
                actor_inputs.source.shape[0],
                OUTER_PLAYER_SLOTS,
                actor_inputs.source.shape[2],
                1,
            ),
        )
        action_launch = actions.launch
        action_target = actions.target
        action_ships = actions.ships
        selection = self._selection_params(actor_inputs, can_act)
        source_active = can_act.any(dim=-1) & (max_launch >= min_fleet_size)
        launch = action_launch[..., 0]
        target = action_target[..., 0]
        ships = action_ships[..., 0]
        _require_valid_discrete_action_slot(
            launch,
            target,
            ships,
            max_launch,
            source_active,
            can_act,
            min_fleet_size,
        )
        if self.config.launch_mode == "binary_after":
            _require_valid_discrete_action_target(
                target,
                source_active,
                can_act,
            )
        params = self._policy_params_for_selected_target(
            selection,
            actor_inputs.source,
            max_launch,
            target.clamp(0, selection.target_values.shape[2] - 1),
            min_fleet_size=min_fleet_size,
        )
        entropy_params = self._policy_params_for_entropy(
            selection,
            actor_inputs.source,
            max_launch,
            min_fleet_size=min_fleet_size,
        )
        launch_log_prob, target_log_prob, size_log_prob = discrete_action_log_probs(
            params,
            launch,
            target,
            ships,
            max_launch,
            source_active,
            self.config.launch_mode,
            min_fleet_size=min_fleet_size,
        )
        (
            launch_entropy,
            target_entropy,
            size_entropy,
            size_mixture_entropy,
            size_logistic_entropy,
        ) = discrete_action_entropy(
            entropy_params,
            max_launch,
            source_active,
            can_act,
            self.config.launch_mode,
            min_fleet_size=min_fleet_size,
            entropy_ship_quantiles=self.config.entropy_ship_quantiles,
        )
        per_player_entity_log_prob = launch_log_prob + target_log_prob + size_log_prob
        per_player_entity_entropy = launch_entropy + target_entropy + size_entropy
        return (
            ModelActionLogProbs(
                launch=launch_log_prob.unsqueeze(-1),
                target=target_log_prob.unsqueeze(-1),
                event=size_log_prob.unsqueeze(-1),
                per_player_entity=per_player_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=launch_entropy.unsqueeze(-1),
                target=target_entropy.unsqueeze(-1),
                event=size_entropy.unsqueeze(-1),
                per_player_entity=per_player_entity_entropy,
                components={
                    "launch": launch_entropy,
                    "target": target_entropy,
                    "fleet_size_full": size_entropy,
                    "fleet_size_mixture": size_mixture_entropy,
                    "fleet_size_logistic": size_logistic_entropy,
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
        action_entity_slots = source_input.shape[2]
        expected_input_shape = (
            source_input.shape[0],
            OUTER_PLAYER_SLOTS,
            action_entity_slots,
            self.head_dim,
        )
        if source_input.shape != expected_input_shape:
            raise ValueError(
                "discrete target source input must have shape "
                f"{expected_input_shape}, got {tuple(source_input.shape)}"
            )
        if target_input.shape != expected_input_shape:
            raise ValueError(
                "discrete target target input must have shape "
                f"{expected_input_shape}, got {tuple(target_input.shape)}"
            )
        if can_act.shape != (
            source_input.shape[0],
            OUTER_PLAYER_SLOTS,
            action_entity_slots,
            action_entity_slots,
        ):
            expected_shape = (
                source_input.shape[0],
                OUTER_PLAYER_SLOTS,
                action_entity_slots,
                action_entity_slots,
            )
            raise ValueError(
                "discrete target can_act must have shape "
                f"{expected_shape}, got {tuple(can_act.shape)}"
            )
        source_role = self.source_role.to(dtype=source_input.dtype)
        target_role = self.target_role.to(dtype=target_input.dtype)
        source_x = self.source_norm(source_input + source_role)
        if self.config.launch_mode == "target_token":
            if self.no_launch_target is None:
                raise RuntimeError("target_token mode requires a no-launch target")
            no_launch_target = self.no_launch_target.to(dtype=target_input.dtype)
            no_launch_target = no_launch_target.expand(
                target_input.shape[0],
                target_input.shape[1],
                1,
                self.head_dim,
            )
            target_input = torch.cat((target_input, no_launch_target), dim=2)
            no_launch_valid = can_act.any(dim=-1, keepdim=True)
            can_act = torch.cat((can_act, no_launch_valid), dim=-1)
        target_x = self.target_norm(target_input + target_role)
        q = self.q(source_x)
        k = self.k(target_x)
        v = self.v(target_x)
        target_logits = torch.einsum("bpsd,bptd->bpst", q, k)
        target_logits = target_logits / math.sqrt(self.head_dim)
        if actor_inputs.pairwise_bias is not None:
            pairwise_bias = actor_inputs.pairwise_bias
            expected_bias_shape = (
                (*target_logits.shape[:-1], action_entity_slots)
                if self.config.launch_mode == "target_token"
                else target_logits.shape
            )
            if pairwise_bias.shape != expected_bias_shape:
                raise ValueError(
                    "discrete target pairwise bias must have shape "
                    f"{tuple(expected_bias_shape)}, got {tuple(pairwise_bias.shape)}"
                )
            if self.config.launch_mode == "target_token":
                no_launch_bias = torch.zeros_like(pairwise_bias[..., :1])
                pairwise_bias = torch.cat((pairwise_bias, no_launch_bias), dim=-1)
            target_logits = target_logits + pairwise_bias.to(dtype=target_logits.dtype)
        continue_logits = None
        if self.continue_source_proj is not None and self.continue_head is not None:
            launch_hidden = self.continue_source_proj(source_input)
            continue_logits = self.continue_head(launch_hidden).squeeze(-1)
        target_logits = target_logits.masked_fill(
            ~can_act,
            torch.finfo(target_logits.dtype).min,
        )
        safe_target_logits = torch.where(
            can_act.any(dim=-1, keepdim=True),
            target_logits,
            torch.zeros_like(target_logits),
        )
        return DiscreteTargetSelectionParams(
            target_logits=safe_target_logits,
            target_values=v,
            continue_logits=continue_logits,
        )

    def _sample_actions(
        self,
        actor_inputs: DiscreteActorInputs,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        *,
        min_fleet_size: int,
        deterministic: bool,
    ) -> tuple[
        DiscreteTargetActions,
        DiscreteTargetSelectionParams,
        DiscreteTargetPolicyParams,
        torch.Tensor,
    ]:
        selection = self._selection_params(actor_inputs, can_act)
        source_active = can_act.any(dim=-1) & (max_launch >= min_fleet_size)
        if deterministic:
            selected_target = selection.target_logits.argmax(dim=-1)
        else:
            selected_target = Categorical(
                logits=selection.target_logits.float()
            ).sample()

        params = self._policy_params_for_selected_target(
            selection,
            actor_inputs.source,
            max_launch,
            selected_target,
            min_fleet_size=min_fleet_size,
        )
        if self.config.launch_mode == "target_token":
            no_launch_target = selection.target_logits.shape[-1] - 1
            launch = (selected_target != no_launch_target) & source_active
            target = torch.where(
                launch,
                selected_target,
                torch.zeros_like(selected_target),
            )
        else:
            continue_logits = _require_continue_logits(params.continue_logits)
            launch = sample_launch(
                continue_logits,
                source_active,
                deterministic=deterministic,
            )
            if self.config.launch_mode == "binary_after":
                target = selected_target
            else:
                target = torch.where(
                    launch,
                    selected_target,
                    torch.zeros_like(selected_target),
                )
        ships = sample_discretized_logistic_mixture(
            params.size_mix_logits,
            params.size_mu,
            params.size_scale,
            max_launch,
            min_fleet_size=min_fleet_size,
            deterministic=deterministic,
            deterministic_mask=launch,
        )
        ships = torch.where(launch, ships, torch.zeros_like(ships))
        return (
            DiscreteTargetActions(
                launch=launch.unsqueeze(-1),
                target=target.unsqueeze(-1),
                ships=ships.unsqueeze(-1),
            ),
            selection,
            params,
            source_active,
        )

    def _policy_params_for_selected_target(
        self,
        selection: DiscreteTargetSelectionParams,
        source_input: torch.Tensor,
        max_launch: torch.Tensor,
        target_index: torch.Tensor,
        *,
        min_fleet_size: int,
    ) -> DiscreteTargetPolicyParams:
        selected_target_values = gather_selected_target_values(
            selection.target_values,
            target_index,
        )
        size_params = self._size_params_from_target_values(
            source_input,
            max_launch,
            selected_target_values,
            min_fleet_size=min_fleet_size,
        )
        return policy_params_for_selected_target(selection, size_params)

    def _policy_params_for_entropy(
        self,
        selection: DiscreteTargetSelectionParams,
        source_input: torch.Tensor,
        max_launch: torch.Tensor,
        *,
        min_fleet_size: int,
    ) -> DiscreteTargetPolicyParams:
        target_index = selection.target_logits.argmax(dim=-1)
        return self._policy_params_for_selected_target(
            selection,
            source_input,
            max_launch,
            target_index,
            min_fleet_size=min_fleet_size,
        )

    def _all_size_params(
        self,
        selection: DiscreteTargetSelectionParams,
        source_input: torch.Tensor,
        max_launch: torch.Tensor,
        *,
        min_fleet_size: int,
    ) -> DiscreteTargetSizeParams:
        batch, players, source_slots, _ = source_input.shape
        target_slots = selection.target_values.shape[2]
        target_values = selection.target_values.unsqueeze(2).expand(
            batch,
            players,
            source_slots,
            target_slots,
            self.head_dim,
        )
        return self._size_params_from_target_values(
            source_input.unsqueeze(3),
            max_launch,
            target_values,
            min_fleet_size=min_fleet_size,
        )

    def _size_params_from_target_values(
        self,
        source_input: torch.Tensor,
        max_launch: torch.Tensor,
        target_values: torch.Tensor,
        *,
        min_fleet_size: int,
    ) -> DiscreteTargetSizeParams:
        source_hidden = self._target_conditioned_hidden(source_input, target_values)

        residual_budget = max_launch.to(dtype=torch.float32).view(
            *max_launch.shape,
            *((1,) * (source_hidden.ndim - max_launch.ndim)),
        )
        support_lo = torch.full_like(residual_budget, float(min_fleet_size))
        support_width = (residual_budget - support_lo + 1.0).clamp_min(1.0)
        rho = torch.sigmoid(self.mean_head(source_hidden).float())
        mu = support_lo + rho * (residual_budget - support_lo).clamp_min(0.0)
        scale_upper = torch.maximum(
            torch.full_like(residual_budget, self.config.scale_max_abs_floor),
            support_width * self.config.scale_max_frac,
        )
        scale = log_interpolate(
            self.config.scale_min,
            scale_upper,
            torch.sigmoid(self.scale_head(source_hidden).float()),
        )
        continue_logits = None
        if self.config.launch_mode == "binary_after":
            continue_head = _require_continue_head(self.continue_head)
            continue_logits = continue_head(source_hidden).squeeze(-1)
        return DiscreteTargetSizeParams(
            size_mix_logits=self.mix_head(source_hidden),
            size_mu=mu,
            size_scale=scale,
            continue_logits=continue_logits,
        )

    def _target_conditioned_hidden(
        self,
        source_input: torch.Tensor,
        target_values: torch.Tensor,
    ) -> torch.Tensor:
        selected_v = self.out(target_values)
        enriched = source_input + selected_v
        enriched = enriched + self.mlp(self.norm2(enriched))
        return self.size_pair_proj(enriched)


def _init_token_parameter(parameter: nn.Parameter) -> None:
    nn.init.normal_(parameter, mean=0.0, std=parameter.shape[-1] ** -0.5)


def policy_params_for_selected_target(
    selection: DiscreteTargetSelectionParams,
    size_params: DiscreteTargetSizeParams,
) -> DiscreteTargetPolicyParams:
    return DiscreteTargetPolicyParams(
        target_logits=selection.target_logits,
        size_mix_logits=size_params.size_mix_logits,
        size_mu=size_params.size_mu,
        size_scale=size_params.size_scale,
        continue_logits=(
            size_params.continue_logits
            if size_params.continue_logits is not None
            else selection.continue_logits
        ),
    )


def gather_selected_target_values(
    target_values: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    gather_index = target_index[..., None].expand(
        *target_index.shape,
        target_values.shape[-1],
    )
    return target_values.gather(dim=2, index=gather_index)


def discrete_action_log_probs(
    params: DiscreteTargetPolicyParams,
    launch: torch.Tensor,
    target: torch.Tensor,
    ships: torch.Tensor,
    residual_budget: torch.Tensor,
    source_active: torch.Tensor,
    launch_mode: str = "binary",
    *,
    min_fleet_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if launch_mode == "target_token":
        launch_log_prob = torch.zeros_like(params.target_logits[..., 0].float())
        no_launch_target = params.target_logits.shape[-1] - 1
        selected_target = torch.where(
            launch,
            target.clamp(0, no_launch_target - 1),
            torch.full_like(target, no_launch_target),
        )
    else:
        continue_logits = _require_continue_logits(params.continue_logits)
        launch_log_prob = -F.binary_cross_entropy_with_logits(
            continue_logits.float(),
            launch.to(dtype=torch.float32),
            reduction="none",
        )
        launch_log_prob = torch.where(
            source_active,
            launch_log_prob,
            torch.zeros_like(launch_log_prob),
        )
        selected_target = target.clamp(0, params.target_logits.shape[-1] - 1)
    target_log_all = F.log_softmax(params.target_logits.float(), dim=-1)
    target_log_prob = target_log_all.gather(
        -1,
        selected_target.unsqueeze(-1),
    ).squeeze(-1)
    size_log_prob = discretized_logistic_mixture_log_prob(
        ships,
        residual_budget,
        params.size_mix_logits,
        params.size_mu,
        params.size_scale,
        min_fleet_size=min_fleet_size,
    )
    target_mask = (
        source_active
        if launch_mode in {"binary_after", "target_token"}
        else launch & source_active
    )
    event_mask = launch & source_active
    return (
        launch_log_prob,
        torch.where(target_mask, target_log_prob, torch.zeros_like(target_log_prob)),
        torch.where(event_mask, size_log_prob, torch.zeros_like(size_log_prob)),
    )


def discrete_action_entropy(
    params: DiscreteTargetPolicyParams,
    residual_budget: torch.Tensor,
    source_active: torch.Tensor,
    can_act: torch.Tensor,
    launch_mode: str = "binary",
    *,
    min_fleet_size: int,
    entropy_ship_quantiles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if launch_mode == "target_token":
        launch_entropy = torch.zeros_like(params.target_logits[..., 0].float())
        no_launch_valid = can_act.any(dim=-1, keepdim=True)
        can_act = torch.cat((can_act, no_launch_valid), dim=-1)
    else:
        continue_logits = _require_continue_logits(params.continue_logits)
        launch_entropy = binary_entropy_from_logits(continue_logits.float())
    target_prob = torch.softmax(params.target_logits.float(), dim=-1)
    target_log_prob = F.log_softmax(params.target_logits.float(), dim=-1)
    target_entropy = (
        -(target_prob * target_log_prob).masked_fill(~can_act, 0.0).sum(dim=-1)
    )

    size_entropy = truncated_logistic_mixture_entropy(
        params.size_mix_logits,
        params.size_mu,
        params.size_scale,
        residual_budget,
        min_fleet_size=min_fleet_size,
        entropy_ship_quantiles=entropy_ship_quantiles,
    )
    size_mix_log_prob = F.log_softmax(params.size_mix_logits.float(), dim=-1)
    size_mix_prob = size_mix_log_prob.exp()
    size_mixture_entropy = -(size_mix_prob * size_mix_log_prob).sum(dim=-1)
    size_logistic_entropy = (
        size_mix_prob * (params.size_scale.float().log() + 2.0)
    ).sum(dim=-1)
    return (
        torch.where(source_active, launch_entropy, torch.zeros_like(launch_entropy)),
        torch.where(
            source_active,
            target_entropy,
            torch.zeros_like(target_entropy),
        ),
        torch.where(
            source_active,
            size_entropy,
            torch.zeros_like(size_entropy),
        ),
        torch.where(
            source_active,
            size_mixture_entropy,
            torch.zeros_like(size_mixture_entropy),
        ),
        torch.where(
            source_active,
            size_logistic_entropy,
            torch.zeros_like(size_logistic_entropy),
        ),
    )


def _require_continue_logits(continue_logits: torch.Tensor | None) -> torch.Tensor:
    if continue_logits is None:
        raise RuntimeError("binary launch modes require continue logits")
    return continue_logits


def _require_continue_head(
    continue_head: OutputProjectionMLP | None,
) -> OutputProjectionMLP:
    if continue_head is None:
        raise RuntimeError("binary_after launch mode requires a continue head")
    return continue_head


def _require_discrete_actions_shape(
    actions: DiscreteTargetActions,
    expected_shape: tuple[int, int, int, int],
) -> None:
    for name, tensor in (
        ("launch", actions.launch),
        ("target", actions.target),
        ("ships", actions.ships),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"actions.{name} must have shape {expected_shape}, got {tensor.shape}"
            )
    if expected_shape[-1] != 1:
        raise ValueError("discrete target actions require one launch slot")
    if actions.launch.dtype != torch.bool:
        raise ValueError(
            f"actions.launch must have dtype torch.bool, got {actions.launch.dtype}"
        )
    if actions.target.dtype != torch.int64:
        raise ValueError(
            f"actions.target must have dtype torch.int64, got {actions.target.dtype}"
        )
    if actions.ships.dtype != torch.int64:
        raise ValueError(
            f"actions.ships must have dtype torch.int64, got {actions.ships.dtype}"
        )


def _require_valid_discrete_action_slot(
    launch: torch.Tensor,
    target: torch.Tensor,
    ships: torch.Tensor,
    remaining: torch.Tensor,
    active: torch.Tensor,
    can_act: torch.Tensor,
    min_fleet_size: int,
) -> None:
    if (launch & ~active).any().item():
        raise ValueError(
            "actions.launch cannot be true after a lane has stopped or is inactive"
        )
    invalid_ships = launch & (ships.lt(min_fleet_size) | ships.gt(remaining))
    if invalid_ships.any().item():
        raise ValueError(
            f"actions.ships must be in {min_fleet_size}..remaining for launched slots"
        )
    target_slots = can_act.shape[-1]
    target_in_range = target.ge(0) & target.lt(target_slots)
    safe_target = target.clamp(0, target_slots - 1)
    target_valid = can_act.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    if (launch & (~target_in_range | ~target_valid)).any().item():
        raise ValueError("actions.target must select a valid target for launched slots")


def _require_valid_discrete_action_target(
    target: torch.Tensor,
    active: torch.Tensor,
    can_act: torch.Tensor,
) -> None:
    target_slots = can_act.shape[-1]
    target_in_range = target.ge(0) & target.lt(target_slots)
    safe_target = target.clamp(0, target_slots - 1)
    target_valid = can_act.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    if (active & (~target_in_range | ~target_valid)).any().item():
        raise ValueError("actions.target must select a valid target for active slots")
