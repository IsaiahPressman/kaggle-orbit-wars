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
    binary_entropy_from_logits,
    sample_launch,
)
from owl.model.actor.config import ActorDiscreteTargetsConfig
from owl.model.base import ModelActionEntropies, ModelActionLogProbs, ModelActions
from owl.rl import ACTION_ENTITY_SLOTS, OUTER_PLAYER_SLOTS


@dataclass(frozen=True)
class DiscreteTargetSelectionParams:
    continue_logits: torch.Tensor
    target_logits: torch.Tensor
    target_values: torch.Tensor


@dataclass(frozen=True)
class DiscreteTargetSizeParams:
    size_mix_logits: torch.Tensor
    size_mu: torch.Tensor
    size_scale: torch.Tensor


@dataclass(frozen=True)
class DiscreteTargetPolicyParams:
    continue_logits: torch.Tensor
    target_logits: torch.Tensor
    size_mix_logits: torch.Tensor
    size_mu: torch.Tensor
    size_scale: torch.Tensor


class DiscreteTargetsActor(nn.Module):
    def __init__(
        self,
        config: ActorDiscreteTargetsConfig,
        *,
        transformer_config: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_heads = transformer_config.n_heads
        self.head_dim = transformer_config.embed_dim // transformer_config.n_heads
        mixtures = config.n_action_mixtures

        self.norm1 = nn.LayerNorm(transformer_config.embed_dim)
        self.q = nn.Linear(transformer_config.embed_dim, transformer_config.embed_dim)
        self.k = nn.Linear(transformer_config.embed_dim, transformer_config.embed_dim)
        self.v = nn.Linear(transformer_config.embed_dim, transformer_config.embed_dim)
        self.out = nn.Linear(transformer_config.embed_dim, transformer_config.embed_dim)
        self.norm2 = nn.LayerNorm(transformer_config.embed_dim)
        self.mlp = FeedForward(transformer_config)
        self.source_proj = nn.Linear(
            transformer_config.embed_dim,
            transformer_config.embed_dim,
        )
        self.continue_head = nn.Linear(transformer_config.embed_dim, 1)
        self.mix_head = nn.Linear(transformer_config.embed_dim, mixtures)
        self.mean_head = nn.Linear(transformer_config.embed_dim, mixtures)
        self.log_scale_head = nn.Linear(transformer_config.embed_dim, mixtures)

    def get_input_layers(self) -> tuple[nn.Module, ...]:
        return (self.source_proj,)

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        return (
            self.continue_head,
            self.mix_head,
            self.mean_head,
            self.log_scale_head,
        )

    def forward(
        self,
        slot_input: torch.Tensor,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        *,
        min_fleet_size: int,
        deterministic: bool,
    ) -> tuple[ModelActions, ModelActionLogProbs, ModelActionEntropies]:
        selection = self._selection_params(slot_input, can_act)
        source_active = can_act.any(dim=-1) & (max_launch >= min_fleet_size)
        launch = sample_launch(
            selection.continue_logits,
            source_active,
            deterministic=deterministic,
        )
        if deterministic:
            target = selection.target_logits.argmax(dim=-1)
        else:
            target = Categorical(logits=selection.target_logits.float()).sample()
        target = torch.where(launch, target, torch.zeros_like(target))
        params = self._size_params(selection, slot_input, max_launch, target)
        ships = sample_discretized_logistic_mixture(
            params.size_mix_logits,
            params.size_mu,
            params.size_scale,
            max_launch,
            min_fleet_size=min_fleet_size,
            deterministic=deterministic,
        )
        ships = torch.where(launch, ships, torch.zeros_like(ships))

        launch_log_prob, target_log_prob, size_log_prob = discrete_action_log_probs(
            params,
            launch,
            target,
            ships,
            max_launch,
            source_active,
            min_fleet_size=min_fleet_size,
        )
        launch_entropy, target_entropy, size_entropy = discrete_action_entropy(
            params,
            self._all_size_params(selection, slot_input, max_launch),
            max_launch,
            source_active,
            can_act,
            min_fleet_size=min_fleet_size,
            max_ship_support=self.config.entropy_ship_support_cap,
        )
        per_player_entity_log_prob = launch_log_prob + target_log_prob + size_log_prob
        per_player_entity_entropy = launch_entropy + target_entropy + size_entropy

        return (
            ModelActions(
                launch=launch.unsqueeze(-1),
                target=target.unsqueeze(-1),
                ships=ships.unsqueeze(-1),
            ),
            ModelActionLogProbs(
                launch=launch_log_prob.unsqueeze(-1),
                target=target_log_prob.unsqueeze(-1),
                angle_and_size=size_log_prob.unsqueeze(-1),
                per_player_entity=per_player_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=launch_entropy.unsqueeze(-1),
                target=target_entropy.unsqueeze(-1),
                angle_and_size=size_entropy.unsqueeze(-1),
                per_player_entity=per_player_entity_entropy,
            ),
        )

    def log_prob(
        self,
        slot_input: torch.Tensor,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        actions: ModelActions,
        *,
        min_fleet_size: int,
    ) -> tuple[ModelActionLogProbs, ModelActionEntropies]:
        _require_discrete_actions_shape(
            actions,
            (
                slot_input.shape[0],
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                1,
            ),
        )
        if actions.target is None:
            raise ValueError("discrete target actions require actions.target")
        selection = self._selection_params(slot_input, can_act)
        source_active = can_act.any(dim=-1) & (max_launch >= min_fleet_size)
        launch = actions.launch[..., 0]
        target = actions.target[..., 0]
        ships = actions.ships[..., 0]
        _require_valid_discrete_action_slot(
            launch,
            target,
            ships,
            max_launch,
            source_active,
            can_act,
            min_fleet_size,
        )
        params = self._size_params(
            selection,
            slot_input,
            max_launch,
            target.clamp(0, ACTION_ENTITY_SLOTS - 1),
        )
        launch_log_prob, target_log_prob, size_log_prob = discrete_action_log_probs(
            params,
            launch,
            target,
            ships,
            max_launch,
            source_active,
            min_fleet_size=min_fleet_size,
        )
        launch_entropy, target_entropy, size_entropy = discrete_action_entropy(
            params,
            self._all_size_params(selection, slot_input, max_launch),
            max_launch,
            source_active,
            can_act,
            min_fleet_size=min_fleet_size,
            max_ship_support=self.config.entropy_ship_support_cap,
        )
        per_player_entity_log_prob = launch_log_prob + target_log_prob + size_log_prob
        per_player_entity_entropy = launch_entropy + target_entropy + size_entropy
        return (
            ModelActionLogProbs(
                launch=launch_log_prob.unsqueeze(-1),
                target=target_log_prob.unsqueeze(-1),
                angle_and_size=size_log_prob.unsqueeze(-1),
                per_player_entity=per_player_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=launch_entropy.unsqueeze(-1),
                target=target_entropy.unsqueeze(-1),
                angle_and_size=size_entropy.unsqueeze(-1),
                per_player_entity=per_player_entity_entropy,
            ),
        )

    def _selection_params(
        self,
        slot_input: torch.Tensor,
        can_act: torch.Tensor,
    ) -> DiscreteTargetSelectionParams:
        if can_act.shape != (
            slot_input.shape[0],
            OUTER_PLAYER_SLOTS,
            ACTION_ENTITY_SLOTS,
            ACTION_ENTITY_SLOTS,
        ):
            expected_shape = (
                slot_input.shape[0],
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                ACTION_ENTITY_SLOTS,
            )
            raise ValueError(
                "discrete target can_act must have shape "
                f"{expected_shape}, got {tuple(can_act.shape)}"
            )
        x = self.norm1(slot_input)
        batch, players, slots, _ = x.shape
        q = self.q(x).view(batch, players, slots, self.n_heads, self.head_dim)
        k = self.k(x).view(batch, players, slots, self.n_heads, self.head_dim)
        v = self.v(x).view(batch, players, slots, self.n_heads, self.head_dim)
        target_logits = torch.einsum("bpshd,bpthd->bpsth", q, k).mean(dim=-1)
        target_logits = target_logits / math.sqrt(self.head_dim)
        target_logits = target_logits.masked_fill(~can_act, torch.finfo(x.dtype).min)
        safe_target_logits = torch.where(
            can_act.any(dim=-1, keepdim=True),
            target_logits,
            torch.zeros_like(target_logits),
        )
        launch_hidden = self.source_proj(slot_input)
        return DiscreteTargetSelectionParams(
            continue_logits=self.continue_head(launch_hidden).squeeze(-1),
            target_logits=safe_target_logits,
            target_values=v,
        )

    def _size_params(
        self,
        selection: DiscreteTargetSelectionParams,
        slot_input: torch.Tensor,
        max_launch: torch.Tensor,
        target_index: torch.Tensor,
    ) -> DiscreteTargetPolicyParams:
        selected_v = gather_target_values(selection.target_values, target_index)
        size_params = self._size_params_from_target_values(
            slot_input,
            max_launch,
            selected_v,
        )
        return DiscreteTargetPolicyParams(
            continue_logits=selection.continue_logits,
            target_logits=selection.target_logits,
            size_mix_logits=size_params.size_mix_logits,
            size_mu=size_params.size_mu,
            size_scale=size_params.size_scale,
        )

    def _all_size_params(
        self,
        selection: DiscreteTargetSelectionParams,
        slot_input: torch.Tensor,
        max_launch: torch.Tensor,
    ) -> DiscreteTargetSizeParams:
        batch, players, source_slots, _ = slot_input.shape
        target_slots = selection.target_values.shape[2]
        target_values = selection.target_values.unsqueeze(2).expand(
            batch,
            players,
            source_slots,
            target_slots,
            self.n_heads,
            self.head_dim,
        )
        return self._size_params_from_target_values(
            slot_input.unsqueeze(3),
            max_launch,
            target_values,
        )

    def _size_params_from_target_values(
        self,
        slot_input: torch.Tensor,
        max_launch: torch.Tensor,
        target_values: torch.Tensor,
    ) -> DiscreteTargetSizeParams:
        selected_v = self.out(target_values.flatten(start_dim=-2))
        enriched = slot_input + selected_v
        enriched = enriched + self.mlp(self.norm2(enriched))
        source_hidden = self.source_proj(enriched)

        residual_budget = (
            max_launch.clamp_min(1)
            .to(dtype=source_hidden.dtype)
            .view(
                *max_launch.shape,
                *((1,) * (source_hidden.ndim - max_launch.ndim)),
            )
        )
        rho = torch.sigmoid(self.mean_head(source_hidden))
        mu = 1.0 + rho * (residual_budget - 1.0)
        raw_log_scale = self.log_scale_head(source_hidden).clamp(
            self.config.min_log_scale,
            self.config.max_log_scale,
        )
        scale = self.config.scale_min + residual_budget * raw_log_scale.exp()
        return DiscreteTargetSizeParams(
            size_mix_logits=self.mix_head(source_hidden),
            size_mu=mu,
            size_scale=scale,
        )


def gather_target_values(
    values: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    gather_index = target_index[..., None, None].expand(
        *target_index.shape,
        values.shape[-2],
        values.shape[-1],
    )
    return values.gather(dim=2, index=gather_index)


def logsubexp(log_x: torch.Tensor, log_y: torch.Tensor) -> torch.Tensor:
    return log_x + torch.log1p(-(log_y - log_x).exp().clamp_max(1.0 - 1e-12))


def logistic_cdf_diff_logprob(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    log_cdf_hi = F.logsigmoid(hi)
    log_cdf_lo = F.logsigmoid(lo)
    left = logsubexp(log_cdf_hi, log_cdf_lo)

    log_sf_lo = F.logsigmoid(-lo)
    log_sf_hi = F.logsigmoid(-hi)
    right = logsubexp(log_sf_lo, log_sf_hi)

    return torch.where((lo + hi) > 0.0, right, left)


def discretized_logistic_mixture_log_prob(
    ships: torch.Tensor,
    residual_budget: torch.Tensor,
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    *,
    min_fleet_size: int,
) -> torch.Tensor:
    dtype = mu.dtype
    n = ships.to(dtype).unsqueeze(-1)
    safe_residual_budget = residual_budget.clamp_min(min_fleet_size)
    residual = safe_residual_budget.to(dtype).unsqueeze(-1)

    valid = (
        (ships >= min_fleet_size)
        & (ships <= residual_budget)
        & (residual_budget >= min_fleet_size)
    )

    lo = (n - 0.5 - mu) / scale
    hi = (n + 0.5 - mu) / scale

    support_lo = (float(min_fleet_size) - 0.5 - mu) / scale
    support_hi = (residual + 0.5 - mu) / scale

    log_bin_mass = logistic_cdf_diff_logprob(lo, hi)
    log_support_mass = logistic_cdf_diff_logprob(support_lo, support_hi)
    log_w = F.log_softmax(mix_logits.float(), dim=-1)
    log_comp = log_w + log_bin_mass.float() - log_support_mass.float()
    log_comp = torch.where(
        valid.unsqueeze(-1),
        log_comp,
        torch.full_like(log_comp, -torch.inf),
    )
    logp = torch.logsumexp(log_comp, dim=-1)
    return torch.where(valid, logp, torch.full_like(logp, -torch.inf))


def sample_discretized_logistic_mixture(
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    residual_budget: torch.Tensor,
    *,
    min_fleet_size: int,
    deterministic: bool,
) -> torch.Tensor:
    support = ship_support(
        residual_budget,
        min_fleet_size=min_fleet_size,
        max_ship_support=int(residual_budget.max().clamp_min(min_fleet_size).item()),
    )
    log_probs = discretized_logistic_mixture_log_prob(
        support,
        residual_budget.unsqueeze(-1),
        mix_logits.unsqueeze(-2),
        mu.unsqueeze(-2),
        scale.unsqueeze(-2),
        min_fleet_size=min_fleet_size,
    )
    valid = support <= residual_budget.unsqueeze(-1)
    log_probs = log_probs.masked_fill(~valid, torch.finfo(log_probs.dtype).min)
    if deterministic:
        support_index = log_probs.argmax(dim=-1)
    else:
        support_index = Categorical(logits=log_probs).sample()
    support = support.expand_as(log_probs)
    return support.gather(dim=-1, index=support_index.unsqueeze(-1)).squeeze(-1)


def ship_support(
    residual_budget: torch.Tensor,
    *,
    min_fleet_size: int,
    max_ship_support: int,
) -> torch.Tensor:
    max_count = max(max_ship_support, 1)
    offsets = torch.arange(max_count, device=residual_budget.device)
    return min_fleet_size + offsets.view(*((1,) * residual_budget.ndim), max_count)


def discrete_action_log_probs(
    params: DiscreteTargetPolicyParams,
    launch: torch.Tensor,
    target: torch.Tensor,
    ships: torch.Tensor,
    residual_budget: torch.Tensor,
    source_active: torch.Tensor,
    *,
    min_fleet_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    launch_log_prob = -F.binary_cross_entropy_with_logits(
        params.continue_logits.float(),
        launch.to(dtype=torch.float32),
        reduction="none",
    )
    launch_log_prob = torch.where(
        source_active,
        launch_log_prob,
        torch.zeros_like(launch_log_prob),
    )
    safe_target = target.clamp(0, ACTION_ENTITY_SLOTS - 1)
    target_log_all = F.log_softmax(params.target_logits.float(), dim=-1)
    target_log_prob = target_log_all.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    size_log_prob = discretized_logistic_mixture_log_prob(
        ships,
        residual_budget,
        params.size_mix_logits,
        params.size_mu,
        params.size_scale,
        min_fleet_size=min_fleet_size,
    )
    event_mask = launch & source_active
    return (
        launch_log_prob,
        torch.where(event_mask, target_log_prob, torch.zeros_like(target_log_prob)),
        torch.where(event_mask, size_log_prob, torch.zeros_like(size_log_prob)),
    )


def discrete_action_entropy(
    params: DiscreteTargetPolicyParams,
    all_size_params: DiscreteTargetSizeParams,
    residual_budget: torch.Tensor,
    source_active: torch.Tensor,
    can_act: torch.Tensor,
    *,
    min_fleet_size: int,
    max_ship_support: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    launch_entropy = binary_entropy_from_logits(params.continue_logits.float())
    target_prob = torch.softmax(params.target_logits.float(), dim=-1)
    target_log_prob = F.log_softmax(params.target_logits.float(), dim=-1)
    target_entropy = (
        -(target_prob * target_log_prob).masked_fill(~can_act, 0.0).sum(dim=-1)
    )

    support = ship_support(
        residual_budget,
        min_fleet_size=min_fleet_size,
        max_ship_support=max_ship_support,
    )
    log_probs = discretized_logistic_mixture_log_prob(
        support.unsqueeze(-2),
        residual_budget.unsqueeze(-1).unsqueeze(-1),
        all_size_params.size_mix_logits.unsqueeze(-2),
        all_size_params.size_mu.unsqueeze(-2),
        all_size_params.size_scale.unsqueeze(-2),
        min_fleet_size=min_fleet_size,
    )
    valid = support.unsqueeze(-2) <= residual_budget.unsqueeze(-1).unsqueeze(-1)
    probs = torch.where(valid, log_probs.exp(), torch.zeros_like(log_probs))
    size_entropy_by_target = -(
        probs * torch.where(valid, log_probs, torch.zeros_like(log_probs))
    ).sum(dim=-1)
    size_entropy = (
        (target_prob * size_entropy_by_target)
        .masked_fill(
            ~can_act,
            0.0,
        )
        .sum(dim=-1)
    )
    launch_probability = torch.sigmoid(params.continue_logits.float())
    return (
        torch.where(source_active, launch_entropy, torch.zeros_like(launch_entropy)),
        torch.where(
            source_active,
            launch_probability * target_entropy,
            torch.zeros_like(target_entropy),
        ),
        torch.where(
            source_active,
            launch_probability * size_entropy,
            torch.zeros_like(size_entropy),
        ),
    )


def _require_discrete_actions_shape(
    actions: ModelActions,
    expected_shape: tuple[int, int, int, int],
) -> None:
    for name, tensor in (
        ("launch", actions.launch),
        ("target", actions.target),
        ("ships", actions.ships),
    ):
        if tensor is None:
            raise ValueError(f"actions.{name} is required for discrete target actions")
        if tensor.shape != expected_shape:
            raise ValueError(
                f"actions.{name} must have shape {expected_shape}, got {tensor.shape}"
            )
    if actions.angle is not None:
        raise ValueError("discrete target actions must not include actions.angle")
    if expected_shape[-1] != 1:
        raise ValueError("discrete target actions require one launch slot")
    if actions.launch.dtype != torch.bool:
        raise ValueError(
            f"actions.launch must have dtype torch.bool, got {actions.launch.dtype}"
        )
    if actions.target is None:
        raise ValueError("discrete target actions require actions.target")
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
    target_in_range = target.ge(0) & target.lt(ACTION_ENTITY_SLOTS)
    safe_target = target.clamp(0, ACTION_ENTITY_SLOTS - 1)
    target_valid = can_act.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    if (launch & (~target_in_range | ~target_valid)).any().item():
        raise ValueError("actions.target must select a valid target for launched slots")
