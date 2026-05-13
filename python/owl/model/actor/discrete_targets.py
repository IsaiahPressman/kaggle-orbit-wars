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
from owl.model.base import (
    InputLayer,
    ModelActionEntropies,
    ModelActionLogProbs,
)
from owl.rl import ACTION_ENTITY_SLOTS, OUTER_PLAYER_SLOTS, DiscreteTargetActions


@dataclass(frozen=True)
class DiscreteActorInputs:
    source: torch.Tensor
    target: torch.Tensor
    pairwise_bias: torch.Tensor | None = None


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
        self.n_heads = 1
        self.head_dim = transformer_config.embed_dim
        mixtures = config.n_action_mixtures

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
        self.continue_source_proj = nn.Linear(
            transformer_config.embed_dim,
            transformer_config.embed_dim,
        )
        self.size_pair_proj = nn.Linear(
            transformer_config.embed_dim,
            transformer_config.embed_dim,
        )
        self.continue_head = OutputProjectionMLP(transformer_config, 1)
        self.mix_head = OutputProjectionMLP(transformer_config, mixtures)
        self.mean_head = OutputProjectionMLP(transformer_config, mixtures)
        self.scale_head = OutputProjectionMLP(transformer_config, mixtures)

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        return (self.source_role, self.target_role)

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        return (
            self.continue_head.out,
            self.mix_head.out,
            self.mean_head.out,
            self.scale_head.out,
        )

    def forward(
        self,
        actor_inputs: DiscreteActorInputs,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        *,
        min_fleet_size: int,
        deterministic: bool,
    ) -> tuple[DiscreteTargetActions, ModelActionLogProbs, ModelActionEntropies]:
        selection = self._selection_params(actor_inputs, can_act)
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
        params = self._policy_params_for_selected_target(
            selection,
            actor_inputs.source,
            max_launch,
            target,
            min_fleet_size=min_fleet_size,
        )
        ships = sample_discretized_logistic_mixture(
            params.size_mix_logits,
            params.size_mu,
            params.size_scale,
            max_launch,
            min_fleet_size=min_fleet_size,
            deterministic=deterministic,
        )
        ships = torch.where(launch, ships, torch.zeros_like(ships))

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
            min_fleet_size=min_fleet_size,
            entropy_ship_quantiles=self.config.entropy_ship_quantiles,
        )
        per_player_entity_log_prob = launch_log_prob + target_log_prob + size_log_prob
        per_player_entity_entropy = launch_entropy + target_entropy + size_entropy

        return (
            DiscreteTargetActions(
                launch=launch.unsqueeze(-1),
                target=target.unsqueeze(-1),
                ships=ships.unsqueeze(-1),
            ),
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
                ACTION_ENTITY_SLOTS,
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
        params = self._policy_params_for_selected_target(
            selection,
            actor_inputs.source,
            max_launch,
            target.clamp(0, ACTION_ENTITY_SLOTS - 1),
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
        expected_input_shape = (
            source_input.shape[0],
            OUTER_PLAYER_SLOTS,
            ACTION_ENTITY_SLOTS,
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
            ACTION_ENTITY_SLOTS,
            ACTION_ENTITY_SLOTS,
        ):
            expected_shape = (
                source_input.shape[0],
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                ACTION_ENTITY_SLOTS,
            )
            raise ValueError(
                "discrete target can_act must have shape "
                f"{expected_shape}, got {tuple(can_act.shape)}"
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
        if actor_inputs.pairwise_bias is not None:
            pairwise_bias = actor_inputs.pairwise_bias
            if pairwise_bias.shape != target_logits.shape:
                raise ValueError(
                    "discrete target pairwise bias must have shape "
                    f"{tuple(target_logits.shape)}, got {tuple(pairwise_bias.shape)}"
                )
            target_logits = target_logits + pairwise_bias.to(dtype=target_logits.dtype)
        target_logits = target_logits.masked_fill(
            ~can_act,
            torch.finfo(target_logits.dtype).min,
        )
        safe_target_logits = torch.where(
            can_act.any(dim=-1, keepdim=True),
            target_logits,
            torch.zeros_like(target_logits),
        )
        launch_hidden = self.continue_source_proj(source_input)
        return DiscreteTargetSelectionParams(
            continue_logits=self.continue_head(launch_hidden).squeeze(-1),
            target_logits=safe_target_logits,
            target_values=v,
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
        selected_v = self.out(target_values)
        enriched = source_input + selected_v
        enriched = enriched + self.mlp(self.norm2(enriched))
        source_hidden = self.size_pair_proj(enriched)

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
        return DiscreteTargetSizeParams(
            size_mix_logits=self.mix_head(source_hidden),
            size_mu=mu,
            size_scale=scale,
        )


def _init_token_parameter(parameter: nn.Parameter) -> None:
    nn.init.normal_(parameter, mean=0.0, std=parameter.shape[-1] ** -0.5)


def policy_params_for_selected_target(
    selection: DiscreteTargetSelectionParams,
    size_params: DiscreteTargetSizeParams,
) -> DiscreteTargetPolicyParams:
    return DiscreteTargetPolicyParams(
        continue_logits=selection.continue_logits,
        target_logits=selection.target_logits,
        size_mix_logits=size_params.size_mix_logits,
        size_mu=size_params.size_mu,
        size_scale=size_params.size_scale,
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


def log_interpolate(
    low: float,
    high: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    log_low = math.log(low)
    high = high.float()
    weight = weight.float()
    return (log_low + weight * (high.log() - log_low)).exp()


def logsubexp(log_x: torch.Tensor, log_y: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(log_x.dtype).eps
    ratio = (log_y - log_x).exp().clamp_max(1.0 - eps)
    return log_x + torch.log1p(-ratio)


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
    mix_logits = mix_logits.float()
    mu = mu.float()
    scale = scale.float()
    n = ships.to(torch.float32).unsqueeze(-1)
    safe_residual_budget = residual_budget.clamp_min(min_fleet_size)
    residual = safe_residual_budget.to(torch.float32).unsqueeze(-1)

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
    log_w = F.log_softmax(mix_logits, dim=-1)
    log_comp = log_w + log_bin_mass - log_support_mass
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
    if not deterministic:
        mix_index = Categorical(logits=mix_logits.float()).sample()
        gather_index = mix_index.unsqueeze(-1)
        selected_mu = mu.gather(-1, gather_index).squeeze(-1).float()
        selected_scale = scale.gather(-1, gather_index).squeeze(-1).float()

        residual = residual_budget.float()
        lo_count = float(min_fleet_size) - 0.5
        hi_count = residual + 0.5
        cdf_lo = torch.sigmoid((lo_count - selected_mu) / selected_scale)
        cdf_hi = torch.sigmoid((hi_count - selected_mu) / selected_scale)
        support_mass = (cdf_hi - cdf_lo).clamp_min(1e-12)
        u = cdf_lo + torch.rand_like(cdf_lo) * support_mass
        y = selected_mu + selected_scale * torch.logit(u.clamp(1e-6, 1.0 - 1e-6))
        ships = y.round().to(dtype=torch.int64).clamp_min(min_fleet_size)
        return torch.minimum(ships, residual_budget.clamp_min(min_fleet_size))

    support = ship_support(
        residual_budget,
        min_fleet_size=min_fleet_size,
        max_ship_support=int(
            residual_budget.max().clamp_min(min_fleet_size).item() - min_fleet_size + 1
        ),
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
    support_index = log_probs.argmax(dim=-1)
    support = support.expand_as(log_probs)
    return support.gather(dim=-1, index=support_index.unsqueeze(-1)).squeeze(-1)


def ship_support(
    residual_budget: torch.Tensor,
    *,
    min_fleet_size: int,
    max_ship_support: int,
) -> torch.Tensor:
    max_residual = int(residual_budget.max().item())
    support_count = max(max_residual - min_fleet_size + 1, 1)
    support_count = min(support_count, max_ship_support)
    support_count = max(support_count, 1)
    offsets = torch.arange(
        support_count,
        device=residual_budget.device,
        dtype=residual_budget.dtype,
    )
    return min_fleet_size + offsets.view(
        *((1,) * residual_budget.ndim),
        support_count,
    )


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
    residual_budget: torch.Tensor,
    source_active: torch.Tensor,
    can_act: torch.Tensor,
    *,
    min_fleet_size: int,
    entropy_ship_quantiles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    launch_entropy = binary_entropy_from_logits(params.continue_logits.float())
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


def truncated_logistic_mixture_entropy(
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    residual_budget: torch.Tensor,
    *,
    min_fleet_size: int,
    entropy_ship_quantiles: int,
) -> torch.Tensor:
    mix_logits = mix_logits.float()
    mu = mu.float()
    scale = scale.float()
    safe_residual_budget = residual_budget.clamp_min(min_fleet_size).float()
    support_lo = float(min_fleet_size) - 0.5
    support_hi = safe_residual_budget.unsqueeze(-1) + 0.5

    cdf_lo = torch.sigmoid((support_lo - mu) / scale)
    cdf_hi = torch.sigmoid((support_hi - mu) / scale)
    support_mass = (cdf_hi - cdf_lo).clamp_min(1e-12)
    quantiles = (
        torch.arange(
            entropy_ship_quantiles,
            device=mu.device,
            dtype=mu.dtype,
        )
        + 0.5
    ) / entropy_ship_quantiles
    quantiles = quantiles.view(*((1,) * mu.ndim), entropy_ship_quantiles)
    cdf_samples = cdf_lo.unsqueeze(-1) + support_mass.unsqueeze(-1) * quantiles
    cdf_samples = cdf_samples.clamp(1e-6, 1.0 - 1e-6)
    samples = mu.unsqueeze(-1) + scale.unsqueeze(-1) * torch.logit(cdf_samples)

    z = (samples.unsqueeze(-1) - mu.unsqueeze(-2).unsqueeze(-2)) / scale.unsqueeze(
        -2,
    ).unsqueeze(-2)
    log_pdf = (
        F.logsigmoid(z) + F.logsigmoid(-z) - scale.log().unsqueeze(-2).unsqueeze(-2)
    )
    component_log_prob = (
        F.log_softmax(mix_logits, dim=-1).unsqueeze(-2).unsqueeze(-2)
        + log_pdf
        - support_mass.log().unsqueeze(-2).unsqueeze(-2)
    )
    log_prob = torch.logsumexp(component_log_prob, dim=-1)
    component_entropy = -log_prob.mean(dim=-1)
    mix_prob = torch.softmax(mix_logits, dim=-1)
    return (mix_prob * component_entropy).sum(dim=-1)


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
    target_in_range = target.ge(0) & target.lt(ACTION_ENTITY_SLOTS)
    safe_target = target.clamp(0, ACTION_ENTITY_SLOTS - 1)
    target_valid = can_act.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    if (launch & (~target_in_range | ~target_valid)).any().item():
        raise ValueError("actions.target must select a valid target for launched slots")
