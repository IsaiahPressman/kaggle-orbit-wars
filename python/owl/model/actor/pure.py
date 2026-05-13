from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, VonMises

from owl.model.actor.common import (
    FeedForward,
    OutputProjectionMLP,
    binary_entropy_from_logits,
    sample_launch,
)
from owl.model.actor.config import ActorPureConfig
from owl.model.actor.discrete_targets import (
    discretized_logistic_mixture_log_prob,
    log_interpolate,
    sample_discretized_logistic_mixture,
    truncated_logistic_mixture_entropy,
)
from owl.model.base import (
    InputLayer,
    ModelActionEntropies,
    ModelActionLogProbs,
)
from owl.rl import ACTION_ENTITY_SLOTS, OUTER_PLAYER_SLOTS, PureActions


@dataclass(frozen=True)
class PureActorInputs:
    source: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor


@dataclass(frozen=True)
class AnglePolicyParams:
    continue_logits: torch.Tensor
    angle_mix_logits: torch.Tensor
    angle_log_w: torch.Tensor
    loc: torch.Tensor
    kappa: torch.Tensor

    def to_distribution_dtype(self) -> AnglePolicyParams:
        angle_mix_logits = self.angle_mix_logits.float()
        return AnglePolicyParams(
            continue_logits=self.continue_logits.float(),
            angle_mix_logits=angle_mix_logits,
            angle_log_w=F.log_softmax(angle_mix_logits, dim=-1),
            loc=self.loc.float(),
            kappa=self.kappa.float(),
        )


@dataclass(frozen=True)
class SizePolicyParams:
    size_mix_logits: torch.Tensor
    size_mu: torch.Tensor
    size_scale: torch.Tensor

    def to_distribution_dtype(self) -> SizePolicyParams:
        return SizePolicyParams(
            size_mix_logits=self.size_mix_logits.float(),
            size_mu=self.size_mu.float(),
            size_scale=self.size_scale.float(),
        )


@dataclass(frozen=True)
class PolicyParams:
    continue_logits: torch.Tensor
    angle_mix_logits: torch.Tensor
    angle_log_w: torch.Tensor
    loc: torch.Tensor
    kappa: torch.Tensor
    size_mix_logits: torch.Tensor
    size_mu: torch.Tensor
    size_scale: torch.Tensor

    def to_distribution_dtype(self) -> PolicyParams:
        angle_mix_logits = self.angle_mix_logits.float()
        return PolicyParams(
            continue_logits=self.continue_logits.float(),
            angle_mix_logits=angle_mix_logits,
            angle_log_w=F.log_softmax(angle_mix_logits, dim=-1),
            loc=self.loc.float(),
            kappa=self.kappa.float(),
            size_mix_logits=self.size_mix_logits.float(),
            size_mu=self.size_mu.float(),
            size_scale=self.size_scale.float(),
        )


class PureActor(nn.Module):
    def __init__(
        self,
        config: ActorPureConfig,
        *,
        embed_dim: int,
        max_per_planet_launches: int,
        activation: Literal["gelu", "silu", "swiglu"],
    ) -> None:
        super().__init__()
        if max_per_planet_launches != 1:
            raise ValueError("pure actor requires max_per_planet_launches=1")
        self.config = config
        self.max_per_planet_launches = max_per_planet_launches
        head_config = _OutputHeadConfig(activation, embed_dim)
        self.head_dim = embed_dim
        self.source_norm = nn.LayerNorm(embed_dim)
        self.target_norm = nn.LayerNorm(embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(head_config)
        self.continue_source_proj = nn.Linear(embed_dim, embed_dim)
        self.angle_source_proj = nn.Linear(embed_dim, embed_dim)
        self.angle_direction_proj = AngleDirectionProjection(head_config)
        self.size_pair_proj = nn.Linear(embed_dim, embed_dim)
        self.actor_heads = LaunchPolicyHeads(
            config,
            embed_dim=embed_dim,
            activation=activation,
        )

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        return ()

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        return (
            self.actor_heads.continue_head.out,
            self.actor_heads.angle_mix_head.out,
            self.actor_heads.dir_head.out,
            self.actor_heads.kappa_head.out,
            self.actor_heads.size_mix_head.out,
            self.actor_heads.mean_head.out,
            self.actor_heads.scale_head.out,
        )

    def forward(
        self,
        actor_inputs: PureActorInputs,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        *,
        min_fleet_size: int,
        deterministic: bool,
    ) -> tuple[PureActions, ModelActionLogProbs, ModelActionEntropies]:
        active = can_act & (max_launch >= min_fleet_size)
        angle_params = self._angle_params(actor_inputs)
        launch = sample_launch(
            angle_params.continue_logits,
            active,
            deterministic=deterministic,
        )
        angle = sample_angle_mixture(angle_params, deterministic=deterministic)
        params = self._policy_params_for_angle(
            angle_params,
            actor_inputs,
            max_launch,
            angle,
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
        angle = torch.where(launch, angle, torch.zeros_like(angle))

        launch_log_prob = -F.binary_cross_entropy_with_logits(
            params.continue_logits,
            launch.to(dtype=params.continue_logits.dtype),
            reduction="none",
        )
        launch_log_prob = torch.where(
            active,
            launch_log_prob,
            torch.zeros_like(launch_log_prob),
        )
        event_mask = active & launch
        event_log_prob = masked_event_log_prob_from_params(
            params,
            angle,
            ships,
            max_launch,
            min_fleet_size,
            event_mask,
        )
        entropy_params = self._policy_params_for_entropy(
            angle_params,
            actor_inputs,
            max_launch,
            min_fleet_size=min_fleet_size,
        )
        (
            launch_entropy,
            event_entropy,
            angle_entropy,
            size_entropy,
            size_mixture_entropy,
            size_logistic_entropy,
        ) = masked_action_entropy_from_params(
            entropy_params,
            max_launch,
            active,
            min_fleet_size=min_fleet_size,
            entropy_ship_quantiles=self.config.entropy_ship_quantiles,
        )
        launch_tensor = launch.unsqueeze(-1)
        angle_tensor = angle.unsqueeze(-1)
        ship_tensor = ships.unsqueeze(-1)
        launch_log_tensor = launch_log_prob.unsqueeze(-1)
        event_log_tensor = event_log_prob.unsqueeze(-1)
        launch_entropy_tensor = launch_entropy.unsqueeze(-1)
        event_entropy_tensor = event_entropy.unsqueeze(-1)
        per_player_entity_log_prob = launch_log_prob + event_log_prob
        per_player_entity_entropy = launch_entropy + event_entropy

        return (
            PureActions(
                launch=launch_tensor,
                angle=angle_tensor,
                ships=ship_tensor,
            ),
            ModelActionLogProbs(
                launch=launch_log_tensor,
                event=event_log_tensor,
                per_player_entity=per_player_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=launch_entropy_tensor,
                event=event_entropy_tensor,
                per_player_entity=per_player_entity_entropy,
                components={
                    "launch": launch_entropy,
                    "angle": angle_entropy,
                    "fleet_size_full": size_entropy,
                    "fleet_size_mixture": size_mixture_entropy,
                    "fleet_size_logistic": size_logistic_entropy,
                    "event": event_entropy,
                },
            ),
        )

    def log_prob(
        self,
        actor_inputs: PureActorInputs,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        actions: PureActions,
        *,
        min_fleet_size: int,
    ) -> tuple[ModelActionLogProbs, ModelActionEntropies]:
        _require_actions_shape(
            actions,
            (
                actor_inputs.source.shape[0],
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                self.max_per_planet_launches,
            ),
        )
        action_launch = actions.launch
        action_angle = actions.angle
        action_ships = actions.ships

        active = can_act & (max_launch >= min_fleet_size)
        angle_params = self._angle_params(actor_inputs)
        launch = action_launch[..., 0]
        angle = action_angle[..., 0]
        ships = action_ships[..., 0]
        _require_valid_action_slot(
            launch,
            angle,
            ships,
            max_launch,
            active,
            min_fleet_size,
        )
        params = self._policy_params_for_angle(
            angle_params,
            actor_inputs,
            max_launch,
            angle,
            min_fleet_size=min_fleet_size,
        )
        launch_log_prob = -F.binary_cross_entropy_with_logits(
            params.continue_logits,
            launch.to(dtype=params.continue_logits.dtype),
            reduction="none",
        )
        launch_log_prob = torch.where(
            active,
            launch_log_prob,
            torch.zeros_like(launch_log_prob),
        )
        event_mask = active & launch
        event_log_prob = masked_event_log_prob_from_params(
            params,
            angle,
            ships,
            max_launch,
            min_fleet_size,
            event_mask,
        )
        entropy_params = self._policy_params_for_entropy(
            angle_params,
            actor_inputs,
            max_launch,
            min_fleet_size=min_fleet_size,
        )
        (
            launch_entropy,
            event_entropy,
            angle_entropy,
            size_entropy,
            size_mixture_entropy,
            size_logistic_entropy,
        ) = masked_action_entropy_from_params(
            entropy_params,
            max_launch,
            active,
            min_fleet_size=min_fleet_size,
            entropy_ship_quantiles=self.config.entropy_ship_quantiles,
        )
        launch_log_tensor = launch_log_prob.unsqueeze(-1)
        event_log_tensor = event_log_prob.unsqueeze(-1)
        launch_entropy_tensor = launch_entropy.unsqueeze(-1)
        event_entropy_tensor = event_entropy.unsqueeze(-1)
        per_player_entity_log_prob = launch_log_prob + event_log_prob
        per_player_entity_entropy = launch_entropy + event_entropy
        return (
            ModelActionLogProbs(
                launch=launch_log_tensor,
                event=event_log_tensor,
                per_player_entity=per_player_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=launch_entropy_tensor,
                event=event_entropy_tensor,
                per_player_entity=per_player_entity_entropy,
                components={
                    "launch": launch_entropy,
                    "angle": angle_entropy,
                    "fleet_size_full": size_entropy,
                    "fleet_size_mixture": size_mixture_entropy,
                    "fleet_size_logistic": size_logistic_entropy,
                    "event": event_entropy,
                },
            ),
        )

    def _angle_params(
        self,
        actor_inputs: PureActorInputs,
    ) -> AnglePolicyParams:
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
                "pure actor source input must have shape "
                f"{expected_input_shape}, got {tuple(source_input.shape)}"
            )
        if target_input.shape != expected_input_shape:
            raise ValueError(
                "pure actor target input must have shape "
                f"{expected_input_shape}, got {tuple(target_input.shape)}"
            )
        if actor_inputs.target_mask.shape != (
            source_input.shape[0],
            ACTION_ENTITY_SLOTS,
        ):
            expected_shape = (source_input.shape[0], ACTION_ENTITY_SLOTS)
            raise ValueError(
                "pure actor target_mask must have shape "
                f"{expected_shape}, got {tuple(actor_inputs.target_mask.shape)}"
            )
        continue_hidden = self.continue_source_proj(source_input)
        angle_hidden = self.angle_source_proj(source_input)
        return self.actor_heads.angle_params(
            continue_hidden,
            angle_hidden,
        ).to_distribution_dtype()

    def _policy_params_for_angle(
        self,
        angle_params: AnglePolicyParams,
        actor_inputs: PureActorInputs,
        max_launch: torch.Tensor,
        angle: torch.Tensor,
        *,
        min_fleet_size: int,
    ) -> PolicyParams:
        selected_target_values = self._selected_angle_target_values(actor_inputs, angle)
        size_params = self._size_params_from_target_values(
            actor_inputs.source,
            max_launch,
            selected_target_values,
            min_fleet_size=min_fleet_size,
        )
        return PolicyParams(
            continue_logits=angle_params.continue_logits,
            angle_mix_logits=angle_params.angle_mix_logits,
            angle_log_w=angle_params.angle_log_w,
            loc=angle_params.loc,
            kappa=angle_params.kappa,
            size_mix_logits=size_params.size_mix_logits,
            size_mu=size_params.size_mu,
            size_scale=size_params.size_scale,
        ).to_distribution_dtype()

    def _policy_params_for_entropy(
        self,
        angle_params: AnglePolicyParams,
        actor_inputs: PureActorInputs,
        max_launch: torch.Tensor,
        *,
        min_fleet_size: int,
    ) -> PolicyParams:
        angle = sample_angle_mixture(angle_params, deterministic=True)
        return self._policy_params_for_angle(
            angle_params,
            actor_inputs,
            max_launch,
            angle,
            min_fleet_size=min_fleet_size,
        )

    def _selected_angle_target_values(
        self,
        actor_inputs: PureActorInputs,
        angle: torch.Tensor,
    ) -> torch.Tensor:
        source_input = actor_inputs.source
        target_input = actor_inputs.target
        angle_features = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        angle_hidden = self.angle_direction_proj(
            angle_features.to(dtype=source_input.dtype)
        )
        source_x = self.source_norm(source_input + angle_hidden)
        target_x = self.target_norm(target_input)
        q = self.q(source_x)
        k = self.k(target_x)
        v = self.v(target_x)
        target_logits = torch.einsum("bpsd,bptd->bpst", q, k)
        target_logits = target_logits / math.sqrt(self.head_dim)

        target_mask = actor_inputs.target_mask[:, None, None, :].expand_as(
            target_logits
        )
        source_indices = torch.arange(ACTION_ENTITY_SLOTS, device=target_logits.device)
        target_mask = target_mask.clone()
        target_mask[:, :, source_indices, source_indices] = False
        target_logits = target_logits.masked_fill(
            ~target_mask,
            torch.finfo(target_logits.dtype).min,
        )
        safe_target_logits = torch.where(
            target_mask.any(dim=-1, keepdim=True),
            target_logits,
            torch.zeros_like(target_logits),
        )
        target_weights = F.softmax(safe_target_logits.float(), dim=-1).to(
            dtype=target_input.dtype
        )
        target_weights = torch.where(
            target_mask,
            target_weights,
            torch.zeros_like(target_weights),
        )
        return torch.einsum("bpst,bptd->bpsd", target_weights, v)

    def _size_params_from_target_values(
        self,
        source_input: torch.Tensor,
        max_launch: torch.Tensor,
        target_values: torch.Tensor,
        *,
        min_fleet_size: int,
    ) -> SizePolicyParams:
        selected_v = self.out(target_values)
        enriched = source_input + selected_v
        enriched = enriched + self.mlp(self.norm2(enriched))
        source_hidden = self.size_pair_proj(enriched)
        return self.actor_heads.size_params(
            source_hidden,
            max_launch,
            min_fleet_size=min_fleet_size,
        ).to_distribution_dtype()


class LaunchPolicyHeads(nn.Module):
    def __init__(
        self,
        config: ActorPureConfig,
        *,
        embed_dim: int,
        activation: Literal["gelu", "silu", "swiglu"],
    ) -> None:
        super().__init__()
        self.config = config
        head_config = _OutputHeadConfig(activation, embed_dim)
        self.continue_head = OutputProjectionMLP(head_config, 1)
        self.angle_mix_head = OutputProjectionMLP(
            head_config,
            config.n_angle_mixtures,
        )
        self.base_dirs = nn.Parameter(torch.empty(config.n_angle_mixtures, 2))
        _init_evenly_spaced_directions(self.base_dirs)
        self.dir_head = OutputProjectionMLP(head_config, config.n_angle_mixtures * 2)
        self.kappa_head = OutputProjectionMLP(head_config, config.n_angle_mixtures)
        self.size_mix_head = OutputProjectionMLP(
            head_config,
            config.n_fleet_size_mixtures,
        )
        self.mean_head = OutputProjectionMLP(head_config, config.n_fleet_size_mixtures)
        self.scale_head = OutputProjectionMLP(head_config, config.n_fleet_size_mixtures)

    def angle_params(
        self,
        continue_hidden: torch.Tensor,
        angle_hidden: torch.Tensor,
    ) -> AnglePolicyParams:
        angle_mixtures = self.config.n_angle_mixtures
        raw_dir = self.dir_head(angle_hidden).view(
            *angle_hidden.shape[:-1],
            angle_mixtures,
            2,
        )
        base_dirs = self.base_dirs.to(dtype=raw_dir.dtype, device=raw_dir.device)
        unit_dir = F.normalize(raw_dir + base_dirs, dim=-1, eps=self.config.dir_eps)
        loc = torch.atan2(unit_dir[..., 1], unit_dir[..., 0])

        kappa = log_interpolate(
            self.config.kappa_min,
            torch.full_like(loc, self.config.kappa_max),
            torch.sigmoid(self.kappa_head(angle_hidden).float()),
        )
        angle_mix_logits = self.angle_mix_head(angle_hidden)
        return AnglePolicyParams(
            continue_logits=self.continue_head(continue_hidden).squeeze(-1),
            angle_mix_logits=angle_mix_logits,
            angle_log_w=F.log_softmax(angle_mix_logits, dim=-1),
            loc=loc,
            kappa=kappa,
        )

    def size_params(
        self,
        size_hidden: torch.Tensor,
        max_launch: torch.Tensor,
        *,
        min_fleet_size: int,
    ) -> SizePolicyParams:
        residual_budget = max_launch.to(dtype=torch.float32).unsqueeze(-1)
        support_lo = torch.full_like(residual_budget, float(min_fleet_size))
        support_width = (residual_budget - support_lo + 1.0).clamp_min(1.0)
        rho = torch.sigmoid(self.mean_head(size_hidden).float())
        mu = support_lo + rho * (residual_budget - support_lo).clamp_min(0.0)
        scale_upper = torch.maximum(
            torch.full_like(residual_budget, self.config.scale_max_abs_floor),
            support_width * self.config.scale_max_frac,
        )
        scale = log_interpolate(
            self.config.scale_min,
            scale_upper,
            torch.sigmoid(self.scale_head(size_hidden).float()),
        )
        return SizePolicyParams(
            size_mix_logits=self.size_mix_head(size_hidden),
            size_mu=mu,
            size_scale=scale,
        )


def _init_evenly_spaced_directions(parameter: nn.Parameter) -> None:
    angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        parameter.shape[0] + 1,
        dtype=parameter.dtype,
        device=parameter.device,
    )[:-1]
    with torch.no_grad():
        parameter[:, 0].copy_(torch.cos(angles))
        parameter[:, 1].copy_(torch.sin(angles))


@dataclass(frozen=True)
class _OutputHeadConfig:
    activation: Literal["gelu", "silu", "swiglu"]
    embed_dim: int
    mlp_ratio: float = 1.0


class AngleDirectionProjection(nn.Module):
    def __init__(self, config: _OutputHeadConfig) -> None:
        super().__init__()
        self.input = nn.Linear(2, config.embed_dim)
        self.output = OutputProjectionMLP(config, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.input(x))


def sample_angle_mixture(
    params: AnglePolicyParams | PolicyParams,
    *,
    deterministic: bool,
) -> torch.Tensor:
    params = params.to_distribution_dtype()
    if deterministic:
        mixture = params.angle_mix_logits.argmax(dim=-1)
    else:
        mixture = Categorical(logits=params.angle_mix_logits).sample()

    gather_index = mixture.unsqueeze(-1)
    loc = torch.gather(params.loc, -1, gather_index).squeeze(-1)
    kappa = torch.gather(params.kappa, -1, gather_index).squeeze(-1)
    if deterministic:
        return loc.remainder(2.0 * math.pi)
    return VonMises(loc, kappa).sample().remainder(2.0 * math.pi)


def masked_event_log_prob_from_params(
    params: PolicyParams,
    angle: torch.Tensor,
    ships: torch.Tensor,
    residual_budget: torch.Tensor,
    min_fleet_size: int,
    event_mask: torch.Tensor,
) -> torch.Tensor:
    safe_angle = torch.where(event_mask, angle, torch.zeros_like(angle))
    safe_ships = torch.where(
        event_mask,
        ships,
        torch.full_like(ships, min_fleet_size),
    )
    safe_residual_budget = torch.where(
        event_mask,
        residual_budget.clamp_min(min_fleet_size),
        torch.full_like(residual_budget, min_fleet_size),
    )
    event_log_prob = event_log_prob_from_params(
        params,
        safe_angle,
        safe_ships,
        safe_residual_budget,
        min_fleet_size,
    )
    return torch.where(
        event_mask,
        event_log_prob,
        torch.zeros_like(event_log_prob),
    )


def event_log_prob_from_params(
    params: PolicyParams,
    angle: torch.Tensor,
    ships: torch.Tensor,
    residual_budget: torch.Tensor,
    min_fleet_size: int,
) -> torch.Tensor:
    params = params.to_distribution_dtype()
    log_angle = von_mises_log_prob(angle, params.loc, params.kappa)
    angle_log_prob = torch.logsumexp(params.angle_log_w + log_angle, dim=-1)
    size_log_prob = discretized_logistic_mixture_log_prob(
        ships,
        residual_budget,
        params.size_mix_logits,
        params.size_mu,
        params.size_scale,
        min_fleet_size=min_fleet_size,
    )
    return angle_log_prob + size_log_prob


def masked_action_entropy_from_params(
    params: PolicyParams,
    residual_budget: torch.Tensor,
    active: torch.Tensor,
    *,
    min_fleet_size: int,
    entropy_ship_quantiles: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    params = params.to_distribution_dtype()
    launch_entropy = binary_entropy_from_logits(params.continue_logits)
    (
        event_entropy,
        angle_entropy,
        size_entropy,
        size_mixture_entropy,
        size_logistic_entropy,
    ) = event_entropy_from_params(
        params,
        residual_budget.clamp_min(min_fleet_size),
        min_fleet_size=min_fleet_size,
        entropy_ship_quantiles=entropy_ship_quantiles,
    )
    launch_probability = torch.sigmoid(params.continue_logits)
    return (
        torch.where(active, launch_entropy, torch.zeros_like(launch_entropy)),
        torch.where(
            active,
            launch_probability * event_entropy,
            torch.zeros_like(event_entropy),
        ),
        torch.where(active, angle_entropy, torch.zeros_like(angle_entropy)),
        torch.where(active, size_entropy, torch.zeros_like(size_entropy)),
        torch.where(
            active,
            size_mixture_entropy,
            torch.zeros_like(size_mixture_entropy),
        ),
        torch.where(
            active,
            size_logistic_entropy,
            torch.zeros_like(size_logistic_entropy),
        ),
    )


def event_entropy_from_params(
    params: PolicyParams,
    residual_budget: torch.Tensor,
    *,
    min_fleet_size: int,
    entropy_ship_quantiles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    angle_mix_probabilities = torch.softmax(params.angle_mix_logits, dim=-1)
    angle_mixture_entropy = -(angle_mix_probabilities * params.angle_log_w).sum(dim=-1)
    angle_entropy = angle_mixture_entropy + (
        angle_mix_probabilities * von_mises_entropy(params.kappa)
    ).sum(dim=-1)
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
        angle_entropy + size_entropy,
        angle_entropy,
        size_entropy,
        size_mixture_entropy,
        size_logistic_entropy,
    )


def von_mises_entropy(kappa: torch.Tensor) -> torch.Tensor:
    log_i0 = torch.log(torch.special.i0e(kappa)) + kappa
    i1_over_i0 = torch.special.i1e(kappa) / torch.special.i0e(kappa)
    return math.log(2.0 * math.pi) + log_i0 - kappa * i1_over_i0


def von_mises_log_prob(
    theta: torch.Tensor,
    loc: torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:
    if theta.ndim == loc.ndim - 1:
        theta = theta.unsqueeze(-1)
    log_i0 = torch.log(torch.special.i0e(kappa)) + kappa
    return kappa * torch.cos(theta - loc) - math.log(2.0 * math.pi) - log_i0


def _per_player_action_entity_log_prob(
    launch_log_prob: torch.Tensor,
    event_log_prob: torch.Tensor,
) -> torch.Tensor:
    return (launch_log_prob + event_log_prob).sum(dim=-1)


def _require_actions_shape(
    actions: PureActions,
    expected_shape: tuple[int, int, int, int],
) -> None:
    for name, tensor in (
        ("launch", actions.launch),
        ("angle", actions.angle),
        ("ships", actions.ships),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"actions.{name} must have shape {expected_shape}, got {tensor.shape}"
            )
    if actions.launch.dtype != torch.bool:
        raise ValueError(
            f"actions.launch must have dtype torch.bool, got {actions.launch.dtype}"
        )
    if actions.angle.dtype != torch.float32:
        raise ValueError(
            f"actions.angle must have dtype torch.float32, got {actions.angle.dtype}"
        )
    if actions.ships.dtype != torch.int64:
        raise ValueError(
            f"actions.ships must have dtype torch.int64, got {actions.ships.dtype}"
        )


def _require_valid_action_slot(
    launch: torch.Tensor,
    angle: torch.Tensor,
    ships: torch.Tensor,
    remaining: torch.Tensor,
    active: torch.Tensor,
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
    launched_active = launch & active
    if (~torch.isfinite(angle) & launched_active).any().item():
        raise ValueError("actions.angle must be finite for launched slots")
