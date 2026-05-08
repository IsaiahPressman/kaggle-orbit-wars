from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta, Binomial, Categorical, VonMises

from owl.model.actor.common import (
    OutputProjectionMLP,
    binary_entropy_from_logits,
    sample_launch,
)
from owl.model.actor.config import ActorPureConfig
from owl.model.base import (
    InputLayer,
    ModelActionEntropies,
    ModelActionLogProbs,
    ModelActions,
)
from owl.rl import ACTION_ENTITY_SLOTS, OUTER_PLAYER_SLOTS


@dataclass(frozen=True)
class PolicyParams:
    continue_logits: torch.Tensor
    mix_logits: torch.Tensor
    log_w: torch.Tensor
    loc: torch.Tensor
    kappa: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor

    def to_distribution_dtype(self) -> PolicyParams:
        mix_logits = self.mix_logits.float()
        return PolicyParams(
            continue_logits=self.continue_logits.float(),
            mix_logits=mix_logits,
            log_w=F.log_softmax(mix_logits, dim=-1),
            loc=self.loc.float(),
            kappa=self.kappa.float(),
            alpha=self.alpha.float(),
            beta=self.beta.float(),
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
        self.config = config
        self.max_per_planet_launches = max_per_planet_launches
        self.launch_slot_tokens = nn.Parameter(
            torch.empty(max_per_planet_launches, embed_dim)
        )
        _init_token_parameter(self.launch_slot_tokens)
        self.slot_dynamic_proj = nn.Linear(9, embed_dim)
        self.actor_gru = MinGRUStack(embed_dim, embed_dim, n_layers=2)
        self.actor_heads = LaunchPolicyHeads(
            config,
            embed_dim=embed_dim,
            activation=activation,
        )

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        return (self.launch_slot_tokens, self.slot_dynamic_proj)

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        return (
            self.actor_heads.continue_head.out,
            self.actor_heads.mix_head.out,
            self.actor_heads.dir_head.out,
            self.actor_heads.kappa_head.out,
            self.actor_heads.size_frac_head.out,
            self.actor_heads.size_conc_head.out,
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
        max_slots = self.max_per_planet_launches
        launch_slots: list[torch.Tensor] = []
        angle_slots: list[torch.Tensor] = []
        ship_slots: list[torch.Tensor] = []
        launch_log_slots: list[torch.Tensor] = []
        event_log_slots: list[torch.Tensor] = []
        launch_entropy_slots: list[torch.Tensor] = []
        event_entropy_slots: list[torch.Tensor] = []

        hidden_state = self.actor_gru.initial_state(
            (*slot_input.shape[:-1],),
            dtype=slot_input.dtype,
            device=slot_input.device,
        )
        remaining = max_launch.clone()
        active = can_act & (remaining >= min_fleet_size)
        last_launch = torch.zeros_like(can_act)
        last_angle_sin = torch.zeros_like(slot_input[..., 0])
        last_angle_cos = torch.zeros_like(slot_input[..., 0])
        last_ships = torch.zeros_like(max_launch)

        for slot in range(max_slots):
            slot_hidden, hidden_state = self.actor_gru(
                self._slot_gru_input(
                    slot_input,
                    slot,
                    active,
                    remaining,
                    max_launch,
                    last_launch,
                    last_angle_sin,
                    last_angle_cos,
                    last_ships,
                    include_dynamic_features=slot > 0,
                ),
                hidden_state,
            )
            params = self.actor_heads(slot_hidden).to_distribution_dtype()
            launch = sample_launch(
                params.continue_logits,
                active,
                deterministic=deterministic,
            )
            angle, ships = self._sample_event(
                params,
                remaining,
                min_fleet_size,
                deterministic,
            )
            event_mask = active & launch
            ships = torch.where(
                launch,
                ships.clamp_min(min_fleet_size),
                torch.zeros_like(ships),
            )
            ships = torch.minimum(ships, remaining)
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
            event_log_prob = masked_event_log_prob_from_params(
                params,
                angle,
                ships,
                remaining,
                min_fleet_size,
                event_mask,
            )
            launch_entropy, event_entropy = masked_action_entropy_from_params(
                params,
                remaining,
                active,
                min_fleet_size=min_fleet_size,
                max_ship_support=self.config.entropy_ship_support_cap,
            )

            launch_slots.append(launch)
            angle_slots.append(angle)
            ship_slots.append(ships)
            launch_log_slots.append(launch_log_prob)
            event_log_slots.append(event_log_prob)
            launch_entropy_slots.append(launch_entropy)
            event_entropy_slots.append(event_entropy)

            remaining = (remaining - ships).clamp_min(0)
            active = active & launch & (remaining >= min_fleet_size)
            last_launch = launch
            last_angle_sin = torch.where(
                launch,
                torch.sin(angle),
                torch.zeros_like(angle),
            ).to(dtype=slot_input.dtype)
            last_angle_cos = torch.where(
                launch,
                torch.cos(angle),
                torch.zeros_like(angle),
            ).to(dtype=slot_input.dtype)
            last_ships = ships

        launch_tensor = torch.stack(launch_slots, dim=-1)
        angle_tensor = torch.stack(angle_slots, dim=-1)
        ship_tensor = torch.stack(ship_slots, dim=-1)
        launch_log_tensor = torch.stack(launch_log_slots, dim=-1)
        event_log_tensor = torch.stack(event_log_slots, dim=-1)
        launch_entropy_tensor = torch.stack(launch_entropy_slots, dim=-1)
        event_entropy_tensor = torch.stack(event_entropy_slots, dim=-1)

        per_player_entity_log_prob = _per_player_action_entity_log_prob(
            launch_log_tensor,
            event_log_tensor,
        )
        per_player_entity_entropy = _per_player_action_entity_log_prob(
            launch_entropy_tensor,
            event_entropy_tensor,
        )

        return (
            ModelActions(
                launch=launch_tensor,
                angle=angle_tensor,
                ships=ship_tensor,
            ),
            ModelActionLogProbs(
                launch=launch_log_tensor,
                angle_and_size=event_log_tensor,
                per_player_entity=per_player_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=launch_entropy_tensor,
                angle_and_size=event_entropy_tensor,
                per_player_entity=per_player_entity_entropy,
                components={
                    "launch": launch_entropy_tensor.sum(dim=-1),
                    "angle_and_size": event_entropy_tensor.sum(dim=-1),
                },
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
        _require_actions_shape(
            actions,
            (
                slot_input.shape[0],
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                self.max_per_planet_launches,
            ),
        )
        if actions.launch is None or actions.angle is None or actions.ships is None:
            raise ValueError("pure actions require launch, angle, and ships")
        action_launch = actions.launch
        action_angle = actions.angle
        action_ships = actions.ships

        launch_log_slots: list[torch.Tensor] = []
        event_log_slots: list[torch.Tensor] = []
        launch_entropy_slots: list[torch.Tensor] = []
        event_entropy_slots: list[torch.Tensor] = []

        hidden_state = self.actor_gru.initial_state(
            (*slot_input.shape[:-1],),
            dtype=slot_input.dtype,
            device=slot_input.device,
        )
        remaining = max_launch.clone()
        active = can_act & (remaining >= min_fleet_size)
        last_launch = torch.zeros_like(can_act)
        last_angle_sin = torch.zeros_like(slot_input[..., 0])
        last_angle_cos = torch.zeros_like(slot_input[..., 0])
        last_ships = torch.zeros_like(max_launch)

        for slot in range(self.max_per_planet_launches):
            slot_hidden, hidden_state = self.actor_gru(
                self._slot_gru_input(
                    slot_input,
                    slot,
                    active,
                    remaining,
                    max_launch,
                    last_launch,
                    last_angle_sin,
                    last_angle_cos,
                    last_ships,
                    include_dynamic_features=slot > 0,
                ),
                hidden_state,
            )
            params = self.actor_heads(slot_hidden).to_distribution_dtype()
            launch = action_launch[..., slot]
            angle = action_angle[..., slot]
            ships = action_ships[..., slot]
            event_mask = active & launch
            _require_valid_action_slot(
                launch,
                angle,
                ships,
                remaining,
                active,
                min_fleet_size,
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
            event_log_prob = masked_event_log_prob_from_params(
                params,
                angle,
                ships,
                remaining,
                min_fleet_size,
                event_mask,
            )
            launch_entropy, event_entropy = masked_action_entropy_from_params(
                params,
                remaining,
                active,
                min_fleet_size=min_fleet_size,
                max_ship_support=self.config.entropy_ship_support_cap,
            )

            launch_log_slots.append(launch_log_prob)
            event_log_slots.append(event_log_prob)
            launch_entropy_slots.append(launch_entropy)
            event_entropy_slots.append(event_entropy)

            ships_used = torch.where(launch, ships, torch.zeros_like(ships))
            remaining = (remaining - ships_used).clamp_min(0)
            active = active & launch & (remaining >= min_fleet_size)
            last_launch = launch
            last_angle_sin = torch.where(
                launch,
                torch.sin(angle),
                torch.zeros_like(angle),
            ).to(dtype=slot_input.dtype)
            last_angle_cos = torch.where(
                launch,
                torch.cos(angle),
                torch.zeros_like(angle),
            ).to(dtype=slot_input.dtype)
            last_ships = ships_used

        launch_log_tensor = torch.stack(launch_log_slots, dim=-1)
        event_log_tensor = torch.stack(event_log_slots, dim=-1)
        launch_entropy_tensor = torch.stack(launch_entropy_slots, dim=-1)
        event_entropy_tensor = torch.stack(event_entropy_slots, dim=-1)
        per_player_entity_log_prob = _per_player_action_entity_log_prob(
            launch_log_tensor,
            event_log_tensor,
        )
        per_player_entity_entropy = _per_player_action_entity_log_prob(
            launch_entropy_tensor,
            event_entropy_tensor,
        )
        return (
            ModelActionLogProbs(
                launch=launch_log_tensor,
                angle_and_size=event_log_tensor,
                per_player_entity=per_player_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=launch_entropy_tensor,
                angle_and_size=event_entropy_tensor,
                per_player_entity=per_player_entity_entropy,
                components={
                    "launch": launch_entropy_tensor.sum(dim=-1),
                    "angle_and_size": event_entropy_tensor.sum(dim=-1),
                },
            ),
        )

    def _sample_event(
        self,
        params: PolicyParams,
        remaining: torch.Tensor,
        min_fleet_size: int,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = params.to_distribution_dtype()
        if deterministic:
            mixture = params.mix_logits.argmax(dim=-1)
        else:
            mixture = Categorical(logits=params.mix_logits).sample()

        gather_index = mixture.unsqueeze(-1)
        loc = torch.gather(params.loc, -1, gather_index).squeeze(-1)
        kappa = torch.gather(params.kappa, -1, gather_index).squeeze(-1)
        alpha = torch.gather(params.alpha, -1, gather_index).squeeze(-1)
        beta = torch.gather(params.beta, -1, gather_index).squeeze(-1)

        if deterministic:
            angle = loc.remainder(2.0 * math.pi)
            ship_mean = (remaining - min_fleet_size).clamp_min(0).to(dtype=alpha.dtype)
            ship_mean = ship_mean * alpha / (alpha + beta)
            ships = ship_mean.round().to(dtype=remaining.dtype) + min_fleet_size
        else:
            angle = VonMises(loc, kappa).sample().remainder(2.0 * math.pi)
            probs = Beta(alpha, beta).sample()
            trials = (remaining - min_fleet_size).clamp_min(0).to(dtype=probs.dtype)
            ships = Binomial(total_count=trials, probs=probs).sample()
            ships = ships.to(dtype=remaining.dtype) + min_fleet_size

        return angle, torch.minimum(ships, remaining.clamp_min(min_fleet_size))

    def _slot_gru_input(
        self,
        slot_input: torch.Tensor,
        slot: int,
        active: torch.Tensor,
        remaining: torch.Tensor,
        initial_max_launch: torch.Tensor,
        last_launch: torch.Tensor,
        last_angle_sin: torch.Tensor,
        last_angle_cos: torch.Tensor,
        last_ships: torch.Tensor,
        *,
        include_dynamic_features: bool,
    ) -> torch.Tensor:
        slot_token = self.launch_slot_tokens[slot].to(dtype=slot_input.dtype)
        slot_context = slot_input + slot_token
        if not include_dynamic_features:
            return slot_context
        dynamic_features = self._slot_dynamic_features(
            slot,
            active,
            remaining,
            initial_max_launch,
            last_launch,
            last_angle_sin,
            last_angle_cos,
            last_ships,
            dtype=slot_input.dtype,
        )
        return slot_context + self.slot_dynamic_proj(dynamic_features)

    def _slot_dynamic_features(
        self,
        slot: int,
        active: torch.Tensor,
        remaining: torch.Tensor,
        initial_max_launch: torch.Tensor,
        last_launch: torch.Tensor,
        last_angle_sin: torch.Tensor,
        last_angle_cos: torch.Tensor,
        last_ships: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        initial_available_ships = initial_max_launch.clamp_min(1).to(dtype=dtype)
        slot_denominator = max(self.max_per_planet_launches - 1, 1)
        slot_fraction = torch.full_like(
            remaining,
            fill_value=slot / slot_denominator,
            dtype=dtype,
        )
        return torch.stack(
            (
                active.to(dtype=dtype),
                remaining.to(dtype=dtype) / self.config.max_ship_normalizer,
                last_launch.to(dtype=dtype),
                last_angle_sin.to(dtype=dtype),
                last_angle_cos.to(dtype=dtype),
                last_ships.to(dtype=dtype) / self.config.max_ship_normalizer,
                remaining.to(dtype=dtype) / initial_available_ships,
                last_ships.to(dtype=dtype) / initial_available_ships,
                slot_fraction,
            ),
            dim=-1,
        )


def _init_token_parameter(parameter: nn.Parameter) -> None:
    nn.init.normal_(parameter, mean=0.0, std=parameter.shape[-1] ** -0.5)


class MinGRUStack(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, *, n_layers: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList(
            MinGRUCell(input_dim if layer == 0 else hidden_dim, hidden_dim)
            for layer in range(n_layers)
        )

    def initial_state(
        self,
        leading_shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        return [
            torch.zeros((*leading_shape, self.hidden_dim), dtype=dtype, device=device)
            for _ in self.cells
        ]

    def forward(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        next_state = []
        layer_input = x
        for cell, layer_state in zip(self.cells, state, strict=True):
            layer_output = cell(layer_input, layer_state)
            next_state.append(layer_output)
            layer_input = layer_output
        return layer_input, next_state


class MinGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.update = nn.Linear(input_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        update = torch.sigmoid(self.update(x))
        candidate = self.candidate(x)
        return torch.lerp(prev, candidate, update)


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
        mixtures = config.n_action_mixtures
        head_config = _OutputHeadConfig(activation, embed_dim)
        self.continue_head = OutputProjectionMLP(head_config, 1)
        self.mix_head = OutputProjectionMLP(head_config, mixtures)
        self.dir_head = OutputProjectionMLP(head_config, mixtures * 2)
        self.kappa_head = OutputProjectionMLP(head_config, mixtures)
        self.size_frac_head = OutputProjectionMLP(head_config, mixtures)
        self.size_conc_head = OutputProjectionMLP(head_config, mixtures)

    def forward(self, x: torch.Tensor) -> PolicyParams:
        mixtures = self.config.n_action_mixtures
        raw_dir = self.dir_head(x).view(*x.shape[:-1], mixtures, 2)
        unit_dir = F.normalize(raw_dir, dim=-1, eps=self.config.dir_eps)
        loc = torch.atan2(unit_dir[..., 1], unit_dir[..., 0])

        kappa = self.config.kappa_min + F.softplus(self.kappa_head(x))
        if self.config.kappa_max is not None:
            kappa = kappa.clamp_max(self.config.kappa_max)

        rho = torch.sigmoid(self.size_frac_head(x))
        tau = self.config.tau_min + F.softplus(self.size_conc_head(x))
        alpha = rho * tau + self.config.alpha_beta_eps
        beta = (1.0 - rho) * tau + self.config.alpha_beta_eps

        mix_logits = self.mix_head(x)
        return PolicyParams(
            continue_logits=self.continue_head(x).squeeze(-1),
            mix_logits=mix_logits,
            log_w=F.log_softmax(mix_logits, dim=-1),
            loc=loc,
            kappa=kappa,
            alpha=alpha,
            beta=beta,
        )


@dataclass(frozen=True)
class _OutputHeadConfig:
    activation: Literal["gelu", "silu", "swiglu"]
    embed_dim: int


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
    log_size = shifted_beta_binomial_log_prob(
        ships,
        residual_budget,
        min_fleet_size,
        params.alpha,
        params.beta,
    )
    return torch.logsumexp(params.log_w + log_angle + log_size, dim=-1)


def masked_action_entropy_from_params(
    params: PolicyParams,
    residual_budget: torch.Tensor,
    active: torch.Tensor,
    *,
    min_fleet_size: int,
    max_ship_support: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = params.to_distribution_dtype()
    launch_entropy = binary_entropy_from_logits(params.continue_logits)
    event_entropy = event_entropy_from_params(
        params,
        residual_budget.clamp_min(min_fleet_size),
        min_fleet_size=min_fleet_size,
        max_ship_support=max_ship_support,
    )
    launch_probability = torch.sigmoid(params.continue_logits)
    return (
        torch.where(active, launch_entropy, torch.zeros_like(launch_entropy)),
        torch.where(
            active,
            launch_probability * event_entropy,
            torch.zeros_like(event_entropy),
        ),
    )


def event_entropy_from_params(
    params: PolicyParams,
    residual_budget: torch.Tensor,
    *,
    min_fleet_size: int,
    max_ship_support: int,
) -> torch.Tensor:
    mix_probabilities = torch.softmax(params.mix_logits, dim=-1)
    mixture_entropy = -(mix_probabilities * params.log_w).sum(dim=-1)
    component_entropy = von_mises_entropy(params.kappa) + beta_binomial_entropy(
        residual_budget,
        min_fleet_size,
        params.alpha,
        params.beta,
        max_ship_support=max_ship_support,
    )
    return mixture_entropy + (mix_probabilities * component_entropy).sum(dim=-1)


def von_mises_entropy(kappa: torch.Tensor) -> torch.Tensor:
    log_i0 = torch.log(torch.special.i0e(kappa)) + kappa
    i1_over_i0 = torch.special.i1e(kappa) / torch.special.i0e(kappa)
    return math.log(2.0 * math.pi) + log_i0 - kappa * i1_over_i0


def beta_binomial_entropy(
    residual_budget: torch.Tensor,
    min_fleet_size: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    max_ship_support: int,
) -> torch.Tensor:
    successes = torch.arange(
        0,
        max_ship_support,
        dtype=alpha.dtype,
        device=alpha.device,
    )
    trials = (residual_budget - min_fleet_size).clamp_min(0).unsqueeze(-1).unsqueeze(-1)
    successes = successes.view(*((1,) * residual_budget.ndim), max_ship_support, 1)
    alpha = alpha.unsqueeze(-2)
    beta = beta.unsqueeze(-2)
    valid = successes <= trials
    successes_safe = torch.minimum(successes, trials)
    log_comb = (
        torch.lgamma(trials + 1.0)
        - torch.lgamma(successes_safe + 1.0)
        - torch.lgamma(trials - successes_safe + 1.0)
    )
    log_prob = (
        log_comb
        + log_beta(successes_safe + alpha, trials - successes_safe + beta)
        - log_beta(alpha, beta)
    )
    log_prob = torch.where(valid, log_prob, torch.full_like(log_prob, -torch.inf))
    probabilities = torch.where(valid, log_prob.exp(), torch.zeros_like(log_prob))
    entropy = -(
        probabilities * torch.where(valid, log_prob, torch.zeros_like(log_prob))
    )
    return entropy.sum(dim=-2)


def von_mises_log_prob(
    theta: torch.Tensor,
    loc: torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:
    if theta.ndim == loc.ndim - 1:
        theta = theta.unsqueeze(-1)
    log_i0 = torch.log(torch.special.i0e(kappa)) + kappa
    return kappa * torch.cos(theta - loc) - math.log(2.0 * math.pi) - log_i0


def shifted_beta_binomial_log_prob(
    ships: torch.Tensor,
    residual_budget: torch.Tensor,
    min_fleet_size: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    dtype = alpha.dtype
    residual = residual_budget.to(device=alpha.device)
    n_ships = ships.to(device=alpha.device)

    trials = (residual - min_fleet_size).clamp_min(0).unsqueeze(-1).to(dtype=dtype)
    successes_raw = (n_ships - min_fleet_size).unsqueeze(-1).to(dtype=dtype)
    valid = (
        residual.unsqueeze(-1).ge(min_fleet_size)
        & n_ships.unsqueeze(-1).ge(min_fleet_size)
        & n_ships.unsqueeze(-1).le(residual.unsqueeze(-1))
    )

    successes = successes_raw.clamp_min(0.0)
    successes = torch.minimum(successes, trials)
    log_comb = (
        torch.lgamma(trials + 1.0)
        - torch.lgamma(successes + 1.0)
        - torch.lgamma(trials - successes + 1.0)
    )
    log_prob = (
        log_comb
        + log_beta(successes + alpha, trials - successes + beta)
        - log_beta(alpha, beta)
    )
    return torch.where(valid, log_prob, torch.full_like(log_prob, -torch.inf))


def log_beta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def _per_player_action_entity_log_prob(
    launch_log_prob: torch.Tensor,
    event_log_prob: torch.Tensor,
) -> torch.Tensor:
    return (launch_log_prob + event_log_prob).sum(dim=-1)


def _require_actions_shape(
    actions: ModelActions,
    expected_shape: tuple[int, int, int, int],
) -> None:
    for name, tensor in (
        ("launch", actions.launch),
        ("angle", actions.angle),
        ("ships", actions.ships),
    ):
        if tensor is None:
            raise ValueError(f"actions.{name} is required for pure actions")
        if tensor.shape != expected_shape:
            raise ValueError(
                f"actions.{name} must have shape {expected_shape}, got {tensor.shape}"
            )
    if actions.target is not None:
        raise ValueError("pure actions must not include actions.target")
    if actions.fleet_bin is not None:
        raise ValueError("pure actions must not include actions.fleet_bin")
    if actions.launch is None or actions.angle is None or actions.ships is None:
        raise ValueError("pure actions require launch, angle, and ships")
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
