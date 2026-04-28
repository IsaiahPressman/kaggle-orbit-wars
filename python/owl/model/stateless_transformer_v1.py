from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal, Self, assert_never, cast

import torch
import torch.nn.functional as F
from pydantic import Field, model_validator
from torch import nn
from torch.distributions import Bernoulli, Beta, Binomial, Categorical, VonMises

from owl.config import BaseConfig
from owl.model.attn import varlen_attention
from owl.model.base import (
    BaseModelAPI,
    ModelActionEntropies,
    ModelActionLogProbs,
    ModelActions,
    ModelEvaluation,
    ModelOutput,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    OUTER_PLAYER_SLOTS,
    ActionPureConfig,
    ObsBatch,
    ObsV1Config,
)

STATELESS_TRANSFORMER_V1: Literal["stateless_transformer_v1"] = (
    "stateless_transformer_v1"
)
_HIDDEN_INIT_GAIN = math.sqrt(2.0)
_INPUT_INIT_GAIN = 1.0
_ACTOR_HEAD_INIT_GAIN = 0.01
_CRITIC_HEAD_INIT_GAIN = 1.0
_INITIAL_KAPPA = 1.0
_INITIAL_SIZE_CONCENTRATION = 2.0


class StatelessTransformerV1Config(BaseConfig):
    model_arch: Literal["stateless_transformer_v1"] = STATELESS_TRANSFORMER_V1
    obs_spec: ObsV1Config = Field(default_factory=ObsV1Config)
    action_spec: ActionPureConfig = Field(default_factory=ActionPureConfig)
    embed_dim: int = Field(default=128, ge=1)
    depth: int = Field(default=4, ge=1)
    n_heads: int = Field(default=8, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    activation: Literal["gelu", "silu", "swiglu"] = "gelu"
    n_angle_mixtures: int = Field(default=4, ge=1)
    kappa_min: float = Field(default=1e-3, gt=0.0)
    kappa_max: float | None = Field(default=200.0, gt=0.0)
    tau_min: float = Field(default=1e-3, gt=0.0)
    alpha_beta_eps: float = Field(default=1e-4, gt=0.0)
    dir_eps: float = Field(default=1e-6, gt=0.0)
    max_ship_normalizer: float = Field(default=250.0, gt=0.0)
    entropy_ship_support_cap: int = Field(default=250, ge=1)

    @model_validator(mode="after")
    def _validate_attention_shape(self) -> Self:
        if self.embed_dim % self.n_heads != 0:
            raise ValueError("n_heads must evenly divide embed_dim")
        return self


type ModelConfig = Annotated[
    StatelessTransformerV1Config, Field(discriminator="model_arch")
]


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


@dataclass(frozen=True)
class PackedSequence:
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    seqlens: torch.Tensor
    max_seqlen: int
    batch_size: int
    padded_seq_len: int


class StatelessTransformerV1(BaseModelAPI):
    def __init__(self, config: StatelessTransformerV1Config) -> None:
        super().__init__()
        self.config = config

        dim = self.config.embed_dim
        self.planet_proj = nn.Linear(self.config.obs_spec.planet_channels, dim)
        self.fleet_proj = nn.Linear(self.config.obs_spec.fleet_channels, dim)
        self.comet_proj = nn.Linear(self.config.obs_spec.comet_channels, dim)
        self.global_proj = nn.Linear(self.config.obs_spec.global_channels, dim)
        self.player_tokens = nn.Embedding(OUTER_PLAYER_SLOTS, dim)

        self.blocks = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(self.config.depth)
        )
        self.final_norm = nn.LayerNorm(dim)

        self.critic_head = nn.Linear(dim, 1)
        self.action_info_proj = nn.Linear(1, dim)
        self.slot_dynamic_proj = nn.Linear(6, dim)
        self.actor_input_proj = nn.Linear(dim * 3, dim)
        self.actor_gru = MinGRUStack(dim, dim, n_layers=2)
        self.actor_heads = LaunchPolicyHeads(self.config)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.apply(_init_module)
        for layer in self.get_input_layers():
            _init_input_layer(layer)
        residual_gain = 1.0 / math.sqrt(2.0 * self.config.depth)
        for module in self.blocks:
            block = cast(TransformerBlock, module)
            _init_linear(block.attn.out, gain=residual_gain)
            _init_linear(block.mlp.down, gain=residual_gain)
        for layer in self.get_output_layers():
            gain = (
                _CRITIC_HEAD_INIT_GAIN
                if layer is self.critic_head
                else _ACTOR_HEAD_INIT_GAIN
            )
            _init_linear(layer, gain=gain)
        _init_direction_bias(
            self.actor_heads.dir_head,
            mixtures=self.config.n_angle_mixtures,
        )
        _init_bias(
            self.actor_heads.kappa_head,
            _softplus_inverse_for_minimum_target(_INITIAL_KAPPA, self.config.kappa_min),
        )
        _init_bias(
            self.actor_heads.size_conc_head,
            _softplus_inverse_for_minimum_target(
                _INITIAL_SIZE_CONCENTRATION, self.config.tau_min
            ),
        )

    def get_input_layers(self) -> tuple[nn.Module, ...]:
        return (
            self.planet_proj,
            self.fleet_proj,
            self.comet_proj,
            self.global_proj,
            self.player_tokens,
            self.action_info_proj,
            self.slot_dynamic_proj,
            self.actor_input_proj,
        )

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        return (
            self.critic_head,
            self.actor_heads.continue_head,
            self.actor_heads.mix_head,
            self.actor_heads.dir_head,
            self.actor_heads.kappa_head,
            self.actor_heads.size_frac_head,
            self.actor_heads.size_conc_head,
        )

    def encode_observations(self, obs: ObsBatch) -> tuple[torch.Tensor, torch.Tensor]:
        global_token = self.global_proj(obs.global_features).unsqueeze(1)
        planet_x = self.planet_proj(obs.planets) + global_token
        fleet_x = self.fleet_proj(obs.fleets) + global_token
        comet_x = self.comet_proj(obs.comets) + global_token
        player_tokens = self.player_tokens.weight.unsqueeze(0).expand(
            obs.planets.shape[0],
            -1,
            -1,
        )

        token_mask = torch.cat(
            (
                obs.planet_mask,
                obs.comet_mask,
                obs.fleet_mask,
                obs.still_playing,
            ),
            dim=1,
        )
        x = torch.cat((planet_x, comet_x, fleet_x, player_tokens), dim=1)
        packed_x, packed = pack_sequence(x, token_mask)

        for block in self.blocks:
            packed_x = block(packed_x, packed)
        x = unpack_sequence(self.final_norm(packed_x), packed)
        return x, token_mask

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
    ) -> ModelOutput:
        hidden, _ = self.encode_observations(obs)
        player_hidden = hidden[:, -OUTER_PLAYER_SLOTS:, :]
        values, winner_probabilities = self._critic(player_hidden, obs.still_playing)
        actions, log_probs, entropies = self._actor(
            hidden,
            obs.can_act,
            obs.max_launch,
            deterministic=deterministic,
        )
        return ModelOutput(
            actions=actions,
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=winner_probabilities,
        )

    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ModelActions,
    ) -> ModelEvaluation:
        hidden, _ = self.encode_observations(obs)
        player_hidden = hidden[:, -OUTER_PLAYER_SLOTS:, :]
        values, winner_probabilities = self._critic(player_hidden, obs.still_playing)
        log_probs, entropies = self._actor_log_prob(
            hidden,
            obs.can_act,
            obs.max_launch,
            actions,
        )
        return ModelEvaluation(
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=winner_probabilities,
        )

    def _critic(
        self,
        player_hidden: torch.Tensor,
        still_playing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if still_playing.shape != player_hidden.shape[:2]:
            raise ValueError(
                "still_playing must have shape "
                f"{tuple(player_hidden.shape[:2])}, got {tuple(still_playing.shape)}"
            )
        if not still_playing.any(dim=1).all():
            raise ValueError(
                "still_playing must include at least one player per batch row"
            )

        logits = self.critic_head(player_hidden).squeeze(-1)
        probabilities = masked_softmax(logits, still_playing, dim=-1)
        values = 2.0 * probabilities - 1.0
        return values, probabilities

    def _actor_inputs(
        self,
        hidden: torch.Tensor,
        max_launch: torch.Tensor,
    ) -> torch.Tensor:
        action_entity_hidden = hidden[:, :ACTION_ENTITY_SLOTS, :]

        player_hidden = hidden[:, -OUTER_PLAYER_SLOTS:, :]
        entity_features = action_entity_hidden[:, None, :, :].expand(
            -1,
            OUTER_PLAYER_SLOTS,
            -1,
            -1,
        )
        player_features = player_hidden[:, :, None, :].expand(
            -1,
            -1,
            ACTION_ENTITY_SLOTS,
            -1,
        )
        max_launch_float = max_launch.to(dtype=action_entity_hidden.dtype)
        action_info = (max_launch_float / self.config.max_ship_normalizer).unsqueeze(-1)
        action_features = self.action_info_proj(action_info)
        return self.actor_input_proj(
            torch.cat((entity_features, player_features, action_features), dim=-1)
        )

    def _actor(
        self,
        hidden: torch.Tensor,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[ModelActions, ModelActionLogProbs, ModelActionEntropies]:
        slot_input = self._actor_inputs(hidden, max_launch)
        max_slots = self.config.action_spec.max_per_planet_launches
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
        active = can_act & (remaining > 0)
        last_launch = torch.zeros_like(can_act)
        last_angle_sin = torch.zeros_like(slot_input[..., 0])
        last_angle_cos = torch.zeros_like(slot_input[..., 0])
        last_ships = torch.zeros_like(max_launch)

        # The configured slot count is a hard truncation: there is no extra
        # terminal stop probability after the final slot.
        for slot in range(max_slots):
            slot_hidden, hidden_state = self.actor_gru(
                self._slot_gru_input(
                    slot_input,
                    active,
                    remaining,
                    last_launch,
                    last_angle_sin,
                    last_angle_cos,
                    last_ships,
                    include_dynamic_features=slot > 0,
                ),
                hidden_state,
            )
            params = self.actor_heads(slot_hidden).to_distribution_dtype()
            launch = self._sample_launch(
                params.continue_logits,
                active,
                deterministic=deterministic,
            )
            angle, ships = self._sample_event(params, remaining, deterministic)
            event_mask = active & launch
            ships = torch.where(launch, ships.clamp_min(1), torch.zeros_like(ships))
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
                event_mask,
            )
            launch_entropy, event_entropy = masked_action_entropy_from_params(
                params,
                remaining,
                active,
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
            active = active & launch & (remaining > 0)
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

        per_player_action_entity_log_prob = _per_player_action_entity_log_prob(
            launch_log_tensor,
            event_log_tensor,
        )
        per_player_action_entity_entropy = _per_player_action_entity_log_prob(
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
                per_player_entity=per_player_action_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=launch_entropy_tensor,
                angle_and_size=event_entropy_tensor,
                per_player_entity=per_player_action_entity_entropy,
            ),
        )

    def _actor_log_prob(
        self,
        hidden: torch.Tensor,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        actions: ModelActions,
    ) -> tuple[ModelActionLogProbs, ModelActionEntropies]:
        _require_actions_shape(
            actions,
            (
                hidden.shape[0],
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                self.config.action_spec.max_per_planet_launches,
            ),
        )

        slot_input = self._actor_inputs(hidden, max_launch)
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
        active = can_act & (remaining > 0)
        last_launch = torch.zeros_like(can_act)
        last_angle_sin = torch.zeros_like(slot_input[..., 0])
        last_angle_cos = torch.zeros_like(slot_input[..., 0])
        last_ships = torch.zeros_like(max_launch)

        # The configured slot count is a hard truncation: there is no extra
        # terminal stop probability after the final slot.
        for slot in range(self.config.action_spec.max_per_planet_launches):
            slot_hidden, hidden_state = self.actor_gru(
                self._slot_gru_input(
                    slot_input,
                    active,
                    remaining,
                    last_launch,
                    last_angle_sin,
                    last_angle_cos,
                    last_ships,
                    include_dynamic_features=slot > 0,
                ),
                hidden_state,
            )
            params = self.actor_heads(slot_hidden).to_distribution_dtype()
            launch = actions.launch[..., slot]
            angle = actions.angle[..., slot]
            ships = actions.ships[..., slot]
            event_mask = active & launch
            _require_valid_action_slot(launch, ships, remaining, active)

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
                event_mask,
            )
            launch_entropy, event_entropy = masked_action_entropy_from_params(
                params,
                remaining,
                active,
                max_ship_support=self.config.entropy_ship_support_cap,
            )

            launch_log_slots.append(launch_log_prob)
            event_log_slots.append(event_log_prob)
            launch_entropy_slots.append(launch_entropy)
            event_entropy_slots.append(event_entropy)

            ships_used = torch.where(launch, ships, torch.zeros_like(ships))
            remaining = (remaining - ships_used).clamp_min(0)
            active = active & launch & (remaining > 0)
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
        per_player_action_entity_log_prob = _per_player_action_entity_log_prob(
            launch_log_tensor,
            event_log_tensor,
        )
        per_player_action_entity_entropy = _per_player_action_entity_log_prob(
            launch_entropy_tensor,
            event_entropy_tensor,
        )
        return (
            ModelActionLogProbs(
                launch=launch_log_tensor,
                angle_and_size=event_log_tensor,
                per_player_entity=per_player_action_entity_log_prob,
            ),
            ModelActionEntropies(
                launch=launch_entropy_tensor,
                angle_and_size=event_entropy_tensor,
                per_player_entity=per_player_action_entity_entropy,
            ),
        )

    @staticmethod
    def _sample_launch(
        logits: torch.Tensor,
        active: torch.Tensor,
        *,
        deterministic: bool,
    ) -> torch.Tensor:
        logits = logits.float()
        if deterministic:
            launch = logits.sigmoid() > 0.5
        else:
            launch = Bernoulli(logits=logits).sample().bool()

        return launch & active

    def _sample_event(
        self,
        params: PolicyParams,
        remaining: torch.Tensor,
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
            ship_mean = (remaining - 1).clamp_min(0).to(dtype=alpha.dtype)
            ship_mean = ship_mean * alpha / (alpha + beta)
            ships = ship_mean.round().to(dtype=remaining.dtype) + 1
        else:
            angle = VonMises(loc, kappa).sample().remainder(2.0 * math.pi)
            probs = Beta(alpha, beta).sample()
            trials = (remaining - 1).clamp_min(0).to(dtype=probs.dtype)
            ships = Binomial(total_count=trials, probs=probs).sample()
            ships = ships.to(dtype=remaining.dtype) + 1

        return angle, torch.minimum(ships, remaining.clamp_min(1))

    def _slot_gru_input(
        self,
        slot_input: torch.Tensor,
        active: torch.Tensor,
        remaining: torch.Tensor,
        last_launch: torch.Tensor,
        last_angle_sin: torch.Tensor,
        last_angle_cos: torch.Tensor,
        last_ships: torch.Tensor,
        *,
        include_dynamic_features: bool,
    ) -> torch.Tensor:
        if not include_dynamic_features:
            return slot_input
        dynamic_features = torch.stack(
            (
                active.to(dtype=slot_input.dtype),
                remaining.to(dtype=slot_input.dtype) / self.config.max_ship_normalizer,
                last_launch.to(dtype=slot_input.dtype),
                last_angle_sin.to(dtype=slot_input.dtype),
                last_angle_cos.to(dtype=slot_input.dtype),
                last_ships.to(dtype=slot_input.dtype) / self.config.max_ship_normalizer,
            ),
            dim=-1,
        )
        return slot_input + self.slot_dynamic_proj(dynamic_features)


class TransformerBlock(nn.Module):
    def __init__(self, config: StatelessTransformerV1Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, packed: PackedSequence) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), packed)
        return x + self.mlp(self.norm2(x))


class FeedForward(nn.Module):
    def __init__(self, config: StatelessTransformerV1Config) -> None:
        super().__init__()
        self.activation = config.activation
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        match config.activation:
            case "gelu" | "silu":
                self.up = nn.Linear(config.embed_dim, hidden_dim)
                self.down = nn.Linear(hidden_dim, config.embed_dim)
            case "swiglu":
                self.gate = nn.Linear(config.embed_dim, hidden_dim)
                self.value = nn.Linear(config.embed_dim, hidden_dim)
                self.down = nn.Linear(hidden_dim, config.embed_dim)
            case _:
                assert_never(config.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        match self.activation:
            case "gelu":
                return self.down(F.gelu(self.up(x)))
            case "silu":
                return self.down(F.silu(self.up(x)))
            case "swiglu":
                return self.down(F.silu(self.gate(x)) * self.value(x))
            case _:
                assert_never(self.activation)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: StatelessTransformerV1Config) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.q = nn.Linear(config.embed_dim, config.embed_dim)
        self.k = nn.Linear(config.embed_dim, config.embed_dim)
        self.v = nn.Linear(config.embed_dim, config.embed_dim)
        self.out = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x: torch.Tensor, packed: PackedSequence) -> torch.Tensor:
        q = self.q(x).view(x.shape[0], self.n_heads, self.head_dim)
        k = self.k(x).view(x.shape[0], self.n_heads, self.head_dim)
        v = self.v(x).view(x.shape[0], self.n_heads, self.head_dim)
        attn = varlen_attention(
            q,
            k,
            v,
            cu_seqlens=packed.cu_seqlens,
            max_seqlen=packed.max_seqlen,
        )
        return self.out(attn.reshape(x.shape[0], self.n_heads * self.head_dim))


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
    def __init__(self, config: StatelessTransformerV1Config) -> None:
        super().__init__()
        self.config = config
        mixtures = config.n_angle_mixtures
        self.continue_head = nn.Linear(config.embed_dim, 1)
        self.mix_head = nn.Linear(config.embed_dim, mixtures)
        self.dir_head = nn.Linear(config.embed_dim, mixtures * 2)
        self.kappa_head = nn.Linear(config.embed_dim, mixtures)
        self.size_frac_head = nn.Linear(config.embed_dim, mixtures)
        self.size_conc_head = nn.Linear(config.embed_dim, mixtures)

    def forward(self, x: torch.Tensor) -> PolicyParams:
        mixtures = self.config.n_angle_mixtures
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


def _init_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        _init_linear(module, gain=_HIDDEN_INIT_GAIN)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=module.embedding_dim**-0.5)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _init_input_layer(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        _init_linear(module, gain=_INPUT_INIT_GAIN)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=module.embedding_dim**-0.5)


def _init_linear(module: nn.Linear, *, gain: float) -> None:
    nn.init.orthogonal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _init_bias(module: nn.Linear, value: float) -> None:
    if module.bias is None:
        raise ValueError("expected linear layer to have a bias")
    nn.init.constant_(module.bias, value)


def _init_direction_bias(module: nn.Linear, *, mixtures: int) -> None:
    if module.bias is None:
        raise ValueError("expected direction head to have a bias")
    angles = torch.arange(
        mixtures,
        dtype=module.bias.dtype,
        device=module.bias.device,
    )
    angles = angles * (2.0 * math.pi / mixtures)
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    with torch.no_grad():
        module.bias.copy_(directions.reshape(-1))


def _softplus_inverse_for_minimum_target(target: float, minimum: float) -> float:
    return math.log(math.expm1(max(target - minimum, 1e-6)))


def pack_sequence(
    x: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, PackedSequence]:
    batch_size, padded_seq_len, _ = x.shape
    flat_mask = token_mask.reshape(-1)
    indices = flat_mask.nonzero(as_tuple=False).flatten()
    seqlens = token_mask.sum(dim=1, dtype=torch.int32)
    if not seqlens.gt(0).all():
        raise ValueError("each batch row must have at least one unmasked token")
    packed = PackedSequence(
        indices=indices,
        cu_seqlens=F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)),
        seqlens=seqlens,
        max_seqlen=int(seqlens.max().item()),
        batch_size=batch_size,
        padded_seq_len=padded_seq_len,
    )
    return x.reshape(batch_size * padded_seq_len, -1)[indices], packed


def unpack_sequence(x: torch.Tensor, packed: PackedSequence) -> torch.Tensor:
    out = torch.zeros(
        (
            packed.batch_size * packed.padded_seq_len,
            x.shape[-1],
        ),
        dtype=x.dtype,
        device=x.device,
    )
    out[packed.indices] = x
    return out.view(packed.batch_size, packed.padded_seq_len, x.shape[-1])


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked_logits, dim=dim)


def event_log_prob_from_params(
    params: PolicyParams,
    angle: torch.Tensor,
    ships: torch.Tensor,
    residual_budget: torch.Tensor,
) -> torch.Tensor:
    params = params.to_distribution_dtype()
    log_angle = von_mises_log_prob(angle, params.loc, params.kappa)
    log_size = shifted_beta_binomial_log_prob(
        ships,
        residual_budget,
        params.alpha,
        params.beta,
    )
    return torch.logsumexp(params.log_w + log_angle + log_size, dim=-1)


def masked_event_log_prob_from_params(
    params: PolicyParams,
    angle: torch.Tensor,
    ships: torch.Tensor,
    residual_budget: torch.Tensor,
    event_mask: torch.Tensor,
) -> torch.Tensor:
    safe_angle = torch.where(event_mask, angle, torch.zeros_like(angle))
    safe_ships = torch.where(event_mask, ships, torch.ones_like(ships))
    safe_residual_budget = torch.where(
        event_mask,
        residual_budget.clamp_min(1),
        torch.ones_like(residual_budget),
    )
    event_log_prob = event_log_prob_from_params(
        params,
        safe_angle,
        safe_ships,
        safe_residual_budget,
    )
    return torch.where(
        event_mask,
        event_log_prob,
        torch.zeros_like(event_log_prob),
    )


def masked_action_entropy_from_params(
    params: PolicyParams,
    residual_budget: torch.Tensor,
    active: torch.Tensor,
    *,
    max_ship_support: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = params.to_distribution_dtype()
    launch_entropy = binary_entropy_from_logits(params.continue_logits)
    event_entropy = event_entropy_from_params(
        params,
        residual_budget.clamp_min(1),
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


def binary_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    return F.binary_cross_entropy_with_logits(logits, probability, reduction="none")


def event_entropy_from_params(
    params: PolicyParams,
    residual_budget: torch.Tensor,
    *,
    max_ship_support: int,
) -> torch.Tensor:
    """Approximate entropy of the augmented latent mixture event.

    This is H(component) plus expected component entropy, not the exact
    marginal entropy of the emitted angle/ship event when components overlap.
    """
    mix_probabilities = torch.softmax(params.mix_logits, dim=-1)
    mixture_entropy = -(mix_probabilities * params.log_w).sum(dim=-1)
    component_entropy = von_mises_entropy(params.kappa) + beta_binomial_entropy(
        residual_budget,
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
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    max_ship_support: int,
) -> torch.Tensor:
    """Entropy contribution over a capped prefix of ship-count support.

    For residual budgets larger than ``max_ship_support`` the tail is
    intentionally unenumerated, making this a truncated-support entropy.
    """
    support = torch.arange(
        1,
        max_ship_support + 1,
        dtype=alpha.dtype,
        device=alpha.device,
    )
    successes = support - 1.0
    trials = (residual_budget - 1).clamp_min(0).unsqueeze(-1).unsqueeze(-1)
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


def _per_player_action_entity_log_prob(
    launch_log_prob: torch.Tensor,
    event_log_prob: torch.Tensor,
) -> torch.Tensor:
    return (launch_log_prob + event_log_prob).sum(dim=-1)


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
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    dtype = alpha.dtype
    residual = residual_budget.to(device=alpha.device)
    n_ships = ships.to(device=alpha.device)

    trials = (residual - 1).clamp_min(0).unsqueeze(-1).to(dtype=dtype)
    successes_raw = (n_ships - 1).unsqueeze(-1).to(dtype=dtype)
    valid = (
        residual.unsqueeze(-1).ge(1)
        & n_ships.unsqueeze(-1).ge(1)
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


def _require_actions_shape(
    actions: ModelActions,
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
    ships: torch.Tensor,
    remaining: torch.Tensor,
    active: torch.Tensor,
) -> None:
    if (launch & ~active).any().item():
        raise ValueError(
            "actions.launch cannot be true after a lane has stopped or is inactive"
        )
    invalid_ships = launch & (ships.lt(1) | ships.gt(remaining))
    if invalid_ships.any().item():
        raise ValueError("actions.ships must be in 1..remaining for launched slots")
