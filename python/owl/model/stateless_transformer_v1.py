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
from owl.model.attn import use_flash_attn, varlen_attention
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
    ActionConfig,
    ActionDiscreteTargetsConfig,
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


class ActorPureConfig(BaseConfig):
    action_spec: Literal["pure"] = "pure"


class ActorDiscreteTargetsConfig(BaseConfig):
    action_spec: Literal["discrete_targets"] = "discrete_targets"
    scale_min: float = Field(default=0.25, gt=0.0)
    min_log_scale: float = -7.0
    max_log_scale: float = 0.5

    @model_validator(mode="after")
    def _validate_scale_clamp(self) -> Self:
        if self.min_log_scale > self.max_log_scale:
            raise ValueError("min_log_scale must be <= max_log_scale")
        return self


type ActorConfig = Annotated[
    ActorPureConfig | ActorDiscreteTargetsConfig,
    Field(discriminator="action_spec"),
]


class StatelessTransformerV1Config(BaseConfig):
    model_arch: Literal["stateless_transformer_v1"] = STATELESS_TRANSFORMER_V1
    action_spec: Literal["pure", "discrete_targets"] = "pure"
    embed_dim: int = Field(default=128, ge=1)
    depth: int = Field(default=4, ge=1)
    n_heads: int = Field(default=8, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    activation: Literal["gelu", "silu", "swiglu"] = "gelu"
    n_action_mixtures: int = Field(default=4, ge=1)
    kappa_min: float = Field(default=1e-3, gt=0.0)
    kappa_max: float | None = Field(default=200.0, gt=0.0)
    tau_min: float = Field(default=1e-3, gt=0.0)
    alpha_beta_eps: float = Field(default=1e-4, gt=0.0)
    dir_eps: float = Field(default=1e-6, gt=0.0)
    max_ship_normalizer: float = Field(default=250.0, gt=0.0)
    entropy_ship_support_cap: int = Field(default=256, ge=1)
    force_flash_attn: bool = False
    actor: ActorConfig = Field(default_factory=ActorPureConfig)

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.embed_dim % self.n_heads != 0:
            raise ValueError("n_heads must evenly divide embed_dim")
        if self.actor.action_spec != self.action_spec:
            raise ValueError("actor action_spec must match model action_spec")
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
    def __init__(
        self,
        config: StatelessTransformerV1Config,
        *,
        obs_spec: ObsV1Config,
        action_spec: ActionConfig,
    ) -> None:
        super().__init__()
        if config.action_spec != action_spec.action_spec:
            raise ValueError("model config action_spec must match env action_spec")
        if config.actor.action_spec != action_spec.action_spec:
            raise ValueError("model actor config must match env action_spec")
        if (
            isinstance(action_spec, ActionDiscreteTargetsConfig)
            and action_spec.max_per_planet_launches != 1
        ):
            raise ValueError(
                "discrete_targets actor requires max_per_planet_launches=1"
            )
        self.config = config
        self.obs_spec = obs_spec
        self.action_spec = action_spec

        dim = self.config.embed_dim
        self.planet_proj = nn.Linear(self.obs_spec.planet_channels, dim)
        self.fleet_proj = nn.Linear(self.obs_spec.fleet_channels, dim)
        self.comet_proj = nn.Linear(self.obs_spec.comet_channels, dim)
        self.global_proj = nn.Linear(self.obs_spec.global_channels, dim)
        self.player_tokens = nn.Embedding(OUTER_PLAYER_SLOTS, dim)

        self.blocks = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(self.config.depth)
        )
        self.final_norm = nn.LayerNorm(dim)

        self.critic_head = nn.Linear(dim, 1)
        self.action_info_proj = nn.Linear(1, dim)
        self.actor_input_proj = nn.Linear(dim * 3, dim)
        self.launch_slot_tokens: nn.Embedding | None
        self.slot_dynamic_proj: nn.Linear | None
        self.actor_gru: MinGRUStack | None
        self.actor_heads: LaunchPolicyHeads | None
        self.discrete_actor: DiscreteTargetActor | None
        if isinstance(action_spec, ActionPureConfig):
            self.launch_slot_tokens = nn.Embedding(
                self.action_spec.max_per_planet_launches,
                dim,
            )
            self.slot_dynamic_proj = nn.Linear(9, dim)
            self.actor_gru = MinGRUStack(dim, dim, n_layers=2)
            self.actor_heads = LaunchPolicyHeads(self.config)
            self.discrete_actor = None
        else:
            self.launch_slot_tokens = None
            self.slot_dynamic_proj = None
            self.actor_gru = None
            self.actor_heads = None
            self.discrete_actor = DiscreteTargetActor(self.config)

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

    def get_input_layers(self) -> tuple[nn.Module, ...]:
        layers: list[nn.Module] = [
            self.planet_proj,
            self.fleet_proj,
            self.comet_proj,
            self.global_proj,
            self.player_tokens,
            self.action_info_proj,
            self.actor_input_proj,
        ]
        if self.launch_slot_tokens is not None:
            layers.append(self.launch_slot_tokens)
        if self.slot_dynamic_proj is not None:
            layers.append(self.slot_dynamic_proj)
        if self.discrete_actor is not None:
            layers.extend(self.discrete_actor.get_input_layers())
        return tuple(layers)

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        layers: list[nn.Linear] = [self.critic_head]
        if self.actor_heads is not None:
            layers.extend(
                (
                    self.actor_heads.continue_head,
                    self.actor_heads.mix_head,
                    self.actor_heads.dir_head,
                    self.actor_heads.kappa_head,
                    self.actor_heads.size_frac_head,
                    self.actor_heads.size_conc_head,
                )
            )
        if self.discrete_actor is not None:
            layers.extend(self.discrete_actor.get_output_layers())
        return tuple(layers)

    def encode_observations(self, obs: ObsBatch) -> tuple[torch.Tensor, torch.Tensor]:
        global_token = self.global_proj(obs.global_features).unsqueeze(1)
        planet_x = self.planet_proj(obs.planets) + global_token
        fleet_x = self.fleet_proj(obs.fleets) + global_token
        comet_x = self.comet_proj(obs.comets) + global_token
        player_tokens = self.player_tokens.weight.to(dtype=global_token.dtype)
        player_tokens = player_tokens.unsqueeze(0).expand(
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
        packed: PackedSequence | None
        should_use_flash = use_flash_attn(x)
        if self.config.force_flash_attn and not should_use_flash:
            raise RuntimeError(
                "force_flash_attn=True requires CUDA fp16/bf16 tensors "
                "and the flash-attn package"
            )
        if should_use_flash:
            x, packed = pack_sequence(x, token_mask)
            block_token_mask = None
        else:
            packed = None
            block_token_mask = token_mask
        for block in self.blocks:
            x = block(x, block_token_mask, packed)
        x = self.final_norm(x)
        if packed is not None:
            x = unpack_sequence(x, packed)
        x = x.masked_fill(~token_mask.unsqueeze(-1), 0.0)
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
        if self.discrete_actor is not None:
            return self._discrete_actor(hidden, can_act, max_launch, deterministic)
        if self.actor_gru is None or self.actor_heads is None:
            raise RuntimeError("pure actor modules are not initialized")
        slot_input = self._actor_inputs(hidden, max_launch)
        max_slots = self.action_spec.max_per_planet_launches
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
        min_fleet_size = self.action_spec.min_fleet_size
        active = can_act & (remaining >= min_fleet_size)
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
            launch = self._sample_launch(
                params.continue_logits,
                active,
                deterministic=deterministic,
            )
            angle, ships = self._sample_event(params, remaining, deterministic)
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
                target=torch.zeros_like(ship_tensor),
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
        if self.discrete_actor is not None:
            return self._discrete_actor_log_prob(hidden, can_act, max_launch, actions)
        if self.actor_gru is None or self.actor_heads is None:
            raise RuntimeError("pure actor modules are not initialized")
        _require_actions_shape(
            actions,
            (
                hidden.shape[0],
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                self.action_spec.max_per_planet_launches,
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
        min_fleet_size = self.action_spec.min_fleet_size
        active = can_act & (remaining >= min_fleet_size)
        last_launch = torch.zeros_like(can_act)
        last_angle_sin = torch.zeros_like(slot_input[..., 0])
        last_angle_cos = torch.zeros_like(slot_input[..., 0])
        last_ships = torch.zeros_like(max_launch)

        # The configured slot count is a hard truncation: there is no extra
        # terminal stop probability after the final slot.
        for slot in range(self.action_spec.max_per_planet_launches):
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
            launch = actions.launch[..., slot]
            angle = actions.angle[..., slot]
            ships = actions.ships[..., slot]
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
        min_fleet_size = self.action_spec.min_fleet_size
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
        if self.launch_slot_tokens is None or self.slot_dynamic_proj is None:
            raise RuntimeError("pure actor recurrent modules are not initialized")
        slot_token = self.launch_slot_tokens.weight[slot].to(dtype=slot_input.dtype)
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
        slot_denominator = max(self.action_spec.max_per_planet_launches - 1, 1)
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

    def _discrete_actor(
        self,
        hidden: torch.Tensor,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        deterministic: bool,
    ) -> tuple[ModelActions, ModelActionLogProbs, ModelActionEntropies]:
        if self.discrete_actor is None:
            raise RuntimeError("discrete actor module is not initialized")
        slot_input = self._actor_inputs(hidden, max_launch)
        return self.discrete_actor(
            slot_input,
            can_act,
            max_launch,
            min_fleet_size=self.action_spec.min_fleet_size,
            max_ship_support=self.config.entropy_ship_support_cap,
            deterministic=deterministic,
        )

    def _discrete_actor_log_prob(
        self,
        hidden: torch.Tensor,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        actions: ModelActions,
    ) -> tuple[ModelActionLogProbs, ModelActionEntropies]:
        if self.discrete_actor is None:
            raise RuntimeError("discrete actor module is not initialized")
        _require_discrete_actions_shape(
            actions,
            (
                hidden.shape[0],
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                self.action_spec.max_per_planet_launches,
            ),
        )
        slot_input = self._actor_inputs(hidden, max_launch)
        log_probs, entropies = self.discrete_actor.log_prob(
            slot_input,
            can_act,
            max_launch,
            actions,
            min_fleet_size=self.action_spec.min_fleet_size,
            max_ship_support=self.config.entropy_ship_support_cap,
        )
        return log_probs, entropies


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


class DiscreteTargetActor(nn.Module):
    def __init__(self, config: StatelessTransformerV1Config) -> None:
        super().__init__()
        self.config = config
        self.actor_config = cast(ActorDiscreteTargetsConfig, config.actor)
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        mixtures = config.n_action_mixtures

        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.q = nn.Linear(config.embed_dim, config.embed_dim)
        self.k = nn.Linear(config.embed_dim, config.embed_dim)
        self.v = nn.Linear(config.embed_dim, config.embed_dim)
        self.out = nn.Linear(config.embed_dim, config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = FeedForward(config)
        self.source_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.continue_head = nn.Linear(config.embed_dim, 1)
        self.mix_head = nn.Linear(config.embed_dim, mixtures)
        self.mean_head = nn.Linear(config.embed_dim, mixtures)
        self.log_scale_head = nn.Linear(config.embed_dim, mixtures)

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
        max_ship_support: int,
        deterministic: bool,
    ) -> tuple[ModelActions, ModelActionLogProbs, ModelActionEntropies]:
        selection = self._selection_params(slot_input, can_act)
        source_active = can_act.any(dim=-1) & (max_launch >= min_fleet_size)
        launch = StatelessTransformerV1._sample_launch(
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
        angle = torch.zeros_like(ships, dtype=slot_input.dtype).float()

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
            max_ship_support=max_ship_support,
        )
        per_player_entity_log_prob = launch_log_prob + target_log_prob + size_log_prob
        per_player_entity_entropy = launch_entropy + target_entropy + size_entropy

        launch_tensor = launch.unsqueeze(-1)
        target_tensor = target.unsqueeze(-1)
        ship_tensor = ships.unsqueeze(-1)
        angle_tensor = angle.unsqueeze(-1)
        return (
            ModelActions(
                launch=launch_tensor,
                angle=angle_tensor,
                target=target_tensor,
                ships=ship_tensor,
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
        max_ship_support: int,
    ) -> tuple[ModelActionLogProbs, ModelActionEntropies]:
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
            max_ship_support=max_ship_support,
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
            self.actor_config.min_log_scale,
            self.actor_config.max_log_scale,
        )
        scale = self.actor_config.scale_min + residual_budget * raw_log_scale.exp()
        return DiscreteTargetSizeParams(
            size_mix_logits=self.mix_head(source_hidden),
            size_mu=mu,
            size_scale=scale,
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: StatelessTransformerV1Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None,
        packed: PackedSequence | None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), token_mask, packed)
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
        self.force_flash_attn = config.force_flash_attn
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.q = nn.Linear(config.embed_dim, config.embed_dim)
        self.k = nn.Linear(config.embed_dim, config.embed_dim)
        self.v = nn.Linear(config.embed_dim, config.embed_dim)
        self.out = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None,
        packed: PackedSequence | None,
    ) -> torch.Tensor:
        if packed is not None:
            seq_len, _ = x.shape
            q = self.q(x).view(seq_len, self.n_heads, self.head_dim)
            k = self.k(x).view(seq_len, self.n_heads, self.head_dim)
            v = self.v(x).view(seq_len, self.n_heads, self.head_dim)
            if self.force_flash_attn and not use_flash_attn(q):
                raise RuntimeError(
                    "force_flash_attn=True requires CUDA fp16/bf16 attention "
                    "projections and the flash-attn package"
                )
            attn = varlen_attention(
                q,
                k,
                v,
                cu_seqlens=packed.cu_seqlens,
                max_seqlen=packed.max_seqlen,
            )
            return self.out(attn.reshape(seq_len, -1))

        if token_mask is None:
            raise RuntimeError("unpacked attention requires a token mask")
        batch_size, seq_len, _ = x.shape
        q = self.q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=token_mask[:, None, None, :],
            dropout_p=0.0,
        )
        attn = attn.transpose(1, 2)
        return self.out(attn.reshape(batch_size, seq_len, -1))


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
        mixtures = config.n_action_mixtures
        self.continue_head = nn.Linear(config.embed_dim, 1)
        self.mix_head = nn.Linear(config.embed_dim, mixtures)
        self.dir_head = nn.Linear(config.embed_dim, mixtures * 2)
        self.kappa_head = nn.Linear(config.embed_dim, mixtures)
        self.size_frac_head = nn.Linear(config.embed_dim, mixtures)
        self.size_conc_head = nn.Linear(config.embed_dim, mixtures)

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


def build_packed_sequence(token_mask: torch.Tensor) -> PackedSequence:
    batch_size, padded_seq_len = token_mask.shape
    flat_mask = token_mask.reshape(-1)
    indices = flat_mask.nonzero(as_tuple=False).flatten()
    seqlens = token_mask.sum(dim=1, dtype=torch.int32)
    if not seqlens.gt(0).all():
        raise ValueError("each batch row must have at least one unmasked token")
    return PackedSequence(
        indices=indices,
        cu_seqlens=F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)),
        seqlens=seqlens,
        max_seqlen=int(seqlens.max().item()),
        batch_size=batch_size,
        padded_seq_len=padded_seq_len,
    )


def pack_tensor(x: torch.Tensor, packed: PackedSequence) -> torch.Tensor:
    return x.reshape(packed.batch_size * packed.padded_seq_len, *x.shape[2:])[
        packed.indices
    ]


def pack_sequence(
    x: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, PackedSequence]:
    packed = build_packed_sequence(token_mask)
    return pack_tensor(x, packed), packed


def unpack_sequence(x: torch.Tensor, packed: PackedSequence) -> torch.Tensor:
    out = torch.zeros(
        (
            packed.batch_size * packed.padded_seq_len,
            *x.shape[1:],
        ),
        dtype=x.dtype,
        device=x.device,
    )
    out[packed.indices] = x
    return out.view(packed.batch_size, packed.padded_seq_len, *x.shape[1:])


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked_logits, dim=dim)


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


def binary_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    return F.binary_cross_entropy_with_logits(logits, probability, reduction="none")


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


def _require_actions_shape(
    actions: ModelActions,
    expected_shape: tuple[int, int, int, int],
) -> None:
    for name, tensor in (
        ("launch", actions.launch),
        ("angle", actions.angle),
        ("target", actions.target),
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
    if actions.target.dtype != torch.int64:
        raise ValueError(
            f"actions.target must have dtype torch.int64, got {actions.target.dtype}"
        )
    if actions.ships.dtype != torch.int64:
        raise ValueError(
            f"actions.ships must have dtype torch.int64, got {actions.ships.dtype}"
        )


def _require_discrete_actions_shape(
    actions: ModelActions,
    expected_shape: tuple[int, int, int, int],
) -> None:
    for name, tensor in (
        ("launch", actions.launch),
        ("angle", actions.angle),
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
    if actions.angle.dtype != torch.float32:
        raise ValueError(
            f"actions.angle must have dtype torch.float32, got {actions.angle.dtype}"
        )
    if actions.target.dtype != torch.int64:
        raise ValueError(
            f"actions.target must have dtype torch.int64, got {actions.target.dtype}"
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
