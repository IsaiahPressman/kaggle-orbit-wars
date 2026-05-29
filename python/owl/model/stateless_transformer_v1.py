from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self, TypeVar, assert_never, cast

import torch
import torch.nn.functional as F
from pydantic import Field, model_validator
from torch import nn

from owl.config import BaseConfig
from owl.model.actor import (
    ActorConfig,
    ActorDiscreteTargetBinsConfig,
    ActorDiscreteTargetsConfig,
    ActorPureConfig,
    DiscreteTargetBinsActor,
    DiscreteTargetsActor,
    PureActor,
)
from owl.model.actor.common import (
    FeedForward,
    OutputProjectionMLP,
    binary_entropy_from_logits,
    sample_launch,
)
from owl.model.actor.discrete_targets import (
    DiscreteActorInputs,
    DiscreteTargetPolicyParams,
    DiscreteTargetSizeParams,
    discrete_action_entropy,
    discretized_logistic_mixture_log_prob,
)
from owl.model.actor.pure import (
    PolicyParams,
    PureActorInputs,
    event_entropy_from_params,
    masked_action_entropy_from_params,
    masked_event_log_prob_from_params,
    sample_angle_mixture,
)
from owl.model.attn import use_flash_attn, varlen_attention
from owl.model.base import (
    BaseModelAPI,
    InputLayer,
    ModelActionEntropies,
    ModelActionKLDivergences,
    ModelActionLogProbs,
    ModelActions,
    ModelEvaluation,
    ModelHiddenState,
    ModelOutput,
    ModelServingOutput,
    ModelTeacherEvaluation,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    OUTER_PLAYER_SLOTS,
    ActionBundle,
    ActionConfig,
    ActionDiscreteTargetBinsConfig,
    ActionDiscreteTargetsConfig,
    ActionMask,
    ActionPureConfig,
    DiscreteTargetActionMask,
    DiscreteTargetActions,
    DiscreteTargetBinActionMask,
    DiscreteTargetBinActions,
    EntityBasedBaseConfig,
    ObsBatch,
    PureActionMask,
    PureActions,
)

__all__ = [
    "STATELESS_TRANSFORMER_V1",
    "ActorDiscreteTargetBinsConfig",
    "ActorDiscreteTargetsConfig",
    "ActorPureConfig",
    "DiscreteActorInputs",
    "DiscreteTargetBinsActor",
    "DiscreteTargetPolicyParams",
    "DiscreteTargetSizeParams",
    "DiscreteTargetsActor",
    "EncodedObservations",
    "FeedForward",
    "MultiHeadSelfAttention",
    "OutputProjectionMLP",
    "PackedSequence",
    "PairwiseBiasMLP",
    "PolicyParams",
    "PureActor",
    "PureActorInputs",
    "StatelessTransformerV1",
    "StatelessTransformerV1Config",
    "binary_entropy_from_logits",
    "build_packed_sequence",
    "build_pairwise_action_features",
    "discrete_action_entropy",
    "discretized_logistic_mixture_log_prob",
    "event_entropy_from_params",
    "masked_action_entropy_from_params",
    "masked_event_log_prob_from_params",
    "masked_softmax",
    "pack_sequence",
    "pack_tensor",
    "sample_angle_mixture",
    "sample_launch",
    "unpack_sequence",
]

STATELESS_TRANSFORMER_V1: Literal["stateless_transformer_v1"] = (
    "stateless_transformer_v1"
)
_HIDDEN_INIT_GAIN = math.sqrt(2.0)
_INPUT_INIT_GAIN = 1.0
_ACTOR_HEAD_INIT_GAIN = 0.01
_CRITIC_HEAD_INIT_GAIN = 1.0
_PAIRWISE_FEATURE_DIM = 6
_NORMALIZED_BOARD_DIAGONAL = math.sqrt(8.0)
_PLANET_NEUTRAL_OWNER = 4
_PLANET_X = 5
_PLANET_Y = 6
_PLANET_NEUTRAL_SHIPS = 13
_PLANET_OWNED_SHIPS = 15
_COMET_SHIPS = 5
_COMET_X = 52
_COMET_Y = 53
_NEUTRAL_SHIP_NORMALIZER = 100.0
_SHIP_NORMALIZER = 500.0
_PLAYER_COUNT_ADAPTER_COUNTS = tuple(range(2, OUTER_PLAYER_SLOTS + 1))

_ActorInputs = DiscreteActorInputs | PureActorInputs
_MIN_PLAYER_COUNT_ADAPTER_COUNT = _PLAYER_COUNT_ADAPTER_COUNTS[0]
_T = TypeVar("_T")


class StatelessTransformerV1Config(BaseConfig):
    model_arch: Literal["stateless_transformer_v1"] = STATELESS_TRANSFORMER_V1
    embed_dim: int = Field(default=128, ge=1)
    depth: int = Field(default=4, ge=1)
    n_heads: int = Field(default=8, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    player_count_adapters_enabled: bool = False
    player_count_adapter_blocks: int = Field(default=0, ge=0)
    n_scratch_tokens: int = Field(default=4, ge=0)
    activation: Literal["gelu", "silu", "swiglu"] = "gelu"
    force_flash_attn: bool = False
    use_learned_pairwise_bias: bool = False
    actor: ActorConfig = Field(default_factory=ActorPureConfig)

    @classmethod
    def subconfig_dirs(cls) -> set[str]:
        return {"actor"}

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.embed_dim % self.n_heads != 0:
            raise ValueError("n_heads must evenly divide embed_dim")
        if int(self.embed_dim * self.mlp_ratio) < 1:
            raise ValueError("embed_dim * mlp_ratio must be at least 1")
        if (
            not self.player_count_adapters_enabled
            and self.player_count_adapter_blocks != 0
        ):
            raise ValueError(
                "player_count_adapter_blocks requires "
                "player_count_adapters_enabled=True"
            )
        if self.player_count_adapter_blocks > self.depth:
            raise ValueError(
                "player_count_adapter_blocks must be less than or equal to depth"
            )
        return self


@dataclass(frozen=True)
class PackedSequence:
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    seqlens: torch.Tensor
    max_seqlen: int
    batch_size: int
    padded_seq_len: int


@dataclass(frozen=True)
class EncodedObservations:
    hidden: torch.Tensor
    token_mask: torch.Tensor
    action_entity_hidden: torch.Tensor
    player_hidden: torch.Tensor
    global_feature_hidden: torch.Tensor
    board_hidden: torch.Tensor
    actor_plan_hidden: torch.Tensor
    critic_value_hidden: torch.Tensor


@dataclass(frozen=True)
class _ValueDistillationOutput:
    values: torch.Tensor
    winner_probabilities: torch.Tensor
    winner_log_probabilities: torch.Tensor


class StatelessTransformerV1(BaseModelAPI):
    def __init__(
        self,
        config: StatelessTransformerV1Config,
        *,
        obs_spec: EntityBasedBaseConfig,
        action_spec: ActionConfig,
    ) -> None:
        super().__init__()
        if config.actor.action_spec != action_spec.action_spec:
            raise ValueError("model actor config must match env action_spec")
        if config.use_learned_pairwise_bias and isinstance(
            action_spec, ActionPureConfig
        ):
            raise ValueError(
                "use_learned_pairwise_bias requires a discrete target action_spec"
            )
        if (
            isinstance(action_spec, ActionDiscreteTargetsConfig)
            and action_spec.max_per_planet_launches != 1
        ):
            raise ValueError(
                "discrete_targets actor requires max_per_planet_launches=1"
            )
        if (
            isinstance(action_spec, ActionPureConfig)
            and action_spec.max_per_planet_launches != 1
        ):
            raise ValueError("pure actor requires max_per_planet_launches=1")
        if isinstance(action_spec, ActionDiscreteTargetBinsConfig):
            actor_config = cast(ActorDiscreteTargetBinsConfig, config.actor)
            if actor_config.n_bins != action_spec.n_bins:
                raise ValueError("model actor n_bins must match env action_spec n_bins")
        if obs_spec.uses_cross_attention and config.force_flash_attn:
            raise ValueError(
                "entity_based_cross_attn_v1 uses padded SDPA and does not support "
                "force_flash_attn=True"
            )
        if obs_spec.uses_cross_attention and config.player_count_adapter_blocks != 0:
            raise ValueError(
                "entity_based_cross_attn_v1 does not support "
                "player_count_adapter_blocks"
            )
        self.config = config
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        dim = self.config.embed_dim
        self.static_planet_proj = ObservationInputStem(
            self.obs_spec.planet_channels,
            self.config,
        )
        self.orbit_planet_proj = ObservationInputStem(
            self.obs_spec.planet_channels,
            self.config,
        )
        self.fleet_proj = ObservationInputStem(
            self.obs_spec.fleet_channels, self.config
        )
        self.comet_proj = ObservationInputStem(
            self.obs_spec.comet_channels, self.config
        )
        self.global_proj = ObservationInputStem(
            self.obs_spec.global_channels,
            self.config,
        )
        self.player_feature_proj = (
            None
            if self.obs_spec.player_feature_channels == 0
            else ObservationInputStem(
                self.obs_spec.player_feature_channels,
                self.config,
            )
        )
        self.target_incoming_proj = (
            None
            if not self.obs_spec.uses_cross_attention
            else ObservationInputStem(
                self.obs_spec.target_incoming_channels,
                self.config,
            )
        )
        self.player_tokens = nn.Parameter(torch.empty(OUTER_PLAYER_SLOTS, dim))
        self.board_tokens = nn.Parameter(torch.empty(self.config.n_scratch_tokens, dim))
        self.actor_plan_tokens = nn.Parameter(torch.empty(OUTER_PLAYER_SLOTS, dim))
        self.critic_value_tokens = nn.Parameter(torch.empty(OUTER_PLAYER_SLOTS, dim))

        shared_depth = self.config.depth - (
            self.config.player_count_adapter_blocks
            if self.config.player_count_adapters_enabled
            else 0
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(shared_depth)
        )
        self.fleet_cross_attn = nn.ModuleList(
            PlanetFleetCrossAttention(self.config)
            for _ in range(shared_depth if self.obs_spec.uses_cross_attention else 0)
        )
        self.fleet_residual_mlps = nn.ModuleList(
            FeedForward(self.config)
            for _ in range(shared_depth if self.obs_spec.uses_cross_attention else 0)
        )
        self.fleet_residual_norms = nn.ModuleList(
            nn.LayerNorm(dim)
            for _ in range(shared_depth if self.obs_spec.uses_cross_attention else 0)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.player_count_adapters = nn.ModuleDict()

        self.critic_head: OutputProjectionMLP | None = None
        self.pairwise_bias_mlp: PairwiseBiasMLP | None = None
        self.source_actor_input_proj: nn.Linear | None = None
        self.target_actor_input_proj: nn.Linear | None = None
        self.actor: (
            PureActor | DiscreteTargetsActor | DiscreteTargetBinsActor | None
        ) = None
        if not self.config.player_count_adapters_enabled:
            self.critic_head = OutputProjectionMLP(self.config, 1)
            self.pairwise_bias_mlp = (
                PairwiseBiasMLP(self.config)
                if self.config.use_learned_pairwise_bias
                else None
            )
            (
                self.source_actor_input_proj,
                self.target_actor_input_proj,
                self.actor,
            ) = _build_actor_modules(self.config, action_spec)
        else:
            for player_count in _PLAYER_COUNT_ADAPTER_COUNTS:
                self.player_count_adapters[str(player_count)] = PlayerCountAdapter(
                    self.config,
                    action_spec,
                )

    def reset_parameters(self) -> None:
        self.apply(_init_module)
        for layer in self.get_input_layers():
            _init_input_layer(layer)
        if isinstance(self.actor, PureActor):
            self.actor.reset_base_dirs()
        for adapter in self.player_count_adapters.values():
            adapter = cast(PlayerCountAdapter, adapter)
            if isinstance(adapter.actor, PureActor):
                adapter.actor.reset_base_dirs()
        residual_gain = 1.0 / math.sqrt(2.0 * self.config.depth)
        for module in self.blocks:
            block = cast(TransformerBlock, module)
            _init_linear(block.attn.out, gain=residual_gain)
            _init_linear(block.mlp.down, gain=residual_gain)
        for module in self.fleet_cross_attn:
            cross_attn = cast(PlanetFleetCrossAttention, module)
            _init_linear(cross_attn.attn.out, gain=residual_gain)
        for module in self.fleet_residual_mlps:
            mlp = cast(FeedForward, module)
            _init_linear(mlp.down, gain=residual_gain)
        for adapter in self.player_count_adapters.values():
            adapter = cast(PlayerCountAdapter, adapter)
            for module in adapter.blocks:
                block = cast(TransformerBlock, module)
                _init_linear(block.attn.out, gain=residual_gain)
                _init_linear(block.mlp.down, gain=residual_gain)
        critic_output_layer_ids = {
            id(adapter.critic_head.out)
            for adapter in cast(
                list[PlayerCountAdapter],
                list(self.player_count_adapters.values()),
            )
        }
        if self.critic_head is not None:
            critic_output_layer_ids.add(id(self.critic_head.out))
        for layer in self.get_output_layers():
            gain = (
                _CRITIC_HEAD_INIT_GAIN
                if id(layer) in critic_output_layer_ids
                else _ACTOR_HEAD_INIT_GAIN
            )
            _init_linear(layer, gain=gain)

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        head_input_layers: tuple[InputLayer, ...]
        if self.player_count_adapters:
            head_input_layers = tuple(
                layer
                for adapter in self.player_count_adapters.values()
                for layer in cast(PlayerCountAdapter, adapter).get_input_layers()
            )
        else:
            actor = self._shared_actor()
            head_input_layers = (
                *(
                    ()
                    if self.pairwise_bias_mlp is None
                    else self.pairwise_bias_mlp.get_input_layers()
                ),
                *actor.get_input_layers(),
            )
        return (
            self.static_planet_proj.input,
            self.orbit_planet_proj.input,
            self.fleet_proj.input,
            self.comet_proj.input,
            self.global_proj.input,
            *(
                ()
                if self.player_feature_proj is None
                else (self.player_feature_proj.input,)
            ),
            *(
                ()
                if self.target_incoming_proj is None
                else (self.target_incoming_proj.input,)
            ),
            self.player_tokens,
            self.board_tokens,
            self.actor_plan_tokens,
            self.critic_value_tokens,
            *head_input_layers,
        )

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        if self.player_count_adapters:
            return tuple(
                layer
                for adapter in self.player_count_adapters.values()
                for layer in cast(PlayerCountAdapter, adapter).get_output_layers()
            )
        if self.critic_head is None:
            raise RuntimeError("shared critic head is not initialized")
        actor = self._shared_actor()
        return (
            self.critic_head.out,
            *(
                ()
                if self.pairwise_bias_mlp is None
                else self.pairwise_bias_mlp.get_output_layers()
            ),
            *actor.get_output_layers(),
        )

    def encode_observations(
        self,
        obs: ObsBatch,
        *,
        action_entity_slots: int = ACTION_ENTITY_SLOTS,
    ) -> EncodedObservations:
        global_x = self.global_proj(obs.global_features)
        global_token = global_x.unsqueeze(1)
        orbiting = obs.orbiting_planets.unsqueeze(-1)
        planet_x = torch.where(
            orbiting,
            self.orbit_planet_proj(obs.planets),
            self.static_planet_proj(obs.planets),
        )
        fleet_x = self.fleet_proj(obs.fleets)
        comet_x = self.comet_proj(obs.comets)
        action_entity_x = torch.cat((planet_x, comet_x), dim=1)
        cross_attention = self.obs_spec.uses_cross_attention
        if cross_attention:
            fleet_target, target_incoming_features = (
                _require_cross_attention_observation_tensors(
                    obs,
                    action_entity_slots=action_entity_slots,
                )
            )
            if self.target_incoming_proj is None:
                raise RuntimeError("target incoming projection is not initialized")
            action_entity_x = action_entity_x + self.target_incoming_proj(
                target_incoming_features
            )
            fleet_mask = obs.entity_mask[:, action_entity_slots:]
            if fleet_mask.shape[1] != fleet_x.shape[1]:
                raise ValueError(
                    "cross-attention fleet mask length must match fleets; got "
                    f"{fleet_mask.shape[1]} and {fleet_x.shape[1]}"
                )
            fleet_x = fleet_x.masked_fill(~fleet_mask.unsqueeze(-1), 0.0)
        else:
            if obs.fleet_target is not None or obs.target_incoming_features is not None:
                raise ValueError(
                    "fleet_target and target_incoming_features require "
                    "entity_based_cross_attn_v1"
                )
            fleet_target = None
            fleet_mask = None
        batch_size = obs.planets.shape[0]
        player_tokens = _expand_tokens(
            self.player_tokens,
            batch_size,
            dtype=global_token.dtype,
        )
        if self.player_feature_proj is not None:
            if obs.player_features is None:
                raise ValueError("player_features are required by this obs_spec")
            player_tokens = player_tokens + self.player_feature_proj(
                obs.player_features
            )
        board_tokens = _expand_tokens(
            self.board_tokens,
            batch_size,
            dtype=global_token.dtype,
        )
        actor_plan_tokens = _expand_tokens(
            self.actor_plan_tokens,
            batch_size,
            dtype=global_token.dtype,
        )
        critic_value_tokens = _expand_tokens(
            self.critic_value_tokens,
            batch_size,
            dtype=global_token.dtype,
        )

        always_on_mask = torch.ones(
            (batch_size, 1 + self.config.n_scratch_tokens),
            dtype=torch.bool,
            device=obs.entity_mask.device,
        )
        if cross_attention:
            action_entity_mask = obs.entity_mask[:, :action_entity_slots]
            token_mask = torch.cat(
                (
                    action_entity_mask,
                    obs.still_playing,
                    always_on_mask,
                    obs.still_playing,
                    obs.still_playing,
                ),
                dim=1,
            )
            x = torch.cat(
                (
                    action_entity_x,
                    player_tokens,
                    global_token,
                    board_tokens,
                    actor_plan_tokens,
                    critic_value_tokens,
                ),
                dim=1,
            )
            cross_attn_mask = _build_planet_fleet_cross_attention_mask(
                obs,
                fleet_target=cast(torch.Tensor, fleet_target),
                fleet_mask=cast(torch.Tensor, fleet_mask),
                action_entity_slots=action_entity_slots,
                query_token_count=x.shape[1],
                n_scratch_tokens=self.config.n_scratch_tokens,
            )
        else:
            token_mask = torch.cat(
                (
                    obs.entity_mask,
                    obs.still_playing,
                    always_on_mask,
                    obs.still_playing,
                    obs.still_playing,
                ),
                dim=1,
            )
            x = torch.cat(
                (
                    action_entity_x,
                    fleet_x,
                    player_tokens,
                    global_token,
                    board_tokens,
                    actor_plan_tokens,
                    critic_value_tokens,
                ),
                dim=1,
            )
            cross_attn_mask = None
        packed: PackedSequence | None
        should_use_flash = use_flash_attn(x) and not cross_attention
        if (
            _requires_flash_attn(x, force_flash_attn=self.config.force_flash_attn)
            and not should_use_flash
        ):
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
        if cross_attention:
            if block_token_mask is None or packed is not None:
                raise RuntimeError("cross-attention observations require padded masks")
            for block, cross_attn, fleet_norm, fleet_mlp in zip(
                self.blocks,
                self.fleet_cross_attn,
                self.fleet_residual_norms,
                self.fleet_residual_mlps,
                strict=True,
            ):
                block = cast(TransformerBlock, block)
                x = x + block.attn(block.norm1(x), block_token_mask, None)
                x = x + cross_attn(
                    x,
                    fleet_x,
                    cast(torch.Tensor, cross_attn_mask),
                    cast(torch.Tensor, fleet_mask),
                )
                x = x + block.mlp(block.norm2(x))
                fleet_x = fleet_x + fleet_mlp(fleet_norm(fleet_x))
                fleet_x = fleet_x.masked_fill(
                    ~cast(torch.Tensor, fleet_mask).unsqueeze(-1),
                    0.0,
                )
        else:
            for block in self.blocks:
                x = block(x, block_token_mask, packed)
        x = self._apply_player_count_adapter_blocks(
            x,
            token_mask,
            packed,
            obs.still_playing,
        )
        x = self.final_norm(x)
        if packed is not None:
            x = unpack_sequence(x, packed)
        x = x.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        entity_count = (
            action_entity_slots if cross_attention else obs.entity_mask.shape[1]
        )
        player_start = entity_count
        global_start = player_start + OUTER_PLAYER_SLOTS
        board_start = global_start + 1
        actor_plan_start = board_start + self.config.n_scratch_tokens
        critic_value_start = actor_plan_start + OUTER_PLAYER_SLOTS
        return EncodedObservations(
            hidden=x,
            token_mask=token_mask,
            action_entity_hidden=x[:, :action_entity_slots, :],
            player_hidden=x[:, player_start:global_start, :],
            global_feature_hidden=x[:, global_start : global_start + 1, :],
            board_hidden=x[:, board_start:actor_plan_start, :],
            actor_plan_hidden=x[:, actor_plan_start:critic_value_start, :],
            critic_value_hidden=x[
                :,
                critic_value_start : critic_value_start + OUTER_PLAYER_SLOTS,
                :,
            ],
        )

    def _apply_player_count_adapter_blocks(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        packed: PackedSequence | None,
        still_playing: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.player_count_adapters
            or self.config.player_count_adapter_blocks == 0
        ):
            return x

        out = x
        for player_count, batch_indices in self._player_count_index_groups(
            still_playing
        ):
            adapter = self._player_count_adapter(player_count)
            if packed is None:
                x_indices = batch_indices.to(device=x.device)
                branch_x = x.index_select(0, x_indices)
                branch_token_mask = token_mask.index_select(
                    0,
                    batch_indices.to(device=token_mask.device),
                )
                for block in adapter.blocks:
                    branch_x = block(branch_x, branch_token_mask, None)
                out = out.index_copy(0, x_indices, branch_x)
                continue

            packed_positions, branch_packed = _packed_subset_for_batch_indices(
                packed,
                batch_indices,
            )
            packed_positions = packed_positions.to(device=x.device)
            branch_x = x.index_select(0, packed_positions)
            for block in adapter.blocks:
                branch_x = block(branch_x, None, branch_packed)
            out = out.index_copy(0, packed_positions, branch_x)
        return out

    def _player_count_index_groups(
        self,
        still_playing: torch.Tensor,
    ) -> list[tuple[int, torch.Tensor]]:
        alive_counts = still_playing.sum(dim=1)
        if not alive_counts.gt(0).all():
            raise ValueError(
                "still_playing must include at least one player per batch row"
            )

        adapter_counts = alive_counts.clamp_min(_MIN_PLAYER_COUNT_ADAPTER_COUNT)
        groups: list[tuple[int, torch.Tensor]] = []
        grouped_rows = 0
        for player_count in _PLAYER_COUNT_ADAPTER_COUNTS:
            batch_indices = (
                (adapter_counts == player_count).nonzero(as_tuple=False).flatten()
            )
            if batch_indices.numel() == 0:
                continue
            groups.append((player_count, batch_indices))
            grouped_rows += batch_indices.numel()
        if grouped_rows != still_playing.shape[0]:
            raise ValueError(
                "player-count adapters support one to four still-playing "
                "players per batch row"
            )
        return groups

    def _player_count_adapter(self, player_count: int) -> PlayerCountAdapter:
        key = str(player_count)
        if key not in self.player_count_adapters:
            raise RuntimeError(
                f"player-count adapter {player_count}p is not initialized"
            )
        return cast(PlayerCountAdapter, self.player_count_adapters[key])

    def _shared_actor(
        self,
    ) -> PureActor | DiscreteTargetsActor | DiscreteTargetBinsActor:
        if self.actor is None:
            raise RuntimeError("shared actor is not initialized")
        return self.actor

    def _critic_head(
        self,
        adapter: PlayerCountAdapter | None,
    ) -> nn.Module:
        if adapter is not None:
            return adapter.critic_head
        if self.critic_head is None:
            raise RuntimeError("shared critic head is not initialized")
        return self.critic_head

    def _actor_module(
        self,
        adapter: PlayerCountAdapter | None,
    ) -> PureActor | DiscreteTargetsActor | DiscreteTargetBinsActor:
        if adapter is not None:
            return adapter.actor
        return self._shared_actor()

    def _actor_input_projections(
        self,
        adapter: PlayerCountAdapter | None,
    ) -> tuple[nn.Linear, nn.Linear]:
        if adapter is not None:
            return adapter.source_actor_input_proj, adapter.target_actor_input_proj
        if self.source_actor_input_proj is None or self.target_actor_input_proj is None:
            raise RuntimeError("actor input projections are not initialized")
        return self.source_actor_input_proj, self.target_actor_input_proj

    def _pairwise_bias_head(
        self,
        adapter: PlayerCountAdapter | None,
    ) -> PairwiseBiasMLP | None:
        if adapter is not None:
            return adapter.pairwise_bias_mlp
        return self.pairwise_bias_mlp

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
        hidden_state: ModelHiddenState | None = None,
    ) -> ModelOutput:
        if hidden_state is not None:
            raise ValueError("StatelessTransformerV1 does not accept hidden_state")
        encoded = self.encode_observations(
            obs,
            action_entity_slots=_action_entity_slots_from_mask(obs.action_mask),
        )
        values, winner_probabilities = self._value_from_encoded(encoded, obs)
        actions, log_probs, entropies = self._actor(
            encoded,
            obs,
            obs.action_mask,
            deterministic=deterministic,
        )
        return ModelOutput(
            actions=actions,
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=winner_probabilities,
        )

    def serve(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
        hidden_state: ModelHiddenState | None = None,
    ) -> ModelServingOutput:
        if hidden_state is not None:
            raise ValueError("StatelessTransformerV1 does not accept hidden_state")
        encoded = self.encode_observations(
            obs,
            action_entity_slots=_action_entity_slots_from_mask(obs.action_mask),
        )
        values, winner_probabilities = self._value_from_encoded(encoded, obs)
        actions = self._actor_actions(
            encoded,
            obs,
            obs.action_mask,
            deterministic=deterministic,
        )
        return ModelServingOutput(
            actions=actions,
            values=values,
            winner_probabilities=winner_probabilities,
        )

    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ModelActions,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
    ) -> ModelEvaluation:
        if hidden_state is not None:
            raise ValueError("StatelessTransformerV1 does not accept hidden_state")
        if dones is not None:
            raise ValueError("StatelessTransformerV1 does not accept dones")
        encoded = self.encode_observations(
            obs,
            action_entity_slots=_action_entity_slots_from_mask(obs.action_mask),
        )
        values, winner_probabilities = self._value_from_encoded(encoded, obs)
        log_probs, entropies = self._actor_log_prob(
            encoded,
            obs,
            obs.action_mask,
            actions,
        )
        return ModelEvaluation(
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=winner_probabilities,
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
        if not isinstance(teacher, StatelessTransformerV1):
            raise ValueError(
                "teacher must be a StatelessTransformerV1-compatible model"
            )
        flat_obs, sequence_shape = _flatten_obs_time_if_sequence(obs)
        flat_actions = _flatten_actions_time_if_sequence(actions, sequence_shape)
        teacher_batch_size = (
            flat_obs.planets.shape[0] if sequence_shape is None else sequence_shape[0]
        )
        if (
            teacher.initial_hidden_state(
                teacher_batch_size,
                device=flat_obs.planets.device,
            )
            is not None
        ):
            raise ValueError(
                "teacher models with recurrent hidden state are not supported"
            )

        student_encoded, student_next_state = self._encode_distillation_observations(
            flat_obs,
            sequence_shape=sequence_shape,
            hidden_state=hidden_state,
            dones=dones,
        )
        student_winner_log_probabilities: torch.Tensor | None = None
        if compute_teacher_value:
            student_value_output = self._value_distillation_from_encoded(
                student_encoded,
                flat_obs,
            )
            student_values = student_value_output.values
            student_winner_probabilities = student_value_output.winner_probabilities
            student_winner_log_probabilities = (
                student_value_output.winner_log_probabilities
            )
        else:
            student_values, student_winner_probabilities = self._value_from_encoded(
                student_encoded,
                flat_obs,
            )
        student_actor_inputs = (
            self._actor_inputs_for_encoded(student_encoded, flat_obs)
            if compute_teacher_action_kl and not self.player_count_adapters
            else None
        )
        student_log_probs, student_entropies = self._actor_log_prob(
            student_encoded,
            flat_obs,
            flat_obs.action_mask,
            flat_actions,
            actor_inputs=student_actor_inputs,
        )
        action_kl: ModelActionKLDivergences | None = None
        teacher_winner_probabilities: torch.Tensor | None = None
        if compute_teacher_action_kl or compute_teacher_value:
            with torch.no_grad():
                teacher_encoded, _teacher_next_state = (
                    teacher._encode_distillation_observations(
                        flat_obs,
                        sequence_shape=sequence_shape,
                        hidden_state=None,
                        dones=dones,
                    )
                )
                if compute_teacher_value:
                    _teacher_values, teacher_winner_probabilities = (
                        teacher._value_from_encoded(
                            teacher_encoded,
                            flat_obs,
                        )
                    )
            if compute_teacher_action_kl:
                action_kl = self._actor_kl_divergence(
                    teacher,
                    student_encoded,
                    teacher_encoded,
                    flat_obs,
                    flat_obs.action_mask,
                    flat_actions,
                    student_actor_inputs=student_actor_inputs,
                )
        student = ModelEvaluation(
            log_probs=student_log_probs,
            entropies=student_entropies,
            values=student_values,
            winner_probabilities=student_winner_probabilities,
            next_hidden_state=student_next_state,
        )
        if sequence_shape is not None:
            student = _unflatten_evaluation(student, sequence_shape)
            if teacher_winner_probabilities is not None:
                teacher_winner_probabilities = _unflatten_time_tensor(
                    teacher_winner_probabilities,
                    sequence_shape,
                )
            if student_winner_log_probabilities is not None:
                student_winner_log_probabilities = _unflatten_time_tensor(
                    student_winner_log_probabilities,
                    sequence_shape,
                )
            if action_kl is not None:
                action_kl = _unflatten_kl_divergences(action_kl, sequence_shape)
        return ModelTeacherEvaluation(
            student=student,
            action_kl=action_kl,
            teacher_winner_probabilities=teacher_winner_probabilities,
            student_winner_log_probabilities=student_winner_log_probabilities,
        )

    def evaluate_action_kl(
        self,
        obs: ObsBatch,
        teacher: BaseModelAPI,
        actions: ModelActions,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
    ) -> ModelActionKLDivergences:
        if not isinstance(teacher, StatelessTransformerV1):
            raise ValueError(
                "teacher must be a StatelessTransformerV1-compatible model"
            )
        flat_obs, sequence_shape = _flatten_obs_time_if_sequence(obs)
        flat_actions = _flatten_actions_time_if_sequence(actions, sequence_shape)
        teacher_batch_size = (
            flat_obs.planets.shape[0] if sequence_shape is None else sequence_shape[0]
        )
        if (
            teacher.initial_hidden_state(
                teacher_batch_size,
                device=flat_obs.planets.device,
            )
            is not None
        ):
            raise ValueError(
                "teacher models with recurrent hidden state are not supported"
            )
        student_encoded, _student_next_state = self._encode_distillation_observations(
            flat_obs,
            sequence_shape=sequence_shape,
            hidden_state=hidden_state,
            dones=dones,
        )
        with torch.no_grad():
            teacher_encoded, _teacher_next_state = (
                teacher._encode_distillation_observations(
                    flat_obs,
                    sequence_shape=sequence_shape,
                    hidden_state=None,
                    dones=dones,
                )
            )
        kl = self._actor_kl_divergence(
            teacher,
            student_encoded,
            teacher_encoded,
            flat_obs,
            flat_obs.action_mask,
            flat_actions,
        )
        if sequence_shape is not None:
            return _unflatten_kl_divergences(kl, sequence_shape)
        return kl

    def _encode_distillation_observations(
        self,
        obs: ObsBatch,
        *,
        sequence_shape: tuple[int, int] | None,
        hidden_state: ModelHiddenState | None,
        dones: torch.Tensor | None,
    ) -> tuple[EncodedObservations, ModelHiddenState | None]:
        if hidden_state is not None:
            raise ValueError("StatelessTransformerV1 does not accept hidden_state")
        if dones is not None and sequence_shape is None:
            raise ValueError("dones require sequence-shaped observations")
        encoded = self.encode_observations(
            obs,
            action_entity_slots=_action_entity_slots_from_mask(obs.action_mask),
        )
        return encoded, None

    def compute_value(
        self,
        obs: ObsBatch,
        *,
        hidden_state: ModelHiddenState | None = None,
    ) -> torch.Tensor:
        if hidden_state is not None:
            raise ValueError("StatelessTransformerV1 does not accept hidden_state")
        encoded = self.encode_observations(
            obs,
            action_entity_slots=_action_entity_slots_from_mask(obs.action_mask),
        )
        values, _winner_probabilities = self._value_from_encoded(encoded, obs)
        return values

    def _value_from_encoded(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.player_count_adapters:
            return self._value_by_player_count(encoded, obs)
        return self._critic(encoded.critic_value_hidden, obs.still_playing)

    def _value_distillation_from_encoded(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
    ) -> _ValueDistillationOutput:
        if self.player_count_adapters:
            return self._value_distillation_by_player_count(encoded, obs)
        return self._critic_distillation(
            encoded.critic_value_hidden,
            obs.still_playing,
        )

    def _value_by_player_count(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
        probability_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
        for player_count, batch_indices in self._player_count_index_groups(
            obs.still_playing
        ):
            adapter = self._player_count_adapter(player_count)
            values, probabilities = self._critic(
                _batch_select(encoded.critic_value_hidden, batch_indices),
                _batch_select(obs.still_playing, batch_indices),
                adapter=adapter,
            )
            value_parts.append((batch_indices, values))
            probability_parts.append((batch_indices, probabilities))

        batch_size = obs.still_playing.shape[0]
        values = _merge_tensors_by_batch(batch_size, value_parts)
        probabilities = _merge_tensors_by_batch(batch_size, probability_parts)
        return values, probabilities

    def _value_distillation_by_player_count(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
    ) -> _ValueDistillationOutput:
        value_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
        probability_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
        log_probability_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
        for player_count, batch_indices in self._player_count_index_groups(
            obs.still_playing
        ):
            adapter = self._player_count_adapter(player_count)
            output = self._critic_distillation(
                _batch_select(encoded.critic_value_hidden, batch_indices),
                _batch_select(obs.still_playing, batch_indices),
                adapter=adapter,
            )
            value_parts.append((batch_indices, output.values))
            probability_parts.append((batch_indices, output.winner_probabilities))
            log_probability_parts.append(
                (batch_indices, output.winner_log_probabilities)
            )

        batch_size = obs.still_playing.shape[0]
        return _ValueDistillationOutput(
            values=_merge_tensors_by_batch(batch_size, value_parts),
            winner_probabilities=_merge_tensors_by_batch(
                batch_size,
                probability_parts,
            ),
            winner_log_probabilities=_merge_tensors_by_batch(
                batch_size,
                log_probability_parts,
            ),
        )

    def _critic(
        self,
        player_hidden: torch.Tensor,
        still_playing: torch.Tensor,
        *,
        adapter: PlayerCountAdapter | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._critic_logits(player_hidden, still_playing, adapter=adapter)
        probabilities = masked_softmax(logits, still_playing, dim=-1)
        values = 2.0 * probabilities - 1.0
        return values, probabilities

    def _critic_distillation(
        self,
        player_hidden: torch.Tensor,
        still_playing: torch.Tensor,
        *,
        adapter: PlayerCountAdapter | None = None,
    ) -> _ValueDistillationOutput:
        logits = self._critic_logits(player_hidden, still_playing, adapter=adapter)
        masked_logits = logits.masked_fill(
            ~still_playing,
            torch.finfo(logits.dtype).min,
        )
        winner_log_probabilities = F.log_softmax(masked_logits, dim=-1)
        winner_probabilities = winner_log_probabilities.exp()
        return _ValueDistillationOutput(
            values=2.0 * winner_probabilities - 1.0,
            winner_probabilities=winner_probabilities,
            winner_log_probabilities=winner_log_probabilities,
        )

    def _critic_logits(
        self,
        player_hidden: torch.Tensor,
        still_playing: torch.Tensor,
        *,
        adapter: PlayerCountAdapter | None = None,
    ) -> torch.Tensor:
        if still_playing.shape != player_hidden.shape[:2]:
            raise ValueError(
                "still_playing must have shape "
                f"{tuple(player_hidden.shape[:2])}, got {tuple(still_playing.shape)}"
            )
        if not still_playing.any(dim=1).all():
            raise ValueError(
                "still_playing must include at least one player per batch row"
            )

        return self._critic_head(adapter)(player_hidden).squeeze(-1)

    def _pure_actor_inputs(
        self,
        encoded: EncodedObservations,
        *,
        adapter: PlayerCountAdapter | None = None,
    ) -> PureActorInputs:
        action_entity_hidden = encoded.action_entity_hidden
        action_entity_slots = action_entity_hidden.shape[1]
        entity_features = action_entity_hidden[:, None, :, :].expand(
            -1,
            OUTER_PLAYER_SLOTS,
            -1,
            -1,
        )
        player_features = encoded.player_hidden[:, :, None, :].expand(
            -1,
            -1,
            action_entity_slots,
            -1,
        )
        plan_features = encoded.actor_plan_hidden[:, :, None, :].expand(
            -1,
            -1,
            action_entity_slots,
            -1,
        )
        source_actor_input_proj, target_actor_input_proj = (
            self._actor_input_projections(adapter)
        )
        source = source_actor_input_proj(
            torch.cat((entity_features, player_features, plan_features), dim=-1)
        )
        target = target_actor_input_proj(
            torch.cat((entity_features, player_features, plan_features), dim=-1)
        )
        return PureActorInputs(
            source=source,
            target=target,
            target_mask=encoded.token_mask[:, :action_entity_slots],
        )

    def _discrete_actor_inputs(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
        *,
        adapter: PlayerCountAdapter | None = None,
    ) -> DiscreteActorInputs:
        action_entity_hidden = encoded.action_entity_hidden
        action_entity_slots = action_entity_hidden.shape[1]
        entity_features = action_entity_hidden[:, None, :, :].expand(
            -1,
            OUTER_PLAYER_SLOTS,
            -1,
            -1,
        )
        player_features = encoded.player_hidden[:, :, None, :].expand(
            -1,
            -1,
            action_entity_slots,
            -1,
        )
        plan_features = encoded.actor_plan_hidden[:, :, None, :].expand(
            -1,
            -1,
            action_entity_slots,
            -1,
        )
        source_actor_input_proj, target_actor_input_proj = (
            self._actor_input_projections(adapter)
        )
        source = source_actor_input_proj(
            torch.cat((entity_features, player_features, plan_features), dim=-1)
        )
        target = target_actor_input_proj(
            torch.cat((entity_features, player_features, plan_features), dim=-1)
        )
        pairwise_bias: torch.Tensor | None = None
        pairwise_bias_mlp = self._pairwise_bias_head(adapter)
        if pairwise_bias_mlp is not None:
            pairwise_bias = pairwise_bias_mlp(build_pairwise_action_features(obs))
            pairwise_bias = pairwise_bias[:, None, :, :].expand(
                -1,
                OUTER_PLAYER_SLOTS,
                -1,
                -1,
            )
        return DiscreteActorInputs(
            source=source,
            target=target,
            pairwise_bias=pairwise_bias,
        )

    def _actor(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
        action_mask: ActionMask,
        *,
        deterministic: bool,
        adapter: PlayerCountAdapter | None = None,
    ) -> tuple[ActionBundle, ModelActionLogProbs, ModelActionEntropies]:
        if adapter is None and self.player_count_adapters:
            return self._actor_by_player_count(
                encoded,
                obs,
                deterministic=deterministic,
            )

        actor = self._actor_module(adapter)
        if isinstance(actor, DiscreteTargetBinsActor):
            if not isinstance(action_mask, DiscreteTargetBinActionMask):
                raise RuntimeError(
                    "discrete_target_bins actor requires a target-bin action mask"
                )
            return actor(
                self._discrete_actor_inputs(encoded, obs, adapter=adapter),
                action_mask.can_act,
                deterministic=deterministic,
            )
        if isinstance(actor, DiscreteTargetsActor):
            if not isinstance(action_mask, DiscreteTargetActionMask):
                raise RuntimeError(
                    "discrete_targets actor requires a discrete-target action mask"
                )
            return actor(
                self._discrete_actor_inputs(encoded, obs, adapter=adapter),
                action_mask.can_act,
                action_mask.max_launch,
                min_fleet_size=self.action_spec.min_fleet_size,
                deterministic=deterministic,
            )
        if not isinstance(action_mask, PureActionMask):
            raise RuntimeError("pure actor requires a pure action mask")
        return actor(
            self._pure_actor_inputs(encoded, adapter=adapter),
            action_mask.can_act,
            action_mask.max_launch,
            min_fleet_size=self.action_spec.min_fleet_size,
            deterministic=deterministic,
        )

    def _actor_by_player_count(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
        *,
        deterministic: bool,
    ) -> tuple[ActionBundle, ModelActionLogProbs, ModelActionEntropies]:
        action_parts: list[tuple[torch.Tensor, ActionBundle]] = []
        log_prob_parts: list[tuple[torch.Tensor, ModelActionLogProbs]] = []
        entropy_parts: list[tuple[torch.Tensor, ModelActionEntropies]] = []
        for player_count, batch_indices in self._player_count_index_groups(
            obs.still_playing
        ):
            adapter = self._player_count_adapter(player_count)
            indexed_encoded = _index_encoded_observations(encoded, batch_indices)
            indexed_obs = _index_obs_batch(obs, batch_indices)
            actions, log_probs, entropies = self._actor(
                indexed_encoded,
                indexed_obs,
                indexed_obs.action_mask,
                deterministic=deterministic,
                adapter=adapter,
            )
            action_parts.append((batch_indices, actions))
            log_prob_parts.append((batch_indices, log_probs))
            entropy_parts.append((batch_indices, entropies))
        batch_size = obs.still_playing.shape[0]
        return (
            _merge_action_bundles_by_batch(batch_size, action_parts),
            _merge_log_probs_by_batch(batch_size, log_prob_parts),
            _merge_entropies_by_batch(batch_size, entropy_parts),
        )

    def _actor_actions(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
        action_mask: ActionMask,
        *,
        deterministic: bool,
        adapter: PlayerCountAdapter | None = None,
    ) -> ActionBundle:
        if adapter is None and self.player_count_adapters:
            return self._actor_actions_by_player_count(
                encoded,
                obs,
                deterministic=deterministic,
            )

        actor = self._actor_module(adapter)
        if isinstance(actor, DiscreteTargetsActor):
            if not isinstance(action_mask, DiscreteTargetActionMask):
                raise RuntimeError(
                    "discrete_targets actor requires a discrete-target action mask"
                )
            return actor.sample_actions(
                self._discrete_actor_inputs(encoded, obs, adapter=adapter),
                action_mask.can_act,
                action_mask.max_launch,
                min_fleet_size=self.action_spec.min_fleet_size,
                deterministic=deterministic,
            )
        if isinstance(actor, DiscreteTargetBinsActor):
            if not isinstance(action_mask, DiscreteTargetBinActionMask):
                raise RuntimeError(
                    "discrete_target_bins actor requires a target-bin action mask"
                )
            return actor.sample_actions(
                self._discrete_actor_inputs(encoded, obs, adapter=adapter),
                action_mask.can_act,
                deterministic=deterministic,
            )

        actions, _log_probs, _entropies = self._actor(
            encoded,
            obs,
            action_mask,
            deterministic=deterministic,
            adapter=adapter,
        )
        return actions

    def _actor_actions_by_player_count(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
        *,
        deterministic: bool,
    ) -> ActionBundle:
        action_parts: list[tuple[torch.Tensor, ActionBundle]] = []
        for player_count, batch_indices in self._player_count_index_groups(
            obs.still_playing
        ):
            adapter = self._player_count_adapter(player_count)
            indexed_encoded = _index_encoded_observations(encoded, batch_indices)
            indexed_obs = _index_obs_batch(obs, batch_indices)
            actions = self._actor_actions(
                indexed_encoded,
                indexed_obs,
                indexed_obs.action_mask,
                deterministic=deterministic,
                adapter=adapter,
            )
            action_parts.append((batch_indices, actions))
        return _merge_action_bundles_by_batch(obs.still_playing.shape[0], action_parts)

    def _actor_log_prob(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
        action_mask: ActionMask,
        actions: ActionBundle,
        *,
        adapter: PlayerCountAdapter | None = None,
        actor_inputs: _ActorInputs | None = None,
    ) -> tuple[ModelActionLogProbs, ModelActionEntropies]:
        if adapter is None and self.player_count_adapters:
            return self._actor_log_prob_by_player_count(encoded, obs, actions)

        actor = self._actor_module(adapter)
        if isinstance(actor, DiscreteTargetBinsActor):
            if not isinstance(action_mask, DiscreteTargetBinActionMask):
                raise RuntimeError(
                    "discrete_target_bins actor requires a target-bin action mask"
                )
            if not isinstance(actions, DiscreteTargetBinActions):
                raise ValueError(
                    "discrete_target_bins actor requires DiscreteTargetBinActions"
                )
            discrete_inputs = _require_discrete_actor_inputs(
                actor_inputs,
                lambda: self._discrete_actor_inputs(encoded, obs, adapter=adapter),
            )
            return actor.log_prob(
                discrete_inputs,
                action_mask.can_act,
                actions,
            )
        if isinstance(actor, DiscreteTargetsActor):
            if not isinstance(action_mask, DiscreteTargetActionMask):
                raise RuntimeError(
                    "discrete_targets actor requires a discrete-target action mask"
                )
            if not isinstance(actions, DiscreteTargetActions):
                raise ValueError(
                    "discrete_targets actor requires DiscreteTargetActions"
                )
            discrete_inputs = _require_discrete_actor_inputs(
                actor_inputs,
                lambda: self._discrete_actor_inputs(encoded, obs, adapter=adapter),
            )
            return actor.log_prob(
                discrete_inputs,
                action_mask.can_act,
                action_mask.max_launch,
                actions,
                min_fleet_size=self.action_spec.min_fleet_size,
            )
        if not isinstance(action_mask, PureActionMask):
            raise RuntimeError("pure actor requires a pure action mask")
        if not isinstance(actions, PureActions):
            raise ValueError("pure actor requires PureActions")
        pure_inputs = _require_pure_actor_inputs(
            actor_inputs,
            lambda: self._pure_actor_inputs(encoded, adapter=adapter),
        )
        return actor.log_prob(
            pure_inputs,
            action_mask.can_act,
            action_mask.max_launch,
            actions,
            min_fleet_size=self.action_spec.min_fleet_size,
        )

    def _actor_inputs_for_encoded(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
        *,
        adapter: PlayerCountAdapter | None = None,
    ) -> _ActorInputs:
        actor = self._actor_module(adapter)
        if isinstance(actor, DiscreteTargetBinsActor | DiscreteTargetsActor):
            return self._discrete_actor_inputs(encoded, obs, adapter=adapter)
        return self._pure_actor_inputs(encoded, adapter=adapter)

    def _actor_log_prob_by_player_count(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
        actions: ActionBundle,
    ) -> tuple[ModelActionLogProbs, ModelActionEntropies]:
        log_prob_parts: list[tuple[torch.Tensor, ModelActionLogProbs]] = []
        entropy_parts: list[tuple[torch.Tensor, ModelActionEntropies]] = []
        for player_count, batch_indices in self._player_count_index_groups(
            obs.still_playing
        ):
            adapter = self._player_count_adapter(player_count)
            indexed_encoded = _index_encoded_observations(encoded, batch_indices)
            indexed_obs = _index_obs_batch(obs, batch_indices)
            indexed_actions = _map_action_bundle(
                actions,
                _batch_selector(batch_indices),
            )
            log_probs, entropies = self._actor_log_prob(
                indexed_encoded,
                indexed_obs,
                indexed_obs.action_mask,
                indexed_actions,
                adapter=adapter,
            )
            log_prob_parts.append((batch_indices, log_probs))
            entropy_parts.append((batch_indices, entropies))
        batch_size = obs.still_playing.shape[0]
        return (
            _merge_log_probs_by_batch(batch_size, log_prob_parts),
            _merge_entropies_by_batch(batch_size, entropy_parts),
        )

    def _actor_kl_divergence(
        self,
        teacher: StatelessTransformerV1,
        student_encoded: EncodedObservations,
        teacher_encoded: EncodedObservations,
        obs: ObsBatch,
        action_mask: ActionMask,
        actions: ActionBundle,
        *,
        student_adapter: PlayerCountAdapter | None = None,
        teacher_adapter: PlayerCountAdapter | None = None,
        student_actor_inputs: _ActorInputs | None = None,
    ) -> ModelActionKLDivergences:
        if (
            student_adapter is None
            and teacher_adapter is None
            and (self.player_count_adapters or teacher.player_count_adapters)
        ):
            return self._actor_kl_divergence_by_player_count(
                teacher,
                student_encoded,
                teacher_encoded,
                obs,
                actions,
            )

        student_actor = self._actor_module(student_adapter)
        teacher_actor = teacher._actor_module(teacher_adapter)
        if isinstance(student_actor, DiscreteTargetBinsActor):
            if not isinstance(teacher_actor, DiscreteTargetBinsActor):
                raise ValueError("teacher action actor must match student actor")
            if not isinstance(action_mask, DiscreteTargetBinActionMask):
                raise RuntimeError(
                    "discrete_target_bins actor requires a target-bin action mask"
                )
            if not isinstance(actions, DiscreteTargetBinActions):
                raise ValueError(
                    "discrete_target_bins actor requires DiscreteTargetBinActions"
                )
            student_inputs = _require_discrete_actor_inputs(
                student_actor_inputs,
                lambda: self._discrete_actor_inputs(
                    student_encoded,
                    obs,
                    adapter=student_adapter,
                ),
            )
            with torch.no_grad():
                teacher_inputs = teacher._discrete_actor_inputs(
                    teacher_encoded,
                    obs,
                    adapter=teacher_adapter,
                )
            return student_actor.kl_divergence(
                student_inputs,
                teacher_actor,
                teacher_inputs,
                action_mask.can_act,
                actions,
            )
        if isinstance(student_actor, DiscreteTargetsActor):
            if not isinstance(teacher_actor, DiscreteTargetsActor):
                raise ValueError("teacher action actor must match student actor")
            if not isinstance(action_mask, DiscreteTargetActionMask):
                raise RuntimeError(
                    "discrete_targets actor requires a discrete-target action mask"
                )
            if not isinstance(actions, DiscreteTargetActions):
                raise ValueError(
                    "discrete_targets actor requires DiscreteTargetActions"
                )
            student_inputs = _require_discrete_actor_inputs(
                student_actor_inputs,
                lambda: self._discrete_actor_inputs(
                    student_encoded,
                    obs,
                    adapter=student_adapter,
                ),
            )
            with torch.no_grad():
                teacher_inputs = teacher._discrete_actor_inputs(
                    teacher_encoded,
                    obs,
                    adapter=teacher_adapter,
                )
            return student_actor.kl_divergence(
                student_inputs,
                teacher_actor,
                teacher_inputs,
                action_mask.can_act,
                action_mask.max_launch,
                actions,
                min_fleet_size=self.action_spec.min_fleet_size,
            )
        if not isinstance(teacher_actor, PureActor):
            raise ValueError("teacher action actor must match student actor")
        if not isinstance(action_mask, PureActionMask):
            raise RuntimeError("pure actor requires a pure action mask")
        if not isinstance(actions, PureActions):
            raise ValueError("pure actor requires PureActions")
        student_pure_inputs = _require_pure_actor_inputs(
            student_actor_inputs,
            lambda: self._pure_actor_inputs(
                student_encoded,
                adapter=student_adapter,
            ),
        )
        with torch.no_grad():
            teacher_pure_inputs = teacher._pure_actor_inputs(
                teacher_encoded,
                adapter=teacher_adapter,
            )
        return student_actor.kl_divergence(
            student_pure_inputs,
            teacher_actor,
            teacher_pure_inputs,
            action_mask.can_act,
            action_mask.max_launch,
            actions,
            min_fleet_size=self.action_spec.min_fleet_size,
        )

    def _actor_kl_divergence_by_player_count(
        self,
        teacher: StatelessTransformerV1,
        student_encoded: EncodedObservations,
        teacher_encoded: EncodedObservations,
        obs: ObsBatch,
        actions: ActionBundle,
    ) -> ModelActionKLDivergences:
        kl_parts: list[tuple[torch.Tensor, ModelActionKLDivergences]] = []
        for player_count, batch_indices in self._player_count_index_groups(
            obs.still_playing
        ):
            student_adapter = (
                self._player_count_adapter(player_count)
                if self.player_count_adapters
                else None
            )
            teacher_adapter = (
                teacher._player_count_adapter(player_count)
                if teacher.player_count_adapters
                else None
            )
            indexed_obs = _index_obs_batch(obs, batch_indices)
            indexed_actions = _map_action_bundle(
                actions,
                _batch_selector(batch_indices),
            )
            kl = self._actor_kl_divergence(
                teacher,
                _index_encoded_observations(student_encoded, batch_indices),
                _index_encoded_observations(teacher_encoded, batch_indices),
                indexed_obs,
                indexed_obs.action_mask,
                indexed_actions,
                student_adapter=student_adapter,
                teacher_adapter=teacher_adapter,
            )
            kl_parts.append((batch_indices, kl))
        return _merge_kl_divergences_by_batch(obs.still_playing.shape[0], kl_parts)


def _require_discrete_actor_inputs(
    actor_inputs: _ActorInputs | None,
    fallback: Callable[[], DiscreteActorInputs],
) -> DiscreteActorInputs:
    if actor_inputs is None:
        return fallback()
    if not isinstance(actor_inputs, DiscreteActorInputs):
        raise TypeError("expected discrete actor inputs")
    return actor_inputs


def _require_pure_actor_inputs(
    actor_inputs: _ActorInputs | None,
    fallback: Callable[[], PureActorInputs],
) -> PureActorInputs:
    if actor_inputs is None:
        return fallback()
    if not isinstance(actor_inputs, PureActorInputs):
        raise TypeError("expected pure actor inputs")
    return actor_inputs


class PlayerCountAdapter(nn.Module):
    def __init__(
        self,
        config: StatelessTransformerV1Config,
        action_spec: ActionConfig,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.player_count_adapter_blocks)
        )
        self.critic_head = OutputProjectionMLP(config, 1)
        self.pairwise_bias_mlp: PairwiseBiasMLP | None = (
            PairwiseBiasMLP(config) if config.use_learned_pairwise_bias else None
        )
        (
            self.source_actor_input_proj,
            self.target_actor_input_proj,
            self.actor,
        ) = _build_actor_modules(config, action_spec)

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        return (
            *(
                ()
                if self.pairwise_bias_mlp is None
                else self.pairwise_bias_mlp.get_input_layers()
            ),
            *self.actor.get_input_layers(),
        )

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        return (
            self.critic_head.out,
            *(
                ()
                if self.pairwise_bias_mlp is None
                else self.pairwise_bias_mlp.get_output_layers()
            ),
            *self.actor.get_output_layers(),
        )


def _build_actor_modules(
    config: StatelessTransformerV1Config,
    action_spec: ActionConfig,
) -> tuple[
    nn.Linear,
    nn.Linear,
    PureActor | DiscreteTargetsActor | DiscreteTargetBinsActor,
]:
    dim = config.embed_dim
    source_actor_input_proj = nn.Linear(dim * 3, dim)
    target_actor_input_proj = nn.Linear(dim * 3, dim)
    if isinstance(action_spec, ActionPureConfig):
        actor: PureActor | DiscreteTargetsActor | DiscreteTargetBinsActor = PureActor(
            cast(ActorPureConfig, config.actor),
            embed_dim=dim,
            max_per_planet_launches=action_spec.max_per_planet_launches,
            activation=config.activation,
        )
    elif isinstance(action_spec, ActionDiscreteTargetsConfig):
        actor = DiscreteTargetsActor(
            cast(ActorDiscreteTargetsConfig, config.actor),
            transformer_config=config,
        )
    else:
        actor = DiscreteTargetBinsActor(
            cast(ActorDiscreteTargetBinsConfig, config.actor),
            transformer_config=config,
        )
    return source_actor_input_proj, target_actor_input_proj, actor


def _batch_select(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tensor.index_select(0, indices.to(device=tensor.device))


def _batch_selector(indices: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def select(tensor: torch.Tensor) -> torch.Tensor:
        return _batch_select(tensor, indices)

    return select


def _index_encoded_observations(
    encoded: EncodedObservations,
    indices: torch.Tensor,
) -> EncodedObservations:
    return EncodedObservations(
        hidden=_batch_select(encoded.hidden, indices),
        token_mask=_batch_select(encoded.token_mask, indices),
        action_entity_hidden=_batch_select(encoded.action_entity_hidden, indices),
        player_hidden=_batch_select(encoded.player_hidden, indices),
        global_feature_hidden=_batch_select(encoded.global_feature_hidden, indices),
        board_hidden=_batch_select(encoded.board_hidden, indices),
        actor_plan_hidden=_batch_select(encoded.actor_plan_hidden, indices),
        critic_value_hidden=_batch_select(encoded.critic_value_hidden, indices),
    )


def _index_obs_batch(obs: ObsBatch, indices: torch.Tensor) -> ObsBatch:
    return ObsBatch(
        planets=_batch_select(obs.planets, indices),
        orbiting_planets=_batch_select(obs.orbiting_planets, indices),
        fleets=_batch_select(obs.fleets, indices),
        fleet_target=(
            None
            if obs.fleet_target is None
            else _batch_select(obs.fleet_target, indices)
        ),
        target_incoming_features=(
            None
            if obs.target_incoming_features is None
            else _batch_select(obs.target_incoming_features, indices)
        ),
        comets=_batch_select(obs.comets, indices),
        entity_mask=_batch_select(obs.entity_mask, indices),
        still_playing=_batch_select(obs.still_playing, indices),
        global_features=_batch_select(obs.global_features, indices),
        action_mask=_map_action_mask(
            obs.action_mask,
            _batch_selector(indices),
        ),
        player_features=(
            None
            if obs.player_features is None
            else _batch_select(obs.player_features, indices)
        ),
    )


def _map_action_mask(
    action_mask: ActionMask,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> ActionMask:
    if isinstance(action_mask, PureActionMask):
        return PureActionMask(
            can_act=fn(action_mask.can_act),
            max_launch=fn(action_mask.max_launch),
        )
    if isinstance(action_mask, DiscreteTargetActionMask):
        return DiscreteTargetActionMask(
            can_act=fn(action_mask.can_act),
            max_launch=fn(action_mask.max_launch),
        )
    if isinstance(action_mask, DiscreteTargetBinActionMask):
        return DiscreteTargetBinActionMask(
            can_act=fn(action_mask.can_act),
        )
    assert_never(action_mask)


def _map_action_bundle(
    actions: ActionBundle,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> ActionBundle:
    if isinstance(actions, PureActions):
        return PureActions(
            launch=fn(actions.launch),
            angle=fn(actions.angle),
            ships=fn(actions.ships),
        )
    if isinstance(actions, DiscreteTargetActions):
        return DiscreteTargetActions(
            launch=fn(actions.launch),
            target=fn(actions.target),
            ships=fn(actions.ships),
        )
    if isinstance(actions, DiscreteTargetBinActions):
        return DiscreteTargetBinActions(
            target=fn(actions.target),
            fleet_bin=fn(actions.fleet_bin),
        )
    assert_never(actions)


def _flatten_obs_time_if_sequence(
    obs: ObsBatch,
) -> tuple[ObsBatch, tuple[int, int] | None]:
    if obs.planets.ndim == 3:
        return obs, None
    if obs.planets.ndim != 4:
        raise ValueError("obs planets must be batch-major or segment-major")
    batch_size, time_steps = obs.planets.shape[:2]
    return (
        ObsBatch(
            planets=_flatten_time_tensor(obs.planets),
            orbiting_planets=_flatten_time_tensor(obs.orbiting_planets),
            fleets=_flatten_time_tensor(obs.fleets),
            fleet_target=(
                None
                if obs.fleet_target is None
                else _flatten_time_tensor(obs.fleet_target)
            ),
            target_incoming_features=(
                None
                if obs.target_incoming_features is None
                else _flatten_time_tensor(obs.target_incoming_features)
            ),
            comets=_flatten_time_tensor(obs.comets),
            entity_mask=_flatten_time_tensor(obs.entity_mask),
            still_playing=_flatten_time_tensor(obs.still_playing),
            global_features=_flatten_time_tensor(obs.global_features),
            action_mask=_map_action_mask(obs.action_mask, _flatten_time_tensor),
            player_features=(
                None
                if obs.player_features is None
                else _flatten_time_tensor(obs.player_features)
            ),
        ),
        (batch_size, time_steps),
    )


def _flatten_actions_time_if_sequence(
    actions: ModelActions,
    sequence_shape: tuple[int, int] | None,
) -> ModelActions:
    if sequence_shape is None:
        return actions
    return _map_action_bundle(actions, _flatten_time_tensor)


def _flatten_time_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _unflatten_time_tensor(
    tensor: torch.Tensor,
    sequence_shape: tuple[int, int],
) -> torch.Tensor:
    batch_size, time_steps = sequence_shape
    return tensor.reshape(batch_size, time_steps, *tensor.shape[1:])


def _merge_tensors_by_batch(
    batch_size: int,
    parts: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if not parts:
        raise ValueError("cannot merge empty player-count outputs")
    first = parts[0][1]
    out = first.new_zeros((batch_size, *first.shape[1:]))
    for indices, tensor in parts:
        out = out.index_copy(0, indices.to(device=out.device), tensor)
    return out


def _merge_tensor_field_by_batch(
    batch_size: int,
    parts: list[tuple[torch.Tensor, _T]],
    field: Callable[[_T], torch.Tensor],
) -> torch.Tensor:
    return _merge_tensors_by_batch(
        batch_size,
        [(indices, field(value)) for indices, value in parts],
    )


def _merge_optional_tensor_field_by_batch(
    batch_size: int,
    parts: list[tuple[torch.Tensor, _T]],
    field: Callable[[_T], torch.Tensor | None],
    *,
    name: str,
) -> torch.Tensor | None:
    first = field(parts[0][1])
    if first is None:
        return None

    tensor_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
    for indices, value in parts:
        tensor = field(value)
        if tensor is None:
            raise RuntimeError(f"expected {name} tensor")
        tensor_parts.append((indices, tensor))
    return _merge_tensors_by_batch(batch_size, tensor_parts)


def _merge_action_bundles_by_batch(
    batch_size: int,
    parts: list[tuple[torch.Tensor, ActionBundle]],
) -> ActionBundle:
    if not parts:
        raise ValueError("cannot merge empty player-count actions")
    first = parts[0][1]
    if isinstance(first, PureActions):
        pure_parts = [
            (indices, cast(PureActions, actions)) for indices, actions in parts
        ]
        return PureActions(
            launch=_merge_tensor_field_by_batch(
                batch_size, pure_parts, lambda actions: actions.launch
            ),
            angle=_merge_tensor_field_by_batch(
                batch_size, pure_parts, lambda actions: actions.angle
            ),
            ships=_merge_tensor_field_by_batch(
                batch_size, pure_parts, lambda actions: actions.ships
            ),
        )
    if isinstance(first, DiscreteTargetActions):
        discrete_parts = [
            (indices, cast(DiscreteTargetActions, actions))
            for indices, actions in parts
        ]
        return DiscreteTargetActions(
            launch=_merge_tensor_field_by_batch(
                batch_size, discrete_parts, lambda actions: actions.launch
            ),
            target=_merge_tensor_field_by_batch(
                batch_size, discrete_parts, lambda actions: actions.target
            ),
            ships=_merge_tensor_field_by_batch(
                batch_size, discrete_parts, lambda actions: actions.ships
            ),
        )
    bin_parts = [
        (indices, cast(DiscreteTargetBinActions, actions)) for indices, actions in parts
    ]
    return DiscreteTargetBinActions(
        target=_merge_tensor_field_by_batch(
            batch_size, bin_parts, lambda actions: actions.target
        ),
        fleet_bin=_merge_tensor_field_by_batch(
            batch_size, bin_parts, lambda actions: actions.fleet_bin
        ),
    )


def _merge_log_probs_by_batch(
    batch_size: int,
    parts: list[tuple[torch.Tensor, ModelActionLogProbs]],
) -> ModelActionLogProbs:
    target = _merge_optional_tensor_field_by_batch(
        batch_size,
        parts,
        lambda log_probs: log_probs.target,
        name="target log-prob",
    )
    return ModelActionLogProbs(
        launch=_merge_tensor_field_by_batch(
            batch_size, parts, lambda log_probs: log_probs.launch
        ),
        event=_merge_tensor_field_by_batch(
            batch_size, parts, lambda log_probs: log_probs.event
        ),
        per_player_entity=_merge_tensor_field_by_batch(
            batch_size, parts, lambda log_probs: log_probs.per_player_entity
        ),
        target=target,
    )


def _merge_entropies_by_batch(
    batch_size: int,
    parts: list[tuple[torch.Tensor, ModelActionEntropies]],
) -> ModelActionEntropies:
    def component_getter(
        name: str,
    ) -> Callable[[ModelActionEntropies], torch.Tensor]:
        def get_component(entropies: ModelActionEntropies) -> torch.Tensor:
            return entropies.components[name]

        return get_component

    first = parts[0][1]
    target = _merge_optional_tensor_field_by_batch(
        batch_size,
        parts,
        lambda entropies: entropies.target,
        name="target entropy",
    )
    components = {
        name: _merge_tensor_field_by_batch(
            batch_size,
            parts,
            component_getter(name),
        )
        for name in first.components
    }
    return ModelActionEntropies(
        launch=_merge_tensor_field_by_batch(
            batch_size, parts, lambda entropies: entropies.launch
        ),
        event=_merge_tensor_field_by_batch(
            batch_size, parts, lambda entropies: entropies.event
        ),
        per_player_entity=_merge_tensor_field_by_batch(
            batch_size, parts, lambda entropies: entropies.per_player_entity
        ),
        target=target,
        components=components,
    )


def _merge_kl_divergences_by_batch(
    batch_size: int,
    parts: list[tuple[torch.Tensor, ModelActionKLDivergences]],
) -> ModelActionKLDivergences:
    def component_getter(
        name: str,
    ) -> Callable[[ModelActionKLDivergences], torch.Tensor]:
        def get_component(kl: ModelActionKLDivergences) -> torch.Tensor:
            return kl.components[name]

        return get_component

    first = parts[0][1]
    target = _merge_optional_tensor_field_by_batch(
        batch_size,
        parts,
        lambda kl: kl.target,
        name="target KL",
    )
    components = {
        name: _merge_tensor_field_by_batch(
            batch_size,
            parts,
            component_getter(name),
        )
        for name in first.components
    }
    return ModelActionKLDivergences(
        launch=_merge_tensor_field_by_batch(
            batch_size,
            parts,
            lambda kl: kl.launch,
        ),
        event=_merge_tensor_field_by_batch(batch_size, parts, lambda kl: kl.event),
        per_player_entity=_merge_tensor_field_by_batch(
            batch_size,
            parts,
            lambda kl: kl.per_player_entity,
        ),
        target=target,
        components=components,
    )


def _unflatten_kl_divergences(
    kl: ModelActionKLDivergences,
    sequence_shape: tuple[int, int],
) -> ModelActionKLDivergences:
    return ModelActionKLDivergences(
        launch=_unflatten_time_tensor(kl.launch, sequence_shape),
        target=(
            None
            if kl.target is None
            else _unflatten_time_tensor(kl.target, sequence_shape)
        ),
        event=_unflatten_time_tensor(kl.event, sequence_shape),
        per_player_entity=_unflatten_time_tensor(
            kl.per_player_entity,
            sequence_shape,
        ),
        components={
            name: _unflatten_time_tensor(component, sequence_shape)
            for name, component in kl.components.items()
        },
    )


def _unflatten_evaluation(
    evaluation: ModelEvaluation,
    sequence_shape: tuple[int, int],
) -> ModelEvaluation:
    return ModelEvaluation(
        log_probs=_unflatten_log_probs(evaluation.log_probs, sequence_shape),
        entropies=_unflatten_entropies(evaluation.entropies, sequence_shape),
        values=_unflatten_time_tensor(evaluation.values, sequence_shape),
        winner_probabilities=_unflatten_time_tensor(
            evaluation.winner_probabilities,
            sequence_shape,
        ),
        next_hidden_state=evaluation.next_hidden_state,
    )


def _unflatten_log_probs(
    log_probs: ModelActionLogProbs,
    sequence_shape: tuple[int, int],
) -> ModelActionLogProbs:
    return ModelActionLogProbs(
        launch=_unflatten_time_tensor(log_probs.launch, sequence_shape),
        target=(
            None
            if log_probs.target is None
            else _unflatten_time_tensor(log_probs.target, sequence_shape)
        ),
        event=_unflatten_time_tensor(log_probs.event, sequence_shape),
        per_player_entity=_unflatten_time_tensor(
            log_probs.per_player_entity,
            sequence_shape,
        ),
    )


def _unflatten_entropies(
    entropies: ModelActionEntropies,
    sequence_shape: tuple[int, int],
) -> ModelActionEntropies:
    return ModelActionEntropies(
        launch=_unflatten_time_tensor(entropies.launch, sequence_shape),
        target=(
            None
            if entropies.target is None
            else _unflatten_time_tensor(entropies.target, sequence_shape)
        ),
        event=_unflatten_time_tensor(entropies.event, sequence_shape),
        per_player_entity=_unflatten_time_tensor(
            entropies.per_player_entity,
            sequence_shape,
        ),
        components={
            name: _unflatten_time_tensor(component, sequence_shape)
            for name, component in entropies.components.items()
        },
    )


class ObservationInputStem(nn.Module):
    def __init__(self, input_dim: int, config: StatelessTransformerV1Config) -> None:
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.activation = config.activation
        self.input = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, config.embed_dim)

    @property
    def in_features(self) -> int:
        return self.input.in_features

    @property
    def out_features(self) -> int:
        return self.output.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return self.output(F.gelu(self.input(x)))
        return self.output(F.silu(self.input(x)))


class PairwiseBiasMLP(nn.Module):
    def __init__(self, config: StatelessTransformerV1Config) -> None:
        super().__init__()
        self.activation = config.activation
        match self.activation:
            case "gelu" | "silu":
                self.up = nn.Linear(_PAIRWISE_FEATURE_DIM, config.embed_dim)
            case "swiglu":
                self.gate = nn.Linear(_PAIRWISE_FEATURE_DIM, config.embed_dim)
                self.value = nn.Linear(_PAIRWISE_FEATURE_DIM, config.embed_dim)
            case _:
                assert_never(self.activation)
        self.out = nn.Linear(config.embed_dim, 1)

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        match self.activation:
            case "gelu" | "silu":
                return (self.up,)
            case "swiglu":
                return (self.gate, self.value)
            case _:
                assert_never(self.activation)

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        return (self.out,)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != _PAIRWISE_FEATURE_DIM:
            raise ValueError(
                "pairwise features must have final dimension "
                f"{_PAIRWISE_FEATURE_DIM}, got {features.shape[-1]}"
            )
        match self.activation:
            case "gelu":
                hidden = F.gelu(self.up(features))
            case "silu":
                hidden = F.silu(self.up(features))
            case "swiglu":
                hidden = F.silu(self.gate(features)) * self.value(features)
            case _:
                assert_never(self.activation)
        return self.out(hidden).squeeze(-1)


def build_pairwise_action_features(obs: ObsBatch) -> torch.Tensor:
    # TODO: Move these pairwise features into the simulator observation contract
    # once they are no longer model-local, so Rust owns the channel layout.
    planet_owners = obs.planets[..., : _PLANET_NEUTRAL_OWNER + 1]
    comet_owners = obs.comets[..., : _PLANET_NEUTRAL_OWNER + 1]
    owners = torch.cat((planet_owners, comet_owners), dim=1)

    planet_neutral = obs.planets[..., _PLANET_NEUTRAL_OWNER]
    planet_ships = torch.where(
        planet_neutral.bool(),
        obs.planets[..., _PLANET_NEUTRAL_SHIPS] * _NEUTRAL_SHIP_NORMALIZER,
        obs.planets[..., _PLANET_OWNED_SHIPS] * _SHIP_NORMALIZER,
    )
    comet_ships = obs.comets[..., _COMET_SHIPS] * _SHIP_NORMALIZER
    ships = torch.cat((planet_ships, comet_ships), dim=1)

    planet_xy = obs.planets[..., (_PLANET_X, _PLANET_Y)]
    comet_xy = obs.comets[..., (_COMET_X, _COMET_Y)]
    xy = torch.cat((planet_xy, comet_xy), dim=1)

    source_ships = ships[:, :, None]
    target_ships = ships[:, None, :]
    has_more_ships = (source_ships > target_ships).to(dtype=obs.planets.dtype)

    neutral_owner = owners[..., _PLANET_NEUTRAL_OWNER]
    target_is_neutral = neutral_owner[:, None, :].expand_as(has_more_ships)

    player_owners = owners[..., :OUTER_PLAYER_SLOTS]
    source_player_owners = player_owners[:, :, None, :]
    target_player_owners = player_owners[:, None, :, :]
    target_is_mine = (source_player_owners * target_player_owners).sum(dim=-1)
    target_is_enemy = (target_player_owners.sum(dim=-1) - target_is_mine).clamp_min(0.0)

    source_xy = xy[:, :, None, :]
    target_xy = xy[:, None, :, :]
    segment = target_xy - source_xy
    distance = segment.norm(dim=-1)
    normalized_distance = (distance / _NORMALIZED_BOARD_DIAGONAL).clamp(0.0, 1.0)

    segment_len_sq = (segment * segment).sum(dim=-1)
    eps = torch.finfo(obs.planets.dtype).eps
    projection = -(source_xy * segment).sum(dim=-1) / segment_len_sq.clamp_min(eps)
    projection = projection.clamp(0.0, 1.0)
    closest_to_sun = source_xy + projection.unsqueeze(-1) * segment
    sun_distance = closest_to_sun.norm(dim=-1)
    sun_proximity = 1.0 - (sun_distance / math.sqrt(2.0)).clamp(0.0, 1.0)

    return torch.stack(
        (
            has_more_ships,
            target_is_neutral,
            target_is_mine,
            target_is_enemy,
            normalized_distance,
            sun_proximity,
        ),
        dim=-1,
    )


def _action_entity_slots_from_mask(action_mask: ActionMask) -> int:
    return action_mask.can_act.shape[2]


def _expand_tokens(
    tokens: torch.Tensor,
    batch_size: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return tokens.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)


def _require_cross_attention_observation_tensors(
    obs: ObsBatch,
    *,
    action_entity_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if obs.fleet_target is None:
        raise ValueError("fleet_target is required by entity_based_cross_attn_v1")
    if obs.target_incoming_features is None:
        raise ValueError(
            "target_incoming_features is required by entity_based_cross_attn_v1"
        )
    if obs.target_incoming_features.shape[:2] != (
        obs.planets.shape[0],
        action_entity_slots,
    ):
        raise ValueError(
            "target_incoming_features must have shape "
            f"(batch, {action_entity_slots}, channels), got "
            f"{tuple(obs.target_incoming_features.shape)}"
        )
    if obs.fleet_target.shape != obs.entity_mask[:, action_entity_slots:].shape:
        raise ValueError(
            "fleet_target must have shape matching the fleet mask tail, got "
            f"{tuple(obs.fleet_target.shape)} and "
            f"{tuple(obs.entity_mask[:, action_entity_slots:].shape)}"
        )
    return obs.fleet_target, obs.target_incoming_features


def _build_planet_fleet_cross_attention_mask(
    obs: ObsBatch,
    *,
    fleet_target: torch.Tensor,
    fleet_mask: torch.Tensor,
    action_entity_slots: int,
    query_token_count: int,
    n_scratch_tokens: int,
) -> torch.Tensor:
    invalid_targets = (
        (fleet_target < 0) | (fleet_target >= action_entity_slots)
    ) & fleet_mask
    if bool(invalid_targets.any().item()):
        raise ValueError(
            "fleet_target contains target indices outside the action entity axis"
        )

    target_slots = torch.arange(
        action_entity_slots,
        device=fleet_target.device,
        dtype=fleet_target.dtype,
    )
    action_entity_mask = (
        fleet_target[:, None, :] == target_slots[None, :, None]
    ) & fleet_mask[:, None, :]

    owner_mask = (obs.fleets[..., :OUTER_PLAYER_SLOTS] > 0.5).transpose(1, 2)
    owner_mask = owner_mask & fleet_mask[:, None, :]
    global_mask = fleet_mask[:, None, :]
    scratch_mask = global_mask.expand(-1, n_scratch_tokens, -1)
    cross_mask = torch.cat(
        (
            action_entity_mask,
            owner_mask,
            global_mask,
            scratch_mask,
            owner_mask,
            owner_mask,
        ),
        dim=1,
    )
    if cross_mask.shape[1] != query_token_count:
        raise RuntimeError(
            "cross-attention query mask shape does not match transformer tokens"
        )
    return cross_mask


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


class PlanetFleetCrossAttention(nn.Module):
    def __init__(self, config: StatelessTransformerV1Config) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.embed_dim)
        self.fleet_norm = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadCrossAttention(config)

    def forward(
        self,
        query: torch.Tensor,
        fleet: torch.Tensor,
        attn_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(fleet_mask.any().item()):
            return torch.zeros_like(query)
        return self.attn(
            self.query_norm(query),
            self.fleet_norm(fleet),
            attn_mask,
            fleet_mask,
        )


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, config: StatelessTransformerV1Config) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.q = nn.Linear(config.embed_dim, config.embed_dim)
        self.k = nn.Linear(config.embed_dim, config.embed_dim)
        self.v = nn.Linear(config.embed_dim, config.embed_dim)
        self.out = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        fleet_len = key_value.shape[1]
        q = self.q(query).view(batch_size, query_len, self.n_heads, self.head_dim)
        k = self.k(key_value).view(batch_size, fleet_len, self.n_heads, self.head_dim)
        v = self.v(key_value).view(batch_size, fleet_len, self.n_heads, self.head_dim)
        valid_query = attn_mask.any(dim=-1)
        fallback_mask = fleet_mask[:, None, :]
        no_fleet = ~fleet_mask.any(dim=-1)
        if bool(no_fleet.any().item()):
            fallback_mask = fallback_mask.expand(-1, query_len, -1).clone()
            fallback_mask[no_fleet, :, 0] = True
        safe_mask = torch.where(
            valid_query.unsqueeze(-1),
            attn_mask,
            fallback_mask,
        )
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=safe_mask[:, None, :, :],
            dropout_p=0.0,
        )
        attn = attn.transpose(1, 2).reshape(batch_size, query_len, -1)
        attn = self.out(attn)
        return attn.masked_fill(~valid_query.unsqueeze(-1), 0.0)


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
            if _requires_flash_attn(
                q, force_flash_attn=self.force_flash_attn
            ) and not use_flash_attn(q):
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


def _requires_flash_attn(
    tensor: torch.Tensor,
    *,
    force_flash_attn: bool,
) -> bool:
    return force_flash_attn and tensor.device.type != "cpu"


def _init_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        _init_linear(module, gain=_HIDDEN_INIT_GAIN)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _init_input_layer(module: InputLayer) -> None:
    if isinstance(module, nn.Linear):
        _init_linear(module, gain=_INPUT_INIT_GAIN)
    elif isinstance(module, nn.Parameter):
        _init_token_parameter(module)


def _init_token_parameter(parameter: nn.Parameter) -> None:
    nn.init.normal_(parameter, mean=0.0, std=parameter.shape[-1] ** -0.5)


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


def _packed_subset_for_batch_indices(
    packed: PackedSequence,
    batch_indices: torch.Tensor,
) -> tuple[torch.Tensor, PackedSequence]:
    index_device = packed.indices.device
    seqlen_device = packed.seqlens.device
    batch_indices = batch_indices.to(device=index_device).sort().values
    selected_rows = torch.zeros(
        (packed.batch_size,),
        dtype=torch.bool,
        device=index_device,
    )
    selected_rows[batch_indices] = True
    packed_batch_indices = torch.div(
        packed.indices,
        packed.padded_seq_len,
        rounding_mode="floor",
    )
    positions = selected_rows[packed_batch_indices].nonzero(as_tuple=False).flatten()
    seqlens = packed.seqlens.index_select(0, batch_indices.to(device=seqlen_device))

    original_indices = packed.indices.index_select(0, positions)
    original_batch = packed_batch_indices.index_select(0, positions)
    token_indices = original_indices.remainder(packed.padded_seq_len)
    subset_row_for_original = torch.empty(
        (packed.batch_size,),
        dtype=torch.long,
        device=index_device,
    )
    subset_row_for_original[batch_indices] = torch.arange(
        batch_indices.numel(),
        device=index_device,
    )
    subset_indices = (
        subset_row_for_original[original_batch] * packed.padded_seq_len + token_indices
    )

    subset_packed = PackedSequence(
        indices=subset_indices,
        cu_seqlens=F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)),
        seqlens=seqlens,
        max_seqlen=packed.max_seqlen,
        batch_size=int(batch_indices.numel()),
        padded_seq_len=packed.padded_seq_len,
    )
    return positions, subset_packed


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
