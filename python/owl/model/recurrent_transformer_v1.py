from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

import torch
from pydantic import Field, model_validator
from torch import nn

from owl.config import BaseConfig
from owl.model.actor import ActorDiscreteTargetsConfig
from owl.model.attn import use_flash_attn
from owl.model.base import (
    ModelActionEntropies,
    ModelActionLogProbs,
    ModelActions,
    ModelEvaluation,
    ModelHiddenState,
    ModelOutput,
    ModelServingOutput,
)
from owl.model.stateless_transformer_v1 import (
    _ACTOR_HEAD_INIT_GAIN,
    _CRITIC_HEAD_INIT_GAIN,
    EncodedObservations,
    PackedSequence,
    StatelessTransformerV1,
    TransformerBlock,
    ValueMode,
    _action_entity_slots_from_mask,
    _init_input_layer,
    _init_linear,
    _init_module,
    _requires_flash_attn,
    build_packed_sequence,
    pack_tensor,
    unpack_sequence,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    MAX_PLANETS,
    OUTER_PLAYER_SLOTS,
    ActionConfig,
    ActionDiscreteTargetsConfig,
    DiscreteTargetActionMask,
    DiscreteTargetActions,
    EntityBasedBaseConfig,
    ObsBatch,
)

RECURRENT_TRANSFORMER_V1: Literal["recurrent_transformer_v1"] = (
    "recurrent_transformer_v1"
)


class RecurrentTransformerV1Config(BaseConfig):
    model_arch: Literal["recurrent_transformer_v1"] = RECURRENT_TRANSFORMER_V1
    embed_dim: int = Field(default=128, ge=1)
    depth: int = Field(default=4, ge=1)
    n_heads: int = Field(default=8, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    player_count_adapters_enabled: Literal[False] = False
    player_count_adapter_blocks: Literal[0] = 0
    n_scratch_tokens: int = Field(default=4, ge=0)
    activation: Literal["gelu", "silu", "swiglu"] = "gelu"
    force_flash_attn: bool = False
    use_learned_pairwise_bias: bool = False
    value_mode: ValueMode = "win_loss"
    recurrence_mode: Literal["global_only", "include_planets"] = "global_only"
    actor: ActorDiscreteTargetsConfig = Field(
        default_factory=ActorDiscreteTargetsConfig
    )

    @classmethod
    def subconfig_dirs(cls) -> set[str]:
        return {"actor"}

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.embed_dim % self.n_heads != 0:
            raise ValueError("n_heads must evenly divide embed_dim")
        if int(self.embed_dim * self.mlp_ratio) < 1:
            raise ValueError("embed_dim * mlp_ratio must be at least 1")
        if self.actor.launch_mode != "binary":
            raise ValueError("recurrent_transformer_v1 requires binary launch mode")
        return self


@dataclass(frozen=True)
class RecurrentTransformerV1HiddenState:
    hidden: torch.Tensor


@dataclass(frozen=True)
class _RecurrentTokenLayout:
    token_indices: torch.Tensor
    player_index: torch.Tensor
    shared_count: int


@dataclass(frozen=True)
class _PackedRecurrentIndex:
    positions: torch.Tensor
    present: torch.Tensor


class RecurrentTransformerV1(StatelessTransformerV1):
    def __init__(
        self,
        config: RecurrentTransformerV1Config,
        *,
        obs_spec: EntityBasedBaseConfig,
        action_spec: ActionConfig,
    ) -> None:
        if not isinstance(action_spec, ActionDiscreteTargetsConfig):
            raise ValueError("recurrent_transformer_v1 requires discrete_targets")
        if action_spec.max_per_planet_launches != 1:
            raise ValueError(
                "recurrent_transformer_v1 requires max_per_planet_launches=1"
            )
        if config.actor.action_spec != "discrete_targets":
            raise ValueError("recurrent_transformer_v1 requires discrete_targets actor")
        if config.actor.launch_mode != "binary":
            raise ValueError("recurrent_transformer_v1 requires binary launch mode")
        if obs_spec.uses_cross_attention:
            raise ValueError(
                "recurrent_transformer_v1 does not support entity_based_cross_attn_v1"
            )
        super().__init__(cast(Any, config), obs_spec=obs_spec, action_spec=action_spec)
        self._recurrence_mode = config.recurrence_mode
        self.blocks = nn.ModuleList(
            RecurrentTransformerBlock(config) for _ in range(config.depth)
        )
        self._recurrent_layout_cache: dict[int, _RecurrentTokenLayout] = {}
        self._recurrent_layout = self._recurrent_layout_for_entity_count(
            obs_spec.max_entities
        )

    def reset_parameters(self) -> None:
        self.apply(_init_module)
        for layer in self.get_input_layers():
            _init_input_layer(layer)
        residual_gain = 1.0 / math.sqrt(2.0 * self.config.depth)
        for module in self.blocks:
            block = cast(RecurrentTransformerBlock, module)
            _init_linear(block.transformer.attn.out, gain=residual_gain)
            _init_linear(block.transformer.mlp.down, gain=residual_gain)
            _init_linear(block.recurrent.out, gain=residual_gain)
        if self.critic_head is None:
            raise RuntimeError("recurrent transformer critic head is not initialized")
        critic_out = self.critic_head.out
        for layer in self.get_output_layers():
            # Recurrent models do not support LoRA, so every output layer is a
            # plain nn.Linear.
            assert isinstance(layer, nn.Linear)
            gain = (
                _CRITIC_HEAD_INIT_GAIN if layer is critic_out else _ACTOR_HEAD_INIT_GAIN
            )
            _init_linear(layer, gain=gain)

    def _recurrent_layout_for_entity_count(
        self,
        entity_count: int,
    ) -> _RecurrentTokenLayout:
        layout = self._recurrent_layout_cache.get(entity_count)
        if layout is None:
            layout = _build_recurrent_token_layout(
                entity_count=entity_count,
                n_scratch_tokens=self.config.n_scratch_tokens,
                include_planets=self._recurrence_mode == "include_planets",
            )
            self._recurrent_layout_cache[entity_count] = layout
        return layout

    def initial_hidden_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> RecurrentTransformerV1HiddenState:
        hidden = torch.zeros(
            (
                self.config.depth,
                batch_size,
                self._recurrent_layout.token_indices.numel(),
                self.config.embed_dim,
            ),
            dtype=next(self.parameters()).dtype,
            device=device,
        )
        return RecurrentTransformerV1HiddenState(hidden=hidden)

    def detach_hidden_state(
        self,
        hidden_state: ModelHiddenState | None,
    ) -> RecurrentTransformerV1HiddenState | None:
        if hidden_state is None:
            return None
        state = _require_recurrent_hidden_state(hidden_state)
        return RecurrentTransformerV1HiddenState(hidden=state.hidden.detach())

    def index_hidden_state(
        self,
        hidden_state: ModelHiddenState | None,
        indices: torch.Tensor,
    ) -> RecurrentTransformerV1HiddenState | None:
        if hidden_state is None:
            return None
        state = _require_recurrent_hidden_state(hidden_state)
        return RecurrentTransformerV1HiddenState(hidden=state.hidden[:, indices])

    def reset_hidden_state(
        self,
        hidden_state: ModelHiddenState | None,
        dones: torch.Tensor,
    ) -> RecurrentTransformerV1HiddenState | None:
        if hidden_state is None:
            return None
        state = _require_recurrent_hidden_state(hidden_state)
        keep = _hidden_keep_mask(
            dones,
            self._recurrent_layout,
            device=state.hidden.device,
        )
        return RecurrentTransformerV1HiddenState(
            hidden=state.hidden * keep[None, :, :, None].to(dtype=state.hidden.dtype)
        )

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
        hidden_state: ModelHiddenState | None = None,
    ) -> ModelOutput:
        state = self._initial_or_validate_hidden_state(obs, hidden_state)
        encoded, next_state = self._encode_sequence(
            obs,
            hidden_state=state,
            dones=None,
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
            next_hidden_state=next_state,
        )

    def serve(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
        hidden_state: ModelHiddenState | None = None,
    ) -> ModelServingOutput:
        state = self._initial_or_validate_hidden_state(obs, hidden_state)
        encoded, next_state = self._encode_sequence(
            obs,
            hidden_state=state,
            dones=None,
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
            next_hidden_state=next_state,
        )

    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ModelActions,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
    ) -> ModelEvaluation:
        state = self._initial_or_validate_hidden_state(obs, hidden_state)
        flat_obs, sequence_shape = _flatten_obs_if_sequence(obs)
        flat_actions = _flatten_actions_if_sequence(actions, sequence_shape)
        flat_dones = None if dones is None else dones
        encoded, next_state = self._encode_sequence(
            flat_obs,
            hidden_state=state,
            sequence_shape=sequence_shape,
            dones=flat_dones,
        )
        values, winner_probabilities = self._value_from_encoded(encoded, flat_obs)
        log_probs, entropies = self._actor_log_prob(
            encoded,
            flat_obs,
            flat_obs.action_mask,
            flat_actions,
        )
        if sequence_shape is not None:
            values = _unflatten_time_tensor(values, sequence_shape)
            winner_probabilities = _unflatten_time_tensor(
                winner_probabilities,
                sequence_shape,
            )
            log_probs = _unflatten_log_probs(log_probs, sequence_shape)
            entropies = _unflatten_entropies(entropies, sequence_shape)
        return ModelEvaluation(
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=winner_probabilities,
            next_hidden_state=next_state,
        )

    def compute_value(
        self,
        obs: ObsBatch,
        *,
        hidden_state: ModelHiddenState | None = None,
    ) -> torch.Tensor:
        state = self._initial_or_validate_hidden_state(obs, hidden_state)
        encoded, _next_state = self._encode_sequence(
            obs,
            hidden_state=state,
            dones=None,
        )
        values, _winner_probabilities = self._value_from_encoded(encoded, obs)
        return values

    def _encode_distillation_observations(
        self,
        obs: ObsBatch,
        *,
        sequence_shape: tuple[int, int] | None,
        hidden_state: ModelHiddenState | None,
        dones: torch.Tensor | None,
    ) -> tuple[EncodedObservations, ModelHiddenState | None]:
        if hidden_state is None:
            batch_size = (
                obs.planets.shape[0] if sequence_shape is None else sequence_shape[0]
            )
            state = self.initial_hidden_state(batch_size, device=obs.planets.device)
        else:
            state = _require_recurrent_hidden_state(hidden_state)
        return self._encode_sequence(
            obs,
            hidden_state=state,
            sequence_shape=sequence_shape,
            dones=dones,
        )

    def _initial_or_validate_hidden_state(
        self,
        obs: ObsBatch,
        hidden_state: ModelHiddenState | None,
    ) -> RecurrentTransformerV1HiddenState:
        if hidden_state is not None:
            return _require_recurrent_hidden_state(hidden_state)
        batch_size = obs.planets.shape[0]
        return self.initial_hidden_state(batch_size, device=obs.planets.device)

    def _encode_sequence(
        self,
        obs: ObsBatch,
        *,
        hidden_state: RecurrentTransformerV1HiddenState,
        dones: torch.Tensor | None,
        sequence_shape: tuple[int, int] | None = None,
    ) -> tuple[EncodedObservations, RecurrentTransformerV1HiddenState]:
        if sequence_shape is None:
            batch_size = obs.planets.shape[0]
            time_steps = 1
        else:
            batch_size, time_steps = sequence_shape
        layout = self._recurrent_layout_for_entity_count(obs.entity_mask.shape[1])
        expected_hidden_shape = (
            self.config.depth,
            batch_size,
            layout.token_indices.numel(),
            self.config.embed_dim,
        )
        if hidden_state.hidden.shape != expected_hidden_shape:
            raise ValueError(
                "hidden_state.hidden must have shape "
                f"{expected_hidden_shape}, got {tuple(hidden_state.hidden.shape)}"
            )
        action_entity_slots = _action_entity_slots_from_mask(obs.action_mask)
        encoded_inputs = self._build_flat_tokens(obs)
        x = encoded_inputs.x
        token_mask = encoded_inputs.token_mask
        recurrent_reset = _recurrent_reset_mask(
            dones,
            batch_size=batch_size,
            time_steps=time_steps,
            layout=layout,
            device=token_mask.device,
        )
        recurrent_active = token_mask[:, layout.token_indices].view(
            batch_size,
            time_steps,
            -1,
        )

        packed: PackedSequence | None
        packed_recurrent_index: _PackedRecurrentIndex | None
        should_use_flash = use_flash_attn(x)
        if (
            _requires_flash_attn(x, force_flash_attn=self.config.force_flash_attn)
            and not should_use_flash
        ):
            raise RuntimeError(
                "force_flash_attn=True requires CUDA fp16/bf16 tensors "
                "and the flash-attn package"
            )
        if should_use_flash:
            packed = build_packed_sequence(token_mask)
            x = pack_tensor(x, packed)
            packed_recurrent_index = _build_packed_recurrent_index(
                packed,
                layout.token_indices,
                device=x.device,
            )
            block_token_mask = None
        else:
            packed = None
            packed_recurrent_index = None
            block_token_mask = token_mask

        next_hidden_layers: list[torch.Tensor] = []
        for layer_index, module in enumerate(self.blocks):
            block = cast(RecurrentTransformerBlock, module)
            x, layer_hidden = block(
                x,
                block_token_mask,
                packed,
                hidden_state.hidden[layer_index],
                reset=recurrent_reset,
                active=recurrent_active,
                layout=layout,
                packed_recurrent_index=packed_recurrent_index,
                batch_size=batch_size,
                time_steps=time_steps,
            )
            next_hidden_layers.append(layer_hidden)
        x = self.final_norm(x)
        if packed is not None:
            x = unpack_sequence(x, packed)
        x = x.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        encoded = self._encoded_from_flat_hidden(
            x,
            token_mask,
            obs,
            action_entity_slots=action_entity_slots,
        )
        return encoded, RecurrentTransformerV1HiddenState(
            hidden=torch.stack(next_hidden_layers, dim=0)
        )

    def _build_flat_tokens(self, obs: ObsBatch) -> _EncodedInputs:
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
                planet_x,
                comet_x,
                fleet_x,
                player_tokens,
                global_token,
                board_tokens,
                actor_plan_tokens,
                critic_value_tokens,
            ),
            dim=1,
        )
        return _EncodedInputs(x=x, token_mask=token_mask)

    def _encoded_from_flat_hidden(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        obs: ObsBatch,
        *,
        action_entity_slots: int = ACTION_ENTITY_SLOTS,
    ) -> EncodedObservations:
        entity_count = obs.entity_mask.shape[1]
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


@dataclass(frozen=True)
class _EncodedInputs:
    x: torch.Tensor
    token_mask: torch.Tensor


class RecurrentTransformerBlock(nn.Module):
    def __init__(self, config: RecurrentTransformerV1Config) -> None:
        super().__init__()
        self.transformer = TransformerBlock(cast(Any, config))
        self.recurrent_norm = nn.LayerNorm(config.embed_dim)
        self.recurrent = MinGRU(config.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None,
        packed: PackedSequence | None,
        hidden_state: torch.Tensor,
        *,
        reset: torch.Tensor,
        active: torch.Tensor,
        layout: _RecurrentTokenLayout,
        packed_recurrent_index: _PackedRecurrentIndex | None,
        batch_size: int,
        time_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.transformer(x, token_mask, packed)
        recurrent_x = _gather_recurrent_tokens(
            x,
            layout,
            packed=packed,
            packed_recurrent_index=packed_recurrent_index,
            batch_size=batch_size,
            time_steps=time_steps,
        )
        recurrent_input = self.recurrent_norm(recurrent_x)
        recurrent_output, next_hidden = self.recurrent(
            recurrent_input,
            hidden_state,
            reset=reset,
            active=active,
        )
        recurrent_x = recurrent_x + self.recurrent.out(recurrent_output)
        x = _scatter_recurrent_tokens(
            x,
            recurrent_x,
            layout,
            packed=packed,
            packed_recurrent_index=packed_recurrent_index,
            batch_size=batch_size,
            time_steps=time_steps,
        )
        return x, next_hidden


class MinGRU(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.candidate = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor,
        *,
        reset: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate(x))
        candidate = torch.tanh(self.candidate(x))
        active_f = active.unsqueeze(-1).to(dtype=x.dtype)
        keep_f = (~reset & active).unsqueeze(-1).to(dtype=x.dtype)
        a = (1.0 - gate) * keep_f
        b = gate * candidate * active_f
        prefix_a, prefix_b = _parallel_affine_scan(a, b)
        hidden = prefix_a * hidden_state.unsqueeze(1).to(dtype=prefix_a.dtype)
        hidden = hidden + prefix_b
        hidden = hidden * active_f
        return hidden, hidden[:, -1].to(dtype=hidden_state.dtype)


def _parallel_affine_scan(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    time_steps = a.shape[1]
    offset = 1
    scan_a = a
    scan_b = b
    while offset < time_steps:
        prev_a = scan_a[:, :-offset]
        prev_b = scan_b[:, :-offset]
        cur_a = scan_a[:, offset:]
        cur_b = scan_b[:, offset:]
        combined_a = cur_a * prev_a
        combined_b = cur_b + cur_a * prev_b
        scan_a = torch.cat((scan_a[:, :offset], combined_a), dim=1)
        scan_b = torch.cat((scan_b[:, :offset], combined_b), dim=1)
        offset *= 2
    return scan_a, scan_b


def _build_recurrent_token_layout(
    *,
    entity_count: int,
    n_scratch_tokens: int,
    include_planets: bool,
) -> _RecurrentTokenLayout:
    player_start = entity_count
    global_start = player_start + OUTER_PLAYER_SLOTS
    board_start = global_start + 1
    actor_plan_start = board_start + n_scratch_tokens
    critic_value_start = actor_plan_start + OUTER_PLAYER_SLOTS
    token_indices = list(range(MAX_PLANETS)) if include_planets else []
    token_indices.extend([global_start, *range(board_start, actor_plan_start)])
    player_index = [-1 for _ in token_indices]
    shared_count = len(token_indices)
    for start in (player_start, actor_plan_start, critic_value_start):
        for player in range(OUTER_PLAYER_SLOTS):
            token_indices.append(start + player)
            player_index.append(player)
    return _RecurrentTokenLayout(
        token_indices=torch.tensor(token_indices, dtype=torch.long),
        player_index=torch.tensor(player_index, dtype=torch.long),
        shared_count=shared_count,
    )


def _recurrent_reset_mask(
    dones: torch.Tensor | None,
    *,
    batch_size: int,
    time_steps: int,
    layout: _RecurrentTokenLayout,
    device: torch.device,
) -> torch.Tensor:
    reset = torch.zeros(
        (batch_size, time_steps, layout.token_indices.numel()),
        dtype=torch.bool,
        device=device,
    )
    if dones is None:
        return reset
    if dones.shape != (batch_size, time_steps, OUTER_PLAYER_SLOTS):
        raise ValueError(
            "dones must have shape "
            f"{(batch_size, time_steps, OUTER_PLAYER_SLOTS)}, got {tuple(dones.shape)}"
        )
    if time_steps == 1:
        return reset
    previous_dones = dones[:, :-1].to(device=device)
    reset[:, 1:, : layout.shared_count] = previous_dones.all(dim=-1, keepdim=True)
    player_index = layout.player_index[layout.shared_count :].to(device=device)
    reset[:, 1:, layout.shared_count :] = previous_dones.index_select(
        dim=-1,
        index=player_index,
    )
    return reset


def _hidden_keep_mask(
    dones: torch.Tensor,
    layout: _RecurrentTokenLayout,
    *,
    device: torch.device,
) -> torch.Tensor:
    if dones.ndim != 2 or dones.shape[1] != OUTER_PLAYER_SLOTS:
        raise ValueError(
            f"dones must have shape (batch, {OUTER_PLAYER_SLOTS}), "
            f"got {tuple(dones.shape)}"
        )
    dones = dones.to(device=device)
    keep = torch.ones(
        (dones.shape[0], layout.token_indices.numel()),
        dtype=torch.bool,
        device=device,
    )
    keep[:, : layout.shared_count] = ~dones.all(dim=-1, keepdim=True)
    player_index = layout.player_index[layout.shared_count :].to(device=device)
    keep[:, layout.shared_count :] = ~dones.index_select(dim=-1, index=player_index)
    return keep


def _build_packed_recurrent_index(
    packed: PackedSequence,
    token_indices: torch.Tensor,
    *,
    device: torch.device,
) -> _PackedRecurrentIndex:
    flat_size = packed.batch_size * packed.padded_seq_len
    inverse = torch.full(
        (flat_size,),
        -1,
        dtype=torch.long,
        device=device,
    )
    inverse[packed.indices] = torch.arange(
        packed.indices.numel(),
        dtype=torch.long,
        device=device,
    )
    rows = torch.arange(packed.batch_size, dtype=torch.long, device=device)
    flat_recurrent = (
        rows[:, None] * packed.padded_seq_len + token_indices.to(device=device)[None, :]
    )
    positions = inverse[flat_recurrent]
    return _PackedRecurrentIndex(
        positions=positions,
        present=positions.ge(0),
    )


def _gather_recurrent_tokens(
    x: torch.Tensor,
    layout: _RecurrentTokenLayout,
    *,
    packed: PackedSequence | None,
    packed_recurrent_index: _PackedRecurrentIndex | None,
    batch_size: int,
    time_steps: int,
) -> torch.Tensor:
    if packed is None:
        return x[:, layout.token_indices.to(device=x.device)].view(
            batch_size,
            time_steps,
            layout.token_indices.numel(),
            x.shape[-1],
        )
    if packed_recurrent_index is None:
        raise RuntimeError("packed recurrent index is required for packed tensors")
    safe_positions = packed_recurrent_index.positions.clamp_min(0)
    recurrent = x[safe_positions]
    recurrent = recurrent.masked_fill(
        ~packed_recurrent_index.present.unsqueeze(-1),
        0.0,
    )
    return recurrent.view(
        batch_size,
        time_steps,
        layout.token_indices.numel(),
        x.shape[-1],
    )


def _scatter_recurrent_tokens(
    x: torch.Tensor,
    recurrent_x: torch.Tensor,
    layout: _RecurrentTokenLayout,
    *,
    packed: PackedSequence | None,
    packed_recurrent_index: _PackedRecurrentIndex | None,
    batch_size: int,
    time_steps: int,
) -> torch.Tensor:
    x = x.clone()
    recurrent_flat = recurrent_x.reshape(
        batch_size * time_steps,
        layout.token_indices.numel(),
        x.shape[-1],
    )
    if packed is None:
        x[:, layout.token_indices.to(device=x.device)] = recurrent_flat[:, :]
        return x
    if packed_recurrent_index is None:
        raise RuntimeError("packed recurrent index is required for packed tensors")
    present = packed_recurrent_index.present
    x[packed_recurrent_index.positions[present]] = recurrent_flat[present]
    return x


def _flatten_obs_if_sequence(obs: ObsBatch) -> tuple[ObsBatch, tuple[int, int] | None]:
    if obs.planets.ndim == 3:
        return obs, None
    if obs.planets.ndim != 4:
        raise ValueError("obs planets must be batch-major or segment-major")
    batch_size, time_steps = obs.planets.shape[:2]
    if not isinstance(obs.action_mask, DiscreteTargetActionMask):
        raise RuntimeError("recurrent_transformer_v1 requires discrete-target masks")
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
            action_mask=DiscreteTargetActionMask(
                can_act=_flatten_time_tensor(obs.action_mask.can_act),
                max_launch=_flatten_time_tensor(obs.action_mask.max_launch),
            ),
            player_features=(
                None
                if obs.player_features is None
                else _flatten_time_tensor(obs.player_features)
            ),
        ),
        (batch_size, time_steps),
    )


def _flatten_actions_if_sequence(
    actions: ModelActions,
    sequence_shape: tuple[int, int] | None,
) -> ModelActions:
    if sequence_shape is None:
        return actions
    if not isinstance(actions, DiscreteTargetActions):
        raise ValueError("recurrent_transformer_v1 requires DiscreteTargetActions")
    return DiscreteTargetActions(
        launch=_flatten_time_tensor(actions.launch),
        target=_flatten_time_tensor(actions.target),
        ships=_flatten_time_tensor(actions.ships),
    )


def _flatten_time_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _unflatten_time_tensor(
    tensor: torch.Tensor,
    sequence_shape: tuple[int, int],
) -> torch.Tensor:
    batch_size, time_steps = sequence_shape
    return tensor.reshape(batch_size, time_steps, *tensor.shape[1:])


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


def _expand_tokens(
    tokens: torch.Tensor,
    batch_size: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return tokens.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)


def _require_recurrent_hidden_state(
    hidden_state: ModelHiddenState,
) -> RecurrentTransformerV1HiddenState:
    if not isinstance(hidden_state, RecurrentTransformerV1HiddenState):
        raise TypeError(
            "hidden_state must be RecurrentTransformerV1HiddenState, "
            f"got {type(hidden_state).__name__}"
        )
    return hidden_state
