from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal, Self, TypeAlias, cast

import torch
import torch.nn.functional as F
from pydantic import Field, model_validator
from torch import nn

from owl.config import BaseConfig
from owl.model.actor import (
    ActorConfig,
    ActorDiscreteTargetsConfig,
    ActorPureConfig,
    DiscreteTargetsActor,
    MinGRUCell,
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
    masked_action_entropy_from_params,
    masked_event_log_prob_from_params,
)
from owl.model.attn import use_flash_attn, varlen_attention
from owl.model.base import (
    BaseModelAPI,
    InputLayer,
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
    EntityBasedConfig,
    ObsBatch,
)

__all__ = [
    "STATELESS_TRANSFORMER_V1",
    "ActorDiscreteTargetsConfig",
    "ActorPureConfig",
    "DiscreteActorInputs",
    "DiscreteTargetPolicyParams",
    "DiscreteTargetSizeParams",
    "DiscreteTargetsActor",
    "EncodedObservations",
    "FeedForward",
    "MinGRUCell",
    "ModelConfig",
    "MultiHeadSelfAttention",
    "OutputProjectionMLP",
    "PackedSequence",
    "PolicyParams",
    "PureActor",
    "StatelessTransformerV1",
    "StatelessTransformerV1Config",
    "binary_entropy_from_logits",
    "build_packed_sequence",
    "discrete_action_entropy",
    "discretized_logistic_mixture_log_prob",
    "masked_action_entropy_from_params",
    "masked_event_log_prob_from_params",
    "masked_softmax",
    "pack_sequence",
    "pack_tensor",
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


class StatelessTransformerV1Config(BaseConfig):
    model_arch: Literal["stateless_transformer_v1"] = STATELESS_TRANSFORMER_V1
    embed_dim: int = Field(default=128, ge=1)
    depth: int = Field(default=4, ge=1)
    n_heads: int = Field(default=8, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    n_scratch_tokens: int = Field(default=4, ge=0)
    activation: Literal["gelu", "silu", "swiglu"] = "gelu"
    force_flash_attn: bool = False
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
        return self


ModelConfig: TypeAlias = Annotated[
    StatelessTransformerV1Config, Field(discriminator="model_arch")
]


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


class StatelessTransformerV1(BaseModelAPI):
    def __init__(
        self,
        config: StatelessTransformerV1Config,
        *,
        obs_spec: EntityBasedConfig,
        action_spec: ActionConfig,
    ) -> None:
        super().__init__()
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
        self.player_tokens = nn.Parameter(torch.empty(OUTER_PLAYER_SLOTS, dim))
        self.board_tokens = nn.Parameter(torch.empty(self.config.n_scratch_tokens, dim))
        self.actor_plan_tokens = nn.Parameter(torch.empty(OUTER_PLAYER_SLOTS, dim))
        self.critic_value_tokens = nn.Parameter(torch.empty(OUTER_PLAYER_SLOTS, dim))

        self.blocks = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(self.config.depth)
        )
        self.final_norm = nn.LayerNorm(dim)

        self.critic_head = OutputProjectionMLP(self.config, 1)
        self.pure_actor_input_proj: nn.Linear | None = None
        self.source_actor_input_proj: nn.Linear | None = None
        self.target_actor_input_proj: nn.Linear | None = None
        self.actor: PureActor | DiscreteTargetsActor
        if isinstance(action_spec, ActionPureConfig):
            self.pure_actor_input_proj = nn.Linear(dim * 2, dim)
            self.actor = PureActor(
                cast(ActorPureConfig, self.config.actor),
                embed_dim=dim,
                max_per_planet_launches=action_spec.max_per_planet_launches,
                activation=self.config.activation,
            )
        else:
            self.source_actor_input_proj = nn.Linear(dim * 3, dim)
            self.target_actor_input_proj = nn.Linear(dim * 3, dim)
            self.actor = DiscreteTargetsActor(
                cast(ActorDiscreteTargetsConfig, self.config.actor),
                transformer_config=self.config,
            )

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
                if layer is self.critic_head.out
                else _ACTOR_HEAD_INIT_GAIN
            )
            _init_linear(layer, gain=gain)

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        return (
            self.static_planet_proj.input,
            self.orbit_planet_proj.input,
            self.fleet_proj.input,
            self.comet_proj.input,
            self.global_proj.input,
            self.player_tokens,
            self.board_tokens,
            self.actor_plan_tokens,
            self.critic_value_tokens,
            *self.actor.get_input_layers(),
        )

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        return (self.critic_head.out, *self.actor.get_output_layers())

    def encode_observations(self, obs: ObsBatch) -> EncodedObservations:
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
        packed: PackedSequence | None
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
        entity_count = obs.entity_mask.shape[1]
        player_start = entity_count
        global_start = player_start + OUTER_PLAYER_SLOTS
        board_start = global_start + 1
        actor_plan_start = board_start + self.config.n_scratch_tokens
        critic_value_start = actor_plan_start + OUTER_PLAYER_SLOTS
        return EncodedObservations(
            hidden=x,
            token_mask=token_mask,
            action_entity_hidden=x[:, :ACTION_ENTITY_SLOTS, :],
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

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
    ) -> ModelOutput:
        encoded = self.encode_observations(obs)
        values, winner_probabilities = self._value_from_encoded(encoded, obs)
        actions, log_probs, entropies = self._actor(
            encoded,
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
        encoded = self.encode_observations(obs)
        values, winner_probabilities = self._value_from_encoded(encoded, obs)
        log_probs, entropies = self._actor_log_prob(
            encoded,
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

    def compute_value(self, obs: ObsBatch) -> torch.Tensor:
        encoded = self.encode_observations(obs)
        values, _winner_probabilities = self._value_from_encoded(encoded, obs)
        return values

    def _value_from_encoded(
        self,
        encoded: EncodedObservations,
        obs: ObsBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._critic(encoded.critic_value_hidden, obs.still_playing)

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

    def _pure_actor_inputs(
        self,
        encoded: EncodedObservations,
    ) -> torch.Tensor:
        action_entity_hidden = encoded.action_entity_hidden
        player_hidden = encoded.player_hidden
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
        if self.pure_actor_input_proj is None:
            raise RuntimeError("pure actor input projection is not initialized")
        return self.pure_actor_input_proj(
            torch.cat((entity_features, player_features), dim=-1)
        )

    def _discrete_actor_inputs(
        self,
        encoded: EncodedObservations,
    ) -> DiscreteActorInputs:
        action_entity_hidden = encoded.action_entity_hidden
        entity_features = action_entity_hidden[:, None, :, :].expand(
            -1,
            OUTER_PLAYER_SLOTS,
            -1,
            -1,
        )
        player_features = encoded.player_hidden[:, :, None, :].expand(
            -1,
            -1,
            ACTION_ENTITY_SLOTS,
            -1,
        )
        plan_features = encoded.actor_plan_hidden[:, :, None, :].expand(
            -1,
            -1,
            ACTION_ENTITY_SLOTS,
            -1,
        )
        if self.source_actor_input_proj is None or self.target_actor_input_proj is None:
            raise RuntimeError("discrete actor input projections are not initialized")
        source = self.source_actor_input_proj(
            torch.cat((entity_features, player_features, plan_features), dim=-1)
        )
        target = self.target_actor_input_proj(
            torch.cat((entity_features, player_features, plan_features), dim=-1)
        )
        return DiscreteActorInputs(source=source, target=target)

    def _actor(
        self,
        encoded: EncodedObservations,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[ModelActions, ModelActionLogProbs, ModelActionEntropies]:
        if isinstance(self.actor, DiscreteTargetsActor):
            return self.actor(
                self._discrete_actor_inputs(encoded),
                can_act,
                max_launch,
                min_fleet_size=self.action_spec.min_fleet_size,
                deterministic=deterministic,
            )
        return self.actor(
            self._pure_actor_inputs(encoded),
            can_act,
            max_launch,
            min_fleet_size=self.action_spec.min_fleet_size,
            deterministic=deterministic,
        )

    def _actor_log_prob(
        self,
        encoded: EncodedObservations,
        can_act: torch.Tensor,
        max_launch: torch.Tensor,
        actions: ModelActions,
    ) -> tuple[ModelActionLogProbs, ModelActionEntropies]:
        if isinstance(self.actor, DiscreteTargetsActor):
            return self.actor.log_prob(
                self._discrete_actor_inputs(encoded),
                can_act,
                max_launch,
                actions,
                min_fleet_size=self.action_spec.min_fleet_size,
            )
        return self.actor.log_prob(
            self._pure_actor_inputs(encoded),
            can_act,
            max_launch,
            actions,
            min_fleet_size=self.action_spec.min_fleet_size,
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


def _expand_tokens(
    tokens: torch.Tensor,
    batch_size: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return tokens.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)


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
