import math
from pathlib import Path
from typing import Any

import owl.model.actor.discrete_target_bins as discrete_target_bins_impl
import owl.model.actor.discrete_targets as discrete_targets_impl
import owl.model.actor.logistic_mixture as logistic_mixture_impl
import owl.model.actor.pure as pure_actor_impl
import owl.model.stateless_transformer_v1 as model_impl
import pytest
import torch
import torch.nn.functional as F
from owl.model import (
    LoRAConfig,
    LoRALinear,
    ModelConfig,
    StatelessTransformerV1,
    StatelessTransformerV1Config,
    apply_lora_to_stateless_transformer,
    fold_lora_adapters,
    load_model_state_dict_allowing_lora,
)
from owl.model.actor.discrete_targets import (
    DiscreteTargetSelectionParams,
    discretized_logistic_mixture_log_prob,
    logsubexp,
    sample_discretized_logistic_mixture,
    ship_support,
)
from owl.model.stateless_transformer_v1 import (
    ActorDiscreteTargetBinsConfig,
    ActorDiscreteTargetsConfig,
    ActorPureConfig,
    DiscreteActorInputs,
    DiscreteTargetBinsActor,
    DiscreteTargetPolicyParams,
    DiscreteTargetsActor,
    FeedForward,
    MultiHeadSelfAttention,
    OutputProjectionMLP,
    PairwiseBiasMLP,
    PolicyParams,
    PureActor,
    PureActorInputs,
    build_pairwise_action_features,
    discrete_action_entropy,
    masked_event_log_prob_from_params,
    pack_sequence,
    unpack_sequence,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    MAX_COMETS,
    MAX_PLANETS,
    OUTER_PLAYER_SLOTS,
    ActionConfig,
    ActionDiscreteTargetBinsConfig,
    ActionDiscreteTargetsConfig,
    ActionPureConfig,
    DiscreteTargetActionMask,
    DiscreteTargetActions,
    DiscreteTargetBinActionMask,
    DiscreteTargetBinActions,
    EntityBasedBaseConfig,
    EntityBasedConfig,
    EntityBasedCrossAttnV1Config,
    EntityBasedExtV1Config,
    EntityBasedExtV2Config,
    ObsBatch,
    PureActionMask,
    PureActions,
    encode_python_observation,
)
from pydantic import TypeAdapter
from torch import nn

_REPO_ROOT = Path(__file__).parents[3]


def _python_obs(**overrides: object) -> dict[str, object]:
    obs = {
        "step": 0,
        "episode_steps": 500,
        "angular_velocity": 0.025,
        "planets": [],
        "initial_planets": [],
        "fleets": [],
        "player": 0,
        "comets": [],
    }
    obs.update(overrides)
    if "initial_planets" not in overrides:
        obs["initial_planets"] = obs["planets"]
    return obs


def _obs_batch(
    *,
    batch_size: int,
    obs_spec: EntityBasedBaseConfig,
    action_spec: ActionConfig,
) -> ObsBatch:
    planets = torch.zeros(
        (batch_size, obs_spec.max_planets, obs_spec.planet_channels),
        dtype=torch.float32,
    )
    orbiting_planets = torch.zeros(
        (batch_size, obs_spec.max_planets),
        dtype=torch.bool,
    )
    fleets = torch.zeros(
        (batch_size, obs_spec.max_fleets, obs_spec.fleet_channels),
        dtype=torch.float32,
    )
    comets = torch.zeros(
        (batch_size, obs_spec.max_comets, obs_spec.comet_channels),
        dtype=torch.float32,
    )
    entity_mask = torch.zeros((batch_size, obs_spec.max_entities), dtype=torch.bool)
    still_playing = torch.ones((batch_size, 4), dtype=torch.bool)
    global_features = torch.zeros(
        (batch_size, obs_spec.global_channels),
        dtype=torch.float32,
    )
    if isinstance(action_spec, ActionPureConfig):
        can_act = torch.zeros((batch_size, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool)
    elif isinstance(action_spec, ActionDiscreteTargetsConfig):
        can_act = torch.zeros(
            (batch_size, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS),
            dtype=torch.bool,
        )
    else:
        can_act = torch.zeros(
            (
                batch_size,
                4,
                ACTION_ENTITY_SLOTS,
                ACTION_ENTITY_SLOTS,
                action_spec.n_bins,
            ),
            dtype=torch.bool,
        )
    max_launch = (
        None
        if isinstance(action_spec, ActionDiscreteTargetBinsConfig)
        else torch.zeros((batch_size, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
    )

    planets[:, 0, 0] = 1.0
    planets[:, 0, 15] = 0.08
    planets[:, 1, 1] = 1.0
    planets[:, 1, 15] = 0.04
    entity_mask[:, :2] = True
    comets[:, 0, 2] = 1.0
    entity_mask[:, MAX_PLANETS] = True
    if isinstance(action_spec, ActionPureConfig):
        can_act[:, 0, 0] = True
        can_act[:, 1, 1] = True
        can_act[:, 2, MAX_PLANETS] = True
    elif isinstance(action_spec, ActionDiscreteTargetsConfig):
        can_act[:, 0, 0, 1] = True
        can_act[:, 0, 0, MAX_PLANETS] = True
        can_act[:, 1, 1, 0] = True
        can_act[:, 2, MAX_PLANETS, 0] = True
    else:
        can_act[:, 0, 0, 1, [0, action_spec.n_bins - 1]] = True
        can_act[:, 0, 0, MAX_PLANETS, [0, 2]] = True
        can_act[:, 1, 1, 0, [0, action_spec.n_bins - 1]] = True
        can_act[:, 2, MAX_PLANETS, 0, [0, 1]] = True
    if max_launch is not None:
        max_launch[:, 0, 0] = 5
        max_launch[:, 1, 1] = 3
        max_launch[:, 2, MAX_PLANETS] = 2

    if isinstance(action_spec, ActionPureConfig):
        assert max_launch is not None
        action_mask = PureActionMask(
            can_act=can_act,
            max_launch=max_launch,
        )
    elif isinstance(action_spec, ActionDiscreteTargetsConfig):
        assert max_launch is not None
        action_mask = DiscreteTargetActionMask(
            can_act=can_act,
            max_launch=max_launch,
        )
    else:
        action_mask = DiscreteTargetBinActionMask(can_act=can_act)

    return ObsBatch(
        planets=planets,
        orbiting_planets=orbiting_planets,
        fleets=fleets,
        fleet_target=(
            None
            if not obs_spec.uses_cross_attention
            else torch.full((batch_size, obs_spec.max_fleets), -1, dtype=torch.int64)
        ),
        target_incoming_features=(
            None
            if not obs_spec.uses_cross_attention
            else torch.zeros(
                (
                    batch_size,
                    ACTION_ENTITY_SLOTS,
                    obs_spec.target_incoming_channels,
                ),
                dtype=torch.float32,
            )
        ),
        comets=comets,
        entity_mask=entity_mask,
        still_playing=still_playing,
        global_features=global_features,
        action_mask=action_mask,
        player_features=(
            None
            if obs_spec.player_feature_channels == 0
            else torch.zeros(
                (batch_size, 4, obs_spec.player_feature_channels),
                dtype=torch.float32,
            )
        ),
    )


def _compacted_obs_batch(
    *,
    batch_size: int,
    obs_spec: EntityBasedBaseConfig,
    action_spec: ActionConfig,
) -> ObsBatch:
    obs = _obs_batch(batch_size=batch_size, obs_spec=obs_spec, action_spec=action_spec)
    action_indices = torch.tensor([0, 1, MAX_PLANETS])
    entity_mask = torch.ones((batch_size, action_indices.numel()), dtype=torch.bool)
    action_mask: PureActionMask | DiscreteTargetActionMask | DiscreteTargetBinActionMask
    if isinstance(obs.action_mask, PureActionMask):
        action_mask = PureActionMask(
            can_act=obs.action_mask.can_act.index_select(2, action_indices),
            max_launch=obs.action_mask.max_launch.index_select(2, action_indices),
        )
    elif isinstance(obs.action_mask, DiscreteTargetActionMask):
        can_act = obs.action_mask.can_act.index_select(2, action_indices)
        can_act = can_act.index_select(3, action_indices)
        action_mask = DiscreteTargetActionMask(
            can_act=can_act,
            max_launch=obs.action_mask.max_launch.index_select(2, action_indices),
        )
    else:
        can_act = obs.action_mask.can_act.index_select(2, action_indices)
        can_act = can_act.index_select(3, action_indices)
        action_mask = DiscreteTargetBinActionMask(can_act=can_act)
    return ObsBatch(
        planets=obs.planets[:, :2, :],
        orbiting_planets=obs.orbiting_planets[:, :2],
        fleets=obs.fleets[:, :0, :],
        comets=obs.comets[:, :1, :],
        entity_mask=entity_mask,
        still_playing=obs.still_playing,
        global_features=obs.global_features,
        action_mask=action_mask,
        player_features=obs.player_features,
    )


def _model(
    config: StatelessTransformerV1Config,
    *,
    obs_spec: EntityBasedBaseConfig | None = None,
    action_spec: ActionConfig | None = None,
) -> StatelessTransformerV1:
    model = StatelessTransformerV1(
        config,
        obs_spec=obs_spec or EntityBasedConfig(),
        action_spec=action_spec or ActionPureConfig(max_per_planet_launches=1),
    )
    model.reset_parameters()
    return model


def _discrete_actor_inputs(
    source: torch.Tensor,
    target: torch.Tensor | None = None,
    pairwise_bias: torch.Tensor | None = None,
) -> DiscreteActorInputs:
    return DiscreteActorInputs(
        source=source,
        target=source if target is None else target,
        pairwise_bias=pairwise_bias,
    )


def _pure_actor_inputs(
    source: torch.Tensor,
    target: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
) -> PureActorInputs:
    return PureActorInputs(
        source=source,
        target=source if target is None else target,
        target_mask=(
            torch.ones(source.shape[0], source.shape[2], dtype=torch.bool)
            if target_mask is None
            else target_mask
        ),
    )


def _zero_target_attention(
    actor: DiscreteTargetsActor | DiscreteTargetBinsActor,
) -> None:
    with torch.no_grad():
        actor.source_role.zero_()
        actor.target_role.zero_()
        for module in (actor.q, actor.k):
            module.weight.zero_()
            module.bias.zero_()


def test_model_config_requires_heads_to_divide_embed_dim() -> None:
    with pytest.raises(ValueError, match="n_heads must evenly divide embed_dim"):
        StatelessTransformerV1Config(embed_dim=30, n_heads=8)


def test_model_config_requires_positive_feedforward_width() -> None:
    with pytest.raises(ValueError, match="embed_dim \\* mlp_ratio must be at least 1"):
        StatelessTransformerV1Config(embed_dim=1, n_heads=1, mlp_ratio=0.5)


def test_model_config_requires_player_count_adapter_blocks_within_depth() -> None:
    with pytest.raises(
        ValueError,
        match="player_count_adapter_blocks must be less than or equal to depth",
    ):
        StatelessTransformerV1Config(
            depth=2,
            player_count_adapters_enabled=True,
            player_count_adapter_blocks=3,
        )


def test_model_config_requires_enabled_player_count_adapters_for_blocks() -> None:
    with pytest.raises(
        ValueError,
        match="player_count_adapter_blocks requires player_count_adapters_enabled=True",
    ):
        StatelessTransformerV1Config(player_count_adapter_blocks=1)


def _is_lora_parameter(name: str) -> bool:
    return name.endswith((".lora_down", ".lora_up"))


def test_model_config_rejects_duplicate_lora_targets() -> None:
    with pytest.raises(ValueError, match=r"lora\.target_modules must not contain"):
        StatelessTransformerV1Config(
            lora=LoRAConfig(rank=2, target_modules=("q", "q")),
        )


def test_model_config_rejects_target_block_count_without_target_modules() -> None:
    # target_block_count only selects transformer-block projections, so it would
    # silently do nothing without target_modules; reject it rather than ignore it.
    with pytest.raises(ValueError, match=r"target_block_count only selects"):
        LoRAConfig(
            rank=2,
            target_modules=(),
            target_block_count=1,
            target_value_head=True,
        )


def test_lora_config_allows_target_block_count_with_target_modules() -> None:
    config = LoRAConfig(rank=2, target_modules=("q",), target_block_count=1)
    assert config.target_block_count == 1


def test_apply_lora_wraps_requested_final_block_modules_and_freezes_base() -> None:
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(
            embed_dim=16,
            depth=3,
            n_heads=4,
            mlp_ratio=2.0,
        ),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    model.reset_parameters()

    application = apply_lora_to_stateless_transformer(
        model,
        LoRAConfig(
            rank=2,
            target_modules=("q", "v", "up"),
            target_block_count=2,
        ),
    )

    assert application.module_count == 6
    assert application.trainable_parameters == 448
    assert not isinstance(model.blocks[0].attn.q, LoRALinear)
    assert isinstance(model.blocks[1].attn.q, LoRALinear)
    assert isinstance(model.blocks[1].attn.v, LoRALinear)
    assert isinstance(model.blocks[1].mlp.up, LoRALinear)
    assert isinstance(model.blocks[2].attn.q, LoRALinear)
    assert all(
        _is_lora_parameter(name) == parameter.requires_grad
        for name, parameter in model.named_parameters()
    )


def test_lora_wrapped_linear_starts_as_noop_and_loads_base_state_dict() -> None:
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=4),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    model.reset_parameters()
    base_state_dict = model.state_dict()
    x = torch.randn(2, 3, 16)
    expected = model.blocks[0].attn.q(x)

    apply_lora_to_stateless_transformer(
        model,
        LoRAConfig(rank=4, target_modules=("q",), target_block_count=1),
    )

    assert isinstance(model.blocks[0].attn.q, LoRALinear)
    assert torch.allclose(model.blocks[0].attn.q(x), expected)
    load_model_state_dict_allowing_lora(model, base_state_dict)


def test_lora_state_dict_loader_rejects_partial_lora_checkpoint() -> None:
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=4),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    model.reset_parameters()
    apply_lora_to_stateless_transformer(
        model,
        LoRAConfig(rank=4, target_modules=("q",), target_block_count=1),
    )
    state_dict = model.state_dict()
    partial_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith("attn.q.lora_up")
    }

    with pytest.raises(RuntimeError, match="missing LoRA model state_dict keys"):
        load_model_state_dict_allowing_lora(model, partial_state_dict)


def test_apply_lora_wraps_value_and_policy_heads() -> None:
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=4),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    model.reset_parameters()

    application = apply_lora_to_stateless_transformer(
        model,
        LoRAConfig(
            rank=2,
            target_modules=(),
            target_value_head=True,
            target_policy_head=True,
        ),
    )

    wrapped = {
        name for name, module in model.named_modules() if isinstance(module, LoRALinear)
    }
    assert "critic_head.out" in wrapped
    assert "critic_head.up" in wrapped
    assert "source_actor_input_proj" in wrapped
    assert "target_actor_input_proj" in wrapped
    assert "actor.q" in wrapped
    assert "actor.actor_heads.continue_head.out" in wrapped
    # target_modules is empty, so no transformer-block projection is wrapped.
    assert not any(name.startswith("blocks.") for name in wrapped)
    assert application.module_count == len(wrapped)
    assert application.trainable_parameters == sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if _is_lora_parameter(name)
    )
    assert all(
        _is_lora_parameter(name) == parameter.requires_grad
        for name, parameter in model.named_parameters()
    )


def test_apply_lora_policy_head_wraps_pairwise_bias_mlp() -> None:
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(
            embed_dim=16,
            depth=1,
            n_heads=4,
            actor={"action_spec": "discrete_targets"},
            use_learned_pairwise_bias=True,
        ),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
    )
    model.reset_parameters()

    application = apply_lora_to_stateless_transformer(
        model,
        LoRAConfig(
            rank=2,
            target_modules=(),
            target_policy_head=True,
        ),
    )

    wrapped = {
        name for name, module in model.named_modules() if isinstance(module, LoRALinear)
    }
    assert "source_actor_input_proj" in wrapped
    assert "target_actor_input_proj" in wrapped
    assert "pairwise_bias_mlp.up" in wrapped
    assert "pairwise_bias_mlp.out" in wrapped
    assert "actor.q" in wrapped
    assert application.module_count == len(wrapped)
    assert all(
        _is_lora_parameter(name) == parameter.requires_grad
        for name, parameter in model.named_parameters()
    )


def test_lora_clamps_adapter_rank_for_degenerate_projections() -> None:
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=4),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    model.reset_parameters()
    apply_lora_to_stateless_transformer(
        model,
        LoRAConfig(rank=8, target_modules=(), target_value_head=True),
    )

    modules = dict(model.named_modules())
    critic_out = modules["critic_head.out"]
    critic_up = modules["critic_head.up"]
    assert isinstance(critic_out, LoRALinear)
    assert isinstance(critic_up, LoRALinear)
    # The embed_dim -> 1 output clamps to min(rank=8, in=16, out=1) == 1, so the
    # adapter is not over-parameterized but still adapts (and stays separate from
    # the frozen base weight).
    assert critic_out.rank == 1
    assert tuple(critic_out.lora_down.shape) == (1, 16)
    assert tuple(critic_out.lora_up.shape) == (1, 1)
    # The embed_dim -> embed_dim hidden projection keeps the configured rank.
    assert critic_up.rank == 8
    # Scaling equals alpha_scale (default 1.0) regardless of how much the rank
    # was clamped, so the clamped output adapter and the full-rank hidden adapter
    # share the same update scale.
    assert critic_out.scaling == pytest.approx(1.0)
    assert critic_up.scaling == pytest.approx(1.0)
    # The clamped adapter is still an exact no-op at init.
    x = torch.randn(2, 3, 16)
    assert torch.allclose(
        critic_out(x), F.linear(x, critic_out.weight, critic_out.bias)
    )


def test_lora_linear_applies_scaled_low_rank_update() -> None:
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 6)
    wrapped = LoRALinear(base, rank=2, alpha_scale=2.0)
    # scaling equals alpha_scale; the scale must reach the final update regardless
    # of where it is folded internally.
    assert wrapped.scaling == pytest.approx(2.0)
    with torch.no_grad():
        wrapped.lora_down.copy_(torch.randn(2, 8))
        wrapped.lora_up.copy_(torch.randn(6, 2))

    x = torch.randn(3, 8)
    expected = F.linear(x, base.weight, base.bias) + wrapped.scaling * F.linear(
        F.linear(x, wrapped.lora_down), wrapped.lora_up
    )

    assert torch.allclose(wrapped(x), expected, atol=1e-6)


def test_fold_lora_adapters_preserves_output_and_removes_wrappers() -> None:
    base = torch.nn.Linear(8, 6)
    wrapped = LoRALinear(base, rank=2, alpha_scale=2.0)
    model = torch.nn.Sequential(wrapped)
    with torch.no_grad():
        wrapped.lora_down.copy_(torch.randn(2, 8))
        wrapped.lora_up.copy_(torch.randn(6, 2))
    x = torch.randn(3, 8)
    expected = model(x)

    folded_count = fold_lora_adapters(model)

    assert folded_count == 1
    assert isinstance(model[0], torch.nn.Linear)
    assert not isinstance(model[0], LoRALinear)
    assert torch.allclose(model(x), expected, atol=1e-6)


def test_reset_parameters_after_lora_raises() -> None:
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=4),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    model.reset_parameters()
    apply_lora_to_stateless_transformer(
        model,
        LoRAConfig(rank=2, target_modules=("q",), target_block_count=1),
    )

    with pytest.raises(RuntimeError, match="before applying LoRA"):
        model.reset_parameters()


@pytest.mark.parametrize(
    ("action_spec", "launch_mode"),
    [
        (ActionPureConfig(max_per_planet_launches=1), None),
        (ActionDiscreteTargetsConfig(max_per_planet_launches=1), "binary"),
        (ActionDiscreteTargetsConfig(max_per_planet_launches=1), "binary_after"),
        (ActionDiscreteTargetsConfig(max_per_planet_launches=1), "target_token"),
        (ActionDiscreteTargetBinsConfig(n_bins=5), None),
    ],
)
def test_lora_head_wrapping_keeps_every_adapter_gradient_active(
    action_spec: ActionConfig, launch_mode: str | None
) -> None:
    # Guards the DDP path: LoRA models train with find_unused_parameters=False,
    # so every wrapped adapter (trunk plus value/policy heads) must receive a
    # gradient on each backward or multi-GPU training would hang on an unused
    # parameter. This exercises every actor variant.
    actor_cfg: dict[str, Any] = {"action_spec": action_spec.action_spec}
    if launch_mode is not None:
        actor_cfg["launch_mode"] = launch_mode
    if isinstance(action_spec, ActionDiscreteTargetBinsConfig):
        actor_cfg["n_bins"] = action_spec.n_bins
    obs_spec = EntityBasedConfig()
    config = StatelessTransformerV1Config(
        embed_dim=16,
        depth=2,
        n_heads=4,
        actor=actor_cfg,
        lora=LoRAConfig(
            rank=4,
            target_modules=("q", "up", "down"),
            target_value_head=True,
            target_policy_head=True,
        ),
    )
    model = StatelessTransformerV1(config, obs_spec=obs_spec, action_spec=action_spec)
    model.reset_parameters()
    assert config.lora is not None
    apply_lora_to_stateless_transformer(model, config.lora)

    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    evaluation = model.evaluate_actions(obs, model.forward(obs).actions)
    loss = (
        evaluation.values.sum()
        + evaluation.log_probs.per_player_entity.sum()
        + evaluation.entropies.per_player_entity.sum()
    )
    model.zero_grad()
    loss.backward()

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    # Only LoRA adapters are trainable; the base model stays fully frozen.
    assert trainable
    assert all(_is_lora_parameter(name) for name, _ in trainable)
    # Every adapter receives a gradient, so DDP has no unused parameter to hang on.
    assert all(parameter.grad is not None for _, parameter in trainable)


def test_actor_pure_config_requires_ordered_kappa_bounds() -> None:
    assert ActorPureConfig().kappa_max == 1_000_000.0
    with pytest.raises(ValueError, match="kappa_min must be <= kappa_max"):
        ActorPureConfig(kappa_min=2.0, kappa_max=1.0)


def test_actor_pure_config_uses_separate_mixture_counts() -> None:
    config = StatelessTransformerV1Config(
        embed_dim=16,
        depth=1,
        n_heads=4,
        actor=ActorPureConfig(n_angle_mixtures=3, n_fleet_size_mixtures=5),
    )
    actor = PureActor(
        config.actor,
        embed_dim=config.embed_dim,
        max_per_planet_launches=1,
        activation=config.activation,
    )
    slot_input = torch.zeros((2, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    max_launch = torch.full(slot_input.shape[:-1], 11, dtype=torch.int64)

    angle_params = actor._angle_params(_pure_actor_inputs(slot_input))
    params = actor._policy_params_for_angle(
        angle_params,
        _pure_actor_inputs(slot_input),
        max_launch,
        torch.zeros(slot_input.shape[:-1]),
        min_fleet_size=2,
    )

    assert params.angle_mix_logits.shape == (2, 4, ACTION_ENTITY_SLOTS, 3)
    assert params.loc.shape == (2, 4, ACTION_ENTITY_SLOTS, 3)
    assert params.size_mix_logits.shape == (2, 4, ACTION_ENTITY_SLOTS, 5)
    assert params.size_mu.shape == (2, 4, ACTION_ENTITY_SLOTS, 5)


def test_pure_actor_angle_attention_selects_directional_target_context() -> None:
    config = StatelessTransformerV1Config(embed_dim=2, depth=1, n_heads=1)
    actor = PureActor(
        ActorPureConfig(),
        embed_dim=config.embed_dim,
        max_per_planet_launches=1,
        activation=config.activation,
    )
    with torch.no_grad():
        for module in (actor.q, actor.k, actor.v, actor.out):
            module.weight.copy_(torch.eye(config.embed_dim))
            module.bias.zero_()
        actor.angle_direction_proj.input.weight.copy_(torch.eye(config.embed_dim))
        actor.angle_direction_proj.input.bias.zero_()
        actor.angle_direction_proj.output.up.weight.copy_(torch.eye(config.embed_dim))
        actor.angle_direction_proj.output.up.bias.zero_()
        actor.angle_direction_proj.output.weight.copy_(torch.eye(config.embed_dim))
        actor.angle_direction_proj.output.bias.zero_()
        actor.source_norm.weight.fill_(1.0)
        actor.source_norm.bias.zero_()
        actor.target_norm.weight.fill_(1.0)
        actor.target_norm.bias.zero_()

    source = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    target = torch.zeros_like(source)
    target[:, :, 1, 0] = 10.0
    target[:, :, 2, 1] = 10.0
    target_mask = torch.zeros((1, ACTION_ENTITY_SLOTS), dtype=torch.bool)
    target_mask[:, 1:3] = True
    actor_inputs = _pure_actor_inputs(source, target=target, target_mask=target_mask)

    right_context = actor._selected_angle_target_values(
        actor_inputs,
        torch.zeros((1, 4, ACTION_ENTITY_SLOTS)),
    )
    up_context = actor._selected_angle_target_values(
        actor_inputs,
        torch.full((1, 4, ACTION_ENTITY_SLOTS), math.pi / 2.0),
    )

    assert right_context[0, 0, 0, 1] > right_context[0, 0, 0, 0]
    assert up_context[0, 0, 0, 0] > up_context[0, 0, 0, 1]


def test_model_config_has_discriminator_tag() -> None:
    config = TypeAdapter(ModelConfig).validate_python(
        {
            "model_arch": "stateless_transformer_v1",
            "embed_dim": 32,
            "n_heads": 4,
        }
    )

    assert config.model_arch == "stateless_transformer_v1"
    assert not config.force_flash_attn


def test_model_config_loads_actor_subconfig_reference() -> None:
    config = StatelessTransformerV1Config.from_file(
        _REPO_ROOT / "configs" / "model" / "stateless_transformer_6m.yaml"
    )

    assert config.actor == ActorDiscreteTargetsConfig()


def test_actor_config_presets_match_defaults() -> None:
    pure = ActorPureConfig.from_file(
        _REPO_ROOT / "configs" / "model" / "actor" / "pure.yaml"
    )
    discrete_targets = ActorDiscreteTargetsConfig.from_file(
        _REPO_ROOT / "configs" / "model" / "actor" / "discrete_targets.yaml"
    )

    assert pure == ActorPureConfig()
    assert discrete_targets == ActorDiscreteTargetsConfig()


def test_model_config_requires_actor_action_spec_match() -> None:
    with pytest.raises(ValueError, match="actor config must match env action_spec"):
        StatelessTransformerV1(
            StatelessTransformerV1Config(actor={"action_spec": "pure"}),
            obs_spec=EntityBasedConfig(),
            action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
        )


def test_model_config_rejects_removed_angle_mixture_alias() -> None:
    with pytest.raises(ValueError, match="n_angle_mixtures"):
        TypeAdapter(ModelConfig).validate_python(
            {
                "model_arch": "stateless_transformer_v1",
                "n_angle_mixtures": 2,
            }
        )


@pytest.mark.parametrize("field_name", ["obs_spec", "action_spec"])
def test_model_config_rejects_env_owned_specs(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        TypeAdapter(ModelConfig).validate_python(
            {
                "model_arch": "stateless_transformer_v1",
                field_name: {},
            }
        )


def test_cross_attention_observation_spec_runs_forward_pass() -> None:
    torch.manual_seed(25)
    obs_spec = EntityBasedCrossAttnV1Config(max_entities=ACTION_ENTITY_SLOTS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(
        StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=4),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    assert obs.fleet_target is not None
    assert obs.target_incoming_features is not None
    obs.entity_mask[:, ACTION_ENTITY_SLOTS] = True
    obs.fleets[:, 0, 0] = 1.0
    obs.fleet_target[:, 0] = 1
    obs.target_incoming_features[:, 1, 2] = 0.01

    output = model(obs)

    assert output.values.shape == (2, OUTER_PLAYER_SLOTS)
    assert isinstance(output.actions, PureActions)


def test_cross_attention_observation_spec_ignores_force_flash_attn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(26)
    obs_spec = EntityBasedCrossAttnV1Config(max_entities=ACTION_ENTITY_SLOTS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(
        StatelessTransformerV1Config(
            embed_dim=16,
            depth=1,
            n_heads=4,
            force_flash_attn=True,
        ),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    assert obs.fleet_target is not None
    obs.entity_mask[:, ACTION_ENTITY_SLOTS] = True
    obs.fleets[:, 0, 0] = 1.0
    obs.fleet_target[:, 0] = 1

    def fail_flash_check(_tensor: torch.Tensor) -> bool:
        pytest.fail("cross-attention observation specs should not check flash-attn")

    monkeypatch.setattr(model_impl, "use_flash_attn", fail_flash_check)
    monkeypatch.setattr(
        model_impl,
        "_requires_flash_attn",
        lambda _tensor, *, force_flash_attn: force_flash_attn,
    )

    output = model(obs)

    assert output.values.shape == (2, OUTER_PLAYER_SLOTS)


def test_model_outputs_do_not_change_with_extra_masked_fleets() -> None:
    torch.manual_seed(148)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(
        StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=4),
        obs_spec=EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 4),
        action_spec=action_spec,
    )
    model.eval()
    compact = _obs_batch(
        batch_size=1,
        obs_spec=EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 1),
        action_spec=action_spec,
    )
    compact.fleets[0, 0, 0] = 1.0
    compact.fleets[0, 0, 3] = 0.25
    compact.entity_mask[0, ACTION_ENTITY_SLOTS] = True
    padded_entity_mask = torch.cat(
        (
            compact.entity_mask,
            torch.zeros((1, 3), dtype=compact.entity_mask.dtype),
        ),
        dim=1,
    )
    padded = ObsBatch(
        planets=compact.planets,
        orbiting_planets=compact.orbiting_planets,
        fleets=torch.cat(
            (
                compact.fleets,
                torch.full(
                    (1, 3, compact.fleets.shape[-1]),
                    123.0,
                    dtype=compact.fleets.dtype,
                ),
            ),
            dim=1,
        ),
        comets=compact.comets,
        entity_mask=padded_entity_mask,
        still_playing=compact.still_playing,
        global_features=compact.global_features,
        action_mask=compact.action_mask,
    )
    padded_zeroed = ObsBatch(
        planets=compact.planets,
        orbiting_planets=compact.orbiting_planets,
        fleets=torch.cat(
            (
                compact.fleets,
                torch.zeros(
                    (1, 3, compact.fleets.shape[-1]),
                    dtype=compact.fleets.dtype,
                ),
            ),
            dim=1,
        ),
        comets=compact.comets,
        entity_mask=padded_entity_mask,
        still_playing=compact.still_playing,
        global_features=compact.global_features,
        action_mask=compact.action_mask,
    )

    with torch.inference_mode():
        compact_output = model(compact, deterministic=True)
        padded_output = model(padded, deterministic=True)
        padded_zeroed_output = model(padded_zeroed, deterministic=True)

    cross_length_atol = 1e-5
    torch.testing.assert_close(
        compact_output.values, padded_output.values, atol=cross_length_atol, rtol=0
    )
    torch.testing.assert_close(
        compact_output.winner_probabilities,
        padded_output.winner_probabilities,
        atol=cross_length_atol,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.actions.launch,
        padded_output.actions.launch,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.actions.angle,
        padded_output.actions.angle,
        atol=cross_length_atol,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.actions.ships,
        padded_output.actions.ships,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.log_probs.launch,
        padded_output.log_probs.launch,
        atol=cross_length_atol,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.log_probs.event,
        padded_output.log_probs.event,
        atol=cross_length_atol,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.log_probs.per_player_entity,
        padded_output.log_probs.per_player_entity,
        atol=cross_length_atol,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.entropies.launch,
        padded_output.entropies.launch,
        atol=cross_length_atol,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.entropies.event,
        padded_output.entropies.event,
        atol=cross_length_atol,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.entropies.per_player_entity,
        padded_output.entropies.per_player_entity,
        atol=cross_length_atol,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_output.values,
        padded_zeroed_output.values,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_output.winner_probabilities,
        padded_zeroed_output.winner_probabilities,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_output.actions.launch,
        padded_zeroed_output.actions.launch,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_output.actions.angle,
        padded_zeroed_output.actions.angle,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_output.actions.ships,
        padded_zeroed_output.actions.ships,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_output.log_probs.launch,
        padded_zeroed_output.log_probs.launch,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_output.log_probs.event,
        padded_zeroed_output.log_probs.event,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_output.log_probs.per_player_entity,
        padded_zeroed_output.log_probs.per_player_entity,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_output.entropies.launch,
        padded_zeroed_output.entropies.launch,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_output.entropies.event,
        padded_zeroed_output.entropies.event,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_output.entropies.per_player_entity,
        padded_zeroed_output.entropies.per_player_entity,
        atol=0,
        rtol=0,
    )


def test_model_constructor_does_not_require_flash_attn_on_cuda_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    _model(StatelessTransformerV1Config(embed_dim=32, depth=1, n_heads=4))


def test_pack_sequence_removes_masked_tokens_and_unpack_restores_layout() -> None:
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, True],
        ]
    )

    packed_x, packed = pack_sequence(x, mask)
    unpacked = unpack_sequence(packed_x, packed)

    assert packed_x.tolist() == [
        x[0, 0].tolist(),
        x[0, 2].tolist(),
        x[1, 1].tolist(),
        x[1, 2].tolist(),
        x[1, 3].tolist(),
    ]
    assert packed.cu_seqlens.tolist() == [0, 2, 5]
    assert packed.max_seqlen == 3
    assert torch.equal(unpacked[mask], x[mask])
    assert torch.equal(unpacked[~mask], torch.zeros_like(unpacked[~mask]))


def test_pack_sequence_accepts_fixed_max_seqlen_capacity() -> None:
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, True],
        ]
    )

    packed_x, packed = pack_sequence(x, mask, max_seqlen=4)

    assert packed_x.shape == (5, 3)
    assert packed.max_seqlen == 4
    assert packed.seqlens.tolist() == [2, 3]
    assert packed.padded_seq_len == 4

    with pytest.raises(ValueError, match="max_seqlen must cover"):
        pack_sequence(x, mask, max_seqlen=2)


def test_pack_sequence_rejects_fully_masked_rows() -> None:
    x = torch.zeros((2, 3, 4))
    mask = torch.tensor([[True, False, False], [False, False, False]])

    with pytest.raises(ValueError, match="at least one unmasked token"):
        pack_sequence(x, mask)


def test_attention_and_swiglu_use_separate_projection_matrices_for_muon() -> None:
    config = StatelessTransformerV1Config(embed_dim=32, n_heads=4, activation="swiglu")

    attn = MultiHeadSelfAttention(config)
    mlp = FeedForward(config)
    output_head = OutputProjectionMLP(config, output_dim=3)
    pairwise_head = PairwiseBiasMLP(config)

    assert attn.q is not attn.k
    assert attn.k is not attn.v
    assert mlp.gate is not mlp.value
    assert output_head.gate is not output_head.value
    assert pairwise_head.gate is not pairwise_head.value
    assert pairwise_head.get_input_layers() == (
        pairwise_head.gate,
        pairwise_head.value,
    )
    pairwise_features = torch.randn((2, 3, 4, 6))
    assert pairwise_head(pairwise_features).shape == (2, 3, 4)


def test_non_flash_attention_uses_regular_shaped_sdpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=4)
    attn = MultiHeadSelfAttention(config)
    x = torch.randn((2, 5, config.embed_dim))
    mask = torch.tensor(
        [
            [True, False, True, True, False],
            [False, True, True, False, True],
        ]
    )

    def disable_flash_attn(q: torch.Tensor) -> bool:
        return q.numel() < 0

    monkeypatch.setattr(model_impl, "use_flash_attn", disable_flash_attn)

    def fail_varlen_attention(
        q: torch.Tensor,  # noqa: ARG001
        k: torch.Tensor,  # noqa: ARG001
        v: torch.Tensor,  # noqa: ARG001
        *,
        cu_seqlens: torch.Tensor,  # noqa: ARG001
        max_seqlen: int,  # noqa: ARG001
    ) -> torch.Tensor:
        raise AssertionError("non-flash attention should not pack through varlen")

    monkeypatch.setattr(model_impl, "varlen_attention", fail_varlen_attention)

    output = attn(x, mask, None)

    assert output.shape == x.shape


def test_unpacked_attention_uses_sdpa_when_projected_q_supports_flash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=4)
    attn = MultiHeadSelfAttention(config)
    x = torch.randn((2, 5, config.embed_dim))
    mask = torch.tensor(
        [
            [True, False, True, True, False],
            [False, True, True, False, True],
        ]
    )
    sdpa_calls = 0
    original_sdpa = F.scaled_dot_product_attention

    def projected_q_uses_flash(q: torch.Tensor) -> bool:
        return q.ndim == 4

    def fail_varlen_attention(
        q: torch.Tensor,  # noqa: ARG001
        k: torch.Tensor,  # noqa: ARG001
        v: torch.Tensor,  # noqa: ARG001
        *,
        cu_seqlens: torch.Tensor,  # noqa: ARG001
        max_seqlen: int,  # noqa: ARG001
    ) -> torch.Tensor:
        raise AssertionError("unpacked attention should use regular-shaped SDPA")

    def counted_sdpa(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal sdpa_calls
        sdpa_calls += 1
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(model_impl, "use_flash_attn", projected_q_uses_flash)
    monkeypatch.setattr(model_impl, "varlen_attention", fail_varlen_attention)
    monkeypatch.setattr(model_impl.F, "scaled_dot_product_attention", counted_sdpa)

    output = attn(x, mask, None)

    assert output.shape == x.shape
    assert sdpa_calls == 1


def test_flash_attention_uses_already_packed_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=4)
    attn = MultiHeadSelfAttention(config)
    x = torch.randn((2, 5, config.embed_dim))
    mask = torch.tensor(
        [
            [True, False, True, True, False],
            [False, True, True, False, True],
        ]
    )
    packed_x, packed = pack_sequence(x, mask)
    calls: list[tuple[torch.Size, list[int], int]] = []

    def enable_flash_attn(q: torch.Tensor) -> bool:
        return q.numel() > 0

    def fail_pack_tensor(
        _x: torch.Tensor,
        _packed: model_impl.PackedSequence,
    ) -> torch.Tensor:
        pytest.fail("attention should not pack tensors")

    def fail_unpack_sequence(
        _x: torch.Tensor,
        _packed: model_impl.PackedSequence,
    ) -> torch.Tensor:
        pytest.fail("attention should not unpack tensors")

    monkeypatch.setattr(model_impl, "use_flash_attn", enable_flash_attn)
    monkeypatch.setattr(model_impl, "pack_tensor", fail_pack_tensor)
    monkeypatch.setattr(model_impl, "unpack_sequence", fail_unpack_sequence)

    def fake_varlen_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        calls.append((q.shape, cu_seqlens.tolist(), max_seqlen))
        assert k.shape == q.shape
        assert v.shape == q.shape
        return torch.zeros_like(q)

    monkeypatch.setattr(model_impl, "varlen_attention", fake_varlen_attention)

    output = attn(packed_x, None, packed)

    assert output.shape == packed_x.shape
    assert calls == [(torch.Size((6, config.n_heads, 4)), [0, 3, 6], 3)]


def test_force_flash_attention_rejects_non_flash_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = StatelessTransformerV1Config(
        embed_dim=16,
        depth=1,
        n_heads=4,
        force_flash_attn=True,
    )
    attn = MultiHeadSelfAttention(config)
    x = torch.randn((2, 5, config.embed_dim))
    mask = torch.tensor(
        [
            [True, False, True, True, False],
            [False, True, True, False, True],
        ]
    )
    packed_x, packed = pack_sequence(x, mask)

    def disable_flash_attn(q: torch.Tensor) -> bool:
        return q.numel() < 0

    def fail_varlen_attention(
        q: torch.Tensor,  # noqa: ARG001
        k: torch.Tensor,  # noqa: ARG001
        v: torch.Tensor,  # noqa: ARG001
        *,
        cu_seqlens: torch.Tensor,  # noqa: ARG001
        max_seqlen: int,  # noqa: ARG001
    ) -> torch.Tensor:
        raise AssertionError("forced flash attention should fail before fallback")

    monkeypatch.setattr(model_impl, "use_flash_attn", disable_flash_attn)
    monkeypatch.setattr(model_impl, "varlen_attention", fail_varlen_attention)
    monkeypatch.setattr(
        model_impl,
        "_requires_flash_attn",
        lambda _tensor, *, force_flash_attn: force_flash_attn,
    )

    with pytest.raises(RuntimeError, match="force_flash_attn=True"):
        attn(packed_x, None, packed)


def test_force_flash_attention_is_ignored_for_cpu_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = StatelessTransformerV1Config(
        embed_dim=16,
        depth=1,
        n_heads=4,
        force_flash_attn=True,
    )
    attn = MultiHeadSelfAttention(config)
    x = torch.randn((2, 5, config.embed_dim))
    mask = torch.tensor(
        [
            [True, False, True, True, False],
            [False, True, True, False, True],
        ]
    )
    packed_x, packed = pack_sequence(x, mask)
    calls = 0

    def disable_flash_attn(q: torch.Tensor) -> bool:
        return q.numel() < 0

    def fake_varlen_attention(
        q: torch.Tensor,
        k: torch.Tensor,  # noqa: ARG001
        v: torch.Tensor,  # noqa: ARG001
        *,
        cu_seqlens: torch.Tensor,  # noqa: ARG001
        max_seqlen: int,  # noqa: ARG001
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.zeros_like(q)

    monkeypatch.setattr(model_impl, "use_flash_attn", disable_flash_attn)
    monkeypatch.setattr(model_impl, "varlen_attention", fake_varlen_attention)

    output = attn(packed_x, None, packed)

    assert output.shape == packed_x.shape
    assert calls == 1


def test_non_flash_encoder_does_not_build_or_pack_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=2,
        n_heads=4,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)

    def disable_flash_attn(q: torch.Tensor) -> bool:
        return q.numel() < 0

    def fail_build_packed_sequence(
        _token_mask: torch.Tensor,
    ) -> model_impl.PackedSequence:
        pytest.fail("SDPA path should not build seqlens")

    def fail_pack_tensor(
        _x: torch.Tensor,
        _packed: model_impl.PackedSequence,
    ) -> torch.Tensor:
        pytest.fail("SDPA path should not pack tensors")

    def fail_unpack_sequence(
        _x: torch.Tensor,
        _packed: model_impl.PackedSequence,
    ) -> torch.Tensor:
        pytest.fail("SDPA path should not unpack tensors")

    monkeypatch.setattr(model_impl, "use_flash_attn", disable_flash_attn)
    monkeypatch.setattr(model_impl, "build_packed_sequence", fail_build_packed_sequence)
    monkeypatch.setattr(model_impl, "pack_tensor", fail_pack_tensor)
    monkeypatch.setattr(model_impl, "unpack_sequence", fail_unpack_sequence)

    encoded = model.encode_observations(obs)

    assert encoded.hidden.shape == (2, obs_spec.max_entities + 17, 32)
    assert encoded.token_mask.shape == (2, obs_spec.max_entities + 17)


def test_count_non_masked_tokens_matches_encoder_token_mask() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    obs.still_playing[1, 2:] = False

    encoded = model.encode_observations(obs)

    assert model.count_non_masked_tokens(obs) == encoded.token_mask.sum()


def test_count_non_masked_tokens_includes_cross_attention_fleets() -> None:
    obs_spec = EntityBasedCrossAttnV1Config(max_entities=ACTION_ENTITY_SLOTS + 2)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    obs.entity_mask[:, ACTION_ENTITY_SLOTS] = True
    assert obs.fleet_target is not None
    obs.fleet_target[:, 0] = 0

    encoded = model.encode_observations(obs)
    fleet_tokens = obs.entity_mask[:, ACTION_ENTITY_SLOTS:].sum()

    assert model.count_non_masked_tokens(obs) == encoded.token_mask.sum() + fleet_tokens


def test_force_flash_encoder_rejects_non_flash_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        force_flash_attn=True,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)

    def disable_flash_attn(q: torch.Tensor) -> bool:
        return q.numel() < 0

    def fail_pack_sequence(
        _x: torch.Tensor,
        _token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, model_impl.PackedSequence]:
        pytest.fail("forced flash attention should fail before packing")

    monkeypatch.setattr(model_impl, "use_flash_attn", disable_flash_attn)
    monkeypatch.setattr(model_impl, "pack_sequence", fail_pack_sequence)
    monkeypatch.setattr(
        model_impl,
        "_requires_flash_attn",
        lambda _tensor, *, force_flash_attn: force_flash_attn,
    )

    with pytest.raises(RuntimeError, match="force_flash_attn=True"):
        model.encode_observations(obs)


def test_force_flash_encoder_ignores_cpu_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        force_flash_attn=True,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)

    def disable_flash_attn(q: torch.Tensor) -> bool:
        return q.numel() < 0

    def fail_pack_sequence(
        _x: torch.Tensor,
        _token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, model_impl.PackedSequence]:
        pytest.fail("CPU force_flash_attn should use the SDPA path")

    monkeypatch.setattr(model_impl, "use_flash_attn", disable_flash_attn)
    monkeypatch.setattr(model_impl, "pack_sequence", fail_pack_sequence)

    encoded = model.encode_observations(obs)

    assert encoded.hidden.shape == (2, obs_spec.max_entities + 17, 32)
    assert encoded.token_mask.shape == (2, obs_spec.max_entities + 17)


def test_orbiting_planets_select_separate_planet_projection() -> None:
    class PassBlock(nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            _token_mask: torch.Tensor | None,
            _packed: model_impl.PackedSequence | None,
        ) -> torch.Tensor:
            return x

    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 1)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(
        StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=4),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    model.blocks = nn.ModuleList([PassBlock()])
    with torch.no_grad():
        for layer in (
            model.static_planet_proj,
            model.orbit_planet_proj,
            model.fleet_proj,
            model.comet_proj,
            model.global_proj,
        ):
            layer.input.weight.zero_()
            layer.input.bias.zero_()
            layer.output.weight.zero_()
            layer.output.bias.zero_()
        model.static_planet_proj.input.weight[:16, 0] = torch.arange(16.0)
        model.static_planet_proj.output.weight[:, :16] = torch.eye(16)
        model.orbit_planet_proj.input.weight[:16, 0] = torch.arange(16.0).flip(0)
        model.orbit_planet_proj.output.weight[:, :16] = torch.eye(16)

    obs = _obs_batch(batch_size=1, obs_spec=obs_spec, action_spec=action_spec)
    obs.planets[:, 0, 0] = 1.0
    hidden_without_orbiting = model.encode_observations(obs).hidden
    obs.orbiting_planets[:, 0] = True
    hidden_with_orbiting = model.encode_observations(obs).hidden

    assert not torch.allclose(
        hidden_without_orbiting[:, 0],
        hidden_with_orbiting[:, 0],
    )


def test_flash_encoder_packs_once_before_trunk_and_unpacks_once_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=3,
        n_heads=4,
        force_flash_attn=True,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    original_pack_tensor = model_impl.pack_tensor
    original_unpack_sequence = model_impl.unpack_sequence
    pack_calls = 0
    unpack_calls = 0
    varlen_calls = 0
    varlen_max_seqlens: list[int] = []

    def enable_flash_attn(q: torch.Tensor) -> bool:
        return q.numel() > 0

    def counted_pack_tensor(
        x: torch.Tensor,
        packed: model_impl.PackedSequence,
    ) -> torch.Tensor:
        nonlocal pack_calls
        pack_calls += 1
        return original_pack_tensor(x, packed)

    def counted_unpack_sequence(
        x: torch.Tensor,
        packed: model_impl.PackedSequence,
    ) -> torch.Tensor:
        nonlocal unpack_calls
        unpack_calls += 1
        return original_unpack_sequence(x, packed)

    def fake_varlen_attention(
        q: torch.Tensor,
        k: torch.Tensor,  # noqa: ARG001
        v: torch.Tensor,  # noqa: ARG001
        *,
        cu_seqlens: torch.Tensor,  # noqa: ARG001
        max_seqlen: int,
    ) -> torch.Tensor:
        nonlocal varlen_calls
        varlen_calls += 1
        varlen_max_seqlens.append(max_seqlen)
        return torch.zeros_like(q)

    monkeypatch.setattr(model_impl, "use_flash_attn", enable_flash_attn)
    monkeypatch.setattr(model_impl, "pack_tensor", counted_pack_tensor)
    monkeypatch.setattr(model_impl, "unpack_sequence", counted_unpack_sequence)
    monkeypatch.setattr(model_impl, "varlen_attention", fake_varlen_attention)

    encoded = model.encode_observations(obs)

    assert encoded.hidden.shape == (2, obs_spec.max_entities + 17, 32)
    assert encoded.token_mask.shape == (2, obs_spec.max_entities + 17)
    assert pack_calls == 1
    assert unpack_calls == 1
    assert varlen_calls == config.depth
    expected_padded_seq_len = (
        obs_spec.max_entities
        + OUTER_PLAYER_SLOTS
        + 1
        + config.n_scratch_tokens
        + OUTER_PLAYER_SLOTS
        + OUTER_PLAYER_SLOTS
    )
    assert varlen_max_seqlens == [expected_padded_seq_len] * config.depth


def test_compile_transformer_trunk_uses_compiled_packed_trunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=2,
        n_heads=4,
        force_flash_attn=True,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    expected_padded_seq_len = (
        obs_spec.max_entities
        + OUTER_PLAYER_SLOTS
        + 1
        + config.n_scratch_tokens
        + OUTER_PLAYER_SLOTS
        + OUTER_PLAYER_SLOTS
    )
    compile_calls: list[dict[str, object]] = []
    run_calls = 0

    def enable_flash_attn(q: torch.Tensor) -> bool:
        return q.numel() > 0

    def fake_varlen_attention(
        q: torch.Tensor,
        k: torch.Tensor,  # noqa: ARG001
        v: torch.Tensor,  # noqa: ARG001
        *,
        cu_seqlens: torch.Tensor,  # noqa: ARG001
        max_seqlen: int,
    ) -> torch.Tensor:
        assert max_seqlen == expected_padded_seq_len
        return torch.zeros_like(q)

    def fake_compile(fn: Any, *args: object, **kwargs: object) -> Any:
        assert args == ()
        compile_calls.append(kwargs)

        def wrapped(
            x: torch.Tensor,
            token_mask: torch.Tensor | None,
            packed: model_impl.PackedSequence | None,
        ) -> torch.Tensor:
            nonlocal run_calls
            run_calls += 1
            assert token_mask is None
            assert packed is not None
            assert packed.max_seqlen == expected_padded_seq_len
            return fn(x, token_mask, packed)

        return wrapped

    monkeypatch.setattr(model_impl, "use_flash_attn", enable_flash_attn)
    monkeypatch.setattr(model_impl, "varlen_attention", fake_varlen_attention)
    eager = model.encode_observations(obs)
    state_keys = set(model.state_dict())
    monkeypatch.setattr(model_impl.torch, "compile", fake_compile)

    compiled = model.compile_transformer_trunk(mode="default")
    encoded = model.encode_observations(obs)

    assert compiled == 1
    assert compile_calls == [{"mode": "default", "dynamic": True}]
    assert run_calls == 1
    assert set(model.state_dict()) == state_keys
    torch.testing.assert_close(encoded.hidden, eager.hidden)
    assert torch.equal(encoded.token_mask, eager.token_mask)


def test_compile_transformer_trunk_rejects_unsupported_trunk_variants() -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    cross_action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    cross_attention_model = _model(
        StatelessTransformerV1Config(
            actor=ActorDiscreteTargetsConfig(),
            embed_dim=32,
            depth=1,
            n_heads=4,
        ),
        obs_spec=EntityBasedCrossAttnV1Config(max_entities=ACTION_ENTITY_SLOTS + 2),
        action_spec=cross_action_spec,
    )
    adapter_model = _model(
        StatelessTransformerV1Config(
            embed_dim=32,
            depth=2,
            n_heads=4,
            player_count_adapters_enabled=True,
            player_count_adapter_blocks=1,
        ),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=action_spec,
    )

    with pytest.raises(RuntimeError, match="does not support cross-attention"):
        cross_attention_model.compile_transformer_trunk(mode="default")
    with pytest.raises(RuntimeError, match="does not support player-count"):
        adapter_model.compile_transformer_trunk(mode="default")


def test_player_count_adapter_blocks_split_depth_and_packed_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PassBlock(nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            _token_mask: torch.Tensor | None,
            _packed: model_impl.PackedSequence | None,
        ) -> torch.Tensor:
            return x

    class RecordingBlock(nn.Module):
        def __init__(self, player_count: int) -> None:
            super().__init__()
            self.player_count = player_count

        def forward(
            self,
            x: torch.Tensor,
            token_mask: torch.Tensor | None,
            packed: model_impl.PackedSequence | None,
        ) -> torch.Tensor:
            assert token_mask is None
            assert packed is not None
            calls.append(
                (
                    self.player_count,
                    tuple(x.shape),
                    packed.cu_seqlens.tolist(),
                    packed.max_seqlen,
                    packed.batch_size,
                )
            )
            delta = torch.zeros_like(x)
            delta[:, 0] = float(self.player_count)
            return x + delta

    calls: list[tuple[int, tuple[int, ...], list[int], int, int]] = []
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 4)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    config = StatelessTransformerV1Config(
        embed_dim=16,
        depth=4,
        n_heads=4,
        n_scratch_tokens=0,
        player_count_adapters_enabled=True,
        player_count_adapter_blocks=2,
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    assert list(model.player_count_adapters.keys()) == ["2", "3", "4"]
    model.blocks = nn.ModuleList([PassBlock(), PassBlock()])
    for player_count, adapter in model.player_count_adapters.items():
        adapter.blocks = nn.ModuleList(
            [RecordingBlock(int(player_count)), RecordingBlock(int(player_count))]
        )
    model.final_norm = nn.Identity()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    obs = _obs_batch(batch_size=3, obs_spec=obs_spec, action_spec=action_spec)
    obs.still_playing = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    obs.entity_mask.zero_()
    obs.entity_mask[0, :2] = True
    obs.entity_mask[1, :3] = True
    obs.entity_mask[2, :4] = True

    monkeypatch.setattr(model_impl, "use_flash_attn", lambda _x: True)

    encoded = model.encode_observations(obs)

    assert len(model.blocks) == 2
    assert all(
        len(adapter.blocks) == 2 for adapter in model.player_count_adapters.values()
    )
    assert [call[0] for call in calls] == [2, 2, 3, 3, 4, 4]
    expected_seqlens = {2: 9, 3: 13, 4: 17}
    expected_max_seqlen = (
        obs_spec.max_entities
        + OUTER_PLAYER_SLOTS
        + 1
        + config.n_scratch_tokens
        + OUTER_PLAYER_SLOTS
        + OUTER_PLAYER_SLOTS
    )
    for player_count, shape, cu_seqlens, max_seqlen, batch_size in calls:
        seqlen = expected_seqlens[player_count]
        assert shape == (seqlen, config.embed_dim)
        assert cu_seqlens == [0, seqlen]
        assert max_seqlen == expected_max_seqlen
        assert batch_size == 1
    torch.testing.assert_close(
        encoded.hidden[:, 0, 0],
        torch.tensor([4.0, 6.0, 8.0]),
    )


def test_player_count_adapter_heads_route_actor_and_critic_by_alive_count() -> None:
    class ConstantCriticHead(nn.Module):
        def __init__(self, logits: list[float]) -> None:
            super().__init__()
            self.register_buffer(
                "logits",
                torch.tensor(logits, dtype=torch.float32).view(
                    1,
                    OUTER_PLAYER_SLOTS,
                    1,
                ),
            )

        def forward(self, player_hidden: torch.Tensor) -> torch.Tensor:
            return self.logits.to(
                device=player_hidden.device,
                dtype=player_hidden.dtype,
            ).expand(player_hidden.shape[0], -1, -1)

    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1, min_fleet_size=1)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        player_count_adapters_enabled=True,
        player_count_adapter_blocks=0,
        actor=ActorPureConfig(n_angle_mixtures=2, n_fleet_size_mixtures=2),
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    assert len(model.blocks) == 1
    assert list(model.player_count_adapters.keys()) == ["2", "3", "4"]
    for player_count, adapter in model.player_count_adapters.items():
        assert len(adapter.blocks) == 0
        assert isinstance(adapter.actor, PureActor)
        with torch.no_grad():
            adapter.actor.actor_heads.continue_head.out.weight.zero_()
            adapter.actor.actor_heads.continue_head.out.bias.fill_(
                10.0 if player_count == "2" else -10.0
            )
        if player_count == "2":
            adapter.critic_head = ConstantCriticHead([2.0, 0.0, 0.0, 0.0])
        elif player_count == "3":
            adapter.critic_head = ConstantCriticHead([0.0, 2.0, 0.0, 0.0])

    obs = _obs_batch(batch_size=3, obs_spec=obs_spec, action_spec=action_spec)
    obs.still_playing = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, False, False, False],
        ]
    )

    output = model(obs, deterministic=True)

    assert isinstance(output.actions, PureActions)
    assert output.actions.launch[0, 0, 0, 0]
    assert not output.actions.launch[1, 0, 0, 0]
    assert output.actions.launch[2, 0, 0, 0]
    assert output.winner_probabilities[0, 0] > output.winner_probabilities[0, 1]
    assert output.winner_probabilities[1, 1] > output.winner_probabilities[1, 0]
    assert output.winner_probabilities[2, 0] == 1.0


def test_autocast_encoder_packs_after_appending_player_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        force_flash_attn=True,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    original_pack_sequence = model_impl.pack_sequence
    pack_calls = 0

    def use_bfloat16_flash_attn(q: torch.Tensor) -> bool:
        return q.dtype == torch.bfloat16

    def counted_pack_sequence(
        x: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, model_impl.PackedSequence]:
        nonlocal pack_calls
        pack_calls += 1
        assert x.dtype == torch.bfloat16
        return original_pack_sequence(x, token_mask, max_seqlen=max_seqlen)

    def fake_varlen_attention(
        q: torch.Tensor,
        k: torch.Tensor,  # noqa: ARG001
        v: torch.Tensor,  # noqa: ARG001
        *,
        cu_seqlens: torch.Tensor,  # noqa: ARG001
        max_seqlen: int,  # noqa: ARG001
    ) -> torch.Tensor:
        assert q.dtype == torch.bfloat16
        return torch.zeros_like(q)

    monkeypatch.setattr(model_impl, "use_flash_attn", use_bfloat16_flash_attn)
    monkeypatch.setattr(model_impl, "pack_sequence", counted_pack_sequence)
    monkeypatch.setattr(model_impl, "varlen_attention", fake_varlen_attention)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        encoded = model.encode_observations(obs)

    assert encoded.hidden.shape == (2, obs_spec.max_entities + 17, 32)
    assert encoded.token_mask.shape == (2, obs_spec.max_entities + 17)
    assert pack_calls == 1


def test_model_initialization_sets_stable_rl_priors() -> None:
    torch.manual_seed(0)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=2,
        n_heads=4,
        actor=ActorPureConfig(n_angle_mixtures=2, n_fleet_size_mixtures=2),
    )
    model = _model(config)

    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            assert torch.allclose(module.weight, torch.ones_like(module.weight))
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    for module in model.get_input_layers():
        if isinstance(module, nn.Linear):
            assert module.bias is not None
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, nn.Parameter):
            assert torch.isfinite(module).all()
            assert not torch.allclose(module, torch.zeros_like(module))

    residual_gain = 1.0 / math.sqrt(2.0 * config.depth)
    for block in model.blocks:
        assert torch.allclose(
            block.attn.out.weight.norm(dim=1),
            torch.full((config.embed_dim,), residual_gain),
            atol=1e-6,
        )
        assert torch.allclose(
            block.mlp.down.weight.norm(dim=1),
            torch.full((config.embed_dim,), residual_gain),
            atol=1e-6,
        )

    assert torch.allclose(
        model.critic_head.weight.norm(dim=1),
        torch.tensor([1.0]),
        atol=1e-6,
    )
    output_layer_ids = {id(layer) for layer in model.get_output_layers()}
    assert id(model.critic_head.out) in output_layer_ids
    assert id(model.critic_head.up) not in output_layer_ids
    for head in (
        model.actor.actor_heads.continue_head,
        model.actor.actor_heads.angle_mix_head,
        model.actor.actor_heads.dir_head,
        model.actor.actor_heads.kappa_head,
        model.actor.actor_heads.size_mix_head,
        model.actor.actor_heads.mean_head,
        model.actor.actor_heads.scale_head,
    ):
        assert id(head.out) in output_layer_ids
        assert id(head.up) not in output_layer_ids
        assert torch.allclose(
            head.weight.norm(dim=1),
            torch.full((head.out_features,), 0.01),
            atol=1e-6,
        )

    slot_input = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    actor_inputs = _pure_actor_inputs(slot_input)
    max_launch = torch.full((1, 4, ACTION_ENTITY_SLOTS), 9, dtype=torch.int64)
    angle_params = model.actor._angle_params(actor_inputs)
    params = model.actor._policy_params_for_angle(
        angle_params,
        actor_inputs,
        max_launch,
        torch.zeros(max_launch.shape),
        min_fleet_size=1,
    )
    assert torch.allclose(
        params.continue_logits,
        torch.zeros_like(params.continue_logits),
    )
    assert torch.allclose(
        angle_params.angle_mix_logits,
        torch.zeros_like(angle_params.angle_mix_logits),
    )
    assert torch.allclose(
        params.size_mix_logits,
        torch.zeros_like(params.size_mix_logits),
    )
    expected_base_angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        params.loc.shape[-1] + 1,
    )[:-1]
    assert torch.allclose(
        torch.stack((torch.cos(params.loc[0, 0, 0]), torch.sin(params.loc[0, 0, 0]))),
        torch.stack((torch.cos(expected_base_angles), torch.sin(expected_base_angles))),
        atol=1e-6,
    )
    actor_config = config.actor
    assert isinstance(actor_config, ActorPureConfig)
    expected_concentration = (
        math.log(actor_config.kappa_min)
        + torch.sigmoid(torch.tensor(0.0))
        * (math.log(actor_config.kappa_max) - math.log(actor_config.kappa_min))
    ).exp()
    assert torch.allclose(
        params.kappa,
        torch.full_like(params.kappa, expected_concentration),
    )
    support_lo = torch.tensor(1.0)
    support_width = torch.tensor(9.0)
    scale_upper = torch.maximum(
        torch.tensor(actor_config.scale_max_abs_floor),
        support_width * actor_config.scale_max_frac,
    )
    expected_scale = (
        math.log(actor_config.scale_min)
        + torch.sigmoid(torch.tensor(0.0))
        * (scale_upper.log() - math.log(actor_config.scale_min))
    ).exp()
    assert torch.allclose(
        params.size_mu,
        torch.full_like(params.size_mu, support_lo + 0.5 * (9.0 - 1.0)),
    )
    assert torch.allclose(
        params.size_scale,
        torch.full_like(params.size_scale, expected_scale),
    )


def test_observation_encoder_returns_structured_token_fields() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 3)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(
        batch_size=2,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )

    encoded = model.encode_observations(obs)

    assert encoded.hidden.shape == (2, obs_spec.max_entities + 17, 32)
    assert encoded.token_mask.shape == (2, obs_spec.max_entities + 17)
    assert encoded.action_entity_hidden.shape == (2, ACTION_ENTITY_SLOTS, 32)
    assert encoded.player_hidden.shape == (2, OUTER_PLAYER_SLOTS, 32)
    assert encoded.global_feature_hidden.shape == (2, 1, 32)
    assert encoded.board_hidden.shape == (2, config.n_scratch_tokens, 32)
    assert encoded.actor_plan_hidden.shape == (2, OUTER_PLAYER_SLOTS, 32)
    assert encoded.critic_value_hidden.shape == (2, OUTER_PLAYER_SLOTS, 32)
    assert encoded.token_mask[:, -4:].all()
    assert encoded.token_mask[:, :MAX_PLANETS].sum().item() == 4
    assert encoded.token_mask[:, MAX_PLANETS].all()
    assert not torch.allclose(
        model.player_tokens[0],
        model.player_tokens[1],
    )


def test_entity_based_ext_v2_player_features_feed_player_tokens() -> None:
    obs_spec = EntityBasedExtV2Config(max_entities=MAX_PLANETS + MAX_COMETS + 3)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(
        StatelessTransformerV1Config(embed_dim=32, depth=1, n_heads=4),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    assert model.player_feature_proj is not None
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    assert obs.player_features is not None

    encoded_without_features = model.encode_observations(obs)
    obs.player_features[:, 0, 0] = 1.0
    encoded_with_features = model.encode_observations(obs)

    assert not torch.allclose(
        encoded_without_features.player_hidden[:, 0],
        encoded_with_features.player_hidden[:, 0],
    )
    with pytest.raises(ValueError, match="player_features are required"):
        model.encode_observations(obs.model_copy(update={"player_features": None}))


def test_entity_based_models_keep_legacy_state_dict_shape() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 3)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    config = StatelessTransformerV1Config(embed_dim=32, depth=1, n_heads=4)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    assert model.player_feature_proj is None
    assert not any(key.startswith("player_feature_proj.") for key in model.state_dict())

    reloaded = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    reloaded.load_state_dict(model.state_dict())


def test_existing_observation_specs_do_not_gain_cross_attention_state() -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    config = StatelessTransformerV1Config(embed_dim=32, depth=1, n_heads=4)
    for obs_spec in (
        EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 3),
        EntityBasedExtV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 3),
        EntityBasedExtV2Config(max_entities=MAX_PLANETS + MAX_COMETS + 3),
    ):
        model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
        assert model.target_incoming_proj is None
        assert not any(
            key.startswith(
                (
                    "target_incoming_proj.",
                    "fleet_cross_attn.",
                    "fleet_residual_mlps.",
                    "fleet_residual_norms.",
                )
            )
            for key in model.state_dict()
        )

        reloaded = _model(config, obs_spec=obs_spec, action_spec=action_spec)
        reloaded.load_state_dict(model.state_dict())


@pytest.mark.parametrize(
    ("config", "action_spec"),
    [
        (
            StatelessTransformerV1Config(
                embed_dim=32,
                depth=1,
                n_heads=4,
                actor=ActorPureConfig(n_angle_mixtures=2, n_fleet_size_mixtures=2),
            ),
            ActionPureConfig(min_fleet_size=1),
        ),
        (
            StatelessTransformerV1Config(
                embed_dim=32,
                depth=1,
                n_heads=4,
                actor=ActorDiscreteTargetsConfig(n_action_mixtures=2),
            ),
            ActionDiscreteTargetsConfig(min_fleet_size=1),
        ),
        (
            StatelessTransformerV1Config(
                embed_dim=32,
                depth=1,
                n_heads=4,
                actor=ActorDiscreteTargetsConfig(
                    launch_mode="target_token",
                    n_action_mixtures=2,
                ),
            ),
            ActionDiscreteTargetsConfig(min_fleet_size=1),
        ),
        (
            StatelessTransformerV1Config(
                embed_dim=32,
                depth=1,
                n_heads=4,
                use_learned_pairwise_bias=True,
                actor=ActorDiscreteTargetsConfig(n_action_mixtures=2),
            ),
            ActionDiscreteTargetsConfig(min_fleet_size=1),
        ),
        (
            StatelessTransformerV1Config(
                embed_dim=32,
                depth=1,
                n_heads=4,
                actor=ActorDiscreteTargetBinsConfig(n_bins=3),
            ),
            ActionDiscreteTargetBinsConfig(min_fleet_size=1, n_bins=3),
        ),
    ],
)
def test_model_accepts_compacted_action_entity_slots(
    config: StatelessTransformerV1Config,
    action_spec: ActionConfig,
) -> None:
    torch.manual_seed(17)
    action_entity_slots = 3
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _compacted_obs_batch(
        batch_size=2,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )

    encoded = model.encode_observations(
        obs,
        action_entity_slots=action_entity_slots,
    )
    output = model(obs, deterministic=True)
    serving = model.serve(obs, deterministic=True)
    evaluation = model.evaluate_actions(obs, output.actions)

    assert encoded.action_entity_hidden.shape == (2, action_entity_slots, 32)
    if isinstance(action_spec, ActionDiscreteTargetBinsConfig):
        expected_action_shape = (2, OUTER_PLAYER_SLOTS, action_entity_slots)
        assert isinstance(output.actions, DiscreteTargetBinActions)
        assert isinstance(serving.actions, DiscreteTargetBinActions)
        assert output.actions.target.shape == expected_action_shape
        assert serving.actions.target.shape == expected_action_shape
    else:
        expected_action_shape = (2, OUTER_PLAYER_SLOTS, action_entity_slots, 1)
        assert isinstance(output.actions, (PureActions, DiscreteTargetActions))
        assert isinstance(serving.actions, (PureActions, DiscreteTargetActions))
        assert output.actions.launch.shape == expected_action_shape
        assert serving.actions.launch.shape == expected_action_shape
    assert evaluation.log_probs.per_player_entity.shape == (
        2,
        OUTER_PLAYER_SLOTS,
        action_entity_slots,
    )
    assert evaluation.entropies.per_player_entity.shape == (
        2,
        OUTER_PLAYER_SLOTS,
        action_entity_slots,
    )


def test_actor_critic_outputs_action_tensors_log_probs_and_values() -> None:
    torch.manual_seed(0)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorPureConfig(n_angle_mixtures=2, n_fleet_size_mixtures=2),
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    obs.still_playing = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )

    output = model(obs)

    expected_action_shape = (2, 4, ACTION_ENTITY_SLOTS, 1)
    assert isinstance(output.actions, PureActions)
    assert output.actions.launch.shape == expected_action_shape
    assert output.actions.launch.dtype == torch.bool
    assert output.actions.angle.shape == expected_action_shape
    assert output.actions.angle.dtype == torch.float32
    assert output.actions.ships.shape == expected_action_shape
    assert output.actions.ships.dtype == torch.int64
    assert output.log_probs.launch.shape == expected_action_shape
    assert output.log_probs.event.shape == expected_action_shape
    assert output.log_probs.per_player_entity.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert output.entropies.launch.shape == expected_action_shape
    assert output.entropies.event.shape == expected_action_shape
    assert output.entropies.per_player_entity.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert torch.allclose(
        output.log_probs.per_player_entity,
        (output.log_probs.launch + output.log_probs.event).sum(dim=-1),
    )
    assert torch.allclose(
        output.entropies.per_player_entity,
        (output.entropies.launch + output.entropies.event).sum(dim=-1),
    )
    assert set(output.entropies.components) == {
        "angle",
        "event",
        "fleet_size_full",
        "fleet_size_logistic",
        "fleet_size_mixture",
        "launch",
    }
    assert torch.allclose(
        output.entropies.components["launch"],
        output.entropies.launch.sum(dim=-1),
    )
    assert torch.allclose(
        output.entropies.components["event"],
        output.entropies.event.sum(dim=-1),
    )
    assert torch.isfinite(output.entropies.per_player_entity).all()
    assert torch.all(output.entropies.launch >= 0)
    assert torch.all(output.entropies.event >= 0)
    assert output.values.shape == (2, 4)
    assert output.winner_probabilities.shape == (2, 4)
    assert torch.allclose(output.winner_probabilities.sum(dim=1), torch.ones(2))
    assert torch.all(output.winner_probabilities[~obs.still_playing] == 0)
    assert torch.all(output.actions.ships[~output.actions.launch] == 0)
    assert torch.all(output.actions.ships.sum(dim=-1) <= obs.action_mask.max_launch)

    evaluation = model.evaluate_actions(obs, output.actions)
    assert torch.allclose(evaluation.log_probs.launch, output.log_probs.launch)
    assert torch.allclose(
        evaluation.log_probs.event,
        output.log_probs.event,
    )
    assert torch.allclose(
        evaluation.log_probs.per_player_entity,
        output.log_probs.per_player_entity,
    )
    assert torch.allclose(evaluation.entropies.launch, output.entropies.launch)
    assert torch.allclose(
        evaluation.entropies.event,
        output.entropies.event,
    )
    assert torch.allclose(
        evaluation.entropies.per_player_entity,
        output.entropies.per_player_entity,
    )
    assert set(evaluation.entropies.components) == {
        "angle",
        "event",
        "fleet_size_full",
        "fleet_size_logistic",
        "fleet_size_mixture",
        "launch",
    }
    assert torch.allclose(
        evaluation.entropies.components["launch"],
        output.entropies.components["launch"],
    )
    assert torch.allclose(
        evaluation.entropies.components["event"],
        output.entropies.components["event"],
    )
    assert torch.allclose(evaluation.values, output.values)
    assert torch.allclose(evaluation.winner_probabilities, output.winner_probabilities)


def test_discrete_targets_actor_outputs_targets_and_replays_log_probs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionDiscreteTargetsConfig(
        max_per_planet_launches=1,
        min_fleet_size=2,
    )
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(
            n_action_mixtures=3,
            entropy_ship_quantiles=8,
        ),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    assert isinstance(model.actor, DiscreteTargetsActor)
    assert model.actor.n_heads == 1
    assert model.actor.head_dim == config.embed_dim

    def fail_all_size_params(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forward/replay should use selected target size params")

    monkeypatch.setattr(
        model.actor,
        "_all_size_params",
        fail_all_size_params,
        raising=False,
    )
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    obs.action_mask.max_launch[:, 0, 0] = 8
    obs.action_mask.max_launch[:, 1, 1] = 4
    obs.action_mask.max_launch[:, 2, MAX_PLANETS] = 3

    output = model(obs)

    expected_action_shape = (2, 4, ACTION_ENTITY_SLOTS, 1)
    assert isinstance(output.actions, DiscreteTargetActions)
    assert output.actions.launch.shape == expected_action_shape
    assert output.actions.target.shape == expected_action_shape
    assert output.actions.target.dtype == torch.int64
    assert output.actions.ships.shape == expected_action_shape
    assert output.log_probs.target is not None
    assert output.log_probs.target.shape == expected_action_shape
    assert set(output.entropies.components) == {
        "launch",
        "target",
        "fleet_size_full",
        "fleet_size_mixture",
        "fleet_size_logistic",
    }
    assert torch.allclose(
        output.entropies.components["launch"],
        output.entropies.launch.squeeze(-1),
    )
    assert output.entropies.target is not None
    assert torch.allclose(
        output.entropies.components["target"],
        output.entropies.target.squeeze(-1),
    )
    assert torch.allclose(
        output.entropies.components["fleet_size_full"],
        output.entropies.event.squeeze(-1),
    )
    assert output.entropies.components["fleet_size_mixture"].shape == (
        2,
        4,
        ACTION_ENTITY_SLOTS,
    )
    assert output.entropies.components["fleet_size_logistic"].shape == (
        2,
        4,
        ACTION_ENTITY_SLOTS,
    )
    launched_target = output.actions.target[..., 0].clamp(0, ACTION_ENTITY_SLOTS - 1)
    target_valid = obs.action_mask.can_act.gather(
        -1, launched_target.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.all(target_valid[output.actions.launch[..., 0]])
    launched_ships = output.actions.ships[output.actions.launch]
    assert torch.all(launched_ships >= action_spec.min_fleet_size)
    assert torch.all(output.actions.ships.sum(dim=-1) <= obs.action_mask.max_launch)

    evaluation = model.evaluate_actions(obs, output.actions)
    assert torch.allclose(evaluation.log_probs.launch, output.log_probs.launch)
    assert torch.allclose(evaluation.log_probs.target, output.log_probs.target)
    assert torch.allclose(
        evaluation.log_probs.event,
        output.log_probs.event,
    )
    assert torch.allclose(
        evaluation.log_probs.per_player_entity,
        output.log_probs.per_player_entity,
    )
    assert set(evaluation.entropies.components) == {
        "launch",
        "target",
        "fleet_size_full",
        "fleet_size_mixture",
        "fleet_size_logistic",
    }
    assert torch.allclose(
        evaluation.entropies.components["launch"],
        output.entropies.components["launch"],
    )
    assert torch.allclose(
        evaluation.entropies.components["target"],
        output.entropies.components["target"],
    )
    assert torch.allclose(
        evaluation.entropies.components["fleet_size_full"],
        output.entropies.components["fleet_size_full"],
    )
    assert torch.allclose(
        evaluation.entropies.components["fleet_size_mixture"],
        output.entropies.components["fleet_size_mixture"],
    )
    assert torch.allclose(
        evaluation.entropies.components["fleet_size_logistic"],
        output.entropies.components["fleet_size_logistic"],
    )


def test_discrete_targets_serving_path_skips_log_probs_and_entropies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(13)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionDiscreteTargetsConfig(
        max_per_planet_launches=1,
        min_fleet_size=2,
    )
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(
            n_action_mixtures=3,
            entropy_ship_quantiles=8,
        ),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    assert isinstance(model.actor, DiscreteTargetsActor)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    obs.action_mask.max_launch[:, 0, 0] = 8
    obs.action_mask.max_launch[:, 1, 1] = 4

    expected = model(obs, deterministic=True)

    def fail_log_prob(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("serving should not compute action log probs")

    def fail_entropy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("serving should not compute action entropies")

    monkeypatch.setattr(
        discrete_targets_impl,
        "discrete_action_log_probs",
        fail_log_prob,
    )
    monkeypatch.setattr(model.actor, "_policy_params_for_entropy", fail_entropy)

    output = model.serve(obs, deterministic=True)

    assert isinstance(output.actions, DiscreteTargetActions)
    assert torch.equal(output.actions.launch, expected.actions.launch)
    assert torch.equal(output.actions.target, expected.actions.target)
    assert torch.equal(output.actions.ships, expected.actions.ships)
    assert torch.allclose(output.values, expected.values)
    assert torch.allclose(output.winner_probabilities, expected.winner_probabilities)


def test_discrete_target_bins_serving_path_skips_log_probs_and_entropies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(17)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionDiscreteTargetBinsConfig(n_bins=7)
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetBinsConfig(n_bins=7),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    assert isinstance(model.actor, DiscreteTargetBinsActor)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)

    expected = model(obs, deterministic=True)

    def fail_log_prob(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("serving should not compute action log probs")

    def fail_entropy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("serving should not compute action entropies")

    monkeypatch.setattr(
        discrete_target_bins_impl,
        "discrete_target_bin_log_probs",
        fail_log_prob,
    )
    monkeypatch.setattr(model.actor, "_entropy", fail_entropy)

    output = model.serve(obs, deterministic=True)

    assert isinstance(output.actions, DiscreteTargetBinActions)
    assert torch.equal(output.actions.target, expected.actions.target)
    assert torch.equal(output.actions.fleet_bin, expected.actions.fleet_bin)
    assert torch.allclose(output.values, expected.values)
    assert torch.allclose(output.winner_probabilities, expected.winner_probabilities)


def test_discrete_target_bins_actor_outputs_bins_and_replays_log_probs() -> None:
    torch.manual_seed(17)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionDiscreteTargetBinsConfig(n_bins=7)
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetBinsConfig(n_bins=7),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)

    output = model(obs)

    expected_action_shape = (2, 4, ACTION_ENTITY_SLOTS)
    assert isinstance(output.actions, DiscreteTargetBinActions)
    assert output.actions.target.shape == expected_action_shape
    assert output.actions.fleet_bin.shape == expected_action_shape
    source_active = obs.action_mask.can_act.flatten(start_dim=-2).any(dim=-1)
    batch_index = torch.arange(2)[:, None, None]
    player_index = torch.arange(4)[None, :, None]
    source_index = torch.arange(ACTION_ENTITY_SLOTS)[None, None, :]
    selected = obs.action_mask.can_act[
        batch_index,
        player_index,
        source_index,
        output.actions.target,
        output.actions.fleet_bin,
    ]
    assert torch.all(selected[source_active])
    assert set(output.entropies.components) == {"target", "fleet_bin"}

    evaluation = model.evaluate_actions(obs, output.actions)
    assert torch.allclose(evaluation.log_probs.target, output.log_probs.target)
    assert torch.allclose(
        evaluation.log_probs.event,
        output.log_probs.event,
    )
    assert torch.allclose(
        evaluation.log_probs.per_player_entity,
        output.log_probs.per_player_entity,
    )

    invalid = DiscreteTargetBinActions(
        target=output.actions.target.clone(),
        fleet_bin=output.actions.fleet_bin.clone(),
    )
    invalid.fleet_bin[0, 0, 0] = 1
    obs.action_mask.can_act[0, 0, 0, :, 1] = False
    with pytest.raises(ValueError, match="valid target-bin pair"):
        model.evaluate_actions(obs, invalid)


def test_pairwise_action_features_use_action_entity_state() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 1)
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    obs = _obs_batch(batch_size=1, obs_spec=obs_spec, action_spec=action_spec)
    obs.planets.zero_()
    obs.comets.zero_()

    obs.planets[0, 0, 0] = 1.0
    obs.planets[0, 0, 5] = -1.0
    obs.planets[0, 0, 15] = 20.0 / 500.0

    obs.planets[0, 1, 4] = 1.0
    obs.planets[0, 1, 5] = 1.0
    obs.planets[0, 1, 13] = 10.0 / 100.0

    obs.planets[0, 2, 1] = 1.0
    obs.planets[0, 2, 6] = 1.0
    obs.planets[0, 2, 15] = 30.0 / 500.0

    obs.comets[0, 0, 2] = 1.0
    obs.comets[0, 0, 5] = 15.0 / 500.0
    obs.comets[0, 0, 52] = -1.0

    features = build_pairwise_action_features(obs)

    assert features.shape == (1, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS, 6)
    torch.testing.assert_close(
        features[0, 0, 1],
        torch.tensor(
            [
                1.0,
                1.0,
                0.0,
                0.0,
                2.0 / math.sqrt(8.0),
                1.0,
            ]
        ),
    )
    torch.testing.assert_close(
        features[0, 0, 2, :4],
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )
    torch.testing.assert_close(
        features[0, 0, 0, 2:],
        torch.tensor([1.0, 0.0, 0.0, 1.0 - 1.0 / math.sqrt(2.0)]),
    )
    torch.testing.assert_close(
        features[0, 0, MAX_PLANETS],
        torch.tensor(
            [
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0 - 1.0 / math.sqrt(2.0),
            ]
        ),
    )


def test_pairwise_action_features_match_simulator_channel_layout() -> None:
    obs = encode_python_observation(
        _python_obs(
            planets=[
                [0, 0, 0.0, 50.0, 2.0, 20, 3],
                [1, -1, 100.0, 50.0, 2.0, 10, 3],
                [2, 1, 50.0, 100.0, 2.0, 30, 3],
                [10, 2, 50.0, 50.0, 1.0, 15, 1],
                [11, -1, 25.0, 50.0, 1.0, 5, 1],
            ],
            comets=[
                {
                    "planet_ids": [10, 11],
                    "paths": [
                        [[50.0, 50.0], [0.0, 50.0], [50.0, 0.0]],
                        [[25.0, 50.0], [25.0, 100.0], [100.0, 100.0]],
                    ],
                    "path_index": 1,
                }
            ],
        ),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
    )

    features = build_pairwise_action_features(obs)

    torch.testing.assert_close(
        features[0, 0, 1],
        torch.tensor(
            [
                1.0,
                1.0,
                0.0,
                0.0,
                2.0 / math.sqrt(8.0),
                1.0,
            ]
        ),
    )
    torch.testing.assert_close(
        features[0, 0, MAX_PLANETS + 1],
        torch.tensor(
            [
                1.0,
                1.0,
                0.0,
                0.0,
                math.sqrt(1.25) / math.sqrt(8.0),
                1.0 - math.sqrt(0.8) / math.sqrt(2.0),
            ]
        ),
    )
    torch.testing.assert_close(
        features[0, 0, 2, :4],
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )
    torch.testing.assert_close(
        features[0, 0, MAX_PLANETS],
        torch.tensor(
            [
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0 - 1.0 / math.sqrt(2.0),
            ]
        ),
    )


def test_learned_pairwise_bias_config_is_discrete_only() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(),
        embed_dim=32,
        depth=1,
        n_heads=4,
        use_learned_pairwise_bias=True,
    )
    model = _model(
        config,
        action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
    )

    assert model.pairwise_bias_mlp is not None
    input_layer_ids = {id(layer) for layer in model.get_input_layers()}
    assert id(model.pairwise_bias_mlp.up) in input_layer_ids

    with pytest.raises(ValueError, match="requires a discrete target action_spec"):
        _model(
            StatelessTransformerV1Config(
                embed_dim=32,
                depth=1,
                n_heads=4,
                use_learned_pairwise_bias=True,
            ),
            action_spec=ActionPureConfig(max_per_planet_launches=1),
        )


def test_swiglu_learned_pairwise_bias_input_layers_are_wired_to_model() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(),
        embed_dim=32,
        depth=1,
        n_heads=4,
        activation="swiglu",
        use_learned_pairwise_bias=True,
    )
    model = _model(
        config,
        action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
    )

    assert model.pairwise_bias_mlp is not None
    input_layer_ids = {id(layer) for layer in model.get_input_layers()}
    assert id(model.pairwise_bias_mlp.gate) in input_layer_ids
    assert id(model.pairwise_bias_mlp.value) in input_layer_ids


def test_discrete_targets_actor_adds_pairwise_bias_before_masking() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    _zero_target_attention(actor)
    slot_input = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    can_act = torch.zeros(
        (1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS),
        dtype=torch.bool,
    )
    can_act[0, 0, 0, 1] = True
    pairwise_bias = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS))
    pairwise_bias[0, 0, 0, 0] = 9.0
    pairwise_bias[0, 0, 0, 1] = 3.0
    pairwise_bias[0, 0, 1, 2] = 5.0

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        selection = actor._selection_params(
            _discrete_actor_inputs(slot_input, pairwise_bias=pairwise_bias),
            can_act,
        )

    assert selection.target_logits.dtype == torch.bfloat16
    assert selection.target_logits[0, 0, 0, 0] == torch.finfo(torch.bfloat16).min
    assert selection.target_logits[0, 0, 0, 1] == torch.tensor(
        3.0,
        dtype=torch.bfloat16,
    )
    assert selection.target_logits[0, 0, 1].eq(0).all()


def test_discrete_targets_target_token_mode_appends_zero_pairwise_bias() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(launch_mode="target_token"),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    _zero_target_attention(actor)
    slot_input = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    can_act = torch.zeros(
        (1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS),
        dtype=torch.bool,
    )
    can_act[0, 0, 0, 1] = True
    pairwise_bias = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS))
    pairwise_bias[0, 0, 0, 1] = 3.0

    selection = actor._selection_params(
        _discrete_actor_inputs(slot_input, pairwise_bias=pairwise_bias),
        can_act,
    )

    assert selection.target_logits.shape[-1] == ACTION_ENTITY_SLOTS + 1
    assert selection.target_logits[0, 0, 0, 1] == 3.0
    assert selection.target_logits[0, 0, 0, ACTION_ENTITY_SLOTS] == 0.0
    assert selection.target_logits[0, 0, 1].eq(0).all()


def test_discrete_targets_teacher_kl_skips_target_and_size_for_no_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(),
        embed_dim=16,
        depth=1,
        n_heads=4,
    )
    student = DiscreteTargetsActor(config.actor, transformer_config=config)
    teacher = DiscreteTargetsActor(config.actor, transformer_config=config)
    _zero_target_attention(student)
    _zero_target_attention(teacher)
    slot_input = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    can_act = torch.zeros(
        (1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS),
        dtype=torch.bool,
    )
    can_act[0, 0, 0, 1] = True
    max_launch = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
    max_launch[0, 0, 0] = 10
    actions = DiscreteTargetActions(
        launch=torch.zeros((1, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.bool),
        target=torch.ones((1, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.int64),
        ships=torch.zeros((1, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.int64),
    )
    size_kl_rows: list[int] = []
    original_size_kl = discrete_targets_impl.logistic_mixture_kl

    def recording_size_kl(*args: Any, **kwargs: Any) -> torch.Tensor:
        residual_budget = args[6]
        assert isinstance(residual_budget, torch.Tensor)
        size_kl_rows.append(residual_budget.numel())
        return original_size_kl(*args, **kwargs)

    monkeypatch.setattr(
        discrete_targets_impl,
        "logistic_mixture_kl",
        recording_size_kl,
    )

    kl = student.kl_divergence(
        _discrete_actor_inputs(slot_input),
        teacher,
        _discrete_actor_inputs(slot_input),
        can_act,
        max_launch,
        actions,
        min_fleet_size=6,
    )

    assert torch.allclose(kl.target, torch.zeros_like(kl.target))
    assert torch.allclose(kl.event, torch.zeros_like(kl.event))
    assert torch.allclose(kl.per_player_entity, kl.launch.squeeze(-1))
    assert size_kl_rows == [0]


def test_discrete_targets_kl_from_teacher_params_matches_kl_divergence() -> None:
    torch.manual_seed(7)
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(),
        embed_dim=16,
        depth=1,
        n_heads=4,
    )
    student = DiscreteTargetsActor(config.actor, transformer_config=config)
    teacher = DiscreteTargetsActor(config.actor, transformer_config=config)
    student_inputs = _discrete_actor_inputs(
        torch.randn((2, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    )
    teacher_inputs = _discrete_actor_inputs(
        torch.randn((2, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    )
    can_act = torch.zeros(
        (2, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS),
        dtype=torch.bool,
    )
    can_act[:, 0, 0, 1] = True
    can_act[:, 0, 1, 0] = True
    max_launch = torch.zeros((2, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
    max_launch[:, 0, 0] = 10
    max_launch[:, 0, 1] = 8
    actions = DiscreteTargetActions(
        launch=torch.ones((2, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.bool),
        target=torch.ones((2, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.int64),
        ships=torch.zeros((2, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.int64),
    )

    reference = student.kl_divergence(
        student_inputs,
        teacher,
        teacher_inputs,
        can_act,
        max_launch,
        actions,
        min_fleet_size=6,
    )
    teacher_params = teacher.teacher_policy_params(
        teacher_inputs,
        can_act,
        max_launch,
        actions,
        min_fleet_size=6,
    )
    split = student.kl_divergence_from_teacher_params(
        student_inputs,
        teacher_params,
        can_act,
        max_launch,
        actions,
        min_fleet_size=6,
    )

    assert torch.allclose(reference.per_player_entity, split.per_player_entity)
    assert torch.allclose(reference.launch, split.launch)
    assert torch.allclose(reference.event, split.event)
    assert reference.target is not None
    assert split.target is not None
    assert torch.allclose(reference.target, split.target)
    assert reference.components.keys() == split.components.keys()
    for key, value in reference.components.items():
        assert torch.allclose(value, split.components[key])


def test_cached_teacher_distillation_matches_inline_teacher() -> None:
    torch.manual_seed(13)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionDiscreteTargetsConfig(
        max_per_planet_launches=1,
        min_fleet_size=2,
    )
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(n_action_mixtures=3, entropy_ship_quantiles=8),
        embed_dim=32,
        depth=2,
        n_heads=4,
    )
    student = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    teacher = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    student.eval()
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    obs.action_mask.max_launch[:, 0, 0] = 8
    obs.action_mask.max_launch[:, 1, 1] = 4
    output = student(obs)

    inline = student.evaluate_actions_with_teacher(
        obs,
        output.actions,
        teacher,
        compute_teacher_action_kl=True,
        compute_teacher_value=True,
    )
    cached_targets = teacher.compute_teacher_distillation_targets(
        obs,
        output.actions,
        compute_action_kl=True,
        compute_value=True,
    )
    cached = student.evaluate_actions_with_cached_teacher(
        obs,
        output.actions,
        cached_targets,
        compute_teacher_action_kl=True,
        compute_teacher_value=True,
    )

    assert inline.action_kl is not None
    assert cached.action_kl is not None
    assert torch.allclose(
        inline.action_kl.per_player_entity,
        cached.action_kl.per_player_entity,
    )
    assert torch.allclose(inline.action_kl.launch, cached.action_kl.launch)
    assert torch.allclose(inline.action_kl.event, cached.action_kl.event)
    assert inline.action_kl.target is not None
    assert cached.action_kl.target is not None
    assert torch.allclose(inline.action_kl.target, cached.action_kl.target)
    assert inline.action_kl.components.keys() == cached.action_kl.components.keys()
    for key, value in inline.action_kl.components.items():
        assert torch.allclose(value, cached.action_kl.components[key])

    assert inline.teacher_winner_probabilities is not None
    assert cached.teacher_winner_probabilities is not None
    assert torch.allclose(
        inline.teacher_winner_probabilities,
        cached.teacher_winner_probabilities,
    )
    assert inline.student_winner_log_probabilities is not None
    assert cached.student_winner_log_probabilities is not None
    assert torch.allclose(
        inline.student_winner_log_probabilities,
        cached.student_winner_log_probabilities,
    )
    assert torch.allclose(
        inline.student.log_probs.per_player_entity,
        cached.student.log_probs.per_player_entity,
    )
    assert torch.allclose(inline.student.values, cached.student.values)


def test_cached_teacher_distillation_value_only_skips_action_params() -> None:
    torch.manual_seed(17)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionDiscreteTargetsConfig(
        max_per_planet_launches=1,
        min_fleet_size=2,
    )
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(),
        embed_dim=16,
        depth=1,
        n_heads=4,
    )
    student = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    teacher = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    student.eval()
    teacher.eval()
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    output = student(obs)

    cached_targets = teacher.compute_teacher_distillation_targets(
        obs,
        output.actions,
        compute_action_kl=False,
        compute_value=True,
    )
    assert cached_targets.action_params is None
    assert cached_targets.winner_probabilities is not None

    cached = student.evaluate_actions_with_cached_teacher(
        obs,
        output.actions,
        cached_targets,
        compute_teacher_action_kl=False,
        compute_teacher_value=True,
    )
    assert cached.action_kl is None
    assert cached.teacher_winner_probabilities is not None


def test_supports_cached_teacher_distillation_by_actor() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    targets_model = _model(
        StatelessTransformerV1Config(
            actor=ActorDiscreteTargetsConfig(),
            embed_dim=16,
            depth=1,
            n_heads=4,
        ),
        obs_spec=obs_spec,
        action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
    )
    assert targets_model.supports_cached_teacher_distillation()
    assert targets_model.supports_cached_value_distillation()
    # Non-discrete-targets actors (pure, and likewise discrete_target_bins) do not
    # implement the cached action-KL path, but value distillation is
    # actor-agnostic, so it stays supported.
    pure_model = _model(
        StatelessTransformerV1Config(
            actor=ActorPureConfig(),
            embed_dim=16,
            depth=1,
            n_heads=4,
        ),
        obs_spec=obs_spec,
    )
    assert not pure_model.supports_cached_teacher_distillation()
    assert pure_model.supports_cached_value_distillation()


def test_logistic_mixture_kl_handles_empty_budget() -> None:
    shape = (0, 2)
    logistic_kl = logistic_mixture_impl.logistic_mixture_kl(
        torch.empty(shape),
        torch.empty(shape),
        torch.empty(shape),
        torch.empty(shape),
        torch.empty(shape),
        torch.empty(shape),
        torch.empty((0,), dtype=torch.int64),
        min_fleet_size=3,
    )

    assert logistic_kl.shape == (0,)


def test_logistic_mixture_kl_ignores_component_permutation() -> None:
    teacher_mix_logits = torch.tensor([[2.0, -0.5]])
    teacher_mu = torch.tensor([[2.0, 8.0]])
    teacher_scale = torch.tensor([[0.7, 1.3]])
    student_mix_logits = teacher_mix_logits.flip(-1)
    student_mu = teacher_mu.flip(-1)
    student_scale = teacher_scale.flip(-1)

    logistic_kl = logistic_mixture_impl.logistic_mixture_kl(
        teacher_mix_logits,
        teacher_mu,
        teacher_scale,
        student_mix_logits,
        student_mu,
        student_scale,
        torch.tensor([12], dtype=torch.int64),
        min_fleet_size=1,
    )

    assert torch.allclose(logistic_kl, torch.zeros_like(logistic_kl), atol=1e-6)


def test_angle_policy_kl_ignores_component_permutation() -> None:
    teacher_mix_logits = torch.tensor([[1.5, -0.25]])
    teacher_loc = torch.tensor([[0.2, 2.4]])
    teacher_kappa = torch.tensor([[3.0, 6.0]])
    student_mix_logits = teacher_mix_logits.flip(-1)
    student_loc = teacher_loc.flip(-1)
    student_kappa = teacher_kappa.flip(-1)
    zeros = torch.zeros_like(teacher_loc)
    ones = torch.ones_like(teacher_loc)
    teacher_params = PolicyParams(
        continue_logits=torch.zeros(teacher_mix_logits.shape[:-1]),
        angle_mix_logits=teacher_mix_logits,
        angle_log_w=F.log_softmax(teacher_mix_logits, dim=-1),
        loc=teacher_loc,
        kappa=teacher_kappa,
        size_mix_logits=zeros,
        size_mu=zeros,
        size_scale=ones,
    )
    student_params = PolicyParams(
        continue_logits=torch.zeros(student_mix_logits.shape[:-1]),
        angle_mix_logits=student_mix_logits,
        angle_log_w=F.log_softmax(student_mix_logits, dim=-1),
        loc=student_loc,
        kappa=student_kappa,
        size_mix_logits=zeros,
        size_mu=zeros,
        size_scale=ones,
    )

    angle_kl = pure_actor_impl.angle_policy_kl(teacher_params, student_params)

    assert torch.allclose(angle_kl, torch.zeros_like(angle_kl), atol=1e-6)


def test_angle_policy_kl_resolves_sharp_component() -> None:
    teacher_mix_logits = torch.tensor([[0.0]])
    teacher_loc = torch.tensor([[0.001]])
    teacher_kappa = torch.tensor([[1_000_000.0]])
    student_loc = torch.tensor([[0.002]])
    zeros = torch.zeros_like(teacher_mix_logits)
    ones = torch.ones_like(teacher_mix_logits)
    teacher_params = PolicyParams(
        continue_logits=torch.zeros(teacher_mix_logits.shape[:-1]),
        angle_mix_logits=teacher_mix_logits,
        angle_log_w=F.log_softmax(teacher_mix_logits, dim=-1),
        loc=teacher_loc,
        kappa=teacher_kappa,
        size_mix_logits=zeros,
        size_mu=zeros,
        size_scale=ones,
    )
    student_params = PolicyParams(
        continue_logits=torch.zeros(teacher_mix_logits.shape[:-1]),
        angle_mix_logits=teacher_mix_logits,
        angle_log_w=F.log_softmax(teacher_mix_logits, dim=-1),
        loc=student_loc,
        kappa=teacher_kappa,
        size_mix_logits=zeros,
        size_mu=zeros,
        size_scale=ones,
    )

    angle_kl = pure_actor_impl.angle_policy_kl(teacher_params, student_params)
    expected_kappa = teacher_kappa.double()
    i1_over_i0 = torch.special.i1e(expected_kappa) / torch.special.i0e(expected_kappa)
    expected = (
        expected_kappa
        * i1_over_i0
        * (1.0 - torch.cos(teacher_loc.double() - student_loc.double()))
    )

    assert torch.allclose(
        angle_kl,
        expected.squeeze(-1).to(dtype=angle_kl.dtype),
        rtol=1e-2,
        atol=1e-3,
    )


def test_angle_policy_kl_matches_closed_form_for_near_aligned_sharp_component() -> None:
    # Regression guard for float32 catastrophic cancellation: at large kappa with
    # a tiny teacher/student loc difference, kappa*cos(theta-loc) loses all
    # precision in float32 (cos of the small difference rounds to 1.0), inflating
    # the KL ~100%. The quadrature must match the closed-form Von Mises KL because
    # the angle math now runs in float64. dloc=1e-4 is the regime float32 breaks
    # (the dloc=1e-3 case above stays accurate even in float32, so it cannot guard
    # this bug).
    teacher_mix_logits = torch.tensor([[0.0]])
    teacher_loc = torch.tensor([[0.0]])
    student_loc = torch.tensor([[1e-4]])
    kappa = torch.tensor([[1_000_000.0]])
    zeros = torch.zeros_like(teacher_mix_logits)
    ones = torch.ones_like(teacher_mix_logits)
    teacher_params = PolicyParams(
        continue_logits=torch.zeros(teacher_mix_logits.shape[:-1]),
        angle_mix_logits=teacher_mix_logits,
        angle_log_w=F.log_softmax(teacher_mix_logits, dim=-1),
        loc=teacher_loc,
        kappa=kappa,
        size_mix_logits=zeros,
        size_mu=zeros,
        size_scale=ones,
    )
    student_params = PolicyParams(
        continue_logits=torch.zeros(teacher_mix_logits.shape[:-1]),
        angle_mix_logits=teacher_mix_logits,
        angle_log_w=F.log_softmax(teacher_mix_logits, dim=-1),
        loc=student_loc,
        kappa=kappa,
        size_mix_logits=zeros,
        size_mu=zeros,
        size_scale=ones,
    )

    angle_kl = pure_actor_impl.angle_policy_kl(teacher_params, student_params)
    expected_kappa = kappa.double()
    i1_over_i0 = torch.special.i1e(expected_kappa) / torch.special.i0e(expected_kappa)
    expected = (
        expected_kappa
        * i1_over_i0
        * (1.0 - torch.cos(teacher_loc.double() - student_loc.double()))
    ).squeeze(-1)

    assert torch.allclose(
        angle_kl.double(),
        expected,
        rtol=0.1,
    )


def test_discrete_target_bins_actor_adds_pairwise_bias_before_masking() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetBinsConfig(n_bins=3),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    actor = DiscreteTargetBinsActor(config.actor, transformer_config=config)
    _zero_target_attention(actor)
    slot_input = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    can_act = torch.zeros(
        (1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS, 3),
        dtype=torch.bool,
    )
    can_act[0, 0, 0, 1, 2] = True
    pairwise_bias = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS))
    pairwise_bias[0, 0, 0, 0] = 9.0
    pairwise_bias[0, 0, 0, 1] = 4.0
    pairwise_bias[0, 0, 1, 2] = 5.0

    selection = actor._selection_params(
        _discrete_actor_inputs(slot_input, pairwise_bias=pairwise_bias),
        can_act,
    )

    assert selection.target_logits[0, 0, 0, 0] == torch.finfo(torch.float32).min
    assert selection.target_logits[0, 0, 0, 1] == 4.0
    assert selection.target_logits[0, 0, 1].eq(0).all()


def test_discrete_targets_output_layers_include_only_second_head_layers() -> None:
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorDiscreteTargetsConfig(),
    )
    model = _model(
        config,
        action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
    )

    output_layer_ids = {id(layer) for layer in model.get_output_layers()}

    assert id(model.critic_head.out) in output_layer_ids
    assert id(model.critic_head.up) not in output_layer_ids
    for head in (
        model.actor.continue_head,
        model.actor.mix_head,
        model.actor.mean_head,
        model.actor.scale_head,
    ):
        assert id(head.out) in output_layer_ids
        assert id(head.up) not in output_layer_ids


def test_learned_token_embeddings_are_input_layers() -> None:
    pure_config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorPureConfig(),
    )
    pure_model = _model(
        pure_config,
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )

    pure_input_layer_ids = {id(layer) for layer in pure_model.get_input_layers()}

    for layer in (
        pure_model.player_tokens,
        pure_model.board_tokens,
        pure_model.actor_plan_tokens,
        pure_model.critic_value_tokens,
        pure_model.actor.actor_heads.base_dirs,
    ):
        assert id(layer) in pure_input_layer_ids

    expected_angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        pure_model.actor.actor_heads.base_dirs.shape[0] + 1,
    )[:-1]
    expected_dirs = torch.stack(
        (torch.cos(expected_angles), torch.sin(expected_angles)),
        dim=-1,
    )
    assert torch.allclose(
        pure_model.actor.actor_heads.base_dirs,
        expected_dirs,
        atol=1e-6,
    )

    discrete_config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorDiscreteTargetsConfig(),
    )
    discrete_model = _model(
        discrete_config,
        action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
    )

    discrete_input_layer_ids = {
        id(layer) for layer in discrete_model.get_input_layers()
    }

    for layer in (
        discrete_model.player_tokens,
        discrete_model.board_tokens,
        discrete_model.actor_plan_tokens,
        discrete_model.critic_value_tokens,
        discrete_model.actor.source_role,
        discrete_model.actor.target_role,
    ):
        assert id(layer) in discrete_input_layer_ids


def test_discrete_targets_actor_masks_target_logits_under_bfloat16_autocast() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    slot_input = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    can_act = torch.zeros(
        (1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS),
        dtype=torch.bool,
    )
    can_act[0, 0, 0, 1] = True

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        selection = actor._selection_params(_discrete_actor_inputs(slot_input), can_act)

    assert selection.target_logits.dtype == torch.bfloat16
    assert selection.target_logits[0, 0, 0, 0] == torch.finfo(torch.bfloat16).min
    assert selection.target_logits[0, 0, 1].eq(0).all()


def test_discrete_targets_default_binary_mode_has_no_no_launch_token() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)

    assert actor.no_launch_target is None
    assert "no_launch_target" not in actor.state_dict()
    assert actor.continue_source_proj is not None
    assert actor.continue_head is not None


def test_discrete_targets_binary_after_mode_has_target_conditioned_continue_head() -> (
    None
):
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(
            launch_mode="binary_after",
            n_action_mixtures=1,
        ),
        embed_dim=4,
        depth=1,
        n_heads=1,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    assert actor.no_launch_target is None
    assert actor.continue_source_proj is None
    assert actor.continue_head is not None
    assert "continue_source_proj.weight" not in actor.state_dict()
    assert "continue_head.out.weight" in actor.state_dict()

    with torch.no_grad():
        for module in (actor.out, actor.size_pair_proj):
            module.weight.copy_(torch.eye(config.embed_dim))
            module.bias.zero_()
        for parameter in actor.mlp.parameters():
            parameter.zero_()
        actor.continue_head.up.weight.copy_(torch.eye(config.embed_dim))
        actor.continue_head.up.bias.zero_()
        actor.continue_head.out.weight.zero_()
        actor.continue_head.out.bias.zero_()
        actor.continue_head.out.weight[0, 0] = 1.0

    target_values = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    target_values[0, 0, 1, 0] = 2.0
    target_values[0, 0, 2, 0] = -2.0
    selection = DiscreteTargetSelectionParams(
        target_logits=torch.zeros((1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS)),
        target_values=target_values,
    )
    source_input = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    max_launch = torch.full((1, 4, ACTION_ENTITY_SLOTS), 10, dtype=torch.int64)

    target_one = actor._policy_params_for_selected_target(
        selection,
        source_input,
        max_launch,
        torch.ones((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64),
        min_fleet_size=1,
    )
    target_two = actor._policy_params_for_selected_target(
        selection,
        source_input,
        max_launch,
        torch.full((1, 4, ACTION_ENTITY_SLOTS), 2, dtype=torch.int64),
        min_fleet_size=1,
    )

    assert target_one.continue_logits is not None
    assert target_two.continue_logits is not None
    assert target_one.continue_logits[0, 0, 0] > target_two.continue_logits[0, 0, 0]


def test_discrete_targets_binary_after_keeps_no_launch_target_log_prob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(
            launch_mode="binary_after",
            n_action_mixtures=1,
        ),
        embed_dim=4,
        depth=1,
        n_heads=1,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    batch_shape = (1, 4, ACTION_ENTITY_SLOTS)
    target_logits = torch.full((*batch_shape, ACTION_ENTITY_SLOTS), -10.0)
    target_logits[0, 0, 0, 2] = 3.0
    selection = DiscreteTargetSelectionParams(
        target_logits=target_logits,
        target_values=torch.zeros((*batch_shape, config.embed_dim)),
    )
    source_input = torch.zeros((*batch_shape, config.embed_dim))
    can_act = torch.zeros((*batch_shape, ACTION_ENTITY_SLOTS), dtype=torch.bool)
    can_act[0, 0, 0, 2] = True
    max_launch = torch.zeros(batch_shape, dtype=torch.int64)
    max_launch[0, 0, 0] = 10

    def selected_policy_params(
        _selection: DiscreteTargetSelectionParams,
        _source_input: torch.Tensor,
        _max_launch: torch.Tensor,
        _target_index: torch.Tensor,
        *,
        min_fleet_size: int,
    ) -> DiscreteTargetPolicyParams:
        return DiscreteTargetPolicyParams(
            target_logits=target_logits,
            continue_logits=torch.full(batch_shape, -10.0),
            size_mix_logits=torch.zeros((*batch_shape, 1)),
            size_mu=torch.full((*batch_shape, 1), float(min_fleet_size)),
            size_scale=torch.ones((*batch_shape, 1)),
        )

    def entropy_policy_params(
        _selection: DiscreteTargetSelectionParams,
        _source_input: torch.Tensor,
        _max_launch: torch.Tensor,
        *,
        min_fleet_size: int,
    ) -> DiscreteTargetPolicyParams:
        params = selected_policy_params(
            _selection,
            _source_input,
            _max_launch,
            torch.zeros(batch_shape, dtype=torch.int64),
            min_fleet_size=min_fleet_size,
        )
        return DiscreteTargetPolicyParams(
            target_logits=params.target_logits,
            continue_logits=torch.full(batch_shape, -10.0),
            size_mix_logits=params.size_mix_logits,
            size_mu=params.size_mu,
            size_scale=params.size_scale,
        )

    monkeypatch.setattr(
        actor,
        "_selection_params",
        lambda _actor_inputs, _can_act: selection,
    )
    monkeypatch.setattr(
        actor,
        "_policy_params_for_selected_target",
        selected_policy_params,
    )
    monkeypatch.setattr(
        actor,
        "_policy_params_for_entropy",
        entropy_policy_params,
    )

    actions, log_probs, _entropies = actor(
        _discrete_actor_inputs(source_input),
        can_act,
        max_launch,
        min_fleet_size=1,
        deterministic=True,
    )

    expected_target_log_prob = F.log_softmax(target_logits[0, 0, 0], dim=-1)[2]
    assert not actions.launch[0, 0, 0, 0]
    assert actions.target[0, 0, 0, 0] == 2
    assert torch.allclose(log_probs.target[0, 0, 0, 0], expected_target_log_prob)


def test_discrete_targets_binary_after_entropy_uses_selected_target_approximation() -> (
    None
):
    batch_shape = (1, 4, ACTION_ENTITY_SLOTS)
    target_logits = torch.full((*batch_shape, ACTION_ENTITY_SLOTS), -100.0)
    target_logits[0, 0, 0, 1] = 0.0
    target_logits[0, 0, 0, 2] = 0.0
    continue_logits = torch.full(batch_shape, -100.0)
    params = DiscreteTargetPolicyParams(
        target_logits=target_logits,
        continue_logits=continue_logits,
        size_mix_logits=torch.zeros((*batch_shape, 1)),
        size_mu=torch.full((*batch_shape, 1), 3.0),
        size_scale=torch.ones((*batch_shape, 1)),
    )
    residual_budget = torch.full(batch_shape, 10, dtype=torch.int64)
    source_active = torch.zeros(batch_shape, dtype=torch.bool)
    source_active[0, 0, 0] = True
    can_act = torch.zeros((*batch_shape, ACTION_ENTITY_SLOTS), dtype=torch.bool)
    can_act[0, 0, 0, 1] = True
    can_act[0, 0, 0, 2] = True

    launch_entropy, *_ = discrete_action_entropy(
        params,
        residual_budget,
        source_active,
        can_act,
        "binary_after",
        min_fleet_size=1,
        entropy_ship_quantiles=4,
    )

    assert torch.allclose(
        launch_entropy[0, 0, 0],
        torch.tensor(0.0),
        atol=1e-5,
    )


def test_discrete_targets_binary_after_entropy_params_use_argmax_target() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(
            launch_mode="binary_after",
            n_action_mixtures=1,
            entropy_ship_quantiles=4,
        ),
        embed_dim=4,
        depth=1,
        n_heads=1,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    with torch.no_grad():
        for module in (actor.out, actor.size_pair_proj):
            module.weight.copy_(torch.eye(config.embed_dim))
            module.bias.zero_()
        for parameter in actor.mlp.parameters():
            parameter.zero_()
        actor.continue_head.up.weight.copy_(torch.eye(config.embed_dim))
        actor.continue_head.up.bias.zero_()
        actor.continue_head.out.weight.zero_()
        actor.continue_head.out.bias.zero_()
        actor.continue_head.out.weight[0, 0] = 1.0

    batch_shape = (1, 4, ACTION_ENTITY_SLOTS)
    target_logits = torch.full((*batch_shape, ACTION_ENTITY_SLOTS), -100.0)
    target_logits[0, 0, 0, 1] = 0.0
    target_logits[0, 0, 0, 2] = 1.0
    target_values = torch.zeros((*batch_shape, config.embed_dim))
    target_values[0, 0, 1, 0] = -10.0
    target_values[0, 0, 2, 0] = 10.0
    selection = DiscreteTargetSelectionParams(
        target_logits=target_logits,
        target_values=target_values,
    )
    source_input = torch.zeros((*batch_shape, config.embed_dim))
    max_launch = torch.full(batch_shape, 10, dtype=torch.int64)
    source_active = torch.zeros(batch_shape, dtype=torch.bool)
    source_active[0, 0, 0] = True
    can_act = torch.zeros((*batch_shape, ACTION_ENTITY_SLOTS), dtype=torch.bool)
    can_act[0, 0, 0, 1] = True
    can_act[0, 0, 0, 2] = True

    params = actor._policy_params_for_entropy(
        selection,
        source_input,
        max_launch,
        min_fleet_size=1,
    )
    launch_entropy, *_ = discrete_action_entropy(
        params,
        max_launch,
        source_active,
        can_act,
        "binary_after",
        min_fleet_size=1,
        entropy_ship_quantiles=4,
    )

    assert params.continue_logits is not None
    assert params.continue_logits.shape == batch_shape
    assert params.continue_logits[0, 0, 0] > 5.0
    assert torch.allclose(
        launch_entropy[0, 0, 0],
        model_impl.binary_entropy_from_logits(params.continue_logits.float())[0, 0, 0],
    )


def test_discrete_targets_target_token_mode_adds_no_launch_target() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(launch_mode="target_token"),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    assert actor.no_launch_target is not None
    assert id(actor.no_launch_target) in {
        id(layer) for layer in actor.get_input_layers()
    }
    assert actor.continue_source_proj is None
    assert actor.continue_head is None
    assert "continue_source_proj.weight" not in actor.state_dict()
    assert "continue_head.out.weight" not in actor.state_dict()
    slot_input = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    can_act = torch.zeros(
        (1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS),
        dtype=torch.bool,
    )
    can_act[0, 0, 0, 1] = True

    selection = actor._selection_params(_discrete_actor_inputs(slot_input), can_act)

    assert selection.target_logits.shape == (
        1,
        4,
        ACTION_ENTITY_SLOTS,
        ACTION_ENTITY_SLOTS + 1,
    )
    assert selection.target_values.shape == (
        1,
        4,
        ACTION_ENTITY_SLOTS + 1,
        config.embed_dim,
    )
    assert selection.target_logits[0, 0, 0, ACTION_ENTITY_SLOTS].isfinite()
    assert selection.target_logits[0, 0, 1].eq(0).all()
    assert selection.continue_logits is None


def test_discrete_targets_target_token_mode_scores_no_launch_as_target() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(
            launch_mode="target_token",
            n_action_mixtures=1,
        ),
        embed_dim=8,
        depth=1,
        n_heads=1,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    batch_shape = (1, 4, ACTION_ENTITY_SLOTS)
    target_logits = torch.full((*batch_shape, ACTION_ENTITY_SLOTS + 1), -10.0)
    target_logits[0, 0, 0, 1] = 2.0
    target_logits[0, 0, 0, ACTION_ENTITY_SLOTS] = 5.0
    params = DiscreteTargetPolicyParams(
        target_logits=target_logits,
        size_mix_logits=torch.zeros((*batch_shape, 1)),
        size_mu=torch.full((*batch_shape, 1), 3.0),
        size_scale=torch.ones((*batch_shape, 1)),
    )
    launch = torch.zeros(batch_shape, dtype=torch.bool)
    target = torch.ones(batch_shape, dtype=torch.int64)
    ships = torch.zeros(batch_shape, dtype=torch.int64)
    residual_budget = torch.full(batch_shape, 10, dtype=torch.int64)
    source_active = torch.zeros(batch_shape, dtype=torch.bool)
    source_active[0, 0, 0] = True

    launch_log_prob, target_log_prob, size_log_prob = (
        discrete_targets_impl.discrete_action_log_probs(
            params,
            launch,
            target,
            ships,
            residual_budget,
            source_active,
            actor.config.launch_mode,
            min_fleet_size=1,
        )
    )

    expected_no_launch = F.log_softmax(target_logits[0, 0, 0].float(), dim=-1)[
        ACTION_ENTITY_SLOTS
    ]
    assert launch_log_prob[0, 0, 0] == 0.0
    assert torch.allclose(target_log_prob[0, 0, 0], expected_no_launch)
    assert size_log_prob[0, 0, 0] == 0.0


def test_discrete_targets_actor_rejects_invalid_replay_target() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=1, obs_spec=obs_spec, action_spec=action_spec)
    obs.action_mask.max_launch[0, 0, 0] = action_spec.min_fleet_size
    output = model(obs, deterministic=True)
    output.actions.launch.zero_()
    output.actions.ships.zero_()
    assert output.actions.target is not None
    output.actions.target.zero_()
    output.actions.launch[0, 0, 0, 0] = True
    output.actions.ships[0, 0, 0, 0] = action_spec.min_fleet_size
    output.actions.target[0, 0, 0, 0] = 2

    with pytest.raises(ValueError, match=r"actions\.target must select a valid target"):
        model.evaluate_actions(obs, output.actions)


def test_discrete_targets_size_log_prob_conditions_on_replayed_target() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(
            n_action_mixtures=1,
            entropy_ship_quantiles=8,
        ),
        embed_dim=4,
        depth=1,
        n_heads=1,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    with torch.no_grad():
        for module in (actor.q, actor.k, actor.continue_head, actor.mix_head):
            module.weight.zero_()
            module.bias.zero_()
        for module in (actor.v, actor.out, actor.size_pair_proj):
            module.weight.copy_(torch.eye(config.embed_dim))
            module.bias.zero_()
        for parameter in actor.mlp.parameters():
            parameter.zero_()
        actor.mean_head.up.weight.copy_(torch.eye(config.embed_dim))
        actor.mean_head.up.bias.zero_()
        actor.mean_head.weight.zero_()
        actor.mean_head.bias.zero_()
        actor.mean_head.weight[0, 0] = 10.0
        actor.scale_head.weight.zero_()
        actor.scale_head.bias.fill_(-3.0)
        actor.source_role.zero_()
        actor.target_role.zero_()

    slot_input = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    slot_input[0, 0, 1] = torch.tensor([2.0, -2.0, 0.0, 0.0])
    slot_input[0, 0, 2] = torch.tensor([-2.0, 2.0, 0.0, 0.0])
    can_act = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS)).bool()
    can_act[0, 0, 0, 1] = True
    can_act[0, 0, 0, 2] = True
    max_launch = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
    max_launch[0, 0, 0] = 10
    actions = DiscreteTargetActions(
        launch=torch.zeros((1, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.bool),
        target=torch.zeros((1, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.int64),
        ships=torch.zeros((1, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.int64),
    )
    actions.launch[0, 0, 0, 0] = True
    actions.ships[0, 0, 0, 0] = 8

    actions.target[0, 0, 0, 0] = 1
    target_one_logp, _ = actor.log_prob(
        _discrete_actor_inputs(slot_input),
        can_act,
        max_launch,
        actions,
        min_fleet_size=1,
    )
    actions.target[0, 0, 0, 0] = 2
    target_two_logp, _ = actor.log_prob(
        _discrete_actor_inputs(slot_input),
        can_act,
        max_launch,
        actions,
        min_fleet_size=1,
    )

    assert not torch.allclose(
        target_one_logp.event[0, 0, 0],
        target_two_logp.event[0, 0, 0],
    )


def test_discrete_targets_replay_entropy_ignores_no_launch_target_placeholder() -> None:
    torch.manual_seed(17)
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(
            n_action_mixtures=2,
            entropy_ship_quantiles=8,
        ),
        embed_dim=8,
        depth=1,
        n_heads=1,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    slot_input = torch.randn((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    can_act = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS)).bool()
    can_act[0, 0, 0, 1] = True
    can_act[0, 0, 0, 2] = True
    max_launch = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
    max_launch[0, 0, 0] = 30
    actions = DiscreteTargetActions(
        launch=torch.zeros((1, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.bool),
        target=torch.zeros((1, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.int64),
        ships=torch.zeros((1, 4, ACTION_ENTITY_SLOTS, 1), dtype=torch.int64),
    )

    actions.target[0, 0, 0, 0] = 1
    _logp_one, entropy_one = actor.log_prob(
        _discrete_actor_inputs(slot_input),
        can_act,
        max_launch,
        actions,
        min_fleet_size=1,
    )
    actions.target[0, 0, 0, 0] = 2
    _logp_two, entropy_two = actor.log_prob(
        _discrete_actor_inputs(slot_input),
        can_act,
        max_launch,
        actions,
        min_fleet_size=1,
    )

    assert torch.allclose(entropy_one.launch, entropy_two.launch)
    assert torch.allclose(entropy_one.target, entropy_two.target)
    assert torch.allclose(entropy_one.event, entropy_two.event)
    assert torch.allclose(entropy_one.per_player_entity, entropy_two.per_player_entity)


def test_discrete_targets_scale_log_interpolates_budget_bounds() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(n_action_mixtures=1),
        embed_dim=4,
        depth=1,
        n_heads=1,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    with torch.no_grad():
        actor.scale_head.weight.zero_()

    slot_input = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    target_values = torch.zeros_like(slot_input)
    max_launch = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
    max_launch[0, 0, 0] = 10
    max_launch[0, 0, 1] = 100

    with torch.no_grad():
        actor.scale_head.bias.fill_(0.0)
    params = actor._size_params_from_target_values(
        slot_input,
        max_launch,
        target_values,
        min_fleet_size=1,
    )
    assert torch.allclose(
        params.size_scale[0, 0, 0, 0],
        torch.tensor(math.sqrt(0.10 * 8.0)),
    )
    assert torch.allclose(
        params.size_scale[0, 0, 1, 0],
        torch.tensor(math.sqrt(0.10 * 50.0)),
    )

    with torch.no_grad():
        actor.scale_head.bias.fill_(-20.0)
    params = actor._size_params_from_target_values(
        slot_input,
        max_launch,
        target_values,
        min_fleet_size=1,
    )
    assert torch.allclose(params.size_scale[0, 0, 0, 0], torch.tensor(0.10))

    with torch.no_grad():
        actor.scale_head.bias.fill_(20.0)
    params = actor._size_params_from_target_values(
        slot_input,
        max_launch,
        target_values,
        min_fleet_size=1,
    )
    assert torch.allclose(params.size_scale[0, 0, 0, 0], torch.tensor(8.0))
    assert torch.allclose(params.size_scale[0, 0, 1, 0], torch.tensor(50.0))


def test_discrete_targets_size_params_respect_min_fleet_size_support() -> None:
    config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(n_action_mixtures=1),
        embed_dim=4,
        depth=1,
        n_heads=1,
    )
    actor = DiscreteTargetsActor(config.actor, transformer_config=config)
    with torch.no_grad():
        actor.mean_head.weight.zero_()
        actor.mean_head.bias.zero_()
        actor.scale_head.weight.zero_()
        actor.scale_head.bias.zero_()

    slot_input = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    target_values = torch.zeros_like(slot_input)
    max_launch = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
    max_launch[0, 0, 0] = 10
    max_launch[0, 0, 1] = 100

    params = actor._size_params_from_target_values(
        slot_input,
        max_launch,
        target_values,
        min_fleet_size=6,
    )

    assert torch.allclose(params.size_mu[0, 0, 0, 0], torch.tensor(8.0))
    assert torch.allclose(
        params.size_scale[0, 0, 0, 0],
        torch.tensor(math.sqrt(0.10 * 8.0)),
    )
    assert torch.allclose(
        params.size_scale[0, 0, 1, 0],
        torch.tensor(math.sqrt(0.10 * 47.5)),
    )


def test_logsubexp_clamps_close_float32_inputs_to_finite_value() -> None:
    log_x = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    log_y = torch.tensor([-1e-8], dtype=torch.float32)

    value = logsubexp(log_x, log_y)
    value.backward()

    assert torch.isfinite(value).all()
    assert log_x.grad is not None
    assert torch.isfinite(log_x.grad).all()


def test_discretized_logistic_mixture_uses_float32_for_bfloat16_inputs() -> None:
    mix_logits = torch.zeros((2, 3), dtype=torch.bfloat16, requires_grad=True)
    mu = torch.tensor(
        [[50.0, 55.0, 60.0], [10.0, 12.0, 14.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    scale = torch.tensor(
        [[0.25, 2.0, 50.0], [0.5, 1.5, 8.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    ships = torch.tensor([55, 12], dtype=torch.int64)
    residual_budget = torch.tensor([100, 20], dtype=torch.int64)

    log_prob = discretized_logistic_mixture_log_prob(
        ships,
        residual_budget,
        mix_logits,
        mu,
        scale,
        min_fleet_size=1,
    )
    log_prob.sum().backward()

    assert log_prob.dtype == torch.float32
    assert torch.isfinite(log_prob).all()
    for tensor in (mix_logits, mu, scale):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_stochastic_discretized_logistic_sampling_uses_inverse_cdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_ship_support(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("stochastic sampling should not enumerate ship support")

    monkeypatch.setattr(logistic_mixture_impl, "ship_support", fail_ship_support)
    torch.manual_seed(5)
    residual_budget = torch.tensor([8, 20], dtype=torch.int64)
    ships = sample_discretized_logistic_mixture(
        mix_logits=torch.zeros((2, 2)),
        mu=torch.tensor([[6.0, 7.0], [12.0, 14.0]]),
        scale=torch.ones((2, 2)),
        residual_budget=residual_budget,
        min_fleet_size=6,
        deterministic=False,
    )

    assert ships.dtype == torch.int64
    assert torch.all(ships >= 6)
    assert torch.all(ships <= residual_budget)


def test_deterministic_discretized_logistic_sampling_skips_empty_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_log_prob(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("empty deterministic mask should skip support scoring")

    monkeypatch.setattr(
        logistic_mixture_impl,
        "discretized_logistic_mixture_log_prob",
        fail_log_prob,
    )
    ships = sample_discretized_logistic_mixture(
        mix_logits=torch.zeros((2, 2)),
        mu=torch.tensor([[500.0, 700.0], [12.0, 14.0]]),
        scale=torch.ones((2, 2)),
        residual_budget=torch.tensor([1000, 20], dtype=torch.int64),
        min_fleet_size=6,
        deterministic=True,
        deterministic_mask=torch.zeros(2, dtype=torch.bool),
    )

    assert torch.equal(ships, torch.zeros(2, dtype=torch.int64))


def test_deterministic_discretized_logistic_sampling_mask_matches_full_map() -> None:
    residual_budget = torch.tensor([[12, 100, 7]], dtype=torch.int64)
    mix_logits = torch.tensor([[[0.0, 0.5], [1.0, -0.5], [-0.25, 0.75]]])
    mu = torch.tensor([[[8.0, 10.0], [30.0, 95.0], [5.0, 7.0]]])
    scale = torch.tensor([[[1.0, 2.0], [3.0, 1.5], [0.5, 1.0]]])
    launch = torch.tensor([[True, False, True]])

    full_ships = sample_discretized_logistic_mixture(
        mix_logits,
        mu,
        scale,
        residual_budget,
        min_fleet_size=6,
        deterministic=True,
    )
    masked_ships = sample_discretized_logistic_mixture(
        mix_logits,
        mu,
        scale,
        residual_budget,
        min_fleet_size=6,
        deterministic=True,
        deterministic_mask=launch,
    )

    assert torch.equal(masked_ships, torch.where(launch, full_ships, 0))


def test_ship_support_counts_only_valid_min_to_residual_entries() -> None:
    residual_budget = torch.tensor([[100, 12]], dtype=torch.int64)

    support = ship_support(residual_budget, min_fleet_size=6, max_ship_support=100)
    capped_support = ship_support(
        residual_budget,
        min_fleet_size=6,
        max_ship_support=4,
    )

    assert support.shape == (1, 1, 95)
    assert torch.equal(support[0, 0, :3], torch.tensor([6, 7, 8]))
    assert support[0, 0, -1] == 100
    assert torch.equal(capped_support[0, 0], torch.tensor([6, 7, 8, 9]))


def test_discrete_targets_entropy_ignores_inactive_source_gradients() -> None:
    mix_logits = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, 2), requires_grad=True)
    mu = torch.full((1, 4, ACTION_ENTITY_SLOTS, 2), 2.0, requires_grad=True)
    scale = torch.full((1, 4, ACTION_ENTITY_SLOTS, 2), 0.5, requires_grad=True)
    continue_logits = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), requires_grad=True)
    target_logits = torch.zeros(
        (1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS),
        requires_grad=True,
    )
    params = DiscreteTargetPolicyParams(
        continue_logits=continue_logits,
        target_logits=target_logits,
        size_mix_logits=mix_logits,
        size_mu=mu,
        size_scale=scale,
    )
    residual_budget = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
    source_active = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool)
    can_act = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS)).bool()

    entropies = discrete_action_entropy(
        params,
        residual_budget,
        source_active,
        can_act,
        min_fleet_size=2,
        entropy_ship_quantiles=4,
    )
    sum(entropy.sum() for entropy in entropies).backward()

    for tensor in (mix_logits, mu, scale, continue_logits, target_logits):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_discrete_targets_quantile_entropy_accounts_for_component_overlap() -> None:
    residual_budget = torch.tensor([100], dtype=torch.int64)
    single_entropy = discrete_targets_impl.truncated_logistic_mixture_entropy(
        torch.zeros((1, 1)),
        torch.full((1, 1), 50.0),
        torch.full((1, 1), 2.0),
        residual_budget,
        min_fleet_size=1,
        entropy_ship_quantiles=32,
    )
    duplicated_entropy = discrete_targets_impl.truncated_logistic_mixture_entropy(
        torch.zeros((1, 2)),
        torch.full((1, 2), 50.0),
        torch.full((1, 2), 2.0),
        residual_budget,
        min_fleet_size=1,
        entropy_ship_quantiles=32,
    )

    assert torch.allclose(duplicated_entropy, single_entropy)


def test_min_fleet_size_masks_and_shifts_ship_distribution() -> None:
    torch.manual_seed(3)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1, min_fleet_size=3)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorPureConfig(n_angle_mixtures=2, n_fleet_size_mixtures=2),
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=1, obs_spec=obs_spec, action_spec=action_spec)
    with torch.no_grad():
        model.actor.actor_heads.continue_head.bias.fill_(100.0)

    output = model(obs, deterministic=True)

    assert not output.actions.launch[0, 2, MAX_PLANETS].any()
    launched_ships = output.actions.ships[output.actions.launch]
    assert launched_ships.numel() > 0
    assert torch.all(launched_ships >= action_spec.min_fleet_size)

    evaluation = model.evaluate_actions(obs, output.actions)
    assert torch.allclose(
        evaluation.log_probs.event,
        output.log_probs.event,
    )

    output.actions.launch.zero_()
    output.actions.ships.zero_()
    assert output.actions.angle is not None
    output.actions.angle.zero_()
    output.actions.launch[0, 0, 0, 0] = True
    output.actions.ships[0, 0, 0, 0] = action_spec.min_fleet_size - 1

    with pytest.raises(ValueError, match=r"actions\.ships must be in 3\.\.remaining"):
        model.evaluate_actions(obs, output.actions)


def test_pure_actor_rejects_multi_launch_action_spec() -> None:
    config = StatelessTransformerV1Config(
        embed_dim=16,
        depth=1,
        n_heads=4,
    )

    with pytest.raises(
        ValueError, match="pure actor requires max_per_planet_launches=1"
    ):
        PureActor(
            ActorPureConfig(),
            embed_dim=config.embed_dim,
            max_per_planet_launches=4,
            activation=config.activation,
        )


def test_actor_distribution_outputs_remain_fp32_under_cpu_bf16_autocast() -> None:
    torch.manual_seed(0)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorPureConfig(n_angle_mixtures=2, n_fleet_size_mixtures=2),
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(obs)
        evaluation = model.evaluate_actions(obs, output.actions)

    assert output.actions.launch.dtype == torch.bool
    assert output.actions.angle is not None
    assert output.actions.angle.dtype == torch.float32
    assert output.actions.ships.dtype == torch.int64
    for tensors in (output.log_probs, output.entropies, evaluation.log_probs):
        assert tensors.launch.dtype == torch.float32
        assert tensors.event.dtype == torch.float32
        assert tensors.per_player_entity.dtype == torch.float32
        assert torch.isfinite(tensors.launch).all()
        assert torch.isfinite(tensors.event).all()
        assert torch.isfinite(tensors.per_player_entity).all()


def test_distribution_helpers_promote_lower_precision_params_to_fp32() -> None:
    torch.manual_seed(0)
    mixtures = 2
    shape = (1, 4, ACTION_ENTITY_SLOTS, mixtures)
    mix_logits = torch.zeros(shape, dtype=torch.bfloat16)
    params = PolicyParams(
        continue_logits=torch.zeros(shape[:-1], dtype=torch.bfloat16),
        angle_mix_logits=mix_logits,
        angle_log_w=torch.log_softmax(mix_logits, dim=-1),
        loc=torch.zeros(shape, dtype=torch.bfloat16),
        kappa=torch.ones(shape, dtype=torch.bfloat16),
        size_mix_logits=mix_logits,
        size_mu=torch.full(shape, 3.0, dtype=torch.bfloat16),
        size_scale=torch.full(shape, 1.0, dtype=torch.bfloat16),
    )
    active = torch.ones(shape[:-1], dtype=torch.bool)
    residual_budget = torch.full(shape[:-1], 5, dtype=torch.int64)
    angle = torch.full(shape[:-1], 0.25, dtype=torch.float32)
    ships = torch.ones(shape[:-1], dtype=torch.int64)
    model = _model(StatelessTransformerV1Config(embed_dim=32, depth=1, n_heads=4))
    assert isinstance(model.actor, model_impl.PureActor)

    launch = model_impl.sample_launch(
        params.continue_logits,
        active,
        deterministic=False,
    )
    sampled_angle = model_impl.sample_angle_mixture(
        params,
        deterministic=False,
    )
    sampled_ships = sample_discretized_logistic_mixture(
        params.size_mix_logits,
        params.size_mu,
        params.size_scale,
        residual_budget,
        min_fleet_size=1,
        deterministic=False,
    )
    event_log_prob = masked_event_log_prob_from_params(
        params,
        angle,
        ships,
        residual_budget,
        1,
        active,
    )

    assert launch.dtype == torch.bool
    assert sampled_angle.dtype == torch.float32
    assert sampled_ships.dtype == torch.int64
    for tensor in (event_log_prob,):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_binary_entropy_from_logits_matches_closed_form() -> None:
    logits = torch.tensor([0.0, math.log(3.0)])
    probabilities = torch.sigmoid(logits)
    expected = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )

    assert torch.allclose(model_impl.binary_entropy_from_logits(logits), expected)


def test_masked_action_entropy_includes_latent_mixture_entropy() -> None:
    mix_logits = torch.zeros((2, 2))
    params = PolicyParams(
        continue_logits=torch.tensor([0.0, 10.0]),
        angle_mix_logits=mix_logits,
        angle_log_w=torch.log_softmax(mix_logits, dim=-1),
        loc=torch.zeros((2, 2)),
        kappa=torch.zeros((2, 2)),
        size_mix_logits=mix_logits,
        size_mu=torch.full((2, 2), 2.0),
        size_scale=torch.ones((2, 2)),
    )
    residual_budget = torch.tensor([3, 3])
    active = torch.tensor([True, False])

    launch_entropy, event_entropy, *_ = model_impl.masked_action_entropy_from_params(
        params,
        residual_budget,
        active,
        min_fleet_size=1,
        entropy_ship_quantiles=8,
    )

    angle_entropy = math.log(2.0) + math.log(2.0 * math.pi)
    size_entropy = model_impl.event_entropy_from_params(
        params,
        residual_budget,
        min_fleet_size=1,
        entropy_ship_quantiles=8,
    )[2]
    expected_event_entropy = angle_entropy + size_entropy[0]
    assert torch.allclose(
        launch_entropy,
        torch.tensor([math.log(2.0), 0.0]),
        atol=1e-6,
    )
    assert torch.allclose(
        event_entropy,
        torch.stack((0.5 * expected_event_entropy, torch.tensor(0.0))),
        atol=1e-6,
    )


def test_actor_log_probs_have_finite_gradients_for_masked_slots() -> None:
    torch.manual_seed(1)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorPureConfig(n_angle_mixtures=2, n_fleet_size_mixtures=2),
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    obs.still_playing = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    output = model(obs)

    model.zero_grad()
    model.evaluate_actions(
        obs,
        output.actions,
    ).log_probs.per_player_entity.sum().backward()

    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_pure_replay_ignores_nan_angles_for_no_launch_slots() -> None:
    torch.manual_seed(1)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorPureConfig(n_angle_mixtures=2, n_fleet_size_mixtures=2),
    )
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    action_shape = (2, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS, 1)
    actions = PureActions(
        launch=torch.zeros(action_shape, dtype=torch.bool),
        angle=torch.full(action_shape, float("nan"), dtype=torch.float32),
        ships=torch.zeros(action_shape, dtype=torch.int64),
    )

    model.zero_grad()
    evaluation = model.evaluate_actions(obs, actions)
    evaluation.log_probs.per_player_entity.sum().backward()

    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert torch.isfinite(evaluation.log_probs.per_player_entity).all()
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_model_rejects_pure_multi_launch_action_spec() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorPureConfig(n_angle_mixtures=2, n_fleet_size_mixtures=2),
    )

    with pytest.raises(
        ValueError, match="pure actor requires max_per_planet_launches=1"
    ):
        StatelessTransformerV1(
            config,
            obs_spec=obs_spec,
            action_spec=ActionPureConfig.model_construct(
                action_spec="pure",
                max_per_planet_launches=3,
                min_fleet_size=1,
            ),
        )


def test_evaluate_actions_rejects_invalid_action_dtypes() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 1)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(
        batch_size=1,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    output = model(obs)
    output.actions.ships = output.actions.ships.to(torch.float32)

    with pytest.raises(
        ValueError, match=r"actions\.ships must have dtype torch\.int64"
    ):
        model.evaluate_actions(obs, output.actions)


@pytest.mark.parametrize("angle", [math.nan, math.inf])
def test_evaluate_actions_rejects_nonfinite_launched_angles(angle: float) -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 1)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(
        batch_size=1,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    output = model(obs)
    output.actions.launch.zero_()
    assert output.actions.angle is not None
    output.actions.angle.zero_()
    output.actions.ships.zero_()
    output.actions.launch[0, 0, 0, 0] = True
    output.actions.angle[0, 0, 0, 0] = angle
    output.actions.ships[0, 0, 0, 0] = action_spec.min_fleet_size
    assert obs.action_mask.max_launch is not None
    obs.action_mask.max_launch[0, 0, 0] = action_spec.min_fleet_size

    with pytest.raises(ValueError, match=r"actions\.angle must be finite"):
        model.evaluate_actions(obs, output.actions)


def test_critic_requires_still_playing_mask_with_live_player() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 1)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(
        batch_size=1,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )

    obs.still_playing.fill_(False)

    with pytest.raises(ValueError, match="at least one player"):
        model(obs)
