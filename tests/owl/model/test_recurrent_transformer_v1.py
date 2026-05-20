from __future__ import annotations

from typing import Literal

import pytest
import torch
from owl.model import (
    ActorDiscreteTargetsConfig,
    ModelConfig,
    RecurrentTransformerV1,
    RecurrentTransformerV1Config,
)
from owl.model.recurrent_transformer_v1 import MinGRU
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    MAX_PLANETS,
    OUTER_PLAYER_SLOTS,
    ActionDiscreteTargetsConfig,
    ActionPureConfig,
    DiscreteTargetActionMask,
    DiscreteTargetActions,
    EntityBasedConfig,
    ObsBatch,
)
from pydantic import TypeAdapter


def _obs_batch(
    *,
    batch_size: int,
    obs_spec: EntityBasedConfig,
    action_spec: ActionDiscreteTargetsConfig,
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
    still_playing = torch.ones((batch_size, OUTER_PLAYER_SLOTS), dtype=torch.bool)
    global_features = torch.zeros(
        (batch_size, obs_spec.global_channels),
        dtype=torch.float32,
    )
    can_act = torch.zeros(
        (
            batch_size,
            OUTER_PLAYER_SLOTS,
            ACTION_ENTITY_SLOTS,
            ACTION_ENTITY_SLOTS,
        ),
        dtype=torch.bool,
    )
    max_launch = torch.zeros(
        (batch_size, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS),
        dtype=torch.int64,
    )
    planets[:, 0, 0] = 1.0
    planets[:, 0, 15] = 0.08
    planets[:, 1, 1] = 1.0
    planets[:, 1, 15] = 0.04
    entity_mask[:, :2] = True
    can_act[:, 0, 0, 1] = True
    can_act[:, 1, 1, 0] = True
    max_launch[:, 0, 0] = action_spec.min_fleet_size
    max_launch[:, 1, 1] = action_spec.min_fleet_size
    return ObsBatch(
        planets=planets,
        orbiting_planets=orbiting_planets,
        fleets=fleets,
        comets=comets,
        entity_mask=entity_mask,
        still_playing=still_playing,
        global_features=global_features,
        action_mask=DiscreteTargetActionMask(
            can_act=can_act,
            max_launch=max_launch,
        ),
    )


def _stack_obs(obs: ObsBatch, time_steps: int) -> ObsBatch:
    if not isinstance(obs.action_mask, DiscreteTargetActionMask):
        raise TypeError("expected discrete target action mask")
    return ObsBatch(
        planets=obs.planets[:, None].expand(-1, time_steps, -1, -1).contiguous(),
        orbiting_planets=obs.orbiting_planets[:, None]
        .expand(-1, time_steps, -1)
        .contiguous(),
        fleets=obs.fleets[:, None].expand(-1, time_steps, -1, -1).contiguous(),
        comets=obs.comets[:, None].expand(-1, time_steps, -1, -1).contiguous(),
        entity_mask=obs.entity_mask[:, None].expand(-1, time_steps, -1).contiguous(),
        still_playing=obs.still_playing[:, None]
        .expand(-1, time_steps, -1)
        .contiguous(),
        global_features=obs.global_features[:, None]
        .expand(-1, time_steps, -1)
        .contiguous(),
        action_mask=DiscreteTargetActionMask(
            can_act=obs.action_mask.can_act[:, None]
            .expand(-1, time_steps, -1, -1, -1)
            .contiguous(),
            max_launch=obs.action_mask.max_launch[:, None]
            .expand(-1, time_steps, -1, -1)
            .contiguous(),
        ),
    )


def _compact_obs(obs: ObsBatch, fleet_indices: torch.Tensor) -> ObsBatch:
    if not isinstance(obs.action_mask, DiscreteTargetActionMask):
        raise TypeError("expected discrete target action mask")
    return ObsBatch(
        planets=obs.planets,
        orbiting_planets=obs.orbiting_planets,
        fleets=obs.fleets[:, fleet_indices, :],
        comets=obs.comets,
        entity_mask=torch.cat(
            (
                obs.entity_mask[:, :ACTION_ENTITY_SLOTS],
                obs.entity_mask[:, ACTION_ENTITY_SLOTS:][:, fleet_indices],
            ),
            dim=1,
        ),
        still_playing=obs.still_playing,
        global_features=obs.global_features,
        action_mask=obs.action_mask,
    )


def _obs_step(obs: ObsBatch, step: int) -> ObsBatch:
    if not isinstance(obs.action_mask, DiscreteTargetActionMask):
        raise TypeError("expected discrete target action mask")
    return ObsBatch(
        planets=obs.planets[:, step],
        orbiting_planets=obs.orbiting_planets[:, step],
        fleets=obs.fleets[:, step],
        comets=obs.comets[:, step],
        entity_mask=obs.entity_mask[:, step],
        still_playing=obs.still_playing[:, step],
        global_features=obs.global_features[:, step],
        action_mask=DiscreteTargetActionMask(
            can_act=obs.action_mask.can_act[:, step],
            max_launch=obs.action_mask.max_launch[:, step],
        ),
    )


def _actions(batch_size: int, time_steps: int) -> DiscreteTargetActions:
    shape = (
        batch_size,
        time_steps,
        OUTER_PLAYER_SLOTS,
        ACTION_ENTITY_SLOTS,
        1,
    )
    return DiscreteTargetActions(
        launch=torch.zeros(shape, dtype=torch.bool),
        target=torch.zeros(shape, dtype=torch.int64),
        ships=torch.zeros(shape, dtype=torch.int64),
    )


def test_recurrent_model_config_has_discriminator_tag() -> None:
    config = TypeAdapter(ModelConfig).validate_python(
        {
            "model_arch": "recurrent_transformer_v1",
            "embed_dim": 32,
            "n_heads": 4,
        }
    )

    assert isinstance(config, RecurrentTransformerV1Config)
    assert config.actor.launch_mode == "binary"
    assert config.recurrence_mode == "global_only"

    include_planets_config = TypeAdapter(ModelConfig).validate_python(
        {
            "model_arch": "recurrent_transformer_v1",
            "embed_dim": 32,
            "n_heads": 4,
            "recurrence_mode": "include_planets",
        }
    )
    assert isinstance(include_planets_config, RecurrentTransformerV1Config)
    assert include_planets_config.recurrence_mode == "include_planets"


def test_recurrent_model_rejects_non_binary_launch_mode() -> None:
    with pytest.raises(ValueError, match="binary launch mode"):
        RecurrentTransformerV1Config(
            actor=ActorDiscreteTargetsConfig(launch_mode="binary_after")
        )


def test_recurrent_model_requires_discrete_target_env() -> None:
    with pytest.raises(ValueError, match="requires discrete_targets"):
        RecurrentTransformerV1(
            RecurrentTransformerV1Config(embed_dim=16, depth=1, n_heads=4),
            obs_spec=EntityBasedConfig(),
            action_spec=ActionPureConfig(),
        )


def test_mingru_parallel_scan_matches_sequential_recurrence() -> None:
    torch.manual_seed(0)
    gru = MinGRU(embed_dim=5)
    x = torch.randn(2, 7, 3, 5)
    h0 = torch.randn(2, 3, 5)
    reset = torch.zeros((2, 7, 3), dtype=torch.bool)
    active = torch.ones((2, 7, 3), dtype=torch.bool)
    reset[0, 3, 1] = True
    active[1, 5, 2] = False

    parallel, final = gru(x, h0, reset=reset, active=active)

    gate = torch.sigmoid(gru.gate(x))
    candidate = torch.tanh(gru.candidate(x))
    hidden = h0
    sequential_steps = []
    for step in range(x.shape[1]):
        keep = (~reset[:, step] & active[:, step]).unsqueeze(-1)
        active_step = active[:, step].unsqueeze(-1)
        hidden = (1.0 - gate[:, step]) * hidden * keep
        hidden = hidden + gate[:, step] * candidate[:, step] * active_step
        hidden = hidden * active_step
        sequential_steps.append(hidden)
    sequential = torch.stack(sequential_steps, dim=1)

    assert torch.allclose(parallel, sequential, atol=1e-6)
    assert torch.allclose(final, sequential[:, -1], atol=1e-6)


def test_recurrent_hidden_reset_uses_env_and_player_dones() -> None:
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 2)
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    model = RecurrentTransformerV1(
        RecurrentTransformerV1Config(embed_dim=16, depth=1, n_heads=4),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    state = model.initial_hidden_state(2, device=torch.device("cpu"))
    state.hidden.fill_(1.0)

    dones = torch.tensor(
        [
            [True, False, False, False],
            [True, True, True, True],
        ]
    )
    reset = model.reset_hidden_state(state, dones)
    assert reset is not None
    layout = model._recurrent_layout

    assert reset.hidden[0, 1].eq(0.0).all()
    assert reset.hidden[0, 0, : layout.shared_count].eq(1.0).all()
    for token, player in enumerate(layout.player_index.tolist()):
        if player == 0:
            assert reset.hidden[0, 0, token].eq(0.0).all()
        elif player >= 0:
            assert reset.hidden[0, 0, token].eq(1.0).all()


def test_recurrent_include_planets_adds_env_level_state() -> None:
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 2)
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    model = RecurrentTransformerV1(
        RecurrentTransformerV1Config(
            embed_dim=16,
            depth=1,
            n_heads=4,
            n_scratch_tokens=3,
            recurrence_mode="include_planets",
        ),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    layout = model._recurrent_layout
    state = model.initial_hidden_state(2, device=torch.device("cpu"))
    state.hidden.fill_(1.0)

    assert layout.token_indices[:MAX_PLANETS].tolist() == list(range(MAX_PLANETS))
    assert layout.shared_count == MAX_PLANETS + 1 + 3
    assert (
        layout.token_indices[MAX_PLANETS : layout.shared_count]
        .ge(obs_spec.max_entities)
        .all()
    )
    assert layout.player_index[: layout.shared_count].eq(-1).all()
    assert state.hidden.shape[2] == MAX_PLANETS + 1 + 3 + 3 * OUTER_PLAYER_SLOTS

    dones = torch.tensor(
        [
            [True, False, False, False],
            [True, True, True, True],
        ]
    )
    reset = model.reset_hidden_state(state, dones)
    assert reset is not None

    assert reset.hidden[0, 0, :MAX_PLANETS].eq(1.0).all()
    assert reset.hidden[0, 1, :MAX_PLANETS].eq(0.0).all()
    for token, player in enumerate(layout.player_index.tolist()):
        if player == 0:
            assert reset.hidden[0, 0, token].eq(0.0).all()


def test_recurrent_include_planets_zeros_inactive_planet_state() -> None:
    torch.manual_seed(0)
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 2)
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    model = RecurrentTransformerV1(
        RecurrentTransformerV1Config(
            embed_dim=16,
            depth=1,
            n_heads=4,
            recurrence_mode="include_planets",
        ),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    model.reset_parameters()
    obs = _obs_batch(batch_size=1, obs_spec=obs_spec, action_spec=action_spec)
    state = model.initial_hidden_state(1, device=torch.device("cpu"))
    state.hidden.fill_(1.0)

    output = model(obs, deterministic=True, hidden_state=state)

    assert output.next_hidden_state is not None
    assert output.next_hidden_state.hidden[0, 0, 2:MAX_PLANETS].eq(0.0).all()


@pytest.mark.parametrize("recurrence_mode", ["global_only", "include_planets"])
def test_recurrent_forward_supports_compacted_runtime_entity_counts(
    recurrence_mode: Literal["global_only", "include_planets"],
) -> None:
    torch.manual_seed(0)
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 3)
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    model = RecurrentTransformerV1(
        RecurrentTransformerV1Config(
            embed_dim=16,
            depth=1,
            n_heads=4,
            recurrence_mode=recurrence_mode,
        ),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    model.reset_parameters()
    obs = _obs_batch(batch_size=1, obs_spec=obs_spec, action_spec=action_spec)
    obs.entity_mask[0, ACTION_ENTITY_SLOTS] = True
    compacted = _compact_obs(obs, torch.tensor([], dtype=torch.long))
    hidden = model.initial_hidden_state(1, device=torch.device("cpu"))

    output = model(compacted, hidden_state=hidden)
    assert output.next_hidden_state is not None
    next_output = model(
        _compact_obs(obs, torch.tensor([0], dtype=torch.long)),
        hidden_state=output.next_hidden_state,
    )

    assert output.next_hidden_state.hidden.shape == hidden.hidden.shape
    assert next_output.next_hidden_state is not None
    assert next_output.next_hidden_state.hidden.shape == hidden.hidden.shape
    cached_layout = model._recurrent_layout_for_entity_count(ACTION_ENTITY_SLOTS)
    assert cached_layout is model._recurrent_layout_for_entity_count(
        ACTION_ENTITY_SLOTS
    )


@pytest.mark.parametrize("recurrence_mode", ["global_only", "include_planets"])
def test_recurrent_sequence_evaluation_matches_stepwise_evaluation(
    recurrence_mode: Literal["global_only", "include_planets"],
) -> None:
    torch.manual_seed(1)
    batch_size = 2
    time_steps = 4
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 2)
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    model = RecurrentTransformerV1(
        RecurrentTransformerV1Config(
            embed_dim=16,
            depth=2,
            n_heads=4,
            recurrence_mode=recurrence_mode,
        ),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    model.reset_parameters()
    obs = _stack_obs(
        _obs_batch(batch_size=batch_size, obs_spec=obs_spec, action_spec=action_spec),
        time_steps,
    )
    actions = _actions(batch_size, time_steps)
    dones = torch.zeros((batch_size, time_steps, OUTER_PLAYER_SLOTS), dtype=torch.bool)
    dones[0, 1, 0] = True
    dones[1, 2, :] = True
    initial_state = model.initial_hidden_state(batch_size, device=torch.device("cpu"))

    sequence = model.evaluate_actions(
        obs,
        actions,
        hidden_state=initial_state,
        dones=dones,
    )

    state = initial_state
    values = []
    log_probs = []
    for step in range(time_steps):
        step_actions = DiscreteTargetActions(
            launch=actions.launch[:, step],
            target=actions.target[:, step],
            ships=actions.ships[:, step],
        )
        output = model.evaluate_actions(
            _obs_step(obs, step),
            step_actions,
            hidden_state=state,
        )
        values.append(output.values)
        log_probs.append(output.log_probs.per_player_entity)
        state = output.next_hidden_state
        if step < time_steps - 1:
            state = model.reset_hidden_state(state, dones[:, step])

    assert torch.allclose(sequence.values, torch.stack(values, dim=1), atol=1e-5)
    assert torch.allclose(
        sequence.log_probs.per_player_entity,
        torch.stack(log_probs, dim=1),
        atol=1e-5,
    )
