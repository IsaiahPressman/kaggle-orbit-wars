import pytest
import torch
from owl.model import ModelConfig, TransformerActorCritic, TransformerV1Config
from owl.model.transformer_v1 import MinGRUCell
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    MAX_COMETS,
    MAX_PLANETS,
    ActionPureConfig,
    ObsBatch,
    ObsV1Config,
)
from pydantic import TypeAdapter


def _obs_batch(
    *,
    batch_size: int,
    obs_spec: ObsV1Config,
    action_spec: ActionPureConfig,
) -> ObsBatch:
    planets = torch.zeros(
        (batch_size, obs_spec.max_planets, obs_spec.planet_channels),
        dtype=torch.float32,
    )
    fleets = torch.zeros(
        (batch_size, obs_spec.max_fleets, obs_spec.fleet_channels),
        dtype=torch.float32,
    )
    comets = torch.zeros(
        (batch_size, obs_spec.max_comets, obs_spec.comet_channels),
        dtype=torch.float32,
    )
    planet_mask = torch.zeros((batch_size, obs_spec.max_planets), dtype=torch.bool)
    fleet_mask = torch.zeros((batch_size, obs_spec.max_fleets), dtype=torch.bool)
    comet_mask = torch.zeros((batch_size, obs_spec.max_comets), dtype=torch.bool)
    still_playing = torch.ones((batch_size, 4), dtype=torch.bool)
    global_features = torch.zeros(
        (batch_size, obs_spec.global_channels),
        dtype=torch.float32,
    )
    can_act = torch.zeros((batch_size, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool)
    max_launch = torch.zeros((batch_size, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)

    planets[:, 0, 0] = 1.0
    planets[:, 0, 13] = 0.08
    planets[:, 1, 1] = 1.0
    planets[:, 1, 13] = 0.04
    planet_mask[:, :2] = True
    comets[:, 0, 2] = 1.0
    comet_mask[:, 0] = True
    can_act[:, 0, 0] = True
    can_act[:, 1, 1] = True
    can_act[:, 2, MAX_PLANETS] = True
    max_launch[:, 0, 0] = 5
    max_launch[:, 1, 1] = 3
    max_launch[:, 2, MAX_PLANETS] = 2

    assert action_spec.max_per_planet_launches >= 1
    return ObsBatch(
        planets=planets,
        fleets=fleets,
        comets=comets,
        planet_mask=planet_mask,
        fleet_mask=fleet_mask,
        comet_mask=comet_mask,
        still_playing=still_playing,
        global_features=global_features,
        can_act=can_act,
        max_launch=max_launch,
    )


def test_model_config_requires_heads_to_divide_embed_dim() -> None:
    with pytest.raises(ValueError, match="n_heads must evenly divide embed_dim"):
        TransformerV1Config(embed_dim=30, n_heads=8)


def test_model_config_has_discriminator_tag() -> None:
    config = TypeAdapter(ModelConfig).validate_python(
        {
            "model_arch": "transformer_v1",
            "embed_dim": 32,
            "n_heads": 4,
        }
    )

    assert config.model_arch == "transformer_v1"


def test_min_gru_cell_matches_paper_equation_without_candidate_tanh() -> None:
    cell = MinGRUCell(input_dim=2, hidden_dim=2)
    with torch.no_grad():
        cell.update.weight.zero_()
        cell.update.bias.copy_(torch.tensor([0.0, 2.0]))
        cell.candidate.weight.copy_(torch.eye(2))
        cell.candidate.bias.zero_()

    x = torch.tensor([[3.0, -4.0]])
    prev = torch.tensor([[1.0, 2.0]])
    update = torch.sigmoid(torch.tensor([[0.0, 2.0]]))
    expected = (1.0 - update) * prev + update * x

    assert torch.allclose(cell(x, prev), expected)


def test_observation_encoder_returns_entity_tokens_plus_player_tokens() -> None:
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 3)
    config = TransformerV1Config(
        obs_spec=obs_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = TransformerActorCritic(config)
    obs = _obs_batch(
        batch_size=2,
        obs_spec=obs_spec,
        action_spec=config.action_spec,
    )

    hidden, mask = model.encode_observations(obs)

    assert hidden.shape == (2, obs_spec.max_entities + 4, 32)
    assert mask.shape == (2, obs_spec.max_entities + 4)
    assert mask[:, -4:].all()
    assert mask[:, :MAX_PLANETS].sum().item() == 4
    assert mask[:, MAX_PLANETS].all()
    assert not torch.allclose(
        model.player_tokens.weight[0],
        model.player_tokens.weight[1],
    )


def test_actor_critic_outputs_action_tensors_log_probs_and_values() -> None:
    torch.manual_seed(0)
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=3)
    config = TransformerV1Config(
        obs_spec=obs_spec,
        action_spec=action_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
        n_angle_mixtures=2,
    )
    model = TransformerActorCritic(config)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    obs.still_playing = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )

    output = model(obs)

    expected_action_shape = (2, 4, ACTION_ENTITY_SLOTS, 3)
    assert output.actions.launch.shape == expected_action_shape
    assert output.actions.launch.dtype == torch.bool
    assert output.actions.angle.shape == expected_action_shape
    assert output.actions.angle.dtype == torch.float32
    assert output.actions.ships.shape == expected_action_shape
    assert output.actions.ships.dtype == torch.int64
    assert output.log_probs.launch.shape == expected_action_shape
    assert output.log_probs.angle_and_size.shape == expected_action_shape
    assert output.log_probs.total.shape == (2,)
    assert output.values.shape == (2, 4)
    assert output.winner_probabilities.shape == (2, 4)
    assert torch.allclose(output.winner_probabilities.sum(dim=1), torch.ones(2))
    assert torch.all(output.winner_probabilities[~obs.still_playing] == 0)
    assert torch.all(output.actions.ships[~output.actions.launch] == 0)
    assert torch.all(output.actions.ships.sum(dim=-1) <= obs.max_launch)
    assert model.slot_dynamic_proj.in_features == 6

    replay_log_probs = model.log_prob(obs, output.actions)
    assert torch.allclose(replay_log_probs.launch, output.log_probs.launch)
    assert torch.allclose(
        replay_log_probs.angle_and_size, output.log_probs.angle_and_size
    )
    assert torch.allclose(replay_log_probs.total, output.log_probs.total)


def test_actor_log_probs_have_finite_gradients_for_masked_slots() -> None:
    torch.manual_seed(1)
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=3)
    config = TransformerV1Config(
        obs_spec=obs_spec,
        action_spec=action_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
        n_angle_mixtures=2,
    )
    model = TransformerActorCritic(config)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)
    obs.still_playing = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    output = model(obs)

    model.zero_grad()
    model.log_prob(obs, output.actions).total.sum().backward()

    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_log_prob_rejects_invalid_action_dtypes() -> None:
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 1)
    config = TransformerV1Config(
        obs_spec=obs_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = TransformerActorCritic(config)
    obs = _obs_batch(
        batch_size=1,
        obs_spec=obs_spec,
        action_spec=config.action_spec,
    )
    output = model(obs)
    output.actions.ships = output.actions.ships.to(torch.float32)

    with pytest.raises(
        ValueError, match=r"actions\.ships must have dtype torch\.int64"
    ):
        model.log_prob(obs, output.actions)


def test_critic_requires_still_playing_mask_with_live_player() -> None:
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 1)
    config = TransformerV1Config(
        obs_spec=obs_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = TransformerActorCritic(config)
    obs = _obs_batch(
        batch_size=1,
        obs_spec=obs_spec,
        action_spec=config.action_spec,
    )

    obs.still_playing.fill_(False)

    with pytest.raises(ValueError, match="at least one player"):
        model(obs)
