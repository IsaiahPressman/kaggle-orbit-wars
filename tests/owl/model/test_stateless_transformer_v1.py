import math

import pytest
import torch
from owl.model import ModelConfig, StatelessTransformerV1, StatelessTransformerV1Config
from owl.model.stateless_transformer_v1 import (
    FeedForward,
    MinGRUCell,
    MultiHeadSelfAttention,
    PolicyParams,
    beta_binomial_entropy,
    masked_action_entropy_from_params,
    masked_event_log_prob_from_params,
    pack_sequence,
    unpack_sequence,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    MAX_COMETS,
    MAX_PLANETS,
    ActionPureConfig,
    ObsBatch,
    ObsV1Config,
)
from pydantic import TypeAdapter
from torch import nn


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
        StatelessTransformerV1Config(embed_dim=30, n_heads=8)


def test_model_config_has_discriminator_tag() -> None:
    config = TypeAdapter(ModelConfig).validate_python(
        {
            "model_arch": "stateless_transformer_v1",
            "embed_dim": 32,
            "n_heads": 4,
        }
    )

    assert config.model_arch == "stateless_transformer_v1"


def test_model_config_accepts_deprecated_angle_mixture_alias() -> None:
    config = TypeAdapter(ModelConfig).validate_python(
        {
            "model_arch": "stateless_transformer_v1",
            "n_angle_mixtures": 2,
        }
    )

    assert config.n_action_mixtures == 2
    assert config.n_angle_mixtures == 2


def test_model_constructor_does_not_require_flash_attn_on_cuda_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=32, depth=1, n_heads=4)
    )


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


def test_pack_sequence_rejects_fully_masked_rows() -> None:
    x = torch.zeros((2, 3, 4))
    mask = torch.tensor([[True, False, False], [False, False, False]])

    with pytest.raises(ValueError, match="at least one unmasked token"):
        pack_sequence(x, mask)


def test_attention_and_swiglu_use_separate_projection_matrices_for_muon() -> None:
    config = StatelessTransformerV1Config(embed_dim=32, n_heads=4, activation="swiglu")

    attn = MultiHeadSelfAttention(config)
    mlp = FeedForward(config)

    assert attn.q is not attn.k
    assert attn.k is not attn.v
    assert mlp.gate is not mlp.value


def test_model_initialization_sets_stable_rl_priors() -> None:
    torch.manual_seed(0)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=2,
        n_heads=4,
        n_action_mixtures=2,
    )
    model = StatelessTransformerV1(config)

    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            assert torch.allclose(module.weight, torch.ones_like(module.weight))
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    for module in model.get_input_layers():
        if isinstance(module, nn.Linear):
            assert module.bias is not None
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))

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
    for head in (
        model.actor_heads.continue_head,
        model.actor_heads.mix_head,
        model.actor_heads.dir_head,
        model.actor_heads.kappa_head,
        model.actor_heads.size_frac_head,
        model.actor_heads.size_conc_head,
    ):
        assert torch.allclose(
            head.weight.norm(dim=1),
            torch.full((head.out_features,), 0.01),
            atol=1e-6,
        )

    params = model.actor_heads(torch.zeros((1, 1, 1, config.embed_dim)))
    assert torch.allclose(
        params.continue_logits,
        torch.full_like(params.continue_logits, -2.0),
    )
    assert torch.allclose(
        params.continue_logits.sigmoid(),
        torch.full_like(params.continue_logits, 0.11920292),
    )
    assert torch.allclose(params.mix_logits, torch.zeros_like(params.mix_logits))
    assert torch.allclose(torch.cos(params.loc), torch.tensor([[[[1.0, -1.0]]]]))
    assert torch.allclose(
        torch.sin(params.loc),
        torch.zeros_like(params.loc),
        atol=1e-6,
    )
    assert torch.allclose(params.kappa, torch.full_like(params.kappa, 1.0))
    assert torch.allclose(
        params.alpha,
        torch.full_like(params.alpha, 1.0 + config.alpha_beta_eps),
    )
    assert torch.allclose(
        params.beta,
        torch.full_like(params.beta, 1.0 + config.alpha_beta_eps),
    )


def test_observation_encoder_returns_entity_tokens_plus_player_tokens() -> None:
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 3)
    config = StatelessTransformerV1Config(
        obs_spec=obs_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = StatelessTransformerV1(config)
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
    config = StatelessTransformerV1Config(
        obs_spec=obs_spec,
        action_spec=action_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
        n_action_mixtures=2,
    )
    model = StatelessTransformerV1(config)
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
    assert output.log_probs.per_player_entity.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert output.entropies.launch.shape == expected_action_shape
    assert output.entropies.angle_and_size.shape == expected_action_shape
    assert output.entropies.per_player_entity.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert torch.allclose(
        output.log_probs.per_player_entity,
        (output.log_probs.launch + output.log_probs.angle_and_size).sum(dim=-1),
    )
    assert torch.allclose(
        output.entropies.per_player_entity,
        (output.entropies.launch + output.entropies.angle_and_size).sum(dim=-1),
    )
    assert torch.isfinite(output.entropies.per_player_entity).all()
    assert output.values.shape == (2, 4)
    assert output.winner_probabilities.shape == (2, 4)
    assert torch.allclose(output.winner_probabilities.sum(dim=1), torch.ones(2))
    assert torch.all(output.winner_probabilities[~obs.still_playing] == 0)
    assert torch.all(output.actions.ships[~output.actions.launch] == 0)
    assert torch.all(output.actions.ships.sum(dim=-1) <= obs.max_launch)
    assert model.launch_slot_tokens.weight.shape == (3, config.embed_dim)
    assert model.slot_dynamic_proj.in_features == 9

    evaluation = model.evaluate_actions(obs, output.actions)
    assert torch.allclose(evaluation.log_probs.launch, output.log_probs.launch)
    assert torch.allclose(
        evaluation.log_probs.angle_and_size,
        output.log_probs.angle_and_size,
    )
    assert torch.allclose(
        evaluation.log_probs.per_player_entity,
        output.log_probs.per_player_entity,
    )
    assert torch.allclose(evaluation.entropies.launch, output.entropies.launch)
    assert torch.allclose(
        evaluation.entropies.angle_and_size,
        output.entropies.angle_and_size,
    )
    assert torch.allclose(
        evaluation.entropies.per_player_entity,
        output.entropies.per_player_entity,
    )
    assert torch.allclose(evaluation.values, output.values)
    assert torch.allclose(evaluation.winner_probabilities, output.winner_probabilities)


def test_launch_slot_embedding_is_added_to_each_slot_input() -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=4)
    config = StatelessTransformerV1Config(
        action_spec=action_spec,
        embed_dim=16,
        depth=1,
        n_heads=4,
    )
    model = StatelessTransformerV1(config)
    slot_input = torch.zeros((2, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    active = torch.zeros(slot_input.shape[:-1], dtype=torch.bool)
    remaining = torch.zeros(slot_input.shape[:-1], dtype=torch.int64)
    initial_max_launch = torch.zeros(slot_input.shape[:-1], dtype=torch.int64)
    last_launch = torch.zeros(slot_input.shape[:-1], dtype=torch.bool)
    last_angle_sin = torch.zeros(slot_input.shape[:-1])
    last_angle_cos = torch.zeros(slot_input.shape[:-1])
    last_ships = torch.zeros(slot_input.shape[:-1], dtype=torch.int64)

    first_slot = model._slot_gru_input(
        slot_input,
        0,
        active,
        remaining,
        initial_max_launch,
        last_launch,
        last_angle_sin,
        last_angle_cos,
        last_ships,
        include_dynamic_features=False,
    )
    second_slot = model._slot_gru_input(
        slot_input,
        1,
        active,
        remaining,
        initial_max_launch,
        last_launch,
        last_angle_sin,
        last_angle_cos,
        last_ships,
        include_dynamic_features=False,
    )

    expected_first_slot = model.launch_slot_tokens.weight[0].view(1, 1, 1, -1)
    assert torch.allclose(first_slot, expected_first_slot.expand_as(first_slot))
    assert not torch.allclose(first_slot, second_slot)


def test_slot_dynamic_features_include_relative_budget_and_slot_fraction() -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=4)
    config = StatelessTransformerV1Config(
        action_spec=action_spec,
        embed_dim=16,
        depth=1,
        n_heads=4,
        max_ship_normalizer=100.0,
    )
    model = StatelessTransformerV1(config)
    active = torch.tensor([[[True, False]]])
    remaining = torch.tensor([[[6, 0]]])
    initial_max_launch = torch.tensor([[[10, 0]]])
    last_launch = torch.tensor([[[True, False]]])
    last_angle_sin = torch.tensor([[[0.25, -0.5]]])
    last_angle_cos = torch.tensor([[[0.75, 0.5]]])
    last_ships = torch.tensor([[[2, 3]]])

    features = model._slot_dynamic_features(
        2,
        active,
        remaining,
        initial_max_launch,
        last_launch,
        last_angle_sin,
        last_angle_cos,
        last_ships,
        dtype=torch.float32,
    )

    expected = torch.tensor(
        [
            [
                [
                    1.0,
                    0.06,
                    1.0,
                    0.25,
                    0.75,
                    0.02,
                    0.6,
                    0.2,
                    2.0 / 3.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    0.5,
                    0.03,
                    0.0,
                    3.0,
                    2.0 / 3.0,
                ],
            ]
        ]
    )
    assert torch.allclose(features, expected)


def test_actor_distribution_outputs_remain_fp32_under_cpu_bf16_autocast() -> None:
    torch.manual_seed(0)
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=2)
    config = StatelessTransformerV1Config(
        obs_spec=obs_spec,
        action_spec=action_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
        n_action_mixtures=2,
    )
    model = StatelessTransformerV1(config)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(obs)
        evaluation = model.evaluate_actions(obs, output.actions)

    assert output.actions.launch.dtype == torch.bool
    assert output.actions.angle.dtype == torch.float32
    assert output.actions.ships.dtype == torch.int64
    for tensors in (output.log_probs, output.entropies, evaluation.log_probs):
        assert tensors.launch.dtype == torch.float32
        assert tensors.angle_and_size.dtype == torch.float32
        assert tensors.per_player_entity.dtype == torch.float32
        assert torch.isfinite(tensors.launch).all()
        assert torch.isfinite(tensors.angle_and_size).all()
        assert torch.isfinite(tensors.per_player_entity).all()


def test_distribution_helpers_promote_lower_precision_params_to_fp32() -> None:
    torch.manual_seed(0)
    mixtures = 2
    shape = (1, 4, ACTION_ENTITY_SLOTS, mixtures)
    mix_logits = torch.zeros(shape, dtype=torch.bfloat16)
    params = PolicyParams(
        continue_logits=torch.zeros(shape[:-1], dtype=torch.bfloat16),
        mix_logits=mix_logits,
        log_w=torch.log_softmax(mix_logits, dim=-1),
        loc=torch.zeros(shape, dtype=torch.bfloat16),
        kappa=torch.ones(shape, dtype=torch.bfloat16),
        alpha=torch.full(shape, 1.5, dtype=torch.bfloat16),
        beta=torch.full(shape, 2.0, dtype=torch.bfloat16),
    )
    active = torch.ones(shape[:-1], dtype=torch.bool)
    residual_budget = torch.full(shape[:-1], 5, dtype=torch.int64)
    angle = torch.full(shape[:-1], 0.25, dtype=torch.float32)
    ships = torch.ones(shape[:-1], dtype=torch.int64)
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=32, depth=1, n_heads=4)
    )

    launch = model._sample_launch(params.continue_logits, active, deterministic=False)
    sampled_angle, sampled_ships = model._sample_event(
        params,
        residual_budget,
        deterministic=False,
    )
    event_log_prob = masked_event_log_prob_from_params(
        params,
        angle,
        ships,
        residual_budget,
        active,
    )
    launch_entropy, event_entropy = masked_action_entropy_from_params(
        params,
        residual_budget,
        active,
        max_ship_support=250,
    )

    assert launch.dtype == torch.bool
    assert sampled_angle.dtype == torch.float32
    assert sampled_ships.dtype == torch.int64
    for tensor in (event_log_prob, launch_entropy, event_entropy):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_action_entropy_is_finite_above_ship_support_cap() -> None:
    mixtures = 1
    shape = (1, 1, 1, mixtures)
    params = PolicyParams(
        continue_logits=torch.zeros(shape[:-1]),
        mix_logits=torch.zeros(shape),
        log_w=torch.zeros(shape),
        loc=torch.zeros(shape),
        kappa=torch.ones(shape),
        alpha=torch.ones(shape),
        beta=torch.ones(shape),
    )
    residual_budget = torch.tensor([[[11]]])
    active = torch.ones(shape[:-1], dtype=torch.bool)

    launch_entropy, event_entropy = masked_action_entropy_from_params(
        params,
        residual_budget,
        active,
        max_ship_support=3,
    )
    _, wider_event_entropy = masked_action_entropy_from_params(
        params,
        residual_budget,
        active,
        max_ship_support=11,
    )

    assert torch.isfinite(launch_entropy).all()
    assert torch.isfinite(event_entropy).all()
    assert torch.all(event_entropy < wider_event_entropy)


def test_actor_log_probs_have_finite_gradients_for_masked_slots() -> None:
    torch.manual_seed(1)
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=3)
    config = StatelessTransformerV1Config(
        obs_spec=obs_spec,
        action_spec=action_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
        n_action_mixtures=2,
    )
    model = StatelessTransformerV1(config)
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


def test_beta_binomial_entropy_uses_static_capped_support() -> None:
    entropy = beta_binomial_entropy(
        torch.tensor([300]),
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([[1.0]], dtype=torch.float64),
        max_ship_support=250,
    )

    expected = (250.0 / 300.0) * torch.log(torch.tensor(300.0, dtype=torch.float64))
    assert torch.allclose(entropy, expected.view(1, 1), atol=1e-3)


def test_k_max_is_hard_truncation_and_replays_without_final_stop_probability() -> None:
    torch.manual_seed(2)
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=3)
    config = StatelessTransformerV1Config(
        obs_spec=obs_spec,
        action_spec=action_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
        n_action_mixtures=2,
    )
    model = StatelessTransformerV1(config)
    obs = _obs_batch(batch_size=1, obs_spec=obs_spec, action_spec=action_spec)
    obs.max_launch[0, 0, 0] = 100
    with torch.no_grad():
        model.actor_heads.continue_head.bias.fill_(100.0)

    output = model(obs, deterministic=True)

    assert output.actions.launch[0, 0, 0].all()
    assert output.actions.ships[0, 0, 0].sum() < obs.max_launch[0, 0, 0]

    evaluation = model.evaluate_actions(obs, output.actions)

    assert torch.allclose(evaluation.log_probs.launch, output.log_probs.launch)
    assert torch.allclose(
        evaluation.log_probs.angle_and_size,
        output.log_probs.angle_and_size,
    )
    assert torch.allclose(
        evaluation.log_probs.per_player_entity,
        output.log_probs.per_player_entity,
    )


def test_evaluate_actions_rejects_invalid_action_dtypes() -> None:
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 1)
    config = StatelessTransformerV1Config(
        obs_spec=obs_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = StatelessTransformerV1(config)
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
        model.evaluate_actions(obs, output.actions)


def test_critic_requires_still_playing_mask_with_live_player() -> None:
    obs_spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 1)
    config = StatelessTransformerV1Config(
        obs_spec=obs_spec,
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = StatelessTransformerV1(config)
    obs = _obs_batch(
        batch_size=1,
        obs_spec=obs_spec,
        action_spec=config.action_spec,
    )

    obs.still_playing.fill_(False)

    with pytest.raises(ValueError, match="at least one player"):
        model(obs)
