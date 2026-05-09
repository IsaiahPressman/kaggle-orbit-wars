import math
from pathlib import Path
from typing import Any

import owl.model.actor.discrete_targets as discrete_targets_impl
import owl.model.stateless_transformer_v1 as model_impl
import pytest
import torch
import torch.nn.functional as F
import yaml
from owl.model import (
    ModelConfig,
    StatelessTransformerV1,
    StatelessTransformerV1Config,
)
from owl.model.actor import ActorConfig
from owl.model.actor.discrete_targets import (
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
    DiscreteTargetPolicyParams,
    DiscreteTargetsActor,
    FeedForward,
    MinGRUCell,
    MultiHeadSelfAttention,
    OutputProjectionMLP,
    PolicyParams,
    PureActor,
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
    DiscreteTargetActions,
    DiscreteTargetBinActions,
    EntityBasedConfig,
    ObsBatch,
    PureActions,
)
from pydantic import TypeAdapter
from torch import nn

_REPO_ROOT = Path(__file__).parents[3]


def _obs_batch(
    *,
    batch_size: int,
    obs_spec: EntityBasedConfig,
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

    return ObsBatch(
        planets=planets,
        orbiting_planets=orbiting_planets,
        fleets=fleets,
        comets=comets,
        entity_mask=entity_mask,
        still_playing=still_playing,
        global_features=global_features,
        can_act=can_act,
        max_launch=max_launch,
    )


def _model(
    config: StatelessTransformerV1Config,
    *,
    obs_spec: EntityBasedConfig | None = None,
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
) -> DiscreteActorInputs:
    return DiscreteActorInputs(
        source=source, target=source if target is None else target
    )


def test_model_config_requires_heads_to_divide_embed_dim() -> None:
    with pytest.raises(ValueError, match="n_heads must evenly divide embed_dim"):
        StatelessTransformerV1Config(embed_dim=30, n_heads=8)


def test_model_config_requires_positive_feedforward_width() -> None:
    with pytest.raises(ValueError, match="embed_dim \\* mlp_ratio must be at least 1"):
        StatelessTransformerV1Config(embed_dim=1, n_heads=1, mlp_ratio=0.5)


def test_actor_pure_config_requires_ordered_kappa_bounds() -> None:
    with pytest.raises(ValueError, match="kappa_min must be <= kappa_max"):
        ActorPureConfig(kappa_min=2.0, kappa_max=1.0)


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
        _REPO_ROOT / "configs" / "model" / "stateless_transformer_tiny.yaml"
    )

    assert config.actor == ActorDiscreteTargetsConfig()


@pytest.mark.parametrize(
    "config_path",
    sorted((_REPO_ROOT / "configs" / "model").glob("*.yaml")),
)
def test_model_config_files_load(config_path: Path) -> None:
    _ = StatelessTransformerV1Config.from_file(config_path)


@pytest.mark.parametrize(
    "config_path",
    sorted((_REPO_ROOT / "configs" / "model" / "actor").glob("*.yaml")),
)
def test_actor_config_files_load(config_path: Path) -> None:
    with config_path.open(encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    _ = TypeAdapter(ActorConfig).validate_python(config_data)


@pytest.mark.parametrize(
    ("filename", "expected_params"),
    [
        ("stateless_transformer_tiny.yaml", 1_207_182),
        ("stateless_transformer_5m.yaml", 5_532_942),
        ("stateless_transformer_20m.yaml", 20_093_402),
        ("stateless_transformer_20m_swiglu.yaml", 20_914_202),
    ],
)
def test_model_config_file_parameter_count(
    filename: str,
    expected_params: int,
) -> None:
    config = StatelessTransformerV1Config.from_file(
        _REPO_ROOT / "configs" / "model" / filename
    )
    model = StatelessTransformerV1(
        config,
        obs_spec=EntityBasedConfig(),
        action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == expected_params


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


def test_model_outputs_do_not_change_with_extra_masked_fleets() -> None:
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
        entity_mask=torch.cat(
            (
                compact.entity_mask,
                torch.zeros((1, 3), dtype=compact.entity_mask.dtype),
            ),
            dim=1,
        ),
        still_playing=compact.still_playing,
        global_features=compact.global_features,
        can_act=compact.can_act,
        max_launch=compact.max_launch,
    )

    with torch.inference_mode():
        compact_output = model(compact, deterministic=True)
        padded_output = model(padded, deterministic=True)

    torch.testing.assert_close(
        compact_output.values, padded_output.values, atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        compact_output.winner_probabilities,
        padded_output.winner_probabilities,
        atol=1e-6,
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
        atol=1e-6,
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
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.log_probs.event,
        padded_output.log_probs.event,
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.log_probs.per_player_entity,
        padded_output.log_probs.per_player_entity,
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.entropies.launch,
        padded_output.entropies.launch,
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.entropies.event,
        padded_output.entropies.event,
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        compact_output.entropies.per_player_entity,
        padded_output.entropies.per_player_entity,
        atol=1e-6,
        rtol=0,
    )


def test_model_constructor_does_not_require_flash_attn_on_cuda_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    _model(StatelessTransformerV1Config(embed_dim=32, depth=1, n_heads=4))


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
    output_head = OutputProjectionMLP(config, output_dim=3)

    assert attn.q is not attn.k
    assert attn.k is not attn.v
    assert mlp.gate is not mlp.value
    assert output_head.gate is not output_head.value


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
        max_seqlen: int,  # noqa: ARG001
    ) -> torch.Tensor:
        nonlocal varlen_calls
        varlen_calls += 1
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
    ) -> tuple[torch.Tensor, model_impl.PackedSequence]:
        nonlocal pack_calls
        pack_calls += 1
        assert x.dtype == torch.bfloat16
        return original_pack_sequence(x, token_mask)

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
        actor=ActorPureConfig(n_action_mixtures=2),
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
        model.actor.actor_heads.mix_head,
        model.actor.actor_heads.dir_head,
        model.actor.actor_heads.kappa_head,
        model.actor.actor_heads.size_frac_head,
        model.actor.actor_heads.size_conc_head,
    ):
        assert id(head.out) in output_layer_ids
        assert id(head.up) not in output_layer_ids
        assert torch.allclose(
            head.weight.norm(dim=1),
            torch.full((head.out_features,), 0.01),
            atol=1e-6,
        )

    params = model.actor.actor_heads(torch.zeros((1, 1, 1, config.embed_dim)))
    assert torch.allclose(
        params.continue_logits,
        torch.zeros_like(params.continue_logits),
    )
    assert torch.allclose(params.mix_logits, torch.zeros_like(params.mix_logits))
    assert torch.allclose(torch.cos(params.loc), torch.ones_like(params.loc))
    assert torch.allclose(
        torch.sin(params.loc),
        torch.zeros_like(params.loc),
        atol=1e-6,
    )
    actor_config = config.actor
    assert isinstance(actor_config, ActorPureConfig)
    expected_concentration = F.softplus(torch.tensor(0.0)) + actor_config.kappa_min
    assert torch.allclose(
        params.kappa,
        torch.full_like(params.kappa, expected_concentration),
    )
    expected_size_concentration = F.softplus(torch.tensor(0.0)) + actor_config.tau_min
    assert torch.allclose(
        params.alpha,
        torch.full_like(
            params.alpha,
            0.5 * expected_size_concentration + actor_config.alpha_beta_eps,
        ),
    )
    assert torch.allclose(
        params.beta,
        torch.full_like(
            params.beta,
            0.5 * expected_size_concentration + actor_config.alpha_beta_eps,
        ),
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


def test_actor_critic_outputs_action_tensors_log_probs_and_values() -> None:
    torch.manual_seed(0)
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorPureConfig(n_action_mixtures=2),
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
    assert set(output.entropies.components) == {"launch", "event"}
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
    assert torch.all(output.actions.ships.sum(dim=-1) <= obs.max_launch)
    assert model.actor.launch_slot_tokens.shape == (1, config.embed_dim)
    assert model.actor.slot_dynamic_proj.in_features == 9

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
    assert set(evaluation.entropies.components) == {"launch", "event"}
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
    obs.max_launch[:, 0, 0] = 8
    obs.max_launch[:, 1, 1] = 4
    obs.max_launch[:, 2, MAX_PLANETS] = 3

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
    target_valid = obs.can_act.gather(-1, launched_target.unsqueeze(-1)).squeeze(-1)
    assert torch.all(target_valid[output.actions.launch[..., 0]])
    launched_ships = output.actions.ships[output.actions.launch]
    assert torch.all(launched_ships >= action_spec.min_fleet_size)
    assert torch.all(output.actions.ships.sum(dim=-1) <= obs.max_launch)

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
    source_active = obs.can_act.flatten(start_dim=-2).any(dim=-1)
    batch_index = torch.arange(2)[:, None, None]
    player_index = torch.arange(4)[None, :, None]
    source_index = torch.arange(ACTION_ENTITY_SLOTS)[None, None, :]
    selected = obs.can_act[
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
    obs.can_act[0, 0, 0, :, 1] = False
    with pytest.raises(ValueError, match="valid target-bin pair"):
        model.evaluate_actions(obs, invalid)


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
        action_spec=ActionPureConfig(max_per_planet_launches=3),
    )

    pure_input_layer_ids = {id(layer) for layer in pure_model.get_input_layers()}

    for layer in (
        pure_model.player_tokens,
        pure_model.board_tokens,
        pure_model.actor_plan_tokens,
        pure_model.critic_value_tokens,
        pure_model.actor.launch_slot_tokens,
    ):
        assert id(layer) in pure_input_layer_ids

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
    obs.max_launch[0, 0, 0] = action_spec.min_fleet_size
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

    monkeypatch.setattr(discrete_targets_impl, "ship_support", fail_ship_support)
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
        actor=ActorPureConfig(n_action_mixtures=2),
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


def test_launch_slot_embedding_is_added_to_each_slot_input() -> None:
    config = StatelessTransformerV1Config(
        embed_dim=16,
        depth=1,
        n_heads=4,
    )
    actor = PureActor(
        ActorPureConfig(),
        embed_dim=config.embed_dim,
        max_per_planet_launches=4,
        activation=config.activation,
    )
    slot_input = torch.zeros((2, 4, ACTION_ENTITY_SLOTS, config.embed_dim))
    active = torch.zeros(slot_input.shape[:-1], dtype=torch.bool)
    remaining = torch.zeros(slot_input.shape[:-1], dtype=torch.int64)
    initial_max_launch = torch.zeros(slot_input.shape[:-1], dtype=torch.int64)
    last_launch = torch.zeros(slot_input.shape[:-1], dtype=torch.bool)
    last_angle_sin = torch.zeros(slot_input.shape[:-1])
    last_angle_cos = torch.zeros(slot_input.shape[:-1])
    last_ships = torch.zeros(slot_input.shape[:-1], dtype=torch.int64)

    first_slot = actor._slot_gru_input(
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
    second_slot = actor._slot_gru_input(
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

    expected_first_slot = actor.launch_slot_tokens[0].view(1, 1, 1, -1)
    assert torch.allclose(first_slot, expected_first_slot.expand_as(first_slot))
    assert not torch.allclose(first_slot, second_slot)


def test_slot_dynamic_features_include_relative_budget_and_slot_fraction() -> None:
    config = StatelessTransformerV1Config(
        embed_dim=16,
        depth=1,
        n_heads=4,
    )
    actor = PureActor(
        ActorPureConfig(max_ship_normalizer=100.0),
        embed_dim=config.embed_dim,
        max_per_planet_launches=4,
        activation=config.activation,
    )
    active = torch.tensor([[[True, False]]])
    remaining = torch.tensor([[[6, 0]]])
    initial_max_launch = torch.tensor([[[10, 0]]])
    last_launch = torch.tensor([[[True, False]]])
    last_angle_sin = torch.tensor([[[0.25, -0.5]]])
    last_angle_cos = torch.tensor([[[0.75, 0.5]]])
    last_ships = torch.tensor([[[2, 3]]])

    features = actor._slot_dynamic_features(
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
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorPureConfig(n_action_mixtures=2),
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
    model = _model(StatelessTransformerV1Config(embed_dim=32, depth=1, n_heads=4))
    assert isinstance(model.actor, model_impl.PureActor)

    launch = model_impl.sample_launch(
        params.continue_logits,
        active,
        deterministic=False,
    )
    sampled_angle, sampled_ships = model.actor._sample_event(
        params,
        residual_budget,
        1,
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
        mix_logits=mix_logits,
        log_w=torch.log_softmax(mix_logits, dim=-1),
        loc=torch.zeros((2, 2)),
        kappa=torch.zeros((2, 2)),
        alpha=torch.ones((2, 2)),
        beta=torch.ones((2, 2)),
    )
    residual_budget = torch.tensor([3, 3])
    active = torch.tensor([True, False])

    launch_entropy, event_entropy = model_impl.masked_action_entropy_from_params(
        params,
        residual_budget,
        active,
        min_fleet_size=1,
        max_ship_support=3,
    )

    expected_event_entropy = math.log(2.0) + math.log(2.0 * math.pi) + math.log(3.0)
    assert torch.allclose(
        launch_entropy,
        torch.tensor([math.log(2.0), 0.0]),
        atol=1e-6,
    )
    assert torch.allclose(
        event_entropy,
        torch.tensor([0.5 * expected_event_entropy, 0.0]),
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
        actor=ActorPureConfig(n_action_mixtures=2),
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


def test_pure_actor_supports_multi_launch_action_spec() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=3)
    config = StatelessTransformerV1Config(
        embed_dim=32,
        depth=1,
        n_heads=4,
        actor=ActorPureConfig(n_action_mixtures=2),
    )

    model = _model(config, obs_spec=obs_spec, action_spec=action_spec)
    obs = _obs_batch(batch_size=2, obs_spec=obs_spec, action_spec=action_spec)

    output = model(obs)
    assert output.actions.launch.shape == (2, 4, ACTION_ENTITY_SLOTS, 3)

    evaluation = model.evaluate_actions(obs, output.actions)
    assert evaluation.log_probs.per_player_entity.shape == (
        2,
        4,
        ACTION_ENTITY_SLOTS,
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
    output.actions.ships[0, 0, 0, 0] = 1

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
