import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from owl.agent import Agent, KaggleObservation
from owl.agent.agent import (
    AGENT_CONFIG_PATH,
    AgentCheckpointConfig,
    AgentConfig,
    apply_max_entities_override,
    apply_targeting_mode_override,
    compact_entities,
    expand_actions_to_full_action_slots,
)
from owl.agent.checkpoint_quantization import (
    FP4_E2M1FN_X2_SCALED_BLOCK16,
    quantize_model_state_dict,
)
from owl.agent.kaggle_observation import (
    FLEET_ANGLE_INDEX,
    FLEET_FROM_PLANET_ID_INDEX,
    FLEET_ID_INDEX,
    FLEET_OWNER_INDEX,
    FLEET_SHIPS_INDEX,
    FLEET_X_INDEX,
    FLEET_Y_INDEX,
    PLANET_ID_INDEX,
    PLANET_OWNER_INDEX,
    PLANET_PRODUCTION_INDEX,
    PLANET_RADIUS_INDEX,
    PLANET_SHIPS_INDEX,
    PLANET_X_INDEX,
    PLANET_Y_INDEX,
)
from owl.model import (
    RecurrentTransformerV1Config,
    StatelessTransformerV1Config,
    create_model,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    COMET_CHANNELS,
    FLEET_CHANNELS,
    GLOBAL_CHANNELS,
    MAX_COMETS,
    MAX_PLANETS,
    PLANET_CHANNELS,
    ActionDiscreteTargetBinsConfig,
    ActionDiscreteTargetsConfig,
    ActionPureConfig,
    DiscreteTargetActionMask,
    DiscreteTargetActions,
    DiscreteTargetBinActionMask,
    DiscreteTargetBinActions,
    EncodedPythonObservation,
    EntityBasedConfig,
    EnvConfig,
    ObsBatch,
    PureActionMask,
    PureActions,
)
from owl.train.config import FullConfig

_ASSERT_AGENT_IMPORT_ISOLATED = Path(__file__).with_name(
    "assert_agent_import_isolated.py"
)
_REPO_ROOT = Path(__file__).parents[3]


def test_kaggle_row_index_constants_match_rust_observation_parser() -> None:
    source = (_REPO_ROOT / "src/rl/obs_spec.rs").read_text(encoding="utf-8")

    assert _rust_row_index(source, "planet id") == PLANET_ID_INDEX
    assert _rust_row_index(source, "planet owner") == PLANET_OWNER_INDEX
    assert _rust_row_index(source, "planet x") == PLANET_X_INDEX
    assert _rust_row_index(source, "planet y") == PLANET_Y_INDEX
    assert _rust_row_index(source, "planet radius") == PLANET_RADIUS_INDEX
    assert _rust_row_index(source, "planet ships") == PLANET_SHIPS_INDEX
    assert _rust_production_row_index(source) == PLANET_PRODUCTION_INDEX
    assert _rust_row_index(source, "fleet id") == FLEET_ID_INDEX
    assert _rust_row_index(source, "fleet owner") == FLEET_OWNER_INDEX
    assert _rust_row_index(source, "fleet x") == FLEET_X_INDEX
    assert _rust_row_index(source, "fleet y") == FLEET_Y_INDEX
    assert _rust_row_index(source, "fleet angle") == FLEET_ANGLE_INDEX
    assert (
        _rust_row_index(
            source,
            "fleet from_planet_id",
        )
        == FLEET_FROM_PLANET_ID_INDEX
    )
    assert _rust_row_index(source, "fleet ships") == FLEET_SHIPS_INDEX


def _rust_row_index(source: str, field_label: str) -> int:
    match = re.search(rf'row\[(\d+)\], "{re.escape(field_label)}"', source)
    assert match is not None
    return int(match.group(1))


def _rust_production_row_index(source: str) -> int:
    match = re.search(r"production: finite_production\(row\[(\d+)\]\)", source)
    assert match is not None
    return int(match.group(1))


def _dynamic_quantized_linear_count(model: torch.nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if type(module).__module__.startswith("torch.ao.nn.quantized.dynamic")
        and type(module).__name__ == "Linear"
    )


def test_agent_import_does_not_load_training_modules() -> None:
    # Import isolation has to be verified in a fresh interpreter
    result = subprocess.run(
        [
            sys.executable,
            str(_ASSERT_AGENT_IMPORT_ISOLATED),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout


def test_agent_config_path_valid() -> None:
    _ = AgentConfig.from_file(AGENT_CONFIG_PATH)


def test_agent_config_rejects_nonpositive_min_fleet_size() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        AgentConfig(deterministic=True, min_fleet_size=0)


def test_agent_config_rejects_unknown_inference_quantization() -> None:
    with pytest.raises(ValueError, match="Input should be 'int8'"):
        AgentConfig.model_validate(
            {
                "deterministic": True,
                "inference_quantization": "fp16",
            }
        )


def test_agent_checkpoint_config_fields_exist_on_full_config() -> None:
    assert set(AgentCheckpointConfig.model_fields) <= set(FullConfig.model_fields)


def test_agent_act_prints_exception_traceback_before_empty_fallback(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent.__new__(Agent)

    def fail_act(_observation: object) -> list[list[float]]:
        raise RuntimeError("agent failure")

    monkeypatch.setattr(agent, "_act", fail_act)

    assert agent.act(object()) == []
    captured = capsys.readouterr()
    assert "RuntimeError exception caught: Traceback (most recent call last):" in (
        captured.out
    )
    assert "RuntimeError: agent failure" in captured.out
    assert captured.err == ""


def test_agent_init_loads_quantized_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_config_path = tmp_path / "agent_config.yaml"
    agent_config_path.write_text(
        "\n".join(
            (
                "deterministic: true",
                "max_entities_override: null",
                "targeting_mode_override: null",
                "min_overage_time: 0.0",
                "fallback_min_overage_time: 5.0",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("owl.agent.agent.AGENT_CONFIG_PATH", agent_config_path)

    model_config = StatelessTransformerV1Config(embed_dim=8, depth=1, n_heads=1)
    env_config = EnvConfig(
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(),
    )
    checkpoint_config_path = tmp_path / "config.yaml"
    AgentCheckpointConfig(env=env_config, model=model_config).to_file(
        checkpoint_config_path
    )

    model = create_model(
        model_config,
        obs_spec=env_config.obs_spec,
        action_spec=env_config.action_spec,
    )
    model.reset_parameters()
    checkpoint_path = tmp_path / "checkpoint_last_best.pt"
    torch.save(
        {
            "model": quantize_model_state_dict(
                model.state_dict(),
                FP4_E2M1FN_X2_SCALED_BLOCK16,
            )
        },
        checkpoint_path,
    )
    fallback_checkpoint_config_path = tmp_path / "fallback_config.yaml"
    AgentCheckpointConfig(env=env_config, model=model_config).to_file(
        fallback_checkpoint_config_path
    )
    fallback_checkpoint_path = tmp_path / "fallback_checkpoint.pt"
    torch.save({"model": model.state_dict()}, fallback_checkpoint_path)

    agent = Agent(
        checkpoint_config_path=checkpoint_config_path,
        checkpoint_path=checkpoint_path,
        fallback_checkpoint_config_path=fallback_checkpoint_config_path,
        fallback_checkpoint_path=fallback_checkpoint_path,
    )

    assert set(agent.model.state_dict()) == set(model.state_dict())
    assert all(parameter.isfinite().all() for parameter in agent.model.parameters())
    assert agent.fallback_checkpoint_config is not None
    assert agent.fallback_model is None
    assert agent._fallback_checkpoint_path == fallback_checkpoint_path


def test_agent_loads_fallback_model_on_second_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = Agent.__new__(Agent)
    agent.config = AgentConfig(deterministic=True, fallback_min_overage_time=5.0)
    agent.fallback_checkpoint_config = object()
    agent.fallback_model = None
    fallback_checkpoint_path = tmp_path / "fallback_checkpoint.pt"
    agent._fallback_checkpoint_path = fallback_checkpoint_path
    loaded: list[tuple[object, Path]] = []

    class FallbackModel:
        pass

    fallback_model = FallbackModel()

    def fake_load_model_from_config(
        *,
        checkpoint_config: object,
        checkpoint_path: Path,
    ) -> object:
        loaded.append((checkpoint_config, checkpoint_path))
        return fallback_model

    monkeypatch.setattr(agent, "_load_model_from_config", fake_load_model_from_config)

    agent._load_fallback_model_if_due(0)

    assert agent.fallback_model is None
    assert loaded == []
    assert capsys.readouterr().out == ""

    agent._load_fallback_model_if_due(1)

    assert agent.fallback_model is fallback_model
    assert loaded == [(agent.fallback_checkpoint_config, fallback_checkpoint_path)]
    assert capsys.readouterr().out.startswith("fallback_init_s=")


def test_agent_does_not_load_fallback_when_routing_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent.__new__(Agent)
    agent.config = AgentConfig(deterministic=True, fallback_min_overage_time=None)
    agent.fallback_checkpoint_config = object()
    agent.fallback_model = None
    agent._fallback_checkpoint_path = tmp_path / "fallback_checkpoint.pt"

    def fail_load_model_from_config(
        *,
        checkpoint_config: object,  # noqa: ARG001
        checkpoint_path: Path,  # noqa: ARG001
    ) -> object:
        raise AssertionError("disabled fallback should not load")

    monkeypatch.setattr(agent, "_load_model_from_config", fail_load_model_from_config)

    agent._load_fallback_model_if_due(1)

    assert agent.fallback_model is None


def test_agent_init_quantizes_linear_layers_for_int8_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    agent_config_path = tmp_path / "agent_config.yaml"
    agent_config_path.write_text(
        "\n".join(
            (
                "deterministic: true",
                "inference_quantization: int8",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("owl.agent.agent.AGENT_CONFIG_PATH", agent_config_path)

    model_config = StatelessTransformerV1Config(embed_dim=8, depth=1, n_heads=1)
    env_config = EnvConfig(
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(),
    )
    checkpoint_config_path = tmp_path / "config.yaml"
    AgentCheckpointConfig(env=env_config, model=model_config).to_file(
        checkpoint_config_path
    )

    model = create_model(
        model_config,
        obs_spec=env_config.obs_spec,
        action_spec=env_config.action_spec,
    )
    model.reset_parameters()
    checkpoint_path = tmp_path / "checkpoint_last_best.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    agent = Agent(
        checkpoint_config_path=checkpoint_config_path,
        checkpoint_path=checkpoint_path,
    )

    assert agent.device == torch.device("cpu")
    output_layer_ids = {id(layer) for layer in agent.model.get_output_layers()}
    linear_layer_ids = {
        id(module)
        for module in agent.model.modules()
        if isinstance(module, torch.nn.Linear)
    }
    assert linear_layer_ids == output_layer_ids
    assert _dynamic_quantized_linear_count(agent.model) > 0
    obs = _obs_batch(max_fleets=0)
    obs.entity_mask[0, 0] = True
    obs.still_playing[:] = True
    output = agent.model.serve(obs, deterministic=True)
    assert output.values.shape == (1, 4)


def test_agent_init_rejects_recurrent_fallback_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_config_path = tmp_path / "agent_config.yaml"
    agent_config_path.write_text(
        "\n".join(
            (
                "deterministic: true",
                "max_entities_override: null",
                "targeting_mode_override: null",
                "min_overage_time: 0.0",
                "fallback_min_overage_time: 5.0",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("owl.agent.agent.AGENT_CONFIG_PATH", agent_config_path)

    env_config = EnvConfig(
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionDiscreteTargetsConfig(),
    )
    primary_model_config = StatelessTransformerV1Config(
        embed_dim=8,
        depth=1,
        n_heads=1,
        actor={"action_spec": "discrete_targets"},
    )
    primary_config_path = tmp_path / "primary_config.yaml"
    AgentCheckpointConfig(env=env_config, model=primary_model_config).to_file(
        primary_config_path
    )
    primary_model = create_model(
        primary_model_config,
        obs_spec=env_config.obs_spec,
        action_spec=env_config.action_spec,
    )
    primary_checkpoint_path = tmp_path / "primary_checkpoint.pt"
    torch.save({"model": primary_model.state_dict()}, primary_checkpoint_path)

    fallback_model_config = RecurrentTransformerV1Config(
        embed_dim=8,
        depth=1,
        n_heads=1,
    )
    fallback_config_path = tmp_path / "fallback_config.yaml"
    AgentCheckpointConfig(env=env_config, model=fallback_model_config).to_file(
        fallback_config_path
    )
    fallback_checkpoint_path = tmp_path / "fallback_checkpoint.pt"
    torch.save({"model": {}}, fallback_checkpoint_path)

    with pytest.raises(ValueError, match="fallback model cannot be recurrent"):
        Agent(
            checkpoint_config_path=primary_config_path,
            checkpoint_path=primary_checkpoint_path,
            fallback_checkpoint_config_path=fallback_config_path,
            fallback_checkpoint_path=fallback_checkpoint_path,
        )


def test_agent_config_max_entities_override_updates_checkpoint_obs_spec() -> None:
    config = AgentCheckpointConfig(
        env=EnvConfig(obs_spec=EntityBasedConfig(max_entities=128)),
        model=StatelessTransformerV1Config(),
    )

    overridden = apply_max_entities_override(config, 256)

    assert config.env.obs_spec.max_entities == 128
    assert overridden.env.obs_spec.max_entities == 256


def test_agent_config_max_entities_override_uses_obs_spec_validation() -> None:
    config = AgentCheckpointConfig(
        env=EnvConfig(obs_spec=EntityBasedConfig(max_entities=128)),
        model=StatelessTransformerV1Config(),
    )

    with pytest.raises(ValueError, match="greater than 44"):
        apply_max_entities_override(
            config,
            2,
        )


def test_agent_config_targeting_mode_override_updates_checkpoint_action_spec() -> None:
    config = AgentCheckpointConfig(
        env=EnvConfig(
            action_spec=ActionDiscreteTargetsConfig(targeting_mode="full_mask")
        ),
        model=StatelessTransformerV1Config(actor={"action_spec": "discrete_targets"}),
    )

    overridden = apply_targeting_mode_override(config, "stop_bad_launch")

    assert config.env.action_spec.targeting_mode == "full_mask"
    assert overridden.env.action_spec.targeting_mode == "stop_bad_launch"


def test_agent_config_targeting_mode_override_updates_target_bins_spec() -> None:
    config = AgentCheckpointConfig(
        env=EnvConfig(
            action_spec=ActionDiscreteTargetBinsConfig(
                n_bins=7,
                targeting_mode="full_mask",
            )
        ),
        model=StatelessTransformerV1Config(
            actor={"action_spec": "discrete_target_bins", "n_bins": 7}
        ),
    )

    overridden = apply_targeting_mode_override(config, "anything_goes")

    assert config.env.action_spec.targeting_mode == "full_mask"
    assert overridden.env.action_spec.targeting_mode == "anything_goes"


def test_agent_config_targeting_mode_override_warns_for_pure_action_spec(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = AgentCheckpointConfig(
        env=EnvConfig(action_spec=ActionPureConfig()),
        model=StatelessTransformerV1Config(),
    )

    overridden = apply_targeting_mode_override(config, "full_mask")

    assert overridden == config
    assert "warning: targeting_mode_override is ignored for pure action_spec" in (
        capsys.readouterr().out
    )


def test_compact_runtime_entities_keeps_active_action_slots_and_fleets() -> None:
    obs = _obs_batch(max_fleets=5)
    action_mask = obs.action_mask
    assert isinstance(action_mask, PureActionMask)
    obs.fleets[0, 1, 0] = 2.0
    obs.fleets[0, 4, 0] = 5.0
    obs.entity_mask[0, 0] = True
    obs.entity_mask[0, 3] = True
    obs.entity_mask[0, MAX_PLANETS + 1] = True
    obs.entity_mask[0, ACTION_ENTITY_SLOTS + 1] = True
    obs.entity_mask[0, ACTION_ENTITY_SLOTS + 4] = True
    action_mask.can_act[0, 0, 0] = True
    action_mask.can_act[0, 1, 3] = True
    action_mask.can_act[0, 2, MAX_PLANETS + 1] = True
    action_mask.max_launch[0, 0, 0] = 9
    action_mask.max_launch[0, 1, 3] = 7
    action_mask.max_launch[0, 2, MAX_PLANETS + 1] = 5

    compacted = compact_entities(obs)
    compact_obs = compacted.obs

    assert compacted.action_entity_indices.tolist() == [0, 3, MAX_PLANETS + 1]
    assert compact_obs.entity_mask.shape == (1, 5)
    assert compact_obs.planets.shape == (1, 2, PLANET_CHANNELS)
    assert compact_obs.comets.shape == (1, 1, COMET_CHANNELS)
    assert compact_obs.fleets.shape == (1, 2, FLEET_CHANNELS)
    assert compact_obs.entity_mask[0].tolist() == [True, True, True, True, True]
    assert compact_obs.fleets[0, :, 0].tolist() == [2.0, 5.0]
    assert isinstance(compact_obs.action_mask, PureActionMask)
    assert compact_obs.action_mask.can_act.shape == (1, 4, 3)
    assert compact_obs.action_mask.can_act[0, :, :].nonzero().tolist() == [
        [0, 0],
        [1, 1],
        [2, 2],
    ]
    assert compact_obs.action_mask.max_launch[0, :, :].tolist() == [
        [9, 0, 0],
        [0, 7, 0],
        [0, 0, 5],
        [0, 0, 0],
    ]


def test_compact_runtime_entities_remaps_cross_attention_fleet_targets() -> None:
    obs = _obs_batch(max_fleets=3)
    obs.fleet_target = torch.full((1, 3), -1, dtype=torch.int64)
    obs.target_incoming_features = torch.zeros((1, ACTION_ENTITY_SLOTS, 2))
    obs.target_incoming_features[0, 3, 0] = 7.0
    obs.target_incoming_features[0, MAX_PLANETS + 1, 1] = 9.0
    obs.entity_mask[0, 0] = True
    obs.entity_mask[0, 3] = True
    obs.entity_mask[0, MAX_PLANETS + 1] = True
    obs.entity_mask[0, ACTION_ENTITY_SLOTS] = True
    obs.entity_mask[0, ACTION_ENTITY_SLOTS + 2] = True
    obs.fleet_target[0, 0] = MAX_PLANETS + 1
    obs.fleet_target[0, 2] = 3

    compacted = compact_entities(obs).obs

    assert compacted.fleet_target is not None
    assert compacted.fleet_target.tolist() == [[2, 1]]
    assert compacted.target_incoming_features is not None
    assert compacted.target_incoming_features[0, :, :].tolist() == [
        [0.0, 0.0],
        [7.0, 0.0],
        [0.0, 9.0],
    ]


def test_compact_runtime_entities_can_preserve_planet_slots() -> None:
    obs = _obs_batch(max_fleets=5)
    action_mask = obs.action_mask
    assert isinstance(action_mask, PureActionMask)
    obs.fleets[0, 1, 0] = 2.0
    obs.fleets[0, 4, 0] = 5.0
    obs.entity_mask[0, 0] = True
    obs.entity_mask[0, 3] = True
    obs.entity_mask[0, MAX_PLANETS + 1] = True
    obs.entity_mask[0, ACTION_ENTITY_SLOTS + 1] = True
    obs.entity_mask[0, ACTION_ENTITY_SLOTS + 4] = True
    action_mask.can_act[0, 1, 3] = True
    action_mask.can_act[0, 2, MAX_PLANETS + 1] = True
    action_mask.max_launch[0, 1, 3] = 7
    action_mask.max_launch[0, 2, MAX_PLANETS + 1] = 5

    compacted = compact_entities(obs, compact_planets=False)
    compact_obs = compacted.obs

    assert compacted.action_entity_indices.tolist() == [
        *range(MAX_PLANETS),
        MAX_PLANETS + 1,
    ]
    assert compact_obs.planets.shape == (1, MAX_PLANETS, PLANET_CHANNELS)
    assert compact_obs.comets.shape == (1, 1, COMET_CHANNELS)
    assert compact_obs.fleets.shape == (1, 2, FLEET_CHANNELS)
    assert compact_obs.entity_mask.shape == (1, MAX_PLANETS + 3)
    assert compact_obs.entity_mask[0, :MAX_PLANETS].tolist() == (
        obs.entity_mask[0, :MAX_PLANETS].tolist()
    )
    assert isinstance(compact_obs.action_mask, PureActionMask)
    assert compact_obs.action_mask.can_act.shape == (1, 4, MAX_PLANETS + 1)
    assert compact_obs.action_mask.can_act[0].nonzero().tolist() == [
        [1, 3],
        [2, MAX_PLANETS],
    ]
    assert compact_obs.action_mask.max_launch[0, 1, 3] == 7
    assert compact_obs.action_mask.max_launch[0, 2, MAX_PLANETS] == 5


def test_compact_runtime_entities_allows_zero_fleets() -> None:
    obs = _obs_batch(max_fleets=5)
    obs.entity_mask[0, 0] = True

    compacted = compact_entities(obs)
    compact_obs = compacted.obs

    assert compacted.action_entity_indices.tolist() == [0]
    assert compact_obs.entity_mask.shape == (1, 1)
    assert compact_obs.fleets.shape == (1, 0, FLEET_CHANNELS)


def test_compact_runtime_entities_compacts_discrete_target_mask() -> None:
    obs = _obs_batch(max_fleets=0)
    obs.entity_mask[0, 0] = True
    obs.entity_mask[0, 3] = True
    obs.entity_mask[0, MAX_PLANETS + 1] = True
    can_act = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS))
    can_act = can_act.bool()
    can_act[0, 0, 0, 3] = True
    can_act[0, 1, 3, MAX_PLANETS + 1] = True
    can_act[0, 2, MAX_PLANETS + 1, 0] = True
    max_launch = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
    max_launch[0, 0, 0] = 9
    max_launch[0, 1, 3] = 7
    max_launch[0, 2, MAX_PLANETS + 1] = 5
    obs.action_mask = DiscreteTargetActionMask(
        can_act=can_act,
        max_launch=max_launch,
    )

    compacted = compact_entities(obs).obs

    assert isinstance(compacted.action_mask, DiscreteTargetActionMask)
    assert compacted.action_mask.can_act.shape == (1, 4, 3, 3)
    assert compacted.action_mask.can_act[0].nonzero().tolist() == [
        [0, 0, 1],
        [1, 1, 2],
        [2, 2, 0],
    ]
    assert compacted.action_mask.max_launch[0].tolist() == [
        [9, 0, 0],
        [0, 7, 0],
        [0, 0, 5],
        [0, 0, 0],
    ]


def test_compact_runtime_entities_compacts_target_bin_mask() -> None:
    obs = _obs_batch(max_fleets=0)
    obs.entity_mask[0, 0] = True
    obs.entity_mask[0, 3] = True
    obs.entity_mask[0, MAX_PLANETS + 1] = True
    can_act = torch.zeros((1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS, 3))
    can_act = can_act.bool()
    can_act[0, 0, 0, 3, 2] = True
    can_act[0, 1, 3, MAX_PLANETS + 1, 1] = True
    can_act[0, 2, MAX_PLANETS + 1, 0, 0] = True
    obs.action_mask = DiscreteTargetBinActionMask(can_act=can_act)

    compacted = compact_entities(obs).obs

    assert isinstance(compacted.action_mask, DiscreteTargetBinActionMask)
    assert compacted.action_mask.can_act.shape == (1, 4, 3, 3, 3)
    assert compacted.action_mask.can_act[0].nonzero().tolist() == [
        [0, 0, 1, 2],
        [1, 1, 2, 1],
        [2, 2, 0, 0],
    ]


def test_expand_actions_to_full_action_slots_remaps_discrete_targets() -> None:
    action_entity_indices = torch.tensor([0, 3, MAX_PLANETS + 1])
    action_shape = (1, 4, 3, 1)
    launch = torch.zeros(action_shape, dtype=torch.bool)
    target = torch.zeros(action_shape, dtype=torch.int64)
    ships = torch.zeros(action_shape, dtype=torch.int64)
    launch[0, 0, 1, 0] = True
    target[0, 0, 1, 0] = 2
    ships[0, 0, 1, 0] = 7

    expanded = expand_actions_to_full_action_slots(
        DiscreteTargetActions(launch=launch, target=target, ships=ships),
        action_entity_indices,
        action_spec=ActionDiscreteTargetsConfig(),
    )

    assert isinstance(expanded, DiscreteTargetActions)
    assert expanded.launch.shape == (1, 4, ACTION_ENTITY_SLOTS, 1)
    assert expanded.launch[0, 0, 3, 0]
    assert expanded.target[0, 0, 3, 0] == MAX_PLANETS + 1
    assert expanded.ships[0, 0, 3, 0] == 7
    assert expanded.launch.sum().item() == 1


def test_expand_actions_to_full_action_slots_remaps_target_bins() -> None:
    action_entity_indices = torch.tensor([0, 3, MAX_PLANETS + 1])
    target = torch.zeros((1, 4, 3), dtype=torch.int64)
    fleet_bin = torch.zeros((1, 4, 3), dtype=torch.int64)
    target[0, 0, 1] = 2
    fleet_bin[0, 0, 1] = 4

    expanded = expand_actions_to_full_action_slots(
        DiscreteTargetBinActions(target=target, fleet_bin=fleet_bin),
        action_entity_indices,
        action_spec=ActionDiscreteTargetBinsConfig(n_bins=5),
    )

    assert isinstance(expanded, DiscreteTargetBinActions)
    assert expanded.target.shape == (1, 4, ACTION_ENTITY_SLOTS)
    assert expanded.target[0, 0, 3] == MAX_PLANETS + 1
    assert expanded.fleet_bin[0, 0, 3] == 4
    assert expanded.fleet_bin.sum().item() == 4


def _obs_batch(*, max_fleets: int) -> ObsBatch:
    return ObsBatch(
        planets=torch.zeros((1, MAX_PLANETS, PLANET_CHANNELS), dtype=torch.float32),
        orbiting_planets=torch.zeros((1, MAX_PLANETS), dtype=torch.bool),
        fleets=torch.zeros((1, max_fleets, FLEET_CHANNELS), dtype=torch.float32),
        comets=torch.zeros((1, MAX_COMETS, COMET_CHANNELS), dtype=torch.float32),
        entity_mask=torch.zeros(
            (1, ACTION_ENTITY_SLOTS + max_fleets), dtype=torch.bool
        ),
        still_playing=torch.zeros((1, 4), dtype=torch.bool),
        global_features=torch.zeros((1, GLOBAL_CHANNELS), dtype=torch.float32),
        action_mask=PureActionMask(
            can_act=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
            max_launch=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64),
        ),
    )


def _raw_observation(*, remaining_overage_time: float = 60.0) -> dict[str, object]:
    planet = [0, 0, 25.0, 50.0, 2.0, 10, 3]
    return {
        "remainingOverageTime": remaining_overage_time,
        "step": 0,
        "planets": [planet],
        "initial_planets": [planet],
        "fleets": [],
        "player": 0,
        "angular_velocity": 0.025,
        "comet_planet_ids": [],
        "next_fleet_id": 0,
        "comets": [],
    }


@pytest.mark.parametrize(
    ("agent_min_fleet_size", "expected_filter_min_size", "expected_filtered_fleets"),
    [
        ("match", 6, 1),
        (8, 8, 2),
    ],
)
def test_agent_act_logs_rust_filtered_fleet_count(
    agent_min_fleet_size: object,
    expected_filter_min_size: int,
    expected_filtered_fleets: int,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1, min_fleet_size=6)
    agent = Agent.__new__(Agent)
    agent.checkpoint_config = SimpleNamespace(
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(),
            action_spec=action_spec,
        ),
        model=StatelessTransformerV1Config(),
    )
    agent.config = AgentConfig(
        deterministic=False,
        min_fleet_size=agent_min_fleet_size,
    )
    agent.device = torch.device("cpu")
    agent.hidden_state = None
    agent.fallback_model = None
    agent.fallback_checkpoint_config = None
    agent._peak_total_ms = 0
    agent._peak_entities = 0

    def fake_encode_python_observation_with_metrics(
        obs: dict[str, object],
        *,
        obs_spec: EntityBasedConfig,
        action_spec: ActionPureConfig,
        fleet_filter_min_size: int,
    ) -> EncodedPythonObservation:
        assert obs_spec == agent.checkpoint_config.env.obs_spec
        assert action_spec == agent.checkpoint_config.env.action_spec
        assert fleet_filter_min_size == expected_filter_min_size
        assert [fleet[FLEET_ID_INDEX] for fleet in obs["fleets"]] == [10, 11, 12]
        batch = _obs_batch(max_fleets=0)
        batch.entity_mask[0, 0] = True
        return EncodedPythonObservation(
            obs=batch,
            filtered_fleets=expected_filtered_fleets,
        )

    class FakeModel:
        def initial_hidden_state(
            self,
            _batch_size: int,
            *,
            device: torch.device,  # noqa: ARG002
        ) -> None:
            return None

        def serve(
            self,
            obs: ObsBatch,
            *,
            deterministic: bool,
            hidden_state: object | None,
        ) -> object:
            assert not deterministic
            assert hidden_state is None
            action_shape = (
                1,
                4,
                obs.action_mask.can_act.shape[2],
                action_spec.max_per_planet_launches,
            )
            return SimpleNamespace(
                actions=PureActions(
                    launch=torch.zeros(action_shape, dtype=torch.bool),
                    angle=torch.zeros(action_shape, dtype=torch.float32),
                    ships=torch.zeros(action_shape, dtype=torch.int64),
                ),
                values=torch.tensor([[0.25, -0.5, 0.0, 0.75]]),
                next_hidden_state=None,
            )

    def fake_actions_to_kaggle(
        obs: dict[str, object],
        player: int,
        actions: PureActions,
        *,
        action_spec: ActionPureConfig,
    ) -> list[list[float]]:
        assert [fleet[FLEET_ID_INDEX] for fleet in obs["fleets"]] == [10, 11, 12]
        assert player == 0
        assert actions.launch.shape == (
            1,
            4,
            ACTION_ENTITY_SLOTS,
            action_spec.max_per_planet_launches,
        )
        return []

    agent.model = FakeModel()
    monkeypatch.setattr(
        "owl.agent.agent.encode_python_observation_with_metrics",
        fake_encode_python_observation_with_metrics,
    )
    monkeypatch.setattr("owl.agent.agent.actions_to_kaggle", fake_actions_to_kaggle)
    raw_observation = _raw_observation()
    raw_observation["planets"] = [
        [0, 0, 25.0, 50.0, 2.0, 10, 3],
        [1, 1, 75.0, 50.0, 2.0, 10, 3],
    ]
    raw_observation["fleets"] = [
        [10, 0, 10.0, 10.0, 0.0, 9, 1],
        [11, 1, 20.0, 20.0, 0.0, 0, 6],
        [12, 2, 30.0, 30.0, 0.0, 1, 8],
    ]

    actions = agent.act(KaggleObservation.model_validate(raw_observation))

    assert actions == []
    assert f"filtered_fleets={expected_filtered_fleets}" in capsys.readouterr().out


def test_agent_act_converts_fake_model_output_to_kaggle_actions() -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    agent = Agent.__new__(Agent)
    agent.checkpoint_config = SimpleNamespace(
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(),
            action_spec=action_spec,
        ),
        model=StatelessTransformerV1Config(),
    )
    agent.config = AgentConfig(deterministic=False)
    agent.device = torch.device("cpu")
    agent.hidden_state = None
    agent.fallback_model = None
    agent.fallback_checkpoint_config = None
    agent._peak_total_ms = 0
    agent._peak_entities = 0
    action_shape = (1, 4, 1, action_spec.max_per_planet_launches)
    launch = torch.zeros(action_shape, dtype=torch.bool)
    angle = torch.zeros(action_shape, dtype=torch.float32)
    ships = torch.zeros(action_shape, dtype=torch.int64)
    launch[0, 0, 0, 0] = True
    angle[0, 0, 0, 0] = 0.5
    ships[0, 0, 0, 0] = action_spec.min_fleet_size

    class FakeModel:
        def initial_hidden_state(
            self,
            _batch_size: int,
            *,
            device: torch.device,  # noqa: ARG002
        ) -> None:
            return None

        def serve(
            self,
            obs: object,
            *,
            deterministic: bool,
            hidden_state: object | None,
        ) -> object:
            assert not deterministic
            assert hidden_state is None
            assert obs.still_playing.tolist() == [[True, False, False, False]]
            assert obs.planets.shape == (1, 1, PLANET_CHANNELS)
            assert obs.entity_mask.shape == (1, 1)
            assert obs.action_mask.can_act.shape == (1, 4, 1)
            return SimpleNamespace(
                actions=PureActions(launch=launch, angle=angle, ships=ships),
                values=torch.tensor([[0.25, -0.5, 0.0, 0.75]]),
                next_hidden_state=None,
            )

    agent.model = FakeModel()

    actions = agent.act(KaggleObservation.model_validate(_raw_observation()))

    assert actions == [[0.0, 0.5, float(action_spec.min_fleet_size)]]


def test_agent_act_preserves_planet_slots_for_recurrent_include_planets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_spec = ActionDiscreteTargetsConfig(min_fleet_size=1)
    agent = Agent.__new__(Agent)
    agent.checkpoint_config = SimpleNamespace(
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(),
            action_spec=action_spec,
        ),
        model=RecurrentTransformerV1Config(
            embed_dim=8,
            depth=1,
            n_heads=1,
            recurrence_mode="include_planets",
        ),
    )
    agent.config = AgentConfig(deterministic=True)
    agent.device = torch.device("cpu")
    agent.hidden_state = None
    agent.fallback_model = None
    agent.fallback_checkpoint_config = None
    agent._peak_total_ms = 0
    agent._peak_entities = 0

    class FakeModel:
        def initial_hidden_state(
            self,
            _batch_size: int,
            *,
            device: torch.device,  # noqa: ARG002
        ) -> str:
            return "hidden"

        def serve(
            self,
            obs: ObsBatch,
            *,
            deterministic: bool,
            hidden_state: object | None,
        ) -> object:
            assert deterministic
            assert hidden_state == "hidden"
            assert obs.planets.shape == (1, MAX_PLANETS, PLANET_CHANNELS)
            assert obs.entity_mask.shape == (1, MAX_PLANETS)
            assert obs.entity_mask[0, 0]
            assert not obs.entity_mask[0, 1:].any()
            assert isinstance(obs.action_mask, DiscreteTargetActionMask)
            assert obs.action_mask.can_act.shape == (
                1,
                4,
                MAX_PLANETS,
                MAX_PLANETS,
            )
            action_shape = (1, 4, MAX_PLANETS, 1)
            return SimpleNamespace(
                actions=DiscreteTargetActions(
                    launch=torch.zeros(action_shape, dtype=torch.bool),
                    target=torch.zeros(action_shape, dtype=torch.int64),
                    ships=torch.zeros(action_shape, dtype=torch.int64),
                ),
                values=torch.tensor([[0.25, -0.5, 0.0, 0.75]]),
                next_hidden_state="next",
            )

    def fake_actions_to_kaggle(
        obs: dict[str, object],
        player: int,
        actions: DiscreteTargetActions,
        *,
        action_spec: ActionDiscreteTargetsConfig,
    ) -> list[list[float]]:
        assert obs["player"] == 0
        assert player == 0
        assert action_spec == agent.checkpoint_config.env.action_spec
        assert actions.launch.shape == (1, 4, ACTION_ENTITY_SLOTS, 1)
        assert actions.target.shape == (1, 4, ACTION_ENTITY_SLOTS, 1)
        assert actions.ships.shape == (1, 4, ACTION_ENTITY_SLOTS, 1)
        return []

    agent.model = FakeModel()
    monkeypatch.setattr("owl.agent.agent.actions_to_kaggle", fake_actions_to_kaggle)

    actions = agent.act(KaggleObservation.model_validate(_raw_observation()))

    assert actions == []
    assert agent.hidden_state == "next"


def test_agent_act_moves_action_bundle_to_cpu_before_kaggle_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    agent = Agent.__new__(Agent)
    agent.checkpoint_config = SimpleNamespace(
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(),
            action_spec=action_spec,
        ),
        model=StatelessTransformerV1Config(),
    )
    agent.config = AgentConfig(deterministic=False)
    agent.device = torch.device("cpu")
    agent.hidden_state = None
    agent.fallback_model = None
    agent.fallback_checkpoint_config = None
    agent._peak_total_ms = 0
    agent._peak_entities = 0

    class FakeModel:
        def initial_hidden_state(
            self,
            _batch_size: int,
            *,
            device: torch.device,  # noqa: ARG002
        ) -> None:
            return None

        def serve(
            self,
            obs: object,
            *,
            deterministic: bool,
            hidden_state: object | None,
        ) -> object:
            assert not deterministic
            assert hidden_state is None
            action_shape = (1, 4, obs.action_mask.can_act.shape[2], 1)
            return SimpleNamespace(
                actions=PureActions(
                    launch=torch.zeros(action_shape, dtype=torch.bool),
                    angle=torch.zeros(action_shape, dtype=torch.float32),
                    ships=torch.zeros(action_shape, dtype=torch.int64),
                ),
                values=torch.tensor([[0.25, -0.5, 0.0, 0.75]]),
                next_hidden_state=None,
            )

    def fake_actions_to_kaggle(
        obs: dict[str, object],
        player: int,
        actions: PureActions,
        *,
        action_spec: ActionPureConfig,
    ) -> list[list[float]]:
        assert obs["player"] == 0
        assert player == 0
        assert action_spec == agent.checkpoint_config.env.action_spec
        assert actions.launch.device.type == "cpu"
        assert actions.angle.device.type == "cpu"
        assert actions.ships.device.type == "cpu"
        assert actions.launch.shape == (
            1,
            4,
            ACTION_ENTITY_SLOTS,
            action_spec.max_per_planet_launches,
        )
        return []

    agent.model = FakeModel()
    monkeypatch.setattr("owl.agent.agent.actions_to_kaggle", fake_actions_to_kaggle)

    actions = agent.act(KaggleObservation.model_validate(_raw_observation()))

    assert actions == []


def test_agent_act_uses_fallback_model_below_configured_overage() -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    agent = Agent.__new__(Agent)
    checkpoint_config = SimpleNamespace(
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(),
            action_spec=action_spec,
        ),
        model=StatelessTransformerV1Config(),
    )
    agent.checkpoint_config = checkpoint_config
    agent.fallback_checkpoint_config = checkpoint_config
    agent.config = AgentConfig(
        deterministic=True,
        min_overage_time=1.0,
        fallback_min_overage_time=10.0,
    )
    agent.device = torch.device("cpu")
    agent.hidden_state = "primary-hidden"
    agent._last_turn_value = float("nan")
    agent._peak_total_ms = 0
    agent._peak_entities = 0

    class PrimaryModel:
        def initial_hidden_state(
            self,
            _batch_size: int,
            *,
            device: torch.device,  # noqa: ARG002
        ) -> None:
            raise AssertionError("primary model should not be used")

        def serve(
            self,
            obs: object,  # noqa: ARG002
            *,
            deterministic: bool,  # noqa: ARG002
            hidden_state: object | None,  # noqa: ARG002
        ) -> object:
            raise AssertionError("primary model should not be used")

    class FallbackModel:
        def initial_hidden_state(
            self,
            _batch_size: int,
            *,
            device: torch.device,  # noqa: ARG002
        ) -> str:
            return "fallback-hidden"

        def serve(
            self,
            obs: object,
            *,
            deterministic: bool,
            hidden_state: object | None,
        ) -> object:
            assert deterministic
            assert hidden_state is None
            action_shape = (
                1,
                4,
                obs.action_mask.can_act.shape[2],
                action_spec.max_per_planet_launches,
            )
            return SimpleNamespace(
                actions=PureActions(
                    launch=torch.zeros(action_shape, dtype=torch.bool),
                    angle=torch.zeros(action_shape, dtype=torch.float32),
                    ships=torch.zeros(action_shape, dtype=torch.int64),
                ),
                values=torch.tensor([[0.25, -0.5, 0.0, 0.75]]),
                next_hidden_state=None,
            )

    agent.model = PrimaryModel()
    agent.fallback_model = FallbackModel()

    actions = agent.act(
        KaggleObservation.model_validate(_raw_observation(remaining_overage_time=5.0))
    )

    assert actions == []
    assert agent.hidden_state == "primary-hidden"


def test_agent_log_prints_one_line_with_metrics(capsys) -> None:
    agent = Agent.__new__(Agent)

    agent.log(
        step=7,
        total_ms=10,
        peak_total_ms=20,
        encode_ms=2,
        inference_ms=7,
        conversion_ms=1,
        self_value=0.25,
        advantage=-0.5,
        player_values=[0.25, -0.5, 0.0, 0.75],
        entity_count=3,
        peak_entities=5,
        filtered_fleets=2,
        remaining_overage_time=59.5,
        fallback_triggered=False,
    )

    assert "\n" not in capsys.readouterr().out.rstrip()


def test_agent_log_prefixes_fallback_trigger(capsys) -> None:
    agent = Agent.__new__(Agent)

    agent.log(
        step=7,
        total_ms=10,
        peak_total_ms=20,
        encode_ms=2,
        inference_ms=7,
        conversion_ms=1,
        self_value=0.25,
        advantage=-0.5,
        player_values=[0.25, -0.5, 0.0, 0.75],
        entity_count=3,
        peak_entities=5,
        filtered_fleets=2,
        remaining_overage_time=4.5,
        fallback_triggered=True,
    )

    assert capsys.readouterr().out.startswith("fallback triggered - step=7 - ")


def test_agent_peak_metrics_exclude_first_step_total_time() -> None:
    agent = Agent.__new__(Agent)
    agent._peak_total_ms = 0
    agent._peak_entities = 0

    assert agent._update_peak_metrics(step=0, total_ms=100, entity_count=3) == (0, 3)
    assert agent._update_peak_metrics(step=1, total_ms=10, entity_count=5) == (10, 5)
    assert agent._update_peak_metrics(step=2, total_ms=7, entity_count=4) == (10, 5)
    assert agent._update_peak_metrics(step=3, total_ms=11, entity_count=2) == (11, 5)


def test_agent_act_logs_model_values_and_entity_count(capsys) -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    agent = Agent.__new__(Agent)
    agent.checkpoint_config = SimpleNamespace(
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(),
            action_spec=action_spec,
        ),
        model=StatelessTransformerV1Config(),
    )
    agent.config = AgentConfig(deterministic=False)
    agent.device = torch.device("cpu")
    agent.hidden_state = None
    agent.fallback_model = None
    agent.fallback_checkpoint_config = None
    agent._peak_total_ms = 0
    agent._peak_entities = 0
    action_shape = (1, 4, 1, action_spec.max_per_planet_launches)

    class FakeModel:
        def initial_hidden_state(
            self,
            _batch_size: int,
            *,
            device: torch.device,  # noqa: ARG002
        ) -> None:
            return None

        def serve(
            self,
            obs: object,
            *,
            deterministic: bool,
            hidden_state: object | None,
        ) -> object:
            assert not deterministic
            assert hidden_state is None
            assert obs.entity_mask.sum().item() == 1
            assert obs.entity_mask.shape == (1, 1)
            assert obs.fleets.shape[1] == 0
            return SimpleNamespace(
                actions=PureActions(
                    launch=torch.zeros(action_shape, dtype=torch.bool),
                    angle=torch.zeros(action_shape, dtype=torch.float32),
                    ships=torch.zeros(action_shape, dtype=torch.int64),
                ),
                values=torch.tensor([[0.25, -0.5, 0.0, 0.75]]),
                next_hidden_state=None,
            )

    agent.model = FakeModel()

    agent.act(KaggleObservation.model_validate(_raw_observation()))

    log_line = capsys.readouterr().out
    assert re.fullmatch(
        r"step=0 - total_ms=\d+ - peak_total_ms=0 - "
        r"encode_ms=\d+ - inference_ms=\d+ - "
        r"conversion_ms=\d+ - value_self=0\.250 - advantage=0\.250 - "
        r"values=\[0\.250,-0\.500,0\.000,0\.750\] - "
        r"entities=1 - "
        r"peak_entities=1 - filtered_fleets=0 - remaining_overage_s=60\.0\n",
        log_line,
    )


def test_agent_act_logs_step_and_value_advantage(capsys) -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    agent = Agent.__new__(Agent)
    agent.checkpoint_config = SimpleNamespace(
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(),
            action_spec=action_spec,
        ),
        model=StatelessTransformerV1Config(),
    )
    agent.config = AgentConfig(deterministic=False)
    agent.device = torch.device("cpu")
    agent._last_turn_value = float("nan")
    agent.hidden_state = None
    agent.fallback_model = None
    agent.fallback_checkpoint_config = None
    agent._peak_total_ms = 0
    agent._peak_entities = 0
    values = iter(
        [
            torch.tensor([[0.25, -0.5, 0.0, 0.75]]),
            torch.tensor([[0.10, -0.5, 0.0, 0.75]]),
        ]
    )

    class FakeModel:
        def initial_hidden_state(
            self,
            _batch_size: int,
            *,
            device: torch.device,  # noqa: ARG002
        ) -> None:
            return None

        def serve(
            self,
            obs: object,
            *,
            deterministic: bool,
            hidden_state: object | None,
        ) -> object:
            assert not deterministic
            assert hidden_state is None
            action_shape = (
                1,
                4,
                obs.action_mask.can_act.shape[2],
                action_spec.max_per_planet_launches,
            )
            return SimpleNamespace(
                actions=PureActions(
                    launch=torch.zeros(action_shape, dtype=torch.bool),
                    angle=torch.zeros(action_shape, dtype=torch.float32),
                    ships=torch.zeros(action_shape, dtype=torch.int64),
                ),
                values=next(values),
                next_hidden_state=None,
            )

    agent.model = FakeModel()

    first_observation = _raw_observation()
    first_observation["step"] = 0
    first_observation["planets"] = [
        [0, 0, 25.0, 50.0, 2.0, 10, 3],
        [1, 1, 50.0, 50.0, 2.0, 10, 3],
        [2, 2, 75.0, 50.0, 2.0, 10, 3],
        [3, 3, 90.0, 50.0, 2.0, 10, 3],
    ]
    first_observation["initial_planets"] = first_observation["planets"]
    agent.act(KaggleObservation.model_validate(first_observation))
    second_observation = _raw_observation()
    second_observation["step"] = 5
    second_observation["planets"] = first_observation["planets"]
    second_observation["initial_planets"] = first_observation["planets"]
    agent.act(KaggleObservation.model_validate(second_observation))

    log_lines = capsys.readouterr().out.splitlines()
    assert len(log_lines) == 2
    assert "step=0" in log_lines[0]
    assert "advantage=0.750" in log_lines[0]
    assert "step=5" in log_lines[1]
    assert "advantage=-0.150" in log_lines[1]
