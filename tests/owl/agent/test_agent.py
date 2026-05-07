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
    compact_entities,
)
from owl.model import ModelActions, StatelessTransformerV1Config
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    COMET_CHANNELS,
    FLEET_CHANNELS,
    GLOBAL_CHANNELS,
    MAX_COMETS,
    MAX_PLANETS,
    PLANET_CHANNELS,
    ActionPureConfig,
    EntityBasedConfig,
    EnvConfig,
    ObsBatch,
)
from owl.train.config import FullConfig

_ASSERT_AGENT_IMPORT_ISOLATED = Path(__file__).with_name(
    "assert_agent_import_isolated.py"
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


def test_agent_checkpoint_config_fields_exist_on_full_config() -> None:
    assert set(AgentCheckpointConfig.model_fields) <= set(FullConfig.model_fields)


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


def test_compact_runtime_entities_keeps_action_slots_and_active_fleets() -> None:
    obs = _obs_batch(max_fleets=5)
    obs.fleets[0, 1, 0] = 2.0
    obs.fleets[0, 4, 0] = 5.0
    obs.entity_mask[0, 0] = True
    obs.entity_mask[0, ACTION_ENTITY_SLOTS + 1] = True
    obs.entity_mask[0, ACTION_ENTITY_SLOTS + 4] = True

    compacted = compact_entities(obs)

    assert compacted.entity_mask.shape == (1, ACTION_ENTITY_SLOTS + 2)
    assert compacted.fleets.shape == (1, 2, FLEET_CHANNELS)
    assert compacted.entity_mask[0, 0]
    assert compacted.entity_mask[0, ACTION_ENTITY_SLOTS:].tolist() == [True, True]
    assert compacted.fleets[0, :, 0].tolist() == [2.0, 5.0]


def test_compact_runtime_entities_allows_zero_fleets() -> None:
    obs = _obs_batch(max_fleets=5)
    obs.entity_mask[0, 0] = True

    compacted = compact_entities(obs)

    assert compacted.entity_mask.shape == (1, ACTION_ENTITY_SLOTS)
    assert compacted.fleets.shape == (1, 0, FLEET_CHANNELS)


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
        can_act=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
        max_launch=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64),
    )


def _raw_observation() -> dict[str, object]:
    planet = [0, 0, 25.0, 50.0, 2.0, 10, 3]
    return {
        "remainingOverageTime": 60.0,
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


@pytest.mark.xfail(reason="Debugging - not submitting real actions")
def test_agent_act_converts_fake_model_output_to_kaggle_actions() -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    agent = Agent.__new__(Agent)
    agent.checkpoint_config = SimpleNamespace(
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(),
            action_spec=action_spec,
        )
    )
    agent.config = AgentConfig(deterministic=False)
    agent.device = torch.device("cpu")
    action_shape = (1, 4, ACTION_ENTITY_SLOTS, action_spec.max_per_planet_launches)
    launch = torch.zeros(action_shape, dtype=torch.bool)
    angle = torch.zeros(action_shape, dtype=torch.float32)
    ships = torch.zeros(action_shape, dtype=torch.int64)
    launch[0, 0, 0, 0] = True
    angle[0, 0, 0, 0] = 0.5
    ships[0, 0, 0, 0] = 1

    class FakeModel:
        def __call__(self, obs: object, *, deterministic: bool) -> object:
            assert not deterministic
            assert obs.still_playing.tolist() == [[True, False, False, False]]
            return SimpleNamespace(
                actions=ModelActions(launch=launch, angle=angle, ships=ships),
                values=torch.tensor([[0.25, -0.5, 0.0, 0.75]]),
            )

    agent.model = FakeModel()

    actions = agent.act(KaggleObservation.model_validate(_raw_observation()))

    assert actions == [[0.0, 0.5, 1.0]]


def test_agent_log_prints_one_line_with_metrics(capsys) -> None:
    agent = Agent.__new__(Agent)

    agent.log(
        total_ms=10,
        encode_ms=2,
        inference_ms=7,
        conversion_ms=1,
        self_value=0.25,
        player_values=[0.25, -0.5, 0.0, 0.75],
        entity_count=3,
        remaining_overage_time=59.5,
    )

    assert "\n" not in capsys.readouterr().out.rstrip()


def test_agent_act_logs_model_values_and_entity_count(capsys) -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    agent = Agent.__new__(Agent)
    agent.checkpoint_config = SimpleNamespace(
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(),
            action_spec=action_spec,
        )
    )
    agent.config = AgentConfig(deterministic=False)
    agent.device = torch.device("cpu")
    action_shape = (1, 4, ACTION_ENTITY_SLOTS, action_spec.max_per_planet_launches)

    class FakeModel:
        def __call__(self, obs: object, *, deterministic: bool) -> object:
            assert not deterministic
            assert obs.entity_mask.sum().item() == 1
            assert obs.entity_mask.shape == (1, ACTION_ENTITY_SLOTS)
            assert obs.fleets.shape[1] == 0
            return SimpleNamespace(
                actions=ModelActions(
                    launch=torch.zeros(action_shape, dtype=torch.bool),
                    angle=torch.zeros(action_shape, dtype=torch.float32),
                    ships=torch.zeros(action_shape, dtype=torch.int64),
                ),
                values=torch.tensor([[0.25, -0.5, 0.0, 0.75]]),
            )

    agent.model = FakeModel()

    agent.act(KaggleObservation.model_validate(_raw_observation()))

    log_line = capsys.readouterr().out
    assert re.fullmatch(
        r"total_ms=\d+ - encode_ms=\d+ - inference_ms=\d+ - conversion_ms=\d+ - "
        r"value_self=0\.250 - values=\[0\.250,-0\.500,0\.000,0\.750\] - "
        r"entities=1 - remaining_overage_s=60\.0\n",
        log_line,
    )
