import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from owl.agent import Agent, KaggleObservation
from owl.agent.agent import AgentCheckpointConfig
from owl.model import ModelActions
from owl.rl import ACTION_ENTITY_SLOTS, ActionPureConfig, EntityBasedConfig
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


def test_agent_checkpoint_config_fields_exist_on_full_config() -> None:
    assert set(AgentCheckpointConfig.model_fields) <= set(FullConfig.model_fields)


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


def test_agent_act_converts_fake_model_output_to_kaggle_actions() -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    agent = Agent.__new__(Agent)
    agent.config = SimpleNamespace(
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(),
            action_spec=action_spec,
        )
    )
    agent.agent_config = SimpleNamespace(deterministic=False)
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
    agent.config = SimpleNamespace(
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(),
            action_spec=action_spec,
        )
    )
    agent.agent_config = SimpleNamespace(deterministic=False)
    agent.device = torch.device("cpu")
    action_shape = (1, 4, ACTION_ENTITY_SLOTS, action_spec.max_per_planet_launches)

    class FakeModel:
        def __call__(self, obs: object, *, deterministic: bool) -> object:
            assert not deterministic
            assert obs.entity_mask.sum().item() == 1
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
