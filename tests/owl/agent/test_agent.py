from types import SimpleNamespace

import torch
from owl.agent import Agent, KaggleObservation
from owl.model import ModelActions
from owl.rl import ACTION_ENTITY_SLOTS, ActionPureConfig, EntityBasedConfig


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
            assert obs.still_playing.tolist() == [[True, True, True, True]]
            return SimpleNamespace(
                actions=ModelActions(launch=launch, angle=angle, ships=ships)
            )

    agent.model = FakeModel()

    actions = agent.act(KaggleObservation.model_validate(_raw_observation()))

    assert actions == [[0.0, 0.5, 1.0]]
