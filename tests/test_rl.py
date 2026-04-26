import numpy as np
import pytest
import torch
from owl.rl import (
    FLEET_CHANNELS,
    MAX_PLANETS,
    PLANET_CHANNELS,
    ObsV1Config,
    VectorizedEnv,
    encode_python_observation,
)


def test_vectorized_env_writes_into_preallocated_torch_buffers() -> None:
    env = VectorizedEnv(n_envs=2, n_players=2, pin_memory=False)
    planet_ptr = env.observations.planets.data_ptr()
    fleet_mask_ptr = env.observations.fleet_mask.data_ptr()

    env.observations.planets.fill_(-7)
    env.observations.fleet_mask.fill_(True)
    obs = env.reset()

    assert obs.planets.data_ptr() == planet_ptr
    assert obs.fleet_mask.data_ptr() == fleet_mask_ptr
    assert np.shares_memory(obs.planets.numpy(), env._planet_obs_np)
    assert torch.any(obs.planets != -7)
    assert torch.any(obs.planet_mask)
    assert torch.all(~obs.fleet_mask)


def test_step_writes_observations_rewards_and_dones_in_place() -> None:
    env = VectorizedEnv(n_envs=2, n_players=2, pin_memory=False)
    reward_ptr = env.rewards.data_ptr()
    done_ptr = env.dones.data_ptr()
    actions = np.zeros((2, 2, 0), dtype=np.float32)

    env.rewards.fill_(123)
    env.dones.fill_(True)
    obs, rewards, dones = env.step(actions)

    assert rewards.data_ptr() == reward_ptr
    assert dones.data_ptr() == done_ptr
    assert obs.planets.shape == (2, MAX_PLANETS, PLANET_CHANNELS)
    assert obs.fleets.shape == (2, env.obs_spec.max_fleets, FLEET_CHANNELS)
    assert torch.equal(rewards, torch.zeros_like(rewards))
    assert torch.equal(dones, torch.zeros_like(dones))


def test_python_observation_encoder_matches_rl_schema_and_masks() -> None:
    planets, fleets, planet_mask, fleet_mask = encode_python_observation(
        {
            "angular_velocity": 0.05,
            "planets": [[0, -1, 25.0, 75.0, 2.0, 50, 3]],
            "fleets": [[1, 0, 10.0, 20.0, np.pi / 2, 0, 25]],
        }
    )

    assert planets.shape == (MAX_PLANETS, PLANET_CHANNELS)
    assert fleets.shape == (MAX_PLANETS * 7, FLEET_CHANNELS)
    assert planet_mask[0]
    assert not planet_mask[1]
    assert fleet_mask[0]
    assert not fleet_mask[1]
    assert planets[0, 4] == 1
    assert planets[0, 5] == pytest.approx(0.25)
    assert planets[0, 9] == 1
    assert planets[0, 15] == pytest.approx(1.0)
    assert fleets[0, 0] == 1
    assert fleets[0, 6] == pytest.approx(1.0)
    assert fleets[0, 7] == pytest.approx(0.0, abs=1e-7)


def test_python_observation_encoder_keeps_largest_fleets_first() -> None:
    spec = ObsV1Config(max_entities=MAX_PLANETS + 1)

    with pytest.warns(UserWarning, match="max_entities exceeded: 1 fleets ignored"):
        _, fleets, _, fleet_mask = encode_python_observation(
            {
                "planets": [],
                "fleets": [
                    [1, 0, 10.0, 20.0, 0.0, 0, 5],
                    [2, 1, 30.0, 40.0, 0.0, 0, 20],
                ],
            },
            spec,
        )

    assert fleet_mask.tolist() == [True]
    assert fleets[0, 1] == 1
    assert fleets[0, 8] == pytest.approx(0.1)
