import numpy as np
import pytest
import torch
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    COMET_CHANNELS,
    FLEET_CHANNELS,
    GLOBAL_CHANNELS,
    MAX_COMET_PATH_LENGTH,
    MAX_COMETS,
    MAX_PLANETS,
    PLANET_CHANNELS,
    ObsV1Config,
    VectorizedEnv,
    encode_python_observation,
)


def test_vectorized_env_writes_into_preallocated_torch_buffers() -> None:
    env = VectorizedEnv(n_envs=2, pin_memory=False)
    planet_ptr = env.observations.planets.data_ptr()
    comet_ptr = env.observations.comets.data_ptr()

    env.observations.planets.fill_(-7)
    env.observations.comets.fill_(-3)
    env.observations.comet_mask.fill_(True)
    env.observations.can_act.fill_(True)
    env.observations.max_launch.fill_(123)
    obs = env.reset()

    assert obs.planets.data_ptr() == planet_ptr
    assert obs.comets.data_ptr() == comet_ptr
    assert np.shares_memory(obs.planets.numpy(), env._planet_obs_np)
    assert torch.any(obs.planets != -7)
    assert torch.any(obs.planet_mask)
    assert torch.all(obs.comets == 0)
    assert torch.all(~obs.comet_mask)
    assert torch.all(~obs.fleet_mask)
    assert torch.any(obs.can_act)
    assert torch.all(obs.max_launch[~obs.can_act] == 0)


def test_step_writes_observations_rewards_and_dones_in_place() -> None:
    env = VectorizedEnv(n_envs=2, two_player_weight=0.0, pin_memory=False)
    reward_ptr = env.rewards.data_ptr()
    done_ptr = env.dones.data_ptr()
    launch = np.zeros((2, 4, ACTION_ENTITY_SLOTS), dtype=np.bool_)
    angle = np.zeros((2, 4, ACTION_ENTITY_SLOTS), dtype=np.float32)
    ships = np.zeros((2, 4, ACTION_ENTITY_SLOTS), dtype=np.int64)

    env.rewards.fill_(123)
    env.dones.fill_(True)
    obs, rewards, dones = env.step(launch, angle, ships)

    assert rewards.data_ptr() == reward_ptr
    assert dones.data_ptr() == done_ptr
    assert obs.planets.shape == (2, MAX_PLANETS, PLANET_CHANNELS)
    assert obs.fleets.shape == (2, env.obs_spec.max_fleets, FLEET_CHANNELS)
    assert obs.comets.shape == (2, MAX_COMETS, COMET_CHANNELS)
    assert obs.global_features.shape == (2, GLOBAL_CHANNELS)
    assert obs.can_act.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert obs.max_launch.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert rewards.shape == (2, 4)
    assert dones.shape == (2, 4)
    assert torch.equal(rewards, torch.zeros_like(rewards))
    assert torch.equal(dones, torch.zeros_like(dones))


def test_two_player_sample_marks_unused_player_slots_done() -> None:
    env = VectorizedEnv(n_envs=2, two_player_weight=1.0, pin_memory=False)
    launch = np.zeros((2, 4, ACTION_ENTITY_SLOTS), dtype=np.bool_)
    angle = np.zeros((2, 4, ACTION_ENTITY_SLOTS), dtype=np.float32)
    ships = np.zeros((2, 4, ACTION_ENTITY_SLOTS), dtype=np.int64)

    _, rewards, dones = env.step(launch, angle, ships)

    assert torch.equal(rewards, torch.zeros_like(rewards))
    assert torch.equal(dones[:, :2], torch.zeros_like(dones[:, :2]))
    assert torch.equal(dones[:, 2:], torch.ones_like(dones[:, 2:]))


def test_vectorized_env_accepts_discriminated_config_dicts() -> None:
    env = VectorizedEnv(
        n_envs=1,
        obs_spec={"obs_spec": "obs_v1", "max_entities": MAX_PLANETS + MAX_COMETS + 1},
        action_spec={"action_spec": "pure"},
        pin_memory=False,
    )

    assert env.obs_spec.max_fleets == 1
    assert env.action_spec.action_spec == "pure"


def test_vectorized_env_rejects_empty_config_dicts() -> None:
    with pytest.raises(ValueError, match="obs_spec"):
        VectorizedEnv(n_envs=1, obs_spec={}, pin_memory=False)

    with pytest.raises(ValueError, match="action_spec"):
        VectorizedEnv(n_envs=1, action_spec={}, pin_memory=False)


def test_python_observation_encoder_matches_rl_schema_and_masks() -> None:
    (
        planets,
        fleets,
        comets,
        planet_mask,
        fleet_mask,
        comet_mask,
        global_features,
        can_act,
        max_launch,
    ) = encode_python_observation(
        {
            "step": 50,
            "angular_velocity": 0.05,
            "planets": [[0, -1, 25.0, 75.0, 2.0, 50, 3]],
            "fleets": [[1, 0, 10.0, 20.0, np.pi / 2, 0, 25]],
            "comets": [],
        }
    )

    assert planets.shape == (MAX_PLANETS, PLANET_CHANNELS)
    assert fleets.shape == (MAX_PLANETS * 7 - MAX_COMETS, FLEET_CHANNELS)
    assert comets.shape == (MAX_COMETS, COMET_CHANNELS)
    assert global_features.shape == (GLOBAL_CHANNELS,)
    assert can_act.shape == (4, ACTION_ENTITY_SLOTS)
    assert max_launch.shape == (4, ACTION_ENTITY_SLOTS)
    assert planet_mask[0]
    assert not planet_mask[1]
    assert fleet_mask[0]
    assert not fleet_mask[1]
    assert not comet_mask[0]
    assert planets[0, 4] == 1
    assert planets[0, 5] == pytest.approx(-0.5)
    assert planets[0, 6] == pytest.approx(0.5)
    assert planets[0, 9] == 1
    assert planets[0, 15] == pytest.approx(1.0)
    assert fleets[0, 0] == 1
    assert fleets[0, 4] == pytest.approx(-0.8)
    assert fleets[0, 5] == pytest.approx(-0.6)
    assert fleets[0, 6] == pytest.approx(1.0)
    assert fleets[0, 7] == pytest.approx(0.0, abs=1e-7)
    assert global_features[0] == pytest.approx(0.1)
    assert global_features[1] == pytest.approx(1.0)
    assert global_features[2] == pytest.approx(1.0)
    assert not can_act.any()
    assert not max_launch.any()


def test_python_observation_encoder_keeps_largest_fleets_first(
    capfd: pytest.CaptureFixture[str],
) -> None:
    spec = ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 1)

    _, fleets, _, _, fleet_mask, _, _, _, _ = encode_python_observation(
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
    assert "max_entities exceeded: 1 fleets ignored" in capfd.readouterr().err


def test_python_observation_encoder_writes_comet_future_paths() -> None:
    path = [[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]]
    planets, _, comets, planet_mask, _, comet_mask, _, can_act, max_launch = (
        encode_python_observation(
            {
                "planets": [[10, 2, 50.0, 50.0, 1.0, 25, 1]],
                "fleets": [],
                "comets": [
                    {
                        "planet_ids": [10],
                        "paths": [path],
                        "path_index": 1,
                    }
                ],
            }
        )
    )

    assert not planet_mask[0]
    assert comet_mask.tolist() == [True, False, False, False]
    assert planets[0].tolist() == [0.0] * PLANET_CHANNELS
    assert comets[0, 2] == 1.0
    assert comets[0, 5] == pytest.approx(25 / 200)
    assert comets[0, 7] == pytest.approx(0.0)
    assert comets[0, 8] == pytest.approx(0.0)
    assert comets[0, 9] == pytest.approx(1.0)
    assert comets[0, 10] == pytest.approx(1.0)
    assert np.all(comets[0, 11 : 7 + MAX_COMET_PATH_LENGTH * 2] == 0.0)
    assert can_act[2, MAX_PLANETS]
    assert max_launch[2, MAX_PLANETS] == 25
