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
    ActionPureConfig,
    ObsV1Config,
    VectorizedEnv,
    encode_python_observation,
)


def test_vectorized_env_writes_into_preallocated_torch_buffers() -> None:
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=ObsV1Config(),
        action_spec=ActionPureConfig(),
        pin_memory=False,
    )
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


def test_vectorized_env_warns_and_disables_pin_memory_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.warns(RuntimeWarning, match="proceeding without pinned memory"):
        env = VectorizedEnv(
            n_envs=1,
            obs_spec=ObsV1Config(),
            action_spec=ActionPureConfig(),
        )

    assert not env.observations.planets.is_pinned()
    assert not env.rewards.is_pinned()


def test_step_writes_observations_rewards_and_dones_in_place() -> None:
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=ObsV1Config(),
        action_spec=ActionPureConfig(),
        two_player_weight=0.0,
        pin_memory=False,
    )
    reward_ptr = env.rewards.data_ptr()
    done_ptr = env.dones.data_ptr()
    action_shape = (
        2,
        4,
        ACTION_ENTITY_SLOTS,
        env.action_spec.max_per_planet_launches,
    )
    launch = np.zeros(action_shape, dtype=np.bool_)
    angle = np.zeros(action_shape, dtype=np.float32)
    ships = np.zeros(action_shape, dtype=np.int64)

    env.rewards.fill_(123)
    env.dones.fill_(True)
    obs, rewards, dones = env.step(launch, angle, ships)

    assert rewards.data_ptr() == reward_ptr
    assert dones.data_ptr() == done_ptr
    assert obs.planets.shape == (2, MAX_PLANETS, PLANET_CHANNELS)
    assert obs.fleets.shape == (2, env.obs_spec.max_fleets, FLEET_CHANNELS)
    assert obs.comets.shape == (2, MAX_COMETS, COMET_CHANNELS)
    assert obs.global_features.shape == (2, GLOBAL_CHANNELS)
    assert obs.still_playing.shape == (2, 4)
    assert obs.can_act.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert obs.max_launch.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert rewards.shape == (2, 4)
    assert dones.shape == (2, 4)
    assert torch.equal(rewards, torch.zeros_like(rewards))
    assert torch.equal(dones, torch.zeros_like(dones))


@pytest.mark.parametrize(
    ("launch_dtype", "angle_dtype", "ships_dtype", "message"),
    [
        (np.float32, np.float32, np.int64, "launch must have dtype bool"),
        (np.bool_, np.float64, np.int64, "angle must have dtype float32"),
        (np.bool_, np.float32, np.float32, "ships must have dtype int64"),
    ],
)
def test_step_rejects_wrong_numpy_action_dtypes(
    launch_dtype: np.dtype,
    angle_dtype: np.dtype,
    ships_dtype: np.dtype,
    message: str,
) -> None:
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=ObsV1Config(),
        action_spec=ActionPureConfig(),
        pin_memory=False,
    )
    shape = (1, 4, ACTION_ENTITY_SLOTS, env.action_spec.max_per_planet_launches)
    launch = np.zeros(shape, dtype=launch_dtype)
    angle = np.zeros(shape, dtype=angle_dtype)
    ships = np.zeros(shape, dtype=ships_dtype)

    with pytest.raises(ValueError, match=message):
        env.step(launch, angle, ships)


@pytest.mark.parametrize(
    ("launch_dtype", "angle_dtype", "ships_dtype", "message"),
    [
        (
            torch.float32,
            torch.float32,
            torch.int64,
            "launch must have dtype torch.bool",
        ),
        (torch.bool, torch.float64, torch.int64, "angle must have dtype torch.float32"),
        (torch.bool, torch.float32, torch.float32, "ships must have dtype torch.int64"),
    ],
)
def test_step_rejects_wrong_torch_action_dtypes(
    launch_dtype: torch.dtype,
    angle_dtype: torch.dtype,
    ships_dtype: torch.dtype,
    message: str,
) -> None:
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=ObsV1Config(),
        action_spec=ActionPureConfig(),
        pin_memory=False,
    )
    shape = (1, 4, ACTION_ENTITY_SLOTS, env.action_spec.max_per_planet_launches)
    launch = torch.zeros(shape, dtype=launch_dtype)
    angle = torch.zeros(shape, dtype=angle_dtype)
    ships = torch.zeros(shape, dtype=ships_dtype)

    with pytest.raises(ValueError, match=message):
        env.step(launch, angle, ships)


@pytest.mark.parametrize(
    ("ship_count", "angle_value", "message"),
    [
        (0, 0.0, "ships must be >= 1"),
        (1, np.inf, "angle must be finite"),
    ],
)
def test_step_rejects_invalid_launched_action_values(
    ship_count: int, angle_value: float, message: str
) -> None:
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=ObsV1Config(),
        action_spec=ActionPureConfig(),
        pin_memory=False,
    )
    obs = env.reset()
    env_index, player, entity = torch.nonzero(obs.can_act, as_tuple=False)[0].tolist()
    shape = (1, 4, ACTION_ENTITY_SLOTS, env.action_spec.max_per_planet_launches)
    launch = np.zeros(shape, dtype=np.bool_)
    angle = np.zeros(shape, dtype=np.float32)
    ships = np.zeros(shape, dtype=np.int64)
    launch[env_index, player, entity, 0] = True
    angle[env_index, player, entity, 0] = angle_value
    ships[env_index, player, entity, 0] = ship_count

    with pytest.raises(ValueError, match=message):
        env.step(launch, angle, ships)


def test_reset_writes_still_playing_from_rust_env_state() -> None:
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=ObsV1Config(),
        action_spec=ActionPureConfig(),
        two_player_weight=1.0,
        pin_memory=False,
    )
    still_playing_ptr = env.observations.still_playing.data_ptr()

    env.observations.still_playing.fill_(True)
    obs = env.reset()

    assert obs.still_playing.data_ptr() == still_playing_ptr
    assert np.shares_memory(obs.still_playing.numpy(), env._still_playing_np)
    assert torch.equal(obs.still_playing.sum(dim=1), torch.full((2,), 2))


def test_two_player_resets_randomize_active_outer_player_slots() -> None:
    env = VectorizedEnv(
        n_envs=32,
        obs_spec=ObsV1Config(),
        action_spec=ActionPureConfig(),
        two_player_weight=1.0,
        pin_memory=False,
    )

    obs = env.reset()

    assert torch.equal(obs.still_playing.sum(dim=1), torch.full((32,), 2))
    assert torch.all(obs.still_playing.any(dim=0))


def test_two_player_sample_marks_unused_player_slots_done() -> None:
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=ObsV1Config(),
        action_spec=ActionPureConfig(),
        two_player_weight=1.0,
        pin_memory=False,
    )
    action_shape = (
        2,
        4,
        ACTION_ENTITY_SLOTS,
        env.action_spec.max_per_planet_launches,
    )
    launch = np.zeros(action_shape, dtype=np.bool_)
    angle = np.zeros(action_shape, dtype=np.float32)
    ships = np.zeros(action_shape, dtype=np.int64)

    _, rewards, dones = env.step(launch, angle, ships)

    assert torch.equal(rewards, torch.zeros_like(rewards))
    assert torch.equal(dones.sum(dim=1), torch.full((2,), 2))
    assert torch.equal(env.observations.still_playing, ~dones)


def test_vectorized_env_accepts_discriminated_config_dicts() -> None:
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=ObsV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 1),
        action_spec=ActionPureConfig(max_per_planet_launches=2),
        pin_memory=False,
    )

    assert env.obs_spec.max_fleets == 1
    assert env.action_spec.action_spec == "pure"
    assert env.action_spec.max_per_planet_launches == 2


def test_action_config_validates_max_per_planet_launches() -> None:
    assert ActionPureConfig().max_per_planet_launches == 3
    assert ActionPureConfig(max_per_planet_launches=4).max_per_planet_launches == 4

    with pytest.raises(ValueError, match="less than or equal to 4"):
        ActionPureConfig(max_per_planet_launches=5)


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
    assert fleets.shape == (ObsV1Config().max_fleets, FLEET_CHANNELS)
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
    assert planets[0, 12] == pytest.approx(2 / 3)
    assert fleets[0, 0] == 1
    assert fleets[0, 4] == pytest.approx(-0.8)
    assert fleets[0, 5] == pytest.approx(-0.6)
    normalized_speed = (1.0 + 5.0 * (np.log(25) / np.log(1000)) ** 1.5) / 6.0
    assert fleets[0, 6] == pytest.approx(0.0, abs=1e-7)
    assert fleets[0, 7] == pytest.approx(normalized_speed)
    assert global_features[0] == pytest.approx(0.1)
    assert global_features[1] == pytest.approx(1.0)
    assert global_features[2] == pytest.approx(1.0)
    assert not can_act.any()
    assert not max_launch.any()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("angular_velocity", np.nan, "angular_velocity must be finite"),
        ("angular_velocity", np.inf, "angular_velocity must be finite"),
        ("episode_steps", 0, "episode_steps must be > 0"),
    ],
)
def test_python_observation_encoder_rejects_invalid_globals(
    field: str, value: float | int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        encode_python_observation(
            {
                field: value,
                "planets": [],
                "fleets": [],
                "comets": [],
            }
        )


@pytest.mark.parametrize("production", [0, 6, -1])
def test_python_observation_encoder_rejects_invalid_planet_production(
    production: int,
) -> None:
    with pytest.raises(ValueError, match="planet production must be between 1 and 5"):
        encode_python_observation(
            {
                "planets": [[0, -1, 25.0, 75.0, 2.0, 50, production]],
                "fleets": [],
                "comets": [],
            }
        )


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
    assert fleets[0, 8] == pytest.approx(20 / 250)
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
    assert comets[0, 5] == pytest.approx(25 / 250)
    assert comets[0, 7] == pytest.approx(2 / MAX_COMET_PATH_LENGTH)
    assert comets[0, 8] == pytest.approx(0.0)
    assert comets[0, 9] == pytest.approx(0.0)
    assert comets[0, 10] == pytest.approx(1.0)
    assert comets[0, 11] == pytest.approx(1.0)
    assert np.all(comets[0, 12 : 8 + MAX_COMET_PATH_LENGTH * 2] == 0.0)
    assert can_act[2, MAX_PLANETS]
    assert max_launch[2, MAX_PLANETS] == 25
