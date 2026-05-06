import json

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
    ActionDiscreteTargetsConfig,
    ActionPureConfig,
    EntityBasedConfig,
    EnvConfig,
    VectorizedEnv,
    encode_entity_based,
    encode_python_observation,
)


def test_vectorized_env_writes_into_preallocated_torch_buffers() -> None:
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(),
        pin_memory=False,
    )
    planet_ptr = env.observations.planets.data_ptr()
    comet_ptr = env.observations.comets.data_ptr()

    env.observations.planets.fill_(-7)
    env.observations.comets.fill_(-3)
    env.observations.entity_mask.fill_(True)
    env.observations.can_act.fill_(True)
    env.observations.max_launch.fill_(123)
    obs = env.reset()

    assert obs.planets.data_ptr() == planet_ptr
    assert obs.comets.data_ptr() == comet_ptr
    assert np.shares_memory(obs.planets.numpy(), env._planet_obs_np)
    assert np.shares_memory(
        obs.orbiting_planets.numpy(),
        env._orbiting_planet_obs_np,
    )
    assert torch.any(obs.planets != -7)
    assert torch.any(obs.entity_mask[:, :MAX_PLANETS])
    assert torch.all(obs.comets == 0)
    assert torch.all(~obs.entity_mask[:, MAX_PLANETS:ACTION_ENTITY_SLOTS])
    assert torch.all(~obs.entity_mask[:, ACTION_ENTITY_SLOTS:])
    assert torch.any(obs.can_act)
    assert torch.all(obs.max_launch[~obs.can_act] == 0)


def test_vectorized_env_warns_and_disables_pin_memory_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.warns(RuntimeWarning, match="proceeding without pinned memory"):
        env = VectorizedEnv(
            n_envs=1,
            obs_spec=EntityBasedConfig(),
            action_spec=ActionPureConfig(),
        )

    assert not env.observations.planets.is_pinned()
    assert not env.rewards.is_pinned()


def test_step_writes_observations_rewards_and_dones_in_place() -> None:
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=EntityBasedConfig(),
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
    obs, rewards, dones, episode_metrics = env.step(launch, angle, ships)

    assert rewards.data_ptr() == reward_ptr
    assert dones.data_ptr() == done_ptr
    assert obs.planets.shape == (2, MAX_PLANETS, PLANET_CHANNELS)
    assert obs.orbiting_planets.shape == (2, MAX_PLANETS)
    assert obs.fleets.shape == (2, env.obs_spec.max_fleets, FLEET_CHANNELS)
    assert obs.comets.shape == (2, MAX_COMETS, COMET_CHANNELS)
    assert obs.global_features.shape == (2, GLOBAL_CHANNELS)
    assert obs.still_playing.shape == (2, 4)
    assert obs.can_act.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert obs.max_launch.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert rewards.shape == (2, 4)
    assert dones.shape == (2, 4)
    assert episode_metrics == {}
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
        obs_spec=EntityBasedConfig(),
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
        obs_spec=EntityBasedConfig(),
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
        obs_spec=EntityBasedConfig(),
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
        obs_spec=EntityBasedConfig(),
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
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(),
        two_player_weight=1.0,
        pin_memory=False,
    )

    obs = env.reset()

    assert torch.equal(obs.still_playing.sum(dim=1), torch.full((32,), 2))
    assert torch.all(obs.still_playing.any(dim=0))


def test_vectorized_env_state_snapshot_is_json_serializable() -> None:
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(),
        two_player_weight=1.0,
        pin_memory=False,
    )
    obs = env.reset()

    snapshot = env.state_snapshot(0)

    json.dumps(snapshot)
    assert snapshot["player_count"] == 2
    assert snapshot["owner_space"] == "outer"
    assert len(snapshot["player_map"]["internal_to_outer"]) == 4
    assert len(snapshot["player_finished"]) == 4
    assert len(snapshot["action_entity_slots"]) == ACTION_ENTITY_SLOTS
    assert len(snapshot["planets"]) > 0
    assert sum(obs.still_playing[0].tolist()) == 2


def test_vectorized_env_terminal_snapshot_preserves_pre_reset_state() -> None:
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(),
        two_player_weight=1.0,
        pin_memory=False,
    )
    env.reset()
    shape = (1, 4, ACTION_ENTITY_SLOTS, env.action_spec.max_per_planet_launches)
    launch = np.zeros(shape, dtype=np.bool_)
    angle = np.zeros(shape, dtype=np.float32)
    ships = np.zeros(shape, dtype=np.int64)

    terminal_snapshot = None
    for _ in range(600):
        obs, _rewards, dones, _episode_metrics = env.step(launch, angle, ships)
        if bool(dones.all()):
            terminal_snapshot = env.terminal_snapshot(0)
            break

    assert terminal_snapshot is not None
    terminal_metrics = env.terminal_metrics(0)
    assert terminal_metrics is not None
    assert terminal_snapshot["step"] > obs.global_features[0, 0].item()
    assert terminal_snapshot["player_count"] == 2


def test_two_player_sample_marks_unused_player_slots_done() -> None:
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=EntityBasedConfig(),
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

    _, rewards, dones, episode_metrics = env.step(launch, angle, ships)

    assert torch.equal(rewards, torch.zeros_like(rewards))
    assert episode_metrics == {}
    assert torch.equal(dones.sum(dim=1), torch.full((2,), 2))
    assert torch.equal(env.observations.still_playing, ~dones)


def test_vectorized_env_accepts_discriminated_config_dicts() -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=2, min_fleet_size=4)
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 1),
        action_spec=action_spec,
        pin_memory=False,
    )

    assert env.obs_spec.max_fleets == 1
    assert env.action_spec.action_spec == "pure"
    assert env.action_spec.max_per_planet_launches == 2
    assert env.action_spec.min_fleet_size == 4


def test_action_config_validates_launch_bounds() -> None:
    assert ActionPureConfig().max_per_planet_launches == 3
    assert ActionPureConfig(max_per_planet_launches=4).max_per_planet_launches == 4
    assert ActionPureConfig(min_fleet_size=5).min_fleet_size == 5

    with pytest.raises(ValueError, match="less than or equal to 4"):
        ActionPureConfig(max_per_planet_launches=5)
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        ActionPureConfig(min_fleet_size=0)


def test_env_config_requires_even_env_count() -> None:
    assert EnvConfig().n_envs == 2

    with pytest.raises(ValueError, match="n_envs must be even"):
        EnvConfig(n_envs=1)


def test_discrete_targets_config_and_env_shapes() -> None:
    config = EnvConfig.model_validate(
        {
            "n_envs": 2,
            "action_spec": {
                "action_spec": "discrete_targets",
                "max_per_planet_launches": 2,
                "min_fleet_size": 4,
            },
            "pin_memory": False,
        }
    )
    assert isinstance(config.action_spec, ActionDiscreteTargetsConfig)
    assert config.action_spec.max_per_planet_launches == 2
    assert config.action_spec.min_fleet_size == 4

    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(),
        action_spec=config.action_spec,
        pin_memory=False,
    )
    obs = env.reset()

    assert obs.can_act.shape == (1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS)
    assert obs.max_launch.shape == (1, 4, ACTION_ENTITY_SLOTS)
    assert obs.max_launch[~obs.can_act.any(dim=-1)].eq(0).all()


def test_discrete_targets_step_uses_int_target_tensor() -> None:
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=2)
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(),
        action_spec=action_spec,
        pin_memory=False,
    )
    shape = (1, 4, ACTION_ENTITY_SLOTS, action_spec.max_per_planet_launches)
    launch = np.zeros(shape, dtype=np.bool_)
    target = np.zeros(shape, dtype=np.int64)
    ships = np.zeros(shape, dtype=np.int64)

    obs, rewards, dones, episode_metrics = env.step(launch, target, ships)

    assert obs.can_act.shape == (1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS)
    assert rewards.shape == (1, 4)
    assert dones.shape == (1, 4)
    assert episode_metrics == {}

    with pytest.raises(ValueError, match="target must have dtype int64"):
        env.step(launch, target.astype(np.float32), ships)


def test_min_fleet_size_controls_action_mask_and_validation() -> None:
    action_spec = ActionPureConfig(min_fleet_size=3)
    (
        _planets,
        _orbiting_planets,
        _fleets,
        _comets,
        _entity_mask,
        _global_features,
        can_act,
        max_launch,
    ) = encode_python_observation(
        {
            "step": 0,
            "angular_velocity": 0.025,
            "planets": [[0, 0, 25.0, 75.0, 2.0, 2, 3]],
            "fleets": [],
            "comets": [],
        },
        action_spec=action_spec,
    )

    assert not can_act[0, 0]
    assert max_launch[0, 0] == 0

    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(),
        action_spec=action_spec,
        pin_memory=False,
    )
    obs = env.reset()
    env_index, player, entity = torch.nonzero(obs.can_act, as_tuple=False)[0].tolist()
    shape = (1, 4, ACTION_ENTITY_SLOTS, action_spec.max_per_planet_launches)
    launch = np.zeros(shape, dtype=np.bool_)
    angle = np.zeros(shape, dtype=np.float32)
    ships = np.zeros(shape, dtype=np.int64)
    launch[env_index, player, entity, 0] = True
    ships[env_index, player, entity, 0] = action_spec.min_fleet_size - 1

    with pytest.raises(ValueError, match="ships must be >= 3"):
        env.step(launch, angle, ships)


def test_python_observation_encoder_writes_discrete_target_mask() -> None:
    (
        _planets,
        _orbiting_planets,
        _fleets,
        _comets,
        entity_mask,
        _global_features,
        can_act,
        max_launch,
    ) = encode_python_observation(
        {
            "step": 0,
            "angular_velocity": 0.025,
            "planets": [
                [0, 0, 25.0, 75.0, 2.0, 10, 3],
                [1, -1, 75.0, 75.0, 2.0, 10, 3],
            ],
            "fleets": [],
            "comets": [],
        },
        action_spec=ActionDiscreteTargetsConfig(),
    )

    assert entity_mask[:2].tolist() == [True, True]
    assert not entity_mask[MAX_PLANETS:ACTION_ENTITY_SLOTS].any()
    assert can_act.shape == (4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS)
    assert not can_act[0, 0, 0]
    assert can_act[0, 0, 1]
    assert not can_act[0, 0, 2]
    assert max_launch[0, 0] == 10


def test_python_observation_encoder_matches_rl_schema_and_masks() -> None:
    (
        planets,
        orbiting_planets,
        fleets,
        comets,
        entity_mask,
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
    assert orbiting_planets.shape == (MAX_PLANETS,)
    assert fleets.shape == (EntityBasedConfig().max_fleets, FLEET_CHANNELS)
    assert comets.shape == (MAX_COMETS, COMET_CHANNELS)
    assert global_features.shape == (GLOBAL_CHANNELS,)
    assert can_act.shape == (4, ACTION_ENTITY_SLOTS)
    assert max_launch.shape == (4, ACTION_ENTITY_SLOTS)
    assert PLANET_CHANNELS == 107
    assert FLEET_CHANNELS == 79
    assert COMET_CHANNELS == 330
    planet_mask = entity_mask[:MAX_PLANETS]
    comet_mask = entity_mask[MAX_PLANETS:ACTION_ENTITY_SLOTS]
    fleet_mask = entity_mask[ACTION_ENTITY_SLOTS:]
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


def test_encode_entity_based_matches_expected_masks_and_masked_values() -> None:
    spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 5)
    min_fleet_size = 12
    planets_in = np.array(
        [
            [0, 0, 25.0, 75.0, 2.0, 50, 3],
            [1, -1, 50.0, 50.0, 1.5, 0, 1],
            [2, 2, 100.0, 0.0, 3.0, 125, 5],
            [3, 3, 0.0, 100.0, 1.0, 10, 2],
            [4, 1, 75.0, 25.0, 2.5, 30, 4],
        ],
        dtype=np.float64,
    )
    fleets_in = np.array(
        [
            [10, 1, 0.0, 100.0, 0.0, 0, 8],
            [11, 3, 100.0, 0.0, np.pi / 2, 1, 27],
            [12, 2, 50.0, 25.0, np.pi, 3, 64],
            [13, 0, 25.0, 50.0, np.pi / 4, 0, 125],
        ],
        dtype=np.float64,
    )
    comet_planet_ids = np.full((1, MAX_COMETS), -1.0, dtype=np.float64)
    comet_planet_ids[0, :2] = [2, 4]
    comet_path_indices = np.array([1.0], dtype=np.float64)
    comet_path_lengths = np.zeros((1, MAX_COMETS), dtype=np.float64)
    comet_path_lengths[0, :2] = [4, 3]
    comet_paths = np.zeros(
        (1, MAX_COMETS, MAX_COMET_PATH_LENGTH, 2),
        dtype=np.float64,
    )
    comet_paths[0, 0, :4] = [
        [0.0, 0.0],
        [50.0, 100.0],
        [100.0, 50.0],
        [25.0, 25.0],
    ]
    comet_paths[0, 1, :3] = [
        [100.0, 100.0],
        [0.0, 50.0],
        [50.0, 0.0],
    ]

    (
        planets,
        orbiting_planets,
        fleets,
        comets,
        entity_mask,
        global_features,
        can_act,
        max_launch,
    ) = encode_entity_based(
        planets_in,
        fleets_in,
        comet_planet_ids,
        comet_path_indices,
        comet_path_lengths,
        comet_paths,
        0.0375,
        120,
        600,
        spec.max_entities,
        min_fleet_size,
    )

    expected_planet_mask = np.zeros(MAX_PLANETS, dtype=np.bool_)
    expected_planet_mask[:3] = True
    expected_fleet_mask = np.zeros(spec.max_fleets, dtype=np.bool_)
    expected_fleet_mask[:4] = True
    expected_comet_mask = np.array([True, True, False, False])
    planet_mask = entity_mask[:MAX_PLANETS]
    comet_mask = entity_mask[MAX_PLANETS:ACTION_ENTITY_SLOTS]
    fleet_mask = entity_mask[ACTION_ENTITY_SLOTS:]
    np.testing.assert_array_equal(planet_mask, expected_planet_mask)
    np.testing.assert_array_equal(fleet_mask, expected_fleet_mask)
    np.testing.assert_array_equal(comet_mask, expected_comet_mask)
    assert orbiting_planets.shape == (MAX_PLANETS,)

    def normalized_position(value: float) -> float:
        return value / 50.0 - 1.0

    def normalized_log_ships(ships: int) -> float:
        return np.log(np.float32(max(ships, 0) + 1)) / np.float32(4.6051702)

    def ship_count_basis(ships: int, buckets: list[int]) -> np.ndarray:
        ships = max(ships, 0)
        values = np.zeros(len(buckets) * 2 + 2, dtype=np.float32)
        if ships == 0:
            values[0] = 1.0
            values[len(buckets)] = 1.0
        elif ships >= buckets[-1]:
            values[len(buckets) - 1] = 1.0
            values[len(buckets) * 2 - 1] = 1.0
        else:
            hi = next(index for index, bucket in enumerate(buckets) if bucket >= ships)
            if buckets[hi] == ships:
                values[hi] = 1.0
                values[len(buckets) + hi] = 1.0
            else:
                lo = hi - 1
                linear_hi = (ships - buckets[lo]) / (buckets[hi] - buckets[lo])
                values[lo] = 1.0 - linear_hi
                values[hi] = linear_hi
                log_hi = (np.log(ships) - np.log(buckets[lo])) / (
                    np.log(buckets[hi]) - np.log(buckets[lo])
                )
                values[len(buckets) + lo] = 1.0 - log_hi
                values[len(buckets) + hi] = log_hi
        if ships > buckets[-1]:
            values[-2] = 1.0
            values[-1] = np.log(max(ships - buckets[-1], 1))
        return values

    def fleet_ship_count_basis(ships: int, min_fleet_size: int) -> np.ndarray:
        buckets = [1]
        next_bucket = 1 << (min_fleet_size - 1).bit_length()
        if next_bucket <= min_fleet_size:
            next_bucket *= 2
        while next_bucket <= 512:
            buckets.append(next_bucket)
            next_bucket *= 2
        values = np.zeros(22, dtype=np.float32)
        if ships == 0:
            return values
        active = ship_count_basis(ships, buckets)
        values[: len(buckets)] = active[: len(buckets)]
        values[10 : 10 + len(buckets)] = active[len(buckets) : len(buckets) * 2]
        values[20:] = active[len(buckets) * 2 :]
        return values

    def normalized_fleet_speed(ships: int) -> float:
        return (1.0 + 5.0 * (np.log(ships) / np.log(1000)) ** 1.5) / 6.0

    def spatial_features(x: float, y: float) -> np.ndarray:
        values: list[float] = []
        for frequency in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
            values.extend(
                [
                    np.sin(np.pi * frequency * x),
                    np.cos(np.pi * frequency * x),
                    np.sin(np.pi * frequency * y),
                    np.cos(np.pi * frequency * y),
                ]
            )

        radius = np.hypot(x, y)
        theta = np.arctan2(y, x)
        values.extend([radius, np.log1p(radius), np.sin(theta), np.cos(theta)])

        for harmonic in [2.0, 3.0, 4.0]:
            values.extend(
                [
                    np.sin(harmonic * theta),
                    np.cos(harmonic * theta),
                ]
            )

        for frequency in [1.0, 2.0, 4.0, 8.0]:
            values.extend(
                [
                    np.sin(np.pi * frequency * radius),
                    np.cos(np.pi * frequency * radius),
                ]
            )

        return np.asarray(values, dtype=np.float32)

    def fleet_motion_features(
        x: float, y: float, velocity_x: float, velocity_y: float
    ) -> np.ndarray:
        speed = np.hypot(velocity_x, velocity_y)
        heading_x = velocity_x / speed if speed > 0.0 else 0.0
        heading_y = velocity_y / speed if speed > 0.0 else 0.0
        radius = np.hypot(x, y)
        radial_velocity = 0.0
        tangential_velocity = 0.0
        if radius > 0.0:
            radial_x = x / radius
            radial_y = y / radius
            radial_velocity = velocity_x * radial_x + velocity_y * radial_y
            tangential_velocity = velocity_x * -radial_y + velocity_y * radial_x
        return np.asarray(
            [
                speed,
                heading_x,
                heading_y,
                radial_velocity,
                tangential_velocity,
            ],
            dtype=np.float32,
        )

    def planet_orbital_velocity(
        x: float, y: float, angular_velocity: float, orbiting: bool
    ) -> np.ndarray:
        if not orbiting:
            return np.zeros(2, dtype=np.float32)
        return np.asarray(
            [
                -angular_velocity * y,
                angular_velocity * x,
            ],
            dtype=np.float32,
        )

    def planet_ship_features(ships: int, owner: int) -> list[float]:
        if owner == -1:
            return [
                ships / 100.0,
                normalized_log_ships(ships),
                0.0,
                0.0,
            ]
        return [
            0.0,
            0.0,
            ships / 500.0,
            normalized_log_ships(ships),
        ]

    def planet_ship_count_features(ships: int, owner: int) -> np.ndarray:
        neutral = ship_count_basis(ships, [0, 1, 2, 4, 8, 16, 32, 64, 99])
        owned = ship_count_basis(
            ships,
            [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        )
        if owner == -1:
            return np.concatenate([neutral, np.zeros_like(owned)])
        return np.concatenate([np.zeros_like(neutral), owned])

    def comet_ship_count_features(ships: int, owner: int) -> np.ndarray:
        neutral = ship_count_basis(ships, [0, 1, 2, 4, 8, 16, 32, 64, 99])
        owned = ship_count_basis(
            ships,
            [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        )
        if owner == -1:
            return np.concatenate([neutral, np.zeros_like(owned)])
        return np.concatenate([np.zeros_like(neutral), owned])

    def normalized_point(point: list[float]) -> np.ndarray:
        return np.asarray(
            [
                normalized_position(point[0]),
                normalized_position(point[1]),
            ],
            dtype=np.float32,
        )

    def fill_comet_path_features(
        row: np.ndarray, path: list[list[float]], path_start: int
    ) -> None:
        base = 52
        current = normalized_point(path[path_start])
        row[base : base + 2] = current
        row[base + 2 : base + 44] = spatial_features(current[0], current[1])

        velocity = np.zeros(2, dtype=np.float32)
        if path_start + 1 < len(path):
            velocity = normalized_point(path[path_start + 1]) - current
        row[base + 44 : base + 46] = velocity
        row[base + 46 : base + 51] = fleet_motion_features(
            current[0],
            current[1],
            velocity[0],
            velocity[1],
        )

        offsets = [1, 2, 4, 8, 16]
        valid_start = base + 51
        positions_start = valid_start + len(offsets)
        spatial_start = positions_start + len(offsets) * 2
        for selected_index, offset in enumerate(offsets):
            selected_path_index = path_start + offset
            if selected_path_index >= len(path):
                continue
            position = normalized_point(path[selected_path_index])
            row[valid_start + selected_index] = 1.0
            position_start = positions_start + selected_index * 2
            row[position_start : position_start + 2] = position
            selected_spatial_start = spatial_start + selected_index * 42
            row[selected_spatial_start : selected_spatial_start + 42] = (
                spatial_features(
                    position[0],
                    position[1],
                )
            )

        row[328:330] = normalized_point(path[-1]) - current

    base_expected_planets = np.array(
        [
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                normalized_position(25.0),
                normalized_position(75.0),
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                2.0 / 3.0,
                *planet_ship_features(50, 0),
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                normalized_position(50.0),
                normalized_position(50.0),
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.5 / 3.0,
                *planet_ship_features(0, -1),
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                normalized_position(0.0),
                normalized_position(100.0),
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0 / 3.0,
                *planet_ship_features(10, 3),
            ],
        ],
        dtype=np.float32,
    )
    expected_planets = np.asarray(
        [
            np.concatenate(
                [
                    row,
                    count_features,
                    spatial_features(row[5], row[6]),
                    planet_orbital_velocity(row[5], row[6], 0.0375, orbiting),
                ]
            )
            for row, ships, owner, orbiting in zip(
                base_expected_planets,
                [50, 0, 10],
                [0, -1, 3],
                [True, True, False],
                strict=True,
            )
            for count_features in [planet_ship_count_features(ships, owner)]
        ],
        dtype=np.float32,
    )
    expected_orbiting_planets = np.array([True, True, False], dtype=np.bool_)
    np.testing.assert_allclose(planets[planet_mask], expected_planets, atol=1e-6)
    np.testing.assert_array_equal(
        orbiting_planets[planet_mask],
        expected_orbiting_planets,
    )

    speed_8 = normalized_fleet_speed(8)
    speed_27 = normalized_fleet_speed(27)
    speed_64 = normalized_fleet_speed(64)
    speed_125 = normalized_fleet_speed(125)
    base_expected_fleets = np.array(
        [
            [
                0.0,
                1.0,
                0.0,
                0.0,
                normalized_position(0.0),
                normalized_position(100.0),
                speed_8,
                0.0,
                8.0 / 500.0,
                normalized_log_ships(8),
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
                normalized_position(100.0),
                normalized_position(0.0),
                np.cos(np.pi / 2) * speed_27,
                np.sin(np.pi / 2) * speed_27,
                27.0 / 500.0,
                normalized_log_ships(27),
            ],
            [
                0.0,
                0.0,
                1.0,
                0.0,
                normalized_position(50.0),
                normalized_position(25.0),
                np.cos(np.pi) * speed_64,
                np.sin(np.pi) * speed_64,
                64.0 / 500.0,
                normalized_log_ships(64),
            ],
            [
                1.0,
                0.0,
                0.0,
                0.0,
                normalized_position(25.0),
                normalized_position(50.0),
                np.cos(np.pi / 4) * speed_125,
                np.sin(np.pi / 4) * speed_125,
                125.0 / 500.0,
                normalized_log_ships(125),
            ],
        ],
        dtype=np.float32,
    )
    expected_fleets = np.asarray(
        [
            np.concatenate(
                [
                    row,
                    fleet_ship_count_basis(ships, min_fleet_size),
                    spatial_features(row[4], row[5]),
                    fleet_motion_features(row[4], row[5], row[6], row[7]),
                ]
            )
            for row, ships in zip(base_expected_fleets, [8, 27, 64, 125], strict=True)
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(fleets[fleet_mask], expected_fleets, atol=1e-6)

    expected_comets = np.zeros((2, COMET_CHANNELS), dtype=np.float32)
    expected_comets[0, 2] = 1.0
    expected_comets[0, 5] = 125.0 / 500.0
    expected_comets[0, 6] = normalized_log_ships(125)
    expected_comets[0, 7:51] = comet_ship_count_features(125, 2)
    expected_comets[0, 51] = 3.0 / MAX_COMET_PATH_LENGTH
    fill_comet_path_features(
        expected_comets[0],
        [
            [0.0, 0.0],
            [50.0, 100.0],
            [100.0, 50.0],
            [25.0, 25.0],
        ],
        1,
    )
    expected_comets[1, 1] = 1.0
    expected_comets[1, 5] = 30.0 / 500.0
    expected_comets[1, 6] = normalized_log_ships(30)
    expected_comets[1, 7:51] = comet_ship_count_features(30, 1)
    expected_comets[1, 51] = 2.0 / MAX_COMET_PATH_LENGTH
    fill_comet_path_features(
        expected_comets[1],
        [
            [100.0, 100.0],
            [0.0, 50.0],
            [50.0, 0.0],
        ],
        1,
    )
    np.testing.assert_allclose(comets[comet_mask], expected_comets, atol=1e-6)

    np.testing.assert_allclose(
        global_features,
        np.array(
            [
                120.0 / 600.0,
                30.0 / 100.0,
                (np.float32(0.0375) - np.float32(0.025)) / np.float32(0.025),
            ],
            dtype=np.float32,
        ),
        atol=0.0,
    )

    expected_can_act = np.zeros((4, ACTION_ENTITY_SLOTS), dtype=np.bool_)
    expected_can_act[0, 0] = True
    expected_can_act[2, MAX_PLANETS] = True
    expected_can_act[1, MAX_PLANETS + 1] = True
    expected_max_launch = np.zeros((4, ACTION_ENTITY_SLOTS), dtype=np.int64)
    expected_max_launch[0, 0] = 50
    expected_max_launch[2, MAX_PLANETS] = 125
    expected_max_launch[1, MAX_PLANETS + 1] = 30
    np.testing.assert_array_equal(can_act, expected_can_act)
    np.testing.assert_array_equal(max_launch, expected_max_launch)


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
    spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 1)

    _, _, fleets, _, entity_mask, _, _, _ = encode_python_observation(
        {
            "planets": [],
            "fleets": [
                [1, 0, 10.0, 20.0, 0.0, 0, 5],
                [2, 1, 30.0, 40.0, 0.0, 0, 20],
            ],
        },
        spec,
    )
    fleet_mask = entity_mask[ACTION_ENTITY_SLOTS:]

    assert fleet_mask.tolist() == [True]
    assert fleets[0, 1] == 1
    assert fleets[0, 8] == pytest.approx(20 / 500.0)
    assert "max_entities exceeded: 1 fleets ignored" in capfd.readouterr().err


def test_python_observation_encoder_writes_comet_future_paths() -> None:
    path = [[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]]
    (
        planets,
        orbiting_planets,
        _fleets,
        comets,
        entity_mask,
        _global_features,
        can_act,
        max_launch,
    ) = encode_python_observation(
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
    planet_mask = entity_mask[:MAX_PLANETS]
    comet_mask = entity_mask[MAX_PLANETS:ACTION_ENTITY_SLOTS]

    assert not planet_mask[0]
    assert comet_mask.tolist() == [True, False, False, False]
    assert planets[0].tolist() == [0.0] * PLANET_CHANNELS
    assert not orbiting_planets[0]
    assert comets[0, 2] == 1.0
    assert comets[0, 5] == pytest.approx(25 / 500.0)
    assert comets[0, 51] == pytest.approx(2 / MAX_COMET_PATH_LENGTH)
    assert comets[0, 52] == pytest.approx(0.0)
    assert comets[0, 53] == pytest.approx(0.0)
    assert comets[0, 96] == pytest.approx(1.0)
    assert comets[0, 97] == pytest.approx(1.0)
    np.testing.assert_array_equal(
        comets[0, 103:108],
        np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert comets[0, 108] == pytest.approx(1.0)
    assert comets[0, 109] == pytest.approx(1.0)
    assert comets[0, 328] == pytest.approx(1.0)
    assert comets[0, 329] == pytest.approx(1.0)
    assert can_act[2, MAX_PLANETS]
    assert max_launch[2, MAX_PLANETS] == 25
