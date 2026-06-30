import json
import math

import numpy as np
import pytest
import torch
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    COMET_CHANNELS,
    CROSS_ATTENTION_FLEET_CHANNELS,
    FLEET_CHANNELS,
    GLOBAL_CHANNELS,
    GLOBAL_EXT_V2_CHANNELS,
    MAX_COMET_PATH_LENGTH,
    MAX_COMETS,
    MAX_PLANETS,
    PLANET_CHANNELS,
    PLAYER_FEATURE_CHANNELS,
    TARGET_INCOMING_CHANNELS,
    ActionDiscreteTargetBinsConfig,
    ActionDiscreteTargetsConfig,
    ActionPureConfig,
    DecodedLaunchActions,
    DiscreteTargetActions,
    DiscreteTargetBinActionMask,
    DiscreteTargetBinActions,
    EntityBasedConfig,
    EntityBasedCrossAttnV1Config,
    EntityBasedExtV1Config,
    EntityBasedExtV2Config,
    EnvConfig,
    ObsBatch,
    PureActions,
    VectorizedEnv,
    actions_to_kaggle,
    encode_entity_based,
    encode_python_observation,
    encode_python_observation_with_metrics,
)
from owl.rs import RlVecEnv as RustRlVecEnv
from owl.rs import encode_entity_based_with_player_features


def _python_obs(**overrides: object) -> dict[str, object]:
    obs = {
        "step": 0,
        "episode_steps": 500,
        "angular_velocity": 0.025,
        "planets": [],
        "initial_planets": [],
        "fleets": [],
        "player": 0,
        "comets": [],
    }
    obs.update(overrides)
    if "initial_planets" not in overrides:
        obs["initial_planets"] = obs["planets"]
    return obs


def _encoded_python_observation(
    obs: dict[str, object],
    *,
    obs_spec: EntityBasedConfig,
    action_spec: ActionPureConfig
    | ActionDiscreteTargetsConfig
    | ActionDiscreteTargetBinsConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    encoded = encode_python_observation(
        obs,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    action_mask = encoded.action_mask
    return (
        encoded.planets[0].numpy(),
        encoded.orbiting_planets[0].numpy(),
        encoded.fleets[0].numpy(),
        encoded.comets[0].numpy(),
        encoded.entity_mask[0].numpy(),
        encoded.global_features[0].numpy(),
        action_mask.can_act[0].numpy(),
        None
        if isinstance(action_mask, DiscreteTargetBinActionMask)
        else action_mask.max_launch[0].numpy(),
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
    env.observations.action_mask.can_act.fill_(True)
    env.observations.action_mask.max_launch.fill_(123)
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
    assert torch.any(obs.action_mask.can_act)
    assert torch.all(obs.action_mask.max_launch[~obs.action_mask.can_act] == 0)


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
    obs, rewards, dones, episode_metrics = env.step(
        PureActions(launch=launch, angle=angle, ships=ships)
    )

    assert rewards.data_ptr() == reward_ptr
    assert dones.data_ptr() == done_ptr
    assert obs.planets.shape == (2, MAX_PLANETS, PLANET_CHANNELS)
    assert obs.orbiting_planets.shape == (2, MAX_PLANETS)
    assert obs.fleets.shape == (2, env.obs_spec.max_fleets, FLEET_CHANNELS)
    assert obs.comets.shape == (2, MAX_COMETS, COMET_CHANNELS)
    assert obs.global_features.shape == (2, GLOBAL_CHANNELS)
    assert obs.still_playing.shape == (2, 4)
    assert obs.action_mask.can_act.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert obs.action_mask.max_launch.shape == (2, 4, ACTION_ENTITY_SLOTS)
    assert rewards.shape == (2, 4)
    assert dones.shape == (2, 4)
    assert episode_metrics == {}
    assert torch.equal(rewards, torch.zeros_like(rewards))
    assert torch.equal(dones, torch.zeros_like(dones))


def test_entity_based_ext_v1_adds_ship_count_one_hot_channels() -> None:
    spec = EntityBasedExtV1Config(
        max_entities=MAX_PLANETS + MAX_COMETS + 2,
        ship_count_one_hot_max=3,
    )

    assert spec.planet_channels == PLANET_CHANNELS + 4
    assert spec.fleet_channels == FLEET_CHANNELS + 3

    encoded = encode_python_observation(
        _python_obs(
            planets=[
                [0, 0, 25.0, 75.0, 2.0, 0, 3],
                [1, 1, 75.0, 25.0, 2.0, 2, 3],
                [2, 1, 50.0, 50.0, 2.0, 4, 3],
            ],
            fleets=[
                [10, 1, 0.0, 100.0, 0.0, 0, 1],
                [11, 1, 100.0, 0.0, 0.0, 1, 5],
            ],
        ),
        obs_spec=spec,
        action_spec=ActionPureConfig(min_fleet_size=1),
    )

    np.testing.assert_array_equal(encoded.planets[0, 0, PLANET_CHANNELS:], [1, 0, 0, 0])
    np.testing.assert_array_equal(encoded.planets[0, 1, PLANET_CHANNELS:], [0, 0, 1, 0])
    np.testing.assert_array_equal(encoded.planets[0, 2, PLANET_CHANNELS:], [0, 0, 0, 1])
    np.testing.assert_array_equal(encoded.fleets[0, 0, FLEET_CHANNELS:], [1, 0, 0])
    np.testing.assert_array_equal(encoded.fleets[0, 1, FLEET_CHANNELS:], [0, 0, 1])
    np.testing.assert_array_equal(encoded.comets.shape, (1, MAX_COMETS, COMET_CHANNELS))


def test_entity_based_ext_v2_adds_global_and_player_summary_features() -> None:
    spec = EntityBasedExtV2Config(max_entities=MAX_PLANETS + MAX_COMETS + 4)

    assert spec.planet_channels == PLANET_CHANNELS
    assert spec.fleet_channels == FLEET_CHANNELS
    assert spec.global_channels == GLOBAL_CHANNELS + GLOBAL_EXT_V2_CHANNELS
    assert spec.player_feature_channels == PLAYER_FEATURE_CHANNELS

    encoded = encode_python_observation(
        _python_obs(
            planets=[
                [0, 0, 25.0, 75.0, 2.0, 100, 3],
                [1, 0, 75.0, 25.0, 2.0, 50, 2],
                [2, 1, 50.0, 25.0, 2.0, 40, 4],
                [3, -1, 25.0, 25.0, 2.0, 30, 5],
                [10, 0, 50.0, 50.0, 1.0, 25, 1],
                [11, 1, 60.0, 50.0, 1.0, 15, 1],
                [12, -1, 70.0, 50.0, 1.0, 20, 1],
            ],
            fleets=[
                [20, 0, 0.0, 100.0, 0.0, 0, 10],
                [21, 0, 100.0, 0.0, 0.0, 1, 20],
                [22, 1, 50.0, 0.0, 0.0, 2, 5],
            ],
            comets=[
                {
                    "planet_ids": [10, 11, 12],
                    "paths": [
                        [[50.0, 50.0], [51.0, 50.0]],
                        [[60.0, 50.0], [61.0, 50.0]],
                        [[70.0, 50.0], [71.0, 50.0]],
                    ],
                    "path_index": 0,
                }
            ],
        ),
        obs_spec=spec,
        action_spec=ActionPureConfig(min_fleet_size=1),
    )

    assert encoded.player_features is not None
    assert encoded.planets.shape == (1, MAX_PLANETS, PLANET_CHANNELS)
    assert encoded.fleets.shape == (1, spec.max_fleets, FLEET_CHANNELS)

    def normalized_aggregate_log_ships(ships: int) -> float:
        return np.log(np.float32(max(ships, 0) + 1)) / np.float32(6.9077554)

    np.testing.assert_allclose(
        encoded.player_features[0, 0].numpy(),
        [
            0.06,
            0.01,
            0.05,
            205.0 / 5000.0,
            normalized_aggregate_log_ships(205),
            25.0 / 5000.0,
            normalized_aggregate_log_ships(25),
            150.0 / 5000.0,
            normalized_aggregate_log_ships(150),
            30.0 / 5000.0,
            normalized_aggregate_log_ships(30),
            0.05,
            0.25,
            0.02,
        ],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        encoded.player_features[0, 1].numpy(),
        [
            0.05,
            0.01,
            0.04,
            60.0 / 5000.0,
            normalized_aggregate_log_ships(60),
            15.0 / 5000.0,
            normalized_aggregate_log_ships(15),
            40.0 / 5000.0,
            normalized_aggregate_log_ships(40),
            5.0 / 5000.0,
            normalized_aggregate_log_ships(5),
            0.025,
            0.25,
            0.01,
        ],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        encoded.global_features[0, GLOBAL_CHANNELS:].numpy(),
        [
            0.06,
            0.01,
            0.05,
            50.0 / 5000.0,
            normalized_aggregate_log_ships(50),
            20.0 / 5000.0,
            normalized_aggregate_log_ships(20),
            30.0 / 5000.0,
            normalized_aggregate_log_ships(30),
            0.25,
            0.025,
            1.0,
            0.0,
            0.0,
        ],
        rtol=1e-6,
    )


@pytest.mark.parametrize("alive_count", [2, 3, 4])
def test_entity_based_ext_v2_adds_alive_player_count_global_one_hot(
    alive_count: int,
) -> None:
    encoded = encode_python_observation(
        _python_obs(
            planets=[
                [
                    player,
                    player,
                    20.0 + 10.0 * player,
                    20.0,
                    2.0,
                    10,
                    1,
                ]
                for player in range(alive_count)
            ],
        ),
        obs_spec=EntityBasedExtV2Config(),
        action_spec=ActionPureConfig(),
    )

    expected = np.zeros(3, dtype=np.float32)
    expected[alive_count - 2] = 1.0
    np.testing.assert_array_equal(
        encoded.global_features[0, GLOBAL_CHANNELS + 11 : GLOBAL_CHANNELS + 14],
        expected,
    )


def test_entity_based_ext_v2_rejects_ship_count_one_hot_hybrid() -> None:
    with pytest.raises(
        ValueError,
        match="entity_based_ext_v2 requires ship_count_one_hot_max=0",
    ):
        encode_entity_based_with_player_features(
            np.zeros((0, 7), dtype=np.float64),
            np.zeros((0, 7), dtype=np.float64),
            np.zeros((0, 7), dtype=np.float64),
            np.full((0, MAX_COMETS), -1.0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, MAX_COMETS), dtype=np.float64),
            np.zeros((0, MAX_COMETS, MAX_COMET_PATH_LENGTH, 2), dtype=np.float64),
            0.025,
            ship_count_one_hot_max=3,
            player_feature_channels=PLAYER_FEATURE_CHANNELS,
        )


def test_vectorized_env_accepts_entity_based_ext_v1_shapes() -> None:
    spec = EntityBasedExtV1Config(ship_count_one_hot_max=5)
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=spec,
        action_spec=ActionPureConfig(),
        pin_memory=False,
    )

    obs = env.reset()

    assert obs.planets.shape == (2, MAX_PLANETS, spec.planet_channels)
    assert obs.fleets.shape == (2, spec.max_fleets, spec.fleet_channels)
    assert obs.comets.shape == (2, MAX_COMETS, COMET_CHANNELS)


def test_vectorized_env_accepts_entity_based_ext_v2_shapes() -> None:
    spec = EntityBasedExtV2Config()
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=spec,
        action_spec=ActionPureConfig(),
        pin_memory=False,
    )

    obs = env.reset()

    assert obs.planets.shape == (2, MAX_PLANETS, PLANET_CHANNELS)
    assert obs.fleets.shape == (2, spec.max_fleets, FLEET_CHANNELS)
    assert obs.global_features.shape == (2, spec.global_channels)
    assert obs.player_features is not None
    assert obs.player_features.shape == (2, 4, PLAYER_FEATURE_CHANNELS)


def test_vectorized_env_accepts_entity_based_cross_attn_shapes() -> None:
    spec = EntityBasedCrossAttnV1Config()
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=spec,
        action_spec=ActionPureConfig(),
        pin_memory=False,
    )

    obs = env.reset()

    assert obs.planets.shape == (2, MAX_PLANETS, PLANET_CHANNELS)
    assert obs.fleets.shape == (2, spec.max_fleets, CROSS_ATTENTION_FLEET_CHANNELS)
    assert obs.fleet_target is not None
    assert obs.fleet_target.shape == (2, spec.max_fleets)
    assert obs.target_incoming_features is not None
    assert obs.target_incoming_features.shape == (
        2,
        ACTION_ENTITY_SLOTS,
        TARGET_INCOMING_CHANNELS,
    )
    assert obs.global_features.shape == (2, spec.global_channels)
    assert obs.player_features is not None
    assert obs.player_features.shape == (2, 4, PLAYER_FEATURE_CHANNELS)


def test_rust_vec_env_keeps_legacy_entity_based_observation_contract() -> None:
    max_entities = MAX_PLANETS + MAX_COMETS + 2
    env = RustRlVecEnv(
        1,
        obs_spec="entity_based",
        action_spec="pure",
        max_entities=max_entities,
    )
    shapes = env.obs_shapes()

    assert len(shapes) == 9
    planet_obs = np.zeros(shapes[0], dtype=np.float32)
    orbiting_planet_obs = np.zeros(shapes[1], dtype=np.bool_)
    fleet_obs = np.zeros(shapes[2], dtype=np.float32)
    comet_obs = np.zeros(shapes[3], dtype=np.float32)
    entity_mask = np.zeros(shapes[4], dtype=np.bool_)
    still_playing = np.zeros(shapes[5], dtype=np.bool_)
    global_obs = np.zeros(shapes[6], dtype=np.float32)
    can_act = np.zeros(shapes[7], dtype=np.bool_)
    max_launch = np.zeros(shapes[8], dtype=np.int64)

    env.reset(
        planet_obs,
        orbiting_planet_obs,
        fleet_obs,
        comet_obs,
        entity_mask,
        still_playing,
        global_obs,
        can_act,
        max_launch,
    )
    env.write_observation(
        "entity_based",
        "pure",
        max_entities,
        0,
        1,
        0,
        "full_mask",
        planet_obs,
        orbiting_planet_obs,
        fleet_obs,
        comet_obs,
        entity_mask,
        still_playing,
        global_obs,
        can_act,
        max_launch,
    )


def test_rust_vec_env_accepts_ext_v2_player_features_as_extra_argument() -> None:
    env = RustRlVecEnv(1, obs_spec="entity_based_ext_v2", action_spec="pure")
    shapes = env.obs_shapes()

    assert len(shapes) == 10
    planet_obs = np.zeros(shapes[0], dtype=np.float32)
    orbiting_planet_obs = np.zeros(shapes[1], dtype=np.bool_)
    fleet_obs = np.zeros(shapes[2], dtype=np.float32)
    comet_obs = np.zeros(shapes[3], dtype=np.float32)
    entity_mask = np.zeros(shapes[4], dtype=np.bool_)
    still_playing = np.zeros(shapes[5], dtype=np.bool_)
    global_obs = np.zeros(shapes[6], dtype=np.float32)
    player_features = np.zeros(shapes[7], dtype=np.float32)
    can_act = np.zeros(shapes[8], dtype=np.bool_)
    max_launch = np.zeros(shapes[9], dtype=np.int64)

    env.reset(
        planet_obs,
        orbiting_planet_obs,
        fleet_obs,
        comet_obs,
        entity_mask,
        still_playing,
        global_obs,
        can_act,
        max_launch,
        player_features,
    )


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
        env.step(PureActions(launch=launch, angle=angle, ships=ships))


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
        env.step(PureActions(launch=launch, angle=angle, ships=ships))


@pytest.mark.parametrize(
    ("ship_count", "angle_value", "message"),
    [
        (0, 0.0, "ships must be >= 6"),
        (6, np.inf, "angle must be finite"),
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
    env_index, player, entity = torch.nonzero(obs.action_mask.can_act, as_tuple=False)[
        0
    ].tolist()
    shape = (1, 4, ACTION_ENTITY_SLOTS, env.action_spec.max_per_planet_launches)
    launch = np.zeros(shape, dtype=np.bool_)
    angle = np.zeros(shape, dtype=np.float32)
    ships = np.zeros(shape, dtype=np.int64)
    launch[env_index, player, entity, 0] = True
    angle[env_index, player, entity, 0] = angle_value
    ships[env_index, player, entity, 0] = ship_count

    with pytest.raises(ValueError, match=message):
        env.step(PureActions(launch=launch, angle=angle, ships=ships))


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
        obs, _rewards, dones, _episode_metrics = env.step(
            PureActions(launch=launch, angle=angle, ships=ships)
        )
        if bool(dones.all()):
            terminal_snapshot = env.terminal_snapshot(0)
            break

    assert terminal_snapshot is not None
    terminal_metrics = env.terminal_metrics(0)
    assert terminal_metrics is not None
    assert "_neutral_planets_captured_per_game" in terminal_metrics
    assert "_neutral_comets_captured_per_game" in terminal_metrics
    assert "_neutral_planet_undershots_per_game" in terminal_metrics
    assert "_neutral_comet_undershots_per_game" in terminal_metrics
    assert terminal_snapshot["step"] > obs.global_features[0, 0].item()
    assert terminal_snapshot["player_count"] == 2

    truncated_obs = env.truncate_envs(np.array([True], dtype=np.bool_))

    assert truncated_obs is env.observations
    assert env.terminal_snapshot(0) is None
    assert env.terminal_metrics(0) is None


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

    _, rewards, dones, episode_metrics = env.step(
        PureActions(launch=launch, angle=angle, ships=ships)
    )

    assert torch.equal(rewards, torch.zeros_like(rewards))
    assert episode_metrics == {}
    assert torch.equal(dones.sum(dim=1), torch.full((2,), 2))
    assert torch.equal(env.observations.still_playing, ~dones)


def test_vectorized_env_accepts_discriminated_config_dicts() -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1, min_fleet_size=4)
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 1),
        action_spec=action_spec,
        pin_memory=False,
    )

    assert env.obs_spec.max_fleets == 1
    assert env.action_spec.action_spec == "pure"
    assert env.action_spec.max_per_planet_launches == 1
    assert env.action_spec.min_fleet_size == 4


def test_action_config_validates_launch_bounds() -> None:
    assert ActionPureConfig().max_per_planet_launches == 1
    assert ActionPureConfig(max_per_planet_launches=1).max_per_planet_launches == 1
    assert ActionPureConfig(min_fleet_size=5).min_fleet_size == 5
    assert ActionDiscreteTargetsConfig().max_per_planet_launches == 1
    assert (
        ActionDiscreteTargetsConfig(max_per_planet_launches=1).max_per_planet_launches
        == 1
    )

    with pytest.raises(ValueError, match="less than or equal to 1"):
        ActionPureConfig(max_per_planet_launches=2)
    with pytest.raises(ValueError, match="less than or equal to 1"):
        ActionDiscreteTargetsConfig(max_per_planet_launches=2)
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        ActionPureConfig(min_fleet_size=0)


def test_env_config_requires_even_env_count() -> None:
    assert EnvConfig().n_envs == 2

    with pytest.raises(ValueError, match="n_envs must be even"):
        EnvConfig(n_envs=1)


@pytest.mark.parametrize("reward_mode", ["ship_ratio", "win_only"])
def test_vectorized_env_accepts_reward_mode_config(reward_mode: str) -> None:
    config = EnvConfig(reward_mode=reward_mode)
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=config.obs_spec,
        action_spec=config.action_spec,
        reward_mode=config.reward_mode,
        pin_memory=False,
    )

    assert env.reward_mode == reward_mode


def test_discrete_targets_config_and_env_shapes() -> None:
    config = EnvConfig.model_validate(
        {
            "n_envs": 2,
            "action_spec": {
                "action_spec": "discrete_targets",
                "max_per_planet_launches": 1,
                "min_fleet_size": 4,
                "targeting_mode": "stop_bad_launch",
            },
            "pin_memory": False,
        }
    )
    assert isinstance(config.action_spec, ActionDiscreteTargetsConfig)
    assert config.action_spec.max_per_planet_launches == 1
    assert config.action_spec.min_fleet_size == 4
    assert config.action_spec.targeting_mode == "stop_bad_launch"

    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(),
        action_spec=config.action_spec,
        pin_memory=False,
    )
    obs = env.reset()

    assert obs.action_mask.can_act.shape == (
        1,
        4,
        ACTION_ENTITY_SLOTS,
        ACTION_ENTITY_SLOTS,
    )
    assert obs.action_mask.max_launch.shape == (1, 4, ACTION_ENTITY_SLOTS)
    assert obs.action_mask.max_launch[~obs.action_mask.can_act.any(dim=-1)].eq(0).all()


def test_discrete_targets_step_uses_int_target_tensor() -> None:
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
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

    obs, rewards, dones, episode_metrics = env.step(
        DiscreteTargetActions(launch=launch, target=target, ships=ships)
    )

    assert obs.action_mask.can_act.shape == (
        1,
        4,
        ACTION_ENTITY_SLOTS,
        ACTION_ENTITY_SLOTS,
    )
    assert rewards.shape == (1, 4)
    assert dones.shape == (1, 4)
    assert episode_metrics == {}

    with pytest.raises(ValueError, match="target must have dtype int64"):
        env.step(
            DiscreteTargetActions(
                launch=launch,
                target=target.astype(np.float32),
                ships=ships,
            )
        )


def test_discrete_target_bins_config_and_env_shapes() -> None:
    config = EnvConfig.model_validate(
        {
            "n_envs": 2,
            "action_spec": {
                "action_spec": "discrete_target_bins",
                "min_fleet_size": 4,
                "n_bins": 11,
                "targeting_mode": "anything_goes",
            },
            "pin_memory": False,
        }
    )
    assert isinstance(config.action_spec, ActionDiscreteTargetBinsConfig)
    assert config.action_spec.min_fleet_size == 4
    assert config.action_spec.n_bins == 11
    assert config.action_spec.targeting_mode == "anything_goes"

    env = VectorizedEnv(
        n_envs=2,
        obs_spec=EntityBasedConfig(),
        action_spec=config.action_spec,
        pin_memory=False,
    )
    obs = env.reset()

    assert obs.action_mask.can_act.shape == (
        2,
        4,
        ACTION_ENTITY_SLOTS,
        ACTION_ENTITY_SLOTS,
        11,
    )
    assert isinstance(obs.action_mask, DiscreteTargetBinActionMask)

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ActionDiscreteTargetBinsConfig(n_bins=11, max_per_planet_launches=1)  # type: ignore[call-arg]


def test_discrete_target_bins_step_uses_target_and_fleet_bin_bundle() -> None:
    action_spec = ActionDiscreteTargetBinsConfig(n_bins=8)
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=EntityBasedConfig(),
        action_spec=action_spec,
        pin_memory=False,
    )
    env.reset()
    shape = (2, 4, ACTION_ENTITY_SLOTS)
    actions = DiscreteTargetBinActions(
        target=torch.zeros(shape, dtype=torch.int64),
        fleet_bin=torch.zeros(shape, dtype=torch.int64),
    )

    obs, rewards, dones, episode_metrics = env.step(actions)

    assert obs.action_mask.can_act.shape == (
        2,
        4,
        ACTION_ENTITY_SLOTS,
        ACTION_ENTITY_SLOTS,
        8,
    )
    assert isinstance(obs.action_mask, DiscreteTargetBinActionMask)
    assert rewards.shape == (2, 4)
    assert dones.shape == (2, 4)
    assert episode_metrics == {}

    invalid_actions = DiscreteTargetBinActions(
        target=actions.target,
        fleet_bin=actions.fleet_bin.to(torch.float32),
    )
    with pytest.raises(ValueError, match=r"fleet_bin must have dtype torch\.int64"):
        env.step(invalid_actions)


def test_vectorized_env_writes_action_masks_for_alternate_specs() -> None:
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
        pin_memory=False,
    )
    obs = env.reset()
    target_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    target_bin_spec = ActionDiscreteTargetBinsConfig(n_bins=7)

    target_mask = env.action_mask_for_spec(target_spec)
    target_bin_mask = env.action_mask_for_spec(target_bin_spec)
    target_obs = env.observation_for_action_spec(target_spec)

    assert obs.action_mask.can_act.shape == (1, 4, ACTION_ENTITY_SLOTS)
    assert target_mask.can_act.shape == (
        1,
        4,
        ACTION_ENTITY_SLOTS,
        ACTION_ENTITY_SLOTS,
    )
    assert target_mask.max_launch.shape == (1, 4, ACTION_ENTITY_SLOTS)
    assert target_bin_mask.can_act.shape == (
        1,
        4,
        ACTION_ENTITY_SLOTS,
        ACTION_ENTITY_SLOTS,
        7,
    )
    assert target_obs.planets is env.observations.planets
    assert target_obs.action_mask.can_act.shape == target_mask.can_act.shape


def test_vectorized_env_reuses_cached_observation_only_for_same_encoding_spec() -> None:
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(max_per_planet_launches=1, min_fleet_size=6),
        pin_memory=False,
    )
    env.reset()

    same_encoding_obs = env.observation_for_spec(
        env.obs_spec,
        ActionDiscreteTargetsConfig(max_per_planet_launches=1, min_fleet_size=6),
    )
    different_encoding_obs = env.observation_for_spec(
        env.obs_spec,
        ActionDiscreteTargetsConfig(max_per_planet_launches=1, min_fleet_size=7),
    )

    assert same_encoding_obs.planets is env.observations.planets
    assert same_encoding_obs.fleets is env.observations.fleets
    assert different_encoding_obs.planets is not env.observations.planets
    assert different_encoding_obs.fleets is not env.observations.fleets
    assert torch.equal(
        different_encoding_obs.still_playing, env.observations.still_playing
    )


def test_vectorized_env_writes_observations_for_alternate_specs() -> None:
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 1),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
        two_player_weight=1.0,
        pin_memory=False,
    )
    obs = env.reset()
    obs_spec = EntityBasedExtV1Config(
        max_entities=MAX_PLANETS + MAX_COMETS + 2,
        ship_count_one_hot_max=5,
    )
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)

    alternate_obs = env.observation_for_spec(obs_spec, action_spec)

    assert alternate_obs.planets.shape == (
        1,
        MAX_PLANETS,
        obs_spec.planet_channels,
    )
    assert alternate_obs.fleets.shape == (
        1,
        obs_spec.max_fleets,
        obs_spec.fleet_channels,
    )
    assert alternate_obs.entity_mask.shape == (1, obs_spec.max_entities)
    assert alternate_obs.action_mask.can_act.shape == (
        1,
        4,
        ACTION_ENTITY_SLOTS,
        ACTION_ENTITY_SLOTS,
    )
    assert torch.equal(alternate_obs.still_playing, obs.still_playing)


def test_step_decoded_actions_validates_shapes() -> None:
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
        pin_memory=False,
    )
    env.reset()
    actions = DecodedLaunchActions(
        valid=torch.zeros((1, 4, 1), dtype=torch.int64),
        from_planet_id=torch.zeros((1, 4, 1), dtype=torch.int64),
        angle=torch.zeros((1, 4, 1), dtype=torch.float32),
        ships=torch.zeros((1, 4, 1), dtype=torch.int64),
    )

    with pytest.raises(ValueError, match=r"valid must have dtype torch\.bool"):
        env.step_decoded_actions(actions)


def test_step_decoded_actions_accepts_mixed_source_action_specs() -> None:
    pure_spec = ActionPureConfig(max_per_planet_launches=1)
    target_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    env = VectorizedEnv(
        n_envs=1,
        obs_spec=EntityBasedConfig(),
        action_spec=pure_spec,
        pin_memory=False,
    )
    obs = env.reset()
    active_players = torch.nonzero(obs.still_playing[0], as_tuple=False).flatten()
    pure_player = int(active_players[0].item())
    target_player = int(active_players[1].item())

    pure_shape = (1, 4, ACTION_ENTITY_SLOTS, 1)
    pure_launch = torch.zeros(pure_shape, dtype=torch.bool)
    pure_angle = torch.zeros(pure_shape, dtype=torch.float32)
    pure_ships = torch.zeros(pure_shape, dtype=torch.int64)
    pure_source = int(
        torch.nonzero(obs.action_mask.can_act[0, pure_player], as_tuple=False)[0]
    )
    pure_launch[0, pure_player, pure_source, 0] = True
    pure_ships[0, pure_player, pure_source, 0] = pure_spec.min_fleet_size
    pure_decoded = env.decode_actions(
        PureActions(launch=pure_launch, angle=pure_angle, ships=pure_ships),
        action_spec=pure_spec,
    )

    target_mask = env.action_mask_for_spec(target_spec)
    target_shape = (1, 4, ACTION_ENTITY_SLOTS, 1)
    target_launch = torch.zeros(target_shape, dtype=torch.bool)
    target = torch.zeros(target_shape, dtype=torch.int64)
    target_ships = torch.zeros(target_shape, dtype=torch.int64)
    target_source, target_index = torch.nonzero(
        target_mask.can_act[0, target_player],
        as_tuple=False,
    )[0].tolist()
    target_launch[0, target_player, target_source, 0] = True
    target[0, target_player, target_source, 0] = target_index
    target_ships[0, target_player, target_source, 0] = target_spec.min_fleet_size
    target_decoded = env.decode_actions(
        DiscreteTargetActions(
            launch=target_launch,
            target=target,
            ships=target_ships,
        ),
        action_spec=target_spec,
    )

    use_pure = torch.zeros((1, 4, 1), dtype=torch.bool)
    use_pure[0, pure_player, 0] = True
    mixed = DecodedLaunchActions(
        valid=torch.where(use_pure, pure_decoded.valid, target_decoded.valid),
        from_planet_id=torch.where(
            use_pure,
            pure_decoded.from_planet_id,
            target_decoded.from_planet_id,
        ),
        angle=torch.where(use_pure, pure_decoded.angle, target_decoded.angle),
        ships=torch.where(use_pure, pure_decoded.ships, target_decoded.ships),
    )

    obs, rewards, dones, episode_metrics = env.step_decoded_actions(mixed)

    assert obs.action_mask.can_act.shape == (1, 4, ACTION_ENTITY_SLOTS)
    assert rewards.shape == (1, 4)
    assert dones.shape == (1, 4)
    assert episode_metrics == {}


def test_actions_to_kaggle_converts_pure_model_actions() -> None:
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    shape = (1, 4, ACTION_ENTITY_SLOTS, action_spec.max_per_planet_launches)
    launch = torch.zeros(shape, dtype=torch.bool)
    angle = torch.zeros(shape, dtype=torch.float32)
    ships = torch.zeros(shape, dtype=torch.int64)
    launch[0, 0, 0, 0] = True
    angle[0, 0, 0, 0] = 1.25
    ships[0, 0, 0, 0] = action_spec.min_fleet_size

    actions = actions_to_kaggle(
        _python_obs(planets=[[0, 0, 25.0, 50.0, 2.0, 10, 3]]),
        0,
        PureActions(launch=launch, angle=angle, ships=ships),
        action_spec=action_spec,
    )

    assert actions == [[0, pytest.approx(1.25), action_spec.min_fleet_size]]
    assert type(actions[0][0]) is int
    assert type(actions[0][2]) is int


def test_actions_to_kaggle_converts_discrete_target_model_actions() -> None:
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    shape = (1, 4, ACTION_ENTITY_SLOTS, action_spec.max_per_planet_launches)
    launch = torch.zeros(shape, dtype=torch.bool)
    target = torch.zeros(shape, dtype=torch.int64)
    ships = torch.zeros(shape, dtype=torch.int64)
    launch[0, 0, 0, 0] = True
    target[0, 0, 0, 0] = 1
    ships[0, 0, 0, 0] = action_spec.min_fleet_size

    actions = actions_to_kaggle(
        _python_obs(
            planets=[
                [0, 0, 25.0, 80.0, 2.0, 10, 3],
                [1, -1, 75.0, 80.0, 2.0, 10, 3],
            ]
        ),
        0,
        DiscreteTargetActions(launch=launch, target=target, ships=ships),
        action_spec=action_spec,
    )

    assert len(actions) == 1
    assert actions[0][0] == 0
    assert np.isfinite(actions[0][1])
    assert actions[0][2] == action_spec.min_fleet_size
    assert type(actions[0][0]) is int
    assert type(actions[0][2]) is int


def test_actions_to_kaggle_respects_discrete_targeting_mode() -> None:
    obs = _python_obs(
        planets=[
            [0, 0, 0.0, 50.0, 2.0, 100, 3],
            [1, -1, 100.0, 50.0, 2.0, 10, 3],
        ]
    )
    shape = (1, 4, ACTION_ENTITY_SLOTS, 1)
    launch = torch.zeros(shape, dtype=torch.bool)
    target = torch.zeros(shape, dtype=torch.int64)
    ships = torch.zeros(shape, dtype=torch.int64)
    launch[0, 0, 0, 0] = True
    target[0, 0, 0, 0] = 1
    ships[0, 0, 0, 0] = 100
    actions = DiscreteTargetActions(launch=launch, target=target, ships=ships)

    stop_bad_launch = actions_to_kaggle(
        obs,
        0,
        actions,
        action_spec=ActionDiscreteTargetsConfig(
            max_per_planet_launches=1,
            targeting_mode="stop_bad_launch",
        ),
    )
    anything_goes = actions_to_kaggle(
        obs,
        0,
        actions,
        action_spec=ActionDiscreteTargetsConfig(
            max_per_planet_launches=1,
            targeting_mode="anything_goes",
        ),
    )

    assert stop_bad_launch == []
    assert len(anything_goes) == 1
    assert anything_goes[0][0] == 0
    assert anything_goes[0][2] == 100
    assert type(anything_goes[0][0]) is int
    assert type(anything_goes[0][2]) is int


def test_actions_to_kaggle_converts_discrete_target_bin_actions() -> None:
    action_spec = ActionDiscreteTargetBinsConfig(n_bins=11)
    shape = (1, 4, ACTION_ENTITY_SLOTS)
    actions = DiscreteTargetBinActions(
        target=torch.zeros(shape, dtype=torch.int64),
        fleet_bin=torch.zeros(shape, dtype=torch.int64),
    )
    actions.target[0, 0, 0] = 1
    actions.fleet_bin[0, 0, 0] = 10

    kaggle_actions = actions_to_kaggle(
        _python_obs(
            planets=[
                [0, 0, 25.0, 80.0, 2.0, 10, 3],
                [1, -1, 75.0, 80.0, 2.0, 10, 3],
            ]
        ),
        0,
        actions,
        action_spec=action_spec,
    )

    assert len(kaggle_actions) == 1
    assert kaggle_actions[0][0] == 0
    assert np.isfinite(kaggle_actions[0][1])
    assert kaggle_actions[0][2] == 10
    assert type(kaggle_actions[0][0]) is int
    assert type(kaggle_actions[0][2]) is int


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
    ) = _encoded_python_observation(
        _python_obs(planets=[[0, 0, 25.0, 75.0, 2.0, 2, 3]]),
        obs_spec=EntityBasedConfig(),
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
    env_index, player, entity = torch.nonzero(obs.action_mask.can_act, as_tuple=False)[
        0
    ].tolist()
    shape = (1, 4, ACTION_ENTITY_SLOTS, action_spec.max_per_planet_launches)
    launch = np.zeros(shape, dtype=np.bool_)
    angle = np.zeros(shape, dtype=np.float32)
    ships = np.zeros(shape, dtype=np.int64)
    launch[env_index, player, entity, 0] = True
    ships[env_index, player, entity, 0] = action_spec.min_fleet_size - 1

    with pytest.raises(ValueError, match="ships must be >= 3"):
        env.step(PureActions(launch=launch, angle=angle, ships=ships))


def test_python_observation_encoder_requires_missing_keys() -> None:
    obs = _python_obs()
    del obs["comets"]

    with pytest.raises(KeyError, match="comets"):
        encode_python_observation(
            obs,
            obs_spec=EntityBasedConfig(),
            action_spec=ActionPureConfig(),
        )


@pytest.mark.parametrize("field", ["step", "episode_steps"])
@pytest.mark.parametrize("value", [1.5, "1", True])
def test_python_observation_encoder_rejects_non_integer_globals(
    field: str, value: object
) -> None:
    with pytest.raises(TypeError, match=rf"obs\['{field}'\] must be an integer"):
        encode_python_observation(
            _python_obs(**{field: value}),
            obs_spec=EntityBasedConfig(),
            action_spec=ActionPureConfig(),
        )


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
    ) = _encoded_python_observation(
        _python_obs(
            planets=[
                [0, 0, 25.0, 75.0, 2.0, 10, 3],
                [1, -1, 75.0, 75.0, 2.0, 10, 3],
            ]
        ),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionDiscreteTargetsConfig(),
    )

    assert entity_mask[:2].tolist() == [True, True]
    assert not entity_mask[MAX_PLANETS:ACTION_ENTITY_SLOTS].any()
    assert can_act.shape == (4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS)
    assert not can_act[0, 0, 0]
    assert can_act[0, 0, 1]
    assert not can_act[0, 0, 2]
    assert max_launch[0, 0] == 10


def test_python_observation_encoder_writes_discrete_target_bin_mask() -> None:
    (
        _planets,
        _orbiting_planets,
        _fleets,
        _comets,
        entity_mask,
        _global_features,
        can_act,
        max_launch,
    ) = _encoded_python_observation(
        _python_obs(
            planets=[
                [0, 0, 25.0, 75.0, 2.0, 10, 3],
                [1, -1, 75.0, 75.0, 2.0, 10, 3],
            ]
        ),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionDiscreteTargetBinsConfig(n_bins=11),
    )

    assert entity_mask[:2].tolist() == [True, True]
    assert can_act.shape == (4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS, 11)
    assert max_launch is None
    assert np.nonzero(can_act[0, 0, 1])[0].tolist() == [0, 6, 7, 8, 9, 10]
    assert not can_act[0, 0, 0].any()
    assert not can_act[0, 0, 2].any()


def test_python_observation_encoder_masks_statically_obstructed_targets() -> None:
    (
        _planets,
        _orbiting_planets,
        _fleets,
        _comets,
        _entity_mask,
        _global_features,
        can_act,
        max_launch,
    ) = _encoded_python_observation(
        _python_obs(
            planets=[
                [0, 0, 0.0, 50.0, 2.0, 10, 3],
                [1, -1, 100.0, 50.0, 2.0, 10, 3],
                [2, -1, 100.0, 80.0, 2.0, 10, 3],
            ]
        ),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionDiscreteTargetsConfig(),
    )

    assert not can_act[0, 0, 1]
    assert can_act[0, 0, 2]
    assert max_launch[0, 0] == 10


@pytest.mark.parametrize("targeting_mode", ["anything_goes", "stop_bad_launch"])
def test_python_observation_encoder_loose_target_modes_do_not_mask_sun_targets(
    targeting_mode: str,
) -> None:
    (
        _planets,
        _orbiting_planets,
        _fleets,
        _comets,
        _entity_mask,
        _global_features,
        can_act,
        max_launch,
    ) = _encoded_python_observation(
        _python_obs(
            planets=[
                [0, 0, 0.0, 50.0, 2.0, 10, 3],
                [1, -1, 100.0, 50.0, 2.0, 10, 3],
            ]
        ),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionDiscreteTargetsConfig(targeting_mode=targeting_mode),
    )

    assert can_act[0, 0, 1]
    assert not can_act[0, 0, 0]
    assert not can_act[0, 0, 2]
    assert max_launch[0, 0] == 10


def test_python_discrete_observation_clears_max_launch_without_targets() -> None:
    (
        _planets,
        _orbiting_planets,
        _fleets,
        _comets,
        _entity_mask,
        _global_features,
        can_act,
        max_launch,
    ) = _encoded_python_observation(
        _python_obs(
            planets=[
                [0, 0, 0.0, 50.0, 2.0, 10, 3],
                [1, -1, 100.0, 50.0, 2.0, 10, 3],
            ]
        ),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionDiscreteTargetsConfig(),
    )

    assert not can_act[0, 0].any()
    assert max_launch[0, 0] == 0


def test_python_observation_encoder_matches_rl_schema_and_masks() -> None:
    encoded = encode_python_observation(
        _python_obs(
            step=50,
            angular_velocity=0.05,
            planets=[[0, -1, 25.0, 75.0, 2.0, 50, 3]],
            fleets=[[1, 0, 10.0, 20.0, np.pi / 2, 0, 25]],
        ),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(),
    )
    assert isinstance(encoded, ObsBatch)
    assert encoded.planets.shape == (1, MAX_PLANETS, PLANET_CHANNELS)

    (
        planets,
        orbiting_planets,
        fleets,
        comets,
        entity_mask,
        global_features,
        can_act,
        max_launch,
    ) = _encoded_python_observation(
        _python_obs(
            step=50,
            angular_velocity=0.05,
            planets=[[0, -1, 25.0, 75.0, 2.0, 50, 3]],
            fleets=[[1, 0, 10.0, 20.0, np.pi / 2, 0, 25]],
        ),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(),
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


def test_python_observation_encoder_marks_only_present_players_still_playing() -> None:
    encoded = encode_python_observation(
        _python_obs(
            planets=[
                [0, 0, 25.0, 75.0, 2.0, 50, 3],
                [1, 1, 75.0, 25.0, 2.0, 50, 3],
            ]
        ),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(),
    )

    assert encoded.still_playing.tolist() == [[True, True, False, False]]


def test_encode_python_observation_returns_rust_filtered_fleet_count() -> None:
    encoded = encode_python_observation_with_metrics(
        _python_obs(
            planets=[
                [0, 0, 25.0, 50.0, 2.0, 10, 3],
                [1, 1, 75.0, 50.0, 2.0, 10, 3],
            ],
            fleets=[
                [10, 0, 10.0, 10.0, 0.0, 0, 1],
                [11, 1, 20.0, 20.0, 0.0, 0, 6],
                [12, 2, 30.0, 30.0, 0.0, 0, 8],
            ],
        ),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(min_fleet_size=6),
        fleet_filter_min_size=8,
    )

    assert encoded.filtered_fleets == 2


def test_encode_python_observation_ext_v2_player_features_include_filtered_fleets() -> (
    None
):
    encoded = encode_python_observation_with_metrics(
        _python_obs(
            planets=[
                [0, 0, 25.0, 50.0, 2.0, 10, 3],
                [1, 1, 75.0, 50.0, 2.0, 20, 3],
            ],
            fleets=[
                [10, 0, 10.0, 10.0, 0.0, 0, 5],
                [11, 1, 20.0, 20.0, 0.0, 1, 6],
                [12, 1, 30.0, 30.0, 0.0, 1, 8],
            ],
        ),
        obs_spec=EntityBasedExtV2Config(max_entities=MAX_PLANETS + MAX_COMETS + 3),
        action_spec=ActionPureConfig(min_fleet_size=6),
        fleet_filter_min_size=8,
    )

    assert encoded.filtered_fleets == 2
    assert encoded.obs.player_features is not None
    assert encoded.obs.entity_mask[0, ACTION_ENTITY_SLOTS:].tolist() == [
        True,
        False,
        False,
    ]
    assert encoded.obs.fleets[0, 0, 8].item() == pytest.approx(8 / 500)

    player_features = encoded.obs.player_features[0].numpy()
    assert player_features[0, 3] == pytest.approx((10 + 5) / 5000)
    assert player_features[0, 4] == pytest.approx(math.log1p(10 + 5) / math.log(1000))
    assert player_features[0, 9] == pytest.approx(5 / 5000)
    assert player_features[0, 10] == pytest.approx(math.log1p(5) / math.log(1000))
    assert player_features[0, 13] == pytest.approx(1 / 100)

    assert player_features[1, 3] == pytest.approx((20 + 6 + 8) / 5000)
    assert player_features[1, 4] == pytest.approx(
        math.log1p(20 + 6 + 8) / math.log(1000)
    )
    assert player_features[1, 9] == pytest.approx((6 + 8) / 5000)
    assert player_features[1, 10] == pytest.approx(math.log1p(6 + 8) / math.log(1000))
    assert player_features[1, 13] == pytest.approx(2 / 100)


def test_encode_python_observation_cross_attn_routes_fleet_arrivals() -> None:
    encoded = encode_python_observation(
        _python_obs(
            planets=[
                [0, 0, 10.0, 70.0, 2.0, 10, 3],
                [1, 1, 50.0, 70.0, 2.0, 10, 3],
            ],
            fleets=[
                [10, 0, 30.0, 70.0, 0.0, 0, 1000],
                [11, 1, 99.0, 10.0, 0.0, 1, 1000],
            ],
        ),
        obs_spec=EntityBasedCrossAttnV1Config(),
        action_spec=ActionPureConfig(min_fleet_size=1),
    )

    assert encoded.fleet_target is not None
    assert encoded.target_incoming_features is not None
    assert encoded.fleet_target[0, 0].item() == 1
    assert encoded.fleet_target[0, 1].item() == -1
    assert encoded.entity_mask[0, ACTION_ENTITY_SLOTS].item()
    assert not encoded.entity_mask[0, ACTION_ENTITY_SLOTS + 1].item()
    eta_start = 4 + 2 + 22
    assert encoded.fleets[0, 0, eta_start + 2].item() == pytest.approx(1.0)
    assert encoded.target_incoming_features[0, 1, 2].item() == pytest.approx(0.01)
    assert encoded.target_incoming_features[0, 1, 16 + 2].item() == pytest.approx(0.2)


def test_encode_entity_based_filters_small_fleets_and_keeps_stranded_players() -> None:
    planets_in = np.array(
        [
            [0, 0, 25.0, 50.0, 2.0, 10, 3],
            [1, 2, 75.0, 50.0, 2.0, 10, 3],
        ],
        dtype=np.float64,
    )
    fleets_in = np.array(
        [
            [10, 0, 10.0, 10.0, 0.0, 9, 5],
            [11, 1, 20.0, 20.0, 0.0, 8, 4],
            [12, 1, 30.0, 30.0, 0.0, 7, 5],
            [13, 2, 40.0, 40.0, 0.0, 9, 3],
            [14, 3, 50.0, 50.0, 0.0, 0, 5],
        ],
        dtype=np.float64,
    )
    comet_planet_ids = np.full((0, MAX_COMETS), -1.0, dtype=np.float64)
    comet_path_indices = np.zeros(0, dtype=np.float64)
    comet_path_lengths = np.zeros((0, MAX_COMETS), dtype=np.float64)
    comet_paths = np.zeros(
        (0, MAX_COMETS, MAX_COMET_PATH_LENGTH, 2),
        dtype=np.float64,
    )

    (
        _planets,
        _orbiting_planets,
        fleets,
        _comets,
        entity_mask,
        _global_features,
        _can_act,
        _target_can_act,
        _max_launch,
        filtered_fleets,
    ) = encode_entity_based(
        planets_in,
        planets_in,
        fleets_in,
        comet_planet_ids,
        comet_path_indices,
        comet_path_lengths,
        comet_paths,
        0.025,
        min_fleet_size=6,
        max_entities=MAX_PLANETS + MAX_COMETS + 5,
    )

    fleet_mask = entity_mask[ACTION_ENTITY_SLOTS:]
    assert filtered_fleets == 3
    np.testing.assert_array_equal(
        fleet_mask,
        np.array([True, True, False, False, False]),
    )
    np.testing.assert_allclose(
        fleets[fleet_mask, 4],
        np.array([-0.4, 0.0], dtype=np.float32),
        atol=1e-6,
    )


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
        target_can_act,
        max_launch,
        filtered_fleets,
    ) = encode_entity_based(
        planets_in,
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
    expected_fleet_mask[:3] = True
    expected_comet_mask = np.array([True, True, False, False])
    planet_mask = entity_mask[:MAX_PLANETS]
    comet_mask = entity_mask[MAX_PLANETS:ACTION_ENTITY_SLOTS]
    fleet_mask = entity_mask[ACTION_ENTITY_SLOTS:]
    np.testing.assert_array_equal(planet_mask, expected_planet_mask)
    np.testing.assert_array_equal(fleet_mask, expected_fleet_mask)
    np.testing.assert_array_equal(comet_mask, expected_comet_mask)
    assert filtered_fleets == 1
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

    speed_27 = normalized_fleet_speed(27)
    speed_64 = normalized_fleet_speed(64)
    speed_125 = normalized_fleet_speed(125)
    base_expected_fleets = np.array(
        [
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
            for row, ships in zip(base_expected_fleets, [27, 64, 125], strict=True)
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
    assert target_can_act.shape == (4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS)
    assert not target_can_act[0, 0, 0]
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
            _python_obs(**{field: value}),
            obs_spec=EntityBasedConfig(),
            action_spec=ActionPureConfig(),
        )


@pytest.mark.parametrize("production", [0, 6, -1])
def test_python_observation_encoder_rejects_invalid_planet_production(
    production: int,
) -> None:
    with pytest.raises(ValueError, match="planet production must be between 1 and 5"):
        encode_python_observation(
            _python_obs(planets=[[0, -1, 25.0, 75.0, 2.0, 50, production]]),
            obs_spec=EntityBasedConfig(),
            action_spec=ActionPureConfig(),
        )


def test_python_observation_encoder_keeps_largest_fleets_first() -> None:
    spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 1)

    _, _, fleets, _, entity_mask, _, _, _ = _encoded_python_observation(
        _python_obs(
            fleets=[
                [1, 0, 10.0, 20.0, 0.0, 0, 5],
                [2, 1, 30.0, 40.0, 0.0, 0, 20],
            ]
        ),
        obs_spec=spec,
        action_spec=ActionPureConfig(),
    )
    fleet_mask = entity_mask[ACTION_ENTITY_SLOTS:]

    assert fleet_mask.tolist() == [True]
    assert fleets[0, 1] == 1
    assert fleets[0, 8] == pytest.approx(20 / 500.0)


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
    ) = _encoded_python_observation(
        _python_obs(
            planets=[[10, 2, 50.0, 50.0, 1.0, 25, 1]],
            comets=[
                {
                    "planet_ids": [10],
                    "paths": [path],
                    "path_index": 1,
                }
            ],
        ),
        obs_spec=EntityBasedConfig(),
        action_spec=ActionPureConfig(),
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


@pytest.mark.parametrize(
    ("comets", "message"),
    [
        (
            [
                {
                    "planet_ids": [1, 2],
                    "paths": [[[0.0, 0.0]]],
                    "path_index": 0,
                }
            ],
            "comet planet_ids and paths must have the same length",
        ),
        (
            [
                {
                    "planet_ids": [1, 2, 3, 4, 5],
                    "paths": [[[0.0, 0.0]]] * 5,
                    "path_index": 0,
                }
            ],
            "comet groups must have at most 4 paths",
        ),
        (
            [
                {
                    "planet_ids": [1, 2, 3],
                    "paths": [[[0.0, 0.0]]] * 3,
                    "path_index": 0,
                },
                {
                    "planet_ids": [4, 5],
                    "paths": [[[0.0, 0.0]]] * 2,
                    "path_index": 0,
                },
            ],
            "observations must have at most 4 active comets",
        ),
        (
            [
                {
                    "planet_ids": [1],
                    "paths": [[[0.0, 0.0]] * (MAX_COMET_PATH_LENGTH + 1)],
                    "path_index": 0,
                }
            ],
            "comet paths must have at most 40 points",
        ),
    ],
)
def test_python_observation_encoder_rejects_comet_truncation(
    comets: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        encode_python_observation(
            _python_obs(comets=comets),
            obs_spec=EntityBasedConfig(),
            action_spec=ActionPureConfig(),
        )
