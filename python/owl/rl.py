from __future__ import annotations

import warnings
from typing import Annotated, Any, Literal, SupportsFloat, SupportsInt, cast

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator

from owl.config import BaseConfig
from owl.rs import RlVecEnv as _RustRlVecEnv
from owl.rs import encode_obs_v1, rl_obs_constants

(
    MAX_PLANETS,
    MAX_COMETS,
    MAX_COMET_PATH_LENGTH,
    ACTION_ENTITY_SLOTS,
    DEFAULT_MAX_ENTITIES,
    PLANET_CHANNELS,
    FLEET_CHANNELS,
    COMET_CHANNELS,
    GLOBAL_CHANNELS,
) = rl_obs_constants()


class ObsV1Config(BaseConfig):
    obs_spec: Literal["obs_v1"] = "obs_v1"
    max_entities: int = Field(default=DEFAULT_MAX_ENTITIES, gt=MAX_PLANETS + MAX_COMETS)

    @property
    def max_planets(self) -> int:
        return MAX_PLANETS

    @property
    def max_fleets(self) -> int:
        return self.max_entities - (self.max_planets + MAX_COMETS)

    @property
    def planet_channels(self) -> int:
        return PLANET_CHANNELS

    @property
    def fleet_channels(self) -> int:
        return FLEET_CHANNELS

    @property
    def max_comets(self) -> int:
        return MAX_COMETS

    @property
    def max_comet_path_length(self) -> int:
        return MAX_COMET_PATH_LENGTH

    @property
    def comet_channels(self) -> int:
        return COMET_CHANNELS

    @property
    def global_channels(self) -> int:
        return GLOBAL_CHANNELS


class ActionPureConfig(BaseConfig):
    """Pure action spec.

    Sharp edge: the action entity axis is ordered as all planet tokens first,
    then comet tokens appended at the end, matching the intended
    planet-plus-comet hidden-token concatenation order.
    """

    action_spec: Literal["pure"] = "pure"
    max_per_planet_launches: int = Field(default=3, ge=1, le=4)
    min_fleet_size: int = Field(default=1, ge=1)


class ActionDiscreteTargetsConfig(BaseConfig):
    """Discrete target action spec.

    The source axis uses the same planet-then-comet action entity slots as the
    pure spec. For each launched source, the target tensor selects an action
    entity slot by integer index.
    """

    action_spec: Literal["discrete_targets"] = "discrete_targets"
    max_per_planet_launches: int = Field(default=3, ge=1, le=4)
    min_fleet_size: int = Field(default=6, ge=1)


type ObsConfig = Annotated[ObsV1Config, Field(discriminator="obs_spec")]
type ActionConfig = Annotated[
    ActionPureConfig | ActionDiscreteTargetsConfig,
    Field(discriminator="action_spec"),
]

OUTER_PLAYER_SLOTS = 4


class EnvConfig(BaseConfig):
    n_envs: int = Field(default=2, ge=1)
    obs_spec: ObsConfig = Field(default_factory=ObsV1Config)
    action_spec: ActionConfig = Field(default_factory=ActionPureConfig)
    two_player_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    pin_memory: bool = True

    @field_validator("n_envs")
    @classmethod
    def _validate_even_env_count(cls, n_envs: int) -> int:
        if n_envs % 2 != 0:
            raise ValueError("n_envs must be even")
        return n_envs


class ObsBatch(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    planets: torch.Tensor
    fleets: torch.Tensor
    comets: torch.Tensor
    entity_mask: torch.Tensor
    still_playing: torch.Tensor
    global_features: torch.Tensor
    can_act: torch.Tensor
    max_launch: torch.Tensor


class VectorizedEnv:
    def __init__(
        self,
        *,
        n_envs: int,
        obs_spec: ObsConfig,
        action_spec: ActionConfig,
        two_player_weight: float = 0.5,
        pin_memory: bool = True,
    ) -> None:
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self._rust = _RustRlVecEnv(
            n_envs,
            two_player_weight,
            self.obs_spec.obs_spec,
            self.action_spec.action_spec,
            self.obs_spec.max_entities,
            self.action_spec.max_per_planet_launches,
            self.action_spec.min_fleet_size,
        )
        if pin_memory and not torch.cuda.is_available():
            warnings.warn(
                "pin_memory=True requires CUDA; proceeding without pinned memory",
                RuntimeWarning,
                stacklevel=2,
            )
            pin_memory = False
        self.n_envs = n_envs
        self.n_players = OUTER_PLAYER_SLOTS
        self.pin_memory_enabled = pin_memory
        self.observations = self._allocate_observations(pin_memory=pin_memory)
        self.rewards = torch.zeros(
            (n_envs, self.n_players), dtype=torch.float32, pin_memory=pin_memory
        )
        self.dones = torch.zeros(
            (n_envs, self.n_players), dtype=torch.bool, pin_memory=pin_memory
        )

        self._planet_obs_np = self.observations.planets.numpy()
        self._fleet_obs_np = self.observations.fleets.numpy()
        self._comet_obs_np = self.observations.comets.numpy()
        self._entity_mask_np = self.observations.entity_mask.numpy()
        self._still_playing_np = self.observations.still_playing.numpy()
        self._global_obs_np = self.observations.global_features.numpy()
        self._can_act_np = self.observations.can_act.numpy()
        self._max_launch_np = self.observations.max_launch.numpy()
        self._rewards_np = self.rewards.numpy()
        self._dones_np = self.dones.numpy()

    def reset(self) -> ObsBatch:
        self._rust.reset(
            self._planet_obs_np,
            self._fleet_obs_np,
            self._comet_obs_np,
            self._entity_mask_np,
            self._still_playing_np,
            self._global_obs_np,
            self._can_act_np,
            self._max_launch_np,
        )
        return self.observations

    def step(
        self,
        launch: np.ndarray | torch.Tensor,
        action_value: np.ndarray | torch.Tensor,
        ships: np.ndarray | torch.Tensor,
    ) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
        launch_array = _actions_to_numpy(
            "launch", launch, dtype=np.bool_, torch_dtype=torch.bool
        )
        ship_array = _actions_to_numpy(
            "ships", ships, dtype=np.int64, torch_dtype=torch.int64
        )
        expected_shape = (
            self.n_envs,
            self.n_players,
            ACTION_ENTITY_SLOTS,
            self.action_spec.max_per_planet_launches,
        )
        _require_action_shape("launch", launch_array, expected_shape)
        _require_action_shape("ships", ship_array, expected_shape)

        if isinstance(self.action_spec, ActionPureConfig):
            angle_array = _actions_to_numpy(
                "angle",
                action_value,
                dtype=np.float32,
                torch_dtype=torch.float32,
            )
            _require_action_shape("angle", angle_array, expected_shape)
            episode_metrics = self._rust.step(
                launch_array,
                angle_array,
                ship_array,
                self._planet_obs_np,
                self._fleet_obs_np,
                self._comet_obs_np,
                self._entity_mask_np,
                self._still_playing_np,
                self._global_obs_np,
                self._can_act_np,
                self._max_launch_np,
                self._rewards_np,
                self._dones_np,
            )
        else:
            target_array = _actions_to_numpy(
                "target",
                action_value,
                dtype=np.int64,
                torch_dtype=torch.int64,
            )
            _require_action_shape("target", target_array, expected_shape)
            episode_metrics = self._rust.step_discrete_targets(
                launch_array,
                target_array,
                ship_array,
                self._planet_obs_np,
                self._fleet_obs_np,
                self._comet_obs_np,
                self._entity_mask_np,
                self._still_playing_np,
                self._global_obs_np,
                self._can_act_np,
                self._max_launch_np,
                self._rewards_np,
                self._dones_np,
            )
        return self.observations, self.rewards, self.dones, episode_metrics

    def _allocate_observations(self, *, pin_memory: bool) -> ObsBatch:
        can_act_shape = (
            (self.n_envs, self.n_players, ACTION_ENTITY_SLOTS)
            if isinstance(self.action_spec, ActionPureConfig)
            else (self.n_envs, self.n_players, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS)
        )
        return ObsBatch(
            planets=torch.zeros(
                (
                    self.n_envs,
                    self.obs_spec.max_planets,
                    self.obs_spec.planet_channels,
                ),
                dtype=torch.float32,
                pin_memory=pin_memory,
            ),
            fleets=torch.zeros(
                (
                    self.n_envs,
                    self.obs_spec.max_fleets,
                    self.obs_spec.fleet_channels,
                ),
                dtype=torch.float32,
                pin_memory=pin_memory,
            ),
            comets=torch.zeros(
                (
                    self.n_envs,
                    self.obs_spec.max_comets,
                    self.obs_spec.comet_channels,
                ),
                dtype=torch.float32,
                pin_memory=pin_memory,
            ),
            entity_mask=torch.zeros(
                (self.n_envs, self.obs_spec.max_entities),
                dtype=torch.bool,
                pin_memory=pin_memory,
            ),
            still_playing=torch.zeros(
                (self.n_envs, self.n_players),
                dtype=torch.bool,
                pin_memory=pin_memory,
            ),
            global_features=torch.zeros(
                (self.n_envs, self.obs_spec.global_channels),
                dtype=torch.float32,
                pin_memory=pin_memory,
            ),
            can_act=torch.zeros(
                can_act_shape,
                dtype=torch.bool,
                pin_memory=pin_memory,
            ),
            max_launch=torch.zeros(
                (self.n_envs, self.n_players, ACTION_ENTITY_SLOTS),
                dtype=torch.int64,
                pin_memory=pin_memory,
            ),
        )


def encode_python_observation(
    obs: dict[str, Any],
    obs_spec: ObsV1Config | None = None,
    action_spec: ActionConfig | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    spec = obs_spec or ObsV1Config()
    action = action_spec or ActionPureConfig()
    comet_planet_ids, comet_path_indices, comet_path_lengths, comet_paths = (
        _comets_to_arrays(obs.get("comets", []))
    )
    encoded = encode_obs_v1(
        _rows_to_array(obs.get("planets", []), name="planets"),
        _rows_to_array(obs.get("fleets", []), name="fleets"),
        comet_planet_ids,
        comet_path_indices,
        comet_path_lengths,
        comet_paths,
        float(cast(SupportsFloat, obs.get("angular_velocity", 0.0))),
        int(cast(SupportsInt, obs.get("step", 0))),
        int(cast(SupportsInt, obs.get("episode_steps", 500))),
        spec.max_entities,
        action.min_fleet_size,
    )
    if isinstance(action, ActionPureConfig):
        return encoded

    (
        planets,
        fleets,
        comets,
        entity_mask,
        global_features,
        source_can_act,
        max_launch,
    ) = encoded
    target_exists = entity_mask[:ACTION_ENTITY_SLOTS]
    source_target_can_act = source_can_act[:, :, None] & target_exists[None, None, :]
    source_indexes = np.arange(ACTION_ENTITY_SLOTS)
    source_target_can_act[:, source_indexes, source_indexes] = False
    return (
        planets,
        fleets,
        comets,
        entity_mask,
        global_features,
        source_target_can_act,
        max_launch,
    )


def _actions_to_numpy(
    name: str,
    actions: np.ndarray | torch.Tensor,
    *,
    dtype: Any,
    torch_dtype: torch.dtype,
) -> np.ndarray:
    if isinstance(actions, torch.Tensor):
        if actions.device.type != "cpu":
            raise ValueError("actions must be on CPU before stepping the Rust env")
        if actions.dtype != torch_dtype:
            raise ValueError(
                f"{name} must have dtype {torch_dtype}, got {actions.dtype}"
            )
        actions = actions.detach().numpy()
    elif isinstance(actions, np.ndarray):
        if actions.dtype != np.dtype(dtype):
            raise ValueError(
                f"{name} must have dtype {np.dtype(dtype).name}, got {actions.dtype}"
            )
    else:
        raise TypeError(f"{name} must be a NumPy array or Torch tensor")
    return np.ascontiguousarray(actions)


def _require_action_shape(
    name: str, action_array: np.ndarray, expected_shape: tuple[int, ...]
) -> None:
    if action_array.shape == expected_shape:
        return
    raise ValueError(
        f"{name} must have shape {expected_shape}, got {action_array.shape}"
    )


def _rows_to_array(rows: Any, *, name: str) -> np.ndarray:
    if not isinstance(rows, list):
        raise TypeError(f"obs['{name}'] must be a list")
    if not rows:
        return np.empty((0, 7), dtype=np.float64)
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"obs['{name}'] must have shape (n, 7)")
    return np.ascontiguousarray(array)


def _comets_to_arrays(
    comets: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(comets, list):
        raise TypeError("obs['comets'] must be a list")

    group_count = len(comets)
    planet_ids = np.full((group_count, MAX_COMETS), -1.0, dtype=np.float64)
    path_indices = np.zeros((group_count,), dtype=np.float64)
    path_lengths = np.zeros((group_count, MAX_COMETS), dtype=np.float64)
    paths = np.zeros(
        (group_count, MAX_COMETS, MAX_COMET_PATH_LENGTH, 2),
        dtype=np.float64,
    )

    for group_index, group in enumerate(comets):
        if not isinstance(group, dict):
            raise TypeError("comet groups must be dicts")
        raw_planet_ids = group.get("planet_ids", [])
        raw_paths = group.get("paths", [])
        if not isinstance(raw_planet_ids, list) or not isinstance(raw_paths, list):
            raise TypeError("comet groups need list planet_ids and paths")
        path_indices[group_index] = float(
            cast(SupportsFloat, group.get("path_index", -1))
        )

        for path_offset, (planet_id, raw_path) in enumerate(
            zip(raw_planet_ids, raw_paths, strict=True)
        ):
            if path_offset >= MAX_COMETS:
                break
            path_array = np.asarray(raw_path, dtype=np.float64)
            if path_array.ndim != 2 or path_array.shape[1] != 2:
                raise ValueError("comet paths must have shape (n, 2)")
            path_len = min(path_array.shape[0], MAX_COMET_PATH_LENGTH)
            planet_ids[group_index, path_offset] = float(cast(SupportsFloat, planet_id))
            path_lengths[group_index, path_offset] = path_len
            paths[group_index, path_offset, :path_len, :] = path_array[:path_len]

    return (
        np.ascontiguousarray(planet_ids),
        np.ascontiguousarray(path_indices),
        np.ascontiguousarray(path_lengths),
        np.ascontiguousarray(paths),
    )
