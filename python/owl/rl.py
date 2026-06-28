from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Integral
from typing import Annotated, Any, Literal, SupportsFloat, TypeAlias, cast

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator

from owl.config import BaseConfig
from owl.rs import RlVecEnv as _RustRlVecEnv
from owl.rs import (
    discrete_target_actions_to_kaggle as _discrete_target_actions_to_kaggle,
)
from owl.rs import (
    discrete_target_bin_actions_to_kaggle as _discrete_target_bin_actions_to_kaggle,
)
from owl.rs import encode_entity_based as encode_entity_based
from owl.rs import (
    encode_entity_based_cross_attn,
    encode_entity_based_with_player_features,
    rl_obs_constants,
    rl_obs_cross_attn_constants,
    rl_obs_ext_v2_constants,
)
from owl.rs import (
    pure_actions_to_kaggle as _pure_actions_to_kaggle,
)

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
(GLOBAL_EXT_V2_CHANNELS, PLAYER_FEATURE_CHANNELS) = rl_obs_ext_v2_constants()
(CROSS_ATTENTION_FLEET_CHANNELS, TARGET_INCOMING_CHANNELS) = (
    rl_obs_cross_attn_constants()
)


class EntityBasedBaseConfig(BaseConfig):
    max_entities: int = Field(default=DEFAULT_MAX_ENTITIES, gt=MAX_PLANETS + MAX_COMETS)

    @property
    def max_planets(self) -> int:
        return MAX_PLANETS

    @property
    def max_fleets(self) -> int:
        return self.max_entities - (self.max_planets + MAX_COMETS)

    @property
    def planet_channels(self) -> int:
        return PLANET_CHANNELS + self._planet_ship_count_one_hot_channels

    @property
    def fleet_channels(self) -> int:
        return FLEET_CHANNELS + self._fleet_ship_count_one_hot_channels

    @property
    def max_comets(self) -> int:
        return MAX_COMETS

    @property
    def comet_channels(self) -> int:
        return COMET_CHANNELS

    @property
    def global_channels(self) -> int:
        return GLOBAL_CHANNELS

    @property
    def player_feature_channels(self) -> int:
        return 0

    @property
    def target_incoming_channels(self) -> int:
        return 0

    @property
    def uses_cross_attention(self) -> bool:
        return False

    @property
    def ship_count_one_hot_encoder_max(self) -> int:
        return 0

    @property
    def _planet_ship_count_one_hot_channels(self) -> int:
        if self.ship_count_one_hot_encoder_max == 0:
            return 0
        return self.ship_count_one_hot_encoder_max + 1

    @property
    def _fleet_ship_count_one_hot_channels(self) -> int:
        return self.ship_count_one_hot_encoder_max


class EntityBasedConfig(EntityBasedBaseConfig):
    obs_spec: Literal["entity_based"] = "entity_based"


class EntityBasedExtV1Config(EntityBasedBaseConfig):
    obs_spec: Literal["entity_based_ext_v1"] = "entity_based_ext_v1"
    ship_count_one_hot_max: int = Field(default=50, ge=1)

    @property
    def ship_count_one_hot_encoder_max(self) -> int:
        return self.ship_count_one_hot_max


class EntityBasedExtV2Config(EntityBasedBaseConfig):
    obs_spec: Literal["entity_based_ext_v2"] = "entity_based_ext_v2"

    @property
    def global_channels(self) -> int:
        return GLOBAL_CHANNELS + GLOBAL_EXT_V2_CHANNELS

    @property
    def player_feature_channels(self) -> int:
        return PLAYER_FEATURE_CHANNELS


class EntityBasedCrossAttnV1Config(EntityBasedBaseConfig):
    obs_spec: Literal["entity_based_cross_attn_v1"] = "entity_based_cross_attn_v1"

    @property
    def fleet_channels(self) -> int:
        return CROSS_ATTENTION_FLEET_CHANNELS

    @property
    def global_channels(self) -> int:
        return GLOBAL_CHANNELS + GLOBAL_EXT_V2_CHANNELS

    @property
    def player_feature_channels(self) -> int:
        return PLAYER_FEATURE_CHANNELS

    @property
    def target_incoming_channels(self) -> int:
        return TARGET_INCOMING_CHANNELS

    @property
    def uses_cross_attention(self) -> bool:
        return True


class ActionPureConfig(BaseConfig):
    """Pure action spec.

    Sharp edge: the action entity axis is ordered as all planet tokens first,
    then comet tokens appended at the end, matching the intended
    planet-plus-comet hidden-token concatenation order.
    """

    action_spec: Literal["pure"] = "pure"
    max_per_planet_launches: int = Field(default=1, ge=1, le=1)
    min_fleet_size: int = Field(default=6, ge=1)


TargetingMode: TypeAlias = Literal["anything_goes", "stop_bad_launch", "full_mask"]


class ActionDiscreteTargetsConfig(BaseConfig):
    """Discrete target action spec.

    The source axis uses the same planet-then-comet action entity slots as the
    pure spec. For each launched source, the target tensor selects an action
    entity slot by integer index.
    """

    action_spec: Literal["discrete_targets"] = "discrete_targets"
    max_per_planet_launches: int = Field(default=1, ge=1, le=1)
    min_fleet_size: int = Field(default=6, ge=1)
    targeting_mode: TargetingMode = "full_mask"


class ActionDiscreteTargetBinsConfig(BaseConfig):
    """Discrete target plus discrete fleet-size-bin action spec."""

    action_spec: Literal["discrete_target_bins"] = "discrete_target_bins"
    min_fleet_size: int = Field(default=6, ge=1)
    n_bins: int = Field(ge=2)
    targeting_mode: TargetingMode = "full_mask"


ObsConfig: TypeAlias = Annotated[
    EntityBasedConfig
    | EntityBasedExtV1Config
    | EntityBasedExtV2Config
    | EntityBasedCrossAttnV1Config,
    Field(discriminator="obs_spec"),
]
ActionConfig: TypeAlias = Annotated[
    ActionPureConfig | ActionDiscreteTargetsConfig | ActionDiscreteTargetBinsConfig,
    Field(discriminator="action_spec"),
]

OUTER_PLAYER_SLOTS = 4


@dataclass
class PureActions:
    launch: torch.Tensor
    angle: torch.Tensor
    ships: torch.Tensor


@dataclass
class DiscreteTargetActions:
    launch: torch.Tensor
    target: torch.Tensor
    ships: torch.Tensor


@dataclass
class DiscreteTargetBinActions:
    target: torch.Tensor
    fleet_bin: torch.Tensor


ActionBundle: TypeAlias = PureActions | DiscreteTargetActions | DiscreteTargetBinActions
RewardMode: TypeAlias = Literal["win_loss", "ship_ratio"]


@dataclass
class DecodedLaunchActions:
    valid: torch.Tensor
    from_planet_id: torch.Tensor
    angle: torch.Tensor
    ships: torch.Tensor


@dataclass(frozen=True)
class PureActionMask:
    can_act: torch.Tensor
    max_launch: torch.Tensor


@dataclass(frozen=True)
class DiscreteTargetActionMask:
    can_act: torch.Tensor
    max_launch: torch.Tensor


@dataclass(frozen=True)
class DiscreteTargetBinActionMask:
    can_act: torch.Tensor


ActionMask: TypeAlias = (
    PureActionMask | DiscreteTargetActionMask | DiscreteTargetBinActionMask
)


class EnvConfig(BaseConfig):
    n_envs: int = Field(default=2, ge=1)
    obs_spec: ObsConfig = Field(default_factory=EntityBasedConfig)
    action_spec: ActionConfig = Field(default_factory=ActionPureConfig)
    two_player_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    reward_mode: RewardMode = "win_loss"
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
    orbiting_planets: torch.Tensor
    fleets: torch.Tensor
    fleet_target: torch.Tensor | None = None
    target_incoming_features: torch.Tensor | None = None
    comets: torch.Tensor
    entity_mask: torch.Tensor
    still_playing: torch.Tensor
    global_features: torch.Tensor
    action_mask: ActionMask
    player_features: torch.Tensor | None = None


@dataclass(frozen=True)
class EncodedPythonObservation:
    obs: ObsBatch
    filtered_fleets: int


class VectorizedEnv:
    def __init__(
        self,
        *,
        n_envs: int,
        obs_spec: ObsConfig,
        action_spec: ActionConfig,
        two_player_weight: float = 0.5,
        reward_mode: RewardMode = "win_loss",
        pin_memory: bool = True,
    ) -> None:
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.reward_mode = reward_mode
        self._rust = _RustRlVecEnv(
            n_envs,
            two_player_weight,
            self.obs_spec.obs_spec,
            self.action_spec.action_spec,
            self.obs_spec.max_entities,
            self.obs_spec.ship_count_one_hot_encoder_max,
            _max_per_planet_launches(self.action_spec),
            self.action_spec.min_fleet_size,
            _n_action_bins(self.action_spec),
            _targeting_mode(self.action_spec),
            reward_mode,
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
        self._orbiting_planet_obs_np = self.observations.orbiting_planets.numpy()
        self._fleet_obs_np = self.observations.fleets.numpy()
        self._fleet_target_np = (
            None
            if self.observations.fleet_target is None
            else self.observations.fleet_target.numpy()
        )
        self._target_incoming_features_np = (
            None
            if self.observations.target_incoming_features is None
            else self.observations.target_incoming_features.numpy()
        )
        self._comet_obs_np = self.observations.comets.numpy()
        self._entity_mask_np = self.observations.entity_mask.numpy()
        self._still_playing_np = self.observations.still_playing.numpy()
        self._global_obs_np = self.observations.global_features.numpy()
        self._player_features_np = (
            None
            if self.observations.player_features is None
            else self.observations.player_features.numpy()
        )
        action_mask = self.observations.action_mask
        self._can_act_np = action_mask.can_act.numpy()
        self._max_launch_np = (
            None
            if isinstance(action_mask, DiscreteTargetBinActionMask)
            else action_mask.max_launch.numpy()
        )
        self._rewards_np = self.rewards.numpy()
        self._dones_np = self.dones.numpy()

    def reset(self) -> ObsBatch:
        if isinstance(self.action_spec, ActionDiscreteTargetBinsConfig):
            self._rust.reset_discrete_target_bins(
                self._planet_obs_np,
                self._orbiting_planet_obs_np,
                self._fleet_obs_np,
                self._comet_obs_np,
                self._entity_mask_np,
                self._still_playing_np,
                self._global_obs_np,
                self._can_act_np,
                self._player_features_np,
                self._fleet_target_np,
                self._target_incoming_features_np,
            )
        else:
            assert self._max_launch_np is not None
            self._rust.reset(
                self._planet_obs_np,
                self._orbiting_planet_obs_np,
                self._fleet_obs_np,
                self._comet_obs_np,
                self._entity_mask_np,
                self._still_playing_np,
                self._global_obs_np,
                self._can_act_np,
                self._max_launch_np,
                self._player_features_np,
                self._fleet_target_np,
                self._target_incoming_features_np,
            )
        return self.observations

    def truncate_envs(self, truncate_mask: np.ndarray | torch.Tensor) -> ObsBatch:
        mask_array = _truncate_mask_to_numpy(truncate_mask, self.n_envs)
        self._rust.truncate_envs(
            mask_array,
            self._planet_obs_np,
            self._orbiting_planet_obs_np,
            self._fleet_obs_np,
            self._comet_obs_np,
            self._entity_mask_np,
            self._still_playing_np,
            self._global_obs_np,
            self._can_act_np,
            self._max_launch_np,
            self._player_features_np,
            self._fleet_target_np,
            self._target_incoming_features_np,
        )
        return self.observations

    def state_snapshot(self, env_index: int) -> dict[str, Any]:
        return self._rust.state_snapshot(env_index)

    def terminal_snapshot(self, env_index: int) -> dict[str, Any] | None:
        return self._rust.terminal_snapshot(env_index)

    def terminal_metrics(self, env_index: int) -> dict[str, float] | None:
        return self._rust.terminal_metrics(env_index)

    def action_mask_for_spec(self, action_spec: ActionConfig) -> ActionMask:
        can_act = torch.zeros(
            _can_act_shape(self.n_envs, self.n_players, action_spec),
            dtype=torch.bool,
            pin_memory=self.pin_memory_enabled,
        )
        if isinstance(action_spec, ActionDiscreteTargetBinsConfig):
            self._rust.write_action_mask_discrete_target_bins(
                action_spec.min_fleet_size,
                action_spec.n_bins,
                action_spec.targeting_mode,
                can_act.numpy(),
            )
            return DiscreteTargetBinActionMask(can_act=can_act)

        max_launch = torch.zeros(
            (self.n_envs, self.n_players, ACTION_ENTITY_SLOTS),
            dtype=torch.int64,
            pin_memory=self.pin_memory_enabled,
        )
        self._rust.write_action_mask(
            action_spec.action_spec,
            action_spec.min_fleet_size,
            0,
            _targeting_mode(action_spec),
            can_act.numpy(),
            max_launch.numpy(),
        )
        if isinstance(action_spec, ActionPureConfig):
            return PureActionMask(can_act=can_act, max_launch=max_launch)
        return DiscreteTargetActionMask(can_act=can_act, max_launch=max_launch)

    def observation_for_action_spec(self, action_spec: ActionConfig) -> ObsBatch:
        return ObsBatch(
            planets=self.observations.planets,
            orbiting_planets=self.observations.orbiting_planets,
            fleets=self.observations.fleets,
            fleet_target=self.observations.fleet_target,
            target_incoming_features=self.observations.target_incoming_features,
            comets=self.observations.comets,
            entity_mask=self.observations.entity_mask,
            still_playing=self.observations.still_playing,
            global_features=self.observations.global_features,
            player_features=self.observations.player_features,
            action_mask=self.action_mask_for_spec(action_spec),
        )

    def observation_for_spec(
        self,
        obs_spec: ObsConfig,
        action_spec: ActionConfig,
    ) -> ObsBatch:
        if self._uses_cached_observation(obs_spec, action_spec):
            return self.observation_for_action_spec(action_spec)

        obs = self._allocate_observations(
            pin_memory=self.pin_memory_enabled,
            obs_spec=obs_spec,
            action_spec=action_spec,
        )
        action_mask = obs.action_mask
        self._rust.write_observation(
            obs_spec.obs_spec,
            action_spec.action_spec,
            obs_spec.max_entities,
            obs_spec.ship_count_one_hot_encoder_max,
            action_spec.min_fleet_size,
            _n_action_bins(action_spec),
            _targeting_mode(action_spec),
            obs.planets.numpy(),
            obs.orbiting_planets.numpy(),
            obs.fleets.numpy(),
            obs.comets.numpy(),
            obs.entity_mask.numpy(),
            obs.still_playing.numpy(),
            obs.global_features.numpy(),
            action_mask.can_act.numpy(),
            None
            if isinstance(action_mask, DiscreteTargetBinActionMask)
            else action_mask.max_launch.numpy(),
            None if obs.player_features is None else obs.player_features.numpy(),
            None if obs.fleet_target is None else obs.fleet_target.numpy(),
            None
            if obs.target_incoming_features is None
            else obs.target_incoming_features.numpy(),
        )
        return obs

    def _uses_cached_observation(
        self,
        obs_spec: ObsConfig,
        action_spec: ActionConfig,
    ) -> bool:
        return (
            obs_spec == self.obs_spec
            and action_spec.min_fleet_size == self.action_spec.min_fleet_size
        )

    def decode_actions(
        self,
        actions: ActionBundle,
        *,
        action_spec: ActionConfig,
    ) -> DecodedLaunchActions:
        decoded = self._allocate_decoded_actions(
            max_actions=_max_decoded_actions(action_spec)
        )
        if isinstance(action_spec, ActionDiscreteTargetBinsConfig):
            if not isinstance(actions, DiscreteTargetBinActions):
                raise TypeError(
                    "discrete_target_bins requires DiscreteTargetBinActions"
                )
            target_array = _actions_to_numpy(
                "target", actions.target, dtype=np.int64, torch_dtype=torch.int64
            )
            fleet_bin_array = _actions_to_numpy(
                "fleet_bin",
                actions.fleet_bin,
                dtype=np.int64,
                torch_dtype=torch.int64,
            )
            expected_bin_shape = (self.n_envs, self.n_players, ACTION_ENTITY_SLOTS)
            _require_action_shape("target", target_array, expected_bin_shape)
            _require_action_shape("fleet_bin", fleet_bin_array, expected_bin_shape)
            self._rust.decode_discrete_target_bin_actions(
                target_array,
                fleet_bin_array,
                action_spec.min_fleet_size,
                action_spec.n_bins,
                action_spec.targeting_mode,
                decoded.valid.numpy(),
                decoded.from_planet_id.numpy(),
                decoded.angle.numpy(),
                decoded.ships.numpy(),
            )
            return decoded

        if isinstance(action_spec, ActionPureConfig):
            if not isinstance(actions, PureActions):
                raise TypeError("pure requires PureActions")
            launch_actions: PureActions | DiscreteTargetActions = actions
        elif isinstance(action_spec, ActionDiscreteTargetsConfig):
            if not isinstance(actions, DiscreteTargetActions):
                raise TypeError("discrete_targets requires DiscreteTargetActions")
            launch_actions = actions
        else:
            raise TypeError("unsupported action spec")

        launch_array = _actions_to_numpy(
            "launch", launch_actions.launch, dtype=np.bool_, torch_dtype=torch.bool
        )
        ship_array = _actions_to_numpy(
            "ships", launch_actions.ships, dtype=np.int64, torch_dtype=torch.int64
        )
        expected_launch_shape = (
            self.n_envs,
            self.n_players,
            ACTION_ENTITY_SLOTS,
            action_spec.max_per_planet_launches,
        )
        _require_action_shape("launch", launch_array, expected_launch_shape)
        _require_action_shape("ships", ship_array, expected_launch_shape)

        if isinstance(action_spec, ActionPureConfig):
            pure_actions = cast(PureActions, launch_actions)
            angle_array = _actions_to_numpy(
                "angle",
                pure_actions.angle,
                dtype=np.float32,
                torch_dtype=torch.float32,
            )
            _require_action_shape("angle", angle_array, expected_launch_shape)
            self._rust.decode_pure_actions(
                launch_array,
                angle_array,
                ship_array,
                action_spec.max_per_planet_launches,
                action_spec.min_fleet_size,
                decoded.valid.numpy(),
                decoded.from_planet_id.numpy(),
                decoded.angle.numpy(),
                decoded.ships.numpy(),
            )
            return decoded

        discrete_actions = cast(DiscreteTargetActions, launch_actions)
        target_array = _actions_to_numpy(
            "target", discrete_actions.target, dtype=np.int64, torch_dtype=torch.int64
        )
        _require_action_shape("target", target_array, expected_launch_shape)
        self._rust.decode_discrete_target_actions(
            launch_array,
            target_array,
            ship_array,
            action_spec.max_per_planet_launches,
            action_spec.min_fleet_size,
            action_spec.targeting_mode,
            decoded.valid.numpy(),
            decoded.from_planet_id.numpy(),
            decoded.angle.numpy(),
            decoded.ships.numpy(),
        )
        return decoded

    def step(
        self,
        actions: ActionBundle,
    ) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
        if isinstance(self.action_spec, ActionDiscreteTargetBinsConfig):
            if not isinstance(actions, DiscreteTargetBinActions):
                raise TypeError(
                    "discrete_target_bins requires DiscreteTargetBinActions"
                )
            target_array = _actions_to_numpy(
                "target", actions.target, dtype=np.int64, torch_dtype=torch.int64
            )
            fleet_bin_array = _actions_to_numpy(
                "fleet_bin",
                actions.fleet_bin,
                dtype=np.int64,
                torch_dtype=torch.int64,
            )
            target_bin_shape = (self.n_envs, self.n_players, ACTION_ENTITY_SLOTS)
            _require_action_shape("target", target_array, target_bin_shape)
            _require_action_shape("fleet_bin", fleet_bin_array, target_bin_shape)
            episode_metrics = self._rust.step_discrete_target_bins(
                target_array,
                fleet_bin_array,
                self._planet_obs_np,
                self._orbiting_planet_obs_np,
                self._fleet_obs_np,
                self._comet_obs_np,
                self._entity_mask_np,
                self._still_playing_np,
                self._global_obs_np,
                self._can_act_np,
                self._rewards_np,
                self._dones_np,
                self._player_features_np,
                self._fleet_target_np,
                self._target_incoming_features_np,
            )
            return self.observations, self.rewards, self.dones, episode_metrics

        if isinstance(self.action_spec, ActionPureConfig):
            if not isinstance(actions, PureActions):
                raise TypeError("pure requires PureActions")
            launch_actions: PureActions | DiscreteTargetActions = actions
        elif isinstance(self.action_spec, ActionDiscreteTargetsConfig):
            if not isinstance(actions, DiscreteTargetActions):
                raise TypeError("discrete_targets requires DiscreteTargetActions")
            launch_actions = actions
        else:
            raise TypeError("unsupported action spec")

        launch_array = _actions_to_numpy(
            "launch", launch_actions.launch, dtype=np.bool_, torch_dtype=torch.bool
        )
        ship_array = _actions_to_numpy(
            "ships", launch_actions.ships, dtype=np.int64, torch_dtype=torch.int64
        )
        expected_shape: tuple[int, ...] = (
            self.n_envs,
            self.n_players,
            ACTION_ENTITY_SLOTS,
            self.action_spec.max_per_planet_launches,
        )
        _require_action_shape("launch", launch_array, expected_shape)
        _require_action_shape("ships", ship_array, expected_shape)

        if isinstance(self.action_spec, ActionPureConfig):
            pure_actions = cast(PureActions, launch_actions)
            assert self._max_launch_np is not None
            angle_array = _actions_to_numpy(
                "angle",
                pure_actions.angle,
                dtype=np.float32,
                torch_dtype=torch.float32,
            )
            _require_action_shape("angle", angle_array, expected_shape)
            episode_metrics = self._rust.step(
                launch_array,
                angle_array,
                ship_array,
                self._planet_obs_np,
                self._orbiting_planet_obs_np,
                self._fleet_obs_np,
                self._comet_obs_np,
                self._entity_mask_np,
                self._still_playing_np,
                self._global_obs_np,
                self._can_act_np,
                self._max_launch_np,
                self._rewards_np,
                self._dones_np,
                self._player_features_np,
                self._fleet_target_np,
                self._target_incoming_features_np,
            )
        elif isinstance(self.action_spec, ActionDiscreteTargetsConfig):
            discrete_actions = cast(DiscreteTargetActions, launch_actions)
            assert self._max_launch_np is not None
            target_array = _actions_to_numpy(
                "target",
                discrete_actions.target,
                dtype=np.int64,
                torch_dtype=torch.int64,
            )
            _require_action_shape("target", target_array, expected_shape)
            episode_metrics = self._rust.step_discrete_targets(
                launch_array,
                target_array,
                ship_array,
                self._planet_obs_np,
                self._orbiting_planet_obs_np,
                self._fleet_obs_np,
                self._comet_obs_np,
                self._entity_mask_np,
                self._still_playing_np,
                self._global_obs_np,
                self._can_act_np,
                self._max_launch_np,
                self._rewards_np,
                self._dones_np,
                self._player_features_np,
                self._fleet_target_np,
                self._target_incoming_features_np,
            )
        else:
            raise TypeError(
                "discrete_target_bins requires a target/fleet_bin action bundle"
            )
        return self.observations, self.rewards, self.dones, episode_metrics

    def step_decoded_actions(
        self,
        actions: DecodedLaunchActions,
    ) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
        valid_array = _actions_to_numpy(
            "valid", actions.valid, dtype=np.bool_, torch_dtype=torch.bool
        )
        from_planet_id_array = _actions_to_numpy(
            "from_planet_id",
            actions.from_planet_id,
            dtype=np.int64,
            torch_dtype=torch.int64,
        )
        angle_array = _actions_to_numpy(
            "angle", actions.angle, dtype=np.float32, torch_dtype=torch.float32
        )
        ship_array = _actions_to_numpy(
            "ships", actions.ships, dtype=np.int64, torch_dtype=torch.int64
        )
        if valid_array.ndim != 3:
            raise ValueError(
                "valid must have shape (n_envs, n_players, max_actions), "
                f"got {valid_array.shape}"
            )
        expected_shape = (self.n_envs, self.n_players, valid_array.shape[2])
        if valid_array.shape[2] <= 0:
            raise ValueError("max_actions must be positive")
        _require_action_shape("valid", valid_array, expected_shape)
        _require_action_shape("from_planet_id", from_planet_id_array, expected_shape)
        _require_action_shape("angle", angle_array, expected_shape)
        _require_action_shape("ships", ship_array, expected_shape)
        episode_metrics = self._rust.step_decoded_actions(
            valid_array,
            from_planet_id_array,
            angle_array,
            ship_array,
            self._planet_obs_np,
            self._orbiting_planet_obs_np,
            self._fleet_obs_np,
            self._comet_obs_np,
            self._entity_mask_np,
            self._still_playing_np,
            self._global_obs_np,
            self._can_act_np,
            self._max_launch_np,
            self._rewards_np,
            self._dones_np,
            self._player_features_np,
            self._fleet_target_np,
            self._target_incoming_features_np,
        )
        return self.observations, self.rewards, self.dones, episode_metrics

    def _allocate_observations(
        self,
        *,
        pin_memory: bool,
        obs_spec: ObsConfig | None = None,
        action_spec: ActionConfig | None = None,
    ) -> ObsBatch:
        obs_spec = self.obs_spec if obs_spec is None else obs_spec
        action_spec = self.action_spec if action_spec is None else action_spec
        can_act_shape = _can_act_shape(self.n_envs, self.n_players, action_spec)
        can_act = torch.zeros(
            can_act_shape,
            dtype=torch.bool,
            pin_memory=pin_memory,
        )
        max_launch = (
            None
            if isinstance(action_spec, ActionDiscreteTargetBinsConfig)
            else torch.zeros(
                (self.n_envs, self.n_players, ACTION_ENTITY_SLOTS),
                dtype=torch.int64,
                pin_memory=pin_memory,
            )
        )
        if isinstance(action_spec, ActionPureConfig):
            action_mask: ActionMask = PureActionMask(
                can_act=can_act,
                max_launch=cast(torch.Tensor, max_launch),
            )
        elif isinstance(action_spec, ActionDiscreteTargetsConfig):
            action_mask = DiscreteTargetActionMask(
                can_act=can_act,
                max_launch=cast(torch.Tensor, max_launch),
            )
        else:
            action_mask = DiscreteTargetBinActionMask(can_act=can_act)
        return ObsBatch(
            planets=torch.zeros(
                (
                    self.n_envs,
                    obs_spec.max_planets,
                    obs_spec.planet_channels,
                ),
                dtype=torch.float32,
                pin_memory=pin_memory,
            ),
            orbiting_planets=torch.zeros(
                (self.n_envs, obs_spec.max_planets),
                dtype=torch.bool,
                pin_memory=pin_memory,
            ),
            fleets=torch.zeros(
                (
                    self.n_envs,
                    obs_spec.max_fleets,
                    obs_spec.fleet_channels,
                ),
                dtype=torch.float32,
                pin_memory=pin_memory,
            ),
            fleet_target=(
                None
                if not obs_spec.uses_cross_attention
                else torch.full(
                    (self.n_envs, obs_spec.max_fleets),
                    -1,
                    dtype=torch.int64,
                    pin_memory=pin_memory,
                )
            ),
            target_incoming_features=(
                None
                if not obs_spec.uses_cross_attention
                else torch.zeros(
                    (
                        self.n_envs,
                        ACTION_ENTITY_SLOTS,
                        obs_spec.target_incoming_channels,
                    ),
                    dtype=torch.float32,
                    pin_memory=pin_memory,
                )
            ),
            comets=torch.zeros(
                (
                    self.n_envs,
                    obs_spec.max_comets,
                    obs_spec.comet_channels,
                ),
                dtype=torch.float32,
                pin_memory=pin_memory,
            ),
            entity_mask=torch.zeros(
                (self.n_envs, obs_spec.max_entities),
                dtype=torch.bool,
                pin_memory=pin_memory,
            ),
            still_playing=torch.zeros(
                (self.n_envs, self.n_players),
                dtype=torch.bool,
                pin_memory=pin_memory,
            ),
            global_features=torch.zeros(
                (self.n_envs, obs_spec.global_channels),
                dtype=torch.float32,
                pin_memory=pin_memory,
            ),
            action_mask=action_mask,
            player_features=(
                None
                if obs_spec.player_feature_channels == 0
                else torch.zeros(
                    (
                        self.n_envs,
                        self.n_players,
                        obs_spec.player_feature_channels,
                    ),
                    dtype=torch.float32,
                    pin_memory=pin_memory,
                )
            ),
        )

    def _allocate_decoded_actions(self, *, max_actions: int) -> DecodedLaunchActions:
        shape = (self.n_envs, self.n_players, max_actions)
        return DecodedLaunchActions(
            valid=torch.zeros(
                shape, dtype=torch.bool, pin_memory=self.pin_memory_enabled
            ),
            from_planet_id=torch.zeros(
                shape, dtype=torch.int64, pin_memory=self.pin_memory_enabled
            ),
            angle=torch.zeros(
                shape, dtype=torch.float32, pin_memory=self.pin_memory_enabled
            ),
            ships=torch.zeros(
                shape, dtype=torch.int64, pin_memory=self.pin_memory_enabled
            ),
        )


def encode_python_observation(
    obs: dict[str, Any],
    obs_spec: EntityBasedBaseConfig,
    action_spec: ActionConfig,
    *,
    fleet_filter_min_size: int | None = None,
) -> ObsBatch:
    return encode_python_observation_with_metrics(
        obs,
        obs_spec,
        action_spec,
        fleet_filter_min_size=fleet_filter_min_size,
    ).obs


def encode_python_observation_with_metrics(
    obs: dict[str, Any],
    obs_spec: EntityBasedBaseConfig,
    action_spec: ActionConfig,
    *,
    fleet_filter_min_size: int | None = None,
) -> EncodedPythonObservation:
    (
        planets_in,
        initial_planets_in,
        fleets_in,
        comet_planet_ids,
        comet_path_indices,
        comet_path_lengths,
        comet_paths,
        angular_velocity,
        step,
        episode_steps,
    ) = _observation_arrays(obs)
    still_playing = _still_playing_from_arrays(
        planets_in,
        fleets_in,
        player=obs["player"],
    )
    if obs_spec.uses_cross_attention:
        cross_encoded = encode_entity_based_cross_attn(
            planets_in,
            initial_planets_in,
            fleets_in,
            comet_planet_ids,
            comet_path_indices,
            comet_path_lengths,
            comet_paths,
            angular_velocity,
            step,
            episode_steps,
            obs_spec.max_entities,
            action_spec.min_fleet_size,
            action_spec.min_fleet_size
            if fleet_filter_min_size is None
            else fleet_filter_min_size,
        )
        (
            planets,
            orbiting_planets,
            fleets,
            fleet_target,
            target_incoming_features,
            comets,
            entity_mask,
            global_features,
            player_features,
            source_can_act,
            source_target_can_act,
            max_launch,
            filtered_fleets,
        ) = cross_encoded
    else:
        entity_encoded = encode_entity_based_with_player_features(
            planets_in,
            initial_planets_in,
            fleets_in,
            comet_planet_ids,
            comet_path_indices,
            comet_path_lengths,
            comet_paths,
            angular_velocity,
            step,
            episode_steps,
            obs_spec.max_entities,
            action_spec.min_fleet_size,
            obs_spec.ship_count_one_hot_encoder_max,
            action_spec.min_fleet_size
            if fleet_filter_min_size is None
            else fleet_filter_min_size,
            obs_spec.player_feature_channels,
        )
        (
            planets,
            orbiting_planets,
            fleets,
            comets,
            entity_mask,
            global_features,
            player_features,
            source_can_act,
            source_target_can_act,
            max_launch,
            filtered_fleets,
        ) = entity_encoded
        fleet_target = None
        target_incoming_features = None
    if isinstance(action_spec, ActionPureConfig):
        return EncodedPythonObservation(
            obs=_encoded_observation_to_batch(
                (
                    planets,
                    orbiting_planets,
                    fleets,
                    fleet_target,
                    target_incoming_features,
                    comets,
                    entity_mask,
                    global_features,
                    None if obs_spec.player_feature_channels == 0 else player_features,
                    source_can_act,
                    max_launch,
                ),
                still_playing=still_playing,
            ),
            filtered_fleets=int(filtered_fleets),
        )
    if isinstance(action_spec, ActionDiscreteTargetBinsConfig):
        if action_spec.targeting_mode == "full_mask":
            target_can_act = source_target_can_act
        else:
            target_can_act = _loose_target_can_act(source_can_act, entity_mask)
        return EncodedPythonObservation(
            obs=_encoded_observation_to_batch(
                (
                    planets,
                    orbiting_planets,
                    fleets,
                    fleet_target,
                    target_incoming_features,
                    comets,
                    entity_mask,
                    global_features,
                    None if obs_spec.player_feature_channels == 0 else player_features,
                    _target_bin_can_act(
                        target_can_act,
                        max_launch,
                        min_fleet_size=action_spec.min_fleet_size,
                        n_bins=action_spec.n_bins,
                    ),
                    None,
                ),
                still_playing=still_playing,
            ),
            filtered_fleets=int(filtered_fleets),
        )
    if action_spec.targeting_mode == "full_mask":
        target_can_act = source_target_can_act
    else:
        target_can_act = _loose_target_can_act(source_can_act, entity_mask)
    target_max_launch = np.where(target_can_act.any(axis=-1), max_launch, 0)
    return EncodedPythonObservation(
        obs=_encoded_observation_to_batch(
            (
                planets,
                orbiting_planets,
                fleets,
                fleet_target,
                target_incoming_features,
                comets,
                entity_mask,
                global_features,
                None if obs_spec.player_feature_channels == 0 else player_features,
                target_can_act,
                target_max_launch,
            ),
            still_playing=still_playing,
        ),
        filtered_fleets=int(filtered_fleets),
    )


def actions_to_kaggle(
    obs: dict[str, Any],
    player: int,
    actions: ActionBundle,
    *,
    action_spec: ActionConfig,
) -> list[list[float]]:
    if isinstance(action_spec, ActionDiscreteTargetBinsConfig):
        if not isinstance(actions, DiscreteTargetBinActions):
            raise TypeError("discrete_target_bins requires DiscreteTargetBinActions")
        return _target_bin_actions_to_kaggle(
            obs,
            player,
            actions.target,
            actions.fleet_bin,
            action_spec=action_spec,
        )
    if isinstance(action_spec, ActionPureConfig):
        if not isinstance(actions, PureActions):
            raise TypeError("pure requires PureActions")
        launch_actions: PureActions | DiscreteTargetActions = actions
    elif isinstance(action_spec, ActionDiscreteTargetsConfig):
        if not isinstance(actions, DiscreteTargetActions):
            raise TypeError("discrete_targets requires DiscreteTargetActions")
        launch_actions = actions
    else:
        raise TypeError("unsupported action spec")

    (
        planets_in,
        initial_planets_in,
        fleets_in,
        comet_planet_ids,
        comet_path_indices,
        comet_path_lengths,
        comet_paths,
        angular_velocity,
        step,
        episode_steps,
    ) = _observation_arrays(obs)
    launch_array = _actions_to_numpy(
        "launch", launch_actions.launch, dtype=np.bool_, torch_dtype=torch.bool
    )
    ship_array = _actions_to_numpy(
        "ships", launch_actions.ships, dtype=np.int64, torch_dtype=torch.int64
    )
    expected_batched_shape = (
        1,
        OUTER_PLAYER_SLOTS,
        ACTION_ENTITY_SLOTS,
        action_spec.max_per_planet_launches,
    )
    _require_action_shape("launch", launch_array, expected_batched_shape)
    _require_action_shape("ships", ship_array, expected_batched_shape)
    launch_array = np.ascontiguousarray(launch_array[0])
    ship_array = np.ascontiguousarray(ship_array[0])

    if isinstance(action_spec, ActionPureConfig):
        pure_actions = cast(PureActions, launch_actions)
        angle_array = _actions_to_numpy(
            "angle",
            pure_actions.angle,
            dtype=np.float32,
            torch_dtype=torch.float32,
        )
        _require_action_shape("angle", angle_array, expected_batched_shape)
        return _sanitize_kaggle_actions(
            _pure_actions_to_kaggle(
                planets_in,
                initial_planets_in,
                fleets_in,
                comet_planet_ids,
                comet_path_indices,
                comet_path_lengths,
                comet_paths,
                angular_velocity,
                step,
                episode_steps,
                int(player),
                launch_array,
                np.ascontiguousarray(angle_array[0]),
                ship_array,
                action_spec.max_per_planet_launches,
                action_spec.min_fleet_size,
            )
        )

    discrete_actions = cast(DiscreteTargetActions, launch_actions)
    target_array = _actions_to_numpy(
        "target",
        discrete_actions.target,
        dtype=np.int64,
        torch_dtype=torch.int64,
    )
    _require_action_shape("target", target_array, expected_batched_shape)
    return _sanitize_kaggle_actions(
        _discrete_target_actions_to_kaggle(
            planets_in,
            initial_planets_in,
            fleets_in,
            comet_planet_ids,
            comet_path_indices,
            comet_path_lengths,
            comet_paths,
            angular_velocity,
            step,
            episode_steps,
            int(player),
            launch_array,
            np.ascontiguousarray(target_array[0]),
            ship_array,
            action_spec.max_per_planet_launches,
            action_spec.min_fleet_size,
            action_spec.targeting_mode,
        )
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


def _truncate_mask_to_numpy(mask: np.ndarray | torch.Tensor, n_envs: int) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        if mask.device.type != "cpu":
            raise ValueError("truncate_mask must be on CPU")
        if mask.dtype != torch.bool:
            raise ValueError(
                f"truncate_mask must have dtype torch.bool, got {mask.dtype}"
            )
        mask = mask.detach().numpy()
    elif isinstance(mask, np.ndarray):
        if mask.dtype != np.dtype(np.bool_):
            raise ValueError(f"truncate_mask must have dtype bool, got {mask.dtype}")
    else:
        raise TypeError("truncate_mask must be a NumPy array or Torch tensor")
    if mask.shape != (n_envs,):
        raise ValueError(f"truncate_mask must have shape {(n_envs,)}, got {mask.shape}")
    return np.ascontiguousarray(mask)


def _encoded_observation_to_batch(
    encoded: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray,
        np.ndarray | None,
    ],
    *,
    still_playing: np.ndarray,
) -> ObsBatch:
    (
        planets,
        orbiting_planets,
        fleets,
        fleet_target,
        target_incoming_features,
        comets,
        entity_mask,
        global_features,
        player_features,
        can_act,
        max_launch,
    ) = encoded
    can_act_tensor = torch.as_tensor(can_act, dtype=torch.bool).unsqueeze(0)
    max_launch_tensor = (
        None
        if max_launch is None
        else torch.as_tensor(max_launch, dtype=torch.int64).unsqueeze(0)
    )
    if max_launch_tensor is None:
        action_mask: ActionMask = DiscreteTargetBinActionMask(can_act=can_act_tensor)
    elif can_act_tensor.ndim == 4:
        action_mask = DiscreteTargetActionMask(
            can_act=can_act_tensor,
            max_launch=max_launch_tensor,
        )
    else:
        action_mask = PureActionMask(
            can_act=can_act_tensor,
            max_launch=max_launch_tensor,
        )
    return ObsBatch(
        planets=torch.as_tensor(planets, dtype=torch.float32).unsqueeze(0),
        orbiting_planets=torch.as_tensor(orbiting_planets, dtype=torch.bool).unsqueeze(
            0
        ),
        fleets=torch.as_tensor(fleets, dtype=torch.float32).unsqueeze(0),
        fleet_target=(
            None
            if fleet_target is None
            else torch.as_tensor(fleet_target, dtype=torch.int64).unsqueeze(0)
        ),
        target_incoming_features=(
            None
            if target_incoming_features is None
            else torch.as_tensor(
                target_incoming_features,
                dtype=torch.float32,
            ).unsqueeze(0)
        ),
        comets=torch.as_tensor(comets, dtype=torch.float32).unsqueeze(0),
        entity_mask=torch.as_tensor(entity_mask, dtype=torch.bool).unsqueeze(0),
        still_playing=torch.as_tensor(still_playing, dtype=torch.bool).unsqueeze(0),
        global_features=torch.as_tensor(global_features, dtype=torch.float32).unsqueeze(
            0
        ),
        action_mask=action_mask,
        player_features=(
            None
            if player_features is None
            else torch.as_tensor(player_features, dtype=torch.float32).unsqueeze(0)
        ),
    )


def _loose_target_can_act(
    source_can_act: np.ndarray, entity_mask: np.ndarray
) -> np.ndarray:
    target_exists = entity_mask[:ACTION_ENTITY_SLOTS].astype(np.bool_, copy=False)
    can_act = source_can_act[:, :, None] & target_exists[None, None, :]
    source_indices = np.arange(ACTION_ENTITY_SLOTS)
    can_act[:, source_indices, source_indices] = False
    return can_act


def _target_bin_actions_to_kaggle(
    obs: dict[str, Any],
    player: int,
    target: np.ndarray | torch.Tensor,
    fleet_bin: np.ndarray | torch.Tensor,
    *,
    action_spec: ActionDiscreteTargetBinsConfig,
) -> list[list[float]]:
    (
        planets_in,
        initial_planets_in,
        fleets_in,
        comet_planet_ids,
        comet_path_indices,
        comet_path_lengths,
        comet_paths,
        angular_velocity,
        step,
        episode_steps,
    ) = _observation_arrays(obs)
    target_array = _actions_to_numpy(
        "target",
        target,
        dtype=np.int64,
        torch_dtype=torch.int64,
    )
    fleet_bin_array = _actions_to_numpy(
        "fleet_bin",
        fleet_bin,
        dtype=np.int64,
        torch_dtype=torch.int64,
    )
    expected_batched_shape = (1, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS)
    _require_action_shape("target", target_array, expected_batched_shape)
    _require_action_shape("fleet_bin", fleet_bin_array, expected_batched_shape)
    return _sanitize_kaggle_actions(
        _discrete_target_bin_actions_to_kaggle(
            planets_in,
            initial_planets_in,
            fleets_in,
            comet_planet_ids,
            comet_path_indices,
            comet_path_lengths,
            comet_paths,
            angular_velocity,
            step,
            episode_steps,
            int(player),
            np.ascontiguousarray(target_array[0]),
            np.ascontiguousarray(fleet_bin_array[0]),
            action_spec.min_fleet_size,
            action_spec.n_bins,
            action_spec.targeting_mode,
        )
    )


def _sanitize_kaggle_actions(actions: list[list[float]]) -> list[list[float]]:
    return [[int(planet), angle, int(ships)] for (planet, angle, ships) in actions]


def _can_act_shape(
    n_envs: int,
    n_players: int,
    action_spec: ActionConfig,
) -> tuple[int, ...]:
    if isinstance(action_spec, ActionPureConfig):
        return (n_envs, n_players, ACTION_ENTITY_SLOTS)
    if isinstance(action_spec, ActionDiscreteTargetsConfig):
        return (n_envs, n_players, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS)
    return (
        n_envs,
        n_players,
        ACTION_ENTITY_SLOTS,
        ACTION_ENTITY_SLOTS,
        action_spec.n_bins,
    )


def _max_per_planet_launches(action_spec: ActionConfig) -> int:
    if isinstance(action_spec, ActionDiscreteTargetBinsConfig):
        return 1
    return action_spec.max_per_planet_launches


def _n_action_bins(action_spec: ActionConfig) -> int:
    if isinstance(action_spec, ActionDiscreteTargetBinsConfig):
        return action_spec.n_bins
    return 0


def _targeting_mode(action_spec: ActionConfig) -> TargetingMode:
    if isinstance(action_spec, ActionPureConfig):
        return "full_mask"
    return action_spec.targeting_mode


def _max_decoded_actions(action_spec: ActionConfig) -> int:
    if isinstance(action_spec, ActionDiscreteTargetBinsConfig):
        return ACTION_ENTITY_SLOTS
    return ACTION_ENTITY_SLOTS * action_spec.max_per_planet_launches


def fleet_bin_to_ships(
    fleet_bin: np.ndarray | torch.Tensor,
    available_ships: np.ndarray | torch.Tensor,
    *,
    n_bins: int,
) -> np.ndarray | torch.Tensor:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    denominator = n_bins - 1
    return (fleet_bin * available_ships + denominator // 2) // denominator


def _target_bin_can_act(
    target_can_act: np.ndarray,
    max_launch: np.ndarray,
    *,
    min_fleet_size: int,
    n_bins: int,
) -> np.ndarray:
    bins = np.arange(n_bins, dtype=np.int64)
    ships = fleet_bin_to_ships(
        bins.reshape(1, 1, n_bins),
        max_launch[..., None].astype(np.int64),
        n_bins=n_bins,
    )
    can_act = np.zeros((*target_can_act.shape, n_bins), dtype=np.bool_)
    can_act[..., 0] = target_can_act
    for fleet_bin in range(1, n_bins):
        ship_count = ships[..., fleet_bin]
        duplicate_later = np.zeros_like(ship_count, dtype=np.bool_)
        for later_bin in range(fleet_bin + 1, n_bins):
            duplicate_later |= ships[..., later_bin] == ship_count
        can_act[..., fleet_bin] = (
            target_can_act
            & (ship_count[:, :, None] >= min_fleet_size)
            & ~duplicate_later[:, :, None]
        )
    return can_act


def _require_action_shape(
    name: str, action_array: np.ndarray, expected_shape: tuple[int, ...]
) -> None:
    if action_array.shape == expected_shape:
        return
    raise ValueError(
        f"{name} must have shape {expected_shape}, got {action_array.shape}"
    )


def _observation_arrays(
    obs: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
    int,
]:
    comet_planet_ids, comet_path_indices, comet_path_lengths, comet_paths = (
        _comets_to_arrays(obs["comets"])
    )
    return (
        _rows_to_array(obs["planets"], name="planets"),
        _rows_to_array(obs["initial_planets"], name="initial_planets"),
        _rows_to_array(obs["fleets"], name="fleets"),
        comet_planet_ids,
        comet_path_indices,
        comet_path_lengths,
        comet_paths,
        float(obs["angular_velocity"]),
        _require_int(obs["step"], name="step"),
        _require_int(obs["episode_steps"], name="episode_steps"),
    )


def _require_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"obs['{name}'] must be an integer")
    return int(value)


def _still_playing_from_arrays(
    planets: np.ndarray,
    fleets: np.ndarray,
    *,
    player: Any,
) -> np.ndarray:
    current_player = _require_int(player, name="player")
    if not 0 <= current_player < OUTER_PLAYER_SLOTS:
        raise ValueError(f"obs['player'] must be in [0, {OUTER_PLAYER_SLOTS})")

    still_playing = np.zeros((OUTER_PLAYER_SLOTS,), dtype=np.bool_)
    still_playing[current_player] = True
    for rows in (planets, fleets):
        for owner in rows[:, 1]:
            owner_id = int(owner)
            if 0 <= owner_id < OUTER_PLAYER_SLOTS:
                still_playing[owner_id] = True
    return still_playing


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
        raw_planet_ids = group["planet_ids"]
        raw_paths = group["paths"]
        if not isinstance(raw_planet_ids, list) or not isinstance(raw_paths, list):
            raise TypeError("comet groups need list planet_ids and paths")
        if len(raw_planet_ids) != len(raw_paths):
            raise ValueError("comet planet_ids and paths must have the same length")
        if len(raw_planet_ids) > MAX_COMETS:
            raise ValueError(f"comet groups must have at most {MAX_COMETS} paths")
        path_indices[group_index] = float(cast(SupportsFloat, group["path_index"]))

        for path_offset, (planet_id, raw_path) in enumerate(
            zip(raw_planet_ids, raw_paths, strict=True)
        ):
            path_array = np.asarray(raw_path, dtype=np.float64)
            if path_array.ndim != 2 or path_array.shape[1] != 2:
                raise ValueError("comet paths must have shape (n, 2)")
            path_len = path_array.shape[0]
            if path_len > MAX_COMET_PATH_LENGTH:
                raise ValueError(
                    f"comet paths must have at most {MAX_COMET_PATH_LENGTH} points"
                )
            planet_ids[group_index, path_offset] = float(cast(SupportsFloat, planet_id))
            path_lengths[group_index, path_offset] = path_len
            paths[group_index, path_offset, :path_len, :] = path_array[:path_len]

    active_comet_count = int((planet_ids >= 0).sum())
    if active_comet_count > MAX_COMETS:
        raise ValueError(f"observations must have at most {MAX_COMETS} active comets")

    return (
        np.ascontiguousarray(planet_ids),
        np.ascontiguousarray(path_indices),
        np.ascontiguousarray(path_lengths),
        np.ascontiguousarray(paths),
    )
