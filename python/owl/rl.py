from __future__ import annotations

from typing import Literal, SupportsFloat, cast

import numpy as np
import torch
from pydantic import BaseModel, Field

from owl.rs import RlVecEnv as _RustRlVecEnv
from owl.rs import encode_obs_v1, rl_obs_constants

MAX_PLANETS, DEFAULT_MAX_ENTITIES, PLANET_CHANNELS, FLEET_CHANNELS = rl_obs_constants()


class ObsV1Config(BaseModel):
    obs_spec: Literal["obs_v1"] = "obs_v1"
    max_entities: int = Field(default=DEFAULT_MAX_ENTITIES, gt=MAX_PLANETS)

    @property
    def max_planets(self) -> int:
        return MAX_PLANETS

    @property
    def max_fleets(self) -> int:
        return self.max_entities - self.max_planets

    @property
    def planet_channels(self) -> int:
        return PLANET_CHANNELS

    @property
    def fleet_channels(self) -> int:
        return FLEET_CHANNELS


class ActionV1Config(BaseModel):
    action_spec: Literal["action_v1"] = "action_v1"
    action_dim: int = 0


class ObsBatch(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    planets: torch.Tensor
    fleets: torch.Tensor
    planet_mask: torch.Tensor
    fleet_mask: torch.Tensor


class VectorizedEnv:
    def __init__(
        self,
        *,
        n_envs: int,
        n_players: int,
        obs_spec: ObsV1Config | None = None,
        action_spec: ActionV1Config | None = None,
        pin_memory: bool = True,
    ) -> None:
        self.obs_spec = obs_spec or ObsV1Config()
        self.action_spec = action_spec or ActionV1Config()
        self._rust = _RustRlVecEnv(
            n_envs,
            n_players,
            self.obs_spec.obs_spec,
            self.obs_spec.max_entities,
            self.action_spec.action_dim,
        )
        self.n_envs = n_envs
        self.n_players = n_players
        self.observations = self._allocate_observations(pin_memory=pin_memory)
        self.rewards = torch.zeros(
            (n_envs, n_players), dtype=torch.float32, pin_memory=pin_memory
        )
        self.dones = torch.zeros(
            (n_envs, n_players), dtype=torch.bool, pin_memory=pin_memory
        )

        self._planet_obs_np = self.observations.planets.numpy()
        self._fleet_obs_np = self.observations.fleets.numpy()
        self._planet_mask_np = self.observations.planet_mask.numpy()
        self._fleet_mask_np = self.observations.fleet_mask.numpy()
        self._rewards_np = self.rewards.numpy()
        self._dones_np = self.dones.numpy()

    def reset(self) -> ObsBatch:
        self._rust.reset(
            self._planet_obs_np,
            self._fleet_obs_np,
            self._planet_mask_np,
            self._fleet_mask_np,
        )
        return self.observations

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[ObsBatch, torch.Tensor, torch.Tensor]:
        action_array = _actions_to_numpy(actions)
        expected_shape = (
            self.n_envs,
            self.n_players,
            self.action_spec.action_dim,
        )
        if action_array.shape != expected_shape:
            msg = f"actions must have shape {expected_shape}, got {action_array.shape}"
            raise ValueError(msg)

        self._rust.step(
            action_array,
            self._planet_obs_np,
            self._fleet_obs_np,
            self._planet_mask_np,
            self._fleet_mask_np,
            self._rewards_np,
            self._dones_np,
        )
        return self.observations, self.rewards, self.dones

    def _allocate_observations(self, *, pin_memory: bool) -> ObsBatch:
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
            planet_mask=torch.zeros(
                (self.n_envs, self.obs_spec.max_planets),
                dtype=torch.bool,
                pin_memory=pin_memory,
            ),
            fleet_mask=torch.zeros(
                (self.n_envs, self.obs_spec.max_fleets),
                dtype=torch.bool,
                pin_memory=pin_memory,
            ),
        )


def encode_python_observation(
    obs: dict[str, object],
    obs_spec: ObsV1Config | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = obs_spec or ObsV1Config()
    return encode_obs_v1(
        _rows_to_array(obs.get("planets", []), name="planets"),
        _rows_to_array(obs.get("fleets", []), name="fleets"),
        float(cast(SupportsFloat, obs.get("angular_velocity", 0.0))),
        spec.max_entities,
    )


def _actions_to_numpy(actions: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(actions, torch.Tensor):
        if actions.device.type != "cpu":
            raise ValueError("actions must be on CPU before stepping the Rust env")
        actions = actions.detach().numpy()
    return np.ascontiguousarray(actions, dtype=np.float32)


def _rows_to_array(rows: object, *, name: str) -> np.ndarray:
    if not isinstance(rows, list):
        raise TypeError(f"obs['{name}'] must be a list")
    if not rows:
        return np.empty((0, 7), dtype=np.float64)
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"obs['{name}'] must have shape (n, 7)")
    return np.ascontiguousarray(array)
