from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Literal

import pytest
import torch
from owl.model import (
    ActorDiscreteTargetsConfig,
    BaseModelAPI,
    ModelActionEntropies,
    ModelActionLogProbs,
    ModelEvaluation,
    ModelOutput,
    RecurrentTransformerV1,
    RecurrentTransformerV1Config,
    StatelessTransformerV1,
    StatelessTransformerV1Config,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    ActionBundle,
    ActionDiscreteTargetBinsConfig,
    ActionDiscreteTargetsConfig,
    ActionPureConfig,
    DiscreteTargetActionMask,
    DiscreteTargetActions,
    DiscreteTargetBinActionMask,
    DiscreteTargetBinActions,
    EntityBasedConfig,
    ObsBatch,
    PureActionMask,
    PureActions,
    VectorizedEnv,
)
from owl.train import ppo
from torch import nn

_OBS_COPY_FIELDS = tuple(
    field
    for field in ObsBatch.model_fields
    if field not in {"action_mask", "player_features"}
)


def _obs_buffer_ptrs(obs: ObsBatch) -> dict[str, int]:
    ptrs = {field: getattr(obs, field).data_ptr() for field in _OBS_COPY_FIELDS}
    if obs.player_features is not None:
        ptrs["player_features"] = obs.player_features.data_ptr()
    ptrs["action_mask.can_act"] = obs.action_mask.can_act.data_ptr()
    if isinstance(obs.action_mask, PureActionMask | DiscreteTargetActionMask):
        ptrs["action_mask.max_launch"] = obs.action_mask.max_launch.data_ptr()
    return ptrs


def _obs_batch(*, n_envs: int, obs_spec: EntityBasedConfig) -> ObsBatch:
    return ObsBatch(
        planets=torch.zeros(
            (n_envs, obs_spec.max_planets, obs_spec.planet_channels),
            dtype=torch.float32,
        ),
        orbiting_planets=torch.zeros(
            (n_envs, obs_spec.max_planets),
            dtype=torch.bool,
        ),
        fleets=torch.zeros(
            (n_envs, obs_spec.max_fleets, obs_spec.fleet_channels),
            dtype=torch.float32,
        ),
        comets=torch.zeros(
            (n_envs, obs_spec.max_comets, obs_spec.comet_channels),
            dtype=torch.float32,
        ),
        entity_mask=torch.zeros((n_envs, obs_spec.max_entities), dtype=torch.bool),
        still_playing=torch.ones((n_envs, 4), dtype=torch.bool),
        global_features=torch.zeros(
            (n_envs, obs_spec.global_channels),
            dtype=torch.float32,
        ),
        action_mask=PureActionMask(
            can_act=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
            max_launch=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64),
        ),
        player_features=(
            None
            if obs_spec.player_feature_channels == 0
            else torch.zeros((n_envs, 4, obs_spec.player_feature_channels))
        ),
    )


def _actions(
    n_envs: int,
    max_launches: int = ActionPureConfig().max_per_planet_launches,
    kind: Literal["pure", "discrete_targets", "discrete_target_bins"] = "pure",
) -> ActionBundle:
    if kind == "discrete_target_bins":
        shape = (n_envs, 4, ACTION_ENTITY_SLOTS)
        return DiscreteTargetBinActions(
            target=torch.zeros(shape, dtype=torch.int64),
            fleet_bin=torch.zeros(shape, dtype=torch.int64),
        )
    shape = (n_envs, 4, ACTION_ENTITY_SLOTS, max_launches)
    if kind == "pure":
        return PureActions(
            launch=torch.zeros(shape, dtype=torch.bool),
            angle=torch.zeros(shape, dtype=torch.float32),
            ships=torch.zeros(shape, dtype=torch.int64),
        )
    return DiscreteTargetActions(
        launch=torch.zeros(shape, dtype=torch.bool),
        target=torch.zeros(shape, dtype=torch.int64),
        ships=torch.zeros(shape, dtype=torch.int64),
    )


def _discrete_obs_batch(*, n_envs: int, obs_spec: EntityBasedConfig) -> ObsBatch:
    obs = _obs_batch(n_envs=n_envs, obs_spec=obs_spec)
    obs.action_mask = DiscreteTargetActionMask(
        can_act=torch.zeros(
            (n_envs, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS),
            dtype=torch.bool,
        ),
        max_launch=obs.action_mask.max_launch,
    )
    return obs


class TinyOrbitEnv:
    def __init__(
        self,
        *,
        n_envs: int,
        episode_length: int = 3,
        two_player: bool = False,
    ) -> None:
        self.n_envs = n_envs
        self.pin_memory_enabled = False
        self.obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 2)
        self.action_spec = ActionPureConfig()
        self.episode_length = episode_length
        self.two_player = two_player
        self._steps = torch.zeros(n_envs, dtype=torch.int64)
        self._targets = torch.arange(n_envs, dtype=torch.float32).remainder(2)

    def reset(self) -> ObsBatch:
        self._steps.zero_()
        return self._obs()

    def step(
        self,
        actions: ActionBundle,
    ) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
        if not isinstance(actions, PureActions):
            raise TypeError("TinyOrbitEnv requires PureActions")
        active = self._still_playing()
        player_launch = actions.launch[:, :, 0, 0].to(dtype=torch.float32)
        reward = torch.where(player_launch.eq(self._targets[:, None]), 1.0, -0.25)
        reward = torch.where(active, reward, torch.zeros_like(reward))
        self._steps += 1
        done = self._steps >= self.episode_length
        self._steps[done] = 0
        dones = done[:, None].expand(-1, 4) | ~active
        return self._obs(), reward, dones, {}

    def _obs(self) -> ObsBatch:
        obs = _obs_batch(n_envs=self.n_envs, obs_spec=self.obs_spec)
        obs.global_features[:, 0] = self._targets
        obs.global_features[:, 1] = self._steps.to(torch.float32) / float(
            self.episode_length
        )
        obs.still_playing = self._still_playing()
        obs.entity_mask[:, :2] = True
        obs.action_mask.can_act[:, :, 0] = obs.still_playing
        obs.action_mask.max_launch[:, :, 0] = obs.still_playing.to(torch.int64)
        return obs

    def _still_playing(self) -> torch.Tensor:
        still_playing = torch.ones((self.n_envs, 4), dtype=torch.bool)
        if self.two_player:
            still_playing[:, 2:] = False
        return still_playing


class TinyOrbitEnvWithMetrics(TinyOrbitEnv):
    def step(
        self,
        actions: ActionBundle,
    ) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
        obs, rewards, dones, _metrics = super().step(actions)
        games_played = float(dones[:, 0].sum().item())
        return obs, rewards, dones, {"total_games_played": [games_played]}


class TinyDiscreteTargetEnv:
    def __init__(self, *, n_envs: int) -> None:
        self.n_envs = n_envs
        self.pin_memory_enabled = False
        self.obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 2)
        self.action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
        self.last_target: torch.Tensor | None = None

    def reset(self) -> ObsBatch:
        return self._obs()

    def step(
        self,
        actions: ActionBundle,
    ) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
        if not isinstance(actions, DiscreteTargetActions):
            raise TypeError("TinyDiscreteTargetEnv requires DiscreteTargetActions")
        assert actions.target.dtype == torch.int64
        self.last_target = actions.target.clone()
        rewards = actions.launch[:, :, 0, 0].to(torch.float32) + actions.ships[
            :,
            :,
            0,
            0,
        ].to(torch.float32)
        dones = torch.zeros((self.n_envs, 4), dtype=torch.bool)
        return self._obs(), rewards, dones, {}

    def _obs(self) -> ObsBatch:
        obs = _discrete_obs_batch(n_envs=self.n_envs, obs_spec=self.obs_spec)
        obs.still_playing.fill_(True)
        obs.entity_mask[:, :2] = True
        obs.action_mask.can_act[:, :, 0, 1] = True
        obs.action_mask.max_launch[:, :, 0] = 3
        return obs


class ReusingObservationEnv(TinyOrbitEnv):
    def __init__(self, *, n_envs: int, episode_length: int = 10) -> None:
        super().__init__(n_envs=n_envs, episode_length=episode_length)
        self._obs_storage = _obs_batch(n_envs=n_envs, obs_spec=self.obs_spec)

    def _obs(self) -> ObsBatch:
        obs = self._obs_storage
        obs.planets.zero_()
        obs.fleets.zero_()
        obs.comets.zero_()
        obs.entity_mask.zero_()
        obs.still_playing.zero_()
        obs.global_features.zero_()
        obs.action_mask.can_act.zero_()
        obs.action_mask.max_launch.zero_()
        obs.global_features[:, 0] = self._targets
        obs.global_features[:, 1] = self._steps.to(torch.float32)
        obs.still_playing.copy_(self._still_playing())
        obs.entity_mask[:, :2] = True
        obs.action_mask.can_act[:, :, 0] = obs.still_playing
        obs.action_mask.max_launch[:, :, 0] = obs.still_playing.to(torch.int64)
        return obs


class TinyOrbitModel(BaseModelAPI):
    def __init__(self) -> None:
        super().__init__()
        self.action_spec = ActionPureConfig()
        self.input_proj = nn.Linear(3, 8)
        self.hidden = nn.Linear(8, 8)
        self.policy = nn.Linear(8, 4)
        self.value = nn.Linear(8, 4)

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
    ) -> ModelOutput:
        hidden = torch.tanh(
            self.hidden(torch.tanh(self.input_proj(obs.global_features)))
        )
        logits = self.policy(hidden)
        dist = torch.distributions.Bernoulli(logits=logits)
        launch = logits.gt(0) if deterministic else dist.sample().bool()
        actions = _actions(
            obs.global_features.shape[0],
            self.action_spec.max_per_planet_launches,
        )
        actions.launch[:, :, 0, 0] = launch & obs.still_playing
        actions.ships[:, :, 0, 0] = actions.launch[:, :, 0, 0].to(torch.int64)
        log_probs = self._log_probs(dist.log_prob(actions.launch[:, :, 0, 0].float()))
        entropies = self._entropies(dist.entropy())
        values = self.value(hidden)
        return ModelOutput(
            actions=actions,
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=torch.softmax(values, dim=-1),
        )

    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ActionBundle,
    ) -> ModelEvaluation:
        hidden = torch.tanh(
            self.hidden(torch.tanh(self.input_proj(obs.global_features)))
        )
        dist = torch.distributions.Bernoulli(logits=self.policy(hidden))
        log_probs = self._log_probs(dist.log_prob(actions.launch[:, :, 0, 0].float()))
        entropies = self._entropies(dist.entropy())
        values = self.value(hidden)
        return ModelEvaluation(
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=torch.softmax(values, dim=-1),
        )

    def compute_value(self, obs: ObsBatch) -> torch.Tensor:
        hidden = torch.tanh(
            self.hidden(torch.tanh(self.input_proj(obs.global_features)))
        )
        return self.value(hidden)

    def reset_parameters(self) -> None:
        self.input_proj.reset_parameters()
        self.hidden.reset_parameters()
        self.policy.reset_parameters()
        self.value.reset_parameters()

    def get_input_layers(self) -> tuple[nn.Module, ...]:
        return (self.input_proj,)

    def get_output_layers(self) -> tuple[nn.Module, ...]:
        return (self.policy, self.value)

    @staticmethod
    def _log_probs(per_player: torch.Tensor) -> ModelActionLogProbs:
        n_envs = per_player.shape[0]
        action_shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
        launch = torch.zeros(action_shape, dtype=per_player.dtype)
        event = torch.zeros_like(launch)
        per_player_entity = torch.zeros(
            (n_envs, 4, ACTION_ENTITY_SLOTS), dtype=per_player.dtype
        )
        launch[:, :, 0, 0] = per_player
        per_player_entity[:, :, 0] = per_player
        return ModelActionLogProbs(
            launch=launch,
            event=event,
            per_player_entity=per_player_entity,
        )

    @staticmethod
    def _entropies(per_player: torch.Tensor) -> ModelActionEntropies:
        n_envs = per_player.shape[0]
        action_shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
        launch = torch.zeros(action_shape, dtype=per_player.dtype)
        event = torch.zeros_like(launch)
        per_player_entity = torch.zeros(
            (n_envs, 4, ACTION_ENTITY_SLOTS), dtype=per_player.dtype
        )
        launch[:, :, 0, 0] = per_player
        per_player_entity[:, :, 0] = per_player
        return ModelActionEntropies(
            launch=launch,
            event=event,
            per_player_entity=per_player_entity,
            components={"launch": per_player_entity},
        )


class CountingForwardModel(TinyOrbitModel):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
    ) -> ModelOutput:
        self.forward_calls += 1
        return super().forward(obs, deterministic=deterministic)


class TinyDiscreteTargetModel(BaseModelAPI):
    def __init__(self) -> None:
        super().__init__()
        self.action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
        self.value = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,  # noqa: ARG002
    ) -> ModelOutput:
        n_envs = obs.global_features.shape[0]
        actions = _actions(
            n_envs,
            self.action_spec.max_per_planet_launches,
            kind="discrete_targets",
        )
        actions.launch[:, :, 0, 0] = obs.still_playing
        assert actions.target is not None
        actions.target[:, :, 0, 0] = 1
        actions.ships[:, :, 0, 0] = 1
        values = self.value.expand(n_envs, 4)
        log_probs = TinyOrbitModel._log_probs(torch.zeros_like(values))
        entropies = TinyOrbitModel._entropies(torch.zeros_like(values))
        return ModelOutput(
            actions=actions,
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=torch.softmax(values, dim=-1),
        )

    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ActionBundle,  # noqa: ARG002
    ) -> ModelEvaluation:
        values = self.value.expand(obs.global_features.shape[0], 4)
        log_probs = TinyOrbitModel._log_probs(torch.zeros_like(values))
        entropies = TinyOrbitModel._entropies(torch.zeros_like(values))
        return ModelEvaluation(
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=torch.softmax(values, dim=-1),
        )

    def compute_value(self, obs: ObsBatch) -> torch.Tensor:
        return self.value.expand(obs.global_features.shape[0], 4)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.value)

    def get_input_layers(self) -> tuple[nn.Module, ...]:
        return ()

    def get_output_layers(self) -> tuple[nn.Module, ...]:
        return ()


class AutocastRecordingModel(TinyOrbitModel):
    def __init__(self, device_type: str) -> None:
        super().__init__()
        self.device_type = device_type
        self.forward_autocast_enabled: list[bool] = []
        self.evaluate_autocast_enabled: list[bool] = []
        self.compute_value_autocast_enabled: list[bool] = []

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
    ) -> ModelOutput:
        self.forward_autocast_enabled.append(
            torch.is_autocast_enabled(self.device_type)
        )
        return super().forward(obs, deterministic=deterministic)

    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ActionBundle,
    ) -> ModelEvaluation:
        self.evaluate_autocast_enabled.append(
            torch.is_autocast_enabled(self.device_type)
        )
        return super().evaluate_actions(obs, actions)

    def compute_value(self, obs: ObsBatch) -> torch.Tensor:
        self.compute_value_autocast_enabled.append(
            torch.is_autocast_enabled(self.device_type)
        )
        return super().compute_value(obs)


class FixedEvaluationModel(BaseModelAPI):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.action_spec = ActionPureConfig()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,  # noqa: ARG002
    ) -> ModelOutput:
        n_envs = obs.global_features.shape[0]
        evaluation = self.evaluate_actions(obs, _actions(n_envs))
        return ModelOutput(
            actions=_actions(n_envs),
            log_probs=evaluation.log_probs,
            entropies=evaluation.entropies,
            values=evaluation.values,
            winner_probabilities=evaluation.winner_probabilities,
        )

    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ActionBundle,  # noqa: ARG002
    ) -> ModelEvaluation:
        values = self.value.expand(obs.global_features.shape[0], 4)
        log_probs = TinyOrbitModel._log_probs(torch.zeros_like(values))
        entropies = TinyOrbitModel._entropies(torch.zeros_like(values))
        return ModelEvaluation(
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=torch.softmax(values, dim=-1),
        )

    def compute_value(self, obs: ObsBatch) -> torch.Tensor:
        return self.value.expand(obs.global_features.shape[0], 4)

    def reset_parameters(self) -> None:
        return None

    def get_input_layers(self) -> tuple[nn.Module, ...]:
        return ()

    def get_output_layers(self) -> tuple[nn.Module, ...]:
        return ()


def _zero_loss_metrics(zero: torch.Tensor) -> ppo._PPOLossMetrics:
    return ppo._PPOLossMetrics(
        loss=zero,
        policy_loss=zero,
        value_loss=zero,
        entropy_loss=zero,
        teacher_kl_loss=zero,
        teacher_value_loss=zero,
        entropy=zero,
        teacher_kl=zero,
        teacher_value_cross_entropy=zero,
        approx_kl=zero,
        clipfrac=zero,
        ratio_mean=zero,
        ratio_max=zero,
        logratio_mean=zero,
        logratio_abs_max=zero,
    )


def test_rollout_buffer_collects_time_major_and_returns_contiguous_segments() -> None:
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 1)
    action_spec = ActionPureConfig()
    buffer = ppo._PPORolloutBuffer(
        horizon=3,
        n_envs=2,
        obs_spec=obs_spec,
        action_spec=action_spec,
        device=torch.device("cpu"),
    )
    obs = _obs_batch(n_envs=2, obs_spec=obs_spec)
    actions = _actions(2)

    for step in range(3):
        obs.global_features.fill_(float(step))
        actions.ships.fill_(step)
        buffer.write_step(
            step,
            obs=obs,
            actions=actions,
            logp=torch.full((2, 4), float(step)),
            values=torch.full((2, 4), float(step + 1)),
            rewards=torch.full((2, 4), float(step + 2)),
            dones=torch.zeros((2, 4), dtype=torch.bool),
        )

    segments = buffer.segment_major()

    assert segments.obs.planets.shape == (
        2,
        3,
        obs_spec.max_planets,
        obs_spec.planet_channels,
    )
    assert segments.obs.orbiting_planets.shape == (
        2,
        3,
        obs_spec.max_planets,
    )
    assert segments.actions.launch.shape == (
        2,
        3,
        4,
        ACTION_ENTITY_SLOTS,
        action_spec.max_per_planet_launches,
    )
    assert segments.logp.shape == (2, 3, 4)
    assert segments.values.shape == (2, 3, 4)
    assert segments.rewards.shape == (2, 3, 4)
    assert segments.obs.global_features.is_contiguous()
    assert torch.equal(
        segments.obs.global_features[0, :, 0],
        torch.tensor([0.0, 1.0, 2.0]),
    )


def test_obs_to_device_clones_cpu_observation_buffers() -> None:
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 1)
    obs = _obs_batch(n_envs=2, obs_spec=obs_spec)
    obs.global_features.fill_(1.0)

    copied = ppo._obs_to_device(obs, torch.device("cpu"))
    obs.global_features.fill_(2.0)

    assert torch.equal(copied.global_features, torch.ones_like(copied.global_features))
    for field, copied_ptr in _obs_buffer_ptrs(copied).items():
        assert copied_ptr != _obs_buffer_ptrs(obs)[field]


def test_env_metrics_are_logged_under_train_prefix() -> None:
    metrics = ppo._mean_env_metrics(
        {
            "game_length_mean": [10.0, 14.0],
            "total_games_played": [1.0, 1.0],
            "full_length_rate": [1.0, 0.0],
            "terminal_ship_count": [20.0, 40.0],
            "launches_per_game": [3.0, 7.0],
            "launch_failures_per_game": [2.0, 4.0],
            "comets_captured_per_game": [1.0, 3.0],
            "_neutral_planets_captured_per_game": [1.0, 3.0],
            "_neutral_comets_captured_per_game": [2.0, 0.0],
            "_neutral_planet_undershots_per_game": [3.0, 1.0],
            "_neutral_comet_undershots_per_game": [4.0, 0.0],
            "neutral_planet_undershot_rate": [0.75, 0.25],
            "neutral_comet_undershot_rate": [2.0 / 3.0],
            "ships_lost_in_combat_per_game": [5.0, 15.0],
            "terminal_planet_occupancy_rate_2p": [0.5, 0.75],
            "terminal_planet_occupancy_rate_4p": [1.0],
            "win_rate_player_0": [1.0, 0.0],
        }
    )

    assert metrics["train/game_length_mean"] == 12.0
    assert metrics["train/total_games_played"] == 2.0
    assert metrics["train/full_length_rate"] == 0.5
    assert metrics["train/terminal_ship_count"] == 30.0
    assert metrics["train/launches_per_game"] == 5.0
    assert metrics["train/launch_failures_per_game"] == 3.0
    assert metrics["train/comets_captured_per_game"] == 2.0
    assert metrics["train/neutral_planet_undershot_rate"] == 0.5
    assert metrics["train/neutral_comet_undershot_rate"] == 4.0 / 6.0
    assert "train/_neutral_planets_captured_per_game" not in metrics
    assert metrics["train/ships_lost_in_combat_per_game"] == 10.0
    assert metrics["train/terminal_planet_occupancy_rate_2p"] == 0.625
    assert metrics["train/terminal_planet_occupancy_rate_4p"] == 1.0
    assert metrics["train/win_rate_player_0"] == 0.5


def test_distributed_env_metrics_reduce_matching_global_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=2,
        initialized=True,
    )

    def fake_all_gather_object(
        value: set[str],
        _context: ppo.DistributedContext,
    ) -> list[set[str]]:
        assert value == {"total_games_played", "z_metric"}
        assert _context is context
        return [
            {"total_games_played", "z_metric"},
            {"a_metric", "total_games_played", "z_metric"},
        ]

    def fake_all_reduce_sum(
        tensor: torch.Tensor,
        _context: ppo.DistributedContext,
    ) -> torch.Tensor:
        assert _context is context
        assert torch.equal(
            tensor,
            torch.tensor(
                [[0.0, 0.0], [2.0, 2.0], [6.0, 2.0]],
                dtype=torch.float64,
            ),
        )
        return torch.tensor(
            [[10.0, 2.0], [4.0, 4.0], [36.0, 5.0]],
            dtype=torch.float64,
        )

    monkeypatch.setattr(ppo, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(ppo, "all_reduce_sum", fake_all_reduce_sum)

    metrics = ppo._mean_env_metrics(
        {"total_games_played": [1.0, 1.0], "z_metric": [2.0, 4.0]},
        context=context,
        device=torch.device("cpu"),
    )

    assert metrics == {
        "train/a_metric": 5.0,
        "train/total_games_played": 4.0,
        "train/z_metric": 7.2,
    }


def test_distributed_weighted_mean_uses_global_sum_and_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=2,
        initialized=True,
    )

    def fake_all_reduce_sum(
        tensor: torch.Tensor,
        _context: ppo.DistributedContext,
    ) -> torch.Tensor:
        assert _context is context
        return tensor + torch.tensor([800.0, 8.0], dtype=tensor.dtype)

    monkeypatch.setattr(ppo, "all_reduce_sum", fake_all_reduce_sum)

    actual = ppo._distributed_weighted_mean(
        torch.tensor([2.0, 4.0]),
        torch.tensor([1.0, 1.0]),
        context,
    )

    assert actual.item() == pytest.approx(80.6)


def test_distributed_backward_weighted_mean_scales_for_ddp_average(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=2,
        initialized=True,
    )

    def fake_all_reduce_sum(
        tensor: torch.Tensor,
        _context: ppo.DistributedContext,
    ) -> torch.Tensor:
        assert _context is context
        return tensor + torch.tensor(8.0, dtype=tensor.dtype)

    monkeypatch.setattr(ppo, "all_reduce_sum", fake_all_reduce_sum)
    values = torch.tensor([2.0, 4.0], requires_grad=True)

    loss = ppo._distributed_backward_weighted_mean(
        values,
        torch.tensor([1.0, 1.0]),
        context,
    )
    loss.backward()

    assert loss.item() == pytest.approx(1.2)
    assert torch.allclose(values.grad, torch.full_like(values, 0.2))


def test_player_segment_returns_preserve_per_player_terminal_rewards() -> None:
    rewards = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    value_mask = torch.tensor(
        [
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, False, False],
            ]
        ]
    )

    returns, return_mask = ppo._player_segment_returns(rewards, value_mask)

    assert torch.equal(returns, torch.tensor([[2.0, -1.0, 0.0, 0.0]]))
    assert torch.equal(return_mask, torch.tensor([[True, True, False, False]]))
    assert ppo._masked_reward_max(rewards, value_mask).item() == 1.0


def test_obs_to_device_uses_explicit_non_blocking_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 1)
    obs = _obs_batch(n_envs=2, obs_spec=obs_spec)
    non_blocking_args: list[bool] = []

    def fake_to(
        self: torch.Tensor,
        *args: Any,  # noqa: ARG001
        **kwargs: Any,
    ) -> torch.Tensor:
        non_blocking_args.append(kwargs["non_blocking"])
        return self

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    ppo._obs_to_device(obs, torch.device("cuda"), non_blocking=True)

    assert non_blocking_args == [True] * len(_obs_buffer_ptrs(obs))


def test_obs_to_device_defaults_to_blocking_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 1)
    obs = _obs_batch(n_envs=2, obs_spec=obs_spec)
    non_blocking_args: list[bool] = []

    def fake_to(
        self: torch.Tensor,
        *args: Any,  # noqa: ARG001
        **kwargs: Any,
    ) -> torch.Tensor:
        non_blocking_args.append(kwargs["non_blocking"])
        return self

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    ppo._obs_to_device(obs, torch.device("cuda"))

    assert non_blocking_args == [False] * len(_obs_buffer_ptrs(obs))


def test_actions_to_cpu_transfer_policy_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _actions(n_envs=2)
    non_blocking_args: list[bool] = []

    def fake_to(
        self: torch.Tensor,
        *args: Any,  # noqa: ARG001
        **kwargs: Any,
    ) -> torch.Tensor:
        non_blocking_args.append(kwargs["non_blocking"])
        return self

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    ppo._actions_to_cpu(actions)
    ppo._actions_to_cpu(actions, non_blocking=True)

    copied_fields = 3
    assert non_blocking_args == [False] * copied_fields + [True] * copied_fields


def test_trainer_sets_static_env_transfer_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TinyOrbitEnv(n_envs=2)
    env.pin_memory_enabled = True
    model = TinyOrbitModel()
    non_blocking_args: list[bool] = []

    def fake_obs_to_device(
        obs: ObsBatch,
        device: torch.device,  # noqa: ARG001
        *,
        non_blocking: bool = False,
    ) -> ObsBatch:
        non_blocking_args.append(non_blocking)
        return obs

    class FakeRolloutBuffer:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(ppo, "_obs_to_device", fake_obs_to_device)
    monkeypatch.setattr(ppo, "_PPORolloutBuffer", FakeRolloutBuffer)

    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(horizon=2),
        device=torch.device("cuda"),
    )

    assert trainer._non_blocking_env_to_device is True
    assert non_blocking_args == [True]


def test_collect_rollout_keeps_pre_step_obs_with_reused_cpu_buffers() -> None:
    torch.manual_seed(4)
    env = ReusingObservationEnv(n_envs=2)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=3,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )
    obs_ptrs = _obs_buffer_ptrs(trainer._obs)

    trainer._collect_rollout()
    segments = trainer.rollout.segment_major()

    assert obs_ptrs == _obs_buffer_ptrs(trainer._obs)
    expected_steps = torch.tensor([0.0, 1.0, 2.0])
    assert torch.equal(
        segments.obs.global_features[:, :, 1], expected_steps.expand(2, -1)
    )
    assert torch.equal(
        trainer._obs.global_features[:, 1],
        torch.full((2,), 3.0),
    )


def test_collect_rollout_does_not_call_teacher_model() -> None:
    torch.manual_seed(4)
    env = TinyOrbitEnv(n_envs=2)
    model = TinyOrbitModel()
    teacher = CountingForwardModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=3,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
        teacher_model=teacher,
        teacher_active=True,
    )

    trainer._collect_rollout()

    assert teacher.forward_calls == 0
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_trainer_smoke_keeps_metrics_finite_and_updates_parameters() -> None:
    torch.manual_seed(0)
    env = TinyOrbitEnv(n_envs=4, episode_length=3)
    model = TinyOrbitModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, eps=1e-5)
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=optimizer,
        config=ppo.PPOConfig(
            horizon=5,
            segments_per_minibatch=2,
            gamma=0.9,
            gae_lambda=0.95,
        ),
        device=torch.device("cpu"),
    )
    before = [param.detach().clone() for param in model.parameters()]

    metrics = trainer.train_iteration()

    for key in (
        "train/return_mean",
        "loss/total_loss",
        "loss/policy_loss",
        "loss/value_loss",
        "loss/entropy_loss",
        "policy/entropy",
        "policy/launch_entropy",
        "policy/approx_kl",
        "policy/clipfrac",
        "policy/ratio_mean",
        "policy/ratio_max",
        "policy/logratio_mean",
        "policy/logratio_abs_max",
        "policy/target_kl_exceeded",
        "policy/target_kl_exceeded_total",
        "optimizer/grad_norm",
        "optimizer/steps",
        "optimizer/learning_rate",
        "optimizer/minibatches_per_update",
        "train/policy_active_ratio",
        "train/advantage_mean",
        "train/advantage_std",
        "train/max_entities",
        "train/1p_rate",
        "train/2p_rate",
        "train/3p_rate",
        "train/4p_rate",
        "train/player_step_total",
        "time/rollout_seconds",
        "time/update_seconds",
        "perf/rollout_sps",
        "perf/update_sps",
    ):
        assert metrics[key] == pytest.approx(float(metrics[key]))
    assert any(
        not torch.allclose(param, old)
        for param, old in zip(model.parameters(), before, strict=True)
    )
    assert metrics["policy/launch_entropy"] == pytest.approx(metrics["policy/entropy"])
    assert metrics["policy/target_kl_exceeded"] == pytest.approx(0.0)
    assert metrics["policy/target_kl_exceeded_total"] == pytest.approx(0.0)
    assert metrics["train/max_entities"] == pytest.approx(2.0)
    assert metrics["train/1p_rate"] == pytest.approx(0.0)
    assert metrics["train/2p_rate"] == pytest.approx(0.0)
    assert metrics["train/3p_rate"] == pytest.approx(0.0)
    assert metrics["train/4p_rate"] == pytest.approx(1.0)
    assert metrics["train/player_step_total"] == pytest.approx(80.0)
    assert metrics["optimizer/steps"] == pytest.approx(2.0)
    assert metrics["optimizer/learning_rate"] == pytest.approx(0.05)

    next_metrics = trainer.train_iteration()

    assert next_metrics["train/player_step_total"] == pytest.approx(160.0)
    assert next_metrics["optimizer/steps"] == pytest.approx(4.0)


def test_trainer_total_games_played_is_cumulative() -> None:
    torch.manual_seed(0)
    env = TinyOrbitEnvWithMetrics(n_envs=4, episode_length=3)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.05, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=3,
            segments_per_minibatch=2,
        ),
        device=torch.device("cpu"),
    )

    metrics = trainer.train_iteration()
    next_metrics = trainer.train_iteration()

    assert metrics["train/total_games_played"] == pytest.approx(4.0)
    assert next_metrics["train/total_games_played"] == pytest.approx(8.0)
    assert trainer.total_games_played == 8


def test_trainer_accumulates_gradients_before_optimizer_step() -> None:
    torch.manual_seed(0)
    env = TinyOrbitEnv(n_envs=4, episode_length=3)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            segments_per_minibatch=1,
            gradient_accumulation_steps=2,
        ),
        device=torch.device("cpu"),
    )

    metrics = trainer.train_iteration()

    assert metrics["optimizer/minibatches_per_update"] == pytest.approx(4.0)
    assert metrics["optimizer/steps"] == pytest.approx(2.0)
    assert trainer.optimizer_steps == 2


def test_trainer_rejects_accumulation_group_that_does_not_divide_envs() -> None:
    env = TinyOrbitEnv(n_envs=4)
    model = TinyOrbitModel()

    with pytest.raises(
        ValueError,
        match=(
            "n_envs must be divisible by segments_per_minibatch "
            r"\* gradient_accumulation_steps"
        ),
    ):
        ppo.PPOTrainer(
            env=env,
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
            config=ppo.PPOConfig(
                horizon=2,
                segments_per_minibatch=2,
                gradient_accumulation_steps=3,
            ),
            device=torch.device("cpu"),
        )


def test_rollout_and_update_model_calls_run_under_autocast() -> None:
    torch.manual_seed(0)
    env = TinyOrbitEnv(n_envs=2, episode_length=3)
    model = AutocastRecordingModel("cpu")
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            dtype="bfloat16",
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )

    trainer._collect_rollout()
    segments = trainer.rollout.segment_major()
    trainer._update_minibatch(
        segments,
        advantages=torch.ones_like(segments.values),
        returns=torch.zeros_like(segments.values),
        policy_mask=torch.ones_like(segments.values, dtype=torch.bool),
        value_mask=torch.ones_like(segments.values, dtype=torch.bool),
        indices=torch.zeros((1,), dtype=torch.int64),
        value_clip_anchor=segments.values,
    )

    assert model.forward_autocast_enabled == [True, True]
    assert model.compute_value_autocast_enabled == [True]
    assert model.evaluate_autocast_enabled == [True]


def test_trainer_masks_inactive_player_slots() -> None:
    torch.manual_seed(1)
    env = TinyOrbitEnv(n_envs=3, episode_length=2, two_player=True)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=3,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )

    metrics = trainer.train_iteration()
    segments = trainer.rollout.segment_major()

    assert not segments.obs.still_playing[:, :, 2:].any()
    assert segments.dones[:, :, 2:].all()
    assert metrics["train/1p_rate"] == pytest.approx(0.0)
    assert metrics["train/2p_rate"] == pytest.approx(1.0)
    assert metrics["train/3p_rate"] == pytest.approx(0.0)
    assert metrics["train/4p_rate"] == pytest.approx(0.0)


def test_player_count_rates_count_raw_alive_players() -> None:
    still_playing = torch.tensor(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
            ],
            [
                [True, True, True, False],
                [True, True, True, True],
            ],
        ]
    )

    metrics = ppo._player_count_rates(still_playing)

    assert metrics["train/1p_rate"].item() == pytest.approx(0.25)
    assert metrics["train/2p_rate"].item() == pytest.approx(0.25)
    assert metrics["train/3p_rate"].item() == pytest.approx(0.25)
    assert metrics["train/4p_rate"].item() == pytest.approx(0.25)


def test_normalize_masked_advantages_uses_valid_policy_samples() -> None:
    normalized = ppo._normalize_masked_advantages(
        torch.tensor([[1.0, 3.0, 100.0]]),
        torch.tensor([[True, True, False]]),
    )

    assert torch.allclose(normalized[:, :2], torch.tensor([[-1.0, 1.0]]))
    assert torch.allclose(normalized[:, 2], torch.tensor([98.0]))


def test_normalize_masked_advantages_uses_distributed_policy_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=2,
        initialized=True,
    )

    def fake_all_reduce_sum(
        tensor: torch.Tensor,
        _context: ppo.DistributedContext,
    ) -> torch.Tensor:
        return tensor + torch.tensor([6.0, 36.0, 1.0])

    monkeypatch.setattr(ppo, "all_reduce_sum", fake_all_reduce_sum)

    normalized = ppo._normalize_masked_advantages(
        torch.tensor([[1.0, 3.0, 100.0]]),
        torch.tensor([[True, True, False]]),
        context=context,
    )

    mean = torch.tensor(10.0 / 3.0)
    variance = torch.tensor(46.0 / 3.0) - mean.pow(2)
    expected = (torch.tensor([[1.0, 3.0, 100.0]]) - mean) / variance.sqrt()
    assert torch.allclose(normalized, expected)


def test_update_minibatch_normalizes_policy_advantages_only() -> None:
    seen: dict[str, torch.Tensor] = {}
    env = TinyOrbitEnv(n_envs=1)
    model = FixedEvaluationModel(value=0.0)
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        config=ppo.PPOConfig(
            horizon=2,
            normalize_advantages=True,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )
    trainer._collect_rollout()
    segments = trainer.rollout.segment_major()

    def fake_ppo_loss(
        *,
        new_logp: torch.Tensor,
        entropy: torch.Tensor,  # noqa: ARG001
        new_values: torch.Tensor,
        old_logp: torch.Tensor,  # noqa: ARG001
        old_values: torch.Tensor,  # noqa: ARG001
        returns: torch.Tensor,
        advantages: torch.Tensor,
        policy_weight: torch.Tensor,
        value_weight: torch.Tensor,  # noqa: ARG001
        config: ppo.PPOConfig,  # noqa: ARG001
        context: ppo.DistributedContext | None = None,  # noqa: ARG001
    ) -> tuple[ppo._PPOLossMetrics, torch.Tensor]:
        seen["advantages"] = advantages.detach().clone()
        seen["returns"] = returns.detach().clone()
        loss = new_values.mean() + 0.0 * new_logp.mean() + 0.0 * policy_weight.mean()
        return _zero_loss_metrics(loss), loss

    trainer._ppo_loss = fake_ppo_loss
    trainer._update_minibatch(
        segments,
        advantages=torch.tensor([[[1.0, 3.0, 100.0, 200.0], [5.0, 7.0, 9.0, 11.0]]]),
        returns=torch.full_like(segments.values, 42.0),
        policy_mask=torch.tensor(
            [[[True, True, False, False], [False, False, False, False]]]
        ),
        value_mask=torch.ones_like(segments.values, dtype=torch.bool),
        indices=torch.zeros((1,), dtype=torch.int64),
        value_clip_anchor=segments.values,
    )

    assert torch.allclose(seen["advantages"][0, 0, :2], torch.tensor([-1.0, 1.0]))
    assert torch.equal(seen["returns"], torch.full_like(seen["returns"], 42.0))


def test_update_minibatch_value_clipping_uses_current_value_anchor() -> None:
    env = TinyOrbitEnv(n_envs=1)
    model = FixedEvaluationModel(value=12.0)
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        config=ppo.PPOConfig(
            horizon=2,
            vf_clip_coef=0.5,
            vf_coef=1.0,
            ent_coef=0.0,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )
    trainer._collect_rollout()
    rollout_segments = trainer.rollout.segment_major()
    segments = ppo._PPORolloutSegments(
        obs=rollout_segments.obs,
        actions=rollout_segments.actions,
        logp=torch.zeros_like(rollout_segments.logp),
        values=torch.zeros_like(rollout_segments.values),
        rewards=rollout_segments.rewards,
        dones=rollout_segments.dones,
    )
    update = trainer._update_minibatch(
        segments,
        advantages=torch.zeros_like(segments.values),
        returns=torch.full_like(segments.values, 10.0),
        policy_mask=torch.zeros_like(segments.values, dtype=torch.bool),
        value_mask=torch.ones_like(segments.values, dtype=torch.bool),
        indices=torch.zeros((1,), dtype=torch.int64),
        value_clip_anchor=torch.full_like(segments.values, 10.0),
    )

    assert update.metrics.value_loss.item() == pytest.approx(2.0)


def test_update_minibatch_steps_before_target_kl_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(6)
    env = TinyOrbitEnv(n_envs=1)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            target_kl=0.01,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )
    trainer._collect_rollout()
    segments = trainer.rollout.segment_major()

    def fake_ppo_loss(
        *,
        new_logp: torch.Tensor,
        entropy: torch.Tensor,  # noqa: ARG001
        new_values: torch.Tensor,  # noqa: ARG001
        old_logp: torch.Tensor,  # noqa: ARG001
        old_values: torch.Tensor,  # noqa: ARG001
        returns: torch.Tensor,  # noqa: ARG001
        advantages: torch.Tensor,  # noqa: ARG001
        policy_weight: torch.Tensor,  # noqa: ARG001
        value_weight: torch.Tensor,  # noqa: ARG001
        config: ppo.PPOConfig,  # noqa: ARG001
        context: ppo.DistributedContext | None = None,  # noqa: ARG001
    ) -> tuple[ppo._PPOLossMetrics, torch.Tensor]:
        loss = new_logp.sum()
        return (
            replace(
                _zero_loss_metrics(loss),
                approx_kl=loss.detach().new_tensor(0.02),
            ),
            loss,
        )

    monkeypatch.setattr(trainer, "_ppo_loss", fake_ppo_loss)

    update = trainer._update_minibatch(
        segments,
        advantages=torch.ones_like(segments.values),
        returns=torch.zeros_like(segments.values),
        policy_mask=ppo._policy_mask(segments.obs),
        value_mask=segments.obs.still_playing,
        indices=torch.zeros((1,), dtype=torch.int64),
        value_clip_anchor=segments.values,
    )

    assert update.target_kl_exceeded
    assert trainer.optimizer_steps == 1
    assert update.grad_norm.item() > 0.0


def test_ppo_epoch_one_uses_shuffled_single_pass_minibatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    env = TinyOrbitEnv(n_envs=6)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            segments_per_minibatch=2,
        ),
        device=torch.device("cpu"),
    )
    trainer._collect_rollout()
    segments = trainer.rollout.segment_major()
    policy_mask = ppo._policy_mask(segments.obs)
    value_mask = segments.obs.still_playing
    advantages = torch.ones_like(segments.rewards)
    returns = advantages + segments.values
    seen: list[torch.Tensor] = []

    def fake_update_minibatch(
        segments: ppo._PPORolloutSegments,
        advantages: torch.Tensor,  # noqa: ARG001
        returns: torch.Tensor,  # noqa: ARG001
        policy_mask: torch.Tensor,  # noqa: ARG001
        value_mask: torch.Tensor,  # noqa: ARG001
        indices: torch.Tensor,
        *,
        value_clip_anchor: torch.Tensor,  # noqa: ARG001
        loss_scale: float = 1.0,  # noqa: ARG001
        step_optimizer: bool = True,  # noqa: ARG001
    ) -> ppo._PPOUpdateResult:
        seen.append(indices.detach().clone())
        zero = segments.logp.new_zeros(())
        return ppo._PPOUpdateResult(
            metrics=_zero_loss_metrics(zero),
            indices=indices,
            new_values=segments.values[indices],
            grad_norm=zero,
        )

    monkeypatch.setattr(trainer, "_update_minibatch", fake_update_minibatch)

    metrics, sampled_segments = trainer._update(
        segments,
        advantages,
        returns,
        policy_mask,
        value_mask,
    )

    assert [indices.shape for indices in seen] == [(2,), (2,), (2,)]
    all_indices = torch.cat(seen)
    assert torch.equal(all_indices.sort().values, torch.arange(6))
    assert metrics["optimizer/minibatches_per_update"] == pytest.approx(3.0)
    assert "optimizer/num_minibatches" not in metrics
    assert "optimizer/num_total_minibatches" not in metrics
    assert "sampling/minibatch_exposure" not in metrics
    assert sampled_segments == 6
    assert metrics["policy/target_kl_exceeded"] == pytest.approx(0.0)


def test_update_uses_no_sync_for_gradient_accumulation_microbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11)
    env = TinyOrbitEnv(n_envs=4)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            segments_per_minibatch=1,
            gradient_accumulation_steps=2,
        ),
        device=torch.device("cpu"),
    )
    trainer._collect_rollout()
    segments = trainer.rollout.segment_major()
    policy_mask = ppo._policy_mask(segments.obs)
    value_mask = segments.obs.still_playing
    advantages = torch.ones_like(segments.rewards)
    returns = advantages + segments.values
    context_enabled: list[bool] = []
    minibatch_context_enabled: list[bool] = []
    active_contexts: list[bool] = []

    @contextmanager
    def fake_model_no_sync_context(
        model_arg: BaseModelAPI,
        *,
        enabled: bool,
    ) -> object:
        assert model_arg is model
        context_enabled.append(enabled)
        active_contexts.append(enabled)
        try:
            yield
        finally:
            active_contexts.pop()

    def fake_update_minibatch(
        segments: ppo._PPORolloutSegments,
        advantages: torch.Tensor,  # noqa: ARG001
        returns: torch.Tensor,  # noqa: ARG001
        policy_mask: torch.Tensor,  # noqa: ARG001
        value_mask: torch.Tensor,  # noqa: ARG001
        indices: torch.Tensor,
        *,
        value_clip_anchor: torch.Tensor,  # noqa: ARG001
        loss_scale: float = 1.0,  # noqa: ARG001
        step_optimizer: bool = True,  # noqa: ARG001
    ) -> ppo._PPOUpdateResult:
        assert len(active_contexts) == 1
        minibatch_context_enabled.append(active_contexts[-1])
        zero = segments.logp.new_zeros(())
        return ppo._PPOUpdateResult(
            metrics=_zero_loss_metrics(zero),
            indices=indices,
            new_values=segments.values[indices],
            grad_norm=zero,
        )

    monkeypatch.setattr(ppo, "model_no_sync_context", fake_model_no_sync_context)
    monkeypatch.setattr(trainer, "_update_minibatch", fake_update_minibatch)

    trainer._update(
        segments,
        advantages,
        returns,
        policy_mask,
        value_mask,
    )

    assert context_enabled == [True, False, True, False]
    assert minibatch_context_enabled == [True, False, True, False]


def test_update_reports_target_kl_guard_when_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(9)
    env = TinyOrbitEnv(n_envs=4)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            target_kl=0.01,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )
    trainer._collect_rollout()
    segments = trainer.rollout.segment_major()
    policy_mask = ppo._policy_mask(segments.obs)
    value_mask = segments.obs.still_playing
    advantages = torch.ones_like(segments.rewards)
    returns = advantages + segments.values
    update_calls = 0

    def fake_update_minibatch(
        segments: ppo._PPORolloutSegments,
        advantages: torch.Tensor,  # noqa: ARG001
        returns: torch.Tensor,  # noqa: ARG001
        policy_mask: torch.Tensor,  # noqa: ARG001
        value_mask: torch.Tensor,  # noqa: ARG001
        indices: torch.Tensor,
        *,
        value_clip_anchor: torch.Tensor,  # noqa: ARG001
        loss_scale: float = 1.0,  # noqa: ARG001
        step_optimizer: bool = True,  # noqa: ARG001
    ) -> ppo._PPOUpdateResult:
        nonlocal update_calls
        update_calls += 1
        zero = segments.logp.new_zeros(())
        metrics = replace(_zero_loss_metrics(zero), approx_kl=zero + 0.02)
        return ppo._PPOUpdateResult(
            metrics=metrics,
            indices=indices,
            new_values=segments.values[indices],
            grad_norm=zero,
            target_kl_exceeded=True,
        )

    monkeypatch.setattr(trainer, "_update_minibatch", fake_update_minibatch)

    metrics, sampled_segments = trainer._update(
        segments,
        advantages,
        returns,
        policy_mask,
        value_mask,
    )

    assert update_calls == 1
    assert metrics["policy/target_kl_exceeded"] == pytest.approx(1.0)
    assert metrics["policy/target_kl_exceeded_total"] == pytest.approx(1.0)
    assert sampled_segments == 1
    assert trainer.target_kl_exceeded_total == 1


def test_train_iteration_update_sps_uses_actual_segments_when_target_kl_stops_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(9)
    env = TinyOrbitEnv(n_envs=4)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            target_kl=0.01,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )

    def fake_update_minibatch(
        segments: ppo._PPORolloutSegments,
        advantages: torch.Tensor,  # noqa: ARG001
        returns: torch.Tensor,  # noqa: ARG001
        policy_mask: torch.Tensor,  # noqa: ARG001
        value_mask: torch.Tensor,  # noqa: ARG001
        indices: torch.Tensor,
        *,
        value_clip_anchor: torch.Tensor,  # noqa: ARG001
        loss_scale: float = 1.0,  # noqa: ARG001
        step_optimizer: bool = True,  # noqa: ARG001
    ) -> ppo._PPOUpdateResult:
        zero = segments.logp.new_zeros(())
        metrics = replace(_zero_loss_metrics(zero), approx_kl=zero + 0.02)
        return ppo._PPOUpdateResult(
            metrics=metrics,
            indices=indices,
            new_values=segments.values[indices],
            grad_norm=zero,
            target_kl_exceeded=True,
        )

    times = iter([0.0, 0.0, 2.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(ppo, "perf_counter", lambda: next(times))
    monkeypatch.setattr(trainer, "_update_minibatch", fake_update_minibatch)

    metrics = trainer.train_iteration()

    assert metrics["policy/target_kl_exceeded"] == pytest.approx(1.0)
    assert metrics["policy/target_kl_exceeded_total"] == pytest.approx(1.0)
    assert "sampling/minibatch_exposure" not in metrics
    assert metrics["perf/update_sps"] == pytest.approx(2.0)


def test_minibatch_indices_repeat_uniform_single_pass_for_each_ppo_epoch() -> None:
    config = ppo.PPOConfig(ppo_epochs=2, segments_per_minibatch=2)
    samples = ppo._minibatch_indices(
        config=config,
        n_segments=4,
        device=torch.device("cpu"),
    )

    assert [sample.shape for sample in samples] == [(2,), (2,), (2,), (2,)]
    for epoch_samples in (samples[:2], samples[2:]):
        epoch_indices = torch.cat(epoch_samples)
        assert torch.equal(epoch_indices.sort().values, torch.arange(4))


def test_ppo_config_defaults_target_kl() -> None:
    assert ppo.PPOConfig().target_kl == pytest.approx(0.03)


def test_trainer_compile_mode_compiles_only_tensor_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_compile(target: Any, *, mode: str) -> Any:
        calls.append((target.__name__, mode))
        return target

    monkeypatch.setattr(ppo.torch, "compile", fake_compile)
    env = TinyOrbitEnv(n_envs=2)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            segments_per_minibatch=1,
            compile_mode="default",
        ),
        device=torch.device("cpu"),
    )

    assert calls == [
        ("_compute_gae_tensors", "default"),
        ("_ppo_loss_components", "default"),
    ]
    metrics = trainer.train_iteration()
    assert metrics["loss/teacher_kl_loss"] == pytest.approx(0.0)
    assert metrics["loss/teacher_value_loss"] == pytest.approx(0.0)


def test_trainer_overwrites_dones_when_envs_terminate_inside_rollout() -> None:
    torch.manual_seed(3)
    env = TinyOrbitEnv(n_envs=3, episode_length=1)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=4,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )

    trainer.train_iteration()
    segments = trainer.rollout.segment_major()

    assert torch.equal(
        segments.dones,
        torch.ones((env.n_envs, trainer.config.horizon, 4), dtype=torch.bool),
    )
    assert torch.isfinite(segments.rewards).all()


def test_discrete_target_rollout_buffer_and_policy_mask() -> None:
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 2)
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    buffer = ppo._PPORolloutBuffer(
        horizon=2,
        n_envs=3,
        obs_spec=obs_spec,
        action_spec=action_spec,
        device=torch.device("cpu"),
    )
    obs = _discrete_obs_batch(n_envs=3, obs_spec=obs_spec)
    obs.still_playing[:, :2] = True
    obs.still_playing[:, 2:] = False
    obs.action_mask.can_act[:, 0, 0, 1] = True
    obs.action_mask.max_launch[:, 0, 0] = 3
    actions = _actions(3, max_launches=1, kind="discrete_targets")
    actions.launch[:, 0, 0, 0] = True
    assert actions.target is not None
    actions.target[:, 0, 0, 0] = 1
    actions.ships[:, 0, 0, 0] = 1

    buffer.write_step(
        0,
        obs=obs,
        actions=actions,
        logp=torch.zeros((3, 4)),
        values=torch.zeros((3, 4)),
        rewards=torch.zeros((3, 4)),
        dones=torch.zeros((3, 4), dtype=torch.bool),
    )
    segments = buffer.segment_major()

    assert segments.obs.action_mask.can_act.shape == (
        3,
        2,
        4,
        ACTION_ENTITY_SLOTS,
        ACTION_ENTITY_SLOTS,
    )
    assert segments.actions.target is not None
    assert segments.actions.target.shape == (3, 2, 4, ACTION_ENTITY_SLOTS, 1)
    assert segments.entity_logp is not None
    assert segments.entity_logp.shape == (3, 2, 4, ACTION_ENTITY_SLOTS)
    policy_mask = ppo._policy_mask(segments.obs)
    assert policy_mask.shape == (3, 2, 4)
    assert policy_mask[:, 0, 0].all()
    assert not policy_mask[:, 0, 1:].any()


def test_discrete_target_bin_rollout_buffer_and_policy_mask() -> None:
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 2)
    action_spec = ActionDiscreteTargetBinsConfig(n_bins=7)
    buffer = ppo._PPORolloutBuffer(
        horizon=2,
        n_envs=3,
        obs_spec=obs_spec,
        action_spec=action_spec,
        device=torch.device("cpu"),
    )
    obs = _obs_batch(n_envs=3, obs_spec=obs_spec)
    obs.action_mask = DiscreteTargetBinActionMask(
        can_act=torch.zeros(
            (3, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS, 7),
            dtype=torch.bool,
        )
    )
    obs.action_mask.can_act[:, 0, 0, 1, [0, 6]] = True
    actions = _actions(3, kind="discrete_target_bins")
    assert actions.fleet_bin is not None
    actions.target[:, 0, 0] = 1
    actions.fleet_bin[:, 0, 0] = 6

    buffer.write_step(
        0,
        obs=obs,
        actions=actions,
        logp=torch.zeros((3, 4)),
        values=torch.zeros((3, 4)),
        rewards=torch.zeros((3, 4)),
        dones=torch.zeros((3, 4), dtype=torch.bool),
    )
    segments = buffer.segment_major()

    assert isinstance(segments.obs.action_mask, DiscreteTargetBinActionMask)
    assert segments.obs.action_mask.can_act.shape == (
        3,
        2,
        4,
        ACTION_ENTITY_SLOTS,
        ACTION_ENTITY_SLOTS,
        7,
    )
    assert isinstance(segments.actions, DiscreteTargetBinActions)
    assert segments.actions.target.shape == (3, 2, 4, ACTION_ENTITY_SLOTS)
    assert segments.actions.fleet_bin.shape == (3, 2, 4, ACTION_ENTITY_SLOTS)
    policy_mask = ppo._policy_mask(segments.obs)
    assert policy_mask.shape == (3, 2, 4)
    assert policy_mask[:, 0, 0].all()
    assert not policy_mask[:, 0, 1:].any()


def test_step_env_passes_discrete_target_tensor() -> None:
    env = TinyDiscreteTargetEnv(n_envs=2)
    actions = _actions(2, max_launches=1, kind="discrete_targets")
    actions.launch[:, :, 0, 0] = True
    assert actions.target is not None
    actions.target[:, :, 0, 0] = 1
    actions.ships[:, :, 0, 0] = 1

    _obs, rewards, _dones, metrics = ppo._step_env(env, actions)

    assert metrics == {}
    assert env.last_target is not None
    assert actions.target is not None
    assert torch.equal(env.last_target, actions.target)
    assert rewards.shape == (2, 4)


def test_discrete_target_train_iteration_runs() -> None:
    env = TinyDiscreteTargetEnv(n_envs=2)
    model = TinyDiscreteTargetModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )

    metrics = trainer.train_iteration()

    assert env.last_target is not None
    for key in (
        "loss/total_loss",
        "policy/entropy",
        "policy/launch_entropy",
        "optimizer/grad_norm",
    ):
        assert metrics[key] == pytest.approx(float(metrics[key]))


def test_discrete_target_transformer_train_iteration_keeps_parameters_finite() -> None:
    torch.manual_seed(0)
    obs_spec = EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 2)
    action_spec = ActionDiscreteTargetsConfig(max_per_planet_launches=1)
    env = VectorizedEnv(
        n_envs=2,
        obs_spec=obs_spec,
        action_spec=action_spec,
        two_player_weight=1.0,
        pin_memory=False,
    )
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(
            actor=ActorDiscreteTargetsConfig(
                n_action_mixtures=2,
                entropy_ship_quantiles=8,
            ),
            embed_dim=32,
            depth=1,
            n_heads=4,
        ),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    model.reset_parameters()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )

    metrics = trainer.train_iteration()

    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert metrics["policy/entropy"] == pytest.approx(
        metrics["policy/launch_entropy"]
        + metrics["policy/target_entropy"]
        + metrics["policy/fleet_size_full_entropy"]
    )
    assert "policy/fleet_size_mixture_entropy" in metrics
    assert "policy/fleet_size_logistic_entropy" in metrics
    assert "policy/event_entropy" not in metrics
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_stateless_transformer_train_iteration_runs_with_teacher() -> None:
    torch.manual_seed(0)
    env = TinyDiscreteTargetEnv(n_envs=2)
    model_config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(
            n_action_mixtures=2,
            entropy_ship_quantiles=8,
        ),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = StatelessTransformerV1(
        model_config,
        obs_spec=env.obs_spec,
        action_spec=env.action_spec,
    )
    teacher_model = StatelessTransformerV1(
        model_config,
        obs_spec=env.obs_spec,
        action_spec=env.action_spec,
    )
    model.reset_parameters()
    teacher_model.reset_parameters()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
        teacher_model=teacher_model,
        teacher_active=True,
    )

    metrics = trainer.train_iteration()

    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert metrics["teacher/kl"] >= 0.0
    assert metrics["teacher/value_cross_entropy"] >= 0.0


def test_teacher_update_uses_combined_student_pass_and_no_grad_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    env = TinyDiscreteTargetEnv(n_envs=2)
    model_config = StatelessTransformerV1Config(
        actor=ActorDiscreteTargetsConfig(
            n_action_mixtures=2,
            entropy_ship_quantiles=8,
        ),
        embed_dim=32,
        depth=1,
        n_heads=4,
    )
    model = StatelessTransformerV1(
        model_config,
        obs_spec=env.obs_spec,
        action_spec=env.action_spec,
    )
    teacher_model = StatelessTransformerV1(
        model_config,
        obs_spec=env.obs_spec,
        action_spec=env.action_spec,
    )
    model.reset_parameters()
    teacher_model.reset_parameters()
    combined_calls = 0
    student_actor_input_grad_enabled: list[bool] = []
    teacher_encode_grad_enabled: list[bool] = []
    teacher_actor_input_grad_enabled: list[bool] = []
    original_combined = model.evaluate_actions_with_teacher
    original_student_actor_inputs = model._discrete_actor_inputs
    original_teacher_encode = teacher_model.encode_observations
    original_teacher_actor_inputs = teacher_model._discrete_actor_inputs

    def combined_wrapper(*args: object, **kwargs: object) -> object:
        nonlocal combined_calls
        combined_calls += 1
        return original_combined(*args, **kwargs)

    def fail_evaluate_actions(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("student evaluate_actions should not be called")

    def fail_evaluate_action_kl(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("student evaluate_action_kl should not be called")

    def fail_teacher_evaluate_actions(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("teacher evaluate_actions should not be called")

    def student_actor_inputs_wrapper(*args: object, **kwargs: object) -> object:
        student_actor_input_grad_enabled.append(torch.is_grad_enabled())
        return original_student_actor_inputs(*args, **kwargs)

    def teacher_encode_wrapper(*args: object, **kwargs: object) -> object:
        teacher_encode_grad_enabled.append(torch.is_grad_enabled())
        return original_teacher_encode(*args, **kwargs)

    def teacher_actor_inputs_wrapper(*args: object, **kwargs: object) -> object:
        teacher_actor_input_grad_enabled.append(torch.is_grad_enabled())
        return original_teacher_actor_inputs(*args, **kwargs)

    monkeypatch.setattr(model, "evaluate_actions_with_teacher", combined_wrapper)
    monkeypatch.setattr(model, "evaluate_actions", fail_evaluate_actions)
    monkeypatch.setattr(model, "evaluate_action_kl", fail_evaluate_action_kl)
    monkeypatch.setattr(
        model,
        "_discrete_actor_inputs",
        student_actor_inputs_wrapper,
    )
    monkeypatch.setattr(
        teacher_model,
        "evaluate_actions",
        fail_teacher_evaluate_actions,
    )
    monkeypatch.setattr(teacher_model, "encode_observations", teacher_encode_wrapper)
    monkeypatch.setattr(
        teacher_model,
        "_discrete_actor_inputs",
        teacher_actor_inputs_wrapper,
    )
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
        teacher_model=teacher_model,
        teacher_active=True,
    )

    trainer.train_iteration()

    assert combined_calls > 0
    assert sum(student_actor_input_grad_enabled) == combined_calls
    assert teacher_encode_grad_enabled
    assert not any(teacher_encode_grad_enabled)
    assert teacher_actor_input_grad_enabled
    assert not any(teacher_actor_input_grad_enabled)


def test_teacher_update_skips_teacher_when_coefficients_are_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    env = TinyDiscreteTargetEnv(n_envs=2)
    model = TinyDiscreteTargetModel()
    teacher_model = TinyDiscreteTargetModel()
    evaluate_calls = 0

    original_evaluate_actions = model.evaluate_actions

    def evaluate_actions_wrapper(*args: object, **kwargs: object) -> object:
        nonlocal evaluate_calls
        evaluate_calls += 1
        return original_evaluate_actions(*args, **kwargs)

    def fail_combined_teacher_eval(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("teacher evaluation should be skipped")

    monkeypatch.setattr(model, "evaluate_actions", evaluate_actions_wrapper)
    monkeypatch.setattr(
        model,
        "evaluate_actions_with_teacher",
        fail_combined_teacher_eval,
    )
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            segments_per_minibatch=1,
            teacher_kl_coef=0.0,
            teacher_value_coef=0.0,
        ),
        device=torch.device("cpu"),
        teacher_model=teacher_model,
        teacher_active=True,
    )

    metrics = trainer.train_iteration()

    assert evaluate_calls > 0
    assert metrics["loss/teacher_kl_loss"] == pytest.approx(0.0)
    assert metrics["loss/teacher_value_loss"] == pytest.approx(0.0)


def test_per_entity_teacher_update_skips_zero_kl_without_shape_error() -> None:
    env = TinyOrbitEnv(n_envs=2)
    model = TinyOrbitModel()
    teacher_model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            ppo_clip_mode="per_entity",
            segments_per_minibatch=1,
            teacher_kl_coef=0.0,
            teacher_value_coef=0.0,
        ),
        device=torch.device("cpu"),
        teacher_model=teacher_model,
        teacher_active=True,
    )

    metrics = trainer.train_iteration()

    assert metrics["loss/teacher_kl_loss"] == pytest.approx(0.0)
    assert metrics["loss/teacher_value_loss"] == pytest.approx(0.0)


def test_detach_loss_metrics_drops_metric_graphs() -> None:
    value = torch.tensor(1.0, requires_grad=True)
    metrics = replace(
        _zero_loss_metrics(value * 2.0),
        entropy_components={"launch": value * 3.0},
        teacher_kl_components={"launch": value * 4.0},
    )

    detached = ppo._detach_loss_metrics(metrics)

    metric_values = (
        detached.loss,
        detached.policy_loss,
        detached.value_loss,
        detached.entropy_loss,
        detached.teacher_kl_loss,
        detached.teacher_value_loss,
        detached.entropy,
        detached.teacher_kl,
        detached.teacher_value_cross_entropy,
        detached.approx_kl,
        detached.clipfrac,
        detached.ratio_mean,
        detached.ratio_max,
        detached.logratio_mean,
        detached.logratio_abs_max,
        detached.entropy_components["launch"],
        detached.teacher_kl_components["launch"],
    )
    assert not any(metric.requires_grad for metric in metric_values)


def test_trainer_rejects_recurrent_teacher() -> None:
    torch.manual_seed(0)
    env = TinyDiscreteTargetEnv(n_envs=2)
    model = TinyDiscreteTargetModel()
    teacher_model = RecurrentTransformerV1(
        RecurrentTransformerV1Config(
            actor=ActorDiscreteTargetsConfig(
                n_action_mixtures=2,
                entropy_ship_quantiles=8,
            ),
            embed_dim=16,
            depth=1,
            n_heads=4,
        ),
        obs_spec=env.obs_spec,
        action_spec=env.action_spec,
    )
    teacher_model.reset_parameters()

    with pytest.raises(ValueError, match="recurrent hidden state"):
        ppo.PPOTrainer(
            env=env,
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
            config=ppo.PPOConfig(
                horizon=2,
                segments_per_minibatch=1,
            ),
            device=torch.device("cpu"),
            teacher_model=teacher_model,
            teacher_active=True,
        )


def test_recurrent_transformer_train_iteration_keeps_parameters_finite() -> None:
    torch.manual_seed(0)
    env = TinyDiscreteTargetEnv(n_envs=2)
    model = RecurrentTransformerV1(
        RecurrentTransformerV1Config(
            actor=ActorDiscreteTargetsConfig(
                n_action_mixtures=2,
                entropy_ship_quantiles=8,
            ),
            embed_dim=16,
            depth=1,
            n_heads=4,
        ),
        obs_spec=env.obs_spec,
        action_spec=env.action_spec,
    )
    model.reset_parameters()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            segments_per_minibatch=1,
        ),
        device=torch.device("cpu"),
    )

    metrics = trainer.train_iteration()

    assert env.last_target is not None
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert trainer.rollout.initial_hidden_state is not None
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
