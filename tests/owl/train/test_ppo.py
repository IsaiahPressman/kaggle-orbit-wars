from __future__ import annotations

from typing import Any

import pytest
import torch
from owl.model import (
    BaseModelAPI,
    ModelActionEntropies,
    ModelActionLogProbs,
    ModelActions,
    ModelEvaluation,
    ModelOutput,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    ActionPureConfig,
    ObsBatch,
    ObsV1Config,
)
from owl.train import ppo
from torch import nn


def _obs_batch(*, n_envs: int, obs_spec: ObsV1Config) -> ObsBatch:
    return ObsBatch(
        planets=torch.zeros(
            (n_envs, obs_spec.max_planets, obs_spec.planet_channels),
            dtype=torch.float32,
        ),
        fleets=torch.zeros(
            (n_envs, obs_spec.max_fleets, obs_spec.fleet_channels),
            dtype=torch.float32,
        ),
        comets=torch.zeros(
            (n_envs, obs_spec.max_comets, obs_spec.comet_channels),
            dtype=torch.float32,
        ),
        planet_mask=torch.zeros((n_envs, obs_spec.max_planets), dtype=torch.bool),
        fleet_mask=torch.zeros((n_envs, obs_spec.max_fleets), dtype=torch.bool),
        comet_mask=torch.zeros((n_envs, obs_spec.max_comets), dtype=torch.bool),
        still_playing=torch.ones((n_envs, 4), dtype=torch.bool),
        global_features=torch.zeros(
            (n_envs, obs_spec.global_channels),
            dtype=torch.float32,
        ),
        can_act=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
        max_launch=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64),
    )


def _actions(
    n_envs: int,
    max_launches: int = ActionPureConfig().max_per_planet_launches,
) -> ModelActions:
    shape = (n_envs, 4, ACTION_ENTITY_SLOTS, max_launches)
    return ModelActions(
        launch=torch.zeros(shape, dtype=torch.bool),
        angle=torch.zeros(shape, dtype=torch.float32),
        ships=torch.zeros(shape, dtype=torch.int64),
    )


class TinyOrbitEnv:
    def __init__(
        self,
        *,
        n_envs: int,
        episode_length: int = 3,
        two_player: bool = False,
    ) -> None:
        self.n_envs = n_envs
        self.obs_spec = ObsV1Config(max_entities=ACTION_ENTITY_SLOTS + 2)
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
        launch: torch.Tensor,
        angle: torch.Tensor,
        ships: torch.Tensor,
    ) -> tuple[ObsBatch, torch.Tensor, torch.Tensor]:
        del angle, ships
        active = self._still_playing()
        player_launch = launch[:, :, 0, 0].to(dtype=torch.float32)
        reward = torch.where(player_launch.eq(self._targets[:, None]), 1.0, -0.25)
        reward = torch.where(active, reward, torch.zeros_like(reward))
        self._steps += 1
        done = self._steps >= self.episode_length
        self._steps[done] = 0
        dones = done[:, None].expand(-1, 4) | ~active
        return self._obs(), reward, dones

    def _obs(self) -> ObsBatch:
        obs = _obs_batch(n_envs=self.n_envs, obs_spec=self.obs_spec)
        obs.global_features[:, 0] = self._targets
        obs.global_features[:, 1] = self._steps.to(torch.float32) / float(
            self.episode_length
        )
        obs.still_playing = self._still_playing()
        obs.planet_mask[:, :2] = True
        obs.can_act[:, :, 0] = obs.still_playing
        obs.max_launch[:, :, 0] = obs.still_playing.to(torch.int64)
        return obs

    def _still_playing(self) -> torch.Tensor:
        still_playing = torch.ones((self.n_envs, 4), dtype=torch.bool)
        if self.two_player:
            still_playing[:, 2:] = False
        return still_playing


class ReusingObservationEnv(TinyOrbitEnv):
    def __init__(self, *, n_envs: int, episode_length: int = 10) -> None:
        super().__init__(n_envs=n_envs, episode_length=episode_length)
        self._obs_storage = _obs_batch(n_envs=n_envs, obs_spec=self.obs_spec)

    def _obs(self) -> ObsBatch:
        obs = self._obs_storage
        obs.planets.zero_()
        obs.fleets.zero_()
        obs.comets.zero_()
        obs.planet_mask.zero_()
        obs.fleet_mask.zero_()
        obs.comet_mask.zero_()
        obs.still_playing.zero_()
        obs.global_features.zero_()
        obs.can_act.zero_()
        obs.max_launch.zero_()
        obs.global_features[:, 0] = self._targets
        obs.global_features[:, 1] = self._steps.to(torch.float32)
        obs.still_playing.copy_(self._still_playing())
        obs.planet_mask[:, :2] = True
        obs.can_act[:, :, 0] = obs.still_playing
        obs.max_launch[:, :, 0] = obs.still_playing.to(torch.int64)
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
        actions: ModelActions,
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

    def get_input_layers(self) -> tuple[nn.Module, ...]:
        return (self.input_proj,)

    def get_output_layers(self) -> tuple[nn.Module, ...]:
        return (self.policy, self.value)

    @staticmethod
    def _log_probs(per_player: torch.Tensor) -> ModelActionLogProbs:
        n_envs = per_player.shape[0]
        action_shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
        launch = torch.zeros(action_shape, dtype=per_player.dtype)
        angle_and_size = torch.zeros_like(launch)
        per_player_entity = torch.zeros(
            (n_envs, 4, ACTION_ENTITY_SLOTS), dtype=per_player.dtype
        )
        launch[:, :, 0, 0] = per_player
        per_player_entity[:, :, 0] = per_player
        return ModelActionLogProbs(
            launch=launch,
            angle_and_size=angle_and_size,
            per_player_entity=per_player_entity,
        )

    @staticmethod
    def _entropies(per_player: torch.Tensor) -> ModelActionEntropies:
        n_envs = per_player.shape[0]
        action_shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
        launch = torch.zeros(action_shape, dtype=per_player.dtype)
        angle_and_size = torch.zeros_like(launch)
        per_player_entity = torch.zeros(
            (n_envs, 4, ACTION_ENTITY_SLOTS), dtype=per_player.dtype
        )
        launch[:, :, 0, 0] = per_player
        per_player_entity[:, :, 0] = per_player
        return ModelActionEntropies(
            launch=launch,
            angle_and_size=angle_and_size,
            per_player_entity=per_player_entity,
        )


def _zero_loss_metrics(zero: torch.Tensor) -> ppo.PPOLossMetrics:
    return ppo.PPOLossMetrics(
        loss=zero,
        policy_loss=zero,
        value_loss=zero,
        entropy=zero,
        approx_kl=zero,
        clipfrac=zero,
        ratio_mean=zero,
        ratio_max=zero,
        logratio_mean=zero,
        logratio_abs_max=zero,
    )


def test_rollout_buffer_collects_time_major_and_returns_contiguous_segments() -> None:
    obs_spec = ObsV1Config(max_entities=ACTION_ENTITY_SLOTS + 1)
    action_spec = ActionPureConfig()
    buffer = ppo.PPORolloutBuffer(
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
    obs_spec = ObsV1Config(max_entities=ACTION_ENTITY_SLOTS + 1)
    obs = _obs_batch(n_envs=2, obs_spec=obs_spec)
    obs.global_features.fill_(1.0)

    copied = ppo._obs_to_device(obs, torch.device("cpu"))
    obs.global_features.fill_(2.0)

    assert torch.equal(copied.global_features, torch.ones_like(copied.global_features))
    for field in ppo._OBS_FIELDS:
        assert getattr(copied, field).data_ptr() != getattr(obs, field).data_ptr()


def test_obs_to_device_uses_non_blocking_for_pinned_cuda_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs_spec = ObsV1Config(max_entities=ACTION_ENTITY_SLOTS + 1)
    obs = _obs_batch(n_envs=2, obs_spec=obs_spec)
    non_blocking_args: list[bool] = []

    def fake_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args
        non_blocking_args.append(kwargs["non_blocking"])
        return self

    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda _self: True)
    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    ppo._obs_to_device(obs, torch.device("cuda"))

    assert non_blocking_args == [True] * len(ppo._OBS_FIELDS)


def test_obs_to_device_blocks_for_unpinned_cuda_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs_spec = ObsV1Config(max_entities=ACTION_ENTITY_SLOTS + 1)
    obs = _obs_batch(n_envs=2, obs_spec=obs_spec)
    non_blocking_args: list[bool] = []

    def fake_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args
        non_blocking_args.append(kwargs["non_blocking"])
        return self

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    ppo._obs_to_device(obs, torch.device("cuda"))

    assert non_blocking_args == [False] * len(ppo._OBS_FIELDS)


def test_actions_to_cpu_transfer_policy_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _actions(n_envs=2)
    non_blocking_args: list[bool] = []

    def fake_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args
        non_blocking_args.append(kwargs["non_blocking"])
        return self

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    ppo._actions_to_cpu(actions)
    ppo._actions_to_cpu(actions, non_blocking=True)

    assert non_blocking_args == [False] * len(ppo._ACTION_FIELDS) + [True] * len(
        ppo._ACTION_FIELDS
    )


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
            n_envs=2,
            segment_sampling=ppo.SegmentSamplingConfig(segments_per_minibatch=1),
        ),
        device=torch.device("cpu"),
    )

    trainer._collect_rollout()
    segments = trainer.rollout.segment_major()

    expected_steps = torch.tensor([0.0, 1.0, 2.0])
    assert torch.equal(
        segments.obs.global_features[:, :, 1], expected_steps.expand(2, -1)
    )
    assert torch.equal(
        trainer._obs.global_features[:, 1],
        torch.full((2,), 3.0),
    )


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
            n_envs=4,
            segment_sampling=ppo.SegmentSamplingConfig(segments_per_minibatch=2),
            update_epochs=2,
            gamma=0.9,
            gae_lambda=0.95,
        ),
        device=torch.device("cpu"),
    )
    before = [param.detach().clone() for param in model.parameters()]

    metrics = trainer.train_iteration()

    for key in (
        "return_mean",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clipfrac",
        "ratio_mean",
        "ratio_max",
        "logratio_mean",
        "logratio_abs_max",
        "grad_norm",
        "num_minibatches_per_epoch",
        "num_total_minibatches",
        "effective_replay_exposure",
        "policy_active_ratio",
        "advantage_mean",
        "advantage_std",
        "priority_mean",
        "priority_entropy",
        "sample_duplicate_frac",
    ):
        assert metrics[key] == pytest.approx(float(metrics[key]))
    assert any(
        not torch.allclose(param, old)
        for param, old in zip(model.parameters(), before, strict=True)
    )


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
            n_envs=3,
            segment_sampling=ppo.SegmentSamplingConfig(segments_per_minibatch=1),
        ),
        device=torch.device("cpu"),
    )

    trainer.train_iteration()
    segments = trainer.rollout.segment_major()

    assert not segments.obs.still_playing[:, :, 2:].any()
    assert segments.dones[:, :, 2:].all()


def test_segment_sampling_advantages_use_policy_mask() -> None:
    advantages = torch.tensor(
        [
            [[1.0, 100.0], [2.0, 200.0]],
            [[3.0, 300.0], [4.0, 400.0]],
        ]
    )
    policy_mask = torch.tensor(
        [
            [[True, False], [True, False]],
            [[False, False], [True, False]],
        ]
    )

    sampling_advantages = ppo._segment_sampling_advantages(advantages, policy_mask)

    assert torch.equal(
        sampling_advantages,
        torch.tensor([[1.0, 2.0], [0.0, 4.0]]),
    )


def test_uniform_replay_one_uses_shuffled_single_pass_minibatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    env = TinyOrbitEnv(n_envs=5)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            n_envs=5,
            update_epochs=1,
            replay_ratio=1.0,
            segment_sampling=ppo.SegmentSamplingConfig(segments_per_minibatch=2),
        ),
        device=torch.device("cpu"),
    )
    bootstrap_values = trainer._collect_rollout()
    segments = trainer.rollout.segment_major()
    policy_mask = ppo._policy_mask(segments.obs)
    value_mask = segments.obs.still_playing
    advantages = torch.ones_like(segments.rewards)
    returns = advantages + segments.values
    seen: list[torch.Tensor] = []

    def fake_update_minibatch(
        segments: ppo.PPORolloutSegments,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        policy_mask: torch.Tensor,
        value_mask: torch.Tensor,
        sample: ppo.SegmentSample,
    ) -> ppo.PPOUpdateResult:
        del advantages, returns, policy_mask, value_mask
        seen.append(sample.indices.detach().clone())
        zero = segments.logp.new_zeros(())
        return ppo.PPOUpdateResult(
            metrics=_zero_loss_metrics(zero),
            indices=sample.indices,
            new_logp=segments.logp[sample.indices],
            new_values=segments.values[sample.indices],
            grad_norm=zero,
        )

    monkeypatch.setattr(trainer, "_update_minibatch", fake_update_minibatch)

    metrics = trainer._update(
        segments,
        advantages,
        returns,
        bootstrap_values,
        policy_mask,
        value_mask,
    )

    assert [indices.shape for indices in seen] == [(2,), (2,), (1,)]
    all_indices = torch.cat(seen)
    assert torch.equal(all_indices.sort().values, torch.arange(5))
    assert metrics["num_minibatches"] == pytest.approx(3.0)
    assert metrics["num_minibatches_per_epoch"] == pytest.approx(3.0)
    assert metrics["num_total_minibatches"] == pytest.approx(3.0)
    assert metrics["effective_replay_exposure"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "config",
    [
        ppo.PPOConfig(
            n_envs=4,
            replay_ratio=2.0,
            segment_sampling=ppo.SegmentSamplingConfig(
                sampling="uniform",
                segments_per_minibatch=2,
            ),
        ),
        ppo.PPOConfig(
            n_envs=4,
            replay_ratio=1.0,
            segment_sampling=ppo.SegmentSamplingConfig(
                sampling="advantage_priority",
                segments_per_minibatch=2,
            ),
        ),
    ],
)
def test_epoch_samples_keeps_replacement_sampling_paths(config: ppo.PPOConfig) -> None:
    advantages = torch.ones((4, 2))

    n_minibatches = ppo._num_minibatches_per_epoch(config)
    samples = ppo._epoch_samples(
        advantages,
        config,
        n_minibatches,
    )

    assert samples == [None for _ in range(n_minibatches)]


def test_ppo_config_defaults_target_kl() -> None:
    assert ppo.PPOConfig().target_kl == pytest.approx(0.03)


def test_trainer_compiles_model_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_compile(target: Any, *, mode: str) -> Any:
        name = getattr(target, "__name__", target.__class__.__name__)
        calls.append((name, mode))
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
            n_envs=2,
            segment_sampling=ppo.SegmentSamplingConfig(segments_per_minibatch=1),
            compile_mode="default",
        ),
        device=torch.device("cpu"),
    )

    assert trainer.model is model
    assert calls == [
        ("TinyOrbitModel", "default"),
        ("evaluate_actions", "default"),
        ("_ppo_loss_tensors", "default"),
    ]


def test_trainer_recomputes_advantages_each_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(2)
    means_seen: list[float] = []
    compute_gae_calls = 0

    def fake_compute_gae(
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        ratios: torch.Tensor | None = None,
        mode: ppo.AdvantageMode = "gae",
        vtrace_rho_clip: float = 1.0,
        vtrace_c_clip: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del dones, last_values, gamma, gae_lambda, ratios, mode, vtrace_rho_clip
        del vtrace_c_clip
        nonlocal compute_gae_calls
        compute_gae_calls += 1
        advantages = torch.full_like(rewards, float(compute_gae_calls))
        return advantages, advantages + values

    def fake_ppo_loss(
        *,
        new_logp: torch.Tensor,
        entropy: torch.Tensor,
        new_values: torch.Tensor,
        old_logp: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        policy_weight: torch.Tensor,
        value_weight: torch.Tensor,
        config: ppo.PPOConfig,
    ) -> ppo.PPOLossMetrics:
        del entropy, old_logp, old_values, returns, value_weight, config
        means_seen.append(float((advantages * policy_weight).mean().item()))
        loss = new_values.mean() + 0.0 * new_logp.mean()
        zero = loss.detach().new_zeros(())
        return ppo.PPOLossMetrics(
            loss=loss,
            policy_loss=zero,
            value_loss=zero,
            entropy=zero,
            approx_kl=zero,
            clipfrac=zero,
            ratio_mean=zero,
            ratio_max=zero,
            logratio_mean=zero,
            logratio_abs_max=zero,
        )

    monkeypatch.setattr(ppo, "compute_gae", fake_compute_gae)
    monkeypatch.setattr(ppo, "ppo_loss", fake_ppo_loss)
    env = TinyOrbitEnv(n_envs=1)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            n_envs=1,
            update_epochs=2,
            segment_sampling=ppo.SegmentSamplingConfig(segments_per_minibatch=1),
            recompute_advantages_each_epoch=True,
        ),
        device=torch.device("cpu"),
    )

    trainer.train_iteration()

    assert compute_gae_calls == 2
    assert len(means_seen) == 2
    assert means_seen[0] != means_seen[1]


def test_trainer_vtrace_recomputes_current_policy_ratios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(4)
    ratios_seen: list[torch.Tensor] = []

    def fake_compute_gae(
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        ratios: torch.Tensor | None = None,
        mode: ppo.AdvantageMode = "gae",
        vtrace_rho_clip: float = 1.0,
        vtrace_c_clip: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del dones, last_values, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip
        assert mode == "gae_vtrace"
        assert ratios is not None
        ratios_seen.append(ratios.detach().clone())
        advantages = torch.ones_like(rewards)
        return advantages, advantages + values

    def fake_ppo_loss(
        *,
        new_logp: torch.Tensor,
        entropy: torch.Tensor,
        new_values: torch.Tensor,
        old_logp: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        policy_weight: torch.Tensor,
        value_weight: torch.Tensor,
        config: ppo.PPOConfig,
    ) -> ppo.PPOLossMetrics:
        del entropy, old_logp, old_values, returns, advantages, policy_weight
        del value_weight, config
        loss = -new_logp.mean() + 0.0 * new_values.mean()
        zero = loss.detach().new_zeros(())
        return ppo.PPOLossMetrics(
            loss=loss,
            policy_loss=zero,
            value_loss=zero,
            entropy=zero,
            approx_kl=zero,
            clipfrac=zero,
            ratio_mean=zero,
            ratio_max=zero,
            logratio_mean=zero,
            logratio_abs_max=zero,
        )

    monkeypatch.setattr(ppo, "compute_gae", fake_compute_gae)
    monkeypatch.setattr(ppo, "ppo_loss", fake_ppo_loss)
    env = TinyOrbitEnv(n_envs=1)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.1, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            n_envs=1,
            update_epochs=2,
            segment_sampling=ppo.SegmentSamplingConfig(segments_per_minibatch=1),
            advantage_mode="gae_vtrace",
        ),
        device=torch.device("cpu"),
    )

    trainer.train_iteration()

    assert len(ratios_seen) == 2
    assert torch.allclose(ratios_seen[0], torch.ones_like(ratios_seen[0]))
    assert not torch.allclose(ratios_seen[1], torch.ones_like(ratios_seen[1]))


def test_trainer_vtrace_refreshes_same_epoch_replay_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(5)
    seen: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def fake_compute_gae(
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        ratios: torch.Tensor | None = None,
        mode: ppo.AdvantageMode = "gae",
        vtrace_rho_clip: float = 1.0,
        vtrace_c_clip: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del dones, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip
        assert mode == "gae_vtrace"
        assert ratios is not None
        seen.append(
            (
                ratios.detach().clone(),
                values.detach().clone(),
                last_values.detach().clone(),
            )
        )
        advantages = torch.ones_like(rewards)
        return advantages, advantages + values

    def fake_sample_segments(
        advantages: torch.Tensor,
        config: ppo.SegmentSamplingConfig,
    ) -> ppo.SegmentSample:
        del config
        return ppo.SegmentSample(
            indices=torch.zeros((1,), dtype=torch.int64),
            importance=torch.ones((1, 1), dtype=advantages.dtype),
            probabilities=torch.ones((advantages.shape[0],), dtype=advantages.dtype),
        )

    monkeypatch.setattr(ppo, "compute_gae", fake_compute_gae)
    monkeypatch.setattr(ppo, "sample_segments", fake_sample_segments)

    env = TinyOrbitEnv(n_envs=1)
    model = TinyOrbitModel()
    trainer = ppo.PPOTrainer(
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5),
        config=ppo.PPOConfig(
            horizon=2,
            n_envs=1,
            update_epochs=1,
            replay_ratio=2.0,
            segment_sampling=ppo.SegmentSamplingConfig(segments_per_minibatch=1),
            advantage_mode="gae_vtrace",
        ),
        device=torch.device("cpu"),
    )
    bootstrap_values = trainer._collect_rollout()
    segments = trainer.rollout.segment_major()
    policy_mask = ppo._policy_mask(segments.obs)
    value_mask = segments.obs.still_playing
    advantages = torch.ones_like(segments.rewards)
    returns = advantages + segments.values
    replacement_logp = torch.full_like(segments.logp[:1], 0.5)
    replacement_values = torch.full_like(segments.values[:1], 2.0)
    replacement_bootstrap = torch.full_like(bootstrap_values, 3.0)

    def fake_update_minibatch(
        segments: ppo.PPORolloutSegments,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        policy_mask: torch.Tensor,
        value_mask: torch.Tensor,
        sample: ppo.SegmentSample,
    ) -> ppo.PPOUpdateResult:
        del segments, advantages, returns, policy_mask, value_mask
        zero = replacement_logp.new_zeros(())
        metrics = ppo.PPOLossMetrics(
            loss=zero,
            policy_loss=zero,
            value_loss=zero,
            entropy=zero,
            approx_kl=zero,
            clipfrac=zero,
            ratio_mean=zero,
            ratio_max=zero,
            logratio_mean=zero,
            logratio_abs_max=zero,
        )
        return ppo.PPOUpdateResult(
            metrics=metrics,
            indices=sample.indices,
            new_logp=replacement_logp,
            new_values=replacement_values,
            grad_norm=zero,
        )

    monkeypatch.setattr(trainer, "_update_minibatch", fake_update_minibatch)
    monkeypatch.setattr(
        trainer,
        "_current_bootstrap_values",
        lambda: replacement_bootstrap,
    )

    trainer._update(
        segments,
        advantages,
        returns,
        bootstrap_values,
        policy_mask,
        value_mask,
    )

    assert len(seen) == 1
    assert torch.allclose(seen[0][0], torch.exp(replacement_logp - segments.logp[:1]))
    assert torch.allclose(seen[0][1], replacement_values)
    assert torch.allclose(seen[0][2], replacement_bootstrap)


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
            n_envs=3,
            segment_sampling=ppo.SegmentSamplingConfig(segments_per_minibatch=1),
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
