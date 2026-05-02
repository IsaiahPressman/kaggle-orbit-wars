from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

import torch
from pydantic import Field

from owl.config import BaseConfig
from owl.model import BaseModelAPI, ModelActions, ModelEvaluation, ModelOutput
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    OUTER_PLAYER_SLOTS,
    ActionConfig,
    ActionPureConfig,
    ObsBatch,
    ObsV1Config,
    VectorizedEnv,
)
from owl.train.advantages import (
    AdvantageMode,
    ComputeGAEFn,
    compile_compute_gae,
    compute_gae,
)
from owl.train.metrics import (
    explained_variance,
    masked_mean,
    masked_std,
    weighted_mean,
)
from owl.train.optimizer import CompositeOptimizer, LRScheduler, Optimizer
from owl.train.sampling import (
    SampleSegmentsFn,
    SegmentSample,
    SegmentSamplingConfig,
    SegmentSamplingMetrics,
    compile_sample_segments,
    sample_segments,
    sample_segments_uniform_single_pass,
    segment_sampling_metrics,
)
from owl.train.utils import (
    TrainingDType,
    assert_finite,
    autocast_context,
    require_same_shape,
)

CompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]


_OBS_FIELDS = tuple(ObsBatch.model_fields)
_ACTION_FIELDS = tuple(ModelActions.__dataclass_fields__)


class PPOConfig(BaseConfig):
    horizon: int = Field(default=64, ge=1)
    checkpoint_freq: int | None = Field(default=None, ge=1_000)
    replay_ratio: float = Field(default=1.0, gt=0.0)
    segment_sampling: SegmentSamplingConfig = Field(
        default_factory=SegmentSamplingConfig
    )
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_coef: float = Field(default=0.2, ge=0.0)
    vf_clip_coef: float = Field(default=0.2, ge=0.0)
    vf_coef: float = Field(default=0.5, ge=0.0)
    ent_coef: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    target_kl: float | None = Field(default=0.03, gt=0.0)
    advantage_mode: AdvantageMode = "gae"
    vtrace_rho_clip: float = Field(default=1.0, gt=0.0)
    vtrace_c_clip: float = Field(default=1.0, gt=0.0)
    recompute_advantages_each_minibatch: bool = True
    normalize_advantages: bool = False
    debug_validate_ppo_loss_inputs: bool = False
    compile_mode: CompileMode | None = None
    dtype: TrainingDType = "float32"


class PPOLossFn(Protocol):
    def __call__(
        self,
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
        config: PPOConfig,
    ) -> PPOLossMetrics: ...


@dataclass(frozen=True)
class PPOLossMetrics:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clipfrac: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_max: torch.Tensor
    logratio_mean: torch.Tensor
    logratio_abs_max: torch.Tensor


@dataclass(frozen=True)
class PPOUpdateResult:
    metrics: PPOLossMetrics
    indices: torch.Tensor
    new_logp: torch.Tensor
    new_values: torch.Tensor
    grad_norm: torch.Tensor


@dataclass(frozen=True)
class PPORolloutSegments:
    """Rollout tensors converted from collection layout [T, N, ...] to [N, T, ...]."""

    obs: ObsBatch
    actions: ModelActions
    logp: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class PPORolloutBuffer:
    """Collect rollouts in time-major layout [T, N, ...]."""

    def __init__(
        self,
        *,
        horizon: int,
        n_envs: int,
        obs_spec: ObsV1Config,
        action_spec: ActionConfig,
        device: torch.device,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_envs <= 0:
            raise ValueError("n_envs must be positive")
        self.horizon = horizon
        self.n_envs = n_envs
        can_act_shape = (
            (horizon, n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS)
            if isinstance(action_spec, ActionPureConfig)
            else (
                horizon,
                n_envs,
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                ACTION_ENTITY_SLOTS,
            )
        )
        self.obs = ObsBatch(
            planets=torch.zeros(
                (horizon, n_envs, obs_spec.max_planets, obs_spec.planet_channels),
                dtype=torch.float32,
                device=device,
            ),
            fleets=torch.zeros(
                (horizon, n_envs, obs_spec.max_fleets, obs_spec.fleet_channels),
                dtype=torch.float32,
                device=device,
            ),
            comets=torch.zeros(
                (horizon, n_envs, obs_spec.max_comets, obs_spec.comet_channels),
                dtype=torch.float32,
                device=device,
            ),
            entity_mask=torch.zeros(
                (horizon, n_envs, obs_spec.max_entities),
                dtype=torch.bool,
                device=device,
            ),
            still_playing=torch.zeros(
                (horizon, n_envs, OUTER_PLAYER_SLOTS),
                dtype=torch.bool,
                device=device,
            ),
            global_features=torch.zeros(
                (horizon, n_envs, obs_spec.global_channels),
                dtype=torch.float32,
                device=device,
            ),
            can_act=torch.zeros(
                can_act_shape,
                dtype=torch.bool,
                device=device,
            ),
            max_launch=torch.zeros(
                (horizon, n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS),
                dtype=torch.int64,
                device=device,
            ),
        )
        action_shape = (
            horizon,
            n_envs,
            OUTER_PLAYER_SLOTS,
            ACTION_ENTITY_SLOTS,
            action_spec.max_per_planet_launches,
        )
        self.actions = ModelActions(
            launch=torch.zeros(action_shape, dtype=torch.bool, device=device),
            ships=torch.zeros(action_shape, dtype=torch.int64, device=device),
            angle=(
                torch.zeros(action_shape, dtype=torch.float32, device=device)
                if isinstance(action_spec, ActionPureConfig)
                else None
            ),
            target=(
                None
                if isinstance(action_spec, ActionPureConfig)
                else torch.zeros(action_shape, dtype=torch.int64, device=device)
            ),
        )
        self.logp = torch.zeros(
            (horizon, n_envs, OUTER_PLAYER_SLOTS),
            dtype=torch.float32,
            device=device,
        )
        self.values = torch.zeros(
            (horizon, n_envs, OUTER_PLAYER_SLOTS),
            dtype=torch.float32,
            device=device,
        )
        self.rewards = torch.zeros(
            (horizon, n_envs, OUTER_PLAYER_SLOTS),
            dtype=torch.float32,
            device=device,
        )
        self.dones = torch.zeros(
            (horizon, n_envs, OUTER_PLAYER_SLOTS),
            dtype=torch.bool,
            device=device,
        )

    def write_step(
        self,
        step: int,
        *,
        obs: ObsBatch,
        actions: ModelActions,
        logp: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        if not 0 <= step < self.horizon:
            raise ValueError(f"step must be in 0..{self.horizon - 1}, got {step}")
        _copy_obs_time_step(self.obs, step, obs)
        _copy_actions_time_step(self.actions, step, actions)
        self.logp[step].copy_(logp)
        self.values[step].copy_(values)
        self.rewards[step].copy_(rewards)
        self.dones[step].copy_(dones)

    def segment_major(self) -> PPORolloutSegments:
        """Return contiguous segment-major/time-second rollout tensors [N, T, ...]."""
        return PPORolloutSegments(
            obs=_obs_segment_major(self.obs),
            actions=_actions_segment_major(self.actions),
            logp=self.logp.transpose(0, 1).contiguous(),
            values=self.values.transpose(0, 1).contiguous(),
            rewards=self.rewards.transpose(0, 1).contiguous(),
            dones=self.dones.transpose(0, 1).contiguous(),
        )


class PPOTrainer:
    def __init__(
        self,
        *,
        config: PPOConfig,
        env: VectorizedEnv,
        model: BaseModelAPI,
        optimizer: Optimizer,
        device: torch.device,
        lr_scheduler: LRScheduler | None = None,
    ) -> None:
        self.env = env
        self.model = model
        model_action_spec = getattr(model, "action_spec", None)
        if model_action_spec != env.action_spec:
            raise ValueError("model and env action_spec must match")
        self._compute_gae = _compile_compute_gae(config.compile_mode)
        self._sample_segments = _compile_sample_segments(config.compile_mode)
        self._ppo_loss = _compile_ppo_loss(config.compile_mode)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device
        self.n_envs = env.n_envs
        self.optimizer_steps = 0
        self._non_blocking_env_to_device = device.type == "cuda" and getattr(
            env, "pin_memory_enabled", False
        )
        self._obs = _obs_to_device(
            env.reset(),
            device,
            non_blocking=self._non_blocking_env_to_device,
        )
        self.rollout = PPORolloutBuffer(
            horizon=config.horizon,
            n_envs=env.n_envs,
            obs_spec=env.obs_spec,
            action_spec=env.action_spec,
            device=device,
        )
        self._last_env_metrics: dict[str, list[float]] = {}

    def train_iteration(self) -> dict[str, float]:
        start = perf_counter()
        rollout_start = perf_counter()
        last_values = self._collect_rollout()
        rollout_elapsed = max(perf_counter() - rollout_start, 1e-12)
        env_metrics = self._last_env_metrics
        segments = self.rollout.segment_major()
        value_mask = segments.obs.still_playing
        policy_mask = _policy_mask(segments.obs)
        ratios = (
            torch.ones_like(segments.logp)
            if self.config.advantage_mode == "puffer_vtrace"
            else None
        )
        advantages, returns = self._compute_gae(
            rewards=segments.rewards,
            values=segments.values,
            dones=segments.dones,
            last_values=last_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            ratios=ratios,
            mode=self.config.advantage_mode,
            vtrace_rho_clip=self.config.vtrace_rho_clip,
            vtrace_c_clip=self.config.vtrace_c_clip,
        )
        update_start = perf_counter()
        metrics = self._update(
            segments,
            advantages,
            returns,
            last_values,
            policy_mask,
            value_mask,
        )
        update_elapsed = max(perf_counter() - update_start, 1e-12)
        player_returns, player_return_mask = _player_segment_returns(
            segments.rewards,
            value_mask,
        )
        metrics["train/return_mean"] = float(
            masked_mean(player_returns, player_return_mask).item()
        )
        metrics["train/return_max"] = float(
            _masked_reward_max(segments.rewards, value_mask).item()
        )
        metrics["train/explained_variance"] = float(
            explained_variance(segments.values, returns, valid_mask=value_mask).item()
        )
        metrics["train/advantage_mean"] = float(
            masked_mean(advantages, policy_mask).item()
        )
        metrics["train/advantage_std"] = float(
            masked_std(advantages, policy_mask).item()
        )
        metrics.update(_mean_env_metrics(env_metrics))
        elapsed = max(perf_counter() - start, 1e-12)
        rollout_steps = self.config.horizon * self.n_envs
        metrics["time/rollout_seconds"] = float(rollout_elapsed)
        metrics["time/update_seconds"] = float(update_elapsed)
        metrics["time/iteration_seconds"] = float(elapsed)
        metrics["perf/rollout_sps"] = float(rollout_steps / rollout_elapsed)
        metrics["perf/update_sps"] = float(rollout_steps / update_elapsed)
        metrics["perf/steps_per_second"] = float(rollout_steps / elapsed)
        return metrics

    def write_checkpoint(
        self,
        path: Path,
        *,
        config: Any,
        config_path: Path,
        env_steps: int,
    ) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": (
                None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
            ),
            "config": config.model_dump(mode="json", round_trip=True),
            "config_path": str(config_path),
            "env_steps": env_steps,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)

    def _collect_rollout(self) -> torch.Tensor:
        self.rollout.rewards.zero_()
        self.rollout.dones.zero_()
        env_metrics: dict[str, list[float]] = {}
        with torch.no_grad():
            for step in range(self.config.horizon):
                with autocast_context(self.config, self.device):
                    output = self.model(self._obs)
                actions = _output_actions(output)
                next_obs, rewards, dones, step_env_metrics = _step_env(
                    self.env, actions
                )
                _extend_env_metrics(env_metrics, step_env_metrics)
                rewards = rewards.to(
                    self.device,
                    non_blocking=self._non_blocking_env_to_device,
                )
                dones = dones.to(
                    self.device,
                    non_blocking=self._non_blocking_env_to_device,
                )
                self.rollout.write_step(
                    step,
                    obs=self._obs,
                    actions=actions,
                    logp=_output_logp(output),
                    values=_output_values(output),
                    rewards=rewards,
                    dones=dones,
                )
                _copy_obs_to_device_(
                    self._obs,
                    next_obs,
                    non_blocking=self._non_blocking_env_to_device,
                )
            with autocast_context(self.config, self.device):
                bootstrap = self.model(self._obs)
            self._last_env_metrics = env_metrics
            return _output_values(bootstrap).detach()

    def _update(
        self,
        segments: PPORolloutSegments,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        bootstrap_values: torch.Tensor,
        policy_mask: torch.Tensor,
        value_mask: torch.Tensor,
    ) -> dict[str, float]:
        loss_metrics: list[PPOLossMetrics] = []
        grad_norms: list[torch.Tensor] = []
        sampling_metrics: list[SegmentSamplingMetrics] = []
        current_values = segments.values.clone()
        current_ratios = torch.ones_like(segments.logp)
        current_advantages = advantages
        current_returns = returns
        current_bootstrap_values = bootstrap_values.clone()
        n_minibatches = _num_minibatches_per_update(self.config, self.n_envs)
        sampled_segments = 0
        update_samples = _update_samples(
            sampling_advantages=_segment_sampling_advantages(
                current_advantages,
                policy_mask,
            ),
            config=self.config,
            n_minibatches=n_minibatches,
        )
        for minibatch_index, update_sample in enumerate(update_samples):
            if self.config.advantage_mode == "puffer_vtrace":
                current_advantages, current_returns = self._compute_current_gae(
                    segments=segments,
                    current_values=current_values,
                    current_bootstrap_values=current_bootstrap_values,
                    current_ratios=current_ratios,
                )
            elif self._should_recompute_advantages(minibatch_index):
                _current_logp, current_values = self._current_segment_logp_values(
                    segments
                )
                current_bootstrap_values = self._current_bootstrap_values()
                current_advantages, current_returns = self._compute_current_gae(
                    segments=segments,
                    current_values=current_values,
                    current_bootstrap_values=current_bootstrap_values,
                    current_ratios=None,
                )
            sampling_advantages = _segment_sampling_advantages(
                current_advantages,
                policy_mask,
            )
            sample = (
                update_sample
                if update_sample is not None
                else self._sample_segments(
                    sampling_advantages,
                    self.config.segment_sampling,
                )
            )
            sampling_metrics.append(
                segment_sampling_metrics(sampling_advantages, sample)
            )
            sampled_segments += int(sample.indices.numel())
            update = self._update_minibatch(
                segments,
                current_advantages,
                current_returns,
                policy_mask,
                value_mask,
                sample,
                value_clip_anchor=current_values,
            )
            loss_metrics.append(update.metrics)
            grad_norms.append(update.grad_norm.detach())
            current_values[update.indices] = update.new_values
            if self.config.advantage_mode == "puffer_vtrace":
                current_ratios[update.indices] = torch.exp(
                    update.new_logp - segments.logp[update.indices]
                )
            if (
                self.config.target_kl is not None
                and update.metrics.approx_kl.item() > self.config.target_kl
            ):
                break

        if not loss_metrics:
            raise RuntimeError("internal error: PPO update produced no minibatches")
        metrics = _mean_loss_metrics(loss_metrics)
        metrics["optimizer/grad_norm"] = float(torch.stack(grad_norms).mean().item())
        metrics["optimizer/steps"] = float(self.optimizer_steps)
        metrics["optimizer/minibatches_per_update"] = float(n_minibatches)
        metrics["sampling/effective_replay_exposure"] = float(
            sampled_segments / self.n_envs
        )
        metrics["train/policy_active_ratio"] = float(policy_mask.float().mean().item())
        metrics["optimizer/learning_rate"] = _current_learning_rate(
            self.optimizer,
            self.lr_scheduler,
        )
        if sampling_metrics:
            metrics.update(_mean_sampling_metrics(sampling_metrics))
        return metrics

    def _should_recompute_advantages(self, minibatch_index: int) -> bool:
        if minibatch_index == 0:
            return False
        return (
            self.config.recompute_advantages_each_minibatch
            or self.config.advantage_mode == "puffer_vtrace"
        )

    def _compute_current_gae(
        self,
        *,
        segments: PPORolloutSegments,
        current_values: torch.Tensor,
        current_bootstrap_values: torch.Tensor,
        current_ratios: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._compute_gae(
            rewards=segments.rewards,
            values=current_values,
            dones=segments.dones,
            last_values=current_bootstrap_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            ratios=current_ratios,
            mode=self.config.advantage_mode,
            vtrace_rho_clip=self.config.vtrace_rho_clip,
            vtrace_c_clip=self.config.vtrace_c_clip,
        )

    def _current_bootstrap_values(self) -> torch.Tensor:
        with torch.no_grad(), autocast_context(self.config, self.device):
            values = self.model.compute_value(self._obs)
        return values.detach()

    def _current_segment_logp_values(
        self,
        segments: PPORolloutSegments,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), autocast_context(self.config, self.device):
            output = self.model.evaluate_actions(
                _flatten_obs_time(segments.obs),
                _flatten_actions_time(segments.actions),
            )
        return (
            _output_logp(output).detach().view_as(segments.logp),
            _output_values(output).detach().view_as(segments.values).clone(),
        )

    def _update_minibatch(
        self,
        segments: PPORolloutSegments,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        policy_mask: torch.Tensor,
        value_mask: torch.Tensor,
        sample: SegmentSample,
        *,
        value_clip_anchor: torch.Tensor,
    ) -> PPOUpdateResult:
        idx = sample.indices
        batch_actions = _flatten_actions_time(_actions_index(segments.actions, idx))
        batch_obs = _flatten_obs_time(_obs_index(segments.obs, idx))
        batch_old_logp = segments.logp[idx]
        batch_old_values = value_clip_anchor[idx]
        batch_returns = returns[idx]
        batch_policy_mask = policy_mask[idx]
        batch_value_mask = value_mask[idx]
        importance = sample.importance
        while importance.ndim < advantages[idx].ndim:
            importance = importance.unsqueeze(-1)
        batch_advantages = advantages[idx]
        if self.config.normalize_advantages:
            batch_advantages = normalize_masked_advantages(
                batch_advantages,
                batch_policy_mask,
            )
        batch_advantages = batch_advantages * importance
        batch_policy_weight = batch_policy_mask.to(dtype=batch_advantages.dtype)
        batch_value_weight = batch_value_mask.to(dtype=batch_advantages.dtype)

        with autocast_context(self.config, self.device):
            output = self.model.evaluate_actions(batch_obs, batch_actions)
        new_logp = _output_logp(output).view_as(batch_old_logp)
        entropy = _output_entropy(output, batch_old_logp)
        new_values = _output_values(output).view_as(batch_old_values)

        metrics = self._ppo_loss(
            new_logp=new_logp,
            entropy=entropy,
            new_values=new_values,
            old_logp=batch_old_logp,
            old_values=batch_old_values,
            returns=batch_returns,
            advantages=batch_advantages,
            policy_weight=batch_policy_weight,
            value_weight=batch_value_weight,
            config=self.config,
        )
        self.optimizer.zero_grad(set_to_none=True)
        metrics.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer_steps += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return PPOUpdateResult(
            metrics=metrics,
            indices=idx,
            new_logp=new_logp.detach(),
            new_values=new_values.detach(),
            grad_norm=grad_norm.detach(),
        )


def _compile_compute_gae(compile_mode: CompileMode | None) -> ComputeGAEFn:
    if compile_mode is None:
        return compute_gae
    return compile_compute_gae(compile_mode)


def _compile_sample_segments(
    compile_mode: CompileMode | None,
) -> SampleSegmentsFn:
    if compile_mode is None:
        return sample_segments
    return compile_sample_segments(compile_mode)


def _compile_ppo_loss(compile_mode: CompileMode | None) -> PPOLossFn:
    if compile_mode is None:
        return ppo_loss
    compiled_tensor_loss = torch.compile(_ppo_loss_tensors, mode=compile_mode)

    def compiled_ppo_loss(
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
        config: PPOConfig,
    ) -> PPOLossMetrics:
        _validate_ppo_loss_inputs_if_debug(
            new_logp=new_logp,
            entropy=entropy,
            new_values=new_values,
            old_logp=old_logp,
            old_values=old_values,
            returns=returns,
            advantages=advantages,
            policy_weight=policy_weight,
            value_weight=value_weight,
            config=config,
        )
        return _ppo_loss_metrics_from_tuple(
            compiled_tensor_loss(
                new_logp,
                entropy,
                new_values,
                old_logp,
                old_values,
                returns,
                advantages,
                policy_weight,
                value_weight,
                config.clip_coef,
                config.vf_clip_coef,
                config.vf_coef,
                config.ent_coef,
            )
        )

    return compiled_ppo_loss


def ppo_loss(
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
    config: PPOConfig,
) -> PPOLossMetrics:
    _validate_ppo_loss_inputs_if_debug(
        new_logp=new_logp,
        entropy=entropy,
        new_values=new_values,
        old_logp=old_logp,
        old_values=old_values,
        returns=returns,
        advantages=advantages,
        policy_weight=policy_weight,
        value_weight=value_weight,
        config=config,
    )
    return _ppo_loss_metrics_from_tuple(
        _ppo_loss_tensors(
            new_logp,
            entropy,
            new_values,
            old_logp,
            old_values,
            returns,
            advantages,
            policy_weight,
            value_weight,
            config.clip_coef,
            config.vf_clip_coef,
            config.vf_coef,
            config.ent_coef,
        )
    )


def _validate_ppo_loss_inputs_if_debug(
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
    config: PPOConfig,
) -> None:
    if not config.debug_validate_ppo_loss_inputs:
        return
    validate_ppo_loss_inputs(
        new_logp=new_logp,
        entropy=entropy,
        new_values=new_values,
        old_logp=old_logp,
        old_values=old_values,
        returns=returns,
        advantages=advantages,
        policy_weight=policy_weight,
        value_weight=value_weight,
    )


def _ppo_loss_metrics_from_tuple(
    tensors: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> PPOLossMetrics:
    (
        loss,
        policy_loss,
        value_loss,
        entropy_loss,
        entropy,
        approx_kl,
        clipfrac,
        ratio_mean,
        ratio_max,
        logratio_mean,
        logratio_abs_max,
    ) = tensors
    return PPOLossMetrics(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        entropy=entropy,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
        ratio_mean=ratio_mean,
        ratio_max=ratio_max,
        logratio_mean=logratio_mean,
        logratio_abs_max=logratio_abs_max,
    )


def _ppo_loss_tensors(
    new_logp: torch.Tensor,
    entropy: torch.Tensor,
    new_values: torch.Tensor,
    old_logp: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    policy_weight: torch.Tensor,
    value_weight: torch.Tensor,
    clip_coef: float,
    vf_clip_coef: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    logratio = new_logp - old_logp
    ratio = logratio.exp()
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss = weighted_mean(torch.max(pg_loss1, pg_loss2), policy_weight)

    value_clipped = old_values + torch.clamp(
        new_values - old_values,
        -vf_clip_coef,
        vf_clip_coef,
    )
    value_loss_unclipped = (new_values - returns).pow(2)
    value_loss_clipped = (value_clipped - returns).pow(2)
    value_loss = 0.5 * weighted_mean(
        torch.max(value_loss_unclipped, value_loss_clipped),
        value_weight,
    )

    entropy_mean = weighted_mean(entropy, policy_weight)
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
    entropy_loss = -ent_coef * entropy_mean
    approx_kl = weighted_mean((ratio - 1.0) - logratio, policy_weight)
    clipfrac = weighted_mean(
        ((ratio - 1.0).abs() > clip_coef).float(),
        policy_weight,
    )
    ratio_mean = weighted_mean(ratio, policy_weight)
    ratio_max = _masked_max_or_zero(ratio, policy_weight > 0)
    logratio_mean = weighted_mean(logratio, policy_weight)
    logratio_abs_max = _masked_max_or_zero(logratio.abs(), policy_weight > 0)

    return (
        loss,
        policy_loss,
        value_loss,
        entropy_loss,
        entropy_mean,
        approx_kl,
        clipfrac,
        ratio_mean,
        ratio_max,
        logratio_mean,
        logratio_abs_max,
    )


def validate_ppo_loss_inputs(
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
) -> None:
    require_same_shape(new_logp, old_logp, left_name="new_logp", right_name="old_logp")
    require_same_shape(new_logp, entropy, left_name="new_logp", right_name="entropy")
    require_same_shape(
        new_logp, old_values, left_name="new_logp", right_name="old_values"
    )
    require_same_shape(
        new_logp, new_values, left_name="new_logp", right_name="new_values"
    )
    require_same_shape(new_logp, returns, left_name="new_logp", right_name="returns")
    require_same_shape(
        new_logp, advantages, left_name="new_logp", right_name="advantages"
    )
    require_same_shape(
        new_logp, policy_weight, left_name="new_logp", right_name="policy_weight"
    )
    require_same_shape(
        new_logp, value_weight, left_name="new_logp", right_name="value_weight"
    )
    for name, tensor in (
        ("new_logp", new_logp),
        ("entropy", entropy),
        ("new_values", new_values),
        ("old_logp", old_logp),
        ("old_values", old_values),
        ("returns", returns),
        ("advantages", advantages),
        ("policy_weight", policy_weight),
        ("value_weight", value_weight),
    ):
        assert_finite(tensor, name)


def _copy_obs_time_step(dst: ObsBatch, step: int, src: ObsBatch) -> None:
    for field in _OBS_FIELDS:
        getattr(dst, field)[step].copy_(getattr(src, field))


def _copy_actions_time_step(
    dst: ModelActions,
    step: int,
    src: ModelActions,
) -> None:
    for field in _ACTION_FIELDS:
        dst_tensor = getattr(dst, field)
        src_tensor = getattr(src, field)
        if dst_tensor is None:
            continue
        if src_tensor is None:
            raise ValueError(f"actions.{field} is required")
        dst_tensor[step].copy_(src_tensor)


def _obs_segment_major(obs: ObsBatch) -> ObsBatch:
    return ObsBatch(
        **{
            field: getattr(obs, field).transpose(0, 1).contiguous()
            for field in _OBS_FIELDS
        }
    )


def _actions_segment_major(actions: ModelActions) -> ModelActions:
    return ModelActions(
        launch=actions.launch.transpose(0, 1).contiguous(),
        ships=actions.ships.transpose(0, 1).contiguous(),
        angle=_optional_actions_segment_major(actions.angle),
        target=_optional_actions_segment_major(actions.target),
    )


def _obs_index(obs: ObsBatch, idx: torch.Tensor) -> ObsBatch:
    return ObsBatch(**{field: getattr(obs, field)[idx] for field in _OBS_FIELDS})


def _actions_index(actions: ModelActions, idx: torch.Tensor) -> ModelActions:
    return ModelActions(
        launch=actions.launch[idx],
        ships=actions.ships[idx],
        angle=_optional_actions_index(actions.angle, idx),
        target=_optional_actions_index(actions.target, idx),
    )


def _obs_to_device(
    obs: ObsBatch,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> ObsBatch:
    if device.type == "cpu":
        return ObsBatch(
            **{
                field: (
                    getattr(obs, field).to(device, non_blocking=non_blocking).clone()
                )
                for field in _OBS_FIELDS
            }
        )

    return ObsBatch(
        **{
            field: getattr(obs, field).to(device, non_blocking=non_blocking)
            for field in _OBS_FIELDS
        }
    )


def _copy_obs_to_device_(
    dst: ObsBatch,
    src: ObsBatch,
    *,
    non_blocking: bool = False,
) -> None:
    for field in _OBS_FIELDS:
        getattr(dst, field).copy_(getattr(src, field), non_blocking=non_blocking)


def _actions_to_cpu(
    actions: ModelActions,
    *,
    non_blocking: bool = False,
) -> ModelActions:
    return ModelActions(
        launch=actions.launch.to("cpu", non_blocking=non_blocking),
        ships=actions.ships.to("cpu", non_blocking=non_blocking),
        angle=_optional_actions_to_cpu(actions.angle, non_blocking=non_blocking),
        target=_optional_actions_to_cpu(actions.target, non_blocking=non_blocking),
    )


def _flatten_tensor_time(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _flatten_obs_time(obs: ObsBatch) -> ObsBatch:
    return ObsBatch(
        **{field: _flatten_tensor_time(getattr(obs, field)) for field in _OBS_FIELDS}
    )


def _flatten_actions_time(actions: ModelActions) -> ModelActions:
    return ModelActions(
        launch=_flatten_tensor_time(actions.launch),
        ships=_flatten_tensor_time(actions.ships),
        angle=_optional_flatten_tensor_time(actions.angle),
        target=_optional_flatten_tensor_time(actions.target),
    )


def _step_env(
    env: Any,
    actions: ModelActions,
) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
    cpu_actions = _actions_to_cpu(actions)
    result = env.step(cpu_actions.launch, cpu_actions.action_value(), cpu_actions.ships)
    if len(result) == 3:
        obs, rewards, dones = result
        return obs, rewards, dones, {}
    return result


def _extend_env_metrics(
    totals: dict[str, list[float]], step_metrics: dict[str, list[float]]
) -> None:
    for key, values in step_metrics.items():
        totals.setdefault(key, []).extend(values)


def _mean_env_metrics(metrics: dict[str, list[float]]) -> dict[str, float]:
    logged: dict[str, float] = {}
    terminal_episodes = 0.0
    for key, values in metrics.items():
        if not values:
            continue
        if key.startswith("terminal_episodes_"):
            value = float(sum(values))
            terminal_episodes += value
        else:
            value = float(sum(values) / len(values))
        logged[f"train/{key}"] = value
    if terminal_episodes > 0:
        logged["train/terminal_episodes"] = terminal_episodes
    return logged


def _player_segment_returns(
    rewards: torch.Tensor,
    value_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_rewards = rewards * value_mask.to(dtype=rewards.dtype)
    return masked_rewards.sum(dim=1), value_mask.any(dim=1)


def _masked_reward_max(rewards: torch.Tensor, value_mask: torch.Tensor) -> torch.Tensor:
    masked_rewards = rewards * value_mask.to(dtype=rewards.dtype)
    return masked_rewards.max()


def _current_learning_rate(
    optimizer: Optimizer,
    lr_scheduler: LRScheduler | None,
) -> float:
    if lr_scheduler is not None:
        return float(lr_scheduler.get_last_lr()[0])
    if isinstance(optimizer, CompositeOptimizer):
        return _torch_optimizer_learning_rate(optimizer.optimizers[0])
    if isinstance(optimizer, torch.optim.Optimizer):
        return _torch_optimizer_learning_rate(optimizer)
    raise TypeError("optimizer must be a torch optimizer or CompositeOptimizer")


def _torch_optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _output_actions(output: ModelOutput) -> ModelActions:
    return output.actions


def _output_logp(output: ModelOutput | ModelEvaluation) -> torch.Tensor:
    return output.log_probs.per_player_entity.sum(dim=-1)


def _output_entropy(
    output: ModelOutput | ModelEvaluation, like: torch.Tensor
) -> torch.Tensor:
    return output.entropies.per_player_entity.sum(dim=-1).view_as(like)


def _output_values(output: ModelOutput | ModelEvaluation) -> torch.Tensor:
    return output.values


def _policy_mask(obs: ObsBatch) -> torch.Tensor:
    can_act = obs.can_act.flatten(start_dim=3).any(dim=-1)
    return obs.still_playing & can_act


def _optional_actions_segment_major(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.transpose(0, 1).contiguous()


def _optional_actions_index(
    tensor: torch.Tensor | None,
    idx: torch.Tensor,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[idx]


def _optional_actions_to_cpu(
    tensor: torch.Tensor | None,
    *,
    non_blocking: bool,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to("cpu", non_blocking=non_blocking)


def _optional_flatten_tensor_time(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return _flatten_tensor_time(tensor)


def _uses_uniform_single_pass_sampling(config: PPOConfig) -> bool:
    return config.segment_sampling.sampling == "uniform" and config.replay_ratio == 1.0


def _num_minibatches_per_update(config: PPOConfig, n_envs: int) -> int:
    segments_per_minibatch = config.segment_sampling.segments_per_minibatch
    if _uses_uniform_single_pass_sampling(config):
        return (n_envs + segments_per_minibatch - 1) // segments_per_minibatch
    return max(1, int(config.replay_ratio * n_envs / segments_per_minibatch))


def _update_samples(
    *,
    sampling_advantages: torch.Tensor,
    config: PPOConfig,
    n_minibatches: int,
) -> list[SegmentSample | None]:
    if _uses_uniform_single_pass_sampling(config):
        samples: list[SegmentSample | None] = []
        samples.extend(
            sample_segments_uniform_single_pass(
                n_segments=sampling_advantages.shape[0],
                segments_per_minibatch=config.segment_sampling.segments_per_minibatch,
                device=sampling_advantages.device,
            )
        )
        return samples
    return [None for _ in range(n_minibatches)]


def _masked_max_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(mask, values, torch.full_like(values, -torch.inf))
    return torch.where(
        mask.any(),
        masked.max(),
        torch.zeros((), dtype=values.dtype, device=values.device),
    )


def normalize_masked_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    require_same_shape(advantages, mask, left_name="advantages", right_name="mask")
    mask_float = mask.to(dtype=advantages.dtype)
    denom = mask_float.sum().clamp_min(1.0)

    mean = (advantages * mask_float).sum() / denom
    var = ((advantages - mean).pow(2) * mask_float).sum() / denom

    return (advantages - mean) / (var.sqrt() + eps)


def _segment_sampling_advantages(
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    masked_advantages = advantages * valid_mask.to(dtype=advantages.dtype)
    if masked_advantages.ndim == 2:
        return masked_advantages
    if masked_advantages.ndim == 3:
        return masked_advantages.detach().abs().sum(dim=-1)
    raise ValueError(
        f"advantages must have shape [N, T] or [N, T, P], got {advantages.shape}"
    )


def _mean_loss_metrics(metrics: list[PPOLossMetrics]) -> dict[str, float]:
    metric_names = (
        ("loss/total_loss", "loss"),
        ("loss/policy_loss", "policy_loss"),
        ("loss/value_loss", "value_loss"),
        ("loss/entropy_loss", "entropy_loss"),
        ("policy/entropy", "entropy"),
        ("policy/approx_kl", "approx_kl"),
        ("policy/clipfrac", "clipfrac"),
        ("policy/ratio_mean", "ratio_mean"),
        ("policy/ratio_max", "ratio_max"),
        ("policy/logratio_mean", "logratio_mean"),
        ("policy/logratio_abs_max", "logratio_abs_max"),
    )
    return {
        output_name: float(
            torch.stack([getattr(metric, attr_name) for metric in metrics])
            .mean()
            .item()
        )
        for output_name, attr_name in metric_names
    }


def _mean_sampling_metrics(metrics: list[SegmentSamplingMetrics]) -> dict[str, float]:
    metric_names = (
        ("sampling/priority_min", "priority_min"),
        ("sampling/priority_mean", "priority_mean"),
        ("sampling/priority_max", "priority_max"),
        ("sampling/priority_entropy", "probability_entropy"),
        ("sampling/sample_duplicate_frac", "duplicate_fraction"),
        ("sampling/importance_mean", "importance_mean"),
        ("sampling/importance_max", "importance_max"),
    )
    return {
        output_name: float(
            torch.stack([getattr(metric, attr_name) for metric in metrics])
            .mean()
            .item()
        )
        for output_name, attr_name in metric_names
    }
