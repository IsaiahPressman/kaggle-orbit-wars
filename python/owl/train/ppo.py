from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol

import torch
from pydantic import Field

from owl.config import BaseConfig
from owl.model import BaseModelAPI, ModelActions, ModelEvaluation, ModelOutput
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    OUTER_PLAYER_SLOTS,
    ActionPureConfig,
    ObsBatch,
    ObsV1Config,
    VectorizedEnv,
)
from owl.train.advantages import AdvantageMode, compute_gae
from owl.train.metrics import (
    explained_variance,
    masked_mean,
    masked_std,
    masked_sum_by_segment,
    weighted_mean,
)
from owl.train.optimizer import LRScheduler, Optimizer
from owl.train.sampling import (
    SegmentSample,
    SegmentSamplingConfig,
    SegmentSamplingMetrics,
    sample_segments,
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
    n_envs: int = Field(default=1, ge=1)
    update_epochs: int = Field(default=4, ge=1)
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
    target_kl: float | None = Field(default=None, gt=0.0)
    normalize_advantages: bool = True
    advantage_eps: float = Field(default=1e-8, gt=0.0)
    advantage_mode: AdvantageMode = "gae"
    vtrace_rho_clip: float = Field(default=1.0, gt=0.0)
    vtrace_c_clip: float = Field(default=1.0, gt=0.0)
    recompute_advantages_each_epoch: bool = False
    compile_mode: CompileMode | None = None
    dtype: TrainingDType = "float32"


class ModelForwardFn(Protocol):
    def __call__(
        self, obs: ObsBatch, *, deterministic: bool = False
    ) -> ModelOutput: ...


class ModelEvaluateActionsFn(Protocol):
    def __call__(self, obs: ObsBatch, actions: ModelActions) -> ModelEvaluation: ...


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
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clipfrac: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_max: torch.Tensor


@dataclass(frozen=True)
class PPOUpdateResult:
    metrics: PPOLossMetrics
    indices: torch.Tensor
    new_logp: torch.Tensor
    new_values: torch.Tensor
    grad_norm: torch.Tensor


@dataclass(frozen=True)
class PPORolloutSegments:
    obs: ObsBatch
    actions: ModelActions
    logp: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class PPORolloutBuffer:
    def __init__(
        self,
        *,
        horizon: int,
        n_envs: int,
        obs_spec: ObsV1Config,
        action_spec: ActionPureConfig,
        device: torch.device,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_envs <= 0:
            raise ValueError("n_envs must be positive")
        self.horizon = horizon
        self.n_envs = n_envs
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
            planet_mask=torch.zeros(
                (horizon, n_envs, obs_spec.max_planets),
                dtype=torch.bool,
                device=device,
            ),
            fleet_mask=torch.zeros(
                (horizon, n_envs, obs_spec.max_fleets),
                dtype=torch.bool,
                device=device,
            ),
            comet_mask=torch.zeros(
                (horizon, n_envs, obs_spec.max_comets),
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
                (horizon, n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS),
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
            angle=torch.zeros(action_shape, dtype=torch.float32, device=device),
            ships=torch.zeros(action_shape, dtype=torch.int64, device=device),
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
        if env.n_envs != config.n_envs:
            raise ValueError(
                f"config.n_envs must match env.n_envs ({env.n_envs}), "
                f"got {config.n_envs}"
            )
        self.env = env
        self.model = model
        self._model_forward = _compile_model_forward(self.model, config.compile_mode)
        self._model_evaluate_actions = _compile_model_evaluate_actions(
            self.model,
            config.compile_mode,
        )
        self._ppo_loss = _compile_ppo_loss(config.compile_mode)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device
        self._obs = _obs_to_device(env.reset(), device)
        self.rollout = PPORolloutBuffer(
            horizon=config.horizon,
            n_envs=config.n_envs,
            obs_spec=env.obs_spec,
            action_spec=env.action_spec,
            device=device,
        )

    def train_iteration(self) -> dict[str, float]:
        start = perf_counter()
        last_values = self._collect_rollout()
        segments = self.rollout.segment_major()
        value_mask = segments.obs.still_playing
        policy_mask = _policy_mask(segments.obs)
        ratios = (
            torch.ones_like(segments.logp)
            if self.config.advantage_mode == "gae_vtrace"
            else None
        )
        advantages, returns = compute_gae(
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
        metrics = self._update(
            segments,
            advantages,
            returns,
            last_values,
            policy_mask,
            value_mask,
        )
        episode_returns = masked_sum_by_segment(segments.rewards, value_mask)
        metrics["return_mean"] = float(episode_returns.mean().item())
        metrics["return_max"] = float(episode_returns.max().item())
        metrics["explained_variance"] = float(
            explained_variance(segments.values, returns, valid_mask=value_mask).item()
        )
        metrics["advantage_mean"] = float(masked_mean(advantages, policy_mask).item())
        metrics["advantage_std"] = float(masked_std(advantages, policy_mask).item())
        elapsed = max(perf_counter() - start, 1e-12)
        metrics["steps_per_second"] = float(
            self.config.horizon * self.config.n_envs / elapsed
        )
        return metrics

    def _collect_rollout(self) -> torch.Tensor:
        self.rollout.rewards.zero_()
        self.rollout.dones.zero_()
        with torch.no_grad():
            for step in range(self.config.horizon):
                with autocast_context(self.config, self.device):
                    output = self._model_forward(self._obs)
                actions = _output_actions(output)
                next_obs, rewards, dones = _step_env(self.env, actions)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                self.rollout.write_step(
                    step,
                    obs=self._obs,
                    actions=actions,
                    logp=_output_logp(output),
                    values=_output_values(output),
                    rewards=rewards,
                    dones=dones,
                )
                self._obs = _obs_to_device(next_obs, self.device)
            with autocast_context(self.config, self.device):
                bootstrap = self._model_forward(self._obs)
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
        current_logp = segments.logp.clone()
        current_values = segments.values.clone()
        current_advantages = advantages
        current_returns = returns
        current_bootstrap_values = bootstrap_values.clone()
        n_minibatches = max(
            1,
            int(
                self.config.replay_ratio
                * self.config.n_envs
                / self.config.segment_sampling.segments_per_minibatch
            ),
        )
        should_stop = False
        for epoch in range(self.config.update_epochs):
            if self._should_recompute_advantages(epoch):
                current_logp, current_values = self._current_segment_logp_values(
                    segments
                )
                current_bootstrap_values = self._current_bootstrap_values()
                current_advantages, current_returns = self._compute_current_gae(
                    segments=segments,
                    current_logp=current_logp,
                    current_values=current_values,
                    current_bootstrap_values=current_bootstrap_values,
                )
            for minibatch_index in range(n_minibatches):
                sampling_advantages = _segment_sampling_advantages(
                    current_advantages,
                    policy_mask,
                )
                sample = sample_segments(
                    sampling_advantages,
                    self.config.segment_sampling,
                )
                sampling_metrics.append(
                    segment_sampling_metrics(sampling_advantages, sample)
                )
                update = self._update_minibatch(
                    segments,
                    current_advantages,
                    current_returns,
                    policy_mask,
                    value_mask,
                    sample,
                )
                loss_metrics.append(update.metrics)
                grad_norms.append(update.grad_norm.detach())
                current_logp[update.indices] = update.new_logp
                current_values[update.indices] = update.new_values
                if (
                    self.config.target_kl is not None
                    and update.metrics.approx_kl.item() > self.config.target_kl
                ):
                    should_stop = True
                    break
                has_next_minibatch = minibatch_index + 1 < n_minibatches
                if (
                    has_next_minibatch
                    and self._should_refresh_advantages_between_minibatches()
                ):
                    current_bootstrap_values = self._current_bootstrap_values()
                    current_advantages, current_returns = self._compute_current_gae(
                        segments=segments,
                        current_logp=current_logp,
                        current_values=current_values,
                        current_bootstrap_values=current_bootstrap_values,
                    )
            if should_stop:
                break

        if not loss_metrics:
            raise RuntimeError("internal error: PPO update produced no minibatches")
        metrics = _mean_loss_metrics(loss_metrics)
        metrics["grad_norm"] = float(torch.stack(grad_norms).mean().item())
        metrics["num_minibatches"] = float(len(loss_metrics))
        if self.lr_scheduler is not None:
            metrics["learning_rate"] = float(self.lr_scheduler.get_last_lr()[0])
        if sampling_metrics:
            metrics.update(_mean_sampling_metrics(sampling_metrics))
        return metrics

    def _should_recompute_advantages(self, epoch: int) -> bool:
        if epoch == 0:
            return False
        return (
            self.config.recompute_advantages_each_epoch
            or self.config.advantage_mode == "gae_vtrace"
        )

    def _should_refresh_advantages_between_minibatches(self) -> bool:
        return (
            self.config.recompute_advantages_each_epoch
            or self.config.advantage_mode == "gae_vtrace"
        )

    def _compute_current_gae(
        self,
        *,
        segments: PPORolloutSegments,
        current_logp: torch.Tensor,
        current_values: torch.Tensor,
        current_bootstrap_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ratios = (
            _policy_ratios(current_logp, segments.logp)
            if self.config.advantage_mode == "gae_vtrace"
            else None
        )
        return compute_gae(
            rewards=segments.rewards,
            values=current_values,
            dones=segments.dones,
            last_values=current_bootstrap_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            ratios=ratios,
            mode=self.config.advantage_mode,
            vtrace_rho_clip=self.config.vtrace_rho_clip,
            vtrace_c_clip=self.config.vtrace_c_clip,
        )

    def _current_segment_logp_values(
        self,
        segments: PPORolloutSegments,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), autocast_context(self.config, self.device):
            output = self._model_evaluate_actions(
                _flatten_obs_time(segments.obs),
                _flatten_actions_time(segments.actions),
            )
        return (
            _output_logp(output).detach().view_as(segments.logp),
            _output_values(output).detach().view_as(segments.values),
        )

    def _current_bootstrap_values(self) -> torch.Tensor:
        with torch.no_grad(), autocast_context(self.config, self.device):
            output = self._model_forward(self._obs)
        return _output_values(output).detach()

    def _update_minibatch(
        self,
        segments: PPORolloutSegments,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        policy_mask: torch.Tensor,
        value_mask: torch.Tensor,
        sample: SegmentSample,
    ) -> PPOUpdateResult:
        idx = sample.indices
        batch_actions = _flatten_actions_time(_actions_index(segments.actions, idx))
        batch_obs = _flatten_obs_time(_obs_index(segments.obs, idx))
        batch_old_logp = segments.logp[idx]
        batch_old_values = segments.values[idx]
        batch_returns = returns[idx]
        batch_policy_mask = policy_mask[idx]
        batch_value_mask = value_mask[idx]
        importance = sample.importance
        while importance.ndim < advantages[idx].ndim:
            importance = importance.unsqueeze(-1)
        batch_advantages = advantages[idx]
        batch_policy_weight = (
            batch_policy_mask.to(dtype=batch_advantages.dtype) * importance
        )
        batch_value_weight = (
            batch_value_mask.to(dtype=batch_advantages.dtype) * importance
        )

        with autocast_context(self.config, self.device):
            output = self._model_evaluate_actions(batch_obs, batch_actions)
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
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return PPOUpdateResult(
            metrics=metrics,
            indices=idx,
            new_logp=new_logp.detach(),
            new_values=new_values.detach(),
            grad_norm=grad_norm.detach(),
        )


def _compile_model_forward(
    model: BaseModelAPI, compile_mode: CompileMode | None
) -> ModelForwardFn:
    if compile_mode is None:
        return model
    compiled: ModelForwardFn = torch.compile(model, mode=compile_mode)
    return compiled


def _compile_model_evaluate_actions(
    model: BaseModelAPI, compile_mode: CompileMode | None
) -> ModelEvaluateActionsFn:
    if compile_mode is None:
        return model.evaluate_actions
    compiled: ModelEvaluateActionsFn = torch.compile(
        model.evaluate_actions,
        mode=compile_mode,
    )
    return compiled


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
                config.normalize_advantages,
                config.advantage_eps,
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
            config.normalize_advantages,
            config.advantage_eps,
            config.clip_coef,
            config.vf_clip_coef,
            config.vf_coef,
            config.ent_coef,
        )
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
    ],
) -> PPOLossMetrics:
    (
        loss,
        policy_loss,
        value_loss,
        entropy,
        approx_kl,
        clipfrac,
        ratio_mean,
        ratio_max,
    ) = tensors
    return PPOLossMetrics(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
        ratio_mean=ratio_mean,
        ratio_max=ratio_max,
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
    normalize_advantages: bool,
    advantage_eps: float,
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
]:
    normalized_advantages = advantages
    if normalize_advantages:
        valid = policy_weight > 0
        normalized_advantages = (advantages - masked_mean(advantages, valid)) / (
            masked_std(advantages, valid) + advantage_eps
        )

    logratio = new_logp - old_logp
    ratio = logratio.exp()
    pg_loss1 = -normalized_advantages * ratio
    pg_loss2 = -normalized_advantages * torch.clamp(
        ratio, 1.0 - clip_coef, 1.0 + clip_coef
    )
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
    approx_kl = weighted_mean((ratio - 1.0) - logratio, policy_weight)
    clipfrac = weighted_mean(
        ((ratio - 1.0).abs() > clip_coef).float(),
        policy_weight,
    )
    ratio_mean = weighted_mean(ratio, policy_weight)
    ratio_max = _masked_max_or_zero(ratio, policy_weight > 0)

    return (
        loss,
        policy_loss,
        value_loss,
        entropy_mean,
        approx_kl,
        clipfrac,
        ratio_mean,
        ratio_max,
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
    if new_values.numel() != returns.numel():
        raise ValueError(
            f"new_values must have {returns.numel()} elements, got {new_values.numel()}"
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
        getattr(dst, field)[step].copy_(getattr(src, field))


def _obs_segment_major(obs: ObsBatch) -> ObsBatch:
    return ObsBatch(
        **{
            field: getattr(obs, field).transpose(0, 1).contiguous()
            for field in _OBS_FIELDS
        }
    )


def _actions_segment_major(actions: ModelActions) -> ModelActions:
    return ModelActions(
        **{
            field: getattr(actions, field).transpose(0, 1).contiguous()
            for field in _ACTION_FIELDS
        }
    )


def _obs_index(obs: ObsBatch, idx: torch.Tensor) -> ObsBatch:
    return ObsBatch(**{field: getattr(obs, field)[idx] for field in _OBS_FIELDS})


def _actions_index(actions: ModelActions, idx: torch.Tensor) -> ModelActions:
    return ModelActions(
        **{field: getattr(actions, field)[idx] for field in _ACTION_FIELDS}
    )


def _obs_to_device(obs: ObsBatch, device: torch.device) -> ObsBatch:
    if device.type == "cpu":
        return ObsBatch(
            **{field: (getattr(obs, field).to(device).clone()) for field in _OBS_FIELDS}
        )

    return ObsBatch(
        **{field: (getattr(obs, field).to(device)) for field in _OBS_FIELDS}
    )


def _actions_to_cpu(actions: ModelActions) -> ModelActions:
    return ModelActions(
        **{field: getattr(actions, field).cpu() for field in _ACTION_FIELDS}
    )


def _flatten_tensor_time(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _flatten_obs_time(obs: ObsBatch) -> ObsBatch:
    return ObsBatch(
        **{field: _flatten_tensor_time(getattr(obs, field)) for field in _OBS_FIELDS}
    )


def _flatten_actions_time(actions: ModelActions) -> ModelActions:
    return ModelActions(
        **{
            field: _flatten_tensor_time(getattr(actions, field))
            for field in _ACTION_FIELDS
        }
    )


def _step_env(
    env: VectorizedEnv,
    actions: ModelActions,
) -> tuple[ObsBatch, torch.Tensor, torch.Tensor]:
    cpu_actions = _actions_to_cpu(actions)
    return env.step(cpu_actions.launch, cpu_actions.angle, cpu_actions.ships)


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
    return obs.still_playing & obs.can_act.any(dim=-1)


def _policy_ratios(new_logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    require_same_shape(new_logp, old_logp, left_name="new_logp", right_name="old_logp")
    return (new_logp - old_logp).exp()


def _masked_max_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(mask, values, torch.full_like(values, -torch.inf))
    return torch.where(
        mask.any(),
        masked.max(),
        torch.zeros((), dtype=values.dtype, device=values.device),
    )


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
    names = (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clipfrac",
        "ratio_mean",
        "ratio_max",
    )
    return {
        name: float(
            torch.stack([getattr(metric, name) for metric in metrics]).mean().item()
        )
        for name in names
    }


def _mean_sampling_metrics(metrics: list[SegmentSamplingMetrics]) -> dict[str, float]:
    metric_names = (
        ("priority_min", "priority_min"),
        ("priority_mean", "priority_mean"),
        ("priority_max", "priority_max"),
        ("priority_entropy", "probability_entropy"),
        ("sample_duplicate_frac", "duplicate_fraction"),
        ("importance_mean", "importance_mean"),
        ("importance_max", "importance_max"),
    )
    return {
        output_name: float(
            torch.stack([getattr(metric, attr_name) for metric in metrics])
            .mean()
            .item()
        )
        for output_name, attr_name in metric_names
    }
