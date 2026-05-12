from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import torch
from pydantic import Field, field_validator

from owl.config import BaseConfig
from owl.model import BaseModelAPI, ModelActions, ModelEvaluation, ModelOutput
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    OUTER_PLAYER_SLOTS,
    ActionBundle,
    ActionConfig,
    ActionDiscreteTargetBinsConfig,
    ActionDiscreteTargetsConfig,
    ActionMask,
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
from owl.train.advantages import compile_compute_gae
from owl.train.distributed import (
    DistributedContext,
    all_gather_object,
    all_reduce_max,
    all_reduce_sum,
    unwrap_model,
)
from owl.train.metrics import (
    explained_variance,
    masked_mean,
    masked_std,
    weighted_mean,
)
from owl.train.optimizer import (
    CompositeOptimizer as _CompositeOptimizer,
)
from owl.train.optimizer import (
    LRScheduler as _LRScheduler,
)
from owl.train.optimizer import (
    Optimizer as _Optimizer,
)
from owl.train.utils import (
    TrainingDType as _TrainingDType,
)
from owl.train.utils import (
    autocast_context as _autocast_context,
)
from owl.train.utils import (
    require_same_shape,
)

CompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]


_OBS_TENSOR_FIELDS = tuple(
    field for field in ObsBatch.model_fields if field != "action_mask"
)


class PPOConfig(BaseConfig):
    horizon: int = Field(default=64, ge=1)
    checkpoint_freq: int | None = Field(default=None, ge=1_000)
    ppo_epochs: int = Field(default=1, ge=1)
    segments_per_minibatch: int = Field(default=1, ge=1)
    gamma: float = Field(default=1.0, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_coef: float = Field(default=0.2, ge=0.0)
    vf_clip_coef: float | None = Field(default=0.2, gt=0.0)
    vf_coef: float = Field(default=0.5, ge=0.0)
    ent_coef: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    target_kl: float | None = Field(default=0.03, gt=0.0)
    normalize_advantages: bool = False
    eval_replay_games: int = Field(default=0, ge=0)
    compile_mode: CompileMode | None = None
    dtype: _TrainingDType = "float32"

    @field_validator("eval_replay_games")
    @classmethod
    def _validate_even_eval_replay_games(cls, value: int) -> int:
        if value % 2 != 0:
            raise ValueError("eval_replay_games must be even")
        return value


@dataclass(frozen=True)
class _PPOLossMetrics:
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
    entropy_components: dict[str, torch.Tensor] = dataclass_field(default_factory=dict)
    backward_loss: torch.Tensor | None = None


@dataclass(frozen=True)
class _PPOUpdateResult:
    metrics: _PPOLossMetrics
    indices: torch.Tensor
    new_values: torch.Tensor
    grad_norm: torch.Tensor
    target_kl_exceeded: bool = False


@dataclass(frozen=True)
class _PPORolloutSegments:
    """Rollout tensors converted from collection layout [T, N, ...] to [N, T, ...]."""

    obs: ObsBatch
    actions: ModelActions
    logp: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


@dataclass(frozen=True)
class PPOCheckpointMetadata:
    env_steps: int
    wandb_run_id: str | None


class _PPORolloutBuffer:
    """Collect rollouts in time-major layout [T, N, ...]."""

    def __init__(
        self,
        *,
        horizon: int,
        n_envs: int,
        obs_spec: EntityBasedConfig,
        action_spec: ActionConfig,
        device: torch.device,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_envs <= 0:
            raise ValueError("n_envs must be positive")
        self.horizon = horizon
        self.n_envs = n_envs
        can_act_shape: tuple[int, ...]
        if isinstance(action_spec, ActionPureConfig):
            can_act_shape = (horizon, n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS)
        elif isinstance(action_spec, ActionDiscreteTargetsConfig):
            can_act_shape = (
                horizon,
                n_envs,
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                ACTION_ENTITY_SLOTS,
            )
        else:
            can_act_shape = (
                horizon,
                n_envs,
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                ACTION_ENTITY_SLOTS,
                action_spec.n_bins,
            )
        can_act = torch.zeros(
            can_act_shape,
            dtype=torch.bool,
            device=device,
        )
        max_launch = (
            None
            if isinstance(action_spec, ActionDiscreteTargetBinsConfig)
            else torch.zeros(
                (horizon, n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS),
                dtype=torch.int64,
                device=device,
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
        self.obs = ObsBatch(
            planets=torch.zeros(
                (horizon, n_envs, obs_spec.max_planets, obs_spec.planet_channels),
                dtype=torch.float32,
                device=device,
            ),
            orbiting_planets=torch.zeros(
                (
                    horizon,
                    n_envs,
                    obs_spec.max_planets,
                ),
                dtype=torch.bool,
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
            action_mask=action_mask,
        )
        if isinstance(action_spec, ActionDiscreteTargetBinsConfig):
            action_shape: tuple[int, ...] = (
                horizon,
                n_envs,
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
            )
            self.actions: ActionBundle = DiscreteTargetBinActions(
                target=torch.zeros(action_shape, dtype=torch.int64, device=device),
                fleet_bin=torch.zeros(action_shape, dtype=torch.int64, device=device),
            )
        else:
            action_shape = (
                horizon,
                n_envs,
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                action_spec.max_per_planet_launches,
            )
            if isinstance(action_spec, ActionPureConfig):
                self.actions = PureActions(
                    launch=torch.zeros(
                        action_shape,
                        dtype=torch.bool,
                        device=device,
                    ),
                    angle=torch.zeros(action_shape, dtype=torch.float32, device=device),
                    ships=torch.zeros(action_shape, dtype=torch.int64, device=device),
                )
            else:
                self.actions = DiscreteTargetActions(
                    launch=torch.zeros(
                        action_shape,
                        dtype=torch.bool,
                        device=device,
                    ),
                    target=torch.zeros(action_shape, dtype=torch.int64, device=device),
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

    def segment_major(self) -> _PPORolloutSegments:
        """Return contiguous segment-major/time-second rollout tensors [N, T, ...]."""
        return _PPORolloutSegments(
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
        optimizer: _Optimizer,
        device: torch.device,
        lr_scheduler: _LRScheduler | None = None,
        distributed_context: DistributedContext | None = None,
    ) -> None:
        self.env = env
        self.model = model
        model_action_spec = getattr(model, "action_spec", None)
        if model_action_spec != env.action_spec:
            raise ValueError("model and env action_spec must match")
        self._compute_gae = compile_compute_gae(config.compile_mode)
        self._ppo_loss = _compile_ppo_loss(config.compile_mode)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device
        self.distributed_context = distributed_context or DistributedContext(
            device=device,
            rank=0,
            local_rank=0,
            world_size=1,
            initialized=False,
        )
        self.n_envs = env.n_envs
        self.optimizer_steps = 0
        self.player_step_total = 0
        self.total_games_played = 0
        self.target_kl_exceeded_total = 0
        self._non_blocking_env_to_device = device.type == "cuda" and getattr(
            env, "pin_memory_enabled", False
        )
        self._obs = _obs_to_device(
            env.reset(),
            device,
            non_blocking=self._non_blocking_env_to_device,
        )
        self.rollout = _PPORolloutBuffer(
            horizon=config.horizon,
            n_envs=env.n_envs,
            obs_spec=env.obs_spec,
            action_spec=env.action_spec,
            device=device,
        )
        self._last_env_metrics: dict[str, list[float]] = {}

    @property
    def world_size(self) -> int:
        return self.distributed_context.world_size

    def train_iteration(self) -> dict[str, float]:
        start = perf_counter()
        rollout_start = perf_counter()
        last_values = self._collect_rollout()
        rollout_elapsed = max(perf_counter() - rollout_start, 1e-12)
        env_metrics = self._last_env_metrics
        segments = self.rollout.segment_major()
        max_entities_seen = segments.obs.entity_mask.sum(dim=-1).max()
        value_mask = segments.obs.still_playing
        policy_mask = _policy_mask(segments.obs)
        advantages, returns = self._compute_gae(
            rewards=segments.rewards,
            values=segments.values,
            dones=segments.dones,
            last_values=last_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        update_start = perf_counter()
        metrics = self._update(
            segments,
            advantages,
            returns,
            policy_mask,
            value_mask,
        )
        update_elapsed = max(perf_counter() - update_start, 1e-12)
        player_returns, player_return_mask = _player_segment_returns(
            segments.rewards,
            value_mask,
        )
        metrics["train/return_mean"] = float(
            self._masked_mean(player_returns, player_return_mask).item()
        )
        metrics["train/return_max"] = float(
            self._masked_max(_masked_reward_max(segments.rewards, value_mask)).item()
        )
        metrics["train/explained_variance"] = float(
            self._explained_variance(
                segments.values,
                returns,
                valid_mask=value_mask,
            ).item()
        )
        metrics["train/advantage_mean"] = float(
            self._masked_mean(advantages, policy_mask).item()
        )
        metrics["train/advantage_std"] = float(
            self._masked_std(advantages, policy_mask).item()
        )
        metrics["train/max_entities"] = float(
            self._masked_max(max_entities_seen).item()
        )
        self.player_step_total += self._sum_int(value_mask.sum())
        metrics["train/player_step_total"] = float(self.player_step_total)
        env_metrics_logged = _mean_env_metrics(
            env_metrics,
            context=self.distributed_context,
            device=self.device,
        )
        total_games_played = env_metrics_logged.get("train/total_games_played")
        if total_games_played is not None:
            self.total_games_played += int(total_games_played)
            env_metrics_logged["train/total_games_played"] = float(
                self.total_games_played
            )
        metrics.update(env_metrics_logged)
        elapsed = self._max_float(max(perf_counter() - start, 1e-12))
        rollout_elapsed = self._max_float(rollout_elapsed)
        update_elapsed = self._max_float(update_elapsed)
        rollout_steps = self.config.horizon * self.n_envs * self.world_size
        update_steps = rollout_steps * metrics["sampling/minibatch_exposure"]
        metrics["time/rollout_seconds"] = float(rollout_elapsed)
        metrics["time/update_seconds"] = float(update_elapsed)
        metrics["time/iteration_seconds"] = float(elapsed)
        metrics["perf/rollout_sps"] = float(rollout_steps / rollout_elapsed)
        metrics["perf/update_sps"] = float(update_steps / update_elapsed)
        metrics["perf/steps_per_second"] = float(rollout_steps / elapsed)
        return metrics

    def write_checkpoint(
        self,
        path: Path,
        *,
        env_steps: int,
        wandb_run_id: str | None = None,
    ) -> None:
        checkpoint = {
            "model": unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": (
                None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
            ),
            "env_steps": env_steps,
            "optimizer_steps": self.optimizer_steps,
            "player_step_total": self.player_step_total,
            "total_games_played": self.total_games_played,
            "target_kl_exceeded_total": self.target_kl_exceeded_total,
            "wandb_run_id": wandb_run_id,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)

    def load_checkpoint(self, path: Path) -> PPOCheckpointMetadata:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a dictionary")

        env_steps = _checkpoint_nonnegative_int(
            checkpoint["env_steps"],
            name="env_steps",
        )
        optimizer_steps = _checkpoint_nonnegative_int(
            checkpoint["optimizer_steps"],
            name="optimizer_steps",
        )
        player_step_total = _checkpoint_nonnegative_int(
            checkpoint["player_step_total"],
            name="player_step_total",
        )
        total_games_played = _checkpoint_nonnegative_int(
            checkpoint["total_games_played"],
            name="total_games_played",
        )
        target_kl_exceeded_total = _checkpoint_nonnegative_int(
            checkpoint["target_kl_exceeded_total"],
            name="target_kl_exceeded_total",
        )
        wandb_run_id = _checkpoint_optional_str(
            checkpoint["wandb_run_id"],
            name="wandb_run_id",
        )
        unwrap_model(self.model).load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler_state = checkpoint["lr_scheduler"]
        if self.lr_scheduler is None:
            if scheduler_state is not None:
                raise ValueError(
                    "checkpoint has lr_scheduler state but trainer does not"
                )
        else:
            if scheduler_state is None:
                raise ValueError("checkpoint is missing lr_scheduler state")
            self.lr_scheduler.load_state_dict(scheduler_state)
        self.optimizer_steps = optimizer_steps
        self.player_step_total = player_step_total
        self.total_games_played = total_games_played
        self.target_kl_exceeded_total = target_kl_exceeded_total
        return PPOCheckpointMetadata(
            env_steps=env_steps,
            wandb_run_id=wandb_run_id,
        )

    def _collect_rollout(self) -> torch.Tensor:
        self.rollout.rewards.zero_()
        self.rollout.dones.zero_()
        env_metrics: dict[str, list[float]] = {}
        with torch.no_grad():
            for step in range(self.config.horizon):
                with _autocast_context(self.config, self.device):
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
            with _autocast_context(self.config, self.device):
                bootstrap = self.model(self._obs)
            self._last_env_metrics = env_metrics
            return _output_values(bootstrap).detach()

    def _update(
        self,
        segments: _PPORolloutSegments,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        policy_mask: torch.Tensor,
        value_mask: torch.Tensor,
    ) -> dict[str, float]:
        loss_metrics: list[_PPOLossMetrics] = []
        grad_norms: list[torch.Tensor] = []
        current_values = segments.values.clone()
        update_samples = _minibatch_indices(
            config=self.config,
            n_segments=self.n_envs,
            device=segments.logp.device,
        )
        n_minibatches = len(update_samples)
        sampled_segments = 0
        target_kl_exceeded = False
        for sample_indices in update_samples:
            sampled_segments += int(sample_indices.numel())
            update = self._update_minibatch(
                segments,
                advantages,
                returns,
                policy_mask,
                value_mask,
                sample_indices,
                value_clip_anchor=current_values,
            )
            loss_metrics.append(update.metrics)
            grad_norms.append(update.grad_norm.detach())
            current_values[update.indices] = update.new_values
            target_kl_exceeded = update.target_kl_exceeded
            if target_kl_exceeded:
                break

        if not loss_metrics:
            raise RuntimeError("internal error: PPO update produced no minibatches")
        metrics = _mean_loss_metrics(loss_metrics)
        if target_kl_exceeded:
            self.target_kl_exceeded_total += 1
        metrics["policy/target_kl_exceeded"] = float(target_kl_exceeded)
        metrics["policy/target_kl_exceeded_total"] = float(
            self.target_kl_exceeded_total
        )
        metrics["optimizer/grad_norm"] = float(
            self._mean_scalar(torch.stack(grad_norms).mean()).item()
        )
        metrics["optimizer/steps"] = float(self.optimizer_steps)
        metrics["optimizer/minibatches_per_update"] = float(n_minibatches)
        metrics["sampling/minibatch_exposure"] = float(
            self._sum_int(torch.tensor(sampled_segments, device=self.device))
            / (self.n_envs * self.world_size)
        )
        metrics["train/policy_active_ratio"] = float(policy_mask.float().mean().item())
        metrics["optimizer/learning_rate"] = _current_learning_rate(
            self.optimizer,
            self.lr_scheduler,
        )
        return self._reduce_mean_metrics(metrics)

    def _update_minibatch(
        self,
        segments: _PPORolloutSegments,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        policy_mask: torch.Tensor,
        value_mask: torch.Tensor,
        indices: torch.Tensor,
        *,
        value_clip_anchor: torch.Tensor,
    ) -> _PPOUpdateResult:
        idx = indices
        batch_actions = _flatten_actions_time(_actions_index(segments.actions, idx))
        batch_obs = _flatten_obs_time(_obs_index(segments.obs, idx))
        batch_old_logp = segments.logp[idx]
        batch_old_values = value_clip_anchor[idx]
        batch_returns = returns[idx]
        batch_policy_mask = policy_mask[idx]
        batch_value_mask = value_mask[idx]
        batch_advantages = advantages[idx]
        if self.config.normalize_advantages:
            batch_advantages = _normalize_masked_advantages(
                batch_advantages,
                batch_policy_mask,
            )
        batch_policy_weight = batch_policy_mask.to(dtype=batch_advantages.dtype)
        batch_value_weight = batch_value_mask.to(dtype=batch_advantages.dtype)

        with _autocast_context(self.config, self.device):
            output = self.model.evaluate_actions(batch_obs, batch_actions)
        new_logp = _output_logp(output).view_as(batch_old_logp)
        entropy = _output_entropy(output, batch_old_logp)
        entropy_components = _output_entropy_components(output, batch_old_logp)
        new_values = _output_values(output).view_as(batch_old_values)

        if self.distributed_context.initialized:
            metrics = _distributed_ppo_loss(
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
                context=self.distributed_context,
            )
        else:
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
        metrics = replace(
            metrics,
            entropy_components={
                name: self._mean_policy_metric(component, batch_policy_weight).detach()
                for name, component in entropy_components.items()
            },
        )
        self.optimizer.zero_grad(set_to_none=True)
        backward_loss = (
            metrics.loss if metrics.backward_loss is None else metrics.backward_loss
        )
        backward_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer_steps += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        target_kl_exceeded = (
            self.config.target_kl is not None
            and metrics.approx_kl.item() > self.config.target_kl
        )

        return _PPOUpdateResult(
            metrics=metrics,
            indices=idx,
            new_values=new_values.detach(),
            grad_norm=grad_norm.detach(),
            target_kl_exceeded=target_kl_exceeded,
        )

    def _mean_policy_metric(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.distributed_context.initialized:
            return _distributed_weighted_mean(values, weights, self.distributed_context)
        return weighted_mean(values, weights)

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.distributed_context.initialized:
            return _distributed_weighted_mean(
                values,
                mask.to(dtype=values.dtype),
                self.distributed_context,
            )
        return masked_mean(values, mask)

    def _masked_std(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.distributed_context.initialized:
            mean = self._masked_mean(values, mask)
            return self._masked_mean((values - mean).pow(2), mask).sqrt()
        return masked_std(values, mask)

    def _masked_max(self, value: torch.Tensor) -> torch.Tensor:
        if self.distributed_context.initialized:
            return all_reduce_max(value, self.distributed_context)
        return value

    def _explained_variance(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.distributed_context.initialized:
            return _distributed_explained_variance(
                predicted,
                target,
                valid_mask=valid_mask,
                context=self.distributed_context,
            )
        return explained_variance(predicted, target, valid_mask=valid_mask)

    def _sum_int(self, value: torch.Tensor) -> int:
        if self.distributed_context.initialized:
            total = all_reduce_sum(value.to(self.device), self.distributed_context)
            return int(total.item())
        return int(value.item())

    def _max_float(self, value: float) -> float:
        if not self.distributed_context.initialized:
            return value
        tensor = torch.tensor(value, device=self.device)
        return float(all_reduce_max(tensor, self.distributed_context).item())

    def _mean_scalar(self, value: torch.Tensor) -> torch.Tensor:
        if not self.distributed_context.initialized:
            return value
        total = all_reduce_sum(value.to(self.device), self.distributed_context)
        return total / self.world_size

    def _reduce_mean_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if not self.distributed_context.initialized or not metrics:
            return metrics
        keys = sorted(metrics)
        values = torch.tensor([metrics[key] for key in keys], device=self.device)
        reduced = all_reduce_sum(values, self.distributed_context) / self.world_size
        return {
            key: float(value)
            for key, value in zip(keys, reduced.detach().cpu().tolist(), strict=True)
        }


def _compile_ppo_loss(
    compile_mode: CompileMode | None,
) -> Callable[..., _PPOLossMetrics]:
    if compile_mode is None:
        return _ppo_loss
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
    ) -> _PPOLossMetrics:
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


def _ppo_loss(
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
) -> _PPOLossMetrics:
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


def _distributed_ppo_loss(
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
    context: DistributedContext,
) -> _PPOLossMetrics:
    logratio = new_logp - old_logp
    ratio = logratio.exp()
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(
        ratio,
        1.0 - config.clip_coef,
        1.0 + config.clip_coef,
    )
    policy_loss_values = torch.max(pg_loss1, pg_loss2)

    if config.vf_clip_coef:
        value_clipped = old_values + torch.clamp(
            new_values - old_values,
            -config.vf_clip_coef,
            config.vf_clip_coef,
        )
        value_loss_unclipped = (new_values - returns).pow(2)
        value_loss_clipped = (value_clipped - returns).pow(2)
        value_loss_values = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)
    else:
        value_loss_values = 0.5 * (new_values - returns).pow(2)

    policy_loss = _distributed_weighted_mean(
        policy_loss_values.detach(),
        policy_weight,
        context,
    )
    value_loss = _distributed_weighted_mean(
        value_loss_values.detach(),
        value_weight,
        context,
    )
    entropy_mean = _distributed_weighted_mean(entropy.detach(), policy_weight, context)
    loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy_mean
    entropy_loss = -config.ent_coef * entropy_mean
    approx_kl_values = (ratio - 1.0) - logratio
    clipfrac_values = ((ratio - 1.0).abs() > config.clip_coef).float()
    backward_policy_loss = _distributed_backward_weighted_mean(
        policy_loss_values,
        policy_weight,
        context,
    )
    backward_value_loss = _distributed_backward_weighted_mean(
        value_loss_values,
        value_weight,
        context,
    )
    backward_entropy = _distributed_backward_weighted_mean(
        entropy,
        policy_weight,
        context,
    )
    backward_loss = (
        backward_policy_loss
        + config.vf_coef * backward_value_loss
        - config.ent_coef * backward_entropy
    )

    return _PPOLossMetrics(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        entropy=entropy_mean,
        approx_kl=_distributed_weighted_mean(
            approx_kl_values.detach(),
            policy_weight,
            context,
        ),
        clipfrac=_distributed_weighted_mean(
            clipfrac_values,
            policy_weight,
            context,
        ),
        ratio_mean=_distributed_weighted_mean(ratio.detach(), policy_weight, context),
        ratio_max=_distributed_masked_max_or_zero(
            ratio.detach(),
            policy_weight > 0,
            context,
        ),
        logratio_mean=_distributed_weighted_mean(
            logratio.detach(),
            policy_weight,
            context,
        ),
        logratio_abs_max=_distributed_masked_max_or_zero(
            logratio.detach().abs(),
            policy_weight > 0,
            context,
        ),
        backward_loss=backward_loss,
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
) -> _PPOLossMetrics:
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
    return _PPOLossMetrics(
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
    vf_clip_coef: float | None,
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

    if vf_clip_coef:
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
    else:
        value_loss = 0.5 * weighted_mean(
            (new_values - returns).pow(2),
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


def _copy_action_mask_time_step(dst: ActionMask, step: int, src: ActionMask) -> None:
    if type(dst) is not type(src):
        raise ValueError(
            f"rollout action-mask type mismatch: expected {type(dst).__name__}, "
            f"got {type(src).__name__}"
        )
    dst.can_act[step].copy_(src.can_act)
    if isinstance(dst, PureActionMask | DiscreteTargetActionMask):
        if not isinstance(src, PureActionMask | DiscreteTargetActionMask):
            raise ValueError("source action mask is missing max_launch")
        dst.max_launch[step].copy_(src.max_launch)


def _action_mask_segment_major(action_mask: ActionMask) -> ActionMask:
    return _map_action_mask(
        action_mask,
        lambda tensor: tensor.transpose(0, 1).contiguous(),
    )


def _action_mask_index(action_mask: ActionMask, idx: torch.Tensor) -> ActionMask:
    return _map_action_mask(action_mask, lambda tensor: tensor[idx])


def _action_mask_to_device(
    action_mask: ActionMask,
    device: torch.device,
    *,
    non_blocking: bool,
    clone: bool,
) -> ActionMask:
    def move(tensor: torch.Tensor) -> torch.Tensor:
        moved = tensor.to(device, non_blocking=non_blocking)
        return moved.clone() if clone else moved

    return _map_action_mask(action_mask, move)


def _copy_action_mask_to_device_(
    dst: ActionMask,
    src: ActionMask,
    *,
    non_blocking: bool,
) -> None:
    if type(dst) is not type(src):
        raise ValueError(
            f"obs action-mask type mismatch: expected {type(dst).__name__}, "
            f"got {type(src).__name__}"
        )
    dst.can_act.copy_(src.can_act, non_blocking=non_blocking)
    if isinstance(dst, PureActionMask | DiscreteTargetActionMask):
        if not isinstance(src, PureActionMask | DiscreteTargetActionMask):
            raise ValueError("source action mask is missing max_launch")
        dst.max_launch.copy_(src.max_launch, non_blocking=non_blocking)


def _action_mask_flatten_time(action_mask: ActionMask) -> ActionMask:
    return _map_action_mask(action_mask, _flatten_tensor_time)


def _map_action_mask(
    action_mask: ActionMask,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> ActionMask:
    if isinstance(action_mask, PureActionMask):
        return PureActionMask(
            can_act=fn(action_mask.can_act),
            max_launch=fn(action_mask.max_launch),
        )
    if isinstance(action_mask, DiscreteTargetActionMask):
        return DiscreteTargetActionMask(
            can_act=fn(action_mask.can_act),
            max_launch=fn(action_mask.max_launch),
        )
    return DiscreteTargetBinActionMask(can_act=fn(action_mask.can_act))


def _map_action_bundle(
    actions: ActionBundle,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> ActionBundle:
    if isinstance(actions, PureActions):
        return PureActions(
            launch=fn(actions.launch),
            angle=fn(actions.angle),
            ships=fn(actions.ships),
        )
    if isinstance(actions, DiscreteTargetActions):
        return DiscreteTargetActions(
            launch=fn(actions.launch),
            target=fn(actions.target),
            ships=fn(actions.ships),
        )
    return DiscreteTargetBinActions(
        target=fn(actions.target),
        fleet_bin=fn(actions.fleet_bin),
    )


def _copy_obs_time_step(dst: ObsBatch, step: int, src: ObsBatch) -> None:
    for field in _OBS_TENSOR_FIELDS:
        dst_tensor = getattr(dst, field)
        src_tensor = getattr(src, field)
        dst_tensor[step].copy_(src_tensor)
    _copy_action_mask_time_step(dst.action_mask, step, src.action_mask)


def _copy_actions_time_step(dst: ActionBundle, step: int, src: ActionBundle) -> None:
    if type(dst) is not type(src):
        raise ValueError(
            f"rollout action bundle type mismatch: expected {type(dst).__name__}, "
            f"got {type(src).__name__}"
        )
    for field in dst.__dataclass_fields__:
        getattr(dst, field)[step].copy_(getattr(src, field))


def _obs_segment_major(obs: ObsBatch) -> ObsBatch:
    return ObsBatch(
        **{
            field: getattr(obs, field).transpose(0, 1).contiguous()
            for field in _OBS_TENSOR_FIELDS
        },
        action_mask=_action_mask_segment_major(obs.action_mask),
    )


def _actions_segment_major(actions: ActionBundle) -> ActionBundle:
    return _map_action_bundle(
        actions,
        lambda tensor: tensor.transpose(0, 1).contiguous(),
    )


def _obs_index(obs: ObsBatch, idx: torch.Tensor) -> ObsBatch:
    return ObsBatch(
        **{field: getattr(obs, field)[idx] for field in _OBS_TENSOR_FIELDS},
        action_mask=_action_mask_index(obs.action_mask, idx),
    )


def _actions_index(actions: ActionBundle, idx: torch.Tensor) -> ActionBundle:
    return _map_action_bundle(actions, lambda tensor: tensor[idx])


def _obs_to_device(
    obs: ObsBatch,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> ObsBatch:
    if device.type == "cpu":
        return ObsBatch(
            **{
                field: getattr(obs, field).to(device, non_blocking=non_blocking).clone()
                for field in _OBS_TENSOR_FIELDS
            },
            action_mask=_action_mask_to_device(
                obs.action_mask,
                device,
                non_blocking=non_blocking,
                clone=True,
            ),
        )

    return ObsBatch(
        **{
            field: getattr(obs, field).to(device, non_blocking=non_blocking)
            for field in _OBS_TENSOR_FIELDS
        },
        action_mask=_action_mask_to_device(
            obs.action_mask,
            device,
            non_blocking=non_blocking,
            clone=False,
        ),
    )


def _copy_obs_to_device_(
    dst: ObsBatch,
    src: ObsBatch,
    *,
    non_blocking: bool = False,
) -> None:
    for field in _OBS_TENSOR_FIELDS:
        dst_tensor = getattr(dst, field)
        src_tensor = getattr(src, field)
        dst_tensor.copy_(src_tensor, non_blocking=non_blocking)
    _copy_action_mask_to_device_(
        dst.action_mask,
        src.action_mask,
        non_blocking=non_blocking,
    )


def _actions_to_cpu(
    actions: ActionBundle,
    *,
    non_blocking: bool = False,
) -> ActionBundle:
    return _map_action_bundle(
        actions,
        lambda tensor: tensor.to("cpu", non_blocking=non_blocking),
    )


def _flatten_tensor_time(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _flatten_obs_time(obs: ObsBatch) -> ObsBatch:
    return ObsBatch(
        **{
            field: _flatten_tensor_time(getattr(obs, field))
            for field in _OBS_TENSOR_FIELDS
        },
        action_mask=_action_mask_flatten_time(obs.action_mask),
    )


def _flatten_actions_time(actions: ActionBundle) -> ActionBundle:
    return _map_action_bundle(actions, _flatten_tensor_time)


def _step_env(
    env: VectorizedEnv,
    actions: ActionBundle,
) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
    cpu_actions = _actions_to_cpu(actions)
    return env.step(cpu_actions)


def _extend_env_metrics(
    totals: dict[str, list[float]], step_metrics: dict[str, list[float]]
) -> None:
    for key, values in step_metrics.items():
        totals.setdefault(key, []).extend(values)


def _mean_env_metrics(
    metrics: dict[str, list[float]],
    *,
    context: DistributedContext | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    if context is not None and context.initialized:
        return _distributed_mean_env_metrics(
            metrics,
            context=context,
            device=device or context.device,
        )

    logged: dict[str, float] = {}
    for key, values in metrics.items():
        if not values:
            continue
        local = torch.tensor(
            [sum(values), len(values)],
            dtype=torch.float64,
            device=device,
        )
        if local[1].item() == 0:
            continue
        if key == "total_games_played":
            logged[f"train/{key}"] = float(local[0].item())
        else:
            logged[f"train/{key}"] = float((local[0] / local[1]).item())
    return logged


def _distributed_mean_env_metrics(
    metrics: dict[str, list[float]],
    *,
    context: DistributedContext,
    device: torch.device,
) -> dict[str, float]:
    key_sets = all_gather_object(set(metrics), context)
    keys = sorted(set().union(*key_sets))
    if not keys:
        return {}

    local = torch.tensor(
        [[sum(metrics.get(key, ())), len(metrics.get(key, ()))] for key in keys],
        dtype=torch.float64,
        device=device,
    )
    totals = all_reduce_sum(local, context)
    logged: dict[str, float] = {}
    for key, total in zip(keys, totals, strict=True):
        if total[1].item() == 0:
            continue
        if key == "total_games_played":
            logged[f"train/{key}"] = float(total[0].item())
        else:
            logged[f"train/{key}"] = float((total[0] / total[1]).item())
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
    optimizer: _Optimizer,
    lr_scheduler: _LRScheduler | None,
) -> float:
    if lr_scheduler is not None:
        return float(lr_scheduler.get_last_lr()[0])
    if isinstance(optimizer, _CompositeOptimizer):
        return _torch_optimizer_learning_rate(optimizer.optimizers[0])
    if isinstance(optimizer, torch.optim.Optimizer):
        return _torch_optimizer_learning_rate(optimizer)
    raise TypeError("optimizer must be a torch optimizer or CompositeOptimizer")


def _torch_optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _checkpoint_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint {name} must be an integer")
    if value < 0:
        raise ValueError(f"checkpoint {name} must be non-negative")
    return value


def _checkpoint_optional_str(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"checkpoint {name} must be a non-empty string or None")
    return value


def _output_actions(output: ModelOutput) -> ModelActions:
    return output.actions


def _output_logp(output: ModelOutput | ModelEvaluation) -> torch.Tensor:
    return output.log_probs.per_player_entity.sum(dim=-1)


def _output_entropy(
    output: ModelOutput | ModelEvaluation, like: torch.Tensor
) -> torch.Tensor:
    return output.entropies.per_player_entity.sum(dim=-1).view_as(like)


def _output_entropy_components(
    output: ModelOutput | ModelEvaluation,
    like: torch.Tensor,
) -> dict[str, torch.Tensor]:
    components = output.entropies.components
    if components:
        return {
            name: _sum_entropy_component(component, like)
            for name, component in components.items()
        }
    fallback = {
        "launch": _sum_entropy_component(output.entropies.launch, like),
        "event": _sum_entropy_component(
            output.entropies.event,
            like,
        ),
    }
    if output.entropies.target is not None:
        fallback["target"] = _sum_entropy_component(output.entropies.target, like)
    return fallback


def _sum_entropy_component(tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if tensor.shape == like.shape:
        return tensor.view_as(like)
    return tensor.flatten(start_dim=2).sum(dim=-1).view_as(like)


def _output_values(output: ModelOutput | ModelEvaluation) -> torch.Tensor:
    return output.values


def _policy_mask(obs: ObsBatch) -> torch.Tensor:
    can_act = obs.action_mask.can_act.flatten(start_dim=3).any(dim=-1)
    return obs.still_playing & can_act


def _minibatch_indices(
    *,
    config: PPOConfig,
    n_segments: int,
    device: torch.device,
) -> list[torch.Tensor]:
    samples: list[torch.Tensor] = []
    for _epoch in range(config.ppo_epochs):
        permutation = torch.randperm(n_segments, device=device)
        samples.extend(permutation.split(config.segments_per_minibatch))
    return samples


def _masked_max_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(mask, values, torch.full_like(values, -torch.inf))
    return torch.where(
        mask.any(),
        masked.max(),
        torch.zeros((), dtype=values.dtype, device=values.device),
    )


def _distributed_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    totals = torch.stack(
        [
            (values * weights).sum(),
            weights.sum().to(dtype=values.dtype),
        ]
    )
    totals = all_reduce_sum(totals, context)
    return totals[0] / totals[1].clamp_min(1e-8)


def _distributed_backward_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    local_numerator = (values * weights).sum()
    global_denominator = all_reduce_sum(weights.sum().to(dtype=values.dtype), context)
    return local_numerator * context.world_size / global_denominator.clamp_min(1e-8)


def _distributed_masked_max_or_zero(
    values: torch.Tensor,
    mask: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    return all_reduce_max(_masked_max_or_zero(values, mask), context)


def _distributed_explained_variance(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    values = target[valid_mask]
    errors = (target - predicted)[valid_mask]
    local = torch.stack(
        [
            values.sum(),
            values.pow(2).sum(),
            errors.sum(),
            errors.pow(2).sum(),
            torch.tensor(values.numel(), dtype=target.dtype, device=target.device),
        ]
    )
    total = all_reduce_sum(local, context)
    count = total[4].clamp_min(1.0)
    target_mean = total[0] / count
    error_mean = total[2] / count
    target_variance = total[1] / count - target_mean.pow(2)
    error_variance = total[3] / count - error_mean.pow(2)
    if target_variance == 0:
        return torch.zeros((), dtype=predicted.dtype, device=predicted.device)
    return 1.0 - error_variance / target_variance


def _normalize_masked_advantages(
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


def _mean_loss_metrics(metrics: list[_PPOLossMetrics]) -> dict[str, float]:
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
    logged: dict[str, float] = {}
    for output_name, attr_name in metric_names:
        logged[output_name] = float(
            torch.stack([getattr(metric, attr_name) for metric in metrics])
            .mean()
            .item()
        )
    for name in _entropy_component_names(metrics):
        logged[f"policy/{name}_entropy"] = float(
            torch.stack([metric.entropy_components[name] for metric in metrics])
            .mean()
            .item()
        )
    return logged


def _entropy_component_names(metrics: list[_PPOLossMetrics]) -> tuple[str, ...]:
    if not metrics:
        return ()
    names = set(metrics[0].entropy_components)
    for metric in metrics[1:]:
        names &= set(metric.entropy_components)
    return tuple(sorted(names))
