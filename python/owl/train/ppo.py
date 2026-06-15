from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from time import perf_counter
from typing import Literal, Self, cast

import torch
from pydantic import Field, model_validator

from owl.config import BaseConfig
from owl.model import (
    ActorDiscreteTargetsConfig,
    BaseModelAPI,
    CachedTeacherDistillationTargets,
    ModelActionKLDivergences,
    ModelActions,
    ModelEvaluation,
    ModelHiddenState,
    ModelOutput,
    ModelTeacherEvaluation,
    StatelessTransformerV1,
    concat_teacher_distillation_targets,
    index_teacher_distillation_targets,
)
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
    EntityBasedBaseConfig,
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
    model_no_sync_context,
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
    ModelCompileMode as _ModelCompileMode,
)
from owl.train.utils import (
    ModelCompileTarget as _ModelCompileTarget,
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
PPOClipMode = Literal["per_player", "per_entity"]
TeacherMode = Literal["last_best", "fixed"]


_OBS_TENSOR_FIELDS = tuple(
    field
    for field in ObsBatch.model_fields
    if field
    not in {
        "action_mask",
        "player_features",
        "fleet_target",
        "target_incoming_features",
    }
)
_OBS_OPTIONAL_TENSOR_FIELDS = (
    "player_features",
    "fleet_target",
    "target_incoming_features",
)


class PPOConfig(BaseConfig):
    horizon: int = Field(default=64, ge=1)
    checkpoint_freq: int | None = Field(default=None, ge=1_000)
    ppo_epochs: int = Field(default=1, ge=1)
    segments_per_minibatch: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    gamma: float = Field(default=1.0, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_coef: float = Field(default=0.2, ge=0.0)
    vf_clip_coef: float | None = Field(default=0.2, gt=0.0)
    vf_coef: float = Field(default=0.5, ge=0.0)
    ent_coef: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    target_kl: float | None = Field(default=0.03, gt=0.0)
    ppo_clip_mode: PPOClipMode = "per_player"
    normalize_advantages: bool = False
    eval_replay_games: int = Field(default=0, ge=0)
    teacher_mode: TeacherMode | None = None
    teacher_init: Path | None = None
    teacher_kl_coef: float = Field(default=0.001, ge=0.0)
    teacher_value_coef: float = Field(default=0.001, ge=0.0)
    teacher_segments_per_minibatch: int = Field(default=32, ge=1)
    compile_mode: CompileMode | None = None
    model_compile: _ModelCompileTarget = "trunk"
    model_compile_mode: _ModelCompileMode = "max-autotune-no-cudagraphs"
    dtype: _TrainingDType = "float32"

    @model_validator(mode="after")
    def _validate_teacher_config(self) -> Self:
        if self.teacher_mode == "fixed" and self.teacher_init is None:
            raise ValueError("rl.teacher_init is required when rl.teacher_mode='fixed'")
        return self


@dataclass(frozen=True)
class _PPOLossMetrics:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    teacher_kl_loss: torch.Tensor
    teacher_value_loss: torch.Tensor
    entropy: torch.Tensor
    teacher_kl: torch.Tensor
    teacher_value_cross_entropy: torch.Tensor
    approx_kl: torch.Tensor
    clipfrac: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_max: torch.Tensor
    logratio_mean: torch.Tensor
    logratio_abs_max: torch.Tensor
    entropy_components: dict[str, torch.Tensor] = dataclass_field(default_factory=dict)
    teacher_kl_components: dict[str, torch.Tensor] = dataclass_field(
        default_factory=dict
    )


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
    entity_logp: torch.Tensor | None = None
    initial_hidden_state: ModelHiddenState | None = None


@dataclass(frozen=True, kw_only=True)
class PPOCheckpointMetadata:
    env_steps: int
    player_step_total: int = 0
    total_games_played: int = 0
    wandb_run_id: str | None = None


class _PPORolloutBuffer:
    """Collect rollouts in time-major layout [T, N, ...]."""

    def __init__(
        self,
        *,
        horizon: int,
        n_envs: int,
        obs_spec: EntityBasedBaseConfig,
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
            fleet_target=(
                None
                if not obs_spec.uses_cross_attention
                else torch.full(
                    (horizon, n_envs, obs_spec.max_fleets),
                    -1,
                    dtype=torch.int64,
                    device=device,
                )
            ),
            target_incoming_features=(
                None
                if not obs_spec.uses_cross_attention
                else torch.zeros(
                    (
                        horizon,
                        n_envs,
                        ACTION_ENTITY_SLOTS,
                        obs_spec.target_incoming_channels,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
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
            player_features=(
                None
                if obs_spec.player_feature_channels == 0
                else torch.zeros(
                    (
                        horizon,
                        n_envs,
                        OUTER_PLAYER_SLOTS,
                        obs_spec.player_feature_channels,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            ),
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
        self.entity_logp = torch.zeros(
            (horizon, n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS),
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
        self.initial_hidden_state: ModelHiddenState | None = None

    def write_step(
        self,
        step: int,
        *,
        obs: ObsBatch,
        actions: ModelActions,
        logp: torch.Tensor,
        entity_logp: torch.Tensor | None = None,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        if not 0 <= step < self.horizon:
            raise ValueError(f"step must be in 0..{self.horizon - 1}, got {step}")
        _copy_obs_time_step(self.obs, step, obs)
        _copy_actions_time_step(self.actions, step, actions)
        self.logp[step].copy_(logp)
        if entity_logp is None:
            self.entity_logp[step].zero_()
        else:
            self.entity_logp[step].copy_(entity_logp)
        self.values[step].copy_(values)
        self.rewards[step].copy_(rewards)
        self.dones[step].copy_(dones)

    def segment_major(self) -> _PPORolloutSegments:
        """Return contiguous segment-major/time-second rollout tensors [N, T, ...]."""
        return _PPORolloutSegments(
            obs=_obs_segment_major(self.obs),
            actions=_actions_segment_major(self.actions),
            logp=self.logp.transpose(0, 1).contiguous(),
            entity_logp=self.entity_logp.transpose(0, 1).contiguous(),
            values=self.values.transpose(0, 1).contiguous(),
            rewards=self.rewards.transpose(0, 1).contiguous(),
            dones=self.dones.transpose(0, 1).contiguous(),
            initial_hidden_state=self.initial_hidden_state,
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
        teacher_model: BaseModelAPI | None = None,
        teacher_active: bool = False,
        distributed_context: DistributedContext | None = None,
    ) -> None:
        self.env = env
        self.model = model
        if model.action_spec != env.action_spec:
            raise ValueError("model and env action_spec must match")
        if teacher_model is not None and teacher_model.action_spec != env.action_spec:
            raise ValueError("teacher model and env action_spec must match")
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
        _validate_minibatch_divisibility(env.n_envs, config)
        self.n_envs = env.n_envs
        self.optimizer_steps = 0
        self.player_step_total = 0
        self.total_games_played = 0
        self.target_kl_exceeded_total = 0
        self._non_blocking_env_to_device = (
            device.type == "cuda" and env.pin_memory_enabled
        )
        self._obs = _obs_to_device(
            env.reset(),
            device,
            non_blocking=self._non_blocking_env_to_device,
        )
        self._hidden_state = model.initial_hidden_state(env.n_envs, device=device)
        self.teacher_model: BaseModelAPI | None = None
        self.teacher_active = False
        self.rollout = _PPORolloutBuffer(
            horizon=config.horizon,
            n_envs=env.n_envs,
            obs_spec=env.obs_spec,
            action_spec=env.action_spec,
            device=device,
        )
        self._last_env_metrics: dict[str, list[float]] = {}
        self.set_teacher_model(teacher_model, active=teacher_active)

    @property
    def world_size(self) -> int:
        return self.distributed_context.world_size

    def set_teacher_model(
        self,
        teacher_model: BaseModelAPI | None,
        *,
        active: bool,
    ) -> None:
        if (
            teacher_model is not None
            and teacher_model.action_spec != self.env.action_spec
        ):
            raise ValueError("teacher model and env action_spec must match")
        teacher_active = teacher_model is not None and active
        if teacher_active:
            if teacher_model is None:
                raise RuntimeError("teacher_active requires a teacher model")
            _require_stateless_teacher(
                teacher_model,
                batch_size=self.n_envs,
                device=self.device,
            )
            teacher_losses_enabled = (
                self.config.teacher_kl_coef > 0.0
                or self.config.teacher_value_coef > 0.0
            )
            if teacher_losses_enabled and not (
                unwrap_model(self.model).supports_cached_teacher_distillation()
                and teacher_model.supports_cached_teacher_distillation()
            ):
                raise ValueError(
                    "teacher distillation requires the discrete_targets actor "
                    "without player-count adapters for both the student and the "
                    "teacher model"
                )
            if (
                self.config.teacher_mode == "fixed"
                and self.config.teacher_kl_coef > 0.0
            ):
                _validate_fixed_teacher_action_compatibility(
                    unwrap_model(self.model),
                    teacher_model,
                )
            teacher_model.eval()
            for parameter in teacher_model.parameters():
                parameter.requires_grad_(False)
        self.teacher_model = teacher_model
        self.teacher_active = teacher_active

    def train_iteration(self) -> dict[str, float]:
        start = perf_counter()
        rollout_start = perf_counter()
        last_values = self._collect_rollout()
        rollout_elapsed = max(perf_counter() - rollout_start, 1e-12)
        env_metrics = self._last_env_metrics
        segments = self.rollout.segment_major()
        teacher_targets: CachedTeacherDistillationTargets | None = None
        teacher_elapsed = 0.0
        teacher_model = self.teacher_model if self.teacher_active else None
        teacher_losses_enabled = (
            self.config.teacher_kl_coef > 0.0 or self.config.teacher_value_coef > 0.0
        )
        if teacher_model is not None and teacher_losses_enabled:
            teacher_start = perf_counter()
            teacher_targets = self._precompute_teacher_targets(segments)
            teacher_elapsed = max(perf_counter() - teacher_start, 1e-12)
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
        metrics, sampled_segments = self._update(
            segments,
            advantages,
            returns,
            policy_mask,
            value_mask,
            teacher_targets,
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
        for key, rate in _player_count_rates(segments.obs.still_playing).items():
            metrics[key] = float(self._mean_scalar(rate).item())
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
        teacher_elapsed = self._max_float(teacher_elapsed)
        update_elapsed = self._max_float(update_elapsed)
        rollout_steps = self.config.horizon * self.n_envs * self.world_size
        update_steps = self.config.horizon * sampled_segments
        metrics["time/rollout_seconds"] = float(rollout_elapsed)
        metrics["time/teacher_seconds"] = float(teacher_elapsed)
        metrics["time/update_seconds"] = float(update_elapsed)
        metrics["time/iteration_seconds"] = float(elapsed)
        metrics["perf/rollout_sps"] = float(rollout_steps / rollout_elapsed)
        metrics["perf/teacher_sps"] = (
            float(rollout_steps / teacher_elapsed) if teacher_elapsed > 0.0 else 0.0
        )
        metrics["perf/update_sps"] = float(update_steps / update_elapsed)
        metrics["perf/steps_per_second"] = float(rollout_steps / elapsed)
        return metrics

    def write_checkpoint(
        self,
        path: Path,
        *,
        env_steps: int,
        wandb_run_id: str | None = None,
        model: BaseModelAPI | None = None,
    ) -> None:
        checkpoint_model = unwrap_model(self.model if model is None else model)
        checkpoint = {
            "model": checkpoint_model.state_dict(),
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

        metadata = _checkpoint_metadata(checkpoint)
        optimizer_steps = _checkpoint_nonnegative_int(
            checkpoint["optimizer_steps"],
            name="optimizer_steps",
        )
        target_kl_exceeded_total = _checkpoint_nonnegative_int(
            checkpoint["target_kl_exceeded_total"],
            name="target_kl_exceeded_total",
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
        self.player_step_total = metadata.player_step_total
        self.total_games_played = metadata.total_games_played
        self.target_kl_exceeded_total = target_kl_exceeded_total
        return metadata

    def load_model_weights(self, path: Path) -> PPOCheckpointMetadata:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a dictionary")

        metadata = _checkpoint_metadata(checkpoint)
        unwrap_model(self.model).load_state_dict(checkpoint["model"])
        self.player_step_total = metadata.player_step_total
        self.total_games_played = metadata.total_games_played
        return metadata

    def _collect_rollout(self) -> torch.Tensor:
        self.rollout.rewards.zero_()
        self.rollout.dones.zero_()
        self.rollout.initial_hidden_state = self.model.detach_hidden_state(
            self._hidden_state
        )
        env_metrics: dict[str, list[float]] = {}
        with torch.no_grad():
            for step in range(self.config.horizon):
                with _autocast_context(self.config, self.device):
                    output = _model_forward(
                        self.model,
                        self._obs,
                        hidden_state=self._hidden_state,
                    )
                self._hidden_state = output.next_hidden_state
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
                    entity_logp=_output_entity_logp(output),
                    values=_output_values(output),
                    rewards=rewards,
                    dones=dones,
                )
                self._hidden_state = self.model.reset_hidden_state(
                    self._hidden_state,
                    dones,
                )
                _copy_obs_to_device_(
                    self._obs,
                    next_obs,
                    non_blocking=self._non_blocking_env_to_device,
                )
            with _autocast_context(self.config, self.device):
                last_values = _model_compute_value(
                    self.model,
                    self._obs,
                    hidden_state=self._hidden_state,
                )
            self._last_env_metrics = env_metrics
            return last_values.detach()

    def _precompute_teacher_targets(
        self,
        segments: _PPORolloutSegments,
    ) -> CachedTeacherDistillationTargets | None:
        """Run the frozen teacher trunk once per iteration over the rollout.

        Caches the teacher's action-distribution params and winner probabilities
        in segment-major layout so each update minibatch can compute the KL /
        value-distillation losses without re-running the teacher trunk. Runs as a
        chunked ``no_grad`` inference loop (chunk size
        ``teacher_segments_per_minibatch`` segments) outside DDP, since the teacher
        is a raw, frozen model.
        """
        teacher_model = self.teacher_model if self.teacher_active else None
        if teacher_model is None:
            return None
        compute_action_kl = self.config.teacher_kl_coef > 0.0
        compute_value = self.config.teacher_value_coef > 0.0
        if not (compute_action_kl or compute_value):
            return None
        chunk_size = self.config.teacher_segments_per_minibatch
        chunks: list[CachedTeacherDistillationTargets] = []
        for start in range(0, self.n_envs, chunk_size):
            stop = min(start + chunk_size, self.n_envs)
            chunk_idx = torch.arange(start, stop, device=segments.logp.device)
            chunk_obs = _obs_index(segments.obs, chunk_idx)
            chunk_actions = _actions_index(segments.actions, chunk_idx)
            with torch.no_grad(), _autocast_context(self.config, self.device):
                chunks.append(
                    teacher_model.compute_teacher_distillation_targets(
                        chunk_obs,
                        chunk_actions,
                        compute_action_kl=compute_action_kl,
                        compute_value=compute_value,
                    )
                )
        return concat_teacher_distillation_targets(chunks)

    def _update(
        self,
        segments: _PPORolloutSegments,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        policy_mask: torch.Tensor,
        value_mask: torch.Tensor,
        teacher_targets: CachedTeacherDistillationTargets | None,
    ) -> tuple[dict[str, float], int]:
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
        accumulation_steps = self.config.gradient_accumulation_steps
        for sample_index, sample_indices in enumerate(update_samples):
            if sample_index % accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
            sampled_segments += int(sample_indices.numel())
            sync_gradients = (sample_index + 1) % accumulation_steps == 0
            with model_no_sync_context(self.model, enabled=not sync_gradients):
                update = self._update_minibatch(
                    segments,
                    advantages,
                    returns,
                    policy_mask,
                    value_mask,
                    sample_indices,
                    teacher_targets=teacher_targets,
                    value_clip_anchor=current_values,
                    loss_scale=1.0 / accumulation_steps,
                    step_optimizer=False,
                )
            loss_metrics.append(update.metrics)
            current_values[update.indices] = update.new_values
            target_kl_exceeded = target_kl_exceeded or update.target_kl_exceeded
            if (sample_index + 1) % accumulation_steps == 0:
                grad_norms.append(self._step_optimizer().detach())
            if target_kl_exceeded and (sample_index + 1) % accumulation_steps == 0:
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
        metrics["train/policy_active_ratio"] = float(policy_mask.float().mean().item())
        metrics["optimizer/learning_rate"] = _current_learning_rate(
            self.optimizer,
            self.lr_scheduler,
        )
        sampled_segment_total = self._sum_int(
            torch.tensor(sampled_segments, device=self.device)
        )
        return self._reduce_mean_metrics(metrics), sampled_segment_total

    def _update_minibatch(
        self,
        segments: _PPORolloutSegments,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        policy_mask: torch.Tensor,
        value_mask: torch.Tensor,
        indices: torch.Tensor,
        *,
        teacher_targets: CachedTeacherDistillationTargets | None = None,
        value_clip_anchor: torch.Tensor,
        loss_scale: float = 1.0,
        step_optimizer: bool = True,
    ) -> _PPOUpdateResult:
        idx = indices
        batch_segment_actions = _actions_index(segments.actions, idx)
        batch_segment_obs = _obs_index(segments.obs, idx)
        batch_hidden_state = self.model.index_hidden_state(
            segments.initial_hidden_state,
            idx,
        )
        teacher_model = self.teacher_model if self.teacher_active else None
        if batch_hidden_state is None:
            batch_actions = _flatten_actions_time(batch_segment_actions)
            batch_obs = _flatten_obs_time(batch_segment_obs)
        else:
            batch_actions = batch_segment_actions
            batch_obs = batch_segment_obs
        batch_old_logp = _old_policy_logp_for_clip_mode(
            segments,
            idx,
            self.config.ppo_clip_mode,
        )
        batch_old_values = value_clip_anchor[idx]
        batch_returns = returns[idx]
        batch_policy_mask = policy_mask[idx]
        batch_value_mask = value_mask[idx]
        batch_entity_policy_mask = (
            _policy_entity_mask(batch_segment_obs)
            if self.config.ppo_clip_mode == "per_entity"
            else None
        )
        batch_advantages = advantages[idx]
        if self.config.normalize_advantages:
            batch_advantages = _normalize_masked_advantages(
                batch_advantages,
                batch_policy_mask,
                context=(
                    self.distributed_context
                    if self.distributed_context.initialized
                    else None
                ),
            )
        batch_policy_weight = batch_policy_mask.to(dtype=batch_advantages.dtype)
        batch_value_weight = batch_value_mask.to(dtype=batch_advantages.dtype)
        batch_entity_policy_weight = (
            None
            if batch_entity_policy_mask is None
            else batch_entity_policy_mask.to(dtype=batch_advantages.dtype)
        )
        compute_teacher_action_kl = (
            teacher_model is not None and self.config.teacher_kl_coef > 0.0
        )
        compute_teacher_value = (
            teacher_model is not None and self.config.teacher_value_coef > 0.0
        )

        with _autocast_context(self.config, self.device):
            if teacher_targets is None:
                output = _model_evaluate_actions(
                    self.model,
                    batch_obs,
                    batch_actions,
                    hidden_state=batch_hidden_state,
                    dones=segments.dones[idx],
                )
                teacher_action_kl = None
                teacher_winner_probabilities = None
                student_winner_log_probabilities = None
            else:
                teacher_evaluation = _model_evaluate_actions_with_cached_teacher(
                    self.model,
                    batch_segment_obs,
                    batch_segment_actions,
                    index_teacher_distillation_targets(teacher_targets, idx),
                    hidden_state=batch_hidden_state,
                    dones=segments.dones[idx],
                    compute_teacher_action_kl=compute_teacher_action_kl,
                    compute_teacher_value=compute_teacher_value,
                )
                output = teacher_evaluation.student
                teacher_action_kl = teacher_evaluation.action_kl
                teacher_winner_probabilities = (
                    teacher_evaluation.teacher_winner_probabilities
                )
                student_winner_log_probabilities = (
                    teacher_evaluation.student_winner_log_probabilities
                )
        new_logp = _output_logp_for_clip_mode(
            output,
            self.config.ppo_clip_mode,
        ).view_as(batch_old_logp)
        entropy = _output_entropy_for_clip_mode(
            output,
            batch_old_logp,
            self.config.ppo_clip_mode,
        )
        entropy_components = _output_entropy_components(output, segments.logp[idx])
        new_values = _output_values(output).view_as(batch_old_values)
        if teacher_model is None:
            teacher_kl = torch.zeros_like(batch_policy_weight)
            teacher_kl_components: dict[str, torch.Tensor] = {}
            teacher_value_loss_values = torch.zeros_like(batch_old_values[..., 0])
        else:
            if not compute_teacher_action_kl:
                teacher_kl = torch.zeros_like(batch_policy_weight)
                teacher_kl_components = {}
            elif teacher_action_kl is None:
                raise RuntimeError("teacher action KL was not computed")
            else:
                teacher_kl = _output_action_kl(
                    teacher_action_kl,
                    batch_policy_weight,
                )
                teacher_kl_components = _output_action_kl_components(
                    teacher_action_kl,
                    segments.logp[idx],
                )
            if not compute_teacher_value:
                teacher_value_loss_values = torch.zeros_like(batch_old_values[..., 0])
            elif teacher_winner_probabilities is None:
                raise RuntimeError("teacher winner probabilities were not computed")
            elif student_winner_log_probabilities is None:
                raise RuntimeError("student winner log probabilities were not computed")
            else:
                teacher_value_loss_values = _teacher_value_cross_entropy(
                    student_winner_log_probabilities.view_as(batch_old_values),
                    teacher_winner_probabilities.view_as(batch_old_values),
                )

        loss_kwargs = {
            "new_logp": new_logp,
            "entropy": entropy,
            "new_values": new_values,
            "old_logp": batch_old_logp,
            "old_values": batch_old_values,
            "returns": batch_returns,
            "advantages": batch_advantages,
            "policy_weight": batch_policy_weight,
            "value_weight": batch_value_weight,
            "config": self.config,
            "context": (
                self.distributed_context
                if self.distributed_context.initialized
                else None
            ),
        }
        if teacher_model is not None:
            loss_kwargs["teacher_kl"] = teacher_kl
            loss_kwargs["teacher_value_loss_values"] = teacher_value_loss_values
        if batch_entity_policy_weight is not None:
            loss_kwargs["entity_policy_weight"] = batch_entity_policy_weight
        metrics, backward_loss = self._ppo_loss(**loss_kwargs)
        metrics = replace(
            metrics,
            entropy_components={
                name: self._mean_policy_metric(component, batch_policy_weight).detach()
                for name, component in entropy_components.items()
            },
            teacher_kl_components={
                name: self._mean_policy_metric(component, batch_policy_weight).detach()
                for name, component in teacher_kl_components.items()
            },
        )
        metrics = _detach_loss_metrics(metrics)
        if step_optimizer:
            self.optimizer.zero_grad(set_to_none=True)
        (backward_loss * loss_scale).backward()
        if step_optimizer:
            grad_norm = self._step_optimizer()
        else:
            grad_norm = backward_loss.detach().new_zeros(())
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

    def _step_optimizer(self) -> torch.Tensor:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer_steps += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return grad_norm

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
) -> Callable[..., tuple[_PPOLossMetrics, torch.Tensor]]:
    if compile_mode is None:
        return _ppo_loss
    compiled_loss_components = torch.compile(_ppo_loss_components, mode=compile_mode)

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
        teacher_kl: torch.Tensor | None = None,
        teacher_value_loss_values: torch.Tensor | None = None,
        context: DistributedContext | None = None,
        entity_policy_weight: torch.Tensor | None = None,
    ) -> tuple[_PPOLossMetrics, torch.Tensor]:
        return _ppo_loss(
            new_logp=new_logp,
            entropy=entropy,
            teacher_kl=teacher_kl,
            new_values=new_values,
            teacher_value_loss_values=teacher_value_loss_values,
            old_logp=old_logp,
            old_values=old_values,
            returns=returns,
            advantages=advantages,
            policy_weight=policy_weight,
            value_weight=value_weight,
            config=config,
            context=context,
            loss_components=compiled_loss_components,
            entity_policy_weight=entity_policy_weight,
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
    teacher_kl: torch.Tensor | None = None,
    teacher_value_loss_values: torch.Tensor | None = None,
    context: DistributedContext | None = None,
    loss_components: Callable[..., tuple[torch.Tensor, ...]] | None = None,
    entity_policy_weight: torch.Tensor | None = None,
) -> tuple[_PPOLossMetrics, torch.Tensor]:
    if loss_components is None:
        loss_components = _ppo_loss_components
    if teacher_kl is None:
        teacher_kl = torch.zeros_like(policy_weight)
    if teacher_value_loss_values is None:
        teacher_value_loss_values = torch.zeros_like(value_weight[..., 0])
    return _ppo_loss_metrics_from_components(
        loss_components(
            new_logp,
            entropy,
            new_values,
            old_logp,
            old_values,
            returns,
            advantages,
            config.clip_coef,
            config.vf_clip_coef,
            config.ppo_clip_mode,
            entity_policy_weight,
        ),
        policy_weight=policy_weight,
        value_weight=value_weight,
        teacher_kl_values=teacher_kl,
        teacher_value_loss_values=teacher_value_loss_values,
        teacher_kl_coef=config.teacher_kl_coef,
        teacher_value_coef=config.teacher_value_coef,
        vf_coef=config.vf_coef,
        ent_coef=config.ent_coef,
        context=context,
    )


def _ppo_loss_metrics_from_components(
    components: tuple[torch.Tensor, ...],
    *,
    policy_weight: torch.Tensor,
    value_weight: torch.Tensor,
    teacher_kl_values: torch.Tensor,
    teacher_value_loss_values: torch.Tensor,
    teacher_kl_coef: float,
    teacher_value_coef: float,
    vf_coef: float,
    ent_coef: float,
    context: DistributedContext | None,
) -> tuple[_PPOLossMetrics, torch.Tensor]:
    (
        policy_loss_values,
        value_loss_values,
        entropy_values,
        approx_kl_values,
        clipfrac_values,
        ratio,
        logratio,
    ) = components
    if context is None or not context.initialized:
        return _local_ppo_loss_metrics(
            policy_loss_values,
            value_loss_values,
            entropy_values,
            approx_kl_values,
            clipfrac_values,
            ratio,
            logratio,
            policy_weight=policy_weight,
            value_weight=value_weight,
            teacher_kl_values=teacher_kl_values,
            teacher_value_loss_values=teacher_value_loss_values,
            teacher_kl_coef=teacher_kl_coef,
            teacher_value_coef=teacher_value_coef,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
        )
    return _distributed_ppo_loss_metrics(
        policy_loss_values,
        value_loss_values,
        entropy_values,
        approx_kl_values,
        clipfrac_values,
        ratio,
        logratio,
        policy_weight=policy_weight,
        value_weight=value_weight,
        teacher_kl_values=teacher_kl_values,
        teacher_value_loss_values=teacher_value_loss_values,
        teacher_kl_coef=teacher_kl_coef,
        teacher_value_coef=teacher_value_coef,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        context=context,
    )


def _local_ppo_loss_metrics(
    policy_loss_values: torch.Tensor,
    value_loss_values: torch.Tensor,
    entropy_values: torch.Tensor,
    approx_kl_values: torch.Tensor,
    clipfrac_values: torch.Tensor,
    ratio: torch.Tensor,
    logratio: torch.Tensor,
    *,
    policy_weight: torch.Tensor,
    value_weight: torch.Tensor,
    teacher_kl_values: torch.Tensor,
    teacher_value_loss_values: torch.Tensor,
    teacher_kl_coef: float,
    teacher_value_coef: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[_PPOLossMetrics, torch.Tensor]:
    policy_loss = weighted_mean(policy_loss_values, policy_weight)
    value_loss = weighted_mean(value_loss_values, value_weight)
    entropy_mean = weighted_mean(entropy_values, policy_weight)
    teacher_kl = weighted_mean(teacher_kl_values, policy_weight)
    teacher_value_cross_entropy = _teacher_value_weighted_mean(
        teacher_value_loss_values,
        value_weight,
    )
    teacher_kl_loss = teacher_kl_coef * teacher_kl
    teacher_value_loss = teacher_value_coef * teacher_value_cross_entropy
    loss = (
        policy_loss
        + vf_coef * value_loss
        - ent_coef * entropy_mean
        + teacher_kl_loss
        + teacher_value_loss
    )

    return (
        _PPOLossMetrics(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=-ent_coef * entropy_mean,
            teacher_kl_loss=teacher_kl_loss,
            teacher_value_loss=teacher_value_loss,
            entropy=entropy_mean,
            teacher_kl=teacher_kl,
            teacher_value_cross_entropy=teacher_value_cross_entropy,
            approx_kl=weighted_mean(approx_kl_values, policy_weight),
            clipfrac=weighted_mean(clipfrac_values, policy_weight),
            ratio_mean=weighted_mean(ratio, policy_weight),
            ratio_max=_masked_max_or_zero(ratio, policy_weight > 0),
            logratio_mean=weighted_mean(logratio, policy_weight),
            logratio_abs_max=_masked_max_or_zero(logratio.abs(), policy_weight > 0),
        ),
        loss,
    )


def _distributed_ppo_loss_metrics(
    policy_loss_values: torch.Tensor,
    value_loss_values: torch.Tensor,
    entropy_values: torch.Tensor,
    approx_kl_values: torch.Tensor,
    clipfrac_values: torch.Tensor,
    ratio: torch.Tensor,
    logratio: torch.Tensor,
    *,
    policy_weight: torch.Tensor,
    value_weight: torch.Tensor,
    teacher_kl_values: torch.Tensor,
    teacher_value_loss_values: torch.Tensor,
    teacher_kl_coef: float,
    teacher_value_coef: float,
    vf_coef: float,
    ent_coef: float,
    context: DistributedContext,
) -> tuple[_PPOLossMetrics, torch.Tensor]:
    teacher_value_weight = _teacher_value_state_weight(
        value_weight,
        dtype=teacher_value_loss_values.dtype,
    )
    if teacher_value_loss_values.shape != teacher_value_weight.shape:
        raise ValueError(
            "teacher value loss values must have shape "
            f"{tuple(teacher_value_weight.shape)}, "
            f"got {tuple(teacher_value_loss_values.shape)}"
        )
    policy_denominator_local = policy_weight.sum().to(dtype=policy_loss_values.dtype)
    value_denominator_local = value_weight.sum().to(dtype=value_loss_values.dtype)
    teacher_value_denominator_local = teacher_value_weight.sum()
    reduced_sums = all_reduce_sum(
        torch.stack(
            [
                (policy_loss_values.detach() * policy_weight).sum(),
                (value_loss_values.detach() * value_weight).sum(),
                (entropy_values.detach() * policy_weight).sum(),
                (teacher_kl_values.detach() * policy_weight).sum(),
                (teacher_value_loss_values.detach() * teacher_value_weight).sum(),
                (approx_kl_values.detach() * policy_weight).sum(),
                (clipfrac_values.detach() * policy_weight).sum(),
                (ratio.detach() * policy_weight).sum(),
                (logratio.detach() * policy_weight).sum(),
                policy_denominator_local,
                value_denominator_local,
                teacher_value_denominator_local,
            ]
        ),
        context,
    )
    policy_denominator = reduced_sums[9].clamp_min(1e-8)
    value_denominator = reduced_sums[10].clamp_min(1e-8)
    teacher_value_denominator = reduced_sums[11].clamp_min(1e-8)
    policy_loss = reduced_sums[0] / policy_denominator
    value_loss = reduced_sums[1] / value_denominator
    entropy_mean = reduced_sums[2] / policy_denominator
    teacher_kl = reduced_sums[3] / policy_denominator
    teacher_value_cross_entropy = reduced_sums[4] / teacher_value_denominator
    approx_kl = reduced_sums[5] / policy_denominator
    clipfrac = reduced_sums[6] / policy_denominator
    ratio_mean = reduced_sums[7] / policy_denominator
    logratio_mean = reduced_sums[8] / policy_denominator
    reduced_maxes = all_reduce_max(
        torch.stack(
            [
                _masked_max_or_zero(ratio.detach(), policy_weight > 0),
                _masked_max_or_zero(logratio.detach().abs(), policy_weight > 0),
            ]
        ),
        context,
    )
    teacher_kl_loss = teacher_kl_coef * teacher_kl
    teacher_value_loss = teacher_value_coef * teacher_value_cross_entropy
    loss = (
        policy_loss
        + vf_coef * value_loss
        - ent_coef * entropy_mean
        + teacher_kl_loss
        + teacher_value_loss
    )
    backward_policy_loss = (
        (policy_loss_values * policy_weight).sum()
        * context.world_size
        / policy_denominator
    )
    backward_value_loss = (
        (value_loss_values * value_weight).sum()
        * context.world_size
        / value_denominator
    )
    backward_entropy = (
        (entropy_values * policy_weight).sum() * context.world_size / policy_denominator
    )
    backward_teacher_kl = (
        (teacher_kl_values * policy_weight).sum()
        * context.world_size
        / policy_denominator
    )
    backward_teacher_value = (
        (teacher_value_loss_values * teacher_value_weight).sum()
        * context.world_size
        / teacher_value_denominator
    )
    backward_loss = (
        backward_policy_loss
        + vf_coef * backward_value_loss
        - ent_coef * backward_entropy
        + teacher_kl_coef * backward_teacher_kl
        + teacher_value_coef * backward_teacher_value
    )

    return (
        _PPOLossMetrics(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=-ent_coef * entropy_mean,
            teacher_kl_loss=teacher_kl_loss,
            teacher_value_loss=teacher_value_loss,
            entropy=entropy_mean,
            teacher_kl=teacher_kl,
            teacher_value_cross_entropy=teacher_value_cross_entropy,
            approx_kl=approx_kl,
            clipfrac=clipfrac,
            ratio_mean=ratio_mean,
            ratio_max=reduced_maxes[0],
            logratio_mean=logratio_mean,
            logratio_abs_max=reduced_maxes[1],
        ),
        backward_loss,
    )


def _ppo_loss_components(
    new_logp: torch.Tensor,
    entropy: torch.Tensor,
    new_values: torch.Tensor,
    old_logp: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
    vf_clip_coef: float | None,
    ppo_clip_mode: PPOClipMode,
    entity_policy_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    if ppo_clip_mode == "per_entity":
        if entity_policy_weight is None:
            raise ValueError(
                "entity_policy_weight is required for per_entity PPO clipping"
            )
        entity_weight = entity_policy_weight
        policy_components = _per_entity_policy_loss_components(
            new_logp,
            old_logp,
            advantages,
            clip_coef,
            entity_weight,
        )
    else:
        entity_weight = None
        policy_components = _per_player_policy_loss_components(
            new_logp,
            old_logp,
            advantages,
            clip_coef,
        )
    (
        policy_loss_values,
        approx_kl_values,
        clipfrac_values,
        ratio,
        logratio,
    ) = policy_components

    if entity_weight is not None:
        entropy = _sum_masked_entities(entropy, entity_weight)

    if vf_clip_coef:
        value_clipped = old_values + torch.clamp(
            new_values - old_values,
            -vf_clip_coef,
            vf_clip_coef,
        )
        value_loss_unclipped = (new_values - returns).pow(2)
        value_loss_clipped = (value_clipped - returns).pow(2)
        value_loss_values = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)
    else:
        value_loss_values = 0.5 * (new_values - returns).pow(2)

    return (
        policy_loss_values,
        value_loss_values,
        entropy,
        approx_kl_values,
        clipfrac_values,
        ratio,
        logratio,
    )


def _per_player_policy_loss_components(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, ...]:
    logratio = new_logp - old_logp
    ratio = logratio.exp()
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss_values = torch.max(pg_loss1, pg_loss2)

    return (
        policy_loss_values,
        (ratio - 1.0) - logratio,
        ((ratio - 1.0).abs() > clip_coef).float(),
        ratio,
        logratio,
    )


def _per_entity_policy_loss_components(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
    entity_policy_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    entity_advantages = advantages.unsqueeze(-1)
    logratio = new_logp - old_logp
    ratio = logratio.exp()
    pg_loss1 = -entity_advantages * ratio
    pg_loss2 = -entity_advantages * torch.clamp(
        ratio,
        1.0 - clip_coef,
        1.0 + clip_coef,
    )
    entity_policy_loss = torch.max(pg_loss1, pg_loss2)
    entity_approx_kl = (ratio - 1.0) - logratio
    entity_clipfrac = ((ratio - 1.0).abs() > clip_coef).float()

    return (
        _sum_masked_entities(entity_policy_loss, entity_policy_weight),
        _sum_masked_entities(entity_approx_kl, entity_policy_weight),
        _mean_masked_entities(entity_clipfrac, entity_policy_weight),
        _mean_masked_entities(ratio, entity_policy_weight),
        _mean_masked_entities(logratio, entity_policy_weight),
    )


def _copy_action_mask_time_step(dst: ActionMask, step: int, src: ActionMask) -> None:
    if type(dst) is not type(src):
        raise ValueError(
            f"rollout action-mask type mismatch: expected {type(dst).__name__}, "
            f"got {type(src).__name__}"
        )
    dst.can_act[step].copy_(src.can_act)
    if isinstance(dst, PureActionMask | DiscreteTargetActionMask):
        src_with_max_launch = cast(PureActionMask | DiscreteTargetActionMask, src)
        dst.max_launch[step].copy_(src_with_max_launch.max_launch)


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
        src_with_max_launch = cast(PureActionMask | DiscreteTargetActionMask, src)
        dst.max_launch.copy_(src_with_max_launch.max_launch, non_blocking=non_blocking)


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


def _map_optional_obs_tensors(
    obs: ObsBatch,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor | None]:
    return {
        field: None if (tensor := getattr(obs, field)) is None else fn(tensor)
        for field in _OBS_OPTIONAL_TENSOR_FIELDS
    }


def _copy_obs_time_step(dst: ObsBatch, step: int, src: ObsBatch) -> None:
    for field in _OBS_TENSOR_FIELDS:
        dst_tensor = getattr(dst, field)
        src_tensor = getattr(src, field)
        dst_tensor[step].copy_(src_tensor)
    for field in _OBS_OPTIONAL_TENSOR_FIELDS:
        dst_tensor = getattr(dst, field)
        src_tensor = getattr(src, field)
        if dst_tensor is None:
            if src_tensor is not None:
                raise ValueError(f"rollout obs has no {field} buffer")
        elif src_tensor is None:
            raise ValueError(f"source obs is missing {field}")
        else:
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
        **_map_optional_obs_tensors(
            obs,
            lambda tensor: tensor.transpose(0, 1).contiguous(),
        ),
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
        **_map_optional_obs_tensors(obs, lambda tensor: tensor[idx]),
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
            **_map_optional_obs_tensors(
                obs,
                lambda tensor: tensor.to(device, non_blocking=non_blocking).clone(),
            ),
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
        **_map_optional_obs_tensors(
            obs,
            lambda tensor: tensor.to(device, non_blocking=non_blocking),
        ),
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
    for field in _OBS_OPTIONAL_TENSOR_FIELDS:
        dst_tensor = getattr(dst, field)
        src_tensor = getattr(src, field)
        if dst_tensor is None:
            if src_tensor is not None:
                raise ValueError(f"destination obs has no {field} buffer")
        elif src_tensor is None:
            raise ValueError(f"source obs is missing {field}")
        else:
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
        **_map_optional_obs_tensors(obs, _flatten_tensor_time),
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
    _add_neutral_undershot_rates(logged, metrics, prefix="train/")
    for key, values in metrics.items():
        if key.startswith("_") or key in _NEUTRAL_UNDERSHOT_RATE_KEYS:
            continue
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
    metric_totals = {
        key: (float(total[0].item()), float(total[1].item()))
        for key, total in zip(keys, totals, strict=True)
    }
    _add_neutral_undershot_rates_from_totals(logged, metric_totals, prefix="train/")
    for key, total in zip(keys, totals, strict=True):
        if key.startswith("_") or key in _NEUTRAL_UNDERSHOT_RATE_KEYS:
            continue
        if total[1].item() == 0:
            continue
        if key == "total_games_played":
            logged[f"train/{key}"] = float(total[0].item())
        else:
            logged[f"train/{key}"] = float((total[0] / total[1]).item())
    return logged


_NEUTRAL_UNDERSHOT_RATE_KEYS = {
    "neutral_planet_undershot_rate",
    "neutral_comet_undershot_rate",
}

_NEUTRAL_UNDERSHOT_RATE_INPUTS = {
    "neutral_planet_undershot_rate": (
        "_neutral_planet_undershots_per_game",
        "_neutral_planets_captured_per_game",
    ),
    "neutral_comet_undershot_rate": (
        "_neutral_comet_undershots_per_game",
        "_neutral_comets_captured_per_game",
    ),
}


def _add_neutral_undershot_rates(
    logged: dict[str, float],
    metrics: dict[str, list[float]],
    *,
    prefix: str,
) -> None:
    totals = {
        key: (float(sum(values)), float(len(values))) for key, values in metrics.items()
    }
    _add_neutral_undershot_rates_from_totals(logged, totals, prefix=prefix)


def _add_neutral_undershot_rates_from_totals(
    logged: dict[str, float],
    totals: dict[str, tuple[float, float]],
    *,
    prefix: str,
) -> None:
    for rate_key, (
        undershot_key,
        captured_key,
    ) in _NEUTRAL_UNDERSHOT_RATE_INPUTS.items():
        undershots = totals.get(undershot_key, (0.0, 0.0))[0]
        captures = totals.get(captured_key, (0.0, 0.0))[0]
        denominator = undershots + captures
        if denominator > 0.0:
            logged[f"{prefix}{rate_key}"] = max(0.0, min(1.0, undershots / denominator))


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


def _checkpoint_metadata(checkpoint: dict[object, object]) -> PPOCheckpointMetadata:
    return PPOCheckpointMetadata(
        env_steps=_checkpoint_nonnegative_int(
            checkpoint["env_steps"],
            name="env_steps",
        ),
        player_step_total=_checkpoint_nonnegative_int(
            checkpoint["player_step_total"],
            name="player_step_total",
        ),
        total_games_played=_checkpoint_nonnegative_int(
            checkpoint["total_games_played"],
            name="total_games_played",
        ),
        wandb_run_id=_checkpoint_optional_str(
            checkpoint["wandb_run_id"],
            name="wandb_run_id",
        ),
    )


def _checkpoint_optional_str(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"checkpoint {name} must be a non-empty string or None")
    return value


def _require_stateless_teacher(
    teacher: BaseModelAPI,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    hidden_state = teacher.initial_hidden_state(batch_size, device=device)
    if hidden_state is not None:
        raise ValueError("teacher models with recurrent hidden state are not supported")


def _validate_fixed_teacher_action_compatibility(
    student: BaseModelAPI,
    teacher: BaseModelAPI,
) -> None:
    if not isinstance(student, StatelessTransformerV1) or not isinstance(
        teacher,
        StatelessTransformerV1,
    ):
        return
    student_actor_config = student.config.actor
    teacher_actor_config = teacher.config.actor
    if not isinstance(
        student_actor_config,
        ActorDiscreteTargetsConfig,
    ) or not isinstance(teacher_actor_config, ActorDiscreteTargetsConfig):
        return
    if student_actor_config.launch_mode != teacher_actor_config.launch_mode:
        raise ValueError(
            "fixed teacher discrete-target launch_mode must match student launch_mode"
        )


def _model_forward(
    model: BaseModelAPI,
    obs: ObsBatch,
    *,
    hidden_state: ModelHiddenState | None,
) -> ModelOutput:
    if hidden_state is None:
        return model(obs)
    return model(obs, hidden_state=hidden_state)


def _model_compute_value(
    model: BaseModelAPI,
    obs: ObsBatch,
    *,
    hidden_state: ModelHiddenState | None,
) -> torch.Tensor:
    if hidden_state is None:
        return model.compute_value(obs)
    return model.compute_value(obs, hidden_state=hidden_state)


def _model_evaluate_actions(
    model: BaseModelAPI,
    obs: ObsBatch,
    actions: ModelActions,
    *,
    hidden_state: ModelHiddenState | None,
    dones: torch.Tensor,
) -> ModelEvaluation:
    if hidden_state is None:
        return model.evaluate_actions(obs, actions)
    return model.evaluate_actions(obs, actions, hidden_state=hidden_state, dones=dones)


def _model_evaluate_actions_with_teacher(
    model: BaseModelAPI,
    obs: ObsBatch,
    actions: ModelActions,
    teacher: BaseModelAPI,
    *,
    hidden_state: ModelHiddenState | None,
    dones: torch.Tensor,
    compute_teacher_action_kl: bool,
    compute_teacher_value: bool,
) -> ModelTeacherEvaluation:
    return model.evaluate_actions_with_teacher(
        obs,
        actions,
        teacher,
        hidden_state=hidden_state,
        dones=dones,
        compute_teacher_action_kl=compute_teacher_action_kl,
        compute_teacher_value=compute_teacher_value,
    )


def _model_evaluate_actions_with_cached_teacher(
    model: BaseModelAPI,
    obs: ObsBatch,
    actions: ModelActions,
    teacher_targets: CachedTeacherDistillationTargets,
    *,
    hidden_state: ModelHiddenState | None,
    dones: torch.Tensor,
    compute_teacher_action_kl: bool,
    compute_teacher_value: bool,
) -> ModelTeacherEvaluation:
    return model.evaluate_actions_with_cached_teacher(
        obs,
        actions,
        teacher_targets,
        hidden_state=hidden_state,
        dones=dones,
        compute_teacher_action_kl=compute_teacher_action_kl,
        compute_teacher_value=compute_teacher_value,
    )


def _teacher_value_cross_entropy(
    student_winner_log_probabilities: torch.Tensor,
    teacher_winner_probabilities: torch.Tensor,
) -> torch.Tensor:
    return (
        -teacher_winner_probabilities.detach() * student_winner_log_probabilities
    ).sum(dim=-1)


def _output_actions(output: ModelOutput) -> ModelActions:
    return output.actions


def _output_logp(output: ModelOutput | ModelEvaluation) -> torch.Tensor:
    return output.log_probs.per_player_entity.sum(dim=-1)


def _output_entity_logp(output: ModelOutput | ModelEvaluation) -> torch.Tensor:
    return output.log_probs.per_player_entity


def _old_policy_logp_for_clip_mode(
    segments: _PPORolloutSegments,
    idx: torch.Tensor,
    ppo_clip_mode: PPOClipMode,
) -> torch.Tensor:
    if ppo_clip_mode == "per_entity":
        if segments.entity_logp is None:
            raise RuntimeError(
                "per_entity PPO clipping requires stored entity log-probs"
            )
        return segments.entity_logp[idx]
    return segments.logp[idx]


def _output_logp_for_clip_mode(
    output: ModelOutput | ModelEvaluation,
    ppo_clip_mode: PPOClipMode,
) -> torch.Tensor:
    if ppo_clip_mode == "per_entity":
        return _output_entity_logp(output)
    return _output_logp(output)


def _output_entropy(
    output: ModelOutput | ModelEvaluation, like: torch.Tensor
) -> torch.Tensor:
    return output.entropies.per_player_entity.sum(dim=-1).view_as(like)


def _output_entity_entropy(output: ModelOutput | ModelEvaluation) -> torch.Tensor:
    return output.entropies.per_player_entity


def _output_entropy_for_clip_mode(
    output: ModelOutput | ModelEvaluation,
    like: torch.Tensor,
    ppo_clip_mode: PPOClipMode,
) -> torch.Tensor:
    if ppo_clip_mode == "per_entity":
        return _output_entity_entropy(output).view_as(like)
    return _output_entropy(output, like)


def _output_action_kl(
    kl: ModelActionKLDivergences,
    like: torch.Tensor,
) -> torch.Tensor:
    return kl.per_player_entity.sum(dim=-1).view_as(like)


def _output_entity_action_kl(kl: ModelActionKLDivergences) -> torch.Tensor:
    return kl.per_player_entity


def _output_action_kl_for_clip_mode(
    kl: ModelActionKLDivergences,
    like: torch.Tensor,
    ppo_clip_mode: PPOClipMode,
) -> torch.Tensor:
    if ppo_clip_mode == "per_entity":
        return _output_entity_action_kl(kl).view_as(like)
    return _output_action_kl(kl, like)


def _output_entropy_components(
    output: ModelOutput | ModelEvaluation,
    like: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        name: _sum_entropy_component(component, like)
        for name, component in output.entropies.components.items()
    }


def _sum_entropy_component(tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if tensor.shape == like.shape:
        return tensor.view_as(like)
    if tensor.shape[: like.ndim] == like.shape:
        return tensor.flatten(start_dim=like.ndim).sum(dim=-1).view_as(like)
    return tensor.flatten(start_dim=2).sum(dim=-1).view_as(like)


def _output_action_kl_components(
    kl: ModelActionKLDivergences,
    like: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        name: _sum_entropy_component(component, like)
        for name, component in kl.components.items()
    }


def _output_values(output: ModelOutput | ModelEvaluation) -> torch.Tensor:
    return output.values


def _policy_mask(obs: ObsBatch) -> torch.Tensor:
    can_act = obs.action_mask.can_act.flatten(start_dim=3).any(dim=-1)
    return obs.still_playing & can_act


def _player_count_rates(still_playing: torch.Tensor) -> dict[str, torch.Tensor]:
    alive_counts = still_playing.sum(dim=-1)
    return {
        f"train/{player_count}p_rate": alive_counts.eq(player_count).float().mean()
        for player_count in range(1, OUTER_PLAYER_SLOTS + 1)
    }


def _policy_entity_mask(obs: ObsBatch) -> torch.Tensor:
    can_act = obs.action_mask.can_act
    if can_act.ndim == obs.still_playing.ndim + 1:
        source_can_act = can_act
    else:
        source_can_act = can_act.flatten(start_dim=4).any(dim=-1)
    return obs.still_playing.unsqueeze(-1) & source_can_act


def _minibatch_indices(
    *,
    config: PPOConfig,
    n_segments: int,
    device: torch.device,
) -> list[torch.Tensor]:
    _validate_minibatch_divisibility(n_segments, config)
    samples: list[torch.Tensor] = []
    for _epoch in range(config.ppo_epochs):
        permutation = torch.randperm(n_segments, device=device)
        samples.extend(permutation.split(config.segments_per_minibatch))
    return samples


def _validate_minibatch_divisibility(n_envs: int, config: PPOConfig) -> None:
    divisor = config.segments_per_minibatch * config.gradient_accumulation_steps
    if n_envs % divisor != 0:
        raise ValueError(
            "n_envs must be divisible by segments_per_minibatch * "
            "gradient_accumulation_steps"
        )


def _masked_max_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(mask, values, torch.full_like(values, -torch.inf))
    return torch.where(
        mask.any(),
        masked.max(),
        torch.zeros((), dtype=values.dtype, device=values.device),
    )


def _sum_masked_entities(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum(dim=-1)


def _mean_masked_entities(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weighted_sum = (values * weights).sum(dim=-1)
    return weighted_sum / weights.sum(dim=-1).clamp_min(1e-8)


def _teacher_value_weighted_mean(
    values: torch.Tensor,
    value_weight: torch.Tensor,
) -> torch.Tensor:
    state_weight = _teacher_value_state_weight(value_weight, dtype=values.dtype)
    if values.shape != state_weight.shape:
        raise ValueError(
            "teacher value loss values must have shape "
            f"{tuple(state_weight.shape)}, got {tuple(values.shape)}"
        )
    return weighted_mean(values, state_weight)


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


def _distributed_teacher_value_weighted_mean(
    values: torch.Tensor,
    value_weight: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    state_weight = _teacher_value_state_weight(value_weight, dtype=values.dtype)
    if values.shape != state_weight.shape:
        raise ValueError(
            "teacher value loss values must have shape "
            f"{tuple(state_weight.shape)}, got {tuple(values.shape)}"
        )
    return _distributed_weighted_mean(values, state_weight, context)


def _distributed_backward_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    local_numerator = (values * weights).sum()
    global_denominator = all_reduce_sum(weights.sum().to(dtype=values.dtype), context)
    return local_numerator * context.world_size / global_denominator.clamp_min(1e-8)


def _distributed_backward_teacher_value_weighted_mean(
    values: torch.Tensor,
    value_weight: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    state_weight = _teacher_value_state_weight(value_weight, dtype=values.dtype)
    if values.shape != state_weight.shape:
        raise ValueError(
            "teacher value loss values must have shape "
            f"{tuple(state_weight.shape)}, got {tuple(values.shape)}"
        )
    return _distributed_backward_weighted_mean(values, state_weight, context)


def _teacher_value_state_weight(
    value_weight: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return value_weight.gt(0).any(dim=-1).to(dtype=dtype)


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
    *,
    context: DistributedContext | None = None,
) -> torch.Tensor:
    require_same_shape(advantages, mask, left_name="advantages", right_name="mask")
    mask_float = mask.to(dtype=advantages.dtype)
    if context is not None and context.initialized:
        totals = all_reduce_sum(
            torch.stack(
                (
                    (advantages * mask_float).sum(),
                    (advantages.square() * mask_float).sum(),
                    mask_float.sum(),
                )
            ),
            context,
        )
        denom = totals[2].clamp_min(1.0)
        mean = totals[0] / denom
        var = totals[1] / denom - mean.pow(2)
    else:
        denom = mask_float.sum().clamp_min(1.0)
        mean = (advantages * mask_float).sum() / denom
        var = ((advantages - mean).pow(2) * mask_float).sum() / denom

    return (advantages - mean) / (var.clamp_min(0.0).sqrt() + eps)


def _mean_loss_metrics(metrics: list[_PPOLossMetrics]) -> dict[str, float]:
    metric_names = (
        ("loss/total_loss", "loss"),
        ("loss/policy_loss", "policy_loss"),
        ("loss/value_loss", "value_loss"),
        ("loss/entropy_loss", "entropy_loss"),
        ("loss/teacher_kl_loss", "teacher_kl_loss"),
        ("loss/teacher_value_loss", "teacher_value_loss"),
        ("policy/entropy", "entropy"),
        ("teacher/kl", "teacher_kl"),
        ("teacher/value_cross_entropy", "teacher_value_cross_entropy"),
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
    for name in _teacher_kl_component_names(metrics):
        logged[f"teacher/{name}_kl"] = float(
            torch.stack([metric.teacher_kl_components[name] for metric in metrics])
            .mean()
            .item()
        )
    return logged


def _detach_loss_metrics(metrics: _PPOLossMetrics) -> _PPOLossMetrics:
    return replace(
        metrics,
        loss=metrics.loss.detach(),
        policy_loss=metrics.policy_loss.detach(),
        value_loss=metrics.value_loss.detach(),
        entropy_loss=metrics.entropy_loss.detach(),
        teacher_kl_loss=metrics.teacher_kl_loss.detach(),
        teacher_value_loss=metrics.teacher_value_loss.detach(),
        entropy=metrics.entropy.detach(),
        teacher_kl=metrics.teacher_kl.detach(),
        teacher_value_cross_entropy=metrics.teacher_value_cross_entropy.detach(),
        approx_kl=metrics.approx_kl.detach(),
        clipfrac=metrics.clipfrac.detach(),
        ratio_mean=metrics.ratio_mean.detach(),
        ratio_max=metrics.ratio_max.detach(),
        logratio_mean=metrics.logratio_mean.detach(),
        logratio_abs_max=metrics.logratio_abs_max.detach(),
        entropy_components={
            name: value.detach() for name, value in metrics.entropy_components.items()
        },
        teacher_kl_components={
            name: value.detach()
            for name, value in metrics.teacher_kl_components.items()
        },
    )


def _entropy_component_names(metrics: list[_PPOLossMetrics]) -> tuple[str, ...]:
    if not metrics:
        return ()
    names = set(metrics[0].entropy_components)
    for metric in metrics[1:]:
        names &= set(metric.entropy_components)
    return tuple(sorted(names))


def _teacher_kl_component_names(metrics: list[_PPOLossMetrics]) -> tuple[str, ...]:
    if not metrics:
        return ()
    names = set(metrics[0].teacher_kl_components)
    for metric in metrics[1:]:
        names &= set(metric.teacher_kl_components)
    return tuple(sorted(names))
