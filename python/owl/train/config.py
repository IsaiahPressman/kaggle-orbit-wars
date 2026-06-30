from typing import Self

from pydantic import Field, model_validator

from owl.config import BaseConfig
from owl.model import ActorDiscreteTargetBinsConfig, ModelConfig
from owl.rl import ActionDiscreteTargetBinsConfig, EnvConfig

from .optimizer import OptimizerConfig
from .ppo import PPOConfig


class RuntimeConfig(BaseConfig):
    n_runtime_gpus: int = Field(default=1, ge=1)


class FullConfig(BaseConfig):
    env: EnvConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    rl: PPOConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _validate_cross_config_constraints(self) -> Self:
        if self.model.actor.action_spec != self.env.action_spec.action_spec:
            raise ValueError("model actor action_spec must match env action_spec")
        if (
            isinstance(self.model.actor, ActorDiscreteTargetBinsConfig)
            and isinstance(self.env.action_spec, ActionDiscreteTargetBinsConfig)
            and self.model.actor.n_bins != self.env.action_spec.n_bins
        ):
            raise ValueError("model actor n_bins must match env action_spec n_bins")
        if self.env.reward_mode == "win_only" and self.model.value_mode != "win_only":
            raise ValueError(
                "env.reward_mode='win_only' requires model.value_mode='win_only'"
            )
        if self.env.reward_mode != "win_only" and self.model.value_mode == "win_only":
            raise ValueError(
                "model.value_mode='win_only' requires env.reward_mode='win_only'"
            )
        divisor = self.rl.segments_per_minibatch * self.rl.gradient_accumulation_steps
        if self.env.n_envs % divisor != 0:
            raise ValueError(
                "env.n_envs must be divisible by rl.segments_per_minibatch * "
                "rl.gradient_accumulation_steps"
            )
        if self.rl.eval_replay_games > self.env.n_envs:
            raise ValueError("rl.eval_replay_games must be <= env.n_envs")
        return self

    @classmethod
    def subconfig_dirs(cls) -> set[str]:
        return {"env", "model", "optimizer", "rl"}
