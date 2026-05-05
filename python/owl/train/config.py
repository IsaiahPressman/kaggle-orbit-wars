from typing import Self

from pydantic import Field, model_validator

from owl.config import BaseConfig
from owl.model import ModelConfig
from owl.rl import EnvConfig

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
    def _validate_model_env_action_spec(self) -> Self:
        if self.model.actor.action_spec != self.env.action_spec.action_spec:
            raise ValueError("model actor action_spec must match env action_spec")
        return self

    @classmethod
    def subconfig_dirs(cls) -> set[str]:
        return {"env", "model", "optimizer", "rl"}
