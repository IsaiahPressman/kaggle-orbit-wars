from typing import Self

from pydantic import model_validator

from owl.config import BaseConfig
from owl.model import ModelConfig
from owl.rl import EnvConfig

from .optimizer import OptimizerConfig
from .ppo import PPOConfig


class FullConfig(BaseConfig):
    env: EnvConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    rl: PPOConfig

    @classmethod
    def subconfig_dirs(cls) -> set[str]:
        return {"env", "model", "optimizer", "rl"}

    @model_validator(mode="after")
    def _validate_model_action_spec_matches_env(self) -> Self:
        env_max_launches = self.env.action_spec.max_per_planet_launches
        model_max_launches = self.model.action_spec.max_per_planet_launches
        if model_max_launches != env_max_launches:
            raise ValueError(
                "model.action_spec.max_per_planet_launches must match "
                "env.action_spec.max_per_planet_launches "
                f"({model_max_launches} != {env_max_launches})"
            )
        return self
