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
