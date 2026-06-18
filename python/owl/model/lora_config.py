from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from owl.config import BaseConfig

LoRATargetModule = Literal["q", "k", "v", "out", "up", "down", "gate", "value"]


class LoRAConfig(BaseConfig):
    rank: int = Field(ge=1)
    alpha: float | None = Field(default=None, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: tuple[LoRATargetModule, ...] = ("q", "v")
    target_block_count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_target_modules(self) -> Self:
        if not self.target_modules:
            raise ValueError("lora.target_modules must not be empty")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("lora.target_modules must not contain duplicates")
        if self.dropout != 0.0:
            raise ValueError("lora.dropout must be 0.0 for PPO fine-tuning")
        return self

    @property
    def scaling_alpha(self) -> float:
        return float(self.rank if self.alpha is None else self.alpha)
