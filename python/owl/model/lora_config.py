from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from owl.config import BaseConfig

LoRATargetModule = Literal["q", "k", "v", "out", "up", "down", "gate", "value"]


class LoRAConfig(BaseConfig):
    rank: int = Field(ge=1)
    alpha: float | None = Field(default=None, gt=0.0)
    target_modules: tuple[LoRATargetModule, ...] = ("q", "v")
    target_block_count: int | None = Field(default=None, ge=1)
    target_value_head: bool = False
    target_policy_head: bool = False

    @model_validator(mode="after")
    def _validate_targets(self) -> Self:
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("lora.target_modules must not contain duplicates")
        if (
            not self.target_modules
            and not self.target_value_head
            and not self.target_policy_head
        ):
            raise ValueError(
                "lora must target at least one of target_modules, "
                "target_value_head, or target_policy_head"
            )
        return self

    @property
    def scaling_alpha(self) -> float:
        return float(self.rank if self.alpha is None else self.alpha)
