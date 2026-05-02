from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from owl.config import BaseConfig


class ActorPureConfig(BaseConfig):
    action_spec: Literal["pure"] = "pure"
    n_action_mixtures: int = Field(default=4, ge=1)
    kappa_min: float = Field(default=1e-3, gt=0.0)
    kappa_max: float | None = Field(default=200.0, gt=0.0)
    tau_min: float = Field(default=1e-3, gt=0.0)
    alpha_beta_eps: float = Field(default=1e-4, gt=0.0)
    dir_eps: float = Field(default=1e-6, gt=0.0)
    max_ship_normalizer: float = Field(default=250.0, gt=0.0)
    entropy_ship_support_cap: int = Field(default=256, ge=1)


class ActorDiscreteTargetsConfig(BaseConfig):
    action_spec: Literal["discrete_targets"] = "discrete_targets"
    n_action_mixtures: int = Field(default=4, ge=1)
    max_ship_normalizer: float = Field(default=250.0, gt=0.0)
    entropy_ship_support_cap: int = Field(default=256, ge=1)
    scale_min: float = Field(default=0.25, gt=0.0)
    min_log_scale: float = -7.0
    max_log_scale: float = 0.5

    @model_validator(mode="after")
    def _validate_scale_clamp(self) -> Self:
        if self.min_log_scale > self.max_log_scale:
            raise ValueError("min_log_scale must be <= max_log_scale")
        return self


type ActorConfig = Annotated[
    ActorPureConfig | ActorDiscreteTargetsConfig,
    Field(discriminator="action_spec"),
]
