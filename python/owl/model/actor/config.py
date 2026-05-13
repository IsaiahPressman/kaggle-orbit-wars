from __future__ import annotations

from typing import Annotated, Literal, Self, TypeAlias

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
    max_ship_normalizer: float = Field(default=500.0, gt=0.0)
    entropy_ship_support_cap: int = Field(default=256, ge=1)

    @model_validator(mode="after")
    def _validate_kappa_bounds(self) -> Self:
        if self.kappa_max is not None and self.kappa_min > self.kappa_max:
            raise ValueError("kappa_min must be <= kappa_max")
        return self


class ActorDiscreteTargetsConfig(BaseConfig):
    action_spec: Literal["discrete_targets"] = "discrete_targets"
    n_action_mixtures: int = Field(default=4, ge=1)
    entropy_ship_quantiles: int = Field(default=16, ge=1)
    scale_min: float = Field(default=0.10, gt=0.0)
    scale_max_frac: float = Field(default=0.50, gt=0.0)
    scale_max_abs_floor: float = Field(default=8.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_scale_bounds(self) -> Self:
        if self.scale_min > self.scale_max_abs_floor:
            raise ValueError("scale_min must be <= scale_max_abs_floor")
        return self


class ActorDiscreteTargetBinsConfig(BaseConfig):
    action_spec: Literal["discrete_target_bins"] = "discrete_target_bins"
    n_bins: int = Field(ge=2)


ActorConfig: TypeAlias = Annotated[
    ActorPureConfig | ActorDiscreteTargetsConfig | ActorDiscreteTargetBinsConfig,
    Field(discriminator="action_spec"),
]
