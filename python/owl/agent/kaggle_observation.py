from pydantic import BaseModel, ConfigDict, Field


class Planet(BaseModel): ...


class Fleet(BaseModel): ...


class Comet(BaseModel): ...


class KaggleObservation(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    remaining_overage_time: float = Field(alias="remainingOverageTime")
    step: int = Field(ge=0, lt=500)
    planets: list[Planet]
    fleets: list[Fleet]
    player: int = Field(ge=0, lt=4)
    angular_velocity: float
    comets: list[Comet]
