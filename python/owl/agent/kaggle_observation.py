from typing import Annotated, Any

from pydantic import BaseModel, Field

KAGGLE_EPISODE_STEPS = 500

Planet = Annotated[
    tuple[int, int, float, float, float, int, int],
    Field(min_length=7, max_length=7),
]
Fleet = Annotated[
    tuple[int, int, float, float, float, int, int],
    Field(min_length=7, max_length=7),
]
Point = Annotated[tuple[float, float], Field(min_length=2, max_length=2)]

PLANET_ID_INDEX = 0
PLANET_OWNER_INDEX = 1
PLANET_X_INDEX = 2
PLANET_Y_INDEX = 3
PLANET_RADIUS_INDEX = 4
PLANET_SHIPS_INDEX = 5
PLANET_PRODUCTION_INDEX = 6

FLEET_ID_INDEX = 0
FLEET_OWNER_INDEX = 1
FLEET_X_INDEX = 2
FLEET_Y_INDEX = 3
FLEET_ANGLE_INDEX = 4
FLEET_FROM_PLANET_ID_INDEX = 5
FLEET_SHIPS_INDEX = 6


class Comet(BaseModel):
    planet_ids: list[int]
    paths: list[list[Point]]
    path_index: int


class KaggleObservation(BaseModel):
    """Strict parser for observations returned by the Kaggle Orbit Wars engine.

    Verified against the installed kaggle_environments source:
    fleet rows are only created after from_planet_id matches an existing planet,
    ship counts are cast with int(...) before storage, and non-finite launch
    angles either error during action processing or produce non-finite positions
    that are removed by the engine's out-of-bounds check before observations are
    returned. That means malformed opponent actions should not require lossy
    row-level repair here.
    """

    remaining_overage_time: float = Field(alias="remainingOverageTime")
    step: int = Field(ge=0, lt=KAGGLE_EPISODE_STEPS)
    planets: list[Planet]
    initial_planets: list[Planet]
    fleets: list[Fleet]
    player: int = Field(ge=0, lt=4)
    angular_velocity: float
    comet_planet_ids: list[int]
    next_fleet_id: int
    comets: list[Comet]

    def to_rl_observation(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "episode_steps": KAGGLE_EPISODE_STEPS,
            "planets": self.planets,
            "initial_planets": self.initial_planets,
            "fleets": self.fleets,
            "player": self.player,
            "angular_velocity": self.angular_velocity,
            "comet_planet_ids": self.comet_planet_ids,
            "next_fleet_id": self.next_fleet_id,
            "comets": [comet.model_dump(mode="json") for comet in self.comets],
        }
