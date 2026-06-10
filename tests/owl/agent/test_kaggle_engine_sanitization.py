import math
import warnings

import pytest
from owl.agent import KaggleObservation


def _make_orbit_wars_env(seed: int = 11):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Using extra keyword arguments on `Field` is deprecated.*",
            category=DeprecationWarning,
        )
        from kaggle_environments import make

        env = make(
            "orbit_wars", debug=True, configuration={"episodeSteps": 20, "seed": seed}
        )
    env.reset(2)
    return env


def _owned_planet(env):
    return next(
        planet
        for planet in env.state[0].observation.planets
        if planet[1] == 0 and planet[5] >= 6
    )


def _center_angle(planet) -> float:
    return math.atan2(50.0 - planet[3], 50.0 - planet[2])


def test_kaggle_engine_ignores_non_integral_source_planet_ids() -> None:
    env = _make_orbit_wars_env()
    source = _owned_planet(env)

    env.step([[[source[0] + 0.5, _center_angle(source), 5.0]], []])

    observation = KaggleObservation.model_validate(env.state[0].observation)
    assert observation.fleets == []


def test_kaggle_engine_casts_ship_counts_before_returning_fleets() -> None:
    env = _make_orbit_wars_env()
    source = _owned_planet(env)

    env.step([[[float(source[0]), _center_angle(source), 5.9]], []])

    raw_fleets = env.state[0].observation.fleets
    assert len(raw_fleets) == 1
    assert raw_fleets[0][5] == float(source[0])
    assert type(raw_fleets[0][5]) is float
    assert raw_fleets[0][6] == 5
    assert type(raw_fleets[0][6]) is int
    observation = KaggleObservation.model_validate(env.state[0].observation)
    assert observation.fleets[0][5] == source[0]
    assert observation.fleets[0][6] == 5


def test_kaggle_engine_removes_nan_angle_fleets_before_returning_observation() -> None:
    env = _make_orbit_wars_env()
    source = _owned_planet(env)

    env.step([[[float(source[0]), float("nan"), 5.0]], []])

    observation = KaggleObservation.model_validate(env.state[0].observation)
    assert observation.fleets == []


def test_kaggle_engine_rejects_infinite_angle_before_returning_observation() -> None:
    env = _make_orbit_wars_env()
    source = _owned_planet(env)

    with pytest.raises(ValueError, match="math domain error"):
        env.step([[[float(source[0]), float("inf"), 5.0]], []])
