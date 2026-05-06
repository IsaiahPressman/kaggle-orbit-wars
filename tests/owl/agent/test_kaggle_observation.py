import pytest
from owl.agent import KaggleObservation
from owl.agent.kaggle_observation import KAGGLE_EPISODE_STEPS
from pydantic import ValidationError


def _raw_observation() -> dict[str, object]:
    planet = [0, 0, 25.0, 50.0, 2.0, 10, 3]
    return {
        "remainingOverageTime": 60.0,
        "step": 0,
        "planets": [planet],
        "initial_planets": [planet],
        "fleets": [],
        "player": 0,
        "angular_velocity": 0.025,
        "comet_planet_ids": [],
        "next_fleet_id": 0,
        "comets": [],
        "extra": "ignored",
    }


def test_kaggle_observation_preserves_row_entities_and_ignores_extra_keys() -> None:
    raw = _raw_observation()
    raw["comets"] = [
        {
            "planet_ids": [10],
            "paths": [[(0.0, 1.0), (2.0, 3.0)]],
            "path_index": 0,
        }
    ]
    observation = KaggleObservation.model_validate(raw)

    assert observation.planets == [(0, 0, 25.0, 50.0, 2.0, 10, 3)]
    assert observation.to_rl_observation()["episode_steps"] == KAGGLE_EPISODE_STEPS
    assert observation.to_rl_observation()["comets"] == [
        {
            "planet_ids": [10],
            "paths": [[[0.0, 1.0], [2.0, 3.0]]],
            "path_index": 0,
        }
    ]


def test_kaggle_observation_rejects_missing_required_keys() -> None:
    raw = _raw_observation()
    del raw["planets"]

    with pytest.raises(ValidationError, match="planets"):
        KaggleObservation.model_validate(raw)


def test_kaggle_observation_rejects_malformed_entity_rows() -> None:
    raw = _raw_observation()
    raw["planets"] = [[0, 0, 25.0]]

    with pytest.raises(ValidationError):
        KaggleObservation.model_validate(raw)
