from owl.rs import assert_release_build

from .kaggle_observation import KaggleObservation


class Agent:
    def __init__(self) -> None:
        assert_release_build()

    def act(self, observation: KaggleObservation) -> list[list[float]]:
        raise NotImplementedError
