import numpy as np

class RlVecEnv:
    n_envs: int
    n_players: int
    max_planets: int
    max_entities: int
    max_fleets: int
    action_dim: int

    def __init__(
        self,
        n_envs: int,
        n_players: int,
        obs_spec: str = ...,
        max_entities: int = ...,
        action_dim: int = ...,
    ) -> None: ...
    def reset(
        self,
        planet_obs: np.ndarray,
        fleet_obs: np.ndarray,
        planet_mask: np.ndarray,
        fleet_mask: np.ndarray,
    ) -> None: ...
    def step(
        self,
        actions: np.ndarray,
        planet_obs: np.ndarray,
        fleet_obs: np.ndarray,
        planet_mask: np.ndarray,
        fleet_mask: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None: ...
    def obs_shapes(
        self,
    ) -> tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int],
        tuple[int, int],
    ]: ...

def hello_from_rust() -> str: ...
def hello_numpy() -> np.ndarray: ...
def assert_release_build() -> None: ...
def rl_obs_constants() -> tuple[int, int, int, int]: ...
def encode_obs_v1(
    planets: np.ndarray,
    fleets: np.ndarray,
    angular_velocity: float,
    max_entities: int = ...,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
