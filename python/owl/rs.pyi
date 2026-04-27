import numpy as np

class RlVecEnv:
    n_envs: int
    n_players: int
    max_planets: int
    max_entities: int
    max_fleets: int

    def __init__(
        self,
        n_envs: int,
        two_player_weight: float = ...,
        obs_spec: str = ...,
        action_spec: str = ...,
        max_entities: int = ...,
    ) -> None: ...
    def reset(
        self,
        planet_obs: np.ndarray,
        fleet_obs: np.ndarray,
        comet_obs: np.ndarray,
        planet_mask: np.ndarray,
        fleet_mask: np.ndarray,
        comet_mask: np.ndarray,
        global_obs: np.ndarray,
        can_act: np.ndarray,
        max_launch: np.ndarray,
    ) -> None: ...
    def step(
        self,
        launch: np.ndarray,
        angle: np.ndarray,
        ships: np.ndarray,
        planet_obs: np.ndarray,
        fleet_obs: np.ndarray,
        comet_obs: np.ndarray,
        planet_mask: np.ndarray,
        fleet_mask: np.ndarray,
        comet_mask: np.ndarray,
        global_obs: np.ndarray,
        can_act: np.ndarray,
        max_launch: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None: ...
    def obs_shapes(
        self,
    ) -> tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ]: ...

def hello_from_rust() -> str: ...
def hello_numpy() -> np.ndarray: ...
def assert_release_build() -> None: ...
def rl_obs_constants() -> tuple[int, int, int, int, int, int, int, int, int]: ...
def encode_obs_v1(
    planets: np.ndarray,
    fleets: np.ndarray,
    comet_planet_ids: np.ndarray,
    comet_path_indices: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_paths: np.ndarray,
    angular_velocity: float,
    step: int = ...,
    episode_steps: int = ...,
    max_entities: int = ...,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]: ...
