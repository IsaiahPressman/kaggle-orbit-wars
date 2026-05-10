import numpy as np

class RlVecEnv:
    n_envs: int
    n_players: int
    max_planets: int
    max_entities: int
    max_fleets: int
    max_per_planet_launches: int
    min_fleet_size: int

    def __init__(
        self,
        n_envs: int,
        two_player_weight: float = ...,
        obs_spec: str = ...,
        action_spec: str = ...,
        max_entities: int = ...,
        max_per_planet_launches: int = ...,
        min_fleet_size: int = ...,
        n_bins: int = ...,
    ) -> None: ...
    def reset(
        self,
        planet_obs: np.ndarray,
        orbiting_planet_obs: np.ndarray,
        fleet_obs: np.ndarray,
        comet_obs: np.ndarray,
        entity_mask: np.ndarray,
        still_playing: np.ndarray,
        global_obs: np.ndarray,
        can_act: np.ndarray,
        max_launch: np.ndarray,
    ) -> None: ...
    def reset_discrete_target_bins(
        self,
        planet_obs: np.ndarray,
        orbiting_planet_obs: np.ndarray,
        fleet_obs: np.ndarray,
        comet_obs: np.ndarray,
        entity_mask: np.ndarray,
        still_playing: np.ndarray,
        global_obs: np.ndarray,
        can_act: np.ndarray,
    ) -> None: ...
    def step(
        self,
        launch: np.ndarray,
        angle: np.ndarray,
        ships: np.ndarray,
        planet_obs: np.ndarray,
        orbiting_planet_obs: np.ndarray,
        fleet_obs: np.ndarray,
        comet_obs: np.ndarray,
        entity_mask: np.ndarray,
        still_playing: np.ndarray,
        global_obs: np.ndarray,
        can_act: np.ndarray,
        max_launch: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, list[float]]: ...
    def step_discrete_target_bins(
        self,
        target: np.ndarray,
        fleet_bin: np.ndarray,
        planet_obs: np.ndarray,
        orbiting_planet_obs: np.ndarray,
        fleet_obs: np.ndarray,
        comet_obs: np.ndarray,
        entity_mask: np.ndarray,
        still_playing: np.ndarray,
        global_obs: np.ndarray,
        can_act: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, list[float]]: ...
    def step_discrete_targets(
        self,
        launch: np.ndarray,
        target: np.ndarray,
        ships: np.ndarray,
        planet_obs: np.ndarray,
        orbiting_planet_obs: np.ndarray,
        fleet_obs: np.ndarray,
        comet_obs: np.ndarray,
        entity_mask: np.ndarray,
        still_playing: np.ndarray,
        global_obs: np.ndarray,
        can_act: np.ndarray,
        max_launch: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, list[float]]: ...
    def obs_shapes(
        self,
    ) -> tuple[
        tuple[int, int, int],
        tuple[int, int],
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, ...],
        tuple[int, int, int],
    ]: ...
    def state_snapshot(self, env_index: int) -> dict[str, object]: ...
    def terminal_snapshot(self, env_index: int) -> dict[str, object] | None: ...
    def terminal_metrics(self, env_index: int) -> dict[str, float] | None: ...
    def write_action_mask(
        self,
        action_spec: str,
        min_fleet_size: int,
        n_bins: int,
        can_act: np.ndarray,
        max_launch: np.ndarray,
    ) -> None: ...
    def write_action_mask_discrete_target_bins(
        self,
        min_fleet_size: int,
        n_bins: int,
        can_act: np.ndarray,
    ) -> None: ...
    def decode_pure_actions(
        self,
        launch: np.ndarray,
        angle: np.ndarray,
        ships: np.ndarray,
        max_per_planet_launches: int,
        min_fleet_size: int,
        valid: np.ndarray,
        from_planet_id: np.ndarray,
        decoded_angle: np.ndarray,
        decoded_ships: np.ndarray,
    ) -> None: ...
    def decode_discrete_target_actions(
        self,
        launch: np.ndarray,
        target: np.ndarray,
        ships: np.ndarray,
        max_per_planet_launches: int,
        min_fleet_size: int,
        valid: np.ndarray,
        from_planet_id: np.ndarray,
        decoded_angle: np.ndarray,
        decoded_ships: np.ndarray,
    ) -> None: ...
    def decode_discrete_target_bin_actions(
        self,
        target: np.ndarray,
        fleet_bin: np.ndarray,
        min_fleet_size: int,
        n_bins: int,
        valid: np.ndarray,
        from_planet_id: np.ndarray,
        decoded_angle: np.ndarray,
        decoded_ships: np.ndarray,
    ) -> None: ...
    def step_decoded_actions(
        self,
        valid: np.ndarray,
        from_planet_id: np.ndarray,
        angle: np.ndarray,
        ships: np.ndarray,
        planet_obs: np.ndarray,
        orbiting_planet_obs: np.ndarray,
        fleet_obs: np.ndarray,
        comet_obs: np.ndarray,
        entity_mask: np.ndarray,
        still_playing: np.ndarray,
        global_obs: np.ndarray,
        can_act: np.ndarray,
        max_launch: np.ndarray | None,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, list[float]]: ...

def assert_release_build() -> None: ...
def rl_obs_constants() -> tuple[int, int, int, int, int, int, int, int, int]: ...
def encode_entity_based(
    planets: np.ndarray,
    initial_planets: np.ndarray,
    fleets: np.ndarray,
    comet_planet_ids: np.ndarray,
    comet_path_indices: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_paths: np.ndarray,
    angular_velocity: float,
    step: int = ...,
    episode_steps: int = ...,
    max_entities: int = ...,
    min_fleet_size: int = ...,
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
def pure_actions_to_kaggle(
    planets: np.ndarray,
    initial_planets: np.ndarray,
    fleets: np.ndarray,
    comet_planet_ids: np.ndarray,
    comet_path_indices: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_paths: np.ndarray,
    angular_velocity: float,
    step: int,
    episode_steps: int,
    player: int,
    launch: np.ndarray,
    angle: np.ndarray,
    ships: np.ndarray,
    max_per_planet_launches: int,
    min_fleet_size: int,
) -> list[list[float]]: ...
def discrete_target_actions_to_kaggle(
    planets: np.ndarray,
    initial_planets: np.ndarray,
    fleets: np.ndarray,
    comet_planet_ids: np.ndarray,
    comet_path_indices: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_paths: np.ndarray,
    angular_velocity: float,
    step: int,
    episode_steps: int,
    player: int,
    launch: np.ndarray,
    target: np.ndarray,
    ships: np.ndarray,
    max_per_planet_launches: int,
    min_fleet_size: int,
) -> list[list[float]]: ...
def discrete_target_bin_actions_to_kaggle(
    planets: np.ndarray,
    initial_planets: np.ndarray,
    fleets: np.ndarray,
    comet_planet_ids: np.ndarray,
    comet_path_indices: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_paths: np.ndarray,
    angular_velocity: float,
    step: int,
    episode_steps: int,
    player: int,
    target: np.ndarray,
    fleet_bin: np.ndarray,
    min_fleet_size: int,
    n_bins: int,
) -> list[list[float]]: ...
