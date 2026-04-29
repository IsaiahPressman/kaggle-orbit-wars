use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::rules_engine::env::{reset, step, PlayerAction};
use crate::rules_engine::state::{PlayerResult, ResetConfig, State};

use super::action_spec::decode_pure_actions;
use super::obs_spec::encode_state;
use super::{
    log_ignored_fleets, require_shape, PlayerMap, ACTION_ENTITY_SLOTS, COMET_CHANNELS,
    DEFAULT_MAX_ENTITIES, FLEET_CHANNELS, GLOBAL_CHANNELS, MAX_COMETS, MAX_PLANETS,
    OUTER_PLAYER_SLOTS, PLANET_CHANNELS,
};

type ObsShapes = (
    (usize, usize, usize),
    (usize, usize, usize),
    (usize, usize, usize),
    (usize, usize),
    (usize, usize),
    (usize, usize),
    (usize, usize),
    (usize, usize),
    (usize, usize, usize),
    (usize, usize, usize),
);

#[pyclass(name = "RlVecEnv")]
pub struct PyRlVecEnv {
    n_envs: usize,
    two_player_weight: f64,
    max_entities: usize,
    max_fleets: usize,
    max_per_planet_launches: usize,
    states: Vec<State>,
    player_maps: Vec<PlayerMap>,
    player_finished: Vec<Vec<bool>>,
}

#[pymethods]
impl PyRlVecEnv {
    #[new]
    #[pyo3(signature = (n_envs, two_player_weight=0.5, obs_spec="obs_v1", action_spec="pure", max_entities=DEFAULT_MAX_ENTITIES, max_per_planet_launches=3))]
    fn new(
        n_envs: usize,
        two_player_weight: f64,
        obs_spec: &str,
        action_spec: &str,
        max_entities: usize,
        max_per_planet_launches: usize,
    ) -> PyResult<Self> {
        if n_envs == 0 {
            return Err(PyValueError::new_err("n_envs must be positive"));
        }
        if !(0.0..=1.0).contains(&two_player_weight) {
            return Err(PyValueError::new_err("two_player_weight must be in [0, 1]"));
        }
        if obs_spec != "obs_v1" {
            return Err(PyValueError::new_err(format!(
                "unsupported obs_spec {obs_spec:?}; expected \"obs_v1\""
            )));
        }
        if action_spec != "pure" {
            return Err(PyValueError::new_err(format!(
                "unsupported action_spec {action_spec:?}; expected \"pure\""
            )));
        }
        if max_entities <= MAX_PLANETS + MAX_COMETS {
            return Err(PyValueError::new_err(format!(
                "max_entities must be greater than MAX_PLANETS + MAX_COMETS ({})",
                MAX_PLANETS + MAX_COMETS
            )));
        }
        if !(1..=4).contains(&max_per_planet_launches) {
            return Err(PyValueError::new_err(
                "max_per_planet_launches must be between 1 and 4",
            ));
        }

        let envs = (0..n_envs)
            .map(|_| reset_one_env(two_player_weight))
            .collect::<Vec<_>>();
        let (states, player_maps) = envs.into_iter().unzip();

        Ok(Self {
            n_envs,
            two_player_weight,
            max_entities,
            max_fleets: max_entities - (MAX_PLANETS + MAX_COMETS),
            max_per_planet_launches,
            states,
            player_maps,
            player_finished: vec![vec![false; OUTER_PLAYER_SLOTS]; n_envs],
        })
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.n_envs
    }

    #[getter]
    fn n_players(&self) -> usize {
        OUTER_PLAYER_SLOTS
    }

    #[getter]
    fn max_planets(&self) -> usize {
        MAX_PLANETS
    }

    #[getter]
    fn max_entities(&self) -> usize {
        self.max_entities
    }

    #[getter]
    fn max_fleets(&self) -> usize {
        self.max_fleets
    }

    #[getter]
    fn max_per_planet_launches(&self) -> usize {
        self.max_per_planet_launches
    }

    #[allow(clippy::too_many_arguments)]
    fn reset(
        &mut self,
        planet_obs: PyReadwriteArrayDyn<'_, f32>,
        fleet_obs: PyReadwriteArrayDyn<'_, f32>,
        comet_obs: PyReadwriteArrayDyn<'_, f32>,
        planet_mask: PyReadwriteArrayDyn<'_, bool>,
        fleet_mask: PyReadwriteArrayDyn<'_, bool>,
        comet_mask: PyReadwriteArrayDyn<'_, bool>,
        still_playing: PyReadwriteArrayDyn<'_, bool>,
        global_obs: PyReadwriteArrayDyn<'_, f32>,
        can_act: PyReadwriteArrayDyn<'_, bool>,
        max_launch: PyReadwriteArrayDyn<'_, i64>,
    ) -> PyResult<()> {
        self.states
            .par_iter_mut()
            .zip_eq(self.player_maps.par_iter_mut())
            .zip_eq(self.player_finished.par_iter_mut())
            .for_each(|((state, player_map), player_finished)| {
                let (new_state, new_player_map) = reset_one_env(self.two_player_weight);
                *state = new_state;
                *player_map = new_player_map;
                player_finished.fill(false);
            });
        self.write_obs(
            planet_obs,
            fleet_obs,
            comet_obs,
            planet_mask,
            fleet_mask,
            comet_mask,
            still_playing,
            global_obs,
            can_act,
            max_launch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        launch: PyReadonlyArrayDyn<'_, bool>,
        angle: PyReadonlyArrayDyn<'_, f32>,
        ships: PyReadonlyArrayDyn<'_, i64>,
        planet_obs: PyReadwriteArrayDyn<'_, f32>,
        fleet_obs: PyReadwriteArrayDyn<'_, f32>,
        comet_obs: PyReadwriteArrayDyn<'_, f32>,
        planet_mask: PyReadwriteArrayDyn<'_, bool>,
        fleet_mask: PyReadwriteArrayDyn<'_, bool>,
        comet_mask: PyReadwriteArrayDyn<'_, bool>,
        still_playing: PyReadwriteArrayDyn<'_, bool>,
        global_obs: PyReadwriteArrayDyn<'_, f32>,
        can_act: PyReadwriteArrayDyn<'_, bool>,
        max_launch: PyReadwriteArrayDyn<'_, i64>,
        rewards: PyReadwriteArrayDyn<'_, f32>,
        dones: PyReadwriteArrayDyn<'_, bool>,
    ) -> PyResult<()> {
        let action_shape = [
            self.n_envs,
            OUTER_PLAYER_SLOTS,
            ACTION_ENTITY_SLOTS,
            self.max_per_planet_launches,
        ];
        require_shape("launch", launch.shape(), &action_shape)?;
        require_shape("angle", angle.shape(), &action_shape)?;
        require_shape("ships", ships.shape(), &action_shape)?;
        require_shape(
            "rewards",
            rewards.shape(),
            &[self.n_envs, OUTER_PLAYER_SLOTS],
        )?;
        require_shape("dones", dones.shape(), &[self.n_envs, OUTER_PLAYER_SLOTS])?;

        let mut rewards = rewards;
        let mut dones = dones;
        let reward_chunks = rewards.as_slice_mut()?.par_chunks_mut(OUTER_PLAYER_SLOTS);
        let done_chunks = dones.as_slice_mut()?.par_chunks_mut(OUTER_PLAYER_SLOTS);
        let actions_per_env =
            OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS * self.max_per_planet_launches;
        let launch_chunks = launch.as_slice()?.par_chunks(actions_per_env);
        let angle_chunks = angle.as_slice()?.par_chunks(actions_per_env);
        let ship_chunks = ships.as_slice()?.par_chunks(actions_per_env);

        let decoded = self
            .states
            .par_iter()
            .zip_eq(self.player_maps.par_iter())
            .zip_eq(launch_chunks)
            .zip_eq(angle_chunks)
            .zip_eq(ship_chunks)
            .enumerate()
            .map(
                |(env_index, ((((state, player_map), launch_chunk), angle_chunk), ship_chunk))| {
                    decode_pure_actions(
                        state,
                        player_map,
                        launch_chunk,
                        angle_chunk,
                        ship_chunk,
                        self.max_per_planet_launches,
                    )
                    .map_err(|err| format!("env {env_index}: {err}"))
                },
            )
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;

        self.states
            .par_iter_mut()
            .zip_eq(self.player_maps.par_iter_mut())
            .zip_eq(self.player_finished.par_iter_mut())
            .zip_eq(decoded.par_iter())
            .zip_eq(reward_chunks)
            .zip_eq(done_chunks)
            .for_each(
                |(
                    ((((state, player_map), player_finished), decoded), reward_chunk),
                    done_chunk,
                )| {
                    step_one_env(
                        state,
                        player_map,
                        player_finished,
                        decoded,
                        reward_chunk,
                        done_chunk,
                        self.two_player_weight,
                    );
                },
            );

        self.write_obs(
            planet_obs,
            fleet_obs,
            comet_obs,
            planet_mask,
            fleet_mask,
            comet_mask,
            still_playing,
            global_obs,
            can_act,
            max_launch,
        )
    }

    fn obs_shapes(&self) -> ObsShapes {
        (
            (self.n_envs, MAX_PLANETS, PLANET_CHANNELS),
            (self.n_envs, self.max_fleets, FLEET_CHANNELS),
            (self.n_envs, MAX_COMETS, COMET_CHANNELS),
            (self.n_envs, MAX_PLANETS),
            (self.n_envs, self.max_fleets),
            (self.n_envs, MAX_COMETS),
            (self.n_envs, OUTER_PLAYER_SLOTS),
            (self.n_envs, GLOBAL_CHANNELS),
            (self.n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS),
            (self.n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS),
        )
    }
}

impl PyRlVecEnv {
    #[allow(clippy::too_many_arguments)]
    fn write_obs(
        &self,
        planet_obs: PyReadwriteArrayDyn<'_, f32>,
        fleet_obs: PyReadwriteArrayDyn<'_, f32>,
        comet_obs: PyReadwriteArrayDyn<'_, f32>,
        planet_mask: PyReadwriteArrayDyn<'_, bool>,
        fleet_mask: PyReadwriteArrayDyn<'_, bool>,
        comet_mask: PyReadwriteArrayDyn<'_, bool>,
        still_playing: PyReadwriteArrayDyn<'_, bool>,
        global_obs: PyReadwriteArrayDyn<'_, f32>,
        can_act: PyReadwriteArrayDyn<'_, bool>,
        max_launch: PyReadwriteArrayDyn<'_, i64>,
    ) -> PyResult<()> {
        require_shape(
            "planet_obs",
            planet_obs.shape(),
            &[self.n_envs, MAX_PLANETS, PLANET_CHANNELS],
        )?;
        require_shape(
            "fleet_obs",
            fleet_obs.shape(),
            &[self.n_envs, self.max_fleets, FLEET_CHANNELS],
        )?;
        require_shape(
            "comet_obs",
            comet_obs.shape(),
            &[self.n_envs, MAX_COMETS, COMET_CHANNELS],
        )?;
        require_shape(
            "planet_mask",
            planet_mask.shape(),
            &[self.n_envs, MAX_PLANETS],
        )?;
        require_shape(
            "fleet_mask",
            fleet_mask.shape(),
            &[self.n_envs, self.max_fleets],
        )?;
        require_shape("comet_mask", comet_mask.shape(), &[self.n_envs, MAX_COMETS])?;
        require_shape(
            "still_playing",
            still_playing.shape(),
            &[self.n_envs, OUTER_PLAYER_SLOTS],
        )?;
        require_shape(
            "global_obs",
            global_obs.shape(),
            &[self.n_envs, GLOBAL_CHANNELS],
        )?;
        require_shape(
            "can_act",
            can_act.shape(),
            &[self.n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS],
        )?;
        require_shape(
            "max_launch",
            max_launch.shape(),
            &[self.n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS],
        )?;

        let mut planet_obs = planet_obs;
        let mut fleet_obs = fleet_obs;
        let mut comet_obs = comet_obs;
        let mut planet_mask = planet_mask;
        let mut fleet_mask = fleet_mask;
        let mut comet_mask = comet_mask;
        let mut still_playing = still_playing;
        let mut global_obs = global_obs;
        let mut can_act = can_act;
        let mut max_launch = max_launch;

        let planets_per_env = MAX_PLANETS * PLANET_CHANNELS;
        let fleets_per_env = self.max_fleets * FLEET_CHANNELS;
        let comets_per_env = MAX_COMETS * COMET_CHANNELS;
        let planet_masks_per_env = MAX_PLANETS;
        let fleet_masks_per_env = self.max_fleets;
        let comet_masks_per_env = MAX_COMETS;
        let still_playing_per_env = OUTER_PLAYER_SLOTS;
        let globals_per_env = GLOBAL_CHANNELS;
        let action_masks_per_env = OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS;

        let ignored_fleets: usize = self
            .states
            .par_iter()
            .zip_eq(self.player_maps.par_iter())
            .zip_eq(self.player_finished.par_iter())
            .zip_eq(planet_obs.as_slice_mut()?.par_chunks_mut(planets_per_env))
            .zip_eq(fleet_obs.as_slice_mut()?.par_chunks_mut(fleets_per_env))
            .zip_eq(comet_obs.as_slice_mut()?.par_chunks_mut(comets_per_env))
            .zip_eq(
                planet_mask
                    .as_slice_mut()?
                    .par_chunks_mut(planet_masks_per_env),
            )
            .zip_eq(
                fleet_mask
                    .as_slice_mut()?
                    .par_chunks_mut(fleet_masks_per_env),
            )
            .zip_eq(
                comet_mask
                    .as_slice_mut()?
                    .par_chunks_mut(comet_masks_per_env),
            )
            .zip_eq(
                still_playing
                    .as_slice_mut()?
                    .par_chunks_mut(still_playing_per_env),
            )
            .zip_eq(global_obs.as_slice_mut()?.par_chunks_mut(globals_per_env))
            .zip_eq(can_act.as_slice_mut()?.par_chunks_mut(action_masks_per_env))
            .zip_eq(
                max_launch
                    .as_slice_mut()?
                    .par_chunks_mut(action_masks_per_env),
            )
            .map(
                |(
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((state, player_map), player_finished),
                                                        planet_obs,
                                                    ),
                                                    fleet_obs,
                                                ),
                                                comet_obs,
                                            ),
                                            planet_mask,
                                        ),
                                        fleet_mask,
                                    ),
                                    comet_mask,
                                ),
                                still_playing,
                            ),
                            global_obs,
                        ),
                        can_act,
                    ),
                    max_launch,
                )| {
                    write_still_playing(state, player_map, player_finished, still_playing);
                    encode_state(
                        state,
                        player_map,
                        self.max_fleets,
                        planet_obs,
                        fleet_obs,
                        comet_obs,
                        planet_mask,
                        fleet_mask,
                        comet_mask,
                        global_obs,
                        can_act,
                        max_launch,
                    )
                },
            )
            .sum();

        log_ignored_fleets(ignored_fleets);
        Ok(())
    }
}

fn write_still_playing(
    state: &State,
    player_map: &PlayerMap,
    player_finished: &[bool],
    still_playing: &mut [bool],
) {
    still_playing.fill(false);
    for player_index in 0..state.config.player_count {
        still_playing[player_map.internal_to_outer(player_index)] = !player_finished[player_index];
    }
}

fn step_one_env(
    state: &mut State,
    player_map: &mut PlayerMap,
    player_finished: &mut [bool],
    decoded: &[PlayerAction],
    reward_chunk: &mut [f32],
    done_chunk: &mut [bool],
    two_player_weight: f64,
) {
    let result = step(state, decoded);
    let should_reset = result_is_terminal(&result.player_results);
    let won_reward = split_won_reward(
        result
            .player_results
            .iter()
            .filter(|result| matches!(result, PlayerResult::Won))
            .count(),
    );

    reward_chunk.fill(0.0);
    done_chunk.fill(true);
    for (player_index, result) in result.player_results.iter().enumerate() {
        let (reward, done) = player_reward_done(*result, player_finished[player_index], won_reward);
        let outer_player = player_map.internal_to_outer(player_index);
        reward_chunk[outer_player] = reward;
        done_chunk[outer_player] = done;
        if done {
            player_finished[player_index] = true;
        }
    }

    if should_reset {
        let (new_state, new_player_map) = reset_one_env(two_player_weight);
        *state = new_state;
        *player_map = new_player_map;
        player_finished.fill(false);
    }
}

fn result_is_terminal(results: &[PlayerResult]) -> bool {
    results
        .iter()
        .all(|result| !matches!(result, PlayerResult::Active))
}

fn sample_reset_config(two_player_weight: f64) -> ResetConfig {
    let player_count = if rand::random_bool(two_player_weight) {
        2
    } else {
        4
    };
    ResetConfig::new(player_count)
}

fn reset_one_env(two_player_weight: f64) -> (State, PlayerMap) {
    let state = reset(sample_reset_config(two_player_weight));
    let player_map = PlayerMap::random(state.config.player_count);
    (state, player_map)
}

fn split_won_reward(winner_count: usize) -> f32 {
    if winner_count <= 1 {
        return 1.0;
    }
    (2.0 - winner_count as f32) / winner_count as f32
}

fn player_reward_done(
    result: PlayerResult,
    previously_finished: bool,
    won_reward: f32,
) -> (f32, bool) {
    if previously_finished {
        return (0.0, true);
    }
    match result {
        PlayerResult::Active => (0.0, false),
        PlayerResult::Lost => (-1.0, true),
        PlayerResult::Won => (won_reward, true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::state::{Planet, SimConfig};

    fn state_with_player_three_eliminated() -> State {
        let planets = vec![
            Planet {
                id: 0,
                owner: 0,
                x: 10.0,
                y: 10.0,
                radius: 2.0,
                ships: 10,
                production: 1,
            },
            Planet {
                id: 1,
                owner: 1,
                x: 90.0,
                y: 10.0,
                radius: 2.0,
                ships: 10,
                production: 1,
            },
            Planet {
                id: 2,
                owner: 2,
                x: 10.0,
                y: 90.0,
                radius: 2.0,
                ships: 10,
                production: 1,
            },
        ];
        State {
            config: SimConfig::new(4),
            step: 0,
            angular_velocity: 0.025,
            initial_planets: planets.clone(),
            planets,
            fleets: Vec::new(),
            next_fleet_id: 0,
            comets: Vec::new(),
            comet_planet_ids: Vec::new(),
        }
    }

    fn state_with_all_players_alive() -> State {
        let planets = vec![
            Planet {
                id: 0,
                owner: 0,
                x: 10.0,
                y: 10.0,
                radius: 2.0,
                ships: 10,
                production: 1,
            },
            Planet {
                id: 1,
                owner: 1,
                x: 90.0,
                y: 10.0,
                radius: 2.0,
                ships: 10,
                production: 1,
            },
            Planet {
                id: 2,
                owner: 2,
                x: 10.0,
                y: 90.0,
                radius: 2.0,
                ships: 10,
                production: 1,
            },
            Planet {
                id: 3,
                owner: 3,
                x: 90.0,
                y: 90.0,
                radius: 2.0,
                ships: 10,
                production: 1,
            },
        ];
        State {
            config: SimConfig::new(4),
            step: 0,
            angular_velocity: 0.025,
            initial_planets: planets.clone(),
            planets,
            fleets: Vec::new(),
            next_fleet_id: 0,
            comets: Vec::new(),
            comet_planet_ids: Vec::new(),
        }
    }

    fn state_with_one_player_alive() -> State {
        let planets = vec![Planet {
            id: 0,
            owner: 0,
            x: 10.0,
            y: 10.0,
            radius: 2.0,
            ships: 10,
            production: 1,
        }];
        State {
            config: SimConfig::new(4),
            step: 0,
            angular_velocity: 0.025,
            initial_planets: planets.clone(),
            planets,
            fleets: Vec::new(),
            next_fleet_id: 0,
            comets: Vec::new(),
            comet_planet_ids: Vec::new(),
        }
    }

    #[test]
    fn nonterminal_horizon_step_does_not_reset_before_returning_done() {
        let actions = vec![Vec::new(); 4];
        let mut state = state_with_all_players_alive();
        let mut player_map = PlayerMap::identity();
        state.step = state.config.episode_steps.saturating_sub(3);
        let expected_step = state.step + 1;
        let mut finished = vec![false; 4];
        let mut rewards = vec![99.0; 4];
        let mut dones = vec![true; 4];

        step_one_env(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
            0.0,
        );

        assert_eq!(state.step, expected_step);
        assert_eq!(rewards, vec![0.0; 4]);
        assert_eq!(dones, vec![false; 4]);
        assert_eq!(finished, vec![false; 4]);
    }

    #[test]
    fn eliminated_player_gets_one_loss_reward_then_sticky_done() {
        let actions = vec![Vec::new(); 4];
        let mut state = state_with_player_three_eliminated();
        let mut player_map = PlayerMap::identity();
        let mut finished = vec![false; 4];
        let mut rewards = vec![99.0; 4];
        let mut dones = vec![false; 4];

        step_one_env(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
            0.0,
        );

        assert_eq!(rewards, vec![0.0, 0.0, 0.0, -1.0]);
        assert_eq!(dones, vec![false, false, false, true]);
        assert_eq!(finished, vec![false, false, false, true]);

        step_one_env(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
            0.0,
        );

        assert_eq!(rewards[3], 0.0);
        assert!(dones[3]);
    }

    #[test]
    fn split_won_reward_averages_one_win_with_tied_losses() {
        assert_eq!(split_won_reward(0), 1.0);
        assert_eq!(split_won_reward(1), 1.0);
        assert_eq!(split_won_reward(2), 0.0);
        assert!((split_won_reward(3) - (-1.0 / 3.0)).abs() <= f32::EPSILON);
        assert_eq!(split_won_reward(4), -0.5);
    }

    #[test]
    fn tied_terminal_winners_get_split_reward() {
        let actions = vec![Vec::new(); 4];
        let mut state = state_with_all_players_alive();
        let mut player_map = PlayerMap::identity();
        state.step = state.config.episode_steps.saturating_sub(2);
        let mut finished = vec![false; 4];
        let mut rewards = vec![99.0; 4];
        let mut dones = vec![false; 4];

        step_one_env(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
            0.0,
        );

        assert_eq!(rewards, vec![-0.5; 4]);
        assert_eq!(dones, vec![true; 4]);
    }

    #[test]
    fn still_playing_marks_only_current_unfinished_player_slots() {
        let state = state_with_player_three_eliminated();
        let player_map = PlayerMap::identity();
        let finished = vec![false, true, false, true];
        let mut still_playing = vec![true; 4];

        write_still_playing(&state, &player_map, &finished, &mut still_playing);

        assert_eq!(still_playing, vec![true, false, true, false]);
    }

    #[test]
    fn still_playing_uses_remapped_outer_player_slots() {
        let mut state = state_with_all_players_alive();
        state.config = SimConfig::new(2);
        let player_map = PlayerMap::from_outer_slots(2, [3, 1, 0, 2]);
        let finished = vec![false, true, false, false];
        let mut still_playing = vec![true; 4];

        write_still_playing(&state, &player_map, &finished, &mut still_playing);

        assert_eq!(still_playing, vec![false, false, false, true]);
    }

    #[test]
    fn rewards_and_dones_use_remapped_outer_player_slots() {
        let actions = vec![Vec::new(); 2];
        let mut state = state_with_all_players_alive();
        state.config = SimConfig::new(2);
        state.planets.retain(|planet| planet.owner <= 1);
        let mut player_map = PlayerMap::from_outer_slots(2, [3, 1, 0, 2]);
        let mut finished = vec![false; 4];
        let mut rewards = vec![99.0; 4];
        let mut dones = vec![false; 4];

        step_one_env(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
            0.0,
        );

        assert_eq!(rewards, vec![0.0; 4]);
        assert_eq!(dones, vec![true, false, true, false]);
    }

    #[test]
    fn still_playing_describes_reset_observation_after_terminal_step() {
        let actions = vec![Vec::new(); 4];
        let mut state = state_with_one_player_alive();
        let mut player_map = PlayerMap::identity();
        let mut finished = vec![false; 4];
        let mut rewards = vec![0.0; 4];
        let mut dones = vec![false; 4];

        step_one_env(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
            0.0,
        );

        let mut still_playing = vec![false; 4];
        write_still_playing(&state, &player_map, &finished, &mut still_playing);

        assert_eq!(dones, vec![true; 4]);
        assert_eq!(still_playing, vec![true; 4]);
    }
}
