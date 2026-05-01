use std::collections::{HashMap, HashSet};

use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::rules_engine::env::{reset, step, PlayerAction};
use crate::rules_engine::state::{FleetLossStats, PlayerResult, ResetConfig, State};

use super::action_spec::{action_entity_slots, decode_pure_actions, ActionEntitySlots};
use super::obs_spec::encode_state_with_action_slots;
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
    min_fleet_size: i64,
    states: Vec<State>,
    player_maps: Vec<PlayerMap>,
    action_slots: Vec<ActionEntitySlots>,
    player_finished: Vec<Vec<bool>>,
    episode_stats: Vec<EpisodeStats>,
}

#[pymethods]
impl PyRlVecEnv {
    #[new]
    #[pyo3(signature = (n_envs, two_player_weight=0.5, obs_spec="obs_v1", action_spec="pure", max_entities=DEFAULT_MAX_ENTITIES, max_per_planet_launches=3, min_fleet_size=1))]
    fn new(
        n_envs: usize,
        two_player_weight: f64,
        obs_spec: &str,
        action_spec: &str,
        max_entities: usize,
        max_per_planet_launches: usize,
        min_fleet_size: i64,
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
        if min_fleet_size < 1 || min_fleet_size > i64::from(i32::MAX) {
            return Err(PyValueError::new_err(
                "min_fleet_size must be between 1 and i32::MAX",
            ));
        }

        let envs = (0..n_envs)
            .map(|_| reset_one_env(two_player_weight))
            .collect::<Vec<_>>();
        let (states, player_maps): (Vec<_>, Vec<_>) = envs.into_iter().unzip();
        let action_slots = states.iter().map(action_entity_slots).collect();

        Ok(Self {
            n_envs,
            two_player_weight,
            max_entities,
            max_fleets: max_entities - (MAX_PLANETS + MAX_COMETS),
            max_per_planet_launches,
            min_fleet_size,
            states,
            player_maps,
            action_slots,
            player_finished: vec![vec![false; OUTER_PLAYER_SLOTS]; n_envs],
            episode_stats: vec![EpisodeStats::default(); n_envs],
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

    #[getter]
    fn min_fleet_size(&self) -> i64 {
        self.min_fleet_size
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
        self.require_obs_shapes(
            &planet_obs,
            &fleet_obs,
            &comet_obs,
            &planet_mask,
            &fleet_mask,
            &comet_mask,
            &still_playing,
            &global_obs,
            &can_act,
            &max_launch,
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
        let action_masks_per_env = OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS;
        let two_player_weight = self.two_player_weight;
        let max_fleets = self.max_fleets;
        let min_fleet_size = self.min_fleet_size;

        let ignored_fleets: usize = self
            .states
            .par_iter_mut()
            .zip_eq(self.player_maps.par_iter_mut())
            .zip_eq(self.action_slots.par_iter_mut())
            .zip_eq(self.player_finished.par_iter_mut())
            .zip_eq(self.episode_stats.par_iter_mut())
            .zip_eq(planet_obs.as_slice_mut()?.par_chunks_mut(planets_per_env))
            .zip_eq(fleet_obs.as_slice_mut()?.par_chunks_mut(fleets_per_env))
            .zip_eq(comet_obs.as_slice_mut()?.par_chunks_mut(comets_per_env))
            .zip_eq(planet_mask.as_slice_mut()?.par_chunks_mut(MAX_PLANETS))
            .zip_eq(fleet_mask.as_slice_mut()?.par_chunks_mut(self.max_fleets))
            .zip_eq(comet_mask.as_slice_mut()?.par_chunks_mut(MAX_COMETS))
            .zip_eq(
                still_playing
                    .as_slice_mut()?
                    .par_chunks_mut(OUTER_PLAYER_SLOTS),
            )
            .zip_eq(global_obs.as_slice_mut()?.par_chunks_mut(GLOBAL_CHANNELS))
            .zip_eq(can_act.as_slice_mut()?.par_chunks_mut(action_masks_per_env))
            .zip_eq(
                max_launch
                    .as_slice_mut()?
                    .par_chunks_mut(action_masks_per_env),
            )
            .map(|item| {
                let (item, max_launch) = item;
                let (item, can_act) = item;
                let (item, global_obs) = item;
                let (item, still_playing) = item;
                let (item, comet_mask) = item;
                let (item, fleet_mask) = item;
                let (item, planet_mask) = item;
                let (item, comet_obs) = item;
                let (item, fleet_obs) = item;
                let (item, planet_obs) = item;
                let ((((state, player_map), action_slots), player_finished), episode_stats) = item;

                {
                    let (new_state, new_player_map) = reset_one_env(two_player_weight);
                    *state = new_state;
                    *player_map = new_player_map;
                    player_finished.fill(false);
                    *episode_stats = EpisodeStats::default();
                    write_one_obs(
                        state,
                        player_map,
                        action_slots,
                        player_finished,
                        max_fleets,
                        min_fleet_size,
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
            })
            .sum();

        log_ignored_fleets(ignored_fleets);
        Ok(())
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
    ) -> PyResult<HashMap<String, Vec<f64>>> {
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
        self.require_obs_shapes(
            &planet_obs,
            &fleet_obs,
            &comet_obs,
            &planet_mask,
            &fleet_mask,
            &comet_mask,
            &still_playing,
            &global_obs,
            &can_act,
            &max_launch,
        )?;

        let mut rewards = rewards;
        let mut dones = dones;
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

        let reward_chunks = rewards.as_slice_mut()?.par_chunks_mut(OUTER_PLAYER_SLOTS);
        let done_chunks = dones.as_slice_mut()?.par_chunks_mut(OUTER_PLAYER_SLOTS);
        let actions_per_env =
            OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS * self.max_per_planet_launches;
        let launch_chunks = launch.as_slice()?.par_chunks(actions_per_env);
        let angle_chunks = angle.as_slice()?.par_chunks(actions_per_env);
        let ship_chunks = ships.as_slice()?.par_chunks(actions_per_env);

        let planets_per_env = MAX_PLANETS * PLANET_CHANNELS;
        let fleets_per_env = self.max_fleets * FLEET_CHANNELS;
        let comets_per_env = MAX_COMETS * COMET_CHANNELS;
        let action_masks_per_env = OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS;
        let max_per_planet_launches = self.max_per_planet_launches;
        let min_fleet_size = self.min_fleet_size;
        let max_fleets = self.max_fleets;
        let two_player_weight = self.two_player_weight;

        let env_results = self
            .states
            .par_iter_mut()
            .zip_eq(self.player_maps.par_iter_mut())
            .zip_eq(self.action_slots.par_iter_mut())
            .zip_eq(self.player_finished.par_iter_mut())
            .zip_eq(self.episode_stats.par_iter_mut())
            .zip_eq(launch_chunks)
            .zip_eq(angle_chunks)
            .zip_eq(ship_chunks)
            .zip_eq(reward_chunks)
            .zip_eq(done_chunks)
            .zip_eq(planet_obs.as_slice_mut()?.par_chunks_mut(planets_per_env))
            .zip_eq(fleet_obs.as_slice_mut()?.par_chunks_mut(fleets_per_env))
            .zip_eq(comet_obs.as_slice_mut()?.par_chunks_mut(comets_per_env))
            .zip_eq(planet_mask.as_slice_mut()?.par_chunks_mut(MAX_PLANETS))
            .zip_eq(fleet_mask.as_slice_mut()?.par_chunks_mut(self.max_fleets))
            .zip_eq(comet_mask.as_slice_mut()?.par_chunks_mut(MAX_COMETS))
            .zip_eq(
                still_playing
                    .as_slice_mut()?
                    .par_chunks_mut(OUTER_PLAYER_SLOTS),
            )
            .zip_eq(global_obs.as_slice_mut()?.par_chunks_mut(GLOBAL_CHANNELS))
            .zip_eq(can_act.as_slice_mut()?.par_chunks_mut(action_masks_per_env))
            .zip_eq(
                max_launch
                    .as_slice_mut()?
                    .par_chunks_mut(action_masks_per_env),
            )
            .enumerate()
            .map(|(env_index, item)| {
                let (item, max_launch) = item;
                let (item, can_act) = item;
                let (item, global_obs) = item;
                let (item, still_playing) = item;
                let (item, comet_mask) = item;
                let (item, fleet_mask) = item;
                let (item, planet_mask) = item;
                let (item, comet_obs) = item;
                let (item, fleet_obs) = item;
                let (item, planet_obs) = item;
                let (item, done_chunk) = item;
                let (item, reward_chunk) = item;
                let (item, ship_chunk) = item;
                let (item, angle_chunk) = item;
                let (item, launch_chunk) = item;
                let ((((state, player_map), action_slots), player_finished), episode_stats) = item;

                {
                    let decoded = decode_pure_actions(
                        state,
                        player_map,
                        action_slots,
                        launch_chunk,
                        angle_chunk,
                        ship_chunk,
                        max_per_planet_launches,
                        min_fleet_size,
                    )
                    .map_err(|err| format!("env {env_index}: {err}"))?;
                    let terminal = step_one_env(
                        state,
                        player_map,
                        player_finished,
                        episode_stats,
                        &decoded,
                        reward_chunk,
                        done_chunk,
                        max_fleets,
                        two_player_weight,
                    );
                    let ignored_fleets = write_one_obs(
                        state,
                        player_map,
                        action_slots,
                        player_finished,
                        max_fleets,
                        min_fleet_size,
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
                    );
                    Ok::<_, String>((terminal, ignored_fleets))
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let mut terminal_metrics = Vec::with_capacity(env_results.len());
        let mut ignored_fleets = 0;
        for (terminal, ignored) in env_results {
            terminal_metrics.push(terminal);
            ignored_fleets += ignored;
        }
        let episode_metrics = collect_terminal_metrics(terminal_metrics);
        log_ignored_fleets(ignored_fleets);
        Ok(episode_metrics)
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
    fn require_obs_shapes(
        &self,
        planet_obs: &PyReadwriteArrayDyn<'_, f32>,
        fleet_obs: &PyReadwriteArrayDyn<'_, f32>,
        comet_obs: &PyReadwriteArrayDyn<'_, f32>,
        planet_mask: &PyReadwriteArrayDyn<'_, bool>,
        fleet_mask: &PyReadwriteArrayDyn<'_, bool>,
        comet_mask: &PyReadwriteArrayDyn<'_, bool>,
        still_playing: &PyReadwriteArrayDyn<'_, bool>,
        global_obs: &PyReadwriteArrayDyn<'_, f32>,
        can_act: &PyReadwriteArrayDyn<'_, bool>,
        max_launch: &PyReadwriteArrayDyn<'_, i64>,
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
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct EpisodeStats {
    occupancy_rate_sum: f64,
    occupancy_rate_turns: u32,
    launches_per_occupied_planet_sum: f64,
    occupied_planet_turns: u32,
    launches_per_launch_sum: f64,
    launched_planet_turns: u32,
    ships_per_launch_sum: i64,
    ships_per_launch_squared_sum: i64,
    launch_count: u32,
    turn_count: u32,
    max_fleet_size: i32,
    planets_captured: u32,
    max_entities_exceeded_turns: u32,
    fleet_losses: FleetLossStats,
}

#[derive(Clone, Debug)]
struct TerminalEpisodeMetrics {
    player_count: usize,
    values: Vec<(&'static str, f64)>,
    win_rates: Vec<(usize, f64)>,
}

impl EpisodeStats {
    fn record_turn(&mut self, state: &State, decoded: &[PlayerAction]) {
        let comet_ids = state
            .comet_planet_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let mut non_comet_planets = 0_usize;
        let mut occupied_planets = 0_usize;
        for planet in state.planets.iter() {
            if comet_ids.contains(&planet.id) {
                continue;
            }
            non_comet_planets += 1;
            if planet.owner != -1 {
                occupied_planets += 1;
            }
        }

        if non_comet_planets > 0 {
            self.occupancy_rate_sum += occupied_planets as f64 / non_comet_planets as f64;
            self.occupancy_rate_turns += 1;
        }

        let mut launches_by_planet = HashMap::<u32, u32>::new();
        for action in decoded.iter().flatten() {
            *launches_by_planet.entry(action.from_planet_id).or_default() += 1;
        }
        let launch_count = launches_by_planet.values().sum::<u32>();
        if occupied_planets > 0 {
            self.launches_per_occupied_planet_sum += launch_count as f64 / occupied_planets as f64;
            self.occupied_planet_turns += 1;
        }
        for launches in launches_by_planet.values() {
            self.launches_per_launch_sum += f64::from(*launches);
            self.launched_planet_turns += 1;
        }
        self.ships_per_launch_sum += decoded
            .iter()
            .flatten()
            .map(|action| i64::from(action.ships))
            .sum::<i64>();
        self.ships_per_launch_squared_sum += decoded
            .iter()
            .flatten()
            .map(|action| i64::from(action.ships).pow(2))
            .sum::<i64>();
        self.launch_count += launch_count;
        self.turn_count += 1;
        self.max_fleet_size = decoded
            .iter()
            .flatten()
            .map(|action| action.ships)
            .max()
            .unwrap_or(0)
            .max(self.max_fleet_size);
    }

    fn record_step_result(
        &mut self,
        state: &State,
        fleet_losses: FleetLossStats,
        max_fleets: usize,
    ) {
        self.fleet_losses.fleets_in_sun += fleet_losses.fleets_in_sun;
        self.fleet_losses.fleets_out_of_bounds += fleet_losses.fleets_out_of_bounds;
        self.fleet_losses.ships_in_sun += fleet_losses.ships_in_sun;
        self.fleet_losses.ships_out_of_bounds += fleet_losses.ships_out_of_bounds;
        if state.fleets.len() > max_fleets {
            self.max_entities_exceeded_turns += 1;
        }
    }

    fn record_planets_captured(&mut self, planets_captured: u32) {
        self.planets_captured += planets_captured;
    }

    fn terminal_metrics(
        &self,
        state: &State,
        player_map: &PlayerMap,
        player_results: &[PlayerResult],
    ) -> TerminalEpisodeMetrics {
        let fleets_lost = self.fleet_losses.fleets_in_sun + self.fleet_losses.fleets_out_of_bounds;
        let ships_lost = self.fleet_losses.ships_in_sun + self.fleet_losses.ships_out_of_bounds;
        let occupancy_key = if state.config.player_count == 2 {
            "total_planet_occupancy_rate_2p"
        } else {
            "total_planet_occupancy_rate_4p"
        };
        let mut values = vec![
            (
                "max_entities_exceeded_per_game",
                f64::from(self.max_entities_exceeded_turns),
            ),
            ("mean_game_length", f64::from(state.step)),
            ("full_length_rate", full_length_value(state)),
            ("terminal_ship_count", terminal_ship_count(state)),
            ("planets_captured", f64::from(self.planets_captured)),
            ("launches_per_game", f64::from(self.launch_count)),
            (
                "launches_per_turn",
                mean_or_zero(
                    f64::from(self.launch_count) / state.config.player_count as f64,
                    self.turn_count,
                ),
            ),
            ("max_fleet_size", f64::from(self.max_fleet_size)),
            ("fleet_size_std", self.fleet_size_std()),
            (
                "mean_launches_per_planet",
                mean_or_zero(
                    self.launches_per_occupied_planet_sum,
                    self.occupied_planet_turns,
                ),
            ),
            ("mean_ships_lost_per_game", f64::from(ships_lost)),
            (
                "mean_ships_lost_in_sun_per_game",
                f64::from(self.fleet_losses.ships_in_sun),
            ),
            (
                "mean_ships_lost_out_of_bounds_per_game",
                f64::from(self.fleet_losses.ships_out_of_bounds),
            ),
            ("mean_fleets_lost_per_game", f64::from(fleets_lost)),
            (
                "mean_fleets_lost_in_sun_per_game",
                f64::from(self.fleet_losses.fleets_in_sun),
            ),
            (
                "mean_fleets_lost_out_of_bounds_per_game",
                f64::from(self.fleet_losses.fleets_out_of_bounds),
            ),
            (
                occupancy_key,
                mean_or_zero(self.occupancy_rate_sum, self.occupancy_rate_turns),
            ),
        ];
        if self.launched_planet_turns > 0 {
            values.push((
                "mean_launches_per_launch",
                self.launches_per_launch_sum / f64::from(self.launched_planet_turns),
            ));
        }
        if self.launch_count > 0 {
            values.push((
                "mean_ships_per_launch",
                self.ships_per_launch_sum as f64 / f64::from(self.launch_count),
            ));
        }

        let win_rates = player_results
            .iter()
            .enumerate()
            .map(|(internal_player, result)| {
                let won = if matches!(result, PlayerResult::Won) {
                    1.0
                } else {
                    0.0
                };
                (player_map.internal_to_outer(internal_player), won)
            })
            .collect();

        TerminalEpisodeMetrics {
            player_count: state.config.player_count,
            values,
            win_rates,
        }
    }

    fn fleet_size_std(&self) -> f64 {
        if self.launch_count == 0 {
            return 0.0;
        }
        let mean = self.ships_per_launch_sum as f64 / f64::from(self.launch_count);
        let square_mean = self.ships_per_launch_squared_sum as f64 / f64::from(self.launch_count);
        (square_mean - mean.powi(2)).max(0.0).sqrt()
    }
}

fn full_length_value(state: &State) -> f64 {
    if state.step >= state.config.episode_steps.saturating_sub(1) {
        1.0
    } else {
        0.0
    }
}

fn mean_or_zero(sum: f64, count: u32) -> f64 {
    if count == 0 {
        0.0
    } else {
        sum / f64::from(count)
    }
}

fn terminal_ship_count(state: &State) -> f64 {
    let planet_ships = state
        .planets
        .iter()
        .map(|planet| i64::from(planet.ships))
        .sum::<i64>();
    let fleet_ships = state
        .fleets
        .iter()
        .map(|fleet| i64::from(fleet.ships))
        .sum::<i64>();
    (planet_ships + fleet_ships) as f64
}

fn collect_terminal_metrics(
    terminals: Vec<Option<TerminalEpisodeMetrics>>,
) -> HashMap<String, Vec<f64>> {
    let mut metrics = HashMap::<String, Vec<f64>>::new();
    for terminal in terminals.into_iter().flatten() {
        for (key, value) in terminal.values {
            metrics.entry(key.to_string()).or_default().push(value);
        }
        for (player, win_rate) in terminal.win_rates {
            metrics
                .entry(format!("win_rate_player_{player}"))
                .or_default()
                .push(win_rate);
        }
        metrics
            .entry(format!("terminal_episodes_{}p", terminal.player_count))
            .or_default()
            .push(1.0);
    }
    metrics
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

#[allow(clippy::too_many_arguments)]
fn write_one_obs(
    state: &State,
    player_map: &PlayerMap,
    action_slots: &mut ActionEntitySlots,
    player_finished: &[bool],
    max_fleets: usize,
    min_fleet_size: i64,
    planet_obs: &mut [f32],
    fleet_obs: &mut [f32],
    comet_obs: &mut [f32],
    planet_mask: &mut [bool],
    fleet_mask: &mut [bool],
    comet_mask: &mut [bool],
    still_playing: &mut [bool],
    global_obs: &mut [f32],
    can_act: &mut [bool],
    max_launch: &mut [i64],
) -> usize {
    write_still_playing(state, player_map, player_finished, still_playing);
    encode_state_with_action_slots(
        state,
        player_map,
        max_fleets,
        planet_obs,
        fleet_obs,
        comet_obs,
        planet_mask,
        fleet_mask,
        comet_mask,
        global_obs,
        can_act,
        max_launch,
        action_slots,
        min_fleet_size,
    )
}

#[allow(clippy::too_many_arguments)]
fn step_one_env(
    state: &mut State,
    player_map: &mut PlayerMap,
    player_finished: &mut [bool],
    episode_stats: &mut EpisodeStats,
    decoded: &[PlayerAction],
    reward_chunk: &mut [f32],
    done_chunk: &mut [bool],
    max_fleets: usize,
    two_player_weight: f64,
) -> Option<TerminalEpisodeMetrics> {
    episode_stats.record_turn(state, decoded);
    let result = step(state, decoded);
    episode_stats.record_step_result(state, result.fleet_losses, max_fleets);
    episode_stats.record_planets_captured(result.planets_captured);
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
        let terminal_metrics =
            episode_stats.terminal_metrics(state, player_map, &result.player_results);
        let (new_state, new_player_map) = reset_one_env(two_player_weight);
        *state = new_state;
        *player_map = new_player_map;
        player_finished.fill(false);
        *episode_stats = EpisodeStats::default();
        return Some(terminal_metrics);
    }
    None
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
    use crate::rules_engine::state::{LaunchAction, Planet, SimConfig};

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
            initial_planets: planets.clone().into(),
            planets: planets.into(),
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
            initial_planets: planets.clone().into(),
            planets: planets.into(),
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
            initial_planets: planets.clone().into(),
            planets: planets.into(),
            fleets: Vec::new(),
            next_fleet_id: 0,
            comets: Vec::new(),
            comet_planet_ids: Vec::new(),
        }
    }

    fn step_one_env_for_test(
        state: &mut State,
        player_map: &mut PlayerMap,
        player_finished: &mut [bool],
        decoded: &[PlayerAction],
        reward_chunk: &mut [f32],
        done_chunk: &mut [bool],
    ) -> Option<TerminalEpisodeMetrics> {
        let mut episode_stats = EpisodeStats::default();
        step_one_env(
            state,
            player_map,
            player_finished,
            &mut episode_stats,
            decoded,
            reward_chunk,
            done_chunk,
            usize::MAX,
            0.0,
        )
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

        step_one_env_for_test(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
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

        step_one_env_for_test(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
        );

        assert_eq!(rewards, vec![0.0, 0.0, 0.0, -1.0]);
        assert_eq!(dones, vec![false, false, false, true]);
        assert_eq!(finished, vec![false, false, false, true]);

        step_one_env_for_test(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
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

        let terminal = step_one_env_for_test(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
        )
        .expect("terminal step should return episode metrics");
        let metrics = collect_terminal_metrics(vec![Some(terminal)]);

        assert_eq!(rewards, vec![-0.5; 4]);
        assert_eq!(dones, vec![true; 4]);
        assert_eq!(metrics["mean_game_length"], vec![499.0]);
        assert_eq!(metrics["full_length_rate"], vec![1.0]);
        assert_eq!(metrics["terminal_ship_count"], vec![44.0]);
        assert_eq!(metrics["launches_per_game"], vec![0.0]);
        assert_eq!(metrics["win_rate_player_0"], vec![1.0]);
        assert_eq!(metrics["win_rate_player_3"], vec![1.0]);
        assert_eq!(metrics["total_planet_occupancy_rate_4p"], vec![1.0]);
        assert_eq!(metrics["mean_fleets_lost_per_game"], vec![0.0]);
    }

    #[test]
    fn terminal_win_rates_use_remapped_outer_player_slots() {
        let actions = vec![Vec::new(); 2];
        let mut state = state_with_all_players_alive();
        state.config = SimConfig::new(2);
        state.planets.retain(|planet| planet.owner <= 1);
        state.planets[0].ships = 20;
        state.step = state.config.episode_steps.saturating_sub(2);
        let mut player_map = PlayerMap::from_outer_slots(2, [3, 1, 0, 2]);
        let mut finished = vec![false; 4];
        let mut rewards = vec![99.0; 4];
        let mut dones = vec![false; 4];

        let terminal = step_one_env_for_test(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
        )
        .expect("terminal step should return episode metrics");
        let metrics = collect_terminal_metrics(vec![Some(terminal)]);

        assert_eq!(rewards, vec![0.0, -1.0, 0.0, 1.0]);
        assert_eq!(metrics["win_rate_player_3"], vec![1.0]);
        assert_eq!(metrics["win_rate_player_1"], vec![0.0]);
        assert!(!metrics.contains_key("win_rate_player_0"));
    }

    #[test]
    fn episode_stats_reports_mean_ships_per_launch() {
        let state = state_with_all_players_alive();
        let player_map = PlayerMap::identity();
        let actions = vec![
            vec![
                LaunchAction {
                    from_planet_id: 0,
                    angle: 0.0,
                    ships: 3,
                },
                LaunchAction {
                    from_planet_id: 0,
                    angle: 0.0,
                    ships: 5,
                },
            ],
            vec![],
            vec![],
            vec![],
        ];
        let mut episode_stats = EpisodeStats::default();

        episode_stats.record_turn(&state, &actions);
        let metrics = episode_stats.terminal_metrics(
            &state,
            &player_map,
            &[
                PlayerResult::Won,
                PlayerResult::Lost,
                PlayerResult::Lost,
                PlayerResult::Lost,
            ],
        );
        let collected = collect_terminal_metrics(vec![Some(metrics)]);

        assert_eq!(collected["mean_ships_per_launch"], vec![4.0]);
        assert_eq!(collected["launches_per_game"], vec![2.0]);
        assert_eq!(collected["launches_per_turn"], vec![0.5]);
        assert_eq!(collected["max_fleet_size"], vec![5.0]);
        assert_eq!(collected["fleet_size_std"], vec![1.0]);
        assert_eq!(collected["planets_captured"], vec![0.0]);
        assert!(!collected.contains_key("mean_fleets_per_launch"));
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

        step_one_env_for_test(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
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

        step_one_env_for_test(
            &mut state,
            &mut player_map,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
        );

        let mut still_playing = vec![false; 4];
        write_still_playing(&state, &player_map, &finished, &mut still_playing);

        assert_eq!(dones, vec![true; 4]);
        assert_eq!(still_playing, vec![true; 4]);
    }
}
