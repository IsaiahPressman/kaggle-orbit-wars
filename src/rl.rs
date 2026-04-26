use std::ffi::CString;

use std::collections::{HashMap, HashSet};

use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4,
    PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::PyErr;
use rayon::prelude::*;

use crate::rules_engine::env::{is_game_terminated, player_alive_flags, reset, step, PlayerAction};
use crate::rules_engine::state::{
    CometGroup, Fleet, Planet, PlayerResult, Point, ResetConfig, SimConfig, State, BOARD_SIZE,
    COMET_SPAWN_STEPS,
};

pub const MAX_PLANETS: usize = 64;
pub const MAX_COMETS: usize = 4;
pub const MAX_COMET_PATH_LENGTH: usize = 40;
pub const DEFAULT_MAX_ENTITIES: usize = 512;
pub const PLANET_CHANNELS: usize = 16;
pub const FLEET_CHANNELS: usize = 10;
pub const COMET_CHANNELS: usize = OWNER_CHANNELS_WITH_NEUTRAL + 2 + MAX_COMET_PATH_LENGTH * 2;
pub const GLOBAL_CHANNELS: usize = 5;
type ObsShapes = (
    (usize, usize, usize),
    (usize, usize, usize),
    (usize, usize, usize),
    (usize, usize),
    (usize, usize),
    (usize, usize),
    (usize, usize),
);
type EncodedObsV1<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<f32>>,
);

const OWNER_CHANNELS_WITH_NEUTRAL: usize = 5;
const OWNER_CHANNELS: usize = 4;
const PRODUCTION_CHANNELS: usize = 5;
const SHIP_NORMALIZER: f32 = 200.0;
const LOG_SHIP_NORMALIZER: f32 = 10.0;
const MIN_ANGULAR_VELOCITY: f32 = 0.025;
const ANGULAR_VELOCITY_SPAN: f32 = 0.025;

#[pyclass(name = "RlVecEnv")]
pub struct PyRlVecEnv {
    n_envs: usize,
    n_players: usize,
    max_entities: usize,
    max_fleets: usize,
    action_dim: usize,
    reset_config: ResetConfig,
    states: Vec<State>,
    player_finished: Vec<Vec<bool>>,
}

#[pymethods]
impl PyRlVecEnv {
    #[new]
    #[pyo3(signature = (n_envs, n_players, obs_spec="obs_v1", max_entities=DEFAULT_MAX_ENTITIES, action_dim=0))]
    fn new(
        n_envs: usize,
        n_players: usize,
        obs_spec: &str,
        max_entities: usize,
        action_dim: usize,
    ) -> PyResult<Self> {
        if n_envs == 0 {
            return Err(PyValueError::new_err("n_envs must be positive"));
        }
        if n_players != 2 && n_players != 4 {
            return Err(PyValueError::new_err(
                "Orbit Wars supports exactly 2 or 4 players",
            ));
        }
        if obs_spec != "obs_v1" {
            return Err(PyValueError::new_err(format!(
                "unsupported obs_spec {obs_spec:?}; expected \"obs_v1\""
            )));
        }
        if max_entities <= MAX_PLANETS + MAX_COMETS {
            return Err(PyValueError::new_err(format!(
                "max_entities must be greater than MAX_PLANETS + MAX_COMETS ({})",
                MAX_PLANETS + MAX_COMETS
            )));
        }

        let reset_config = ResetConfig::new(n_players);
        let states = (0..n_envs)
            .map(|_| reset(reset_config.clone()))
            .collect::<Vec<_>>();

        Ok(Self {
            n_envs,
            n_players,
            max_entities,
            max_fleets: max_entities - (MAX_PLANETS + MAX_COMETS),
            action_dim,
            reset_config,
            states,
            player_finished: vec![vec![false; n_players]; n_envs],
        })
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.n_envs
    }

    #[getter]
    fn n_players(&self) -> usize {
        self.n_players
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
    fn action_dim(&self) -> usize {
        self.action_dim
    }

    #[allow(clippy::too_many_arguments)]
    fn reset(
        &mut self,
        py: Python<'_>,
        planet_obs: PyReadwriteArrayDyn<'_, f32>,
        fleet_obs: PyReadwriteArrayDyn<'_, f32>,
        comet_obs: PyReadwriteArrayDyn<'_, f32>,
        planet_mask: PyReadwriteArrayDyn<'_, bool>,
        fleet_mask: PyReadwriteArrayDyn<'_, bool>,
        comet_mask: PyReadwriteArrayDyn<'_, bool>,
        global_obs: PyReadwriteArrayDyn<'_, f32>,
    ) -> PyResult<()> {
        self.states
            .par_iter_mut()
            .zip_eq(self.player_finished.par_iter_mut())
            .for_each(|(state, player_finished)| {
                *state = reset(self.reset_config.clone());
                player_finished.fill(false);
            });
        self.write_obs(
            py,
            planet_obs,
            fleet_obs,
            comet_obs,
            planet_mask,
            fleet_mask,
            comet_mask,
            global_obs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArrayDyn<'_, f32>,
        planet_obs: PyReadwriteArrayDyn<'_, f32>,
        fleet_obs: PyReadwriteArrayDyn<'_, f32>,
        comet_obs: PyReadwriteArrayDyn<'_, f32>,
        planet_mask: PyReadwriteArrayDyn<'_, bool>,
        fleet_mask: PyReadwriteArrayDyn<'_, bool>,
        comet_mask: PyReadwriteArrayDyn<'_, bool>,
        global_obs: PyReadwriteArrayDyn<'_, f32>,
        rewards: PyReadwriteArrayDyn<'_, f32>,
        dones: PyReadwriteArrayDyn<'_, bool>,
    ) -> PyResult<()> {
        let action_shape = [self.n_envs, self.n_players, self.action_dim];
        require_shape("actions", actions.shape(), &action_shape)?;
        require_shape("rewards", rewards.shape(), &[self.n_envs, self.n_players])?;
        require_shape("dones", dones.shape(), &[self.n_envs, self.n_players])?;

        let mut rewards = rewards;
        let mut dones = dones;
        let reward_chunks = rewards.as_slice_mut()?.par_chunks_mut(self.n_players);
        let done_chunks = dones.as_slice_mut()?.par_chunks_mut(self.n_players);

        if self.action_dim == 0 {
            self.states
                .par_iter_mut()
                .zip_eq(self.player_finished.par_iter_mut())
                .zip_eq(reward_chunks)
                .zip_eq(done_chunks)
                .for_each(|(((state, player_finished), reward_chunk), done_chunk)| {
                    let decoded = vec![Vec::new(); self.n_players];
                    step_one_env(
                        state,
                        player_finished,
                        &decoded,
                        reward_chunk,
                        done_chunk,
                        &self.reset_config,
                    );
                });
        } else {
            let actions = actions.as_slice()?;
            let action_chunks = actions.par_chunks(self.n_players * self.action_dim);
            self.states
                .par_iter_mut()
                .zip_eq(self.player_finished.par_iter_mut())
                .zip_eq(action_chunks)
                .zip_eq(reward_chunks)
                .zip_eq(done_chunks)
                .for_each(
                    |((((state, player_finished), action_chunk), reward_chunk), done_chunk)| {
                        let decoded = decode_actions(action_chunk, self.n_players, self.action_dim);
                        step_one_env(
                            state,
                            player_finished,
                            &decoded,
                            reward_chunk,
                            done_chunk,
                            &self.reset_config,
                        );
                    },
                );
        }

        self.write_obs(
            py,
            planet_obs,
            fleet_obs,
            comet_obs,
            planet_mask,
            fleet_mask,
            comet_mask,
            global_obs,
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
            (self.n_envs, GLOBAL_CHANNELS),
        )
    }
}

fn step_one_env(
    state: &mut State,
    player_finished: &mut [bool],
    decoded: &[PlayerAction],
    reward_chunk: &mut [f32],
    done_chunk: &mut [bool],
    reset_config: &ResetConfig,
) {
    let result = step(state, decoded);
    let should_reset = is_game_terminated(state);
    let alive = player_alive_flags(state);

    for (player_index, result) in result.player_results.iter().enumerate() {
        let (reward, done) =
            player_reward_done(*result, alive[player_index], player_finished[player_index]);
        reward_chunk[player_index] = reward;
        done_chunk[player_index] = done;
        if done {
            player_finished[player_index] = true;
        }
    }

    if should_reset {
        *state = reset(reset_config.clone());
        player_finished.fill(false);
    }
}

impl PyRlVecEnv {
    #[allow(clippy::too_many_arguments)]
    fn write_obs(
        &self,
        py: Python<'_>,
        planet_obs: PyReadwriteArrayDyn<'_, f32>,
        fleet_obs: PyReadwriteArrayDyn<'_, f32>,
        comet_obs: PyReadwriteArrayDyn<'_, f32>,
        planet_mask: PyReadwriteArrayDyn<'_, bool>,
        fleet_mask: PyReadwriteArrayDyn<'_, bool>,
        comet_mask: PyReadwriteArrayDyn<'_, bool>,
        global_obs: PyReadwriteArrayDyn<'_, f32>,
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
            "global_obs",
            global_obs.shape(),
            &[self.n_envs, GLOBAL_CHANNELS],
        )?;

        let mut planet_obs = planet_obs;
        let mut fleet_obs = fleet_obs;
        let mut comet_obs = comet_obs;
        let mut planet_mask = planet_mask;
        let mut fleet_mask = fleet_mask;
        let mut comet_mask = comet_mask;
        let mut global_obs = global_obs;

        let planets_per_env = MAX_PLANETS * PLANET_CHANNELS;
        let fleets_per_env = self.max_fleets * FLEET_CHANNELS;
        let comets_per_env = MAX_COMETS * COMET_CHANNELS;
        let planet_masks_per_env = MAX_PLANETS;
        let fleet_masks_per_env = self.max_fleets;
        let comet_masks_per_env = MAX_COMETS;
        let globals_per_env = GLOBAL_CHANNELS;

        let ignored_fleets: usize = self
            .states
            .par_iter()
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
            .zip_eq(global_obs.as_slice_mut()?.par_chunks_mut(globals_per_env))
            .map(
                |(
                    (
                        (((((state, planet_obs), fleet_obs), comet_obs), planet_mask), fleet_mask),
                        comet_mask,
                    ),
                    global_obs,
                )| {
                    encode_state(
                        state,
                        self.max_fleets,
                        planet_obs,
                        fleet_obs,
                        comet_obs,
                        planet_mask,
                        fleet_mask,
                        comet_mask,
                        global_obs,
                    )
                },
            )
            .sum();

        warn_ignored_fleets(py, ignored_fleets)
    }
}

fn decode_actions(action_chunk: &[f32], n_players: usize, action_dim: usize) -> Vec<PlayerAction> {
    if action_dim == 0 {
        return vec![Vec::new(); n_players];
    }
    decode_nonempty_actions(action_chunk, n_players, action_dim)
}

fn decode_nonempty_actions(
    _action_chunk: &[f32],
    _n_players: usize,
    _action_dim: usize,
) -> Vec<PlayerAction> {
    unimplemented!("action_spec decoding is not implemented yet")
}

fn player_reward_done(result: PlayerResult, alive: bool, previously_finished: bool) -> (f32, bool) {
    if previously_finished {
        return (0.0, true);
    }
    match result {
        PlayerResult::NotDone if alive => (0.0, false),
        PlayerResult::NotDone => (-1.0, true),
        PlayerResult::Loss => (-1.0, true),
        PlayerResult::Win => (1.0, true),
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_state(
    state: &State,
    max_fleets: usize,
    planet_obs: &mut [f32],
    fleet_obs: &mut [f32],
    comet_obs: &mut [f32],
    planet_mask: &mut [bool],
    fleet_mask: &mut [bool],
    comet_mask: &mut [bool],
    global_obs: &mut [f32],
) -> usize {
    let comet_ids = state
        .comet_planet_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let non_comet_planets = state
        .planets
        .iter()
        .filter(|planet| !comet_ids.contains(&planet.id))
        .collect::<Vec<_>>();
    assert!(
        non_comet_planets.len() <= MAX_PLANETS,
        "max_planets exceeded: {} planets present, max is {MAX_PLANETS}",
        non_comet_planets.len()
    );

    planet_obs.fill(0.0);
    fleet_obs.fill(0.0);
    comet_obs.fill(0.0);
    planet_mask.fill(false);
    fleet_mask.fill(false);
    comet_mask.fill(false);
    global_obs.fill(0.0);

    let mut fleets = state.fleets.iter().collect::<Vec<_>>();
    fleets.sort_by(|left, right| right.ships.cmp(&left.ships).then(left.id.cmp(&right.id)));
    let ignored_fleets = fleets.len().saturating_sub(max_fleets);

    for (planet_index, planet) in non_comet_planets.iter().enumerate() {
        planet_mask[planet_index] = true;
        let row_start = planet_index * PLANET_CHANNELS;
        let row = &mut planet_obs[row_start..row_start + PLANET_CHANNELS];

        let owner_index = if planet.owner == -1 {
            4
        } else {
            planet.owner as usize
        };
        row[owner_index] = 1.0;
        row[OWNER_CHANNELS_WITH_NEUTRAL] = normalize_position(planet.x);
        row[OWNER_CHANNELS_WITH_NEUTRAL + 1] = normalize_position(planet.y);

        let production = planet.production.clamp(1, PRODUCTION_CHANNELS as i32) as usize - 1;
        row[OWNER_CHANNELS_WITH_NEUTRAL + 2 + production] = 1.0;

        row[12] = (planet.radius / 10.0) as f32;
        row[13] = normalize_ships(planet.ships);
        row[14] = normalize_log_ships(planet.ships);
        row[15] = normalize_angular_velocity(state.angular_velocity);
    }

    for (fleet_index, fleet) in fleets.iter().take(max_fleets).enumerate() {
        fleet_mask[fleet_index] = true;
        let row_start = fleet_index * FLEET_CHANNELS;
        let row = &mut fleet_obs[row_start..row_start + FLEET_CHANNELS];

        row[fleet.owner as usize] = 1.0;
        row[OWNER_CHANNELS] = normalize_position(fleet.x);
        row[OWNER_CHANNELS + 1] = normalize_position(fleet.y);
        row[OWNER_CHANNELS + 2] = fleet.angle.sin() as f32;
        row[OWNER_CHANNELS + 3] = fleet.angle.cos() as f32;
        row[OWNER_CHANNELS + 4] = normalize_ships(fleet.ships);
        row[OWNER_CHANNELS + 5] = normalize_log_ships(fleet.ships);
    }

    encode_comets(state, comet_obs, comet_mask);
    encode_global(state, global_obs);

    ignored_fleets
}

fn normalize_angular_velocity(angular_velocity: f64) -> f32 {
    ((angular_velocity as f32) - MIN_ANGULAR_VELOCITY) / ANGULAR_VELOCITY_SPAN
}

fn normalize_position(value: f64) -> f32 {
    ((value / BOARD_SIZE) * 2.0 - 1.0) as f32
}

fn encode_comets(state: &State, comet_obs: &mut [f32], comet_mask: &mut [bool]) {
    let planets_by_id = state
        .planets
        .iter()
        .map(|planet| (planet.id, planet))
        .collect::<HashMap<_, _>>();

    let mut comet_index = 0;
    for group in &state.comets {
        for (path_offset, planet_id) in group.planet_ids.iter().enumerate() {
            if comet_index >= MAX_COMETS {
                return;
            }
            let Some(planet) = planets_by_id.get(planet_id) else {
                continue;
            };
            let Some(path) = group.paths.get(path_offset) else {
                continue;
            };

            comet_mask[comet_index] = true;
            let row_start = comet_index * COMET_CHANNELS;
            let row = &mut comet_obs[row_start..row_start + COMET_CHANNELS];

            let owner_index = if planet.owner == -1 {
                4
            } else {
                planet.owner as usize
            };
            row[owner_index] = 1.0;
            row[OWNER_CHANNELS_WITH_NEUTRAL] = normalize_ships(planet.ships);
            row[OWNER_CHANNELS_WITH_NEUTRAL + 1] = normalize_log_ships(planet.ships);

            let path_start = group.path_index.max(0) as usize;
            let path_values_start = OWNER_CHANNELS_WITH_NEUTRAL + 2;
            for (future_index, point) in path
                .iter()
                .skip(path_start)
                .take(MAX_COMET_PATH_LENGTH)
                .enumerate()
            {
                let value_start = path_values_start + future_index * 2;
                row[value_start] = normalize_position(point.x);
                row[value_start + 1] = normalize_position(point.y);
            }

            comet_index += 1;
        }
    }
}

fn encode_global(state: &State, global_obs: &mut [f32]) {
    global_obs[0] = state.step as f32 / state.config.episode_steps as f32;
    global_obs[1] = steps_until_next_comet_spawn(state.step) as f32 / 100.0;
    global_obs[2] = normalize_angular_velocity(state.angular_velocity);
    global_obs[3] = active_comet_count(state) as f32 / MAX_COMETS as f32;
    global_obs[4] = player_alive_flags(state)
        .into_iter()
        .filter(|alive| *alive)
        .count() as f32
        / state.config.player_count as f32;
}

fn steps_until_next_comet_spawn(step: u32) -> u32 {
    COMET_SPAWN_STEPS
        .iter()
        .copied()
        .find(|spawn_step| *spawn_step > step)
        .map_or(0, |spawn_step| spawn_step - step)
}

fn active_comet_count(state: &State) -> usize {
    state
        .comets
        .iter()
        .map(|group| group.planet_ids.len())
        .sum()
}

fn normalize_ships(ships: i32) -> f32 {
    ships as f32 / SHIP_NORMALIZER
}

fn normalize_log_ships(ships: i32) -> f32 {
    ((ships.max(0) as f32) + 1.0).ln() / LOG_SHIP_NORMALIZER
}

fn require_shape(name: &str, actual: &[usize], expected: &[usize]) -> PyResult<()> {
    if actual == expected {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{name} must have shape {expected:?}, got {actual:?}"
    )))
}

fn require_shape_suffix(name: &str, actual: &[usize], expected_last_dim: usize) -> PyResult<()> {
    if actual.len() == 2 && actual[1] == expected_last_dim {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{name} must have shape (n, {expected_last_dim}), got {actual:?}"
    )))
}

fn warn_ignored_fleets(py: Python<'_>, ignored_fleets: usize) -> PyResult<()> {
    if ignored_fleets == 0 {
        return Ok(());
    }
    let message = CString::new(format!(
        "max_entities exceeded: {ignored_fleets} fleets ignored"
    ))
    .expect("warning message does not contain nul bytes");
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), &message, 0)
}

#[allow(clippy::too_many_arguments)]
fn state_from_arrays(
    planets: PyReadonlyArray2<'_, f64>,
    fleets: PyReadonlyArray2<'_, f64>,
    comet_planet_ids: PyReadonlyArray2<'_, f64>,
    comet_path_indices: PyReadonlyArray1<'_, f64>,
    comet_path_lengths: PyReadonlyArray2<'_, f64>,
    comet_paths: PyReadonlyArray4<'_, f64>,
    angular_velocity: f64,
    step: u32,
    episode_steps: u32,
) -> PyResult<State> {
    let planet_rows = planets.as_array();
    let fleet_rows = fleets.as_array();
    let comet_planet_id_rows = comet_planet_ids.as_array();
    let comet_path_index_rows = comet_path_indices.as_array();
    let comet_path_length_rows = comet_path_lengths.as_array();
    let comet_path_rows = comet_paths.as_array();

    let planets = planet_rows
        .rows()
        .into_iter()
        .map(|row| {
            Ok(Planet {
                id: finite_u32(row[0], "planet id")?,
                owner: finite_i32(row[1], "planet owner")?,
                x: finite_f64(row[2], "planet x")?,
                y: finite_f64(row[3], "planet y")?,
                radius: finite_f64(row[4], "planet radius")?,
                ships: finite_i32(row[5], "planet ships")?,
                production: finite_i32(row[6], "planet production")?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let fleets = fleet_rows
        .rows()
        .into_iter()
        .map(|row| {
            Ok(Fleet {
                id: finite_u32(row[0], "fleet id")?,
                owner: finite_i32(row[1], "fleet owner")?,
                x: finite_f64(row[2], "fleet x")?,
                y: finite_f64(row[3], "fleet y")?,
                angle: finite_f64(row[4], "fleet angle")?,
                from_planet_id: finite_u32(row[5], "fleet from_planet_id")?,
                ships: finite_i32(row[6], "fleet ships")?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    if comet_planet_id_rows.shape()[0] != comet_path_index_rows.shape()[0]
        || comet_planet_id_rows.shape()[0] != comet_path_length_rows.shape()[0]
        || comet_planet_id_rows.shape()[0] != comet_path_rows.shape()[0]
    {
        return Err(PyValueError::new_err(
            "comet arrays must agree on the comet group dimension",
        ));
    }

    let mut comets = Vec::with_capacity(comet_planet_id_rows.shape()[0]);
    let mut flattened_comet_planet_ids = Vec::new();
    for group_index in 0..comet_planet_id_rows.shape()[0] {
        let mut group = CometGroup {
            planet_ids: Vec::new(),
            paths: Vec::new(),
            path_index: finite_i32(comet_path_index_rows[group_index], "comet path_index")?,
        };
        for path_offset in 0..MAX_COMETS {
            let planet_id = comet_planet_id_rows[[group_index, path_offset]];
            if planet_id < 0.0 {
                continue;
            }
            let path_len = finite_usize(
                comet_path_length_rows[[group_index, path_offset]],
                "comet path length",
            )?;
            group
                .planet_ids
                .push(finite_u32(planet_id, "comet planet_id")?);
            flattened_comet_planet_ids.push(finite_u32(planet_id, "comet planet_id")?);
            let mut path = Vec::with_capacity(path_len);
            for path_index in 0..path_len {
                path.push(Point::new(
                    finite_f64(
                        comet_path_rows[[group_index, path_offset, path_index, 0]],
                        "comet path x",
                    )?,
                    finite_f64(
                        comet_path_rows[[group_index, path_offset, path_index, 1]],
                        "comet path y",
                    )?,
                ));
            }
            group.paths.push(path);
        }
        if !group.planet_ids.is_empty() {
            comets.push(group);
        }
    }

    let mut config = SimConfig::new(4);
    config.episode_steps = episode_steps;

    Ok(State {
        config,
        step,
        angular_velocity,
        initial_planets: planets.clone(),
        planets,
        fleets,
        next_fleet_id: 0,
        comets,
        comet_planet_ids: flattened_comet_planet_ids,
    })
}

fn finite_f64(value: f64, name: &str) -> PyResult<f64> {
    if value.is_finite() {
        return Ok(value);
    }
    Err(PyValueError::new_err(format!("{name} must be finite")))
}

fn finite_i32(value: f64, name: &str) -> PyResult<i32> {
    finite_f64(value, name)?;
    Ok(value as i32)
}

fn finite_u32(value: f64, name: &str) -> PyResult<u32> {
    finite_f64(value, name)?;
    if value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be non-negative"
        )));
    }
    Ok(value as u32)
}

fn finite_usize(value: f64, name: &str) -> PyResult<usize> {
    finite_f64(value, name)?;
    if value < 0.0 || value > MAX_COMET_PATH_LENGTH as f64 {
        return Err(PyValueError::new_err(format!(
            "{name} must be between 0 and {MAX_COMET_PATH_LENGTH}"
        )));
    }
    Ok(value as usize)
}

#[pyfunction]
pub fn rl_obs_constants() -> (usize, usize, usize, usize, usize, usize, usize, usize) {
    (
        MAX_PLANETS,
        MAX_COMETS,
        MAX_COMET_PATH_LENGTH,
        DEFAULT_MAX_ENTITIES,
        PLANET_CHANNELS,
        FLEET_CHANNELS,
        COMET_CHANNELS,
        GLOBAL_CHANNELS,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    planets,
    fleets,
    comet_planet_ids,
    comet_path_indices,
    comet_path_lengths,
    comet_paths,
    angular_velocity,
    step=0,
    episode_steps=500,
    max_entities=DEFAULT_MAX_ENTITIES
))]
pub fn encode_obs_v1<'py>(
    py: Python<'py>,
    planets: PyReadonlyArray2<'py, f64>,
    fleets: PyReadonlyArray2<'py, f64>,
    comet_planet_ids: PyReadonlyArray2<'py, f64>,
    comet_path_indices: PyReadonlyArray1<'py, f64>,
    comet_path_lengths: PyReadonlyArray2<'py, f64>,
    comet_paths: PyReadonlyArray4<'py, f64>,
    angular_velocity: f64,
    step: u32,
    episode_steps: u32,
    max_entities: usize,
) -> PyResult<EncodedObsV1<'py>> {
    if max_entities <= MAX_PLANETS + MAX_COMETS {
        return Err(PyValueError::new_err(format!(
            "max_entities must be greater than MAX_PLANETS + MAX_COMETS ({})",
            MAX_PLANETS + MAX_COMETS
        )));
    }
    require_shape_suffix("planets", planets.shape(), 7)?;
    require_shape_suffix("fleets", fleets.shape(), 7)?;
    let comet_group_count = comet_planet_ids.shape()[0];
    require_shape(
        "comet_planet_ids",
        comet_planet_ids.shape(),
        &[comet_group_count, MAX_COMETS],
    )?;
    require_shape(
        "comet_path_indices",
        comet_path_indices.shape(),
        &[comet_group_count],
    )?;
    require_shape(
        "comet_path_lengths",
        comet_path_lengths.shape(),
        &[comet_group_count, MAX_COMETS],
    )?;
    require_shape(
        "comet_paths",
        comet_paths.shape(),
        &[comet_group_count, MAX_COMETS, MAX_COMET_PATH_LENGTH, 2],
    )?;

    let state = state_from_arrays(
        planets,
        fleets,
        comet_planet_ids,
        comet_path_indices,
        comet_path_lengths,
        comet_paths,
        angular_velocity,
        step,
        episode_steps,
    )?;
    let max_fleets = max_entities - (MAX_PLANETS + MAX_COMETS);
    let mut planet_obs = Array2::<f32>::zeros((MAX_PLANETS, PLANET_CHANNELS));
    let mut fleet_obs = Array2::<f32>::zeros((max_fleets, FLEET_CHANNELS));
    let mut comet_obs = Array2::<f32>::zeros((MAX_COMETS, COMET_CHANNELS));
    let mut planet_mask = Array1::<bool>::from_elem(MAX_PLANETS, false);
    let mut fleet_mask = Array1::<bool>::from_elem(max_fleets, false);
    let mut comet_mask = Array1::<bool>::from_elem(MAX_COMETS, false);
    let mut global_obs = Array1::<f32>::zeros(GLOBAL_CHANNELS);

    let ignored_fleets = encode_state(
        &state,
        max_fleets,
        planet_obs
            .as_slice_mut()
            .expect("newly allocated planet array is contiguous"),
        fleet_obs
            .as_slice_mut()
            .expect("newly allocated fleet array is contiguous"),
        comet_obs
            .as_slice_mut()
            .expect("newly allocated comet array is contiguous"),
        planet_mask
            .as_slice_mut()
            .expect("newly allocated planet mask is contiguous"),
        fleet_mask
            .as_slice_mut()
            .expect("newly allocated fleet mask is contiguous"),
        comet_mask
            .as_slice_mut()
            .expect("newly allocated comet mask is contiguous"),
        global_obs
            .as_slice_mut()
            .expect("newly allocated global array is contiguous"),
    );
    warn_ignored_fleets(py, ignored_fleets)?;

    Ok((
        planet_obs.into_pyarray(py),
        fleet_obs.into_pyarray(py),
        comet_obs.into_pyarray(py),
        planet_mask.into_pyarray(py),
        fleet_mask.into_pyarray(py),
        comet_mask.into_pyarray(py),
        global_obs.into_pyarray(py),
    ))
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRlVecEnv>()?;
    m.add_function(wrap_pyfunction!(rl_obs_constants, m)?)?;
    m.add_function(wrap_pyfunction!(encode_obs_v1, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn eliminated_player_gets_one_loss_reward_then_sticky_done() {
        let reset_config = ResetConfig::new(4);
        let actions = vec![Vec::new(); 4];
        let mut state = state_with_player_three_eliminated();
        let mut finished = vec![false; 4];
        let mut rewards = vec![99.0; 4];
        let mut dones = vec![false; 4];

        step_one_env(
            &mut state,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
            &reset_config,
        );

        assert_eq!(rewards, vec![0.0, 0.0, 0.0, -1.0]);
        assert_eq!(dones, vec![false, false, false, true]);
        assert_eq!(finished, vec![false, false, false, true]);

        step_one_env(
            &mut state,
            &mut finished,
            &actions,
            &mut rewards,
            &mut dones,
            &reset_config,
        );

        assert_eq!(rewards[3], 0.0);
        assert!(dones[3]);
    }

    #[test]
    fn angular_velocity_normalization_maps_generated_range_to_zero_one() {
        assert_eq!(normalize_angular_velocity(0.025), 0.0);
        assert_eq!(normalize_angular_velocity(0.05), 1.0);
    }
}
