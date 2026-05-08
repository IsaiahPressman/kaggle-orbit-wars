use std::collections::{HashMap, HashSet};

use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::rules_engine::env::{reset, step, PlayerAction};
use crate::rules_engine::state::{
    FleetLossStats, PlayerResult, ResetConfig, State, StaticTargetCache, BOARD_SIZE, CENTER,
    SUN_RADIUS,
};

use super::action_spec::{
    action_entity_slots, decode_discrete_target_actions, decode_pure_actions, ActionEntitySlots,
    RlActionSpec,
};
use super::obs_spec::encode_state_with_action_slots;
use super::{
    log_ignored_fleets, require_shape, PlayerMap, ACTION_ENTITY_SLOTS, COMET_CHANNELS,
    DEFAULT_MAX_ENTITIES, FLEET_CHANNELS, GLOBAL_CHANNELS, MAX_COMETS, MAX_PLANETS,
    OUTER_PLAYER_SLOTS, PLANET_CHANNELS,
};

type ObsShapes = (
    (usize, usize, usize),
    (usize, usize),
    (usize, usize, usize),
    (usize, usize, usize),
    (usize, usize),
    (usize, usize),
    (usize, usize),
    Vec<usize>,
    (usize, usize, usize),
);

#[pyclass(name = "RlVecEnv")]
pub struct PyRlVecEnv {
    n_envs: usize,
    two_player_weight: f64,
    max_entities: usize,
    max_fleets: usize,
    action_spec: RlActionSpec,
    max_per_planet_launches: usize,
    min_fleet_size: i64,
    states: Vec<State>,
    player_maps: Vec<PlayerMap>,
    action_slots: Vec<ActionEntitySlots>,
    player_finished: Vec<Vec<bool>>,
    episode_stats: Vec<EpisodeStats>,
    last_terminal_metrics: Vec<Option<TerminalEpisodeMetrics>>,
    last_terminal_snapshots: Vec<Option<StateSnapshot>>,
}

#[pymethods]
impl PyRlVecEnv {
    #[new]
    #[pyo3(signature = (n_envs, two_player_weight=0.5, obs_spec="entity_based", action_spec="pure", max_entities=DEFAULT_MAX_ENTITIES, max_per_planet_launches=3, min_fleet_size=1))]
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
        if obs_spec != "entity_based" {
            return Err(PyValueError::new_err(format!(
                "unsupported obs_spec {obs_spec:?}; expected \"entity_based\""
            )));
        }
        let Some(action_spec) = RlActionSpec::parse(action_spec) else {
            return Err(PyValueError::new_err(format!(
                "unsupported action_spec {action_spec:?}; expected \"pure\" or \"discrete_targets\""
            )));
        };
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
            action_spec,
            max_per_planet_launches,
            min_fleet_size,
            states,
            player_maps,
            action_slots,
            player_finished: vec![vec![false; OUTER_PLAYER_SLOTS]; n_envs],
            episode_stats: vec![EpisodeStats::default(); n_envs],
            last_terminal_metrics: vec![None; n_envs],
            last_terminal_snapshots: vec![None; n_envs],
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
        orbiting_planet_obs: PyReadwriteArrayDyn<'_, bool>,
        fleet_obs: PyReadwriteArrayDyn<'_, f32>,
        comet_obs: PyReadwriteArrayDyn<'_, f32>,
        entity_mask: PyReadwriteArrayDyn<'_, bool>,
        still_playing: PyReadwriteArrayDyn<'_, bool>,
        global_obs: PyReadwriteArrayDyn<'_, f32>,
        can_act: PyReadwriteArrayDyn<'_, bool>,
        max_launch: PyReadwriteArrayDyn<'_, i64>,
    ) -> PyResult<()> {
        self.require_obs_shapes(
            &planet_obs,
            &orbiting_planet_obs,
            &fleet_obs,
            &comet_obs,
            &entity_mask,
            &still_playing,
            &global_obs,
            &can_act,
            &max_launch,
        )?;

        let mut planet_obs = planet_obs;
        let mut orbiting_planet_obs = orbiting_planet_obs;
        let mut fleet_obs = fleet_obs;
        let mut comet_obs = comet_obs;
        let mut entity_mask = entity_mask;
        let mut still_playing = still_playing;
        let mut global_obs = global_obs;
        let mut can_act = can_act;
        let mut max_launch = max_launch;

        let planets_per_env = MAX_PLANETS * PLANET_CHANNELS;
        let orbiting_planets_per_env = MAX_PLANETS;
        let fleets_per_env = self.max_fleets * FLEET_CHANNELS;
        let comets_per_env = MAX_COMETS * COMET_CHANNELS;
        let action_masks_per_env = self.action_spec.can_act_len();
        let two_player_weight = self.two_player_weight;
        let max_fleets = self.max_fleets;
        let min_fleet_size = self.min_fleet_size;
        let action_spec = self.action_spec;
        let max_launch_per_env = OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS;

        let ignored_fleets: usize = self
            .states
            .par_iter_mut()
            .zip_eq(self.player_maps.par_iter_mut())
            .zip_eq(self.action_slots.par_iter_mut())
            .zip_eq(self.player_finished.par_iter_mut())
            .zip_eq(self.episode_stats.par_iter_mut())
            .zip_eq(planet_obs.as_slice_mut()?.par_chunks_mut(planets_per_env))
            .zip_eq(
                orbiting_planet_obs
                    .as_slice_mut()?
                    .par_chunks_mut(orbiting_planets_per_env),
            )
            .zip_eq(fleet_obs.as_slice_mut()?.par_chunks_mut(fleets_per_env))
            .zip_eq(comet_obs.as_slice_mut()?.par_chunks_mut(comets_per_env))
            .zip_eq(
                entity_mask
                    .as_slice_mut()?
                    .par_chunks_mut(self.max_entities),
            )
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
                    .par_chunks_mut(max_launch_per_env),
            )
            .map(|item| {
                let (item, max_launch) = item;
                let (item, can_act) = item;
                let (item, global_obs) = item;
                let (item, still_playing) = item;
                let (item, entity_mask) = item;
                let (item, comet_obs) = item;
                let (item, fleet_obs) = item;
                let (item, orbiting_planet_obs) = item;
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
                        action_spec,
                        max_fleets,
                        min_fleet_size,
                        planet_obs,
                        orbiting_planet_obs,
                        fleet_obs,
                        comet_obs,
                        entity_mask,
                        still_playing,
                        global_obs,
                        can_act,
                        max_launch,
                    )
                }
            })
            .sum();

        self.last_terminal_metrics.fill(None);
        self.last_terminal_snapshots.fill(None);
        log_ignored_fleets(ignored_fleets);
        Ok(())
    }

    fn state_snapshot<'py>(&self, py: Python<'py>, env_index: usize) -> PyResult<Py<PyAny>> {
        let snapshot = self.current_snapshot(env_index)?;
        snapshot_to_py(py, &snapshot)
    }

    fn terminal_snapshot<'py>(&self, py: Python<'py>, env_index: usize) -> PyResult<Py<PyAny>> {
        if env_index >= self.n_envs {
            return Err(PyValueError::new_err(format!(
                "env_index must be < {}, got {env_index}",
                self.n_envs
            )));
        }
        let Some(snapshot) = &self.last_terminal_snapshots[env_index] else {
            return Ok(py.None());
        };
        snapshot_to_py(py, snapshot)
    }

    fn terminal_metrics<'py>(&self, py: Python<'py>, env_index: usize) -> PyResult<Py<PyAny>> {
        if env_index >= self.n_envs {
            return Err(PyValueError::new_err(format!(
                "env_index must be < {}, got {env_index}",
                self.n_envs
            )));
        }
        let Some(metrics) = &self.last_terminal_metrics[env_index] else {
            return Ok(py.None());
        };
        terminal_metrics_to_py(py, metrics)
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        launch: PyReadonlyArrayDyn<'_, bool>,
        angle: PyReadonlyArrayDyn<'_, f32>,
        ships: PyReadonlyArrayDyn<'_, i64>,
        planet_obs: PyReadwriteArrayDyn<'_, f32>,
        orbiting_planet_obs: PyReadwriteArrayDyn<'_, bool>,
        fleet_obs: PyReadwriteArrayDyn<'_, f32>,
        comet_obs: PyReadwriteArrayDyn<'_, f32>,
        entity_mask: PyReadwriteArrayDyn<'_, bool>,
        still_playing: PyReadwriteArrayDyn<'_, bool>,
        global_obs: PyReadwriteArrayDyn<'_, f32>,
        can_act: PyReadwriteArrayDyn<'_, bool>,
        max_launch: PyReadwriteArrayDyn<'_, i64>,
        rewards: PyReadwriteArrayDyn<'_, f32>,
        dones: PyReadwriteArrayDyn<'_, bool>,
    ) -> PyResult<HashMap<String, Vec<f64>>> {
        if self.action_spec != RlActionSpec::Pure {
            return Err(PyValueError::new_err(
                "step with angle actions requires action_spec \"pure\"",
            ));
        }
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
            &orbiting_planet_obs,
            &fleet_obs,
            &comet_obs,
            &entity_mask,
            &still_playing,
            &global_obs,
            &can_act,
            &max_launch,
        )?;

        let mut rewards = rewards;
        let mut dones = dones;
        let mut planet_obs = planet_obs;
        let mut orbiting_planet_obs = orbiting_planet_obs;
        let mut fleet_obs = fleet_obs;
        let mut comet_obs = comet_obs;
        let mut entity_mask = entity_mask;
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
        let orbiting_planets_per_env = MAX_PLANETS;
        let fleets_per_env = self.max_fleets * FLEET_CHANNELS;
        let comets_per_env = MAX_COMETS * COMET_CHANNELS;
        let action_masks_per_env = self.action_spec.can_act_len();
        let max_launch_per_env = OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS;
        let max_per_planet_launches = self.max_per_planet_launches;
        let min_fleet_size = self.min_fleet_size;
        let max_fleets = self.max_fleets;
        let two_player_weight = self.two_player_weight;
        let action_spec = self.action_spec;

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
            .zip_eq(
                orbiting_planet_obs
                    .as_slice_mut()?
                    .par_chunks_mut(orbiting_planets_per_env),
            )
            .zip_eq(fleet_obs.as_slice_mut()?.par_chunks_mut(fleets_per_env))
            .zip_eq(comet_obs.as_slice_mut()?.par_chunks_mut(comets_per_env))
            .zip_eq(
                entity_mask
                    .as_slice_mut()?
                    .par_chunks_mut(self.max_entities),
            )
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
                    .par_chunks_mut(max_launch_per_env),
            )
            .enumerate()
            .map(|(env_index, item)| {
                let (item, max_launch) = item;
                let (item, can_act) = item;
                let (item, global_obs) = item;
                let (item, still_playing) = item;
                let (item, entity_mask) = item;
                let (item, comet_obs) = item;
                let (item, fleet_obs) = item;
                let (item, orbiting_planet_obs) = item;
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
                        action_spec,
                        max_fleets,
                        min_fleet_size,
                        planet_obs,
                        orbiting_planet_obs,
                        fleet_obs,
                        comet_obs,
                        entity_mask,
                        still_playing,
                        global_obs,
                        can_act,
                        max_launch,
                    );
                    Ok::<_, String>(StepOneOutput {
                        terminal_metrics: terminal.metrics,
                        terminal_snapshot: terminal.snapshot,
                        ignored_fleets,
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let mut terminal_metrics = Vec::with_capacity(env_results.len());
        let mut ignored_fleets = 0;
        for (env_index, result) in env_results.into_iter().enumerate() {
            self.last_terminal_snapshots[env_index] = result.terminal_snapshot;
            self.last_terminal_metrics[env_index] = result.terminal_metrics.clone();
            terminal_metrics.push(result.terminal_metrics);
            let ignored = result.ignored_fleets;
            ignored_fleets += ignored;
        }
        let episode_metrics = collect_terminal_metrics(terminal_metrics);
        log_ignored_fleets(ignored_fleets);
        Ok(episode_metrics)
    }

    #[allow(clippy::too_many_arguments)]
    fn step_discrete_targets(
        &mut self,
        launch: PyReadonlyArrayDyn<'_, bool>,
        target: PyReadonlyArrayDyn<'_, i64>,
        ships: PyReadonlyArrayDyn<'_, i64>,
        planet_obs: PyReadwriteArrayDyn<'_, f32>,
        orbiting_planet_obs: PyReadwriteArrayDyn<'_, bool>,
        fleet_obs: PyReadwriteArrayDyn<'_, f32>,
        comet_obs: PyReadwriteArrayDyn<'_, f32>,
        entity_mask: PyReadwriteArrayDyn<'_, bool>,
        still_playing: PyReadwriteArrayDyn<'_, bool>,
        global_obs: PyReadwriteArrayDyn<'_, f32>,
        can_act: PyReadwriteArrayDyn<'_, bool>,
        max_launch: PyReadwriteArrayDyn<'_, i64>,
        rewards: PyReadwriteArrayDyn<'_, f32>,
        dones: PyReadwriteArrayDyn<'_, bool>,
    ) -> PyResult<HashMap<String, Vec<f64>>> {
        if self.action_spec != RlActionSpec::DiscreteTargets {
            return Err(PyValueError::new_err(
                "step_discrete_targets requires action_spec \"discrete_targets\"",
            ));
        }
        let action_shape = [
            self.n_envs,
            OUTER_PLAYER_SLOTS,
            ACTION_ENTITY_SLOTS,
            self.max_per_planet_launches,
        ];
        require_shape("launch", launch.shape(), &action_shape)?;
        require_shape("target", target.shape(), &action_shape)?;
        require_shape("ships", ships.shape(), &action_shape)?;
        require_shape(
            "rewards",
            rewards.shape(),
            &[self.n_envs, OUTER_PLAYER_SLOTS],
        )?;
        require_shape("dones", dones.shape(), &[self.n_envs, OUTER_PLAYER_SLOTS])?;
        self.require_obs_shapes(
            &planet_obs,
            &orbiting_planet_obs,
            &fleet_obs,
            &comet_obs,
            &entity_mask,
            &still_playing,
            &global_obs,
            &can_act,
            &max_launch,
        )?;

        let mut rewards = rewards;
        let mut dones = dones;
        let mut planet_obs = planet_obs;
        let mut orbiting_planet_obs = orbiting_planet_obs;
        let mut fleet_obs = fleet_obs;
        let mut comet_obs = comet_obs;
        let mut entity_mask = entity_mask;
        let mut still_playing = still_playing;
        let mut global_obs = global_obs;
        let mut can_act = can_act;
        let mut max_launch = max_launch;

        let reward_chunks = rewards.as_slice_mut()?.par_chunks_mut(OUTER_PLAYER_SLOTS);
        let done_chunks = dones.as_slice_mut()?.par_chunks_mut(OUTER_PLAYER_SLOTS);
        let actions_per_env =
            OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS * self.max_per_planet_launches;
        let launch_chunks = launch.as_slice()?.par_chunks(actions_per_env);
        let target_chunks = target.as_slice()?.par_chunks(actions_per_env);
        let ship_chunks = ships.as_slice()?.par_chunks(actions_per_env);

        let planets_per_env = MAX_PLANETS * PLANET_CHANNELS;
        let orbiting_planets_per_env = MAX_PLANETS;
        let fleets_per_env = self.max_fleets * FLEET_CHANNELS;
        let comets_per_env = MAX_COMETS * COMET_CHANNELS;
        let action_masks_per_env = self.action_spec.can_act_len();
        let max_launch_per_env = OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS;
        let max_per_planet_launches = self.max_per_planet_launches;
        let min_fleet_size = self.min_fleet_size;
        let max_fleets = self.max_fleets;
        let two_player_weight = self.two_player_weight;
        let action_spec = self.action_spec;

        let env_results = self
            .states
            .par_iter_mut()
            .zip_eq(self.player_maps.par_iter_mut())
            .zip_eq(self.action_slots.par_iter_mut())
            .zip_eq(self.player_finished.par_iter_mut())
            .zip_eq(self.episode_stats.par_iter_mut())
            .zip_eq(launch_chunks)
            .zip_eq(target_chunks)
            .zip_eq(ship_chunks)
            .zip_eq(reward_chunks)
            .zip_eq(done_chunks)
            .zip_eq(planet_obs.as_slice_mut()?.par_chunks_mut(planets_per_env))
            .zip_eq(
                orbiting_planet_obs
                    .as_slice_mut()?
                    .par_chunks_mut(orbiting_planets_per_env),
            )
            .zip_eq(fleet_obs.as_slice_mut()?.par_chunks_mut(fleets_per_env))
            .zip_eq(comet_obs.as_slice_mut()?.par_chunks_mut(comets_per_env))
            .zip_eq(
                entity_mask
                    .as_slice_mut()?
                    .par_chunks_mut(self.max_entities),
            )
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
                    .par_chunks_mut(max_launch_per_env),
            )
            .enumerate()
            .map(|(env_index, item)| {
                let (item, max_launch) = item;
                let (item, can_act) = item;
                let (item, global_obs) = item;
                let (item, still_playing) = item;
                let (item, entity_mask) = item;
                let (item, comet_obs) = item;
                let (item, fleet_obs) = item;
                let (item, orbiting_planet_obs) = item;
                let (item, planet_obs) = item;
                let (item, done_chunk) = item;
                let (item, reward_chunk) = item;
                let (item, ship_chunk) = item;
                let (item, target_chunk) = item;
                let (item, launch_chunk) = item;
                let ((((state, player_map), action_slots), player_finished), episode_stats) = item;

                {
                    let decoded = decode_discrete_target_actions(
                        state,
                        player_map,
                        action_slots,
                        launch_chunk,
                        target_chunk,
                        ship_chunk,
                        max_per_planet_launches,
                        min_fleet_size,
                    )
                    .map_err(|err| format!("env {env_index}: {err}"))?;
                    episode_stats.record_launch_failures(decoded.launch_failures);
                    let terminal = step_one_env(
                        state,
                        player_map,
                        player_finished,
                        episode_stats,
                        &decoded.actions,
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
                        action_spec,
                        max_fleets,
                        min_fleet_size,
                        planet_obs,
                        orbiting_planet_obs,
                        fleet_obs,
                        comet_obs,
                        entity_mask,
                        still_playing,
                        global_obs,
                        can_act,
                        max_launch,
                    );
                    Ok::<_, String>(StepOneOutput {
                        terminal_metrics: terminal.metrics,
                        terminal_snapshot: terminal.snapshot,
                        ignored_fleets,
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let mut terminal_metrics = Vec::with_capacity(env_results.len());
        let mut ignored_fleets = 0;
        for (env_index, result) in env_results.into_iter().enumerate() {
            self.last_terminal_snapshots[env_index] = result.terminal_snapshot;
            self.last_terminal_metrics[env_index] = result.terminal_metrics.clone();
            terminal_metrics.push(result.terminal_metrics);
            let ignored = result.ignored_fleets;
            ignored_fleets += ignored;
        }
        let episode_metrics = collect_terminal_metrics(terminal_metrics);
        log_ignored_fleets(ignored_fleets);
        Ok(episode_metrics)
    }

    fn obs_shapes(&self) -> ObsShapes {
        (
            (self.n_envs, MAX_PLANETS, PLANET_CHANNELS),
            (self.n_envs, MAX_PLANETS),
            (self.n_envs, self.max_fleets, FLEET_CHANNELS),
            (self.n_envs, MAX_COMETS, COMET_CHANNELS),
            (self.n_envs, self.max_entities),
            (self.n_envs, OUTER_PLAYER_SLOTS),
            (self.n_envs, GLOBAL_CHANNELS),
            match self.action_spec {
                RlActionSpec::Pure => vec![self.n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS],
                RlActionSpec::DiscreteTargets => vec![
                    self.n_envs,
                    OUTER_PLAYER_SLOTS,
                    ACTION_ENTITY_SLOTS,
                    ACTION_ENTITY_SLOTS,
                ],
            },
            (self.n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS),
        )
    }
}

impl PyRlVecEnv {
    fn current_snapshot(&self, env_index: usize) -> PyResult<StateSnapshot> {
        if env_index >= self.n_envs {
            return Err(PyValueError::new_err(format!(
                "env_index must be < {}, got {env_index}",
                self.n_envs
            )));
        }
        Ok(StateSnapshot::from_env(
            &self.states[env_index],
            &self.player_maps[env_index],
            &self.action_slots[env_index],
            &self.player_finished[env_index],
        ))
    }
}

impl PyRlVecEnv {
    #[allow(clippy::too_many_arguments)]
    fn require_obs_shapes(
        &self,
        planet_obs: &PyReadwriteArrayDyn<'_, f32>,
        orbiting_planet_obs: &PyReadwriteArrayDyn<'_, bool>,
        fleet_obs: &PyReadwriteArrayDyn<'_, f32>,
        comet_obs: &PyReadwriteArrayDyn<'_, f32>,
        entity_mask: &PyReadwriteArrayDyn<'_, bool>,
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
            "orbiting_planet_obs",
            orbiting_planet_obs.shape(),
            &[self.n_envs, MAX_PLANETS],
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
            "entity_mask",
            entity_mask.shape(),
            &[self.n_envs, self.max_entities],
        )?;
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
        let can_act_shape = match self.action_spec {
            RlActionSpec::Pure => vec![self.n_envs, OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS],
            RlActionSpec::DiscreteTargets => vec![
                self.n_envs,
                OUTER_PLAYER_SLOTS,
                ACTION_ENTITY_SLOTS,
                ACTION_ENTITY_SLOTS,
            ],
        };
        require_shape("can_act", can_act.shape(), &can_act_shape)?;
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
    launches_per_occupied_planet_sum: f64,
    occupied_planet_turns: u32,
    launches_per_launch_sum: f64,
    launched_planet_turns: u32,
    ships_per_launch_sum: i64,
    ships_per_launch_squared_sum: i64,
    launch_count: u32,
    launch_failures: u32,
    turn_count: u32,
    max_fleet_size: i32,
    min_fleet_size: Option<i32>,
    planets_captured: u32,
    comets_captured: u32,
    fleets_lost_in_combat: u32,
    ships_lost_in_combat: i64,
    max_entities_exceeded_turns: u32,
    fleet_losses: FleetLossStats,
}

#[derive(Clone, Debug)]
struct TerminalEpisodeMetrics {
    values: Vec<(&'static str, f64)>,
    win_rates: Vec<(usize, f64)>,
}

#[derive(Clone, Debug)]
struct StateSnapshot {
    state: State,
    player_map: PlayerMap,
    action_slots: ActionEntitySlots,
    player_finished: Vec<bool>,
}

impl StateSnapshot {
    fn from_env(
        state: &State,
        player_map: &PlayerMap,
        action_slots: &ActionEntitySlots,
        player_finished: &[bool],
    ) -> Self {
        let mut state = state.clone();
        state.orbit_paths.clear();
        state.static_planet_ids.clear();
        state.static_planet_mask.clear();
        state.static_target_cache = StaticTargetCache::empty();
        Self {
            state,
            player_map: *player_map,
            action_slots: *action_slots,
            player_finished: player_finished.to_vec(),
        }
    }
}

fn snapshot_to_py(py: Python<'_>, snapshot: &StateSnapshot) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("step", snapshot.state.step)?;
    dict.set_item("episode_steps", snapshot.state.config.episode_steps)?;
    dict.set_item("ship_speed", snapshot.state.config.ship_speed)?;
    dict.set_item("comet_speed", snapshot.state.config.comet_speed)?;
    dict.set_item("angular_velocity", snapshot.state.angular_velocity)?;
    dict.set_item("player_count", snapshot.state.config.player_count)?;
    dict.set_item("board_size", BOARD_SIZE)?;
    dict.set_item("center", CENTER)?;
    dict.set_item("sun_radius", SUN_RADIUS)?;
    dict.set_item("owner_space", "outer")?;
    dict.set_item("player_map", player_map_to_py(py, snapshot)?)?;
    dict.set_item("player_finished", player_finished_to_py(py, snapshot)?)?;
    dict.set_item("action_entity_slots", action_slots_to_py(py, snapshot)?)?;
    dict.set_item("planets", planets_to_py(py, snapshot)?)?;
    dict.set_item("fleets", fleets_to_py(py, snapshot)?)?;
    dict.set_item("comets", comets_to_py(py, snapshot)?)?;
    Ok(dict.into_any().unbind())
}

fn terminal_metrics_to_py(py: Python<'_>, metrics: &TerminalEpisodeMetrics) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (key, value) in &metrics.values {
        dict.set_item(*key, *value)?;
    }
    for (player, win_rate) in &metrics.win_rates {
        dict.set_item(format!("win_rate_player_{player}"), *win_rate)?;
    }
    Ok(dict.into_any().unbind())
}

fn player_map_to_py<'py>(
    py: Python<'py>,
    snapshot: &StateSnapshot,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let internal_to_outer = PyList::empty(py);
    for player in 0..OUTER_PLAYER_SLOTS {
        internal_to_outer.append(snapshot.player_map.internal_to_outer(player))?;
    }
    let outer_to_internal = PyList::empty(py);
    for player in 0..OUTER_PLAYER_SLOTS {
        match snapshot.player_map.outer_to_internal(player) {
            Some(internal) => outer_to_internal.append(internal)?,
            None => outer_to_internal.append(py.None())?,
        }
    }
    dict.set_item("internal_to_outer", internal_to_outer)?;
    dict.set_item("outer_to_internal", outer_to_internal)?;
    Ok(dict)
}

fn player_finished_to_py<'py>(
    py: Python<'py>,
    snapshot: &StateSnapshot,
) -> PyResult<Bound<'py, PyList>> {
    let values = PyList::empty(py);
    for outer_player in 0..OUTER_PLAYER_SLOTS {
        let finished = snapshot
            .player_map
            .outer_to_internal(outer_player)
            .and_then(|internal| snapshot.player_finished.get(internal))
            .copied()
            .unwrap_or(true);
        values.append(finished)?;
    }
    Ok(values)
}

fn action_slots_to_py<'py>(
    py: Python<'py>,
    snapshot: &StateSnapshot,
) -> PyResult<Bound<'py, PyList>> {
    let slots = PyList::empty(py);
    for slot in snapshot.action_slots {
        match slot {
            Some(slot) => slots.append(slot.planet_id)?,
            None => slots.append(py.None())?,
        }
    }
    Ok(slots)
}

fn planets_to_py<'py>(py: Python<'py>, snapshot: &StateSnapshot) -> PyResult<Bound<'py, PyList>> {
    let planets = PyList::empty(py);
    let comet_ids = snapshot
        .state
        .comet_planet_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    for planet in snapshot.state.planets.iter() {
        let item = PyDict::new(py);
        item.set_item("id", planet.id)?;
        item.set_item("owner", outer_owner(planet.owner, &snapshot.player_map))?;
        item.set_item("internal_owner", planet.owner)?;
        item.set_item("x", planet.x)?;
        item.set_item("y", planet.y)?;
        item.set_item("radius", planet.radius)?;
        item.set_item("ships", planet.ships)?;
        item.set_item("production", planet.production)?;
        item.set_item("is_comet", comet_ids.contains(&planet.id))?;
        planets.append(item)?;
    }
    Ok(planets)
}

fn fleets_to_py<'py>(py: Python<'py>, snapshot: &StateSnapshot) -> PyResult<Bound<'py, PyList>> {
    let fleets = PyList::empty(py);
    for fleet in &snapshot.state.fleets {
        let item = PyDict::new(py);
        item.set_item("id", fleet.id)?;
        item.set_item("owner", outer_owner(fleet.owner, &snapshot.player_map))?;
        item.set_item("internal_owner", fleet.owner)?;
        item.set_item("x", fleet.x)?;
        item.set_item("y", fleet.y)?;
        item.set_item("angle", fleet.angle)?;
        item.set_item("from_planet_id", fleet.from_planet_id)?;
        item.set_item("ships", fleet.ships)?;
        fleets.append(item)?;
    }
    Ok(fleets)
}

fn comets_to_py<'py>(py: Python<'py>, snapshot: &StateSnapshot) -> PyResult<Bound<'py, PyList>> {
    let comets = PyList::empty(py);
    for comet in &snapshot.state.comets {
        let item = PyDict::new(py);
        item.set_item("planet_ids", &comet.planet_ids)?;
        item.set_item("path_index", comet.path_index)?;
        let paths = PyList::empty(py);
        for path in &comet.paths {
            let points = PyList::empty(py);
            for point in path {
                points.append((point.x, point.y))?;
            }
            paths.append(points)?;
        }
        item.set_item("paths", paths)?;
        comets.append(item)?;
    }
    Ok(comets)
}

fn outer_owner(owner: i32, player_map: &PlayerMap) -> i32 {
    if owner < 0 {
        return owner;
    }
    let player = owner as usize;
    if player >= OUTER_PLAYER_SLOTS {
        return owner;
    }
    player_map.internal_to_outer(player) as i32
}

struct StepOneOutput {
    terminal_metrics: Option<TerminalEpisodeMetrics>,
    terminal_snapshot: Option<StateSnapshot>,
    ignored_fleets: usize,
}

struct TerminalStep {
    metrics: Option<TerminalEpisodeMetrics>,
    snapshot: Option<StateSnapshot>,
}

impl EpisodeStats {
    fn record_turn(&mut self, state: &State, decoded: &[PlayerAction]) {
        let comet_ids = state
            .comet_planet_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let mut occupied_planets = 0_usize;
        for planet in state.planets.iter() {
            if comet_ids.contains(&planet.id) {
                continue;
            }
            if planet.owner != -1 {
                occupied_planets += 1;
            }
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
        let launched_ships = decoded.iter().flatten().map(|action| action.ships);
        if let Some(turn_max_fleet_size) = launched_ships.clone().max() {
            self.max_fleet_size = turn_max_fleet_size.max(self.max_fleet_size);
        }
        if let Some(turn_min_fleet_size) = launched_ships.min() {
            self.min_fleet_size = Some(
                self.min_fleet_size
                    .map_or(turn_min_fleet_size, |fleet_size| {
                        fleet_size.min(turn_min_fleet_size)
                    }),
            );
        }
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

    fn record_comets_captured(&mut self, comets_captured: u32) {
        self.comets_captured += comets_captured;
    }

    fn record_ships_lost_in_combat(&mut self, ships_lost_in_combat: i64) {
        self.ships_lost_in_combat += ships_lost_in_combat;
    }

    fn record_fleets_lost_in_combat(&mut self, fleets_lost_in_combat: u32) {
        self.fleets_lost_in_combat += fleets_lost_in_combat;
    }

    fn record_launch_failures(&mut self, launch_failures: u32) {
        self.launch_failures += launch_failures;
    }

    fn terminal_metrics(
        &self,
        state: &State,
        player_map: &PlayerMap,
        player_results: &[PlayerResult],
    ) -> TerminalEpisodeMetrics {
        let fleets_lost_to_sun_or_oob =
            self.fleet_losses.fleets_in_sun + self.fleet_losses.fleets_out_of_bounds;
        let fleets_lost = fleets_lost_to_sun_or_oob + self.fleets_lost_in_combat;
        let ships_lost = i64::from(self.fleet_losses.ships_in_sun)
            + i64::from(self.fleet_losses.ships_out_of_bounds)
            + self.ships_lost_in_combat;
        let ships_lost_to_sun_or_oob =
            self.fleet_losses.ships_in_sun + self.fleet_losses.ships_out_of_bounds;
        let occupancy_key = if state.config.player_count == 2 {
            "terminal_planet_occupancy_rate_2p"
        } else {
            "terminal_planet_occupancy_rate_4p"
        };
        let mut values = vec![
            ("total_games_played", 1.0),
            (
                "max_entities_exceeded_per_game",
                f64::from(self.max_entities_exceeded_turns),
            ),
            ("game_length_mean", f64::from(state.step)),
            ("full_length_rate", full_length_value(state)),
            ("terminal_ship_count", terminal_ship_count(state)),
            (
                "planets_captured_per_game",
                f64::from(self.planets_captured),
            ),
            ("comets_captured_per_game", f64::from(self.comets_captured)),
            ("launches_per_game", f64::from(self.launch_count)),
            ("launch_failures_per_game", f64::from(self.launch_failures)),
            (
                "launches_per_turn",
                mean_or_zero(
                    f64::from(self.launch_count) / state.config.player_count as f64,
                    self.turn_count,
                ),
            ),
            ("fleet_size_max", f64::from(self.max_fleet_size)),
            (
                "fleet_size_min",
                f64::from(self.min_fleet_size.unwrap_or(0)),
            ),
            ("fleet_size_std", self.fleet_size_std()),
            (
                "launches_per_planet_mean",
                mean_or_zero(
                    self.launches_per_occupied_planet_sum,
                    self.occupied_planet_turns,
                ),
            ),
            (
                "ships_lost_in_combat_per_game",
                self.ships_lost_in_combat as f64,
            ),
            (
                "fleets_lost_in_combat_per_game",
                f64::from(self.fleets_lost_in_combat),
            ),
            ("ships_lost_per_game_mean", ships_lost as f64),
            (
                "ships_lost_in_sun_per_game_mean",
                f64::from(self.fleet_losses.ships_in_sun),
            ),
            (
                "ships_lost_out_of_bounds_per_game_mean",
                f64::from(self.fleet_losses.ships_out_of_bounds),
            ),
            ("fleets_lost_per_game_mean", f64::from(fleets_lost)),
            (
                "fleets_lost_in_sun_per_game_mean",
                f64::from(self.fleet_losses.fleets_in_sun),
            ),
            (
                "fleets_lost_out_of_bounds_per_game_mean",
                f64::from(self.fleet_losses.fleets_out_of_bounds),
            ),
            (occupancy_key, terminal_planet_occupancy_rate(state)),
        ];
        if let Some(rate) = loss_rate(f64::from(ships_lost_to_sun_or_oob), ships_lost as f64) {
            values.push(("ships_lost_to_sun_or_oob_rate", rate));
        }
        if let Some(rate) = loss_rate(f64::from(fleets_lost_to_sun_or_oob), f64::from(fleets_lost))
        {
            values.push(("fleets_lost_to_sun_or_oob_rate", rate));
        }
        if self.launched_planet_turns > 0 {
            values.push((
                "launches_per_launch_mean",
                self.launches_per_launch_sum / f64::from(self.launched_planet_turns),
            ));
        }
        if self.launch_count > 0 {
            values.push((
                "ships_per_launch_mean",
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

        TerminalEpisodeMetrics { values, win_rates }
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

fn terminal_planet_occupancy_rate(state: &State) -> f64 {
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
    if non_comet_planets == 0 {
        0.0
    } else {
        occupied_planets as f64 / non_comet_planets as f64
    }
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
    }
    if metrics.contains_key("ships_lost_per_game_mean")
        || metrics.contains_key("fleets_lost_per_game_mean")
    {
        add_loss_rate_metrics(&mut metrics);
    }
    metrics
}

fn add_loss_rate_metrics(metrics: &mut HashMap<String, Vec<f64>>) {
    let ships_lost_to_sun_or_oob = metric_sum(metrics, "ships_lost_in_sun_per_game_mean")
        + metric_sum(metrics, "ships_lost_out_of_bounds_per_game_mean");
    let ships_lost_total = metric_sum(metrics, "ships_lost_per_game_mean");
    if let Some(rate) = loss_rate(ships_lost_to_sun_or_oob, ships_lost_total) {
        metrics.insert("ships_lost_to_sun_or_oob_rate".to_string(), vec![rate]);
    }

    let fleets_lost_to_sun_or_oob = metric_sum(metrics, "fleets_lost_in_sun_per_game_mean")
        + metric_sum(metrics, "fleets_lost_out_of_bounds_per_game_mean");
    let fleets_lost_total = metric_sum(metrics, "fleets_lost_per_game_mean");
    if let Some(rate) = loss_rate(fleets_lost_to_sun_or_oob, fleets_lost_total) {
        metrics.insert("fleets_lost_to_sun_or_oob_rate".to_string(), vec![rate]);
    }
}

fn metric_sum(metrics: &HashMap<String, Vec<f64>>, key: &str) -> f64 {
    metrics
        .get(key)
        .map(|values| values.iter().sum())
        .unwrap_or(0.0)
}

fn loss_rate(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator <= 0.0 {
        return None;
    }
    Some((numerator / denominator).clamp(0.0, 1.0))
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
    action_spec: RlActionSpec,
    max_fleets: usize,
    min_fleet_size: i64,
    planet_obs: &mut [f32],
    orbiting_planet_obs: &mut [bool],
    fleet_obs: &mut [f32],
    comet_obs: &mut [f32],
    entity_mask: &mut [bool],
    still_playing: &mut [bool],
    global_obs: &mut [f32],
    can_act: &mut [bool],
    max_launch: &mut [i64],
) -> usize {
    write_still_playing(state, player_map, player_finished, still_playing);
    let (planet_mask, tail_mask) = entity_mask.split_at_mut(MAX_PLANETS);
    let (comet_mask, fleet_mask) = tail_mask.split_at_mut(MAX_COMETS);
    encode_state_with_action_slots(
        action_spec,
        state,
        player_map,
        max_fleets,
        planet_obs,
        orbiting_planet_obs,
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
) -> TerminalStep {
    episode_stats.record_turn(state, decoded);
    let result = step(state, decoded);
    episode_stats.record_step_result(state, result.fleet_losses, max_fleets);
    episode_stats.record_planets_captured(result.planets_captured);
    episode_stats.record_comets_captured(result.comets_captured);
    episode_stats.record_fleets_lost_in_combat(result.fleets_lost_in_combat);
    episode_stats.record_ships_lost_in_combat(result.ships_lost_in_combat);
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
        let terminal_action_slots = action_entity_slots(state);
        let terminal_snapshot =
            StateSnapshot::from_env(state, player_map, &terminal_action_slots, player_finished);
        let (new_state, new_player_map) = reset_one_env(two_player_weight);
        *state = new_state;
        *player_map = new_player_map;
        player_finished.fill(false);
        *episode_stats = EpisodeStats::default();
        return TerminalStep {
            metrics: Some(terminal_metrics),
            snapshot: Some(terminal_snapshot),
        };
    }
    TerminalStep {
        metrics: None,
        snapshot: None,
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
    use crate::rules_engine::state::{LaunchAction, Planet, SimConfig, StaticTargetCache};

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
            orbit_paths: Vec::new(),
            static_planet_ids: Vec::new(),
            static_planet_mask: Vec::new(),
            static_target_cache: StaticTargetCache::empty(),
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
            orbit_paths: Vec::new(),
            static_planet_ids: Vec::new(),
            static_planet_mask: Vec::new(),
            static_target_cache: StaticTargetCache::empty(),
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
            orbit_paths: Vec::new(),
            static_planet_ids: Vec::new(),
            static_planet_mask: Vec::new(),
            static_target_cache: StaticTargetCache::empty(),
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
        .metrics
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
        assert_eq!(metrics["game_length_mean"], vec![499.0]);
        assert_eq!(metrics["total_games_played"], vec![1.0]);
        assert_eq!(metrics["full_length_rate"], vec![1.0]);
        assert_eq!(metrics["terminal_ship_count"], vec![44.0]);
        assert_eq!(metrics["launches_per_game"], vec![0.0]);
        assert_eq!(metrics["launch_failures_per_game"], vec![0.0]);
        assert_eq!(metrics["win_rate_player_0"], vec![1.0]);
        assert_eq!(metrics["win_rate_player_3"], vec![1.0]);
        assert_eq!(metrics["terminal_planet_occupancy_rate_4p"], vec![1.0]);
        assert_eq!(metrics["fleets_lost_per_game_mean"], vec![0.0]);
    }

    #[test]
    fn terminal_planet_occupancy_rate_counts_only_terminal_non_comet_planets() {
        let mut state = state_with_all_players_alive();
        state.planets.push(Planet {
            id: 4,
            owner: -1,
            x: 50.0,
            y: 50.0,
            radius: 2.0,
            ships: 0,
            production: 1,
        });
        state.planets.push(Planet {
            id: 5,
            owner: -1,
            x: 60.0,
            y: 60.0,
            radius: 2.0,
            ships: 0,
            production: 1,
        });
        state.planets.push(Planet {
            id: 6,
            owner: 0,
            x: 70.0,
            y: 70.0,
            radius: 2.0,
            ships: 10,
            production: 1,
        });
        state.comet_planet_ids.push(6);

        assert_eq!(terminal_planet_occupancy_rate(&state), 4.0 / 6.0);
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
    fn episode_stats_reports_ships_per_launch_mean() {
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
        episode_stats.record_launch_failures(3);
        episode_stats.record_step_result(
            &state,
            FleetLossStats {
                fleets_in_sun: 2,
                fleets_out_of_bounds: 3,
                ships_in_sun: 3,
                ships_out_of_bounds: 6,
            },
            128,
        );
        episode_stats.record_ships_lost_in_combat(11);
        episode_stats.record_fleets_lost_in_combat(4);
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
        let mut second_episode_stats = EpisodeStats::default();
        second_episode_stats.record_step_result(
            &state,
            FleetLossStats {
                fleets_in_sun: 1,
                fleets_out_of_bounds: 0,
                ships_in_sun: 10,
                ships_out_of_bounds: 0,
            },
            128,
        );
        second_episode_stats.record_ships_lost_in_combat(90);
        second_episode_stats.record_fleets_lost_in_combat(9);
        let second_metrics = second_episode_stats.terminal_metrics(
            &state,
            &player_map,
            &[
                PlayerResult::Won,
                PlayerResult::Lost,
                PlayerResult::Lost,
                PlayerResult::Lost,
            ],
        );
        assert_eq!(
            metrics
                .values
                .iter()
                .find(|(key, _)| *key == "ships_lost_to_sun_or_oob_rate")
                .map(|(_, value)| *value),
            Some(9.0 / 20.0)
        );
        assert_eq!(
            metrics
                .values
                .iter()
                .find(|(key, _)| *key == "fleets_lost_to_sun_or_oob_rate")
                .map(|(_, value)| *value),
            Some(5.0 / 9.0)
        );
        let collected = collect_terminal_metrics(vec![Some(metrics), Some(second_metrics)]);

        assert_eq!(collected["ships_per_launch_mean"], vec![4.0]);
        assert_eq!(collected["launches_per_game"], vec![2.0, 0.0]);
        assert_eq!(collected["launch_failures_per_game"], vec![3.0, 0.0]);
        assert_eq!(collected["ships_lost_in_combat_per_game"], vec![11.0, 90.0]);
        assert_eq!(collected["fleets_lost_in_combat_per_game"], vec![4.0, 9.0]);
        assert_eq!(collected["ships_lost_per_game_mean"], vec![20.0, 100.0]);
        assert_eq!(collected["fleets_lost_per_game_mean"], vec![9.0, 10.0]);
        assert_eq!(
            collected["ships_lost_to_sun_or_oob_rate"],
            vec![19.0 / 120.0]
        );
        assert_eq!(
            collected["fleets_lost_to_sun_or_oob_rate"],
            vec![6.0 / 19.0]
        );
        assert_eq!(collected["launches_per_turn"], vec![0.5, 0.0]);
        assert_eq!(collected["fleet_size_max"], vec![5.0, 0.0]);
        assert_eq!(collected["fleet_size_min"], vec![3.0, 0.0]);
        assert_eq!(collected["fleet_size_std"], vec![1.0, 0.0]);
        assert_eq!(collected["planets_captured_per_game"], vec![0.0, 0.0]);
        assert_eq!(collected["comets_captured_per_game"], vec![0.0, 0.0]);
        assert!(!collected.contains_key("fleets_per_launch_mean"));
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
