mod action_spec;
mod obs_spec;
mod vec_env;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::RngExt;

use crate::rules_engine::state::MAX_PLAYERS;

use obs_spec::encode_obs_v1;
use vec_env::PyRlVecEnv;

pub const MAX_PLANETS: usize = 40;
pub const MAX_COMETS: usize = 4;
pub const MAX_COMET_PATH_LENGTH: usize = 40;
pub const DEFAULT_MAX_ENTITIES: usize = 1024;
const BASE_PLANET_CHANNELS: usize = 15;
const BASE_FLEET_CHANNELS: usize = 10;
const CARTESIAN_FOURIER_FREQUENCY_COUNT: usize = 6;
const RADIAL_FOURIER_FREQUENCY_COUNT: usize = 4;
const ANGULAR_HARMONIC_COUNT: usize = 3;
const SPATIAL_CHANNELS: usize = CARTESIAN_FOURIER_FREQUENCY_COUNT * 4
    + 4
    + ANGULAR_HARMONIC_COUNT * 2
    + RADIAL_FOURIER_FREQUENCY_COUNT * 2;
const FLEET_MOTION_CHANNELS: usize = 5;
pub const PLANET_CHANNELS: usize = BASE_PLANET_CHANNELS + SPATIAL_CHANNELS;
pub const FLEET_CHANNELS: usize = BASE_FLEET_CHANNELS + SPATIAL_CHANNELS + FLEET_MOTION_CHANNELS;
pub const COMET_CHANNELS: usize = OWNER_CHANNELS_WITH_NEUTRAL + 3 + MAX_COMET_PATH_LENGTH * 2;
pub const GLOBAL_CHANNELS: usize = 3;
pub const OUTER_PLAYER_SLOTS: usize = MAX_PLAYERS;
pub const ACTION_ENTITY_SLOTS: usize = MAX_PLANETS + MAX_COMETS;

const OWNER_CHANNELS_WITH_NEUTRAL: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PlayerMap {
    internal_to_outer: [usize; OUTER_PLAYER_SLOTS],
    outer_to_internal: [Option<usize>; OUTER_PLAYER_SLOTS],
}

impl PlayerMap {
    pub(super) fn identity() -> Self {
        Self::from_outer_slots(OUTER_PLAYER_SLOTS, [0, 1, 2, 3])
    }

    pub(super) fn random(player_count: usize) -> Self {
        assert!(
            player_count == 2 || player_count == MAX_PLAYERS,
            "Orbit Wars supports exactly 2 or 4 players"
        );

        let mut slots = [0, 1, 2, 3];
        let mut rng = rand::rng();
        for index in (1..OUTER_PLAYER_SLOTS).rev() {
            let swap_index = rng.random_range(0..=index);
            slots.swap(index, swap_index);
        }

        Self::from_outer_slots(player_count, slots)
    }

    pub(super) fn internal_to_outer(&self, player: usize) -> usize {
        self.internal_to_outer[player]
    }

    pub(super) fn outer_to_internal(&self, player: usize) -> Option<usize> {
        self.outer_to_internal[player]
    }

    pub(super) fn owner_channel(&self, owner: i32) -> usize {
        if owner == -1 {
            return OWNER_CHANNELS_WITH_NEUTRAL - 1;
        }
        self.internal_to_outer(owner as usize)
    }

    pub(super) fn from_outer_slots(
        player_count: usize,
        slots: [usize; OUTER_PLAYER_SLOTS],
    ) -> Self {
        assert!(
            player_count == 2 || player_count == MAX_PLAYERS,
            "Orbit Wars supports exactly 2 or 4 players"
        );

        let mut internal_to_outer = [0; OUTER_PLAYER_SLOTS];
        let mut outer_to_internal = [None; OUTER_PLAYER_SLOTS];
        for internal in 0..OUTER_PLAYER_SLOTS {
            let outer = slots[internal];
            internal_to_outer[internal] = outer;
            if internal < player_count {
                outer_to_internal[outer] = Some(internal);
            }
        }

        Self {
            internal_to_outer,
            outer_to_internal,
        }
    }
}

pub(super) fn require_shape(name: &str, actual: &[usize], expected: &[usize]) -> PyResult<()> {
    if actual == expected {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{name} must have shape {expected:?}, got {actual:?}"
    )))
}

pub(super) fn log_ignored_fleets(ignored_fleets: usize) {
    if ignored_fleets == 0 {
        return;
    }
    eprintln!("max_entities exceeded: {ignored_fleets} fleets ignored");
}

#[pyfunction]
pub fn rl_obs_constants() -> (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
) {
    (
        MAX_PLANETS,
        MAX_COMETS,
        MAX_COMET_PATH_LENGTH,
        ACTION_ENTITY_SLOTS,
        DEFAULT_MAX_ENTITIES,
        PLANET_CHANNELS,
        FLEET_CHANNELS,
        COMET_CHANNELS,
        GLOBAL_CHANNELS,
    )
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRlVecEnv>()?;
    m.add_function(wrap_pyfunction!(rl_obs_constants, m)?)?;
    m.add_function(wrap_pyfunction!(encode_obs_v1, m)?)?;
    Ok(())
}
