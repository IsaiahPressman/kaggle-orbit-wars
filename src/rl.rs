mod action_spec;
mod obs_spec;
mod vec_env;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use obs_spec::encode_obs_v1;
use vec_env::PyRlVecEnv;

pub const MAX_PLANETS: usize = 64;
pub const MAX_COMETS: usize = 4;
pub const MAX_COMET_PATH_LENGTH: usize = 40;
pub const DEFAULT_MAX_ENTITIES: usize = 512;
pub const PLANET_CHANNELS: usize = 16;
pub const FLEET_CHANNELS: usize = 10;
pub const COMET_CHANNELS: usize = OWNER_CHANNELS_WITH_NEUTRAL + 2 + MAX_COMET_PATH_LENGTH * 2;
pub const GLOBAL_CHANNELS: usize = 5;
pub const OUTER_PLAYER_SLOTS: usize = 4;
pub const ACTION_ENTITY_SLOTS: usize = MAX_PLANETS + MAX_COMETS;

const OWNER_CHANNELS_WITH_NEUTRAL: usize = 5;

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
