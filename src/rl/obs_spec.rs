use std::collections::{HashMap, HashSet};

use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::rules_engine::state::{
    CometGroup, Fleet, Planet, Point, SimConfig, State, BOARD_SIZE, COMET_SPAWN_STEPS,
};

use super::action_spec::encode_action_spec;
use super::{
    log_ignored_fleets, require_shape, ACTION_ENTITY_SLOTS, COMET_CHANNELS, DEFAULT_MAX_ENTITIES,
    FLEET_CHANNELS, GLOBAL_CHANNELS, MAX_COMETS, MAX_COMET_PATH_LENGTH, MAX_PLANETS,
    OUTER_PLAYER_SLOTS, PLANET_CHANNELS,
};

type EncodedObsV1<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<i64>>,
);

const OWNER_CHANNELS_WITH_NEUTRAL: usize = 5;
const OWNER_CHANNELS: usize = 4;
const PRODUCTION_CHANNELS: usize = 5;
const SHIP_NORMALIZER: f32 = 250.0;
const LOG_SHIP_NORMALIZER: f32 = 4.6051702;
const MIN_ANGULAR_VELOCITY: f32 = 0.025;
const ANGULAR_VELOCITY_SPAN: f32 = 0.025;

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_state(
    state: &State,
    max_fleets: usize,
    planet_obs: &mut [f32],
    fleet_obs: &mut [f32],
    comet_obs: &mut [f32],
    planet_mask: &mut [bool],
    fleet_mask: &mut [bool],
    comet_mask: &mut [bool],
    global_obs: &mut [f32],
    can_act: &mut [bool],
    max_launch: &mut [i64],
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
    can_act.fill(false);
    max_launch.fill(0);

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

        row[12] = (planet.radius / 3.0) as f32;
        row[13] = normalize_ships(planet.ships);
        row[14] = normalize_log_ships(planet.ships);
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
    encode_action_spec(state, can_act, max_launch);

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
}

fn steps_until_next_comet_spawn(step: u32) -> u32 {
    COMET_SPAWN_STEPS
        .iter()
        .copied()
        .find(|spawn_step| *spawn_step > step)
        .map_or(0, |spawn_step| spawn_step - step)
}

fn normalize_ships(ships: i32) -> f32 {
    ships as f32 / SHIP_NORMALIZER
}

fn normalize_log_ships(ships: i32) -> f32 {
    ((ships.max(0) as f32) + 1.0).ln() / LOG_SHIP_NORMALIZER
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

fn require_shape_suffix(name: &str, actual: &[usize], expected_last_dim: usize) -> PyResult<()> {
    if actual.len() == 2 && actual[1] == expected_last_dim {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{name} must have shape (n, {expected_last_dim}), got {actual:?}"
    )))
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
    let mut can_act = Array2::<bool>::from_elem((OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS), false);
    let mut max_launch = Array2::<i64>::zeros((OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS));

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
        can_act
            .as_slice_mut()
            .expect("newly allocated can_act array is contiguous"),
        max_launch
            .as_slice_mut()
            .expect("newly allocated max_launch array is contiguous"),
    );
    log_ignored_fleets(ignored_fleets);

    Ok((
        planet_obs.into_pyarray(py),
        fleet_obs.into_pyarray(py),
        comet_obs.into_pyarray(py),
        planet_mask.into_pyarray(py),
        fleet_mask.into_pyarray(py),
        comet_mask.into_pyarray(py),
        global_obs.into_pyarray(py),
        can_act.into_pyarray(py),
        max_launch.into_pyarray(py),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angular_velocity_normalization_maps_generated_range_to_zero_one() {
        assert_eq!(normalize_angular_velocity(0.025), 0.0);
        assert_eq!(normalize_angular_velocity(0.05), 1.0);
    }
}
