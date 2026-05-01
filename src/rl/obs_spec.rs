use std::collections::HashSet;

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
use crate::rules_engine::utils::{fleet_speed, is_orbiting};

use super::action_spec::{action_entity_slots, encode_action_spec, ActionEntitySlots};
use super::{
    log_ignored_fleets, require_shape, PlayerMap, ACTION_ENTITY_SLOTS, COMET_CHANNELS,
    DEFAULT_MAX_ENTITIES, FLEET_CHANNELS, GLOBAL_CHANNELS, MAX_COMETS, MAX_COMET_PATH_LENGTH,
    MAX_PLANETS, OUTER_PLAYER_SLOTS, PLANET_CHANNELS,
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
const INTEGER_TOLERANCE: f64 = 1e-9;

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_state(
    state: &State,
    player_map: &PlayerMap,
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
    min_fleet_size: i64,
) -> usize {
    let mut action_slots = [None; ACTION_ENTITY_SLOTS];
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
        &mut action_slots,
        min_fleet_size,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_state_with_action_slots(
    state: &State,
    player_map: &PlayerMap,
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
    action_slots: &mut ActionEntitySlots,
    min_fleet_size: i64,
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
    let ignored_fleets = fleets.len().saturating_sub(max_fleets);
    if ignored_fleets > 0 {
        fleets.sort_by(|left, right| right.ships.cmp(&left.ships).then(left.id.cmp(&right.id)));
    }

    for (planet_index, planet) in non_comet_planets.iter().enumerate() {
        planet_mask[planet_index] = true;
        let row_start = planet_index * PLANET_CHANNELS;
        let row = &mut planet_obs[row_start..row_start + PLANET_CHANNELS];

        let owner_index = player_map.owner_channel(planet.owner);
        row[owner_index] = 1.0;
        row[OWNER_CHANNELS_WITH_NEUTRAL] = normalize_position(planet.x);
        row[OWNER_CHANNELS_WITH_NEUTRAL + 1] = normalize_position(planet.y);

        let production = production_channel(planet.production);
        row[OWNER_CHANNELS_WITH_NEUTRAL + 2 + production] = 1.0;

        row[12] = (planet.radius / 3.0) as f32;
        row[13] = normalize_ships(planet.ships);
        row[14] = normalize_log_ships(planet.ships);
        row[15] = f32::from(is_orbiting(planet.position(), planet.radius));
    }

    for (fleet_index, fleet) in fleets.iter().take(max_fleets).enumerate() {
        fleet_mask[fleet_index] = true;
        let row_start = fleet_index * FLEET_CHANNELS;
        let row = &mut fleet_obs[row_start..row_start + FLEET_CHANNELS];

        row[player_map.owner_channel(fleet.owner)] = 1.0;
        row[OWNER_CHANNELS] = normalize_position(fleet.x);
        row[OWNER_CHANNELS + 1] = normalize_position(fleet.y);
        let speed = fleet_speed(fleet.ships, state.config.ship_speed);
        row[OWNER_CHANNELS + 2] = (fleet.angle.cos() * speed / state.config.ship_speed) as f32;
        row[OWNER_CHANNELS + 3] = (fleet.angle.sin() * speed / state.config.ship_speed) as f32;
        row[OWNER_CHANNELS + 4] = normalize_ships(fleet.ships);
        row[OWNER_CHANNELS + 5] = normalize_log_ships(fleet.ships);
    }

    encode_comets(state, player_map, comet_obs, comet_mask);
    encode_global(state, global_obs);
    *action_slots = action_entity_slots(state);
    encode_action_spec(
        state,
        player_map,
        action_slots,
        can_act,
        max_launch,
        min_fleet_size,
    );

    ignored_fleets
}

fn normalize_angular_velocity(angular_velocity: f64) -> f32 {
    ((angular_velocity as f32) - MIN_ANGULAR_VELOCITY) / ANGULAR_VELOCITY_SPAN
}

fn normalize_position(value: f64) -> f32 {
    ((value / BOARD_SIZE) * 2.0 - 1.0) as f32
}

fn encode_comets(
    state: &State,
    player_map: &PlayerMap,
    comet_obs: &mut [f32],
    comet_mask: &mut [bool],
) {
    let mut comet_index = 0;
    for group in &state.comets {
        for (path_offset, planet_id) in group.planet_ids.iter().enumerate() {
            if comet_index >= MAX_COMETS {
                return;
            }
            let Some(planet) = state.planets.get(*planet_id) else {
                continue;
            };
            let Some(path) = group.paths.get(path_offset) else {
                continue;
            };

            comet_mask[comet_index] = true;
            let row_start = comet_index * COMET_CHANNELS;
            let row = &mut comet_obs[row_start..row_start + COMET_CHANNELS];

            let owner_index = player_map.owner_channel(planet.owner);
            row[owner_index] = 1.0;
            row[OWNER_CHANNELS_WITH_NEUTRAL] = normalize_ships(planet.ships);
            row[OWNER_CHANNELS_WITH_NEUTRAL + 1] = normalize_log_ships(planet.ships);

            let path_start = group.path_index.max(0) as usize;
            let remaining_steps = path.len().saturating_sub(path_start);
            row[OWNER_CHANNELS_WITH_NEUTRAL + 2] =
                remaining_steps as f32 / MAX_COMET_PATH_LENGTH as f32;
            let path_values_start = OWNER_CHANNELS_WITH_NEUTRAL + 3;
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

    finite_f64(angular_velocity, "angular_velocity")?;
    if episode_steps == 0 {
        return Err(PyValueError::new_err("episode_steps must be > 0"));
    }

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
                production: finite_production(row[6])?,
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
        initial_planets: planets.clone().into(),
        planets: planets.into(),
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
    let rounded = finite_integer(value, name)?;
    if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
        return Err(PyValueError::new_err(format!("{name} must fit in i32")));
    }
    Ok(rounded as i32)
}

fn finite_u32(value: f64, name: &str) -> PyResult<u32> {
    let rounded = finite_integer(value, name)?;
    if rounded < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be non-negative"
        )));
    }
    if rounded > f64::from(u32::MAX) {
        return Err(PyValueError::new_err(format!("{name} must fit in u32")));
    }
    Ok(rounded as u32)
}

fn finite_usize(value: f64, name: &str) -> PyResult<usize> {
    let rounded = finite_integer(value, name)?;
    if rounded < 0.0 || rounded > MAX_COMET_PATH_LENGTH as f64 {
        return Err(PyValueError::new_err(format!(
            "{name} must be between 0 and {MAX_COMET_PATH_LENGTH}"
        )));
    }
    Ok(rounded as usize)
}

fn finite_production(value: f64) -> PyResult<i32> {
    let production = finite_i32(value, "planet production")?;
    if (1..=PRODUCTION_CHANNELS as i32).contains(&production) {
        return Ok(production);
    }
    Err(PyValueError::new_err(format!(
        "planet production must be between 1 and {PRODUCTION_CHANNELS}"
    )))
}

fn finite_integer(value: f64, name: &str) -> PyResult<f64> {
    finite_f64(value, name)?;
    let rounded = value.round();
    if (value - rounded).abs() <= INTEGER_TOLERANCE {
        return Ok(rounded);
    }
    Err(PyValueError::new_err(format!("{name} must be an integer")))
}

fn production_channel(production: i32) -> usize {
    assert!(
        (1..=PRODUCTION_CHANNELS as i32).contains(&production),
        "planet production must be between 1 and {PRODUCTION_CHANNELS}, got {production}"
    );
    production as usize - 1
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
    max_entities=DEFAULT_MAX_ENTITIES,
    min_fleet_size=1
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
    min_fleet_size: i64,
) -> PyResult<EncodedObsV1<'py>> {
    if max_entities <= MAX_PLANETS + MAX_COMETS {
        return Err(PyValueError::new_err(format!(
            "max_entities must be greater than MAX_PLANETS + MAX_COMETS ({})",
            MAX_PLANETS + MAX_COMETS
        )));
    }
    if min_fleet_size < 1 || min_fleet_size > i64::from(i32::MAX) {
        return Err(PyValueError::new_err(
            "min_fleet_size must be between 1 and i32::MAX",
        ));
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
        &PlayerMap::identity(),
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
        min_fleet_size,
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
    use crate::rules_engine::state::{Fleet, Planet, SimConfig, State};

    #[test]
    fn angular_velocity_normalization_maps_generated_range_to_zero_one() {
        assert_eq!(normalize_angular_velocity(0.025), 0.0);
        assert_eq!(normalize_angular_velocity(0.05), 1.0);
    }

    #[test]
    fn integer_fields_round_values_within_tolerance() {
        assert_eq!(finite_i32(4.0 + 5e-10, "value").unwrap(), 4);
        assert_eq!(finite_u32(7.0 - 5e-10, "value").unwrap(), 7);
        assert_eq!(finite_usize(3.0 + 5e-10, "value").unwrap(), 3);
    }

    #[test]
    fn integer_fields_reject_fractional_values() {
        assert!(finite_i32(4.1, "value").is_err());
        assert!(finite_u32(7.9, "value").is_err());
        assert!(finite_usize(3.5, "value").is_err());
    }

    #[test]
    fn planet_production_requires_documented_one_hot_range() {
        assert_eq!(finite_production(1.0).unwrap(), 1);
        assert_eq!(finite_production(5.0).unwrap(), 5);
        assert_eq!(production_channel(1), 0);
        assert_eq!(production_channel(5), 4);

        assert!(finite_production(0.0).is_err());
        assert!(finite_production(6.0).is_err());
        assert!(finite_production(-1.0).is_err());
    }

    #[test]
    #[should_panic(expected = "planet production must be between 1 and 5")]
    fn encode_state_rejects_invalid_planet_production() {
        let state = State {
            config: SimConfig::new(2),
            step: 0,
            angular_velocity: 0.025,
            initial_planets: Vec::new().into(),
            planets: vec![Planet {
                id: 0,
                owner: 0,
                x: 50.0,
                y: 50.0,
                radius: 2.0,
                ships: 10,
                production: 0,
            }]
            .into(),
            fleets: Vec::new(),
            next_fleet_id: 0,
            comets: Vec::new(),
            comet_planet_ids: Vec::new(),
        };
        let mut planet_obs = vec![0.0; MAX_PLANETS * PLANET_CHANNELS];
        let mut fleet_obs = Vec::new();
        let mut comet_obs = vec![0.0; MAX_COMETS * COMET_CHANNELS];
        let mut planet_mask = vec![false; MAX_PLANETS];
        let mut fleet_mask = Vec::new();
        let mut comet_mask = vec![false; MAX_COMETS];
        let mut global_obs = vec![0.0; GLOBAL_CHANNELS];
        let mut can_act = vec![false; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut max_launch = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];

        encode_state(
            &state,
            &PlayerMap::identity(),
            0,
            &mut planet_obs,
            &mut fleet_obs,
            &mut comet_obs,
            &mut planet_mask,
            &mut fleet_mask,
            &mut comet_mask,
            &mut global_obs,
            &mut can_act,
            &mut max_launch,
            1,
        );
    }

    #[test]
    fn encode_state_writes_owners_and_action_masks_to_remapped_outer_slots() {
        let state = State {
            config: SimConfig::new(2),
            step: 0,
            angular_velocity: 0.025,
            initial_planets: Vec::new().into(),
            planets: vec![Planet {
                id: 0,
                owner: 1,
                x: 50.0,
                y: 50.0,
                radius: 2.0,
                ships: 10,
                production: 1,
            }]
            .into(),
            fleets: vec![Fleet {
                id: 0,
                owner: 0,
                x: 50.0,
                y: 50.0,
                angle: 0.0,
                from_planet_id: 0,
                ships: 5,
            }],
            next_fleet_id: 1,
            comets: Vec::new(),
            comet_planet_ids: Vec::new(),
        };
        let player_map = PlayerMap::from_outer_slots(2, [3, 1, 0, 2]);
        let mut planet_obs = vec![0.0; MAX_PLANETS * PLANET_CHANNELS];
        let mut fleet_obs = vec![0.0; FLEET_CHANNELS];
        let mut comet_obs = vec![0.0; MAX_COMETS * COMET_CHANNELS];
        let mut planet_mask = vec![false; MAX_PLANETS];
        let mut fleet_mask = vec![false; 1];
        let mut comet_mask = vec![false; MAX_COMETS];
        let mut global_obs = vec![0.0; GLOBAL_CHANNELS];
        let mut can_act = vec![false; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut max_launch = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];

        encode_state(
            &state,
            &player_map,
            1,
            &mut planet_obs,
            &mut fleet_obs,
            &mut comet_obs,
            &mut planet_mask,
            &mut fleet_mask,
            &mut comet_mask,
            &mut global_obs,
            &mut can_act,
            &mut max_launch,
            1,
        );

        assert_eq!(planet_obs[1], 1.0);
        assert_eq!(fleet_obs[3], 1.0);
        assert!(can_act[ACTION_ENTITY_SLOTS]);
        assert_eq!(max_launch[ACTION_ENTITY_SLOTS], 10);
    }

    #[test]
    fn encode_state_respects_min_fleet_size_action_mask() {
        let state = State {
            config: SimConfig::new(2),
            step: 0,
            angular_velocity: 0.025,
            initial_planets: Vec::new().into(),
            planets: vec![Planet {
                id: 0,
                owner: 0,
                x: 50.0,
                y: 50.0,
                radius: 2.0,
                ships: 2,
                production: 1,
            }]
            .into(),
            fleets: Vec::new(),
            next_fleet_id: 0,
            comets: Vec::new(),
            comet_planet_ids: Vec::new(),
        };
        let mut planet_obs = vec![0.0; MAX_PLANETS * PLANET_CHANNELS];
        let mut fleet_obs = Vec::new();
        let mut comet_obs = vec![0.0; MAX_COMETS * COMET_CHANNELS];
        let mut planet_mask = vec![false; MAX_PLANETS];
        let mut fleet_mask = Vec::new();
        let mut comet_mask = vec![false; MAX_COMETS];
        let mut global_obs = vec![0.0; GLOBAL_CHANNELS];
        let mut can_act = vec![false; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut max_launch = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];

        encode_state(
            &state,
            &PlayerMap::identity(),
            0,
            &mut planet_obs,
            &mut fleet_obs,
            &mut comet_obs,
            &mut planet_mask,
            &mut fleet_mask,
            &mut comet_mask,
            &mut global_obs,
            &mut can_act,
            &mut max_launch,
            3,
        );

        assert!(!can_act[0]);
        assert_eq!(max_launch[0], 0);
    }
}
