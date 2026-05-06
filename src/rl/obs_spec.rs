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

use super::action_spec::{
    action_entity_slots, encode_action_spec, sorted_comet_planet_ids, ActionEntitySlots,
    RlActionSpec,
};
use super::{
    log_ignored_fleets, require_shape, PlayerMap, ACTION_ENTITY_SLOTS, COMET_CHANNELS,
    DEFAULT_MAX_ENTITIES, FLEET_CHANNELS, GLOBAL_CHANNELS, MAX_COMETS, MAX_COMET_PATH_LENGTH,
    MAX_PLANETS, OUTER_PLAYER_SLOTS, PLANET_CHANNELS,
};

type EncodedEntityBased<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<i64>>,
);

const OWNER_CHANNELS_WITH_NEUTRAL: usize = 5;
const OWNER_CHANNELS: usize = 4;
const PRODUCTION_CHANNELS: usize = 5;
const NEUTRAL_SHIP_NORMALIZER: f32 = 100.0;
const SHIP_NORMALIZER: f32 = 500.0;
// ln(100) ~= 4.6051702
const LOG_SHIP_NORMALIZER: f32 = 4.6051702;
const MIN_ANGULAR_VELOCITY: f32 = 0.025;
const ANGULAR_VELOCITY_SPAN: f32 = 0.025;
const INTEGER_TOLERANCE: f64 = 1e-9;
const BASE_PLANET_CHANNELS: usize = 17;
const BASE_FLEET_CHANNELS: usize = 10;
const MAX_PLANET_SPAWN: i32 = 99;
const NEUTRAL_SHIP_COUNT_BUCKETS: [i32; 9] = [0, 1, 2, 4, 8, 16, 32, 64, MAX_PLANET_SPAWN];
const PLANET_SHIP_COUNT_BUCKETS: [i32; 12] = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
const FLEET_SHIP_COUNT_BUCKET_CAPACITY: usize = 10;
const FLEET_SHIP_COUNT_MAX_BUCKET: i32 = 512;
const COMET_SHIP_COUNT_BUCKETS: [i32; 11] = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
const SHIP_COUNT_OVERFLOW_CHANNELS: usize = 2;
const CARTESIAN_FOURIER_FREQUENCIES: [f32; 6] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
const RADIAL_FOURIER_FREQUENCIES: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
const PLANET_ORBITAL_CHANNELS: usize = 2;
const COMET_BASE_CHANNELS: usize = OWNER_CHANNELS_WITH_NEUTRAL
    + 3
    + ship_count_feature_count(NEUTRAL_SHIP_COUNT_BUCKETS.len())
    + ship_count_feature_count(COMET_SHIP_COUNT_BUCKETS.len());
const COMET_SELECTED_FUTURE_OFFSETS: [usize; 5] = [1, 2, 4, 8, 16];

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_state(
    action_spec: RlActionSpec,
    state: &State,
    player_map: &PlayerMap,
    max_fleets: usize,
    planet_obs: &mut [f32],
    orbiting_planet_obs: &mut [bool],
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
        &mut action_slots,
        min_fleet_size,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_state_with_action_slots(
    action_spec: RlActionSpec,
    state: &State,
    player_map: &PlayerMap,
    max_fleets: usize,
    planet_obs: &mut [f32],
    orbiting_planet_obs: &mut [bool],
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
    orbiting_planet_obs.fill(false);
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
        if planet.owner == -1 {
            row[13] = normalize_neutral_ships(planet.ships);
            row[14] = normalize_log_ships(planet.ships);
        } else {
            row[15] = normalize_ships(planet.ships);
            row[16] = normalize_log_ships(planet.ships);
        }
        let neutral_count_start = BASE_PLANET_CHANNELS;
        let owned_count_start =
            neutral_count_start + ship_count_feature_count(NEUTRAL_SHIP_COUNT_BUCKETS.len());
        let count_end =
            owned_count_start + ship_count_feature_count(PLANET_SHIP_COUNT_BUCKETS.len());
        if planet.owner == -1 {
            encode_ship_count_features(
                &mut row[neutral_count_start..owned_count_start],
                planet.ships,
                &NEUTRAL_SHIP_COUNT_BUCKETS,
            );
        } else {
            encode_ship_count_features(
                &mut row[owned_count_start..count_end],
                planet.ships,
                &PLANET_SHIP_COUNT_BUCKETS,
            );
        }
        let position_x = row[OWNER_CHANNELS_WITH_NEUTRAL];
        let position_y = row[OWNER_CHANNELS_WITH_NEUTRAL + 1];
        let spatial_start = count_end;
        encode_spatial_features(
            &mut row[spatial_start..spatial_start + spatial_feature_count()],
            position_x,
            position_y,
        );
        let orbiting = is_orbiting(planet.position(), planet.radius);
        encode_planet_orbital_velocity(
            &mut row[spatial_start + spatial_feature_count()..],
            position_x,
            position_y,
            state.angular_velocity,
            orbiting,
        );
        orbiting_planet_obs[planet_index] = orbiting;
    }

    for (fleet_index, fleet) in fleets.iter().take(max_fleets).enumerate() {
        fleet_mask[fleet_index] = true;
        let row_start = fleet_index * FLEET_CHANNELS;
        let row = &mut fleet_obs[row_start..row_start + FLEET_CHANNELS];

        row[player_map.owner_channel(fleet.owner)] = 1.0;
        row[OWNER_CHANNELS] = normalize_position(fleet.x);
        row[OWNER_CHANNELS + 1] = normalize_position(fleet.y);
        let speed = fleet_speed(fleet.ships, state.config.ship_speed);
        let velocity_x = (fleet.angle.cos() * speed / state.config.ship_speed) as f32;
        let velocity_y = (fleet.angle.sin() * speed / state.config.ship_speed) as f32;
        row[OWNER_CHANNELS + 2] = velocity_x;
        row[OWNER_CHANNELS + 3] = velocity_y;
        row[OWNER_CHANNELS + 4] = normalize_ships(fleet.ships);
        row[OWNER_CHANNELS + 5] = normalize_log_ships(fleet.ships);
        let count_start = BASE_FLEET_CHANNELS;
        let count_end = count_start + ship_count_feature_count(FLEET_SHIP_COUNT_BUCKET_CAPACITY);
        encode_fleet_ship_count_features(
            &mut row[count_start..count_end],
            fleet.ships,
            min_fleet_size,
        );
        let position_x = row[OWNER_CHANNELS];
        let position_y = row[OWNER_CHANNELS + 1];
        let spatial_start = count_end;
        encode_spatial_features(
            &mut row[spatial_start..spatial_start + spatial_feature_count()],
            position_x,
            position_y,
        );
        encode_fleet_motion_features(
            &mut row[spatial_start + spatial_feature_count()..],
            position_x,
            position_y,
            velocity_x,
            velocity_y,
        );
    }

    encode_comets(state, player_map, comet_obs, comet_mask);
    encode_global(state, global_obs);
    *action_slots = action_entity_slots(state);
    encode_action_spec(
        action_spec,
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

fn spatial_feature_count() -> usize {
    CARTESIAN_FOURIER_FREQUENCIES.len() * 4 + 4 + 3 * 2 + RADIAL_FOURIER_FREQUENCIES.len() * 2
}

fn encode_spatial_features(row: &mut [f32], x: f32, y: f32) {
    assert_eq!(row.len(), spatial_feature_count());

    let mut index = 0;
    for frequency in CARTESIAN_FOURIER_FREQUENCIES {
        row[index] = (std::f32::consts::PI * frequency * x).sin();
        row[index + 1] = (std::f32::consts::PI * frequency * x).cos();
        row[index + 2] = (std::f32::consts::PI * frequency * y).sin();
        row[index + 3] = (std::f32::consts::PI * frequency * y).cos();
        index += 4;
    }

    let radius = x.hypot(y);
    let theta = y.atan2(x);
    row[index] = radius;
    row[index + 1] = radius.ln_1p();
    row[index + 2] = theta.sin();
    row[index + 3] = theta.cos();
    index += 4;

    for harmonic in 2..=4 {
        let harmonic_theta = theta * harmonic as f32;
        row[index] = harmonic_theta.sin();
        row[index + 1] = harmonic_theta.cos();
        index += 2;
    }

    for frequency in RADIAL_FOURIER_FREQUENCIES {
        row[index] = (std::f32::consts::PI * frequency * radius).sin();
        row[index + 1] = (std::f32::consts::PI * frequency * radius).cos();
        index += 2;
    }

    assert_eq!(index, row.len());
}

fn encode_fleet_motion_features(row: &mut [f32], x: f32, y: f32, velocity_x: f32, velocity_y: f32) {
    assert_eq!(row.len(), 5);
    row.fill(0.0);

    let speed = velocity_x.hypot(velocity_y);
    if speed > 0.0 {
        row[1] = velocity_x / speed;
        row[2] = velocity_y / speed;
    }
    row[0] = speed;

    let radius = x.hypot(y);
    if radius == 0.0 {
        return;
    }
    let radial_x = x / radius;
    let radial_y = y / radius;
    row[3] = velocity_x * radial_x + velocity_y * radial_y;
    row[4] = velocity_x * -radial_y + velocity_y * radial_x;
}

fn encode_planet_orbital_velocity(
    row: &mut [f32],
    x: f32,
    y: f32,
    angular_velocity: f64,
    orbiting: bool,
) {
    assert_eq!(row.len(), PLANET_ORBITAL_CHANNELS);
    row.fill(0.0);
    if !orbiting {
        return;
    }
    let angular_velocity = angular_velocity as f32;
    row[0] = -angular_velocity * y;
    row[1] = angular_velocity * x;
}

fn encode_comets(
    state: &State,
    player_map: &PlayerMap,
    comet_obs: &mut [f32],
    comet_mask: &mut [bool],
) {
    for (comet_index, planet_id) in sorted_comet_planet_ids(state).into_iter().enumerate() {
        let Some(planet) = state.planets.get(planet_id) else {
            continue;
        };
        let Some((path, path_index)) = state.comets.iter().find_map(|group| {
            group
                .planet_ids
                .iter()
                .position(|candidate_id| *candidate_id == planet_id)
                .and_then(|path_offset| {
                    group
                        .paths
                        .get(path_offset)
                        .map(|path| (path, group.path_index))
                })
        }) else {
            continue;
        };

        comet_mask[comet_index] = true;
        let row_start = comet_index * COMET_CHANNELS;
        let row = &mut comet_obs[row_start..row_start + COMET_CHANNELS];

        let owner_index = player_map.owner_channel(planet.owner);
        row[owner_index] = 1.0;
        row[OWNER_CHANNELS_WITH_NEUTRAL] = normalize_ships(planet.ships);
        row[OWNER_CHANNELS_WITH_NEUTRAL + 1] = normalize_log_ships(planet.ships);
        let neutral_count_start = OWNER_CHANNELS_WITH_NEUTRAL + 2;
        let owned_count_start =
            neutral_count_start + ship_count_feature_count(NEUTRAL_SHIP_COUNT_BUCKETS.len());
        let count_end =
            owned_count_start + ship_count_feature_count(COMET_SHIP_COUNT_BUCKETS.len());
        if planet.owner == -1 {
            encode_ship_count_features(
                &mut row[neutral_count_start..owned_count_start],
                planet.ships,
                &NEUTRAL_SHIP_COUNT_BUCKETS,
            );
        } else {
            encode_ship_count_features(
                &mut row[owned_count_start..count_end],
                planet.ships,
                &COMET_SHIP_COUNT_BUCKETS,
            );
        }

        let path_start = path_index.max(0) as usize;
        let remaining_steps = path.len().saturating_sub(path_start);
        row[count_end] = remaining_steps as f32 / MAX_COMET_PATH_LENGTH as f32;
        encode_comet_path_features(&mut row[COMET_BASE_CHANNELS..], path, path_start);
    }
}

fn encode_comet_path_features(row: &mut [f32], path: &[Point], path_start: usize) {
    assert_eq!(
        row.len(),
        2 + spatial_feature_count()
            + 2
            + 5
            + COMET_SELECTED_FUTURE_OFFSETS.len()
            + COMET_SELECTED_FUTURE_OFFSETS.len() * 2
            + COMET_SELECTED_FUTURE_OFFSETS.len() * spatial_feature_count()
            + 2
    );
    row.fill(0.0);
    let Some(current) = path.get(path_start).map(normalized_point) else {
        return;
    };

    row[0] = current.0;
    row[1] = current.1;
    let spatial_start = 2;
    encode_spatial_features(
        &mut row[spatial_start..spatial_start + spatial_feature_count()],
        current.0,
        current.1,
    );

    let velocity_start = spatial_start + spatial_feature_count();
    let mut velocity_x = 0.0;
    let mut velocity_y = 0.0;
    if let Some(next) = path.get(path_start + 1).map(normalized_point) {
        velocity_x = next.0 - current.0;
        velocity_y = next.1 - current.1;
    }
    row[velocity_start] = velocity_x;
    row[velocity_start + 1] = velocity_y;
    encode_fleet_motion_features(
        &mut row[velocity_start + 2..velocity_start + 2 + 5],
        current.0,
        current.1,
        velocity_x,
        velocity_y,
    );

    let valid_start = velocity_start + 2 + 5;
    let positions_start = valid_start + COMET_SELECTED_FUTURE_OFFSETS.len();
    let selected_spatial_start = positions_start + COMET_SELECTED_FUTURE_OFFSETS.len() * 2;
    for (selected_index, offset) in COMET_SELECTED_FUTURE_OFFSETS.into_iter().enumerate() {
        let Some(position) = path.get(path_start + offset).map(normalized_point) else {
            continue;
        };
        row[valid_start + selected_index] = 1.0;

        let position_start = positions_start + selected_index * 2;
        row[position_start] = position.0;
        row[position_start + 1] = position.1;

        let selected_spatial_row_start =
            selected_spatial_start + selected_index * spatial_feature_count();
        encode_spatial_features(
            &mut row
                [selected_spatial_row_start..selected_spatial_row_start + spatial_feature_count()],
            position.0,
            position.1,
        );
    }

    let displacement_start =
        selected_spatial_start + COMET_SELECTED_FUTURE_OFFSETS.len() * spatial_feature_count();
    let final_position = normalized_point(path.last().expect("non-empty path has final position"));
    row[displacement_start] = final_position.0 - current.0;
    row[displacement_start + 1] = final_position.1 - current.1;
    assert_eq!(displacement_start + 2, row.len());
}

fn normalized_point(point: &Point) -> (f32, f32) {
    (normalize_position(point.x), normalize_position(point.y))
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

fn normalize_neutral_ships(ships: i32) -> f32 {
    ships as f32 / NEUTRAL_SHIP_NORMALIZER
}

fn normalize_ships(ships: i32) -> f32 {
    ships as f32 / SHIP_NORMALIZER
}

fn normalize_log_ships(ships: i32) -> f32 {
    ((ships.max(0) as f32) + 1.0).ln() / LOG_SHIP_NORMALIZER
}

const fn ship_count_feature_count(bucket_count: usize) -> usize {
    bucket_count * 2 + SHIP_COUNT_OVERFLOW_CHANNELS
}

fn encode_ship_count_features(row: &mut [f32], ships: i32, buckets: &[i32]) {
    assert_eq!(row.len(), ship_count_feature_count(buckets.len()));
    row.fill(0.0);

    let ships = ships.max(0);
    let max_bucket = *buckets
        .last()
        .expect("ship-count bucket grid must not be empty");
    if ships == 0 {
        row[0] = 1.0;
        row[buckets.len()] = 1.0;
    } else if ships >= max_bucket {
        row[buckets.len() - 1] = 1.0;
        row[buckets.len() * 2 - 1] = 1.0;
    } else {
        let hi = buckets
            .partition_point(|bucket| *bucket < ships)
            .min(buckets.len() - 1);
        if buckets[hi] == ships {
            row[hi] = 1.0;
            row[buckets.len() + hi] = 1.0;
        } else {
            let lo = hi - 1;
            let lo_bucket = buckets[lo] as f32;
            let hi_bucket = buckets[hi] as f32;
            let ships_f32 = ships as f32;
            let linear_hi = (ships_f32 - lo_bucket) / (hi_bucket - lo_bucket);
            row[lo] = 1.0 - linear_hi;
            row[hi] = linear_hi;

            let log_lo = lo_bucket.ln();
            let log_hi = hi_bucket.ln();
            let log_weight_hi = (ships_f32.ln() - log_lo) / (log_hi - log_lo);
            row[buckets.len() + lo] = 1.0 - log_weight_hi;
            row[buckets.len() + hi] = log_weight_hi;
        }
    }

    let overflow_start = buckets.len() * 2;
    if ships > max_bucket {
        row[overflow_start] = 1.0;
        row[overflow_start + 1] = ((ships - max_bucket).max(1) as f32).ln();
    }
}

fn encode_fleet_ship_count_features(row: &mut [f32], ships: i32, min_fleet_size: i64) {
    assert_eq!(
        row.len(),
        ship_count_feature_count(FLEET_SHIP_COUNT_BUCKET_CAPACITY)
    );
    row.fill(0.0);

    let mut buckets = [0; FLEET_SHIP_COUNT_BUCKET_CAPACITY];
    let bucket_count = fleet_ship_count_buckets(min_fleet_size, &mut buckets);
    let ships = ships.max(0);
    if ships == 0 {
        return;
    }

    let mut active = [0.0; ship_count_feature_count(FLEET_SHIP_COUNT_BUCKET_CAPACITY)];
    let active_len = ship_count_feature_count(bucket_count);
    encode_ship_count_features(&mut active[..active_len], ships, &buckets[..bucket_count]);
    row[..bucket_count].copy_from_slice(&active[..bucket_count]);
    row[FLEET_SHIP_COUNT_BUCKET_CAPACITY..FLEET_SHIP_COUNT_BUCKET_CAPACITY + bucket_count]
        .copy_from_slice(&active[bucket_count..bucket_count * 2]);
    row[FLEET_SHIP_COUNT_BUCKET_CAPACITY * 2] = active[bucket_count * 2];
    row[FLEET_SHIP_COUNT_BUCKET_CAPACITY * 2 + 1] = active[bucket_count * 2 + 1];
}

fn fleet_ship_count_buckets(min_fleet_size: i64, buckets: &mut [i32; 10]) -> usize {
    buckets[0] = 1;
    let mut len = 1;
    let min_fleet_size = min_fleet_size.clamp(1, i64::from(i32::MAX)) as u32;
    let mut next_bucket = min_fleet_size.next_power_of_two();
    if next_bucket <= min_fleet_size {
        next_bucket = next_bucket.saturating_mul(2);
    }
    while next_bucket <= FLEET_SHIP_COUNT_MAX_BUCKET as u32 && len < buckets.len() {
        buckets[len] = next_bucket
            .try_into()
            .expect("fleet ship-count bucket fits in i32");
        len += 1;
        next_bucket = next_bucket.saturating_mul(2);
    }
    len
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
pub fn encode_entity_based<'py>(
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
) -> PyResult<EncodedEntityBased<'py>> {
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
    let mut orbiting_planet_obs = Array1::<bool>::from_elem(MAX_PLANETS, false);
    let mut fleet_obs = Array2::<f32>::zeros((max_fleets, FLEET_CHANNELS));
    let mut comet_obs = Array2::<f32>::zeros((MAX_COMETS, COMET_CHANNELS));
    let mut entity_mask = Array1::<bool>::from_elem(max_entities, false);
    let mut global_obs = Array1::<f32>::zeros(GLOBAL_CHANNELS);
    let mut can_act = Array2::<bool>::from_elem((OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS), false);
    let mut max_launch = Array2::<i64>::zeros((OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS));
    let (planet_mask, tail_mask) = entity_mask
        .as_slice_mut()
        .expect("newly allocated entity mask is contiguous")
        .split_at_mut(MAX_PLANETS);
    let (comet_mask, fleet_mask) = tail_mask.split_at_mut(MAX_COMETS);

    let ignored_fleets = encode_state(
        RlActionSpec::Pure,
        &state,
        &PlayerMap::identity(),
        max_fleets,
        planet_obs
            .as_slice_mut()
            .expect("newly allocated planet array is contiguous"),
        orbiting_planet_obs
            .as_slice_mut()
            .expect("newly allocated orbiting planet array is contiguous"),
        fleet_obs
            .as_slice_mut()
            .expect("newly allocated fleet array is contiguous"),
        comet_obs
            .as_slice_mut()
            .expect("newly allocated comet array is contiguous"),
        planet_mask,
        fleet_mask,
        comet_mask,
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
        orbiting_planet_obs.into_pyarray(py),
        fleet_obs.into_pyarray(py),
        comet_obs.into_pyarray(py),
        entity_mask.into_pyarray(py),
        global_obs.into_pyarray(py),
        can_act.into_pyarray(py),
        max_launch.into_pyarray(py),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::state::{Fleet, Planet, SimConfig, State};

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "expected {actual} to be within 1e-6 of {expected}"
        );
    }

    #[test]
    fn angular_velocity_normalization_maps_generated_range_to_zero_one() {
        assert_eq!(normalize_angular_velocity(0.025), 0.0);
        assert_eq!(normalize_angular_velocity(0.05), 1.0);
    }

    #[test]
    fn spatial_feature_count_matches_public_channel_widths() {
        assert_eq!(
            BASE_PLANET_CHANNELS
                + ship_count_feature_count(NEUTRAL_SHIP_COUNT_BUCKETS.len())
                + ship_count_feature_count(PLANET_SHIP_COUNT_BUCKETS.len())
                + spatial_feature_count()
                + PLANET_ORBITAL_CHANNELS,
            PLANET_CHANNELS
        );
        assert_eq!(
            BASE_FLEET_CHANNELS
                + ship_count_feature_count(FLEET_SHIP_COUNT_BUCKET_CAPACITY)
                + spatial_feature_count()
                + 5,
            FLEET_CHANNELS
        );
    }

    #[test]
    fn ship_count_features_two_hot_linear_and_log_space() {
        let mut row = [0.0; 26];

        encode_ship_count_features(&mut row, 64, &PLANET_SHIP_COUNT_BUCKETS);

        assert_close(row[7], 1.0);
        assert_close(row[PLANET_SHIP_COUNT_BUCKETS.len() + 7], 1.0);

        encode_ship_count_features(&mut row, 96, &PLANET_SHIP_COUNT_BUCKETS);

        assert_close(row[7], 0.5);
        assert_close(row[8], 0.5);
        let log_hi = (96.0_f32.ln() - 64.0_f32.ln()) / (128.0_f32.ln() - 64.0_f32.ln());
        assert_close(row[PLANET_SHIP_COUNT_BUCKETS.len() + 7], 1.0 - log_hi);
        assert_close(row[PLANET_SHIP_COUNT_BUCKETS.len() + 8], log_hi);
    }

    #[test]
    fn ship_count_features_use_zero_bucket_and_overflow_channels() {
        let mut row = [1.0; 26];

        encode_ship_count_features(&mut row, 0, &PLANET_SHIP_COUNT_BUCKETS);

        assert_close(row[0], 1.0);
        assert_close(row[PLANET_SHIP_COUNT_BUCKETS.len()], 1.0);
        assert_eq!(row.iter().filter(|value| **value != 0.0).count(), 2);

        encode_ship_count_features(&mut row, 1200, &PLANET_SHIP_COUNT_BUCKETS);

        assert_close(row[PLANET_SHIP_COUNT_BUCKETS.len() - 1], 1.0);
        assert_close(row[PLANET_SHIP_COUNT_BUCKETS.len() * 2 - 1], 1.0);
        assert_close(row[PLANET_SHIP_COUNT_BUCKETS.len() * 2], 1.0);
        assert_close(
            row[PLANET_SHIP_COUNT_BUCKETS.len() * 2 + 1],
            (176.0_f32).ln(),
        );
    }

    #[test]
    fn fleet_ship_count_features_start_at_one_then_next_power_above_min_fleet_size() {
        let mut row = [1.0; 22];

        encode_fleet_ship_count_features(&mut row, 2, 4);

        assert_close(row[0], 6.0 / 7.0);
        assert_close(row[1], 1.0 / 7.0);
        assert_close(row[FLEET_SHIP_COUNT_BUCKET_CAPACITY], 2.0 / 3.0);
        assert_close(row[FLEET_SHIP_COUNT_BUCKET_CAPACITY + 1], 1.0 / 3.0);
        assert_eq!(row.iter().filter(|value| **value != 0.0).count(), 4);

        encode_fleet_ship_count_features(&mut row, 6, 4);

        assert_close(row[0], 2.0 / 7.0);
        assert_close(row[1], 5.0 / 7.0);
        let log_hi = 6.0_f32.ln() / 8.0_f32.ln();
        assert_close(row[FLEET_SHIP_COUNT_BUCKET_CAPACITY], 1.0 - log_hi);
        assert_close(row[FLEET_SHIP_COUNT_BUCKET_CAPACITY + 1], log_hi);
    }

    #[test]
    fn fleet_motion_features_zero_radial_basis_at_sun_center() {
        let mut row = [1.0; 5];

        encode_fleet_motion_features(&mut row, 0.0, 0.0, 0.3, 0.4);

        assert_close(row[0], 0.5);
        assert_close(row[1], 0.6);
        assert_close(row[2], 0.8);
        assert_eq!(row[3], 0.0);
        assert_eq!(row[4], 0.0);
    }

    #[test]
    fn fleet_motion_features_zero_heading_for_stationary_fleet() {
        let mut row = [1.0; 5];

        encode_fleet_motion_features(&mut row, 1.0, 0.0, 0.0, 0.0);

        assert_eq!(row, [0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn planet_orbital_velocity_uses_normalized_tangent_direction_only_for_orbiting_planets() {
        let mut row = [1.0; PLANET_ORBITAL_CHANNELS];

        encode_planet_orbital_velocity(&mut row, 0.5, -0.25, 0.04, true);

        assert_close(row[0], 0.01);
        assert_close(row[1], 0.02);

        encode_planet_orbital_velocity(&mut row, 0.5, -0.25, 0.04, false);

        assert_eq!(row, [0.0, 0.0]);
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
        let mut orbiting_planet_obs = vec![false; MAX_PLANETS];
        let mut fleet_obs = Vec::new();
        let mut comet_obs = vec![0.0; MAX_COMETS * COMET_CHANNELS];
        let mut planet_mask = vec![false; MAX_PLANETS];
        let mut fleet_mask = Vec::new();
        let mut comet_mask = vec![false; MAX_COMETS];
        let mut global_obs = vec![0.0; GLOBAL_CHANNELS];
        let mut can_act = vec![false; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut max_launch = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];

        encode_state(
            RlActionSpec::Pure,
            &state,
            &PlayerMap::identity(),
            0,
            &mut planet_obs,
            &mut orbiting_planet_obs,
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
        let mut orbiting_planet_obs = vec![false; MAX_PLANETS];
        let mut fleet_obs = vec![0.0; FLEET_CHANNELS];
        let mut comet_obs = vec![0.0; MAX_COMETS * COMET_CHANNELS];
        let mut planet_mask = vec![false; MAX_PLANETS];
        let mut fleet_mask = vec![false; 1];
        let mut comet_mask = vec![false; MAX_COMETS];
        let mut global_obs = vec![0.0; GLOBAL_CHANNELS];
        let mut can_act = vec![false; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut max_launch = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];

        encode_state(
            RlActionSpec::Pure,
            &state,
            &player_map,
            1,
            &mut planet_obs,
            &mut orbiting_planet_obs,
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
        let mut orbiting_planet_obs = vec![false; MAX_PLANETS];
        let mut fleet_obs = Vec::new();
        let mut comet_obs = vec![0.0; MAX_COMETS * COMET_CHANNELS];
        let mut planet_mask = vec![false; MAX_PLANETS];
        let mut fleet_mask = Vec::new();
        let mut comet_mask = vec![false; MAX_COMETS];
        let mut global_obs = vec![0.0; GLOBAL_CHANNELS];
        let mut can_act = vec![false; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut max_launch = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];

        encode_state(
            RlActionSpec::Pure,
            &state,
            &PlayerMap::identity(),
            0,
            &mut planet_obs,
            &mut orbiting_planet_obs,
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
