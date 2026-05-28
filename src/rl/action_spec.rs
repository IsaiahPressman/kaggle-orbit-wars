use std::collections::HashSet;
use std::f64::consts::PI;

use crate::rules_engine::env::PlayerAction;
use crate::rules_engine::state::{
    LaunchAction, Planet, Point, State, StaticTargetArc, BOARD_SIZE, CENTER,
};
use crate::rules_engine::utils::{
    angle_between, best_static_target_angle, distance, fleet_speed, is_orbiting, launch_start,
    orbit_position, point_along, point_to_segment_distance, swept_pair_hit,
};

use super::{PlayerMap, ACTION_ENTITY_SLOTS, MAX_COMETS, MAX_PLANETS, OUTER_PLAYER_SLOTS};

const ROOT_EPS: f64 = 1e-7;
const QUADRATIC_EPS: f64 = 1e-12;
const DYNAMIC_TARGET_EPS: f64 = 1e-6;
const COLLISION_EPS: f64 = 1e-6;
const ANGLE_EPS: f64 = 1e-9;
const ANGLE_CHOICE_EPS: f64 = 1e-4;
const MAX_ANGLE_EDGE_MARGIN: f64 = 2e-2;
const TARGET_WINDOW_LIMIT: usize = 32;
const DYNAMIC_BLOCKER_SAMPLE_FRACTIONS: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const TAU: f64 = PI * 2.0;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RlActionSpec {
    Pure,
    DiscreteTargets {
        targeting_mode: TargetingMode,
    },
    DiscreteTargetBins {
        n_bins: usize,
        targeting_mode: TargetingMode,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TargetingMode {
    AnythingGoes,
    StopBadLaunch,
    FullMask,
}

impl TargetingMode {
    pub(super) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "anything_goes" => Ok(Self::AnythingGoes),
            "stop_bad_launch" => Ok(Self::StopBadLaunch),
            "full_mask" => Ok(Self::FullMask),
            _ => Err(format!(
                "unsupported targeting_mode {value:?}; expected \"anything_goes\", \"stop_bad_launch\", or \"full_mask\""
            )),
        }
    }

    const fn uses_full_mask(self) -> bool {
        matches!(self, Self::FullMask)
    }
}

impl RlActionSpec {
    pub(super) fn parse(value: &str, n_bins: usize, targeting_mode: &str) -> Result<Self, String> {
        match value {
            "pure" => {
                TargetingMode::parse(targeting_mode)?;
                Ok(Self::Pure)
            }
            "discrete_targets" => Ok(Self::DiscreteTargets {
                targeting_mode: TargetingMode::parse(targeting_mode)?,
            }),
            "discrete_target_bins" => {
                let targeting_mode = TargetingMode::parse(targeting_mode)?;
                if n_bins < 2 {
                    return Err("n_bins must be >= 2 for action_spec \"discrete_target_bins\""
                        .to_string());
                }
                let spec = Self::DiscreteTargetBins {
                    n_bins,
                    targeting_mode,
                };
                spec.checked_can_act_len().ok_or_else(|| {
                    "n_bins is too large for the discrete_target_bins can_act shape".to_string()
                })?;
                Ok(spec)
            },
            _ => Err(format!(
                "unsupported action_spec {value:?}; expected \"pure\", \"discrete_targets\", or \"discrete_target_bins\""
            )),
        }
    }

    pub(super) fn can_act_len(self) -> usize {
        self.checked_can_act_len()
            .expect("validated action spec can_act shape should fit in usize")
    }

    fn checked_can_act_len(self) -> Option<usize> {
        match self {
            Self::Pure => OUTER_PLAYER_SLOTS.checked_mul(ACTION_ENTITY_SLOTS),
            Self::DiscreteTargets { .. } => OUTER_PLAYER_SLOTS
                .checked_mul(ACTION_ENTITY_SLOTS)?
                .checked_mul(ACTION_ENTITY_SLOTS),
            Self::DiscreteTargetBins { n_bins, .. } => OUTER_PLAYER_SLOTS
                .checked_mul(ACTION_ENTITY_SLOTS)?
                .checked_mul(ACTION_ENTITY_SLOTS)?
                .checked_mul(n_bins),
        }
    }

    pub(super) const fn max_launch_len(self) -> usize {
        match self {
            Self::Pure | Self::DiscreteTargets { .. } => OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS,
            Self::DiscreteTargetBins { .. } => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ActionEntitySlot {
    pub(super) planet_id: u32,
}

pub(super) type ActionEntitySlots = [Option<ActionEntitySlot>; ACTION_ENTITY_SLOTS];

#[derive(Debug)]
pub(super) struct DecodedDiscreteTargetActions {
    pub(super) actions: Vec<PlayerAction>,
    pub(super) launch_failures: u32,
}

pub(super) fn fleet_bin_to_ships(fleet_bin: usize, available_ships: i64, n_bins: usize) -> i64 {
    debug_assert!(n_bins >= 2);
    if fleet_bin == 0 || available_ships <= 0 {
        return 0;
    }
    let numerator = fleet_bin as u128 * available_ships as u128;
    let denominator = (n_bins - 1) as u128;
    ((numerator * 2 + denominator) / (2 * denominator)) as i64
}

#[cfg(test)]
fn ships_to_fleet_bin(ships: i64, available_ships: i64, n_bins: usize) -> usize {
    debug_assert!(n_bins >= 2);
    if ships <= 0 || available_ships <= 0 {
        return 0;
    }
    let clamped_ships = ships.min(available_ships) as u128;
    let numerator = clamped_ships * (n_bins - 1) as u128;
    let denominator = available_ships as u128;
    ((numerator * 2 + denominator) / (2 * denominator)) as usize
}

fn fleet_bin_keeps_ship_count(fleet_bin: usize, available_ships: i64, n_bins: usize) -> bool {
    let ship_count = fleet_bin_to_ships(fleet_bin, available_ships, n_bins);
    (fleet_bin + 1..n_bins)
        .all(|later_bin| fleet_bin_to_ships(later_bin, available_ships, n_bins) != ship_count)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_pure_actions(
    state: &State,
    player_map: &PlayerMap,
    entities: &ActionEntitySlots,
    launch: &[bool],
    angle: &[f32],
    ships: &[i64],
    max_per_planet_launches: usize,
    min_fleet_size: i64,
) -> Result<Vec<PlayerAction>, String> {
    let mut actions = vec![Vec::new(); state.config.player_count];
    for outer_player in 0..OUTER_PLAYER_SLOTS {
        let player_offset = outer_player * ACTION_ENTITY_SLOTS * max_per_planet_launches;
        let Some(internal_player) = player_map
            .outer_to_internal(outer_player)
            .filter(|player| *player < state.config.player_count)
        else {
            let player_launches = &launch
                [player_offset..player_offset + ACTION_ENTITY_SLOTS * max_per_planet_launches];
            if player_launches.iter().any(|launched| *launched) {
                return Err(format!("player slot {outer_player} is inactive"));
            }
            continue;
        };
        let player_actions = &mut actions[internal_player];
        for (entity_index, planet) in entities.iter().enumerate() {
            let entity_offset = player_offset + entity_index * max_per_planet_launches;
            let mut spent_ships = 0_i64;
            for launch_index in 0..max_per_planet_launches {
                let action_index = entity_offset + launch_index;
                if !launch[action_index] {
                    break;
                }
                let Some(slot) = planet else {
                    return Err(format!(
                        "player {outer_player} cannot launch from empty action entity slot {entity_index}"
                    ));
                };
                let Some(planet) = planet_for_slot(state, *slot) else {
                    return Err(format!(
                        "player {outer_player} cannot launch from stale action entity slot {entity_index}"
                    ));
                };
                let ship_count = ships[action_index];
                if ship_count < min_fleet_size {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} ships must be >= {min_fleet_size}"
                    ));
                }
                let ship_count_i32 = i32::try_from(ship_count).map_err(|_| {
                    format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} ships must fit in i32"
                    )
                })?;
                let launch_angle = angle[action_index];
                if !launch_angle.is_finite() {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} angle must be finite"
                    ));
                }
                if planet.owner != internal_player as i32 {
                    return Err(format!(
                        "player {outer_player} cannot launch from planet {} owned by {}",
                        planet.id, planet.owner
                    ));
                }
                spent_ships += ship_count;
                if spent_ships > i64::from(planet.ships) {
                    return Err(format!(
                        "planet {} has {} ships, cannot launch {spent_ships}",
                        planet.id, planet.ships
                    ));
                }
                player_actions.push(LaunchAction {
                    from_planet_id: planet.id,
                    angle: f64::from(launch_angle),
                    ships: ship_count_i32,
                });
            }
        }
    }
    Ok(actions)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_discrete_target_actions(
    state: &State,
    player_map: &PlayerMap,
    entities: &ActionEntitySlots,
    launch: &[bool],
    target: &[i64],
    ships: &[i64],
    max_per_planet_launches: usize,
    min_fleet_size: i64,
    targeting_mode: TargetingMode,
) -> Result<DecodedDiscreteTargetActions, String> {
    let mut actions = vec![Vec::new(); state.config.player_count];
    let mut launch_failures = 0_u32;
    let mut orbit_target_cache = OrbitTargetCache::new(state, entities, min_fleet_size);
    for outer_player in 0..OUTER_PLAYER_SLOTS {
        let player_offset = outer_player * ACTION_ENTITY_SLOTS * max_per_planet_launches;
        let Some(internal_player) = player_map
            .outer_to_internal(outer_player)
            .filter(|player| *player < state.config.player_count)
        else {
            let player_launches = &launch
                [player_offset..player_offset + ACTION_ENTITY_SLOTS * max_per_planet_launches];
            if player_launches.iter().any(|launched| *launched) {
                return Err(format!("player slot {outer_player} is inactive"));
            }
            continue;
        };
        let player_actions = &mut actions[internal_player];
        for (entity_index, source_slot) in entities.iter().enumerate() {
            let entity_offset = player_offset + entity_index * max_per_planet_launches;
            let mut spent_ships = 0_i64;
            for launch_index in 0..max_per_planet_launches {
                let action_index = entity_offset + launch_index;
                if !launch[action_index] {
                    break;
                }
                let Some(source_slot) = source_slot else {
                    return Err(format!(
                        "player {outer_player} cannot launch from empty action entity slot {entity_index}"
                    ));
                };
                let Some(source) = planet_for_slot(state, *source_slot) else {
                    return Err(format!(
                        "player {outer_player} cannot launch from stale action entity slot {entity_index}"
                    ));
                };
                let target_index = target[action_index];
                if !(0..ACTION_ENTITY_SLOTS as i64).contains(&target_index) {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} target must be in [0, {ACTION_ENTITY_SLOTS})"
                    ));
                }
                let target_index = target_index as usize;
                if target_index == entity_index {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} cannot target itself"
                    ));
                }
                let Some(target_slot) = entities[target_index] else {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} cannot target empty action entity slot {target_index}"
                    ));
                };
                let Some(target_planet) = planet_for_slot(state, target_slot) else {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} cannot target stale action entity slot {target_index}"
                    ));
                };
                let ship_count = ships[action_index];
                if ship_count < min_fleet_size {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} ships must be >= {min_fleet_size}"
                    ));
                }
                let ship_count_i32 = i32::try_from(ship_count).map_err(|_| {
                    format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} ships must fit in i32"
                    )
                })?;
                if source.owner != internal_player as i32 {
                    return Err(format!(
                        "player {outer_player} cannot launch from planet {} owned by {}",
                        source.id, source.owner
                    ));
                }
                if spent_ships + ship_count > i64::from(source.ships) {
                    return Err(format!(
                        "planet {} has {} ships, cannot launch {}",
                        source.id,
                        source.ships,
                        spent_ships + ship_count
                    ));
                }
                let Some(angle) = target_angle(
                    state,
                    source,
                    target_planet,
                    ship_count_i32,
                    &mut orbit_target_cache,
                    targeting_mode,
                )?
                else {
                    launch_failures += 1;
                    continue;
                };
                spent_ships += ship_count;
                player_actions.push(LaunchAction {
                    from_planet_id: source.id,
                    angle,
                    ships: ship_count_i32,
                });
            }
        }
    }
    Ok(DecodedDiscreteTargetActions {
        actions,
        launch_failures,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_discrete_target_bin_actions(
    state: &State,
    player_map: &PlayerMap,
    entities: &ActionEntitySlots,
    target: &[i64],
    fleet_bin: &[i64],
    n_bins: usize,
    min_fleet_size: i64,
    targeting_mode: TargetingMode,
) -> Result<DecodedDiscreteTargetActions, String> {
    if n_bins < 2 {
        return Err("n_bins must be >= 2".to_string());
    }
    let mut actions = vec![Vec::new(); state.config.player_count];
    let mut launch_failures = 0_u32;
    let mut orbit_target_cache = OrbitTargetCache::new(state, entities, min_fleet_size);
    let mut entity_planets = [None; ACTION_ENTITY_SLOTS];
    let mut entity_static = [false; ACTION_ENTITY_SLOTS];
    for (entity_index, slot) in entities.iter().enumerate() {
        if let Some(planet) = slot.and_then(|slot| planet_for_slot(state, slot)) {
            entity_planets[entity_index] = Some(planet);
            entity_static[entity_index] = is_static_planet_cached(state, planet);
        }
    }
    for outer_player in 0..OUTER_PLAYER_SLOTS {
        let player_offset = outer_player * ACTION_ENTITY_SLOTS;
        let Some(internal_player) = player_map
            .outer_to_internal(outer_player)
            .filter(|player| *player < state.config.player_count)
        else {
            for entity_index in 0..ACTION_ENTITY_SLOTS {
                let action_index = player_offset + entity_index;
                let fleet_bin = fleet_bin[action_index];
                if !(0..n_bins as i64).contains(&fleet_bin) {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} fleet_bin must be in [0, {n_bins})"
                    ));
                }
                if fleet_bin != 0 {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} cannot launch from inactive player slot"
                    ));
                }
            }
            continue;
        };
        let player_actions = &mut actions[internal_player];
        for (entity_index, source_slot) in entities.iter().enumerate() {
            let action_index = player_offset + entity_index;
            let fleet_bin = fleet_bin[action_index];
            if !(0..n_bins as i64).contains(&fleet_bin) {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} fleet_bin must be in [0, {n_bins})"
                ));
            }
            let fleet_bin = fleet_bin as usize;
            if fleet_bin == 0 {
                continue;
            }
            let Some(source_slot) = source_slot else {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} cannot launch from empty action entity slot"
                ));
            };
            let Some(source) = planet_for_slot(state, *source_slot) else {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} cannot launch from stale action entity slot"
                ));
            };
            if source.owner != internal_player as i32 {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} cannot launch from planet {} owned by {}",
                    source.id, source.owner
                ));
            }
            if i64::from(source.ships) < min_fleet_size {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} planet {} has {} ships, below min_fleet_size {min_fleet_size}",
                    source.id, source.ships
                ));
            }
            let ship_count = fleet_bin_to_ships(fleet_bin, i64::from(source.ships), n_bins);
            let target_index = target[action_index];
            if !(0..ACTION_ENTITY_SLOTS as i64).contains(&target_index) {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} target must be in [0, {ACTION_ENTITY_SLOTS})"
                ));
            }
            let target_index = target_index as usize;
            if target_index == entity_index {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} cannot target itself"
                ));
            }
            let Some(target_slot) = entities[target_index] else {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} cannot target empty action entity slot {target_index}"
                ));
            };
            let Some(target_planet) = planet_for_slot(state, target_slot) else {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} cannot target stale action entity slot {target_index}"
                ));
            };
            if !target_eligible_for_mode(
                state,
                source,
                entity_index,
                target_index,
                entity_static[entity_index],
                &entity_planets,
                &entity_static,
                targeting_mode,
            ) {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} target-bin pair is masked by can_act"
                ));
            }
            if ship_count < min_fleet_size {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} fleet_bin {fleet_bin} maps to {ship_count} ships, below min_fleet_size {min_fleet_size}"
                ));
            }
            let ship_count_i32 = i32::try_from(ship_count).map_err(|_| {
                format!(
                    "player {outer_player} entity slot {entity_index} fleet_bin {fleet_bin} maps to ships that must fit in i32"
                )
            })?;
            if !fleet_bin_keeps_ship_count(fleet_bin, i64::from(source.ships), n_bins) {
                return Err(format!(
                    "player {outer_player} entity slot {entity_index} fleet_bin {fleet_bin} duplicates a higher bin"
                ));
            }
            let Some(angle) = target_angle(
                state,
                source,
                target_planet,
                ship_count_i32,
                &mut orbit_target_cache,
                targeting_mode,
            )?
            else {
                launch_failures += 1;
                continue;
            };
            player_actions.push(LaunchAction {
                from_planet_id: source.id,
                angle,
                ships: ship_count_i32,
            });
        }
    }
    Ok(DecodedDiscreteTargetActions {
        actions,
        launch_failures,
    })
}

pub(super) fn encode_action_spec(
    action_spec: RlActionSpec,
    state: &State,
    player_map: &PlayerMap,
    entities: &ActionEntitySlots,
    can_act: &mut [bool],
    mut max_launch: Option<&mut [i64]>,
    min_fleet_size: i64,
) {
    let mut entity_planets = [None; ACTION_ENTITY_SLOTS];
    let mut entity_static = [false; ACTION_ENTITY_SLOTS];
    if matches!(
        action_spec,
        RlActionSpec::DiscreteTargets { .. } | RlActionSpec::DiscreteTargetBins { .. }
    ) {
        for (entity_index, slot) in entities.iter().enumerate() {
            if let Some(planet) = slot.and_then(|slot| planet_for_slot(state, slot)) {
                entity_planets[entity_index] = Some(planet);
                entity_static[entity_index] = is_static_planet_cached(state, planet);
            }
        }
    }

    for (entity_index, slot) in entities.iter().enumerate() {
        let Some(slot) = slot else {
            continue;
        };
        let Some(planet) = planet_for_slot(state, *slot) else {
            continue;
        };
        if i64::from(planet.ships) < min_fleet_size || planet.owner < 0 {
            continue;
        }
        let player = planet.owner as usize;
        if player >= state.config.player_count {
            continue;
        }
        let mut source_can_act = false;
        match action_spec {
            RlActionSpec::Pure => {
                let index =
                    player_map.internal_to_outer(player) * ACTION_ENTITY_SLOTS + entity_index;
                can_act[index] = true;
                source_can_act = true;
            },
            RlActionSpec::DiscreteTargets { targeting_mode } => {
                let base = (player_map.internal_to_outer(player) * ACTION_ENTITY_SLOTS
                    + entity_index)
                    * ACTION_ENTITY_SLOTS;
                let source_static = entity_static[entity_index];
                for target_index in 0..ACTION_ENTITY_SLOTS {
                    let eligible = target_eligible_for_mode(
                        state,
                        planet,
                        entity_index,
                        target_index,
                        source_static,
                        &entity_planets,
                        &entity_static,
                        targeting_mode,
                    );
                    can_act[base + target_index] = eligible;
                    source_can_act |= eligible;
                }
            },
            RlActionSpec::DiscreteTargetBins {
                n_bins,
                targeting_mode,
            } => {
                let target_base = (player_map.internal_to_outer(player) * ACTION_ENTITY_SLOTS
                    + entity_index)
                    * ACTION_ENTITY_SLOTS
                    * n_bins;
                let source_static = entity_static[entity_index];
                for target_index in 0..ACTION_ENTITY_SLOTS {
                    let eligible = target_eligible_for_mode(
                        state,
                        planet,
                        entity_index,
                        target_index,
                        source_static,
                        &entity_planets,
                        &entity_static,
                        targeting_mode,
                    );
                    let bin_base = target_base + target_index * n_bins;
                    can_act[bin_base] = eligible;
                    for fleet_bin in 1..n_bins {
                        let ship_count =
                            fleet_bin_to_ships(fleet_bin, i64::from(planet.ships), n_bins);
                        can_act[bin_base + fleet_bin] = eligible
                            && ship_count >= min_fleet_size
                            && fleet_bin_keeps_ship_count(
                                fleet_bin,
                                i64::from(planet.ships),
                                n_bins,
                            );
                    }
                    source_can_act |= eligible;
                }
            },
        }
        if source_can_act && action_spec.max_launch_len() > 0 {
            let index = player_map.internal_to_outer(player) * ACTION_ENTITY_SLOTS + entity_index;
            max_launch
                .as_deref_mut()
                .expect("max_launch buffer required for this action spec")[index] =
                i64::from(planet.ships);
        }
    }
}

pub(super) fn action_entity_slots(state: &State) -> ActionEntitySlots {
    let comet_ids = state
        .comet_planet_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut entities = [None; ACTION_ENTITY_SLOTS];
    for (entity_index, planet) in state
        .planets
        .iter()
        .filter(|planet| !comet_ids.contains(&planet.id))
        .take(MAX_PLANETS)
        .enumerate()
    {
        entities[entity_index] = Some(ActionEntitySlot {
            planet_id: planet.id,
        });
    }

    for (comet_index, planet_id) in sorted_comet_planet_ids(state).into_iter().enumerate() {
        entities[MAX_PLANETS + comet_index] = Some(ActionEntitySlot { planet_id });
    }
    entities
}

#[allow(clippy::too_many_arguments)]
fn target_eligible_for_mode(
    state: &State,
    source: &Planet,
    source_index: usize,
    target_index: usize,
    source_static: bool,
    entity_planets: &[Option<&Planet>; ACTION_ENTITY_SLOTS],
    entity_static: &[bool; ACTION_ENTITY_SLOTS],
    targeting_mode: TargetingMode,
) -> bool {
    if target_index == source_index {
        return false;
    }
    let Some(target) = entity_planets[target_index] else {
        return false;
    };
    if !targeting_mode.uses_full_mask() {
        return true;
    }
    if !entity_static[target_index] {
        return true;
    }
    if source_static && !state.static_target_cache.is_empty() {
        state
            .static_target_cache
            .get(source.id, target.id)
            .is_some()
    } else {
        best_live_static_target_angle(state, source, target).is_some()
    }
}

pub(super) fn sorted_comet_planet_ids(state: &State) -> Vec<u32> {
    let mut planet_ids = state
        .comet_planet_ids
        .iter()
        .copied()
        .filter(|planet_id| state.planets.get(*planet_id).is_some())
        .collect::<Vec<_>>();
    planet_ids.sort_unstable();
    planet_ids.dedup();
    planet_ids.truncate(MAX_COMETS);
    planet_ids
}

fn planet_for_slot(state: &State, slot: ActionEntitySlot) -> Option<&Planet> {
    state.planets.get(slot.planet_id)
}

#[derive(Clone, Copy, Debug)]
struct TargetCandidate {
    angle: f64,
    end: Point,
    time: f64,
}

#[derive(Clone, Copy, Debug)]
struct TargetWindow {
    start_time: f64,
    end_time: f64,
    segment_index: usize,
    segment_start: Point,
    segment_end: Point,
}

#[derive(Clone, Copy, Debug)]
struct AngleSpan {
    start: f64,
    end: f64,
}

#[derive(Clone, Copy, Debug)]
struct InstantArc {
    center: f64,
    half_angle: f64,
}

#[derive(Clone, Copy, Debug)]
struct CircleBlocker {
    id: u32,
    center: Point,
    radius: f64,
}

struct OrbitTargetCache<'a> {
    state: &'a State,
    entities: &'a ActionEntitySlots,
    min_fleet_size: i32,
    sources: Option<Vec<&'a Planet>>,
    min_speed: f64,
    static_blockers: Vec<CircleBlocker>,
    dynamic_blocker_ids: Vec<u32>,
    comet_paths: Vec<Option<&'a [Point]>>,
    orbit_path_indices: Vec<Option<usize>>,
    paths: Vec<(u32, Vec<Point>)>,
}

impl<'a> OrbitTargetCache<'a> {
    fn new(state: &'a State, entities: &'a ActionEntitySlots, min_fleet_size: i64) -> Self {
        let min_fleet_size = min_fleet_size.clamp(1, i64::from(i32::MAX)) as i32;
        let mut static_blockers = Vec::new();
        let mut dynamic_blocker_ids = Vec::new();
        for planet in state.planets.iter() {
            if is_static_planet_cached(state, planet) {
                static_blockers.push(CircleBlocker {
                    id: planet.id,
                    center: planet.position(),
                    radius: planet.radius + COLLISION_EPS,
                });
            } else {
                dynamic_blocker_ids.push(planet.id);
            }
        }
        let mut comet_paths = vec![None; crate::rules_engine::state::MAX_PLANET_ID as usize];
        for group in &state.comets {
            let path_start = group.path_index.max(0) as usize;
            for (path_offset, planet_id) in group.planet_ids.iter().copied().enumerate() {
                let Some(path) = group.paths.get(path_offset) else {
                    continue;
                };
                if path_start < path.len() {
                    comet_paths[planet_id as usize] = Some(&path[path_start..]);
                }
            }
        }
        let mut orbit_path_indices = vec![None; crate::rules_engine::state::MAX_PLANET_ID as usize];
        for (index, path) in state.orbit_paths.iter().enumerate() {
            orbit_path_indices[path.planet_id as usize] = Some(index);
        }
        Self {
            state,
            entities,
            min_fleet_size,
            sources: None,
            min_speed: fleet_speed(min_fleet_size, state.config.ship_speed),
            static_blockers,
            dynamic_blocker_ids,
            comet_paths,
            orbit_path_indices,
            paths: Vec::new(),
        }
    }

    fn path_for(&mut self, target: &Planet) -> Option<&[Point]> {
        if let Some(index) = self.paths.iter().position(|(id, _)| *id == target.id) {
            return Some(self.paths[index].1.as_slice());
        }

        if let Some(path_index) = self
            .orbit_path_indices
            .get(target.id as usize)
            .and_then(|index| *index)
        {
            let path = &self.state.orbit_paths[path_index];
            let start = self.state.step.saturating_sub(1) as usize;
            if self.state.step == 0 && path.points.len() >= 2 {
                let mut points = Vec::with_capacity(path.points.len() + 1);
                points.push(target.position());
                points.push(target.position());
                points.extend_from_slice(&path.points[1..]);
                self.paths.push((target.id, points));
                return self.paths.last().map(|(_, path)| path.as_slice());
            }
            if start < path.points.len() {
                return Some(&path.points[start..]);
            }
        }

        let initial_target = self.state.initial_planets.get(target.id)?;
        if self.sources.is_none() {
            self.sources = Some(
                self.entities
                    .iter()
                    .filter_map(|slot| slot.and_then(|slot| planet_for_slot(self.state, slot)))
                    .filter(|planet| planet.owner >= 0 && planet.ships >= self.min_fleet_size)
                    .collect(),
            );
        }
        let max_time = self
            .sources
            .as_ref()?
            .iter()
            .map(|source| orbit_time_bounds(source, initial_target, target, self.min_speed).1)
            .fold(0.0, f64::max);
        let point_count = max_time.ceil().max(1.0) as usize + 1;
        let path = (0..point_count)
            .map(|tick| orbit_target_at(self.state, initial_target, target, tick as f64))
            .collect::<Vec<_>>();
        self.paths.push((target.id, path));
        self.paths.last().map(|(_, path)| path.as_slice())
    }

    fn comet_path_for(&self, target: &Planet) -> Option<&'a [Point]> {
        self.comet_paths
            .get(target.id as usize)
            .and_then(|path| *path)
    }

    fn dynamic_path_for(&mut self, blocker: &Planet) -> Option<&[Point]> {
        if let Some(path) = self.comet_path_for(blocker) {
            return Some(path);
        }
        self.path_for(blocker)
    }
}

fn target_angle(
    state: &State,
    source: &Planet,
    target: &Planet,
    ships: i32,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
    targeting_mode: TargetingMode,
) -> Result<Option<f64>, String> {
    let speed = fleet_speed(ships, state.config.ship_speed);
    if is_dynamic_planet_cached(state, target) {
        return Ok(dynamic_target_angle(
            state,
            source,
            target,
            speed,
            orbit_target_cache,
            targeting_mode,
        ));
    }

    Ok(static_target_angle(
        state,
        source,
        target,
        speed,
        orbit_target_cache,
        targeting_mode,
    ))
}

fn dynamic_target_angle(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
    targeting_mode: TargetingMode,
) -> Option<f64> {
    if state.comet_planet_ids.contains(&target.id) {
        let path = orbit_target_cache.comet_path_for(target)?;
        let windows = piecewise_linear_target_windows(source, target.radius, speed, path);
        if windows.is_empty() {
            return None;
        }
        return choose_dynamic_window_angle(
            state,
            source,
            target,
            speed,
            &windows,
            orbit_target_cache,
            targeting_mode,
        );
    }

    let windows = orbiting_target_windows(state, source, target, speed, orbit_target_cache)?;
    choose_dynamic_window_angle(
        state,
        source,
        target,
        speed,
        &windows,
        orbit_target_cache,
        targeting_mode,
    )
}

fn direct_target_angle(source: &Planet, target: &Planet) -> f64 {
    angle_between(source.position(), target.position())
}

fn static_target_candidate(
    source: &Planet,
    target: &Planet,
    angle: f64,
    speed: f64,
) -> TargetCandidate {
    let start = launch_start(source, angle);
    let target_pos = target.position();
    let distance_to_target = distance(start, target_pos);
    let end_distance = (distance_to_target - target.radius).max(0.0);
    TargetCandidate {
        angle,
        end: point_along(start, angle, end_distance),
        time: end_distance / speed.max(f64::EPSILON),
    }
}

fn static_target_angle(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
    targeting_mode: TargetingMode,
) -> Option<f64> {
    let reference_angle = direct_target_angle(source, target);
    let max_time = static_target_max_time(source, target, speed);
    let midpoint_candidate = static_target_candidate(source, target, reference_angle, speed);
    if !is_dynamic_planet_cached(state, source)
        && static_target_cache_contains(state, source.id, target.id, 0.0)
        && !goes_out_of_bounds(source, midpoint_candidate)
        && static_candidate_has_target_margin(source, target, reference_angle, midpoint_candidate)
        && !shot_hits_dynamic_blocker(
            state,
            source,
            target,
            reference_angle,
            speed,
            midpoint_candidate.time,
            orbit_target_cache,
        )
    {
        return Some(reference_angle);
    }
    if !goes_out_of_bounds(source, midpoint_candidate)
        && static_candidate_has_target_margin(source, target, reference_angle, midpoint_candidate)
        && !shot_hits_sun(source, reference_angle, speed, midpoint_candidate.time)
        && !shot_hits_blocker(
            state,
            source,
            target,
            reference_angle,
            speed,
            midpoint_candidate.time,
            orbit_target_cache,
        )
    {
        return Some(reference_angle);
    }

    if let Some(angle) = cached_static_target_angle(
        state,
        source,
        target,
        speed,
        max_time,
        reference_angle,
        orbit_target_cache,
    ) {
        return Some(angle);
    }

    if let Some(angle) = optimistic_static_target_angle(
        state,
        source,
        target,
        speed,
        max_time,
        reference_angle,
        orbit_target_cache,
    ) {
        return Some(angle);
    }

    let target_spans = static_target_arc_spans(source, target);
    if target_spans.is_empty() {
        return matches!(targeting_mode, TargetingMode::AnythingGoes).then_some(reference_angle);
    }

    let sun_spans = sun_forbidden_spans(source, speed, max_time, reference_angle);
    let sun_safe_spans = subtract_spans(&target_spans, &sun_spans);
    if sun_safe_spans.is_empty() {
        return matches!(targeting_mode, TargetingMode::AnythingGoes).then_some(reference_angle);
    }
    let first_sun_avoiding = choose_center_first_with(&sun_safe_spans, |relative_angle| {
        let angle = normalize_angle(reference_angle + relative_angle);
        let candidate = static_target_candidate(source, target, angle, speed);
        !goes_out_of_bounds(source, candidate)
            && static_candidate_has_target_margin(source, target, angle, candidate)
    })
    .map(|relative_angle| normalize_angle(reference_angle + relative_angle));

    if let Some(relative_angle) = choose_center_first_with(&target_spans, |relative_angle| {
        let angle = normalize_angle(reference_angle + relative_angle);
        let candidate = static_target_candidate(source, target, angle, speed);
        !goes_out_of_bounds(source, candidate)
            && static_candidate_has_target_margin(source, target, angle, candidate)
            && !shot_hits_sun(source, angle, speed, candidate.time)
            && !shot_hits_blocker(
                state,
                source,
                target,
                angle,
                speed,
                candidate.time,
                orbit_target_cache,
            )
    }) {
        return Some(normalize_angle(reference_angle + relative_angle));
    }

    let mut feasible = sun_safe_spans;
    subtract_blocker_forbidden_spans(
        state,
        source,
        target,
        speed,
        max_time,
        reference_angle,
        orbit_target_cache,
        &mut feasible,
    );
    choose_center_first_with(&feasible, |relative_angle| {
        let angle = normalize_angle(reference_angle + relative_angle);
        let candidate = static_target_candidate(source, target, angle, speed);
        !goes_out_of_bounds(source, candidate)
            && static_candidate_has_target_margin(source, target, angle, candidate)
            && !shot_hits_blocker(
                state,
                source,
                target,
                angle,
                speed,
                candidate.time,
                orbit_target_cache,
            )
    })
    .map(|relative_angle| normalize_angle(reference_angle + relative_angle))
    .or_else(|| {
        first_sun_avoiding.or_else(|| {
            matches!(targeting_mode, TargetingMode::AnythingGoes).then_some(reference_angle)
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn cached_static_target_angle(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
    max_time: f64,
    reference_angle: f64,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
) -> Option<f64> {
    if is_dynamic_planet_cached(state, source)
        || is_dynamic_planet_cached(state, target)
        || state.static_target_cache.is_empty()
    {
        return None;
    }
    choose_center_first_static_arcs(
        state.static_target_cache.arcs(source.id, target.id),
        |relative_angle| {
            let angle = normalize_angle(reference_angle + relative_angle);
            let candidate = static_target_candidate(source, target, angle, speed);
            !goes_out_of_bounds(source, candidate)
                && static_candidate_has_target_margin(source, target, angle, candidate)
                && !shot_hits_dynamic_blocker(
                    state,
                    source,
                    target,
                    angle,
                    speed,
                    max_time.min(candidate.time),
                    orbit_target_cache,
                )
        },
    )
    .map(|relative_angle| normalize_angle(reference_angle + relative_angle))
}

fn static_target_cache_contains(state: &State, source_id: u32, target_id: u32, angle: f64) -> bool {
    !state.static_target_cache.is_empty()
        && state
            .static_target_cache
            .arcs(source_id, target_id)
            .iter()
            .any(|span| span.start <= angle && angle <= span.end)
}

#[allow(clippy::too_many_arguments)]
fn optimistic_static_target_angle(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
    max_time: f64,
    reference_angle: f64,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
) -> Option<f64> {
    let candidate_angle =
        if !is_dynamic_planet_cached(state, source) && !state.static_target_cache.is_empty() {
            state.static_target_cache.get(source.id, target.id)?
        } else {
            reference_angle
        };
    let candidate = static_target_candidate(source, target, candidate_angle, speed);
    if !is_dynamic_planet_cached(state, source) && !state.static_target_cache.is_empty() {
        return (!goes_out_of_bounds(source, candidate)
            && static_candidate_has_target_margin(source, target, candidate_angle, candidate)
            && !shot_hits_dynamic_blocker(
                state,
                source,
                target,
                candidate_angle,
                speed,
                max_time.min(candidate.time),
                orbit_target_cache,
            ))
        .then_some(candidate_angle);
    }
    (!goes_out_of_bounds(source, candidate)
        && static_candidate_has_target_margin(source, target, candidate_angle, candidate)
        && !shot_hits_sun(source, candidate_angle, speed, candidate.time)
        && !shot_hits_blocker(
            state,
            source,
            target,
            candidate_angle,
            speed,
            max_time.min(candidate.time),
            orbit_target_cache,
        ))
    .then_some(candidate_angle)
}

fn static_candidate_has_target_margin(
    source: &Planet,
    target: &Planet,
    angle: f64,
    _candidate: TargetCandidate,
) -> bool {
    let start = launch_start(source, angle);
    let radius = (target.radius - DYNAMIC_TARGET_EPS).max(0.0);
    point_to_segment_distance(
        target.position(),
        start,
        point_along(start, angle, BOARD_SIZE * 2.0),
    ) <= radius
}

fn static_target_arc_spans(source: &Planet, target: &Planet) -> Vec<AngleSpan> {
    let target_distance = distance(source.position(), target.position());
    let target_radius = (target.radius - DYNAMIC_TARGET_EPS).max(0.0);
    let half_angle = if target_distance <= target_radius {
        PI
    } else if target_radius > 0.0 {
        (target_radius / target_distance).asin()
    } else {
        0.0
    };
    (half_angle > ANGLE_EPS)
        .then_some(AngleSpan {
            start: -half_angle,
            end: half_angle,
        })
        .into_iter()
        .collect()
}

fn static_target_max_time(source: &Planet, target: &Planet, speed: f64) -> f64 {
    let max_distance = distance(source.position(), target.position()) + target.radius;
    max_distance / speed.max(f64::EPSILON)
}

fn best_live_static_target_angle(state: &State, source: &Planet, target: &Planet) -> Option<f64> {
    if !state.static_planet_ids.is_empty() {
        return best_static_target_angle(
            source,
            target,
            state
                .static_planet_ids
                .iter()
                .filter_map(|planet_id| state.planets.get(*planet_id))
                .filter(|planet| planet.id != source.id && planet.id != target.id),
        );
    }

    let blockers = state
        .planets
        .iter()
        .filter(|planet| {
            planet.id != source.id
                && planet.id != target.id
                && !state.comet_planet_ids.contains(&planet.id)
                && !is_orbiting_planet(state, planet)
        })
        .collect::<Vec<_>>();
    best_static_target_angle(source, target, blockers.iter().copied())
}

#[cfg(test)]
fn orbiting_target_candidates(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
) -> Option<Vec<TargetCandidate>> {
    let initial_target = state.initial_planets.get(target.id)?;
    let path = orbit_target_cache.path_for(target)?;
    let (min_time, max_time) = orbit_time_bounds(source, initial_target, target, speed);
    Some(piecewise_linear_target_candidates_in_time_range(
        source,
        target.radius,
        speed,
        path,
        min_time,
        max_time,
    ))
}

fn orbiting_target_windows(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
) -> Option<Vec<TargetWindow>> {
    let initial_target = state.initial_planets.get(target.id)?;
    let path = orbit_target_cache.path_for(target)?;
    let (min_time, max_time) = orbit_time_bounds(source, initial_target, target, speed);
    Some(piecewise_linear_target_windows_in_time_range(
        source,
        target.radius,
        speed,
        path,
        min_time,
        max_time,
    ))
}

fn orbit_target_at(state: &State, initial_target: &Planet, target: &Planet, time: f64) -> Point {
    if time == 0.0 {
        target.position()
    } else {
        // Post-reset observations store orbiting planets at the phase from
        // the previous completed simulator step.
        orbit_position(
            initial_target.position(),
            state.angular_velocity,
            state.step.saturating_sub(1) as f64 + time,
        )
    }
}

fn orbit_time_bounds(
    source: &Planet,
    initial_target: &Planet,
    target: &Planet,
    speed: f64,
) -> (f64, f64) {
    let center = Point::new(CENTER, CENTER);
    let orbit_radius = distance(initial_target.position(), center);
    let source_orbit_distance = distance(source.position(), center);
    let min_distance = (source_orbit_distance - orbit_radius).abs();
    let max_distance = source_orbit_distance + orbit_radius;
    let clearance = source.radius + 0.1 + target.radius;
    (
        ((min_distance - clearance) / speed).max(0.0),
        ((max_distance - clearance) / speed).max(0.0),
    )
}

#[cfg(test)]
fn piecewise_linear_target_candidates(
    source: &Planet,
    radius: f64,
    speed: f64,
    path: &[Point],
) -> Vec<TargetCandidate> {
    piecewise_linear_target_candidates_in_time_range(
        source,
        radius,
        speed,
        path,
        0.0,
        f64::INFINITY,
    )
}

#[cfg(test)]
fn piecewise_linear_target_candidates_in_time_range(
    source: &Planet,
    radius: f64,
    speed: f64,
    path: &[Point],
    min_time: f64,
    max_time: f64,
) -> Vec<TargetCandidate> {
    let mut candidates = Vec::new();
    if path.len() < 2 || speed <= 0.0 {
        return candidates;
    }

    let source_pos = source.position();
    let clearance = source.radius + 0.1 + radius;
    let first_segment = min_time.floor().max(0.0) as usize;
    let last_segment = max_time.ceil().max(1.0) as usize;
    let last_segment = last_segment.min(path.len().saturating_sub(1));
    for segment_index in first_segment..last_segment {
        let segment = [path[segment_index], path[segment_index + 1]];
        let start = segment[0];
        let end = segment[1];
        let segment_time = segment_index as f64;
        if !segment_distance_band_can_intersect(
            source_pos,
            start,
            end,
            clearance,
            speed,
            segment_time,
        ) {
            continue;
        }
        for fraction in piecewise_linear_intercept_fractions(
            source_pos,
            start,
            end,
            clearance,
            speed,
            segment_time,
        ) {
            let time = segment_time + fraction;
            if time + ROOT_EPS < min_time || time - ROOT_EPS > max_time {
                continue;
            }
            let target_pos = lerp(start, end, fraction);
            let angle = angle_between(source_pos, target_pos);
            candidates.push(TargetCandidate {
                angle,
                end: point_along(launch_start(source, angle), angle, speed * time),
                time,
            });
        }
    }
    candidates.sort_by(|left, right| left.time.total_cmp(&right.time));
    candidates
}

fn piecewise_linear_target_windows(
    source: &Planet,
    radius: f64,
    speed: f64,
    path: &[Point],
) -> Vec<TargetWindow> {
    piecewise_linear_target_windows_in_time_range(source, radius, speed, path, 0.0, f64::INFINITY)
}

fn piecewise_linear_target_windows_in_time_range(
    source: &Planet,
    radius: f64,
    speed: f64,
    path: &[Point],
    min_time: f64,
    max_time: f64,
) -> Vec<TargetWindow> {
    let mut windows = Vec::new();
    if path.len() < 2 || speed <= 0.0 {
        return windows;
    }

    let source_pos = source.position();
    let source_clearance = source.radius + 0.1;
    let target_radius = (radius - DYNAMIC_TARGET_EPS).max(0.0);
    let first_segment = min_time.floor().max(0.0) as usize;
    let last_segment = max_time.ceil().max(1.0) as usize;
    let last_segment = last_segment.min(path.len().saturating_sub(1));
    for segment_index in first_segment..last_segment {
        let start = path[segment_index];
        let end = path[segment_index + 1];
        let segment_time = segment_index as f64;
        if !target_window_distance_band_can_intersect(
            source_pos,
            start,
            end,
            source_clearance,
            target_radius,
            speed,
            segment_time,
        ) {
            continue;
        }

        let lower_fraction = (min_time - segment_time).clamp(0.0, 1.0);
        let upper_fraction = (max_time - segment_time).clamp(0.0, 1.0);
        if upper_fraction - lower_fraction <= ROOT_EPS {
            continue;
        }

        let mut roots = vec![lower_fraction, upper_fraction];
        push_distance_linear_roots(
            &mut roots,
            source_pos,
            start,
            end,
            source_clearance + target_radius + speed * segment_time,
            speed,
        );
        push_distance_linear_roots(
            &mut roots,
            source_pos,
            start,
            end,
            source_clearance - target_radius + speed * segment_time,
            speed,
        );
        push_distance_linear_roots(
            &mut roots,
            source_pos,
            start,
            end,
            target_radius - source_clearance - speed * segment_time,
            -speed,
        );
        roots.sort_by(f64::total_cmp);
        roots.dedup_by(|left, right| (*left - *right).abs() <= ROOT_EPS);

        for pair in roots.windows(2) {
            let start_fraction = pair[0].max(lower_fraction);
            let end_fraction = pair[1].min(upper_fraction);
            if end_fraction - start_fraction <= ROOT_EPS {
                continue;
            }
            let mid_time = segment_time + (start_fraction + end_fraction) / 2.0;
            let target_pos = lerp(start, end, mid_time - segment_time);
            if instantaneous_target_arc(
                source_pos,
                source_clearance,
                target_pos,
                target_radius,
                speed,
                mid_time,
                angle_between(source_pos, target_pos),
            )
            .is_none()
            {
                continue;
            }
            windows.push(TargetWindow {
                start_time: segment_time + start_fraction,
                end_time: segment_time + end_fraction,
                segment_index,
                segment_start: start,
                segment_end: end,
            });
            if windows.len() >= TARGET_WINDOW_LIMIT {
                return windows;
            }
        }
    }
    windows
}

fn target_window_distance_band_can_intersect(
    source_pos: Point,
    start: Point,
    end: Point,
    source_clearance: f64,
    target_radius: f64,
    speed: f64,
    segment_time: f64,
) -> bool {
    let min_distance = point_to_segment_distance(source_pos, start, end);
    let max_distance = distance(source_pos, start).max(distance(source_pos, end));
    let min_reachable = (source_clearance + speed * segment_time - target_radius).max(0.0);
    let max_reachable = source_clearance + speed * (segment_time + 1.0) + target_radius;
    max_distance >= min_reachable && min_distance <= max_reachable
}

fn push_distance_linear_roots(
    roots: &mut Vec<f64>,
    source_pos: Point,
    start: Point,
    end: Point,
    distance_at_segment_start: f64,
    distance_slope: f64,
) {
    let offset = Point::new(start.x - source_pos.x, start.y - source_pos.y);
    let velocity = Point::new(end.x - start.x, end.y - start.y);
    let q2 = velocity.x * velocity.x + velocity.y * velocity.y - distance_slope * distance_slope;
    let q1 = 2.0
        * (offset.x * velocity.x + offset.y * velocity.y
            - distance_at_segment_start * distance_slope);
    let q0 = offset.x * offset.x + offset.y * offset.y
        - distance_at_segment_start * distance_at_segment_start;
    if q2.abs() <= QUADRATIC_EPS {
        if q1.abs() > QUADRATIC_EPS {
            let root = -q0 / q1;
            if distance_at_segment_start + distance_slope * root >= -ROOT_EPS {
                push_unit_root(roots, root);
            }
        }
        return;
    }
    let discriminant = q1 * q1 - 4.0 * q2 * q0;
    if discriminant < -QUADRATIC_EPS {
        return;
    }
    let sqrt_discriminant = discriminant.max(0.0).sqrt();
    for root in [
        (-q1 - sqrt_discriminant) / (2.0 * q2),
        (-q1 + sqrt_discriminant) / (2.0 * q2),
    ] {
        if distance_at_segment_start + distance_slope * root >= -ROOT_EPS {
            push_unit_root(roots, root);
        }
    }
}

fn choose_dynamic_window_angle(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
    windows: &[TargetWindow],
    orbit_target_cache: &mut OrbitTargetCache<'_>,
    targeting_mode: TargetingMode,
) -> Option<f64> {
    let mut first_sun_avoiding = None;
    let mut first_midpoint = None;
    for window in windows {
        let reference_angle = window_midpoint_angle(source, *window);
        first_midpoint.get_or_insert(reference_angle);

        let target_spans = target_arc_spans_for_window(source, target.radius, speed, *window);
        if target_spans.is_empty() {
            continue;
        }

        for centerline_candidate in dynamic_window_centerline_candidates(source, speed, *window) {
            let Some(candidate) = dynamic_window_candidate_for_angle(
                source,
                target.radius,
                speed,
                *window,
                centerline_candidate.angle,
            ) else {
                continue;
            };
            if !goes_out_of_bounds(source, candidate)
                && !shot_hits_sun(source, candidate.angle, speed, candidate.time)
                && !shot_hits_blocker(
                    state,
                    source,
                    target,
                    candidate.angle,
                    speed,
                    window.end_time,
                    orbit_target_cache,
                )
            {
                return Some(candidate.angle);
            }
        }

        let sun_spans = sun_forbidden_spans(source, speed, window.end_time, reference_angle);
        let sun_safe_spans = subtract_spans(&target_spans, &sun_spans);
        if sun_safe_spans.is_empty() {
            continue;
        }
        if first_sun_avoiding.is_none() {
            first_sun_avoiding = choose_center_first_with(&sun_safe_spans, |relative_angle| {
                let angle = normalize_angle(reference_angle + relative_angle);
                dynamic_window_candidate_for_angle(source, target.radius, speed, *window, angle)
                    .is_some_and(|candidate| !goes_out_of_bounds(source, candidate))
            })
            .map(|relative_angle| normalize_angle(reference_angle + relative_angle));
        }

        if let Some(relative_angle) = choose_center_first_with(&target_spans, |relative_angle| {
            let angle = normalize_angle(reference_angle + relative_angle);
            dynamic_window_candidate_for_angle(source, target.radius, speed, *window, angle)
                .is_some_and(|candidate| {
                    !goes_out_of_bounds(source, candidate)
                        && !shot_hits_sun(source, candidate.angle, speed, candidate.time)
                        && !shot_hits_blocker(
                            state,
                            source,
                            target,
                            candidate.angle,
                            speed,
                            window.end_time,
                            orbit_target_cache,
                        )
                })
        }) {
            return Some(normalize_angle(reference_angle + relative_angle));
        }

        let mut feasible = sun_safe_spans;
        subtract_blocker_forbidden_spans(
            state,
            source,
            target,
            speed,
            window.end_time,
            reference_angle,
            orbit_target_cache,
            &mut feasible,
        );
        if let Some(relative_angle) = choose_center_first_with(&feasible, |relative_angle| {
            let angle = normalize_angle(reference_angle + relative_angle);
            dynamic_window_candidate_for_angle(source, target.radius, speed, *window, angle)
                .is_some_and(|candidate| {
                    !goes_out_of_bounds(source, candidate)
                        && !shot_hits_blocker(
                            state,
                            source,
                            target,
                            candidate.angle,
                            speed,
                            window.end_time,
                            orbit_target_cache,
                        )
                })
        }) {
            return Some(normalize_angle(reference_angle + relative_angle));
        }
    }

    if let Some(angle) = first_sun_avoiding {
        return Some(angle);
    }
    if matches!(targeting_mode, TargetingMode::AnythingGoes) {
        return first_midpoint.map(normalize_angle);
    }
    None
}

fn dynamic_window_candidate_for_angle(
    source: &Planet,
    target_radius: f64,
    speed: f64,
    window: TargetWindow,
    angle: f64,
) -> Option<TargetCandidate> {
    let start = launch_start(source, angle);
    let direction = Point::new(angle.cos(), angle.sin());
    let target_velocity = Point::new(
        window.segment_end.x - window.segment_start.x,
        window.segment_end.y - window.segment_start.y,
    );
    let segment_time = window.segment_index as f64;
    let relative_start = Point::new(
        start.x - window.segment_start.x + target_velocity.x * segment_time,
        start.y - window.segment_start.y + target_velocity.y * segment_time,
    );
    let relative_velocity = Point::new(
        speed * direction.x - target_velocity.x,
        speed * direction.y - target_velocity.y,
    );
    let radius = (target_radius - DYNAMIC_TARGET_EPS).max(0.0);
    first_time_inside_moving_circle(
        relative_start,
        relative_velocity,
        radius,
        window.start_time,
        window.end_time,
    )
    .map(|time| TargetCandidate {
        angle,
        end: point_along(start, angle, speed * time),
        time,
    })
}

fn first_time_inside_moving_circle(
    relative_start: Point,
    relative_velocity: Point,
    radius: f64,
    min_time: f64,
    max_time: f64,
) -> Option<f64> {
    if max_time < min_time {
        return None;
    }
    let q2 = relative_velocity.x * relative_velocity.x + relative_velocity.y * relative_velocity.y;
    let q1 =
        2.0 * (relative_start.x * relative_velocity.x + relative_start.y * relative_velocity.y);
    let q0 =
        relative_start.x * relative_start.x + relative_start.y * relative_start.y - radius * radius;
    let value_at = |time: f64| q2 * time * time + q1 * time + q0;
    if value_at(min_time) <= ROOT_EPS {
        return Some(min_time);
    }
    if q2 <= QUADRATIC_EPS {
        return None;
    }
    let discriminant = q1 * q1 - 4.0 * q2 * q0;
    if discriminant < -QUADRATIC_EPS {
        return None;
    }
    let sqrt_discriminant = discriminant.max(0.0).sqrt();
    let enter = (-q1 - sqrt_discriminant) / (2.0 * q2);
    let exit = (-q1 + sqrt_discriminant) / (2.0 * q2);
    let time = enter.max(min_time);
    (time <= exit + ROOT_EPS && time <= max_time + ROOT_EPS)
        .then_some(time.clamp(min_time, max_time))
}

fn dynamic_window_centerline_candidates(
    source: &Planet,
    speed: f64,
    window: TargetWindow,
) -> [TargetCandidate; 3] {
    let mid_time = (window.start_time + window.end_time) / 2.0;
    [
        dynamic_window_centerline_candidate(source, speed, window, window.start_time),
        dynamic_window_centerline_candidate(source, speed, window, mid_time),
        dynamic_window_centerline_candidate(source, speed, window, window.end_time),
    ]
}

fn dynamic_window_centerline_candidate(
    source: &Planet,
    speed: f64,
    window: TargetWindow,
    time: f64,
) -> TargetCandidate {
    let target_pos = target_position_in_window(window, time);
    let angle = angle_between(source.position(), target_pos);
    TargetCandidate {
        angle,
        end: point_along(launch_start(source, angle), angle, speed * time),
        time,
    }
}

fn window_midpoint_angle(source: &Planet, window: TargetWindow) -> f64 {
    let time = (window.start_time + window.end_time) / 2.0;
    angle_between(source.position(), target_position_in_window(window, time))
}

fn target_arc_spans_for_window(
    source: &Planet,
    target_radius: f64,
    speed: f64,
    window: TargetWindow,
) -> Vec<AngleSpan> {
    let source_pos = source.position();
    let source_clearance = source.radius + 0.1;
    let target_radius = (target_radius - DYNAMIC_TARGET_EPS).max(0.0);
    let reference_angle = window_midpoint_angle(source, window);
    let mid_time = (window.start_time + window.end_time) / 2.0;
    instantaneous_target_arc(
        source_pos,
        source_clearance,
        target_position_in_window(window, mid_time),
        target_radius,
        speed,
        mid_time,
        reference_angle,
    )
    .map(|arc| {
        arc_to_spans(
            normalize_angle(reference_angle + arc.center),
            arc.half_angle,
            reference_angle,
        )
    })
    .unwrap_or_default()
}

fn instantaneous_target_arc(
    source_pos: Point,
    source_clearance: f64,
    target_pos: Point,
    target_radius: f64,
    speed: f64,
    time: f64,
    reference_angle: f64,
) -> Option<InstantArc> {
    let radius_from_source = source_clearance + speed * time;
    if radius_from_source <= 0.0 {
        return None;
    }
    let center_distance = distance(source_pos, target_pos);
    if center_distance <= ROOT_EPS {
        return (radius_from_source <= target_radius + ROOT_EPS).then_some(InstantArc {
            center: 0.0,
            half_angle: PI,
        });
    }
    if center_distance + radius_from_source <= target_radius + ROOT_EPS {
        return Some(InstantArc {
            center: angle_delta(angle_between(source_pos, target_pos), reference_angle),
            half_angle: PI,
        });
    }
    if (center_distance - radius_from_source).abs() > target_radius + ROOT_EPS {
        return None;
    }
    let cos_half = (radius_from_source * radius_from_source + center_distance * center_distance
        - target_radius * target_radius)
        / (2.0 * radius_from_source * center_distance);
    if !(-1.0 - ROOT_EPS..=1.0 + ROOT_EPS).contains(&cos_half) {
        return None;
    }
    Some(InstantArc {
        center: angle_delta(angle_between(source_pos, target_pos), reference_angle),
        half_angle: cos_half.clamp(-1.0, 1.0).acos(),
    })
}

fn target_position_in_window(window: TargetWindow, time: f64) -> Point {
    lerp(
        window.segment_start,
        window.segment_end,
        (time - window.segment_index as f64).clamp(0.0, 1.0),
    )
}

fn sun_forbidden_spans(
    source: &Planet,
    speed: f64,
    time: f64,
    reference_angle: f64,
) -> Vec<AngleSpan> {
    forbidden_circle_spans(
        source.position(),
        Point::new(CENTER, CENTER),
        crate::rules_engine::state::SUN_RADIUS + COLLISION_EPS,
        source.radius + 0.1 + speed * time,
        reference_angle,
    )
}

fn shot_hits_sun(source: &Planet, angle: f64, speed: f64, time: f64) -> bool {
    shot_hits_circle(
        source,
        angle,
        speed,
        time,
        Point::new(CENTER, CENTER),
        crate::rules_engine::state::SUN_RADIUS + COLLISION_EPS,
    )
}

fn shot_hits_circle(
    source: &Planet,
    angle: f64,
    speed: f64,
    time: f64,
    center: Point,
    radius: f64,
) -> bool {
    let start = launch_start(source, angle);
    let end = point_along(start, angle, speed * time);
    swept_aabb_overlaps(start, end, center, center, radius)
        && point_to_segment_distance(center, start, end) < radius
}

#[allow(clippy::too_many_arguments)]
fn shot_hits_blocker(
    state: &State,
    source: &Planet,
    target: &Planet,
    angle: f64,
    speed: f64,
    time: f64,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
) -> bool {
    for blocker in &orbit_target_cache.static_blockers {
        if blocker.id != source.id
            && blocker.id != target.id
            && shot_hits_circle(source, angle, speed, time, blocker.center, blocker.radius)
        {
            return true;
        }
    }

    for blocker_index in 0..orbit_target_cache.dynamic_blocker_ids.len() {
        let blocker_id = orbit_target_cache.dynamic_blocker_ids[blocker_index];
        let Some(blocker) = state.planets.get(blocker_id) else {
            continue;
        };
        if blocker.id == source.id || blocker.id == target.id {
            continue;
        }
        let path = orbit_target_cache.dynamic_path_for(blocker);
        if path.is_some_and(|path| {
            shot_hits_moving_circle(
                source,
                angle,
                speed,
                time,
                path,
                blocker.radius + COLLISION_EPS,
            )
        }) {
            return true;
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn shot_hits_dynamic_blocker(
    state: &State,
    source: &Planet,
    target: &Planet,
    angle: f64,
    speed: f64,
    time: f64,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
) -> bool {
    for blocker_index in 0..orbit_target_cache.dynamic_blocker_ids.len() {
        let blocker_id = orbit_target_cache.dynamic_blocker_ids[blocker_index];
        let Some(blocker) = state.planets.get(blocker_id) else {
            continue;
        };
        if blocker.id == source.id || blocker.id == target.id {
            continue;
        }
        let path = orbit_target_cache.dynamic_path_for(blocker);
        if path.is_some_and(|path| {
            shot_hits_moving_circle(
                source,
                angle,
                speed,
                time,
                path,
                blocker.radius + COLLISION_EPS,
            )
        }) {
            return true;
        }
    }
    false
}

fn shot_hits_moving_circle(
    source: &Planet,
    angle: f64,
    speed: f64,
    max_time: f64,
    path: &[Point],
    radius: f64,
) -> bool {
    if path.is_empty() || max_time <= 0.0 {
        return false;
    }
    let start = launch_start(source, angle);
    let last_segment = max_time.ceil().max(1.0) as usize;
    let last_segment = last_segment.min(path.len().saturating_sub(1));
    for segment_index in 0..last_segment {
        let start_time = segment_index as f64;
        let end_time = (segment_index as f64 + 1.0).min(max_time);
        if end_time - start_time <= ROOT_EPS {
            continue;
        }
        let fleet_start = point_along(start, angle, speed * start_time);
        let fleet_end = point_along(start, angle, speed * end_time);
        let blocker_start = path[segment_index];
        let blocker_end = lerp(
            path[segment_index],
            path[segment_index + 1],
            end_time - start_time,
        );
        if !swept_aabb_overlaps(fleet_start, fleet_end, blocker_start, blocker_end, radius) {
            continue;
        }
        if blocker_start == blocker_end {
            if point_to_segment_distance(blocker_start, fleet_start, fleet_end) < radius {
                return true;
            }
        } else if swept_pair_hit(fleet_start, fleet_end, blocker_start, blocker_end, radius) {
            return true;
        }
    }
    false
}

fn swept_aabb_overlaps(
    fleet_start: Point,
    fleet_end: Point,
    blocker_start: Point,
    blocker_end: Point,
    radius: f64,
) -> bool {
    let fleet_min_x = fleet_start.x.min(fleet_end.x);
    let fleet_max_x = fleet_start.x.max(fleet_end.x);
    let fleet_min_y = fleet_start.y.min(fleet_end.y);
    let fleet_max_y = fleet_start.y.max(fleet_end.y);
    let blocker_min_x = blocker_start.x.min(blocker_end.x) - radius;
    let blocker_max_x = blocker_start.x.max(blocker_end.x) + radius;
    let blocker_min_y = blocker_start.y.min(blocker_end.y) - radius;
    let blocker_max_y = blocker_start.y.max(blocker_end.y) + radius;

    !(fleet_max_x < blocker_min_x
        || fleet_min_x > blocker_max_x
        || fleet_max_y < blocker_min_y
        || fleet_min_y > blocker_max_y)
}

#[allow(clippy::too_many_arguments)]
fn subtract_blocker_forbidden_spans(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
    time: f64,
    reference_angle: f64,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
    feasible: &mut Vec<AngleSpan>,
) {
    let max_ray_distance = source.radius + 0.1 + speed * time;
    let source_pos = source.position();
    for blocker in &orbit_target_cache.static_blockers {
        if blocker.id == source.id || blocker.id == target.id {
            continue;
        }
        let blocker_spans = forbidden_circle_spans_overlapping(
            feasible,
            source_pos,
            blocker.center,
            blocker.radius,
            max_ray_distance,
            reference_angle,
        );
        if blocker_spans.is_empty() || !spans_overlap(feasible, &blocker_spans) {
            continue;
        }
        *feasible = subtract_spans(feasible, &blocker_spans);
        if feasible.is_empty() {
            return;
        }
    }

    for blocker_index in 0..orbit_target_cache.dynamic_blocker_ids.len() {
        let blocker_id = orbit_target_cache.dynamic_blocker_ids[blocker_index];
        let Some(blocker) = state.planets.get(blocker_id) else {
            continue;
        };
        if blocker.id == source.id || blocker.id == target.id {
            continue;
        }
        let blocker_spans = orbit_target_cache
            .dynamic_path_for(blocker)
            .map(|path| {
                moving_circle_forbidden_spans(
                    source,
                    blocker.radius + COLLISION_EPS,
                    speed,
                    path,
                    time,
                    reference_angle,
                    feasible,
                )
            })
            .unwrap_or_default();

        if blocker_spans.is_empty() || !spans_overlap(feasible, &blocker_spans) {
            continue;
        }
        *feasible = subtract_spans(feasible, &blocker_spans);
        if feasible.is_empty() {
            return;
        }
    }
}

fn moving_circle_forbidden_spans(
    source: &Planet,
    radius: f64,
    speed: f64,
    path: &[Point],
    max_time: f64,
    reference_angle: f64,
    feasible: &[AngleSpan],
) -> Vec<AngleSpan> {
    let mut spans = Vec::new();
    if path.is_empty() || max_time <= 0.0 || feasible.is_empty() {
        return spans;
    }

    let source_pos = source.position();
    let source_clearance = source.radius + 0.1;
    let mut sample_times = Vec::with_capacity(DYNAMIC_BLOCKER_SAMPLE_FRACTIONS.len() * 2);
    for fraction in DYNAMIC_BLOCKER_SAMPLE_FRACTIONS {
        let time = max_time * fraction;
        sample_times.push(time);
        let center = path_position_at(path, time);
        let radial_time =
            ((distance(source_pos, center) - source_clearance) / speed).clamp(0.0, max_time);
        sample_times.push(radial_time);
    }
    sample_times.sort_by(f64::total_cmp);
    sample_times.dedup_by(|left, right| (*left - *right).abs() <= ROOT_EPS);

    for time in sample_times {
        if let Some(arc) = instantaneous_target_arc(
            source_pos,
            source_clearance,
            path_position_at(path, time),
            radius,
            speed,
            time,
            reference_angle,
        ) {
            if !arc_overlaps_spans(arc.center, arc.half_angle, feasible) {
                continue;
            }
            spans.extend(arc_to_spans(
                normalize_angle(reference_angle + arc.center),
                arc.half_angle,
                reference_angle,
            ));
        }
    }
    merge_spans(spans)
}

fn path_position_at(path: &[Point], time: f64) -> Point {
    if path.len() == 1 {
        return path[0];
    }
    let segment_index = time.floor().clamp(0.0, path.len().saturating_sub(2) as f64) as usize;
    lerp(
        path[segment_index],
        path[segment_index + 1],
        (time - segment_index as f64).clamp(0.0, 1.0),
    )
}

fn forbidden_circle_spans_overlapping(
    feasible: &[AngleSpan],
    source_pos: Point,
    center: Point,
    radius: f64,
    max_ray_distance: f64,
    reference_angle: f64,
) -> Vec<AngleSpan> {
    if feasible.is_empty() {
        return Vec::new();
    }
    let center_distance = distance(source_pos, center);
    if center_distance <= radius {
        return vec![AngleSpan {
            start: -PI,
            end: PI,
        }];
    }
    if center_distance - radius > max_ray_distance {
        return Vec::new();
    }
    let half_angle = (radius / center_distance).clamp(-1.0, 1.0).asin();
    let center = angle_delta(angle_between(source_pos, center), reference_angle);
    if !arc_overlaps_spans(center, half_angle, feasible) {
        return Vec::new();
    }
    arc_to_spans(
        normalize_angle(reference_angle + center),
        half_angle,
        reference_angle,
    )
}

fn forbidden_circle_spans(
    source_pos: Point,
    center: Point,
    radius: f64,
    max_ray_distance: f64,
    reference_angle: f64,
) -> Vec<AngleSpan> {
    let center_distance = distance(source_pos, center);
    if center_distance <= radius {
        return vec![AngleSpan {
            start: -PI,
            end: PI,
        }];
    }
    if center_distance - radius > max_ray_distance {
        return Vec::new();
    }
    let half_angle = (radius / center_distance).clamp(-1.0, 1.0).asin();
    arc_to_spans(
        angle_between(source_pos, center),
        half_angle,
        reference_angle,
    )
}

fn arc_overlaps_spans(center: f64, half_angle: f64, spans: &[AngleSpan]) -> bool {
    if half_angle >= PI {
        return !spans.is_empty();
    }
    let raw_start = center - half_angle;
    let raw_end = center + half_angle;
    for offset in [-TAU, 0.0, TAU] {
        let start = (raw_start + offset).max(-PI);
        let end = (raw_end + offset).min(PI);
        if end - start <= ANGLE_EPS {
            continue;
        }
        if spans
            .iter()
            .any(|span| end > span.start && start < span.end)
        {
            return true;
        }
    }
    false
}

fn arc_to_spans(center: f64, half_angle: f64, reference_angle: f64) -> Vec<AngleSpan> {
    if half_angle >= PI {
        return vec![AngleSpan {
            start: -PI,
            end: PI,
        }];
    }

    let center = angle_delta(center, reference_angle);
    let raw_start = center - half_angle;
    let raw_end = center + half_angle;
    let mut spans = Vec::with_capacity(2);
    for offset in [-TAU, 0.0, TAU] {
        let start = raw_start + offset;
        let end = raw_end + offset;
        let clipped_start = start.max(-PI);
        let clipped_end = end.min(PI);
        if clipped_end - clipped_start > ANGLE_EPS {
            spans.push(AngleSpan {
                start: clipped_start,
                end: clipped_end,
            });
        }
    }
    merge_spans(spans)
}

fn subtract_spans(spans: &[AngleSpan], forbidden: &[AngleSpan]) -> Vec<AngleSpan> {
    let mut current = spans.to_vec();
    for forbidden_span in forbidden {
        let mut next = Vec::with_capacity(current.len() + 1);
        for span in current {
            if forbidden_span.end <= span.start || forbidden_span.start >= span.end {
                next.push(span);
                continue;
            }
            if forbidden_span.start > span.start {
                next.push(AngleSpan {
                    start: span.start,
                    end: forbidden_span.start,
                });
            }
            if forbidden_span.end < span.end {
                next.push(AngleSpan {
                    start: forbidden_span.end,
                    end: span.end,
                });
            }
        }
        current = next;
        if current.is_empty() {
            break;
        }
    }
    current
}

fn spans_overlap(left: &[AngleSpan], right: &[AngleSpan]) -> bool {
    left.iter().any(|left_span| {
        right
            .iter()
            .any(|right_span| right_span.end > left_span.start && right_span.start < left_span.end)
    })
}

fn merge_spans(mut spans: Vec<AngleSpan>) -> Vec<AngleSpan> {
    spans.sort_by(|left, right| left.start.total_cmp(&right.start));
    let mut merged: Vec<AngleSpan> = Vec::with_capacity(spans.len());
    for span in spans {
        if span.end - span.start <= ANGLE_EPS {
            continue;
        }
        if let Some(last) = merged.last_mut() {
            if span.start <= last.end + ANGLE_EPS {
                last.end = last.end.max(span.end);
                continue;
            }
        }
        merged.push(span);
    }
    merged
}

fn choose_center_first_with(
    spans: &[AngleSpan],
    mut is_valid: impl FnMut(f64) -> bool,
) -> Option<f64> {
    let mut best = None;
    for span in spans {
        let width = span.end - span.start;
        if width <= ANGLE_EPS {
            continue;
        }
        let edge_margin = (width * 0.25).min(MAX_ANGLE_EDGE_MARGIN);
        let center_candidate = 0.0_f64.clamp(span.start, span.end);
        let center_candidate = if (center_candidate - span.start).abs() <= ANGLE_EPS {
            (span.start + edge_margin).min((span.start + span.end) / 2.0)
        } else if (center_candidate - span.end).abs() <= ANGLE_EPS {
            (span.end - edge_margin).max((span.start + span.end) / 2.0)
        } else {
            center_candidate
        };
        for candidate in [
            center_candidate,
            span.start + width * 0.25,
            (span.start + span.end) / 2.0,
            span.start + width * 0.75,
            (span.start + edge_margin).min((span.start + span.end) / 2.0),
            (span.end - edge_margin).max((span.start + span.end) / 2.0),
            (span.start + ANGLE_CHOICE_EPS).min((span.start + span.end) / 2.0),
            (span.end - ANGLE_CHOICE_EPS).max((span.start + span.end) / 2.0),
        ] {
            if !is_valid(candidate) {
                continue;
            }
            let score = candidate.abs();
            if best.is_none_or(|(best_score, _)| score < best_score) {
                best = Some((score, candidate));
            }
        }
    }
    best.map(|(_, angle)| angle)
}

fn choose_center_first_static_arcs(
    spans: &[StaticTargetArc],
    mut is_valid: impl FnMut(f64) -> bool,
) -> Option<f64> {
    let mut best = None;
    for span in spans {
        let width = span.end - span.start;
        if width <= ANGLE_EPS {
            continue;
        }
        let edge_margin = (width * 0.25).min(MAX_ANGLE_EDGE_MARGIN);
        let center_candidate = 0.0_f64.clamp(span.start, span.end);
        let center_candidate = if (center_candidate - span.start).abs() <= ANGLE_EPS {
            (span.start + edge_margin).min((span.start + span.end) / 2.0)
        } else if (center_candidate - span.end).abs() <= ANGLE_EPS {
            (span.end - edge_margin).max((span.start + span.end) / 2.0)
        } else {
            center_candidate
        };
        for candidate in [
            center_candidate,
            span.start + width * 0.25,
            (span.start + span.end) / 2.0,
            span.start + width * 0.75,
            (span.start + edge_margin).min((span.start + span.end) / 2.0),
            (span.end - edge_margin).max((span.start + span.end) / 2.0),
            (span.start + ANGLE_CHOICE_EPS).min((span.start + span.end) / 2.0),
            (span.end - ANGLE_CHOICE_EPS).max((span.start + span.end) / 2.0),
        ] {
            if !is_valid(candidate) {
                continue;
            }
            let score = candidate.abs();
            if best.is_none_or(|(best_score, _)| score < best_score) {
                best = Some((score, candidate));
            }
        }
    }
    best.map(|(_, angle)| angle)
}

fn normalize_angle(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

fn angle_delta(angle: f64, reference_angle: f64) -> f64 {
    normalize_angle(angle - reference_angle)
}

#[cfg(test)]
fn segment_distance_band_can_intersect(
    source_pos: Point,
    start: Point,
    end: Point,
    clearance: f64,
    speed: f64,
    segment_time: f64,
) -> bool {
    let min_distance = point_to_segment_distance(source_pos, start, end);
    let max_distance = distance(source_pos, start).max(distance(source_pos, end));
    let min_reachable = clearance + speed * segment_time;
    let max_reachable = clearance + speed * (segment_time + 1.0);
    max_distance >= min_reachable && min_distance <= max_reachable
}

#[cfg(test)]
fn piecewise_linear_intercept_fractions(
    source_pos: Point,
    start: Point,
    end: Point,
    clearance: f64,
    speed: f64,
    segment_time: f64,
) -> Vec<f64> {
    let offset = Point::new(start.x - source_pos.x, start.y - source_pos.y);
    let velocity = Point::new(end.x - start.x, end.y - start.y);
    let reachable_at_start = clearance + speed * segment_time;
    let q2 = velocity.x * velocity.x + velocity.y * velocity.y - speed * speed;
    let q1 = 2.0 * (offset.x * velocity.x + offset.y * velocity.y - reachable_at_start * speed);
    let q0 = offset.x * offset.x + offset.y * offset.y - reachable_at_start * reachable_at_start;
    let mut roots = Vec::with_capacity(2);
    if q2.abs() <= QUADRATIC_EPS {
        if q1.abs() > QUADRATIC_EPS {
            push_unit_root(&mut roots, -q0 / q1);
        }
        return roots;
    }
    let discriminant = q1 * q1 - 4.0 * q2 * q0;
    if discriminant < -QUADRATIC_EPS {
        return roots;
    }
    let sqrt_discriminant = discriminant.max(0.0).sqrt();
    push_unit_root(&mut roots, (-q1 - sqrt_discriminant) / (2.0 * q2));
    push_unit_root(&mut roots, (-q1 + sqrt_discriminant) / (2.0 * q2));
    roots.sort_by(f64::total_cmp);
    roots.dedup_by(|left, right| (*left - *right).abs() <= ROOT_EPS);
    roots
}

fn push_unit_root(roots: &mut Vec<f64>, root: f64) {
    if (-ROOT_EPS..=1.0 + ROOT_EPS).contains(&root) {
        roots.push(root.clamp(0.0, 1.0));
    }
}

fn goes_out_of_bounds(source: &Planet, candidate: TargetCandidate) -> bool {
    debug_assert!(candidate.time >= 0.0);
    let start = launch_start(source, candidate.angle);
    !point_in_bounds(start) || !point_in_bounds(candidate.end)
}

fn point_in_bounds(point: Point) -> bool {
    (0.0..=BOARD_SIZE).contains(&point.x) && (0.0..=BOARD_SIZE).contains(&point.y)
}

fn is_dynamic_planet_cached(state: &State, planet: &Planet) -> bool {
    !is_static_planet_cached(state, planet)
}

fn is_static_planet_cached(state: &State, planet: &Planet) -> bool {
    if let Some(is_static) = state.static_planet_mask.get(planet.id as usize) {
        return *is_static;
    }
    !state.comet_planet_ids.contains(&planet.id) && !is_orbiting_planet(state, planet)
}

fn is_orbiting_planet(state: &State, planet: &Planet) -> bool {
    is_orbiting(
        state
            .initial_planets
            .get(planet.id)
            .map_or(planet.position(), Planet::position),
        planet.radius,
    )
}

#[cfg(test)]
fn bisect_root(f: &impl Fn(f64) -> f64, mut left: f64, mut right: f64) -> f64 {
    let mut left_value = f(left);
    for _ in 0..64 {
        let mid = (left + right) / 2.0;
        let mid_value = f(mid);
        if mid_value.abs() <= ROOT_EPS {
            return mid;
        }
        if left_value.signum() == mid_value.signum() {
            left = mid;
            left_value = mid_value;
        } else {
            right = mid;
        }
    }
    (left + right) / 2.0
}

fn lerp(start: Point, end: Point, fraction: f64) -> Point {
    Point::new(
        start.x + (end.x - start.x) * fraction,
        start.y + (end.y - start.y) * fraction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::env::{reset, reset_with_rng, step, step_with_injections};
    use crate::rules_engine::generation::RandomSource;
    use crate::rules_engine::state::{
        CometGroup, CometSpawnInjection, OrbitPath, ResetConfig, SimConfig, StaticTargetCache,
        StepInjections,
    };
    use crate::rules_engine::utils::{
        static_ray_hits_planet, static_ray_hits_sun, static_target_rays, swept_pair_hit,
    };

    #[test]
    fn parse_rejects_invalid_targeting_mode_for_pure_action_spec() {
        let err = RlActionSpec::parse("pure", 0, "not_a_targeting_mode")
            .expect_err("pure should validate targeting_mode");
        assert!(err.contains("unsupported targeting_mode"));
    }

    #[test]
    fn parse_rejects_invalid_targeting_mode_for_discrete_action_specs() {
        let err = RlActionSpec::parse("discrete_targets", 0, "not_a_targeting_mode")
            .expect_err("discrete target specs should validate targeting_mode");
        assert!(err.contains("unsupported targeting_mode"));

        let err = RlActionSpec::parse("discrete_target_bins", 7, "not_a_targeting_mode")
            .expect_err("target-bin specs should validate targeting_mode");
        assert!(err.contains("unsupported targeting_mode"));
    }

    fn one_planet_state() -> State {
        let planets = vec![Planet {
            id: 7,
            owner: 0,
            x: 50.0,
            y: 50.0,
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

    fn state_from_planets(planets: Vec<Planet>) -> State {
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

    fn planet(id: u32, owner: i32, x: f64, y: f64, radius: f64, ships: i32) -> Planet {
        Planet {
            id,
            owner,
            x,
            y,
            radius,
            ships,
            production: 1,
        }
    }

    fn replay_77263963_static_target_state(source_ships: i32) -> State {
        let planets = vec![
            planet(
                2,
                -1,
                84.12388824387287,
                2.965933424497905,
                2.09861228866811,
                82,
            ),
            planet(
                7,
                -1,
                30.247979474405184,
                5.056628318443259,
                1.6931471805599454,
                24,
            ),
            planet(
                10,
                -1,
                68.39179262370835,
                5.052350137026821,
                2.09861228866811,
                57,
            ),
            planet(
                15,
                1,
                19.862568169475338,
                7.366978768343088,
                1.0,
                source_ships,
            ),
        ];
        let mut config = ResetConfig::new(4);
        config.step = Some(25);
        config.angular_velocity = Some(0.043356178890437275);
        config.planets = Some(planets.clone());
        config.initial_planets = Some(planets);
        reset(config)
    }

    fn entity_index_for_planet(entities: &ActionEntitySlots, planet_id: u32) -> usize {
        entities
            .iter()
            .position(|slot| slot.is_some_and(|slot| slot.planet_id == planet_id))
            .unwrap_or_else(|| panic!("planet {planet_id} should be in action entity slots"))
    }

    fn decode_single_discrete_target(
        state: &State,
        source_id: u32,
        target_id: u32,
        ship_count: i32,
    ) -> DecodedDiscreteTargetActions {
        let entities = action_entity_slots(state);
        let source_index = entity_index_for_planet(&entities, source_id);
        let target_index = entity_index_for_planet(&entities, target_id);
        let mut launch = vec![false; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let action_index = ACTION_ENTITY_SLOTS + source_index;
        launch[action_index] = true;
        targets[action_index] = target_index as i64;
        ships[action_index] = i64::from(ship_count);

        decode_discrete_target_actions(
            state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("valid replay discrete target should decode")
    }

    fn discrete_target_can_act(
        state: &State,
        source_id: u32,
        target_id: u32,
        min_fleet_size: i64,
    ) -> bool {
        let entities = action_entity_slots(state);
        let source_index = entity_index_for_planet(&entities, source_id);
        let target_index = entity_index_for_planet(&entities, target_id);
        let spec = RlActionSpec::DiscreteTargets {
            targeting_mode: TargetingMode::FullMask,
        };
        let mut can_act = vec![false; spec.can_act_len()];
        let mut max_launch = vec![0; spec.max_launch_len()];

        encode_action_spec(
            spec,
            state,
            &PlayerMap::identity(),
            &entities,
            &mut can_act,
            Some(&mut max_launch),
            min_fleet_size,
        );

        let player = state.planets.get(source_id).expect("source").owner as usize;
        let base = (PlayerMap::identity().internal_to_outer(player) * ACTION_ENTITY_SLOTS
            + source_index)
            * ACTION_ENTITY_SLOTS;
        can_act[base + target_index]
    }

    fn run_until_planet_changes(state: &mut State, planet_id: u32, initial_ships: i32) -> bool {
        let empty_actions = vec![Vec::new(); state.config.player_count];
        for _ in 0..80 {
            if state
                .planets
                .get(planet_id)
                .is_some_and(|target| target.ships != initial_ships)
            {
                return true;
            }
            step(state, &empty_actions);
        }
        state
            .planets
            .get(planet_id)
            .is_some_and(|target| target.ships != initial_ships)
    }

    fn dense_orbiting_target_candidate(
        state: &State,
        source: &Planet,
        target: &Planet,
        speed: f64,
    ) -> Option<TargetCandidate> {
        let initial_target = state.initial_planets.get(target.id)?;
        let source_pos = source.position();
        let center = Point::new(CENTER, CENTER);
        let orbit_radius = distance(initial_target.position(), center);
        let source_orbit_distance = distance(source_pos, center);
        let min_distance = (source_orbit_distance - orbit_radius).abs();
        let max_distance = source_orbit_distance + orbit_radius;
        let clearance = source.radius + 0.1 + target.radius;
        let min_time = ((min_distance - clearance) / speed).max(0.0);
        let max_time = ((max_distance - clearance) / speed).max(0.0);
        let target_at = |time: f64| {
            if time == 0.0 {
                target.position()
            } else {
                // Match the simulator's observed orbit phase, not the next step.
                orbit_position(
                    initial_target.position(),
                    state.angular_velocity,
                    state.step.saturating_sub(1) as f64 + time,
                )
            }
        };
        let impact = |time: f64| distance(target_at(time), source_pos) - clearance - speed * time;
        let candidate_at = |time: f64| {
            let target_pos = target_at(time);
            let angle = angle_between(source_pos, target_pos);
            TargetCandidate {
                angle,
                end: point_along(launch_start(source, angle), angle, speed * time),
                time,
            }
        };

        if max_time <= min_time {
            return (impact(min_time) <= ROOT_EPS).then(|| candidate_at(min_time));
        }
        let mut prev_time = min_time;
        let mut prev_value = impact(prev_time);
        if prev_value <= ROOT_EPS {
            return Some(candidate_at(prev_time));
        }
        for sample_index in 1..=16_384 {
            let time = min_time + (max_time - min_time) * f64::from(sample_index) / 16_384.0;
            let value = impact(time);
            if value <= ROOT_EPS {
                let root_time = if prev_value.signum() != value.signum() {
                    bisect_root(&impact, prev_time, time)
                } else {
                    time
                };
                return Some(candidate_at(root_time));
            }
            prev_time = time;
            prev_value = value;
        }
        None
    }

    fn cached_orbiting_target_candidate(
        state: &State,
        source: &Planet,
        target: &Planet,
        speed: f64,
    ) -> Option<TargetCandidate> {
        let entities = action_entity_slots(state);
        let mut cache = OrbitTargetCache::new(state, &entities, 1);
        orbiting_target_candidates(state, source, target, speed, &mut cache)?
            .into_iter()
            .next()
    }

    fn assert_candidate_hits_swept_segment(
        source: &Planet,
        target_radius: f64,
        path: &[Point],
        speed: f64,
        candidate: TargetCandidate,
    ) {
        assert!(
            candidate_hits_swept_segment(source, target_radius, path, speed, candidate),
            "candidate at time {} did not hit a swept target segment",
            candidate.time,
        );
    }

    #[derive(Clone, Debug)]
    struct AuditRng {
        state: u64,
    }

    impl AuditRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f64(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((self.state >> 11) as f64) / ((1_u64 << 53) as f64)
        }
    }

    impl RandomSource for AuditRng {
        fn randint(&mut self, low: i32, high: i32) -> i32 {
            low + (self.next_f64() * f64::from(high - low + 1)).floor() as i32
        }

        fn uniform(&mut self, low: f64, high: f64) -> f64 {
            low + self.next_f64() * (high - low)
        }
    }

    #[derive(Default)]
    struct AngleAuditStats {
        cases: usize,
        static_cases: usize,
        dynamic_cases: usize,
        old_found: usize,
        new_found: usize,
        both_found: usize,
        old_arrived: usize,
        new_arrived: usize,
        both_arrived: usize,
        new_only_arrived: usize,
        old_only_arrived: usize,
        new_faster: usize,
        new_equal: usize,
        new_slower: usize,
        arrival_tick_delta_sum: i64,
        midpoint_valid: usize,
        midpoint_selected: usize,
        outer_valid: usize,
        outer_matched_or_improved: usize,
        dynamic_old_arrived: usize,
        dynamic_new_arrived: usize,
        dynamic_new_only_arrived: usize,
        dynamic_old_only_arrived: usize,
    }

    fn old_target_angle(
        state: &State,
        source: &Planet,
        target: &Planet,
        ships: i32,
    ) -> Option<f64> {
        let speed = fleet_speed(ships, state.config.ship_speed);
        if is_dynamic_planet_cached(state, target) {
            old_dynamic_target_angle(state, source, target, speed)
        } else {
            old_static_target_angle(state, source, target)
        }
    }

    fn old_static_target_angle(state: &State, source: &Planet, target: &Planet) -> Option<f64> {
        static_target_rays(source, target)
            .into_iter()
            .find(|ray| {
                !static_ray_hits_sun(source, *ray)
                    && !old_static_blockers(state, source, target)
                        .iter()
                        .any(|blocker| static_ray_hits_planet(source, *ray, blocker))
            })
            .map(|ray| ray.angle)
    }

    fn old_static_blockers<'a>(
        state: &'a State,
        source: &Planet,
        target: &Planet,
    ) -> Vec<&'a Planet> {
        if !state.static_planet_ids.is_empty() {
            return state
                .static_planet_ids
                .iter()
                .filter_map(|planet_id| state.planets.get(*planet_id))
                .filter(|planet| planet.id != source.id && planet.id != target.id)
                .collect();
        }
        state
            .planets
            .iter()
            .filter(|planet| {
                planet.id != source.id
                    && planet.id != target.id
                    && !state.comet_planet_ids.contains(&planet.id)
                    && !is_orbiting_planet(state, planet)
            })
            .collect()
    }

    fn old_dynamic_target_angle(
        state: &State,
        source: &Planet,
        target: &Planet,
        speed: f64,
    ) -> Option<f64> {
        let entities = action_entity_slots(state);
        let mut cache = OrbitTargetCache::new(state, &entities, 1);
        let candidates = if state.comet_planet_ids.contains(&target.id) {
            let path = cache.dynamic_path_for(target)?;
            piecewise_linear_target_candidates(source, target.radius, speed, path)
        } else {
            orbiting_target_candidates(state, source, target, speed, &mut cache)?
        };
        choose_old_dynamic_candidate(state, source, target, candidates)
            .map(|candidate| candidate.angle)
    }

    fn choose_old_dynamic_candidate(
        state: &State,
        source: &Planet,
        target: &Planet,
        candidates: Vec<TargetCandidate>,
    ) -> Option<TargetCandidate> {
        candidates
            .iter()
            .copied()
            .find(|candidate| {
                !old_candidate_hits_sun(source, *candidate)
                    && !goes_out_of_bounds(source, *candidate)
                    && !old_candidate_hits_static_blocker(state, source, target, *candidate)
            })
            .or_else(|| {
                candidates.iter().copied().find(|candidate| {
                    !old_candidate_hits_sun(source, *candidate)
                        && !goes_out_of_bounds(source, *candidate)
                })
            })
    }

    fn old_candidate_hits_sun(source: &Planet, candidate: TargetCandidate) -> bool {
        point_to_segment_distance(
            Point::new(CENTER, CENTER),
            launch_start(source, candidate.angle),
            candidate.end,
        ) < crate::rules_engine::state::SUN_RADIUS
    }

    fn old_candidate_hits_static_blocker(
        state: &State,
        source: &Planet,
        target: &Planet,
        candidate: TargetCandidate,
    ) -> bool {
        old_static_blockers(state, source, target)
            .iter()
            .any(|blocker| {
                point_to_segment_distance(
                    blocker.position(),
                    launch_start(source, candidate.angle),
                    candidate.end,
                ) < blocker.radius
            })
    }

    fn new_target_angle_for_audit(
        state: &State,
        source: &Planet,
        target: &Planet,
        ships: i32,
    ) -> Option<f64> {
        let entities = action_entity_slots(state);
        let mut cache = OrbitTargetCache::new(state, &entities, 1);
        target_angle(
            state,
            source,
            target,
            ships,
            &mut cache,
            TargetingMode::FullMask,
        )
        .expect("audit target angle should not error")
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum LaunchOutcome {
        Arrived(usize),
        HitObstacle { planet_id: u32, tick: usize },
        Lost(usize),
        NoContact,
    }

    fn arrival_tick(
        state: &State,
        source: &Planet,
        target: &Planet,
        angle: f64,
        ships: i32,
    ) -> Option<usize> {
        match launch_outcome(state, source, target, angle, ships) {
            LaunchOutcome::Arrived(tick) => Some(tick),
            LaunchOutcome::HitObstacle { .. }
            | LaunchOutcome::Lost(_)
            | LaunchOutcome::NoContact => None,
        }
    }

    fn launch_outcome(
        state: &State,
        source: &Planet,
        target: &Planet,
        angle: f64,
        ships: i32,
    ) -> LaunchOutcome {
        let mut launched = state.clone();
        let mut control = state.clone();
        let mut launch_actions = vec![Vec::new(); state.config.player_count];
        launch_actions[source.owner as usize].push(LaunchAction {
            from_planet_id: source.id,
            angle,
            ships,
        });
        let empty_actions = vec![Vec::new(); state.config.player_count];
        let no_spawn = StepInjections {
            comet_spawn: Some(CometSpawnInjection::Skip),
        };
        let mut launch_rng = AuditRng::new(1);
        let mut control_rng = AuditRng::new(1);
        for tick in 1..=120 {
            let actions = if tick == 1 {
                &launch_actions
            } else {
                &empty_actions
            };
            step_with_injections(&mut launched, actions, &mut launch_rng, no_spawn.clone());
            step_with_injections(
                &mut control,
                &empty_actions,
                &mut control_rng,
                no_spawn.clone(),
            );
            if target_differs_from_control(&launched, &control, target.id) {
                return LaunchOutcome::Arrived(tick);
            }
            if let Some(planet_id) =
                first_non_target_planet_difference(&launched, &control, source.id, target.id)
            {
                return LaunchOutcome::HitObstacle { planet_id, tick };
            }
            if launched.fleets.len() < control.fleets.len() + usize::from(tick == 1) {
                return LaunchOutcome::Lost(tick);
            }
        }
        LaunchOutcome::NoContact
    }

    fn target_differs_from_control(launched: &State, control: &State, target_id: u32) -> bool {
        match (
            launched.planets.get(target_id),
            control.planets.get(target_id),
        ) {
            (Some(left), Some(right)) => left.owner != right.owner || left.ships != right.ships,
            (None, None) => false,
            _ => true,
        }
    }

    fn first_non_target_planet_difference(
        launched: &State,
        control: &State,
        source_id: u32,
        target_id: u32,
    ) -> Option<u32> {
        launched
            .planets
            .iter()
            .chain(control.planets.iter())
            .map(|planet| planet.id)
            .find(|planet_id| {
                *planet_id != source_id
                    && *planet_id != target_id
                    && target_differs_from_control(launched, control, *planet_id)
            })
    }

    fn audit_case(
        stats: &mut AngleAuditStats,
        state: &State,
        source_id: u32,
        target_id: u32,
        ships: i32,
    ) {
        let Some(source) = state.planets.get(source_id) else {
            return;
        };
        let Some(target) = state.planets.get(target_id) else {
            return;
        };
        if source.owner < 0 || source.ships < ships || source.id == target.id {
            return;
        }

        let target_is_dynamic = is_dynamic_planet_cached(state, target);
        stats.cases += 1;
        if target_is_dynamic {
            stats.dynamic_cases += 1;
        } else {
            stats.static_cases += 1;
        }

        let old_angle = old_target_angle(state, source, target, ships);
        let new_angle = new_target_angle_for_audit(state, source, target, ships);
        stats.old_found += usize::from(old_angle.is_some());
        stats.new_found += usize::from(new_angle.is_some());
        stats.both_found += usize::from(old_angle.is_some() && new_angle.is_some());

        let old_tick =
            old_angle.and_then(|angle| arrival_tick(state, source, target, angle, ships));
        let new_tick =
            new_angle.and_then(|angle| arrival_tick(state, source, target, angle, ships));
        if !target_is_dynamic {
            audit_static_angle_properties(stats, state, source, target, ships, new_angle);
        }
        stats.old_arrived += usize::from(old_tick.is_some());
        stats.new_arrived += usize::from(new_tick.is_some());
        stats.both_arrived += usize::from(old_tick.is_some() && new_tick.is_some());
        stats.new_only_arrived += usize::from(old_tick.is_none() && new_tick.is_some());
        stats.old_only_arrived += usize::from(old_tick.is_some() && new_tick.is_none());
        if old_tick.is_some() && new_tick.is_none() {
            println!(
                "old-only arrival: step={} source={} target={} dynamic={} ships={} old_angle={old_angle:?} new_angle={new_angle:?}",
                state.step, source.id, target.id, target_is_dynamic, ships
            );
        }
        if target_is_dynamic {
            stats.dynamic_old_arrived += usize::from(old_tick.is_some());
            stats.dynamic_new_arrived += usize::from(new_tick.is_some());
            stats.dynamic_new_only_arrived += usize::from(old_tick.is_none() && new_tick.is_some());
            stats.dynamic_old_only_arrived += usize::from(old_tick.is_some() && new_tick.is_none());
        }
        if let (Some(old_tick), Some(new_tick)) = (old_tick, new_tick) {
            match new_tick.cmp(&old_tick) {
                std::cmp::Ordering::Less => stats.new_faster += 1,
                std::cmp::Ordering::Equal => stats.new_equal += 1,
                std::cmp::Ordering::Greater => stats.new_slower += 1,
            }
            stats.arrival_tick_delta_sum += old_tick as i64 - new_tick as i64;
        }
    }

    fn audit_static_angle_properties(
        stats: &mut AngleAuditStats,
        state: &State,
        source: &Planet,
        target: &Planet,
        ships: i32,
        new_angle: Option<f64>,
    ) {
        let rays = static_target_rays(source, target);
        let center_angle = direct_target_angle(source, target);
        if arrival_tick(state, source, target, center_angle, ships).is_some() {
            stats.midpoint_valid += 1;
            if new_angle.is_some_and(|angle| angle_delta(angle, center_angle).abs() <= 1e-6) {
                stats.midpoint_selected += 1;
            }
        }

        if let Some(old_edge) = rays
            .iter()
            .skip(1)
            .find(|ray| arrival_tick(state, source, target, ray.angle, ships).is_some())
        {
            stats.outer_valid += 1;
            let old_offset = angle_delta(old_edge.angle, center_angle).abs();
            if new_angle
                .is_some_and(|angle| angle_delta(angle, center_angle).abs() <= old_offset + 1e-6)
            {
                stats.outer_matched_or_improved += 1;
            }
        }
    }

    fn challenging_audit_states() -> Vec<State> {
        let mut states = vec![
            state_from_planets(vec![
                planet(0, 0, 10.0, 80.0, 2.0, 500),
                planet(1, -1, 70.0, 80.0, 6.0, 100),
                planet(10, -1, 35.0, 80.0, 2.0, 20),
            ]),
            state_from_planets(vec![
                planet(0, 0, 10.0, 96.0, 2.0, 500),
                planet(1, -1, 35.0, 96.0, 2.0, 100),
                planet(10, -1, 70.0, 96.0, 6.0, 20),
            ]),
            state_from_planets(vec![
                planet(0, 0, 10.0, 80.0, 2.0, 500),
                planet(1, -1, 35.0, 80.0, 2.0, 20),
                planet(10, -1, 70.0, 80.0, 6.0, 20),
            ]),
        ];
        states[0].comet_planet_ids = vec![10];
        states[0].comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(35.0, 80.0); 40]],
            path_index: 0,
        }];
        states[1].comet_planet_ids = vec![10];
        states[1].comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(70.0, 96.0); 40]],
            path_index: 0,
        }];
        states[2].angular_velocity = 0.0;
        states[2].comet_planet_ids = vec![10];
        states[2].comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(70.0, 80.0); 40]],
            path_index: 0,
        }];
        states
    }

    fn known_adversarial_example_cases() -> Vec<(&'static str, State, u32, u32, i32)> {
        let mut delayed_comet_window = state_from_planets(vec![
            planet(0, 0, 10.0, 96.0, 2.0, 500),
            planet(1, -1, 35.0, 96.0, 5.0, 100),
            planet(10, -1, 70.0, 96.0, 6.0, 20),
        ]);
        delayed_comet_window.comet_planet_ids = vec![10];
        let mut delayed_path = vec![Point::new(70.0, 96.0); 32];
        delayed_path.extend(vec![Point::new(70.0, 75.0); 80]);
        delayed_comet_window.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![delayed_path],
            path_index: 0,
        }];

        let mut crossing_dynamic_blocker = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 500),
            planet(10, -1, 70.0, 80.0, 6.0, 20),
            planet(11, -1, 35.0, 8.0, 1.5, 20),
        ]);
        crossing_dynamic_blocker.comet_planet_ids = vec![10, 11];
        crossing_dynamic_blocker.comets = vec![CometGroup {
            planet_ids: vec![10, 11],
            paths: vec![
                vec![Point::new(70.0, 80.0); 80],
                (0..80)
                    .map(|tick| Point::new(35.0, 8.0 + 8.0 * f64::from(tick)))
                    .collect(),
            ],
            path_index: 0,
        }];

        let seed_900_after_70 = skipped_comet_seed_state(900, 70);
        vec![
            (
                "delayed comet window after blocked early windows",
                delayed_comet_window,
                0,
                10,
                6,
            ),
            (
                "crossing dynamic blocker requires offset",
                crossing_dynamic_blocker,
                0,
                10,
                20,
            ),
            (
                "seed 900 static target interior angle",
                seed_900_after_70.clone(),
                0,
                2,
                100,
            ),
            (
                "seed 900 orbiting target interior angle",
                seed_900_after_70,
                0,
                13,
                100,
            ),
        ]
    }

    fn skipped_comet_seed_state(seed: u64, steps: usize) -> State {
        let empty_actions = vec![Vec::new(); 4];
        let no_spawn = StepInjections {
            comet_spawn: Some(CometSpawnInjection::Skip),
        };
        let mut rng = AuditRng::new(seed);
        let mut state = reset_with_rng(ResetConfig::new(4), &mut rng);
        for _ in 0..steps {
            step_with_injections(&mut state, &empty_actions, &mut rng, no_spawn.clone());
        }
        state
    }

    fn audit_known_adversarial_examples(stats: &mut AngleAuditStats) {
        for (name, state, source_id, target_id, ships) in known_adversarial_example_cases() {
            audit_case(stats, &state, source_id, target_id, ships);
            let source = state
                .planets
                .get(source_id)
                .unwrap_or_else(|| panic!("{name}: missing source {source_id}"));
            let target = state
                .planets
                .get(target_id)
                .unwrap_or_else(|| panic!("{name}: missing target {target_id}"));
            let angle = new_target_angle_for_audit(&state, source, target, ships)
                .unwrap_or_else(|| panic!("{name}: solver did not find an angle"));
            let outcome = launch_outcome(&state, source, target, angle, ships);
            assert!(
                matches!(outcome, LaunchOutcome::Arrived(_)),
                "{name}: selected angle {angle} did not arrive; outcome={outcome:?}"
            );
        }
    }

    #[test]
    #[ignore = "expensive selected-angle quality audit; run with --ignored --nocapture"]
    fn audit_selected_target_angle_quality() {
        let mut stats = AngleAuditStats::default();
        for state in challenging_audit_states() {
            audit_state_pairs(&mut stats, &state, 100, 64);
        }
        audit_known_adversarial_examples(&mut stats);

        let empty_actions = vec![Vec::new(); 4];
        for seed in 0..24_u64 {
            let mut rng = AuditRng::new(seed + 100);
            let mut state = reset_with_rng(ResetConfig::new(4), &mut rng);
            audit_state_pairs(&mut stats, &state, 40, 96);
            let no_spawn = StepInjections {
                comet_spawn: Some(CometSpawnInjection::Skip),
            };
            for _ in 0..70 {
                step_with_injections(&mut state, &empty_actions, &mut rng, no_spawn.clone());
            }
            audit_state_pairs(&mut stats, &state, 40, 96);
        }

        println!("selected-angle audit, cases: {}", stats.cases);
        println!(
            "selected-angle audit, found old/new/both: {}/{}/{}",
            stats.old_found, stats.new_found, stats.both_found
        );
        println!(
            "selected-angle audit, arrived old/new/both: {}/{}/{}",
            stats.old_arrived, stats.new_arrived, stats.both_arrived
        );
        println!(
            "selected-angle audit, arrivals old-only/new-only: {}/{}",
            stats.old_only_arrived, stats.new_only_arrived
        );
        println!(
            "selected-angle audit, arrival tick comparison among both arrivals, new faster/equal/slower: {}/{}/{}; total old-new tick delta: {}",
            stats.new_faster, stats.new_equal, stats.new_slower, stats.arrival_tick_delta_sum
        );
        println!(
            "selected-angle audit, static midpoint valid/selected: {}/{}",
            stats.midpoint_valid, stats.midpoint_selected
        );
        println!(
            "selected-angle audit, static outer matched-or-improved/valid: {}/{}",
            stats.outer_matched_or_improved, stats.outer_valid
        );
        println!(
            "selected-angle audit, dynamic arrivals old/new: {}/{}; dynamic arrivals old-only/new-only: {}/{}",
            stats.dynamic_old_arrived,
            stats.dynamic_new_arrived,
            stats.dynamic_old_only_arrived,
            stats.dynamic_new_only_arrived
        );

        assert_eq!(
            stats.old_only_arrived, 0,
            "new selected angles should not lose arrivals found by the previous algorithm"
        );
        assert_eq!(
            stats.midpoint_valid, stats.midpoint_selected,
            "valid static midpoint rays should still be selected"
        );
        assert_eq!(
            stats.outer_valid, stats.outer_matched_or_improved,
            "valid static edge rays should be matched or improved toward center"
        );
    }

    fn audit_state_pairs(
        stats: &mut AngleAuditStats,
        state: &State,
        max_sources: usize,
        max_cases: usize,
    ) {
        let mut cases = 0;
        for source in state
            .planets
            .iter()
            .filter(|planet| planet.owner >= 0 && planet.ships >= 12)
            .take(max_sources)
        {
            let ships = (source.ships / 2).clamp(6, 100);
            for target in state.planets.iter().filter(|target| target.id != source.id) {
                audit_case(stats, state, source.id, target.id, ships);
                cases += 1;
                if cases >= max_cases {
                    return;
                }
            }
        }
    }

    #[test]
    #[ignore = "expensive adversarial selected-angle search; run with --ignored --nocapture"]
    fn adversarial_selected_target_angle_search() {
        let mut cases = 0_usize;
        let mut brute_arrivals = 0_usize;
        let mut solver_misses = 0_usize;
        let mut solver_obstacle_hits = 0_usize;
        let mut old_finds_missed = 0_usize;

        for state in adversarial_search_states() {
            for source in state
                .planets
                .iter()
                .filter(|planet| planet.owner >= 0 && planet.ships >= 12)
            {
                let ships = (source.ships / 2).clamp(6, 100);
                for target in state.planets.iter().filter(|target| target.id != source.id) {
                    cases += 1;
                    let brute = brute_force_arrival_angle(&state, source, target, ships);
                    let solver_angle = new_target_angle_for_audit(&state, source, target, ships);
                    let solver_outcome = solver_angle
                        .map(|angle| launch_outcome(&state, source, target, angle, ships));
                    let solver_tick = match solver_outcome {
                        Some(LaunchOutcome::Arrived(tick)) => Some(tick),
                        _ => None,
                    };
                    let old_angle = old_target_angle(&state, source, target, ships);
                    let old_outcome =
                        old_angle.map(|angle| launch_outcome(&state, source, target, angle, ships));
                    let old_tick = match old_outcome {
                        Some(LaunchOutcome::Arrived(tick)) => Some(tick),
                        _ => None,
                    };

                    if brute.is_some() {
                        brute_arrivals += 1;
                    }
                    if let Some((brute_angle, brute_tick)) = brute {
                        if solver_tick.is_none() {
                            solver_misses += 1;
                            solver_obstacle_hits += usize::from(matches!(
                                solver_outcome,
                                Some(LaunchOutcome::HitObstacle { .. })
                            ));
                            old_finds_missed += usize::from(old_tick.is_some());
                            println!(
                                "adversarial miss, step={} source={} target={} target_dynamic={} ships={} brute_angle={} brute_tick={} old_angle={old_angle:?} old_outcome={old_outcome:?} solver_angle={solver_angle:?} solver_outcome={solver_outcome:?}",
                                state.step,
                                source.id,
                                target.id,
                                is_dynamic_planet_cached(&state, target),
                                ships,
                                brute_angle,
                                brute_tick,
                            );
                        }
                    }
                }
            }
        }

        println!("adversarial search, cases: {cases}");
        println!("adversarial search, brute arrivals: {brute_arrivals}");
        println!("adversarial search, solver misses: {solver_misses}");
        println!("adversarial search, solver miss obstacle-hits: {solver_obstacle_hits}");
        println!("adversarial search, solver misses old-found: {old_finds_missed}");

        assert_eq!(solver_misses, 0);
    }

    fn adversarial_search_states() -> Vec<State> {
        let mut states = challenging_audit_states();
        let empty_actions = vec![Vec::new(); 4];
        for seed in 0..16_u64 {
            let mut rng = AuditRng::new(seed + 900);
            let mut state = reset_with_rng(ResetConfig::new(4), &mut rng);
            states.push(state.clone());
            let no_spawn = StepInjections {
                comet_spawn: Some(CometSpawnInjection::Skip),
            };
            for _ in 0..70 {
                step_with_injections(&mut state, &empty_actions, &mut rng, no_spawn.clone());
            }
            states.push(state);
        }
        states
    }

    fn brute_force_arrival_angle(
        state: &State,
        source: &Planet,
        target: &Planet,
        ships: i32,
    ) -> Option<(f64, usize)> {
        let mut best = None;
        for angle in brute_force_candidate_angles(state, source, target, ships) {
            let Some(tick) = arrival_tick(state, source, target, angle, ships) else {
                continue;
            };
            if best.is_none_or(|(_, best_tick)| tick < best_tick) {
                best = Some((angle, tick));
            }
        }
        best
    }

    fn brute_force_candidate_angles(
        state: &State,
        source: &Planet,
        target: &Planet,
        ships: i32,
    ) -> Vec<f64> {
        let mut angles = Vec::new();
        let speed = fleet_speed(ships, state.config.ship_speed);
        if is_dynamic_planet_cached(state, target) {
            push_dynamic_brute_force_angles(state, source, target, speed, &mut angles);
        } else {
            push_static_brute_force_angles(source, target, &mut angles);
        }
        if let Some(angle) = old_target_angle(state, source, target, ships) {
            angles.push(angle);
        }
        if let Some(angle) = new_target_angle_for_audit(state, source, target, ships) {
            angles.push(angle);
        }
        dedup_angles(&mut angles);
        angles
    }

    fn push_static_brute_force_angles(source: &Planet, target: &Planet, angles: &mut Vec<f64>) {
        let center = direct_target_angle(source, target);
        let distance_to_target = distance(source.position(), target.position());
        let target_radius = (target.radius - DYNAMIC_TARGET_EPS).max(0.0);
        let half_angle = if distance_to_target <= target_radius {
            PI
        } else if target_radius > 0.0 {
            (target_radius / distance_to_target).asin()
        } else {
            0.0
        };
        for sample in 0..=96 {
            let fraction = f64::from(sample) / 96.0;
            angles.push(normalize_angle(
                center - half_angle + 2.0 * half_angle * fraction,
            ));
        }
        for ray in static_target_rays(source, target) {
            angles.push(ray.angle);
        }
    }

    fn push_dynamic_brute_force_angles(
        state: &State,
        source: &Planet,
        target: &Planet,
        speed: f64,
        angles: &mut Vec<f64>,
    ) {
        let entities = action_entity_slots(state);
        let mut cache = OrbitTargetCache::new(state, &entities, 1);
        if state.comet_planet_ids.contains(&target.id) {
            if let Some(path) = cache.dynamic_path_for(target) {
                for candidate in
                    piecewise_linear_target_candidates(source, target.radius, speed, path)
                {
                    angles.push(candidate.angle);
                }
                for window in
                    piecewise_linear_target_windows_uncapped(source, target.radius, speed, path)
                {
                    push_window_brute_force_angles(source, target.radius, speed, window, angles);
                }
            }
            return;
        }
        if let Some(path) = cache.path_for(target) {
            let Some(initial_target) = state.initial_planets.get(target.id) else {
                return;
            };
            let (min_time, max_time) = orbit_time_bounds(source, initial_target, target, speed);
            for candidate in piecewise_linear_target_candidates_in_time_range(
                source,
                target.radius,
                speed,
                path,
                min_time,
                max_time,
            ) {
                angles.push(candidate.angle);
            }
            for window in piecewise_linear_target_windows_in_time_range_uncapped(
                source,
                target.radius,
                speed,
                path,
                min_time,
                max_time,
            ) {
                push_window_brute_force_angles(source, target.radius, speed, window, angles);
            }
        }
    }

    fn push_window_brute_force_angles(
        source: &Planet,
        target_radius: f64,
        speed: f64,
        window: TargetWindow,
        angles: &mut Vec<f64>,
    ) {
        for candidate in dynamic_window_centerline_candidates(source, speed, window) {
            angles.push(candidate.angle);
        }
        let reference_angle = window_midpoint_angle(source, window);
        for span in target_arc_spans_for_window(source, target_radius, speed, window) {
            for relative_angle in [
                span.start + ANGLE_CHOICE_EPS,
                (span.start + span.end) / 2.0,
                span.end - ANGLE_CHOICE_EPS,
            ] {
                angles.push(normalize_angle(reference_angle + relative_angle));
            }
        }
    }

    fn piecewise_linear_target_windows_uncapped(
        source: &Planet,
        radius: f64,
        speed: f64,
        path: &[Point],
    ) -> Vec<TargetWindow> {
        piecewise_linear_target_windows_in_time_range_uncapped(
            source,
            radius,
            speed,
            path,
            0.0,
            f64::INFINITY,
        )
    }

    fn piecewise_linear_target_windows_in_time_range_uncapped(
        source: &Planet,
        radius: f64,
        speed: f64,
        path: &[Point],
        min_time: f64,
        max_time: f64,
    ) -> Vec<TargetWindow> {
        let mut windows = Vec::new();
        if path.len() < 2 || speed <= 0.0 {
            return windows;
        }

        let source_pos = source.position();
        let source_clearance = source.radius + 0.1;
        let target_radius = (radius - DYNAMIC_TARGET_EPS).max(0.0);
        let first_segment = min_time.floor().max(0.0) as usize;
        let last_segment = max_time.ceil().max(1.0) as usize;
        let last_segment = last_segment.min(path.len().saturating_sub(1));
        for segment_index in first_segment..last_segment {
            let start = path[segment_index];
            let end = path[segment_index + 1];
            let segment_time = segment_index as f64;
            if !target_window_distance_band_can_intersect(
                source_pos,
                start,
                end,
                source_clearance,
                target_radius,
                speed,
                segment_time,
            ) {
                continue;
            }

            let lower_fraction = (min_time - segment_time).clamp(0.0, 1.0);
            let upper_fraction = (max_time - segment_time).clamp(0.0, 1.0);
            if upper_fraction - lower_fraction <= ROOT_EPS {
                continue;
            }

            let mut roots = vec![lower_fraction, upper_fraction];
            push_distance_linear_roots(
                &mut roots,
                source_pos,
                start,
                end,
                source_clearance + target_radius + speed * segment_time,
                speed,
            );
            push_distance_linear_roots(
                &mut roots,
                source_pos,
                start,
                end,
                source_clearance - target_radius + speed * segment_time,
                speed,
            );
            push_distance_linear_roots(
                &mut roots,
                source_pos,
                start,
                end,
                target_radius - source_clearance - speed * segment_time,
                -speed,
            );
            roots.sort_by(f64::total_cmp);
            roots.dedup_by(|left, right| (*left - *right).abs() <= ROOT_EPS);

            for pair in roots.windows(2) {
                let start_fraction = pair[0].max(lower_fraction);
                let end_fraction = pair[1].min(upper_fraction);
                if end_fraction - start_fraction <= ROOT_EPS {
                    continue;
                }
                let mid_time = segment_time + (start_fraction + end_fraction) / 2.0;
                let target_pos = lerp(start, end, mid_time - segment_time);
                if instantaneous_target_arc(
                    source_pos,
                    source_clearance,
                    target_pos,
                    target_radius,
                    speed,
                    mid_time,
                    angle_between(source_pos, target_pos),
                )
                .is_none()
                {
                    continue;
                }
                windows.push(TargetWindow {
                    start_time: segment_time + start_fraction,
                    end_time: segment_time + end_fraction,
                    segment_index,
                    segment_start: start,
                    segment_end: end,
                });
            }
        }
        windows
    }

    fn dedup_angles(angles: &mut Vec<f64>) {
        angles.sort_by(f64::total_cmp);
        angles.dedup_by(|left, right| angle_delta(*left, *right).abs() <= 1e-6);
    }

    fn candidate_hits_swept_segment(
        source: &Planet,
        target_radius: f64,
        path: &[Point],
        speed: f64,
        candidate: TargetCandidate,
    ) -> bool {
        assert!(
            path.len() >= 2,
            "candidate validation requires a path segment"
        );
        let segment = (candidate.time.floor() as usize).min(path.len() - 2);
        let segment_time = segment as f64;
        let fleet_start = point_along(
            launch_start(source, candidate.angle),
            candidate.angle,
            speed * segment_time,
        );
        let fleet_end = point_along(
            launch_start(source, candidate.angle),
            candidate.angle,
            speed * (segment_time + 1.0),
        );

        swept_pair_hit(
            fleet_start,
            fleet_end,
            path[segment],
            path[segment + 1],
            target_radius,
        )
    }

    fn test_orbit_path(target: &Planet, angular_velocity: f64, point_count: u32) -> OrbitPath {
        OrbitPath {
            planet_id: target.id,
            points: (0..point_count)
                .map(|tick| {
                    if tick == 0 {
                        target.position()
                    } else {
                        orbit_position(target.position(), angular_velocity, f64::from(tick))
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn discrete_targets_mask_uses_source_target_square() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 10),
            planet(1, -1, 70.0, 80.0, 3.0, 10),
            planet(10, 2, 30.0, 80.0, 1.0, 10),
        ]);
        state.comet_planet_ids = vec![10];
        state.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(30.0, 80.0), Point::new(32.0, 80.0)]],
            path_index: 0,
        }];
        let entities = action_entity_slots(&state);
        let spec = RlActionSpec::DiscreteTargets {
            targeting_mode: TargetingMode::FullMask,
        };
        let mut can_act = vec![false; spec.can_act_len()];
        let mut max_launch = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];

        encode_action_spec(
            spec,
            &state,
            &PlayerMap::identity(),
            &entities,
            &mut can_act,
            Some(&mut max_launch),
            1,
        );

        let base = 0;
        assert!(!can_act[base]);
        assert!(can_act[base + 1]);
        assert!(can_act[base + MAX_PLANETS]);
        assert!(!can_act[base + 2]);
        assert_eq!(max_launch[0], 10);
        assert_eq!(entities[MAX_PLANETS].map(|slot| slot.planet_id), Some(10));
    }

    #[test]
    fn discrete_targets_mask_rejects_sun_blocked_static_targets() {
        let state = state_from_planets(vec![
            planet(0, 0, 0.0, 50.0, 2.0, 10),
            planet(1, -1, 100.0, 50.0, 2.0, 10),
            planet(2, -1, 100.0, 80.0, 2.0, 10),
        ]);
        let entities = action_entity_slots(&state);
        let spec = RlActionSpec::DiscreteTargets {
            targeting_mode: TargetingMode::FullMask,
        };
        let mut can_act = vec![false; spec.can_act_len()];
        let mut max_launch = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];

        encode_action_spec(
            spec,
            &state,
            &PlayerMap::identity(),
            &entities,
            &mut can_act,
            Some(&mut max_launch),
            1,
        );

        assert!(
            !can_act[1],
            "source-target ray through the sun should be masked"
        );
        assert!(
            can_act[2],
            "unobstructed static target should remain eligible"
        );
    }

    #[test]
    fn loose_discrete_target_modes_do_not_mask_statically_obstructed_targets() {
        let state = state_from_planets(vec![
            planet(0, 0, 0.0, 50.0, 2.0, 10),
            planet(1, -1, 100.0, 50.0, 2.0, 10),
        ]);
        let entities = action_entity_slots(&state);

        for targeting_mode in [TargetingMode::AnythingGoes, TargetingMode::StopBadLaunch] {
            let spec = RlActionSpec::DiscreteTargets { targeting_mode };
            let mut can_act = vec![false; spec.can_act_len()];
            let mut max_launch = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];

            encode_action_spec(
                spec,
                &state,
                &PlayerMap::identity(),
                &entities,
                &mut can_act,
                Some(&mut max_launch),
                1,
            );

            assert!(can_act[1]);
            assert!(!can_act[0]);
            assert!(!can_act[2]);
            assert_eq!(max_launch[0], 10);
        }
    }

    #[test]
    fn loose_discrete_target_bin_modes_do_not_mask_statically_obstructed_targets() {
        let state = state_from_planets(vec![
            planet(0, 0, 0.0, 50.0, 2.0, 10),
            planet(1, -1, 100.0, 50.0, 2.0, 10),
        ]);
        let entities = action_entity_slots(&state);

        for targeting_mode in [TargetingMode::AnythingGoes, TargetingMode::StopBadLaunch] {
            let spec = RlActionSpec::DiscreteTargetBins {
                n_bins: 5,
                targeting_mode,
            };
            let mut can_act = vec![false; spec.can_act_len()];

            encode_action_spec(
                spec,
                &state,
                &PlayerMap::identity(),
                &entities,
                &mut can_act,
                None,
                1,
            );

            let target_base = 5;
            assert!(can_act[target_base]);
            assert!(can_act[target_base + 4]);
            assert!(!can_act[0..5].iter().any(|eligible| *eligible));
            assert!(!can_act[10..15].iter().any(|eligible| *eligible));
        }
    }

    #[test]
    fn fleet_bin_mapping_uses_half_up_rounding() {
        let ships = (0..5)
            .map(|fleet_bin| fleet_bin_to_ships(fleet_bin, 10, 5))
            .collect::<Vec<_>>();
        assert_eq!(ships, vec![0, 3, 5, 8, 10]);
        assert_eq!(ships_to_fleet_bin(3, 10, 5), 1);
        assert_eq!(ships_to_fleet_bin(8, 10, 5), 3);
    }

    #[test]
    fn discrete_target_bins_mask_keeps_only_higher_duplicate_bins() {
        let state = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 3),
            planet(1, -1, 70.0, 80.0, 3.0, 10),
        ]);
        let entities = action_entity_slots(&state);
        let spec = RlActionSpec::DiscreteTargetBins {
            n_bins: 8,
            targeting_mode: TargetingMode::FullMask,
        };
        let mut can_act = vec![false; spec.can_act_len()];

        encode_action_spec(
            spec,
            &state,
            &PlayerMap::identity(),
            &entities,
            &mut can_act,
            None,
            1,
        );

        let base = 8;
        assert_eq!(
            &can_act[base..base + 8],
            &[true, false, false, true, false, true, false, true]
        );
    }

    #[test]
    fn discrete_target_bins_decode_target_and_fleet_bin() {
        let state = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 10),
            planet(1, -1, 70.0, 80.0, 3.0, 10),
        ]);
        let entities = action_entity_slots(&state);
        let mut targets = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut fleet_bins = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        targets[0] = 1;
        fleet_bins[0] = 2;

        let decoded = decode_discrete_target_bin_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &targets,
            &fleet_bins,
            5,
            1,
            TargetingMode::FullMask,
        )
        .expect("valid target-bin action should decode");

        assert_eq!(decoded.launch_failures, 0);
        assert_eq!(decoded.actions[0].len(), 1);
        assert_eq!(decoded.actions[0][0].from_planet_id, 0);
        assert_eq!(decoded.actions[0][0].ships, 5);
    }

    #[test]
    fn discrete_target_bins_zero_bin_decodes_as_no_launch() {
        let state = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 10),
            planet(1, -1, 70.0, 80.0, 3.0, 10),
        ]);
        let entities = action_entity_slots(&state);
        let mut targets = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let fleet_bins = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        targets[0] = 1;

        let decoded = decode_discrete_target_bin_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &targets,
            &fleet_bins,
            5,
            1,
            TargetingMode::FullMask,
        )
        .expect("zero fleet bin should decode as a no-op");

        assert_eq!(decoded.launch_failures, 0);
        assert!(decoded.actions.iter().all(Vec::is_empty));
    }

    #[test]
    fn static_source_static_target_cache_gates_full_mask_mask() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 5.0, 80.0, 2.0, 10),
            planet(1, -1, 95.0, 80.0, 2.0, 10),
        ]);
        state.static_target_cache =
            StaticTargetCache::new(crate::rules_engine::state::MAX_PLANET_ID as usize);
        state.static_target_cache.set(0, 1, 0.25);
        assert!(discrete_target_can_act(&state, 0, 1, 1));

        state.static_target_cache =
            StaticTargetCache::new(crate::rules_engine::state::MAX_PLANET_ID as usize);
        assert!(!discrete_target_can_act(&state, 0, 1, 1));
    }

    #[test]
    fn reset_precomputes_static_target_cache() {
        let planets = vec![
            planet(0, 0, 0.0, 50.0, 2.0, 10),
            planet(1, -1, 100.0, 50.0, 2.0, 10),
            planet(2, -1, 100.0, 80.0, 2.0, 10),
        ];
        let mut config = ResetConfig::new(4);
        config.planets = Some(planets.clone());
        config.initial_planets = Some(planets);
        config.angular_velocity = Some(0.025);

        let state = reset(config);

        assert!(state.static_target_cache.get(0, 1).is_none());
        assert!(state.static_target_cache.get(0, 2).is_some());
    }

    #[test]
    fn dynamic_target_launch_crossing_sun_is_counted_as_launch_failure() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 20.0, 50.0, 2.0, 100),
            planet(10, -1, 80.0, 50.0, 1.0, 20),
        ]);
        state.comet_planet_ids = vec![10];
        state.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(80.0, 50.0); 20]],
            path_index: 0,
        }];
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = MAX_PLANETS as i64;
        ships[0] = 100;

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("dynamic target launch failure should decode as a no-op");

        assert_eq!(decoded.launch_failures, 1);
        assert!(decoded.actions.iter().all(Vec::is_empty));
    }

    #[test]
    fn full_mask_sun_blocked_static_target_decodes_as_no_op() {
        let planets = vec![
            planet(0, 0, 0.0, 50.0, 2.0, 100),
            planet(1, -1, 100.0, 50.0, 2.0, 20),
            planet(2, -1, 100.0, 80.0, 2.0, 20),
        ];
        let mut config = ResetConfig::new(4);
        config.planets = Some(planets.clone());
        config.initial_planets = Some(planets);
        config.angular_velocity = Some(0.025);
        let state = reset(config);
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = 1;
        ships[0] = 100;

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("sun-blocked static target should decode as no-op");

        assert_eq!(decoded.launch_failures, 1);
        assert!(decoded.actions.iter().all(Vec::is_empty));
    }

    #[test]
    fn stop_bad_launch_replaces_sun_crossing_static_target_with_no_op() {
        let state = state_from_planets(vec![
            planet(0, 0, 0.0, 50.0, 2.0, 100),
            planet(1, -1, 100.0, 50.0, 2.0, 20),
        ]);
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = 1;
        ships[0] = 100;

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::StopBadLaunch,
        )
        .expect("stop_bad_launch should decode sun-crossing target as no-op");

        assert_eq!(decoded.launch_failures, 1);
        assert!(decoded.actions.iter().all(Vec::is_empty));
    }

    #[test]
    fn stop_bad_launch_falls_back_to_sun_safe_blocked_static_target() {
        let state = state_from_planets(vec![
            planet(0, 0, 10.0, 96.0, 2.0, 100),
            planet(1, -1, 35.0, 96.0, 8.0, 20),
            planet(2, -1, 70.0, 96.0, 2.0, 20),
        ]);
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = 2;
        ships[0] = 100;

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::StopBadLaunch,
        )
        .expect("stop_bad_launch should submit a blocker-only fallback");

        assert_eq!(decoded.launch_failures, 0);
        assert_eq!(decoded.actions[0].len(), 1);
        assert!(decoded.actions[0][0].angle.abs() <= 1e-6);
    }

    #[test]
    fn full_mask_falls_back_to_sun_safe_blocked_dynamic_target() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 10.0, 96.0, 2.0, 100),
            planet(1, -1, 35.0, 96.0, 8.0, 20),
            planet(10, -1, 70.0, 96.0, 2.0, 20),
        ]);
        state.comet_planet_ids = vec![10];
        state.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(70.0, 96.0); 40]],
            path_index: 0,
        }];
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = MAX_PLANETS as i64;
        ships[0] = 100;

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("full_mask should submit a blocker-only dynamic fallback");

        assert_eq!(decoded.launch_failures, 0);
        assert_eq!(decoded.actions[0].len(), 1);
        assert!(decoded.actions[0][0].angle.abs() <= 1e-6);
    }

    #[test]
    fn anything_goes_submits_sun_crossing_static_target() {
        let state = state_from_planets(vec![
            planet(0, 0, 0.0, 50.0, 2.0, 100),
            planet(1, -1, 100.0, 50.0, 2.0, 20),
        ]);
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = 1;
        ships[0] = 100;

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::AnythingGoes,
        )
        .expect("anything_goes should submit sun-crossing target");

        assert_eq!(decoded.launch_failures, 0);
        assert_eq!(decoded.actions[0].len(), 1);
        assert_eq!(decoded.actions[0][0].from_planet_id, 0);
    }

    #[test]
    fn anything_goes_target_bins_submit_sun_crossing_static_target() {
        let state = state_from_planets(vec![
            planet(0, 0, 0.0, 50.0, 2.0, 100),
            planet(1, -1, 100.0, 50.0, 2.0, 20),
        ]);
        let entities = action_entity_slots(&state);
        let mut targets = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut fleet_bins = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        targets[0] = 1;
        fleet_bins[0] = 4;

        let decoded = decode_discrete_target_bin_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &targets,
            &fleet_bins,
            5,
            1,
            TargetingMode::AnythingGoes,
        )
        .expect("anything_goes target bins should submit sun-crossing target");

        assert_eq!(decoded.launch_failures, 0);
        assert_eq!(decoded.actions[0].len(), 1);
        assert_eq!(decoded.actions[0][0].from_planet_id, 0);
    }

    #[test]
    fn stop_bad_launch_target_bins_replace_sun_crossing_static_target_with_no_op() {
        let state = state_from_planets(vec![
            planet(0, 0, 0.0, 50.0, 2.0, 100),
            planet(1, -1, 100.0, 50.0, 2.0, 20),
        ]);
        let entities = action_entity_slots(&state);
        let mut targets = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut fleet_bins = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        targets[0] = 1;
        fleet_bins[0] = 4;

        let decoded = decode_discrete_target_bin_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &targets,
            &fleet_bins,
            5,
            1,
            TargetingMode::StopBadLaunch,
        )
        .expect("stop_bad_launch target bins should decode sun-crossing target as no-op");

        assert_eq!(decoded.launch_failures, 1);
        assert!(decoded.actions.iter().all(Vec::is_empty));
    }

    #[test]
    fn full_mask_target_bins_reject_masked_nonzero_target_bin() {
        let planets = vec![
            planet(0, 0, 0.0, 50.0, 2.0, 100),
            planet(1, -1, 100.0, 50.0, 2.0, 20),
            planet(2, -1, 100.0, 80.0, 2.0, 20),
        ];
        let mut config = ResetConfig::new(4);
        config.planets = Some(planets.clone());
        config.initial_planets = Some(planets);
        config.angular_velocity = Some(0.025);
        let state = reset(config);
        let entities = action_entity_slots(&state);
        let mut targets = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        let mut fleet_bins = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];
        targets[0] = 1;
        fleet_bins[0] = 4;

        let err = decode_discrete_target_bin_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &targets,
            &fleet_bins,
            5,
            1,
            TargetingMode::FullMask,
        )
        .expect_err("nonzero target-bin actions must obey full-mask can_act");

        assert!(err.contains("masked by can_act"));
    }

    #[test]
    fn dynamic_target_launch_ending_out_of_bounds_is_counted_as_launch_failure() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 90.0, 80.0, 2.0, 100),
            planet(10, -1, 120.0, 80.0, 1.0, 20),
        ]);
        state.comet_planet_ids = vec![10];
        state.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(120.0, 80.0); 20]],
            path_index: 0,
        }];
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = MAX_PLANETS as i64;
        ships[0] = 100;

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("dynamic target launch failure should decode as a no-op");

        assert_eq!(decoded.launch_failures, 1);
        assert!(decoded.actions.iter().all(Vec::is_empty));
    }

    #[test]
    fn discrete_targets_rejects_invalid_targets() {
        let state = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 10),
            planet(1, -1, 70.0, 80.0, 3.0, 10),
        ]);
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        ships[0] = 1;

        let err = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect_err("self target should fail");
        assert!(err.contains("cannot target itself"));

        targets[0] = 2;
        let err = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect_err("empty target should fail");
        assert!(err.contains("cannot target empty action entity slot 2"));
    }

    #[test]
    fn discrete_static_target_tangent_fallback_hits_in_simulator() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 10.0, 99.0, 2.0, 200),
            planet(1, -1, 40.0, 99.0, 1.0, 100),
            planet(2, -1, 70.0, 99.0, 5.0, 100),
        ]);
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = 2;
        ships[0] = 100;
        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("valid discrete target should decode");

        let angle = decoded.actions[0][0].angle;
        assert!(
            (0.02..0.06).contains(&angle.abs()),
            "arc subtraction should pick the nearest unblocked angle, got {angle}",
        );

        step(&mut state, &decoded.actions);

        assert!(run_until_planet_changes(&mut state, 2, 100));
        assert_eq!(state.planets.get(1).expect("blocker").ships, 100);
    }

    #[test]
    fn replay_static_target_to_57_ship_planet_arrives_for_slow_fleets() {
        for ship_count in [1, 2, 6, 11] {
            let state = replay_77263963_static_target_state(ship_count);
            assert!(discrete_target_can_act(
                &state,
                15,
                10,
                i64::from(ship_count)
            ));
            let decoded = decode_single_discrete_target(&state, 15, 10, ship_count);

            assert_eq!(decoded.launch_failures, 0);
            assert_eq!(decoded.actions[1].len(), 1);
            let action = &decoded.actions[1][0];
            let source = state.planets.get(15).expect("source");
            let target = state.planets.get(10).expect("target");

            assert!(
                matches!(
                    launch_outcome(&state, source, target, action.angle, ship_count),
                    LaunchOutcome::Arrived(_)
                ),
                "{ship_count}-ship replay launch should arrive at the 57-ship planet"
            );
            assert_eq!(
                state.planets.get(7).expect("blocker").ships,
                24,
                "regression setup should keep the 24-ship blocker unchanged before launch"
            );
        }
    }

    #[test]
    fn replay_far_static_target_blocked_by_57_ship_planet_is_masked_for_slow_fleets() {
        for ship_count in [1, 2, 6] {
            let state = replay_77263963_static_target_state(ship_count);
            assert!(!discrete_target_can_act(
                &state,
                15,
                2,
                i64::from(ship_count)
            ));

            let decoded = decode_single_discrete_target(&state, 15, 2, ship_count);
            assert_eq!(decoded.launch_failures, 0);
            assert_eq!(decoded.actions[1].len(), 1);

            let source = state.planets.get(15).expect("source");
            let target = state.planets.get(2).expect("target");
            let outcome = launch_outcome(
                &state,
                source,
                target,
                decoded.actions[1][0].angle,
                ship_count,
            );
            assert!(
                matches!(outcome, LaunchOutcome::HitObstacle { planet_id: 7, .. }),
                "selected masked launch should hit blocker, got {outcome:?}"
            );
        }
    }

    #[test]
    fn discrete_comet_target_arc_subtracts_static_blocker() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 10.0, 96.0, 2.0, 500),
            planet(1, -1, 35.0, 96.0, 2.0, 100),
            planet(10, -1, 70.0, 96.0, 6.0, 20),
        ]);
        state.comet_planet_ids = vec![10];
        state.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(70.0, 96.0); 40]],
            path_index: 0,
        }];
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = MAX_PLANETS as i64;
        ships[0] = 100;

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("valid comet target should decode");

        assert_eq!(decoded.launch_failures, 0);
        assert_eq!(decoded.actions[0].len(), 1);
        let angle = decoded.actions[0][0].angle;
        assert!(
            (0.07..0.12).contains(&angle.abs()),
            "dynamic arc subtraction should route around blocker, got {angle}",
        );

        step(&mut state, &decoded.actions);

        assert!(run_until_planet_changes(&mut state, 10, 20));
        assert_eq!(state.planets.get(1).expect("blocker").ships, 100);
    }

    #[test]
    fn discrete_static_target_arc_subtracts_dynamic_blocker() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 500),
            planet(1, -1, 70.0, 80.0, 6.0, 100),
            planet(10, -1, 35.0, 80.0, 2.0, 20),
        ]);
        state.comet_planet_ids = vec![10];
        state.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(35.0, 80.0); 40]],
            path_index: 0,
        }];
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = 1;
        ships[0] = 100;

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("valid static target should decode");

        assert_eq!(decoded.launch_failures, 0);
        let angle = decoded.actions[0][0].angle;
        assert!(
            (0.07..0.12).contains(&angle.abs()),
            "static target should route around dynamic blocker, got {angle}",
        );

        step(&mut state, &decoded.actions);

        assert!(run_until_planet_changes(&mut state, 1, 100));
        assert_eq!(state.planets.get(10).expect("blocker").ships, 20);
    }

    #[test]
    fn discrete_comet_target_arc_subtracts_dynamic_blocker() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 500),
            planet(1, -1, 35.0, 80.0, 2.0, 20),
            planet(10, -1, 70.0, 80.0, 6.0, 20),
        ]);
        state.angular_velocity = 0.0;
        state.comet_planet_ids = vec![10];
        state.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(70.0, 80.0); 40]],
            path_index: 0,
        }];
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = MAX_PLANETS as i64;
        ships[0] = 100;

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("valid comet target should decode");

        assert_eq!(decoded.launch_failures, 0);
        let angle = decoded.actions[0][0].angle;
        assert!(
            (0.07..0.12).contains(&angle.abs()),
            "dynamic target should route around dynamic blocker, got {angle}",
        );

        step(&mut state, &decoded.actions);

        assert!(run_until_planet_changes(&mut state, 10, 20));
        assert_eq!(state.planets.get(1).expect("blocker").ships, 20);
    }

    #[test]
    fn discrete_orbiting_target_hits_in_simulator() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 97.0, 50.0, 3.0, 500),
            planet(1, -1, 50.0, 20.0, 3.0, 100),
        ]);
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = 1;
        ships[0] = 300;
        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("valid orbiting target should decode");

        step(&mut state, &decoded.actions);

        assert!(run_until_planet_changes(&mut state, 1, 100));
    }

    #[test]
    fn orbiting_target_candidate_solves_inflated_radius_intercept() {
        let state = state_from_planets(vec![
            planet(0, 0, 97.0, 50.0, 3.0, 500),
            planet(1, -1, 50.0, 20.0, 3.0, 100),
        ]);
        let source = state.planets.get(0).expect("source");
        let target = state.planets.get(1).expect("target");
        let speed = fleet_speed(300, state.config.ship_speed);

        let candidate = cached_orbiting_target_candidate(&state, source, target, speed)
            .expect("orbiting target should have an intercept");
        let initial_target = state.initial_planets.get(1).expect("initial target");
        let target_pos = orbit_position(
            initial_target.position(),
            state.angular_velocity,
            state.step as f64 + candidate.time,
        );
        let clearance = source.radius + 0.1 + target.radius;
        let residual = distance(target_pos, source.position()) - clearance - speed * candidate.time;

        assert!(residual.abs() <= 1e-3, "residual {residual}");
        assert!((candidate.angle - angle_between(source.position(), target_pos)).abs() <= 1e-3);
    }

    #[test]
    fn orbiting_target_candidate_falls_back_to_scan_when_target_is_faster() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 52.0, 50.0, 1.0, 500),
            planet(1, -1, 70.0, 50.0, 1.0, 100),
        ]);
        state.angular_velocity = 1.0;
        let source = state.planets.get(0).expect("source");
        let target = state.planets.get(1).expect("target");
        let speed = fleet_speed(1, state.config.ship_speed);

        assert!(
            speed
                <= state.angular_velocity.abs()
                    * distance(target.position(), Point::new(CENTER, CENTER))
        );
        assert!(cached_orbiting_target_candidate(&state, source, target, speed).is_some());
    }

    #[test]
    fn orbiting_target_candidate_matches_dense_solver_and_hits_in_simulator() {
        for (target_x, target_y, angular_velocity, ships) in [
            (50.0, 20.0, 0.025, 300),
            (75.0, 50.0, 0.025, 80),
            (50.0, 80.0, -0.025, 200),
        ] {
            let mut state = state_from_planets(vec![
                planet(0, 0, 97.0, 50.0, 3.0, 500),
                planet(1, -1, target_x, target_y, 3.0, 100),
            ]);
            state.angular_velocity = angular_velocity;
            let source = state.planets.get(0).expect("source");
            let target = state.planets.get(1).expect("target");
            let speed = fleet_speed(ships, state.config.ship_speed);

            let fast = cached_orbiting_target_candidate(&state, source, target, speed)
                .expect("fast solver should find an intercept");
            let dense = dense_orbiting_target_candidate(&state, source, target, speed)
                .expect("dense solver should find an intercept");

            assert!(
                (fast.time - dense.time).abs() <= 1e-3,
                "fast time {} did not match dense time {} for target ({target_x}, {target_y})",
                fast.time,
                dense.time,
            );
            assert!(
                (fast.angle - dense.angle).abs() <= 1e-3,
                "fast angle {} did not match dense angle {} for target ({target_x}, {target_y})",
                fast.angle,
                dense.angle,
            );

            let mut sim_state = state.clone();
            let entities = action_entity_slots(&sim_state);
            let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
            let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
            let mut launched_ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
            launch[0] = true;
            targets[0] = 1;
            launched_ships[0] = i64::from(ships);
            let decoded = decode_discrete_target_actions(
                &sim_state,
                &PlayerMap::identity(),
                &entities,
                &launch,
                &targets,
                &launched_ships,
                1,
                1,
                TargetingMode::FullMask,
            )
            .expect("valid orbiting target action should decode");

            step(&mut sim_state, &decoded.actions);
            assert!(
                run_until_planet_changes(&mut sim_state, 1, 100),
                "decoded action should hit target ({target_x}, {target_y})",
            );
        }
    }

    #[test]
    fn orbiting_target_candidate_uses_observed_generated_reset_phase() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 95.0, 67.0, 3.0, 500),
            planet(1, -1, 50.0, 20.0, 1.0, 100),
        ]);
        state.step = 1;
        state.angular_velocity = 0.05;
        let source = state.planets.get(0).expect("source");
        let target = state.planets.get(1).expect("target");
        let speed = fleet_speed(40, state.config.ship_speed);

        let fast = cached_orbiting_target_candidate(&state, source, target, speed)
            .expect("fast solver should find an intercept");
        let dense = dense_orbiting_target_candidate(&state, source, target, speed)
            .expect("dense solver should find an intercept");

        assert!(
            (fast.time - dense.time).abs() <= 1e-3,
            "fast time {} did not match dense time {}",
            fast.time,
            dense.time,
        );
        assert!(
            (fast.angle - dense.angle).abs() <= 1e-3,
            "fast angle {} did not match dense angle {}",
            fast.angle,
            dense.angle,
        );
    }

    #[test]
    fn piecewise_linear_comet_candidate_solves_segment_intercept() {
        let source = planet(0, 0, 10.0, 80.0, 2.0, 500);
        let target_radius = 1.0;
        let speed = fleet_speed(100, SimConfig::new(4).ship_speed);
        let path = (0..30)
            .map(|index| Point::new(35.0 + f64::from(index) * 2.0, 80.0))
            .collect::<Vec<_>>();

        let candidate = piecewise_linear_target_candidates(&source, target_radius, speed, &path)
            .into_iter()
            .next()
            .expect("linear comet path should have an intercept");
        let segment = candidate.time.floor() as usize;
        let fraction = candidate.time - segment as f64;
        let target_pos = lerp(path[segment], path[segment + 1], fraction);
        let clearance = source.radius + 0.1 + target_radius;
        let residual = distance(source.position(), target_pos) - clearance - speed * candidate.time;

        assert!(residual.abs() <= 1e-5, "residual {residual}");
        assert!((candidate.angle - angle_between(source.position(), target_pos)).abs() <= 1e-12);
    }

    #[test]
    fn piecewise_linear_comet_candidates_hit_swept_segments() {
        let sources = [
            planet(0, 0, 10.0, 80.0, 2.0, 500),
            planet(1, 0, 90.0, 20.0, 3.0, 500),
            planet(2, 0, 50.0, 50.0, 1.5, 500),
        ];
        let speeds = [
            fleet_speed(6, 6.0),
            fleet_speed(40, 6.0),
            fleet_speed(500, 6.0),
        ];
        let paths = [
            (0..35)
                .map(|index| Point::new(35.0 + f64::from(index) * 1.7, 80.0))
                .collect::<Vec<_>>(),
            (0..35)
                .map(|index| {
                    Point::new(80.0 - f64::from(index) * 1.5, 15.0 + f64::from(index) * 1.3)
                })
                .collect::<Vec<_>>(),
            (0..35)
                .map(|index| {
                    let time = f64::from(index);
                    Point::new(20.0 + time * 2.0, 70.0 - time * 0.8)
                })
                .collect::<Vec<_>>(),
        ];

        let mut checked = 0;
        for source in sources {
            for speed in speeds {
                for path in &paths {
                    for candidate in piecewise_linear_target_candidates(&source, 1.25, speed, path)
                    {
                        assert_candidate_hits_swept_segment(&source, 1.25, path, speed, candidate);
                        checked += 1;
                    }
                }
            }
        }

        assert!(
            checked > 10,
            "test cases should produce meaningful comet candidates"
        );
    }

    #[test]
    fn piecewise_linear_orbiting_candidates_hit_swept_segments() {
        let target_positions = [
            Point::new(50.0, 20.0),
            Point::new(78.0, 50.0),
            Point::new(35.0, 76.0),
        ];
        let sources = [
            planet(0, 0, 12.0, 50.0, 2.0, 500),
            planet(1, 0, 88.0, 18.0, 3.0, 500),
            planet(2, 0, 45.0, 88.0, 1.5, 500),
        ];
        let speeds = [
            fleet_speed(6, 6.0),
            fleet_speed(40, 6.0),
            fleet_speed(500, 6.0),
        ];

        let mut checked = 0;
        for (target_index, target_pos) in target_positions.into_iter().enumerate() {
            let target = planet(
                10 + target_index as u32,
                -1,
                target_pos.x,
                target_pos.y,
                1.25,
                100,
            );
            let orbit_path = test_orbit_path(&target, 0.035, 120);
            let mut state = state_from_planets(vec![target.clone()]);
            state.step = 1;
            state.angular_velocity = 0.035;
            state.orbit_paths = vec![orbit_path];

            for source in &sources {
                for speed in speeds {
                    let (min_time, max_time) = orbit_time_bounds(source, &target, &target, speed);
                    let path = &state.orbit_paths[0].points;
                    for candidate in piecewise_linear_target_candidates_in_time_range(
                        source,
                        target.radius,
                        speed,
                        path,
                        min_time,
                        max_time,
                    ) {
                        assert_candidate_hits_swept_segment(
                            source,
                            target.radius,
                            path,
                            speed,
                            candidate,
                        );
                        checked += 1;
                    }
                }
            }
        }

        assert!(
            checked > 10,
            "test cases should produce meaningful orbiting candidates"
        );
    }

    #[test]
    fn cached_orbit_path_matches_step_zero_swept_phase() {
        let source = planet(0, 0, 12.0, 50.0, 2.0, 500);
        let target = planet(1, -1, 50.0, 20.0, 1.25, 100);
        let mut state = state_from_planets(vec![source.clone(), target.clone()]);
        state.step = 0;
        state.angular_velocity = 0.035;
        state.orbit_paths = vec![test_orbit_path(&target, state.angular_velocity, 120)];
        let speed = fleet_speed(500, state.config.ship_speed);

        let candidate = cached_orbiting_target_candidate(&state, &source, &target, speed)
            .expect("cached step-zero orbit path should have an intercept");
        let mut simulator_path = Vec::with_capacity(state.orbit_paths[0].points.len() + 1);
        simulator_path.push(target.position());
        simulator_path.push(target.position());
        simulator_path.extend_from_slice(&state.orbit_paths[0].points[1..]);

        assert_candidate_hits_swept_segment(
            &source,
            target.radius,
            &simulator_path,
            speed,
            candidate,
        );
    }

    #[test]
    fn analytic_orbiting_candidates_are_compared_to_swept_segments() {
        let sources = [
            planet(0, 0, 12.0, 50.0, 2.0, 500),
            planet(1, 0, 88.0, 18.0, 3.0, 500),
            planet(2, 0, 45.0, 88.0, 1.5, 500),
            planet(3, 0, 20.0, 20.0, 2.5, 500),
            planet(4, 0, 80.0, 80.0, 2.0, 500),
        ];
        let speeds = [
            fleet_speed(6, 6.0),
            fleet_speed(40, 6.0),
            fleet_speed(500, 6.0),
        ];
        let angular_velocities = [0.025, 0.035, 0.05];
        let target_angles: [f64; 9] = [0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6];

        let mut analytic_checked = 0;
        let mut analytic_hits = 0;
        let mut piecewise_checked = 0;
        let mut piecewise_hits = 0;

        for angular_velocity in angular_velocities {
            for target_angle in target_angles {
                let target = planet(
                    10,
                    -1,
                    CENTER + target_angle.cos() * 30.0,
                    CENTER + target_angle.sin() * 30.0,
                    1.25,
                    100,
                );
                let orbit_path = test_orbit_path(&target, angular_velocity, 120);
                let mut state = state_from_planets(vec![target.clone()]);
                state.step = 1;
                state.angular_velocity = angular_velocity;
                state.orbit_paths = vec![orbit_path];
                let path = &state.orbit_paths[0].points;

                for source in &sources {
                    for speed in speeds {
                        if let Some(candidate) =
                            dense_orbiting_target_candidate(&state, source, &target, speed)
                        {
                            analytic_checked += 1;
                            if candidate_hits_swept_segment(
                                source,
                                target.radius,
                                path,
                                speed,
                                candidate,
                            ) {
                                analytic_hits += 1;
                            }
                        }

                        let (min_time, max_time) =
                            orbit_time_bounds(source, &target, &target, speed);
                        if let Some(candidate) = piecewise_linear_target_candidates_in_time_range(
                            source,
                            target.radius,
                            speed,
                            path,
                            min_time,
                            max_time,
                        )
                        .into_iter()
                        .next()
                        {
                            piecewise_checked += 1;
                            if candidate_hits_swept_segment(
                                source,
                                target.radius,
                                path,
                                speed,
                                candidate,
                            ) {
                                piecewise_hits += 1;
                            }
                        }
                    }
                }
            }
        }

        println!(
            "analytic swept hits: {analytic_hits}/{analytic_checked}; piecewise swept hits: {piecewise_hits}/{piecewise_checked}",
        );
        assert!(analytic_checked > 100);
        assert!(piecewise_checked > 100);
        assert_eq!(analytic_hits, analytic_checked);
        assert_eq!(piecewise_hits, piecewise_checked);
    }

    #[test]
    fn discrete_comet_target_hits_in_simulator() {
        let mut path = Vec::new();
        for index in 0..30 {
            path.push(Point::new(35.0 + f64::from(index) * 2.0, 80.0));
        }
        let mut state = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 500),
            planet(10, -1, 35.0, 80.0, 1.0, 20),
        ]);
        state.comet_planet_ids = vec![10];
        state.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![path],
            path_index: 0,
        }];
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = MAX_PLANETS as i64;
        ships[0] = 100;
        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("valid comet target should decode");

        step(&mut state, &decoded.actions);

        assert!(run_until_planet_changes(&mut state, 10, 20));
    }

    #[test]
    fn discrete_comet_target_without_intercept_is_no_op() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 500),
            planet(10, -1, 90.0, 80.0, 1.0, 20),
        ]);
        state.comet_planet_ids = vec![10];
        state.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(90.0, 80.0), Point::new(94.0, 80.0)]],
            path_index: 0,
        }];
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = MAX_PLANETS as i64;
        ships[0] = 1;

        for targeting_mode in [
            TargetingMode::AnythingGoes,
            TargetingMode::StopBadLaunch,
            TargetingMode::FullMask,
        ] {
            let decoded = decode_discrete_target_actions(
                &state,
                &PlayerMap::identity(),
                &entities,
                &launch,
                &targets,
                &ships,
                1,
                1,
                targeting_mode,
            )
            .expect("unreachable comet target should decode as a no-op");

            assert_eq!(decoded.launch_failures, 1);
            assert!(decoded.actions.iter().all(Vec::is_empty));
        }
    }

    #[test]
    fn discrete_comet_target_with_exhausted_path_is_no_op() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 500),
            planet(10, -1, 70.0, 80.0, 1.0, 20),
        ]);
        state.comet_planet_ids = vec![10];
        state.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(70.0, 80.0), Point::new(70.0, 80.0)]],
            path_index: 2,
        }];
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut targets = vec![0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        targets[0] = MAX_PLANETS as i64;
        ships[0] = 100;

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
            TargetingMode::FullMask,
        )
        .expect("exhausted comet path should decode as a no-op");

        assert_eq!(decoded.launch_failures, 1);
        assert!(decoded.actions.iter().all(Vec::is_empty));
    }

    #[test]
    fn pure_launch_errors_when_ship_count_is_zero() {
        let player_map = PlayerMap::identity();
        let state = one_planet_state();
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;

        let err = decode_pure_actions(
            &state,
            &player_map,
            &entities,
            &launch,
            &angle,
            &ships,
            1,
            1,
        )
        .expect_err("zero ships should fail");

        assert!(err.contains("ships must be >= 1"));
    }

    #[test]
    fn pure_launch_errors_when_ship_count_is_below_min_fleet_size() {
        let player_map = PlayerMap::identity();
        let state = one_planet_state();
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        ships[0] = 2;

        let err = decode_pure_actions(
            &state,
            &player_map,
            &entities,
            &launch,
            &angle,
            &ships,
            1,
            3,
        )
        .expect_err("undersized fleet should fail");

        assert!(err.contains("ships must be >= 3"));
    }

    #[test]
    fn pure_launch_errors_when_ship_count_exceeds_i32() {
        let player_map = PlayerMap::identity();
        let state = one_planet_state();
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        ships[0] = i64::from(i32::MAX) + 1;

        let err = decode_pure_actions(
            &state,
            &player_map,
            &entities,
            &launch,
            &angle,
            &ships,
            1,
            1,
        )
        .expect_err("oversized ships should fail");

        assert!(err.contains("ships must fit in i32"));
    }

    #[test]
    fn pure_launch_errors_when_angle_is_not_finite() {
        let player_map = PlayerMap::identity();
        let state = one_planet_state();
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        angle[0] = f32::INFINITY;
        ships[0] = 1;

        let err = decode_pure_actions(
            &state,
            &player_map,
            &entities,
            &launch,
            &angle,
            &ships,
            1,
            1,
        )
        .expect_err("non-finite angle should fail");

        assert!(err.contains("angle must be finite"));
    }

    #[test]
    fn pure_launch_errors_when_player_does_not_own_source() {
        let player_map = PlayerMap::identity();
        let state = one_planet_state();
        let entities = action_entity_slots(&state);
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[ACTION_ENTITY_SLOTS] = true;
        ships[ACTION_ENTITY_SLOTS] = 1;

        let err = decode_pure_actions(
            &state,
            &player_map,
            &entities,
            &launch,
            &angle,
            &ships,
            1,
            1,
        )
        .expect_err("wrong owner should fail");

        assert!(err.contains("player 1 cannot launch from planet 7 owned by 0"));
    }

    #[test]
    fn pure_launch_errors_when_total_launches_exceed_source_ships() {
        let player_map = PlayerMap::identity();
        let state = one_planet_state();
        let entities = action_entity_slots(&state);
        let max_per_planet_launches = 2;
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        launch[0] = true;
        ships[0] = 6;
        launch[1] = true;
        ships[1] = 5;

        let err = decode_pure_actions(
            &state,
            &player_map,
            &entities,
            &launch,
            &angle,
            &ships,
            max_per_planet_launches,
            1,
        )
        .expect_err("overspending should fail");

        assert!(err.contains("planet 7 has 10 ships, cannot launch 11"));
    }

    #[test]
    fn pure_launch_emits_multiple_actions_until_first_false_slot() {
        let player_map = PlayerMap::identity();
        let state = one_planet_state();
        let entities = action_entity_slots(&state);
        let max_per_planet_launches = 3;
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        launch[0] = true;
        ships[0] = 2;
        launch[1] = true;
        ships[1] = 3;
        launch[2] = false;
        ships[2] = 4;

        let actions = decode_pure_actions(
            &state,
            &player_map,
            &entities,
            &launch,
            &angle,
            &ships,
            max_per_planet_launches,
            1,
        )
        .expect("valid actions should decode");

        assert_eq!(actions[0].len(), 2);
        assert_eq!(actions[0][0].ships, 2);
        assert_eq!(actions[0][1].ships, 3);
    }

    #[test]
    fn pure_launch_decodes_from_remapped_outer_player_slot() {
        let player_map = PlayerMap::from_outer_slots(2, [3, 1, 0, 2]);
        let state = one_planet_state();
        let entities = action_entity_slots(&state);
        let max_per_planet_launches = 1;
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let outer_player = 3;
        let action_index = outer_player * ACTION_ENTITY_SLOTS * max_per_planet_launches;
        launch[action_index] = true;
        ships[action_index] = 4;

        let actions = decode_pure_actions(
            &state,
            &player_map,
            &entities,
            &launch,
            &angle,
            &ships,
            max_per_planet_launches,
            1,
        )
        .expect("valid remapped outer slot should decode");

        assert_eq!(actions[0].len(), 1);
        assert_eq!(actions[0][0].ships, 4);
        assert!(actions[1].is_empty());
    }

    #[test]
    fn pure_comet_action_uses_reserved_comet_slots_after_planets() {
        let mut state = one_planet_state();
        state.planets.push(Planet {
            id: 8,
            owner: 1,
            x: 20.0,
            y: 20.0,
            radius: 2.0,
            ships: 5,
            production: 1,
        });
        state.comets.push(CometGroup {
            planet_ids: vec![8],
            paths: Vec::new(),
            path_index: 0,
        });
        state.comet_planet_ids.push(8);

        let slots = action_entity_slots(&state);

        assert_eq!(slots[0].map(|slot| slot.planet_id), Some(7));
        assert!(slots[1..MAX_PLANETS].iter().all(Option::is_none));
        assert_eq!(slots[MAX_PLANETS].map(|slot| slot.planet_id), Some(8));
    }

    #[test]
    fn pure_launch_decodes_against_cached_slot_mapping() {
        let player_map = PlayerMap::identity();
        let mut observed_state = one_planet_state();
        observed_state.planets.push(Planet {
            id: 8,
            owner: 0,
            x: 20.0,
            y: 20.0,
            radius: 2.0,
            ships: 5,
            production: 1,
        });
        let entities = action_entity_slots(&observed_state);
        let current_state = observed_state.clone();

        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        ships[0] = 1;

        let actions = decode_pure_actions(
            &current_state,
            &player_map,
            &entities,
            &launch,
            &angle,
            &ships,
            1,
            1,
        )
        .expect("cached slot should decode by observed planet id");

        assert_eq!(actions[0][0].from_planet_id, 7);
    }

    #[test]
    fn pure_launch_errors_when_cached_slot_planet_is_missing() {
        let player_map = PlayerMap::identity();
        let observed_state = one_planet_state();
        let entities = action_entity_slots(&observed_state);
        let mut current_state = observed_state.clone();
        current_state.planets.clear();

        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        ships[0] = 1;

        let err = decode_pure_actions(
            &current_state,
            &player_map,
            &entities,
            &launch,
            &angle,
            &ships,
            1,
            1,
        )
        .expect_err("missing cached slot planet should fail");

        assert!(err.contains("stale action entity slot"));
    }
}
