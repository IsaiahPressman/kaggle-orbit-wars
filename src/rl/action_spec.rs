use std::collections::HashSet;

use crate::rules_engine::env::PlayerAction;
use crate::rules_engine::state::{LaunchAction, Planet, Point, State, CENTER, SUN_RADIUS};
use crate::rules_engine::utils::{
    distance, fleet_speed, is_orbiting, orbit_position, point_to_segment_distance,
};

use super::{PlayerMap, ACTION_ENTITY_SLOTS, MAX_COMETS, MAX_PLANETS, OUTER_PLAYER_SLOTS};

const TARGET_EPS: f64 = 1e-6;
const ROOT_EPS: f64 = 1e-7;
const QUADRATIC_EPS: f64 = 1e-12;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RlActionSpec {
    Pure,
    DiscreteTargets,
}

impl RlActionSpec {
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value {
            "pure" => Some(Self::Pure),
            "discrete_targets" => Some(Self::DiscreteTargets),
            _ => None,
        }
    }

    pub(super) const fn can_act_len(self) -> usize {
        match self {
            Self::Pure => OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS,
            Self::DiscreteTargets => OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS * ACTION_ENTITY_SLOTS,
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
    pub(super) comet_launch_failures: u32,
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
                if ship_count > i64::from(i32::MAX) {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} ships must fit in i32"
                    ));
                }
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
                    ships: ship_count as i32,
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
) -> Result<DecodedDiscreteTargetActions, String> {
    let mut actions = vec![Vec::new(); state.config.player_count];
    let mut comet_launch_failures = 0_u32;
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
                if ship_count > i64::from(i32::MAX) {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} ships must fit in i32"
                    ));
                }
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
                    ship_count as i32,
                    &mut orbit_target_cache,
                ) else {
                    comet_launch_failures += 1;
                    continue;
                };
                spent_ships += ship_count;
                player_actions.push(LaunchAction {
                    from_planet_id: source.id,
                    angle,
                    ships: ship_count as i32,
                });
            }
        }
    }
    Ok(DecodedDiscreteTargetActions {
        actions,
        comet_launch_failures,
    })
}

pub(super) fn encode_action_spec(
    action_spec: RlActionSpec,
    state: &State,
    player_map: &PlayerMap,
    entities: &ActionEntitySlots,
    can_act: &mut [bool],
    max_launch: &mut [i64],
    min_fleet_size: i64,
) {
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
        match action_spec {
            RlActionSpec::Pure => {
                let index =
                    player_map.internal_to_outer(player) * ACTION_ENTITY_SLOTS + entity_index;
                can_act[index] = true;
            },
            RlActionSpec::DiscreteTargets => {
                let base = (player_map.internal_to_outer(player) * ACTION_ENTITY_SLOTS
                    + entity_index)
                    * ACTION_ENTITY_SLOTS;
                for (target_index, target_slot) in entities.iter().enumerate() {
                    can_act[base + target_index] =
                        target_slot.is_some() && target_index != entity_index;
                }
            },
        }
        let index = player_map.internal_to_outer(player) * ACTION_ENTITY_SLOTS + entity_index;
        max_launch[index] = i64::from(planet.ships);
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

struct OrbitTargetCache<'a> {
    state: &'a State,
    entities: &'a ActionEntitySlots,
    min_fleet_size: i32,
    sources: Option<Vec<&'a Planet>>,
    min_speed: f64,
    paths: Vec<(u32, Vec<Point>)>,
}

impl<'a> OrbitTargetCache<'a> {
    fn new(state: &'a State, entities: &'a ActionEntitySlots, min_fleet_size: i64) -> Self {
        let min_fleet_size = min_fleet_size.clamp(1, i64::from(i32::MAX)) as i32;
        Self {
            state,
            entities,
            min_fleet_size,
            sources: None,
            min_speed: fleet_speed(min_fleet_size, state.config.ship_speed),
            paths: Vec::new(),
        }
    }

    fn path_for(&mut self, target: &Planet) -> Option<&[Point]> {
        if let Some(index) = self.paths.iter().position(|(id, _)| *id == target.id) {
            return Some(self.paths[index].1.as_slice());
        }

        if let Some(path) = self
            .state
            .orbit_paths
            .iter()
            .find(|path| path.planet_id == target.id)
        {
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
}

fn target_angle(
    state: &State,
    source: &Planet,
    target: &Planet,
    ships: i32,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
) -> Option<f64> {
    let speed = fleet_speed(ships, state.config.ship_speed);
    if state.comet_planet_ids.contains(&target.id) {
        let candidates = comet_target_candidates(state, source, target, speed)?;
        return choose_candidate(state, source, target, candidates)
            .map(|candidate| candidate.angle);
    }

    if is_orbiting(
        state
            .initial_planets
            .get(target.id)
            .map_or(target.position(), Planet::position),
        target.radius,
    ) {
        return Some(
            orbiting_target_candidate(state, source, target, speed, orbit_target_cache)
                .map_or_else(
                    || angle_between(source.position(), target.position()),
                    |candidate| candidate.angle,
                ),
        );
    }

    let candidates = static_target_candidates(source, target, speed);
    Some(
        choose_candidate(state, source, target, candidates).map_or_else(
            || angle_between(source.position(), target.position()),
            |candidate| candidate.angle,
        ),
    )
}

fn static_target_candidates(source: &Planet, target: &Planet, speed: f64) -> Vec<TargetCandidate> {
    let source_pos = source.position();
    let target_pos = target.position();
    let base_angle = angle_between(source_pos, target_pos);
    let mut candidates = vec![candidate_for_angle(
        source,
        target_pos,
        target.radius,
        speed,
        base_angle,
    )];
    let distance_to_target = distance(source_pos, target_pos);
    let radius = (target.radius - TARGET_EPS).max(0.0);
    if distance_to_target > radius && radius > 0.0 {
        let half_angle = (radius / distance_to_target).asin();
        candidates.push(candidate_for_angle(
            source,
            target_pos,
            target.radius,
            speed,
            base_angle + half_angle,
        ));
        candidates.push(candidate_for_angle(
            source,
            target_pos,
            target.radius,
            speed,
            base_angle - half_angle,
        ));
    }
    candidates
}

fn orbiting_target_candidate(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
    orbit_target_cache: &mut OrbitTargetCache<'_>,
) -> Option<TargetCandidate> {
    let initial_target = state.initial_planets.get(target.id)?;
    let path = orbit_target_cache.path_for(target)?;
    let (min_time, max_time) = orbit_time_bounds(source, initial_target, target, speed);
    piecewise_linear_target_candidates_in_time_range(
        source,
        target.radius,
        speed,
        path,
        min_time,
        max_time,
    )
    .into_iter()
    .next()
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

fn comet_target_candidates(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
) -> Option<Vec<TargetCandidate>> {
    let (group, path_offset) = state.comets.iter().find_map(|group| {
        group
            .planet_ids
            .iter()
            .position(|planet_id| *planet_id == target.id)
            .map(|path_offset| (group, path_offset))
    })?;
    let path = group.paths.get(path_offset)?;
    let path_start = group.path_index.max(0) as usize;
    if path_start >= path.len() {
        return None;
    }
    let remaining = &path[path_start..];
    let candidates = piecewise_linear_target_candidates(source, target.radius, speed, remaining);
    (!candidates.is_empty()).then_some(candidates)
}

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

fn choose_candidate(
    state: &State,
    source: &Planet,
    target: &Planet,
    candidates: Vec<TargetCandidate>,
) -> Option<TargetCandidate> {
    if candidates.is_empty() {
        return None;
    }
    candidates
        .iter()
        .copied()
        .find(|candidate| {
            !hits_sun(source, *candidate) && !hits_static_blocker(state, source, target, *candidate)
        })
        .or_else(|| {
            candidates
                .iter()
                .copied()
                .find(|candidate| !hits_sun(source, *candidate))
        })
        .or_else(|| candidates.first().copied())
}

fn hits_sun(source: &Planet, candidate: TargetCandidate) -> bool {
    point_to_segment_distance(
        Point::new(CENTER, CENTER),
        launch_start(source, candidate.angle),
        candidate.end,
    ) < SUN_RADIUS
}

fn hits_static_blocker(
    state: &State,
    source: &Planet,
    target: &Planet,
    candidate: TargetCandidate,
) -> bool {
    let comet_ids = state
        .comet_planet_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let start = launch_start(source, candidate.angle);
    state.planets.iter().any(|planet| {
        planet.id != source.id
            && planet.id != target.id
            && !comet_ids.contains(&planet.id)
            && !is_orbiting(
                state
                    .initial_planets
                    .get(planet.id)
                    .map_or(planet.position(), Planet::position),
                planet.radius,
            )
            && point_to_segment_distance(planet.position(), start, candidate.end) < planet.radius
    })
}

fn candidate_for_angle(
    source: &Planet,
    target_pos: Point,
    target_radius: f64,
    speed: f64,
    angle: f64,
) -> TargetCandidate {
    let start = launch_start(source, angle);
    let dir = Point::new(angle.cos(), angle.sin());
    let to_target = Point::new(target_pos.x - start.x, target_pos.y - start.y);
    let projection = to_target.x * dir.x + to_target.y * dir.y;
    let perpendicular_squared =
        (to_target.x * to_target.x + to_target.y * to_target.y) - projection * projection;
    let hit_distance = if perpendicular_squared < target_radius * target_radius {
        projection - (target_radius * target_radius - perpendicular_squared.max(0.0)).sqrt()
    } else {
        projection
    }
    .max(0.0);
    TargetCandidate {
        angle,
        end: point_along(start, angle, hit_distance),
        time: hit_distance / speed,
    }
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

fn launch_start(source: &Planet, angle: f64) -> Point {
    point_along(source.position(), angle, source.radius + 0.1)
}

fn point_along(start: Point, angle: f64, distance: f64) -> Point {
    Point::new(
        start.x + angle.cos() * distance,
        start.y + angle.sin() * distance,
    )
}

fn angle_between(start: Point, end: Point) -> f64 {
    (end.y - start.y).atan2(end.x - start.x)
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
    use crate::rules_engine::env::step;
    use crate::rules_engine::state::{CometGroup, OrbitPath, SimConfig};
    use crate::rules_engine::utils::swept_pair_hit;

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
        orbiting_target_candidate(state, source, target, speed, &mut cache)
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
        let mut can_act = vec![false; RlActionSpec::DiscreteTargets.can_act_len()];
        let mut max_launch = vec![0; OUTER_PLAYER_SLOTS * ACTION_ENTITY_SLOTS];

        encode_action_spec(
            RlActionSpec::DiscreteTargets,
            &state,
            &PlayerMap::identity(),
            &entities,
            &mut can_act,
            &mut max_launch,
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
        )
        .expect_err("empty target should fail");
        assert!(err.contains("cannot target empty action entity slot 2"));
    }

    #[test]
    fn discrete_static_target_tangent_fallback_hits_in_simulator() {
        let mut state = state_from_planets(vec![
            planet(0, 0, 10.0, 80.0, 2.0, 200),
            planet(1, -1, 40.0, 80.0, 1.0, 100),
            planet(2, -1, 70.0, 80.0, 8.0, 100),
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
        )
        .expect("valid discrete target should decode");

        step(&mut state, &decoded.actions);

        assert!(run_until_planet_changes(&mut state, 2, 100));
        assert_eq!(state.planets.get(1).expect("blocker").ships, 100);
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

        let decoded = decode_discrete_target_actions(
            &state,
            &PlayerMap::identity(),
            &entities,
            &launch,
            &targets,
            &ships,
            1,
            1,
        )
        .expect("unreachable comet target should decode as a no-op");

        assert_eq!(decoded.comet_launch_failures, 1);
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
