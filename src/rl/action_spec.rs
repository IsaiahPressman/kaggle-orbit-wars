use std::collections::HashSet;

use crate::rules_engine::env::PlayerAction;
use crate::rules_engine::state::{LaunchAction, Planet, Point, State, CENTER, SUN_RADIUS};
use crate::rules_engine::utils::{
    distance, fleet_speed, is_orbiting, orbit_position, point_to_segment_distance,
};

use super::{PlayerMap, ACTION_ENTITY_SLOTS, MAX_COMETS, MAX_PLANETS, OUTER_PLAYER_SLOTS};

const TARGET_EPS: f64 = 1e-6;
const ROOT_EPS: f64 = 1e-7;
const ROOT_STEP: f64 = 0.25;
const ORBIT_TARGET_HORIZON: f64 = 200.0;

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
    planet_id: u32,
}

pub(super) type ActionEntitySlots = [Option<ActionEntitySlot>; ACTION_ENTITY_SLOTS];

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
                spent_ships += ship_count;
                if spent_ships > i64::from(source.ships) {
                    return Err(format!(
                        "planet {} has {} ships, cannot launch {spent_ships}",
                        source.id, source.ships
                    ));
                }
                let angle = target_angle(state, source, target_planet, ship_count as i32);
                player_actions.push(LaunchAction {
                    from_planet_id: source.id,
                    angle,
                    ships: ship_count as i32,
                });
            }
        }
    }
    Ok(actions)
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

fn target_angle(state: &State, source: &Planet, target: &Planet, ships: i32) -> f64 {
    let speed = fleet_speed(ships, state.config.ship_speed);
    let candidates = if state.comet_planet_ids.contains(&target.id) {
        comet_target_candidates(state, source, target, speed)
    } else if is_orbiting(
        state
            .initial_planets
            .get(target.id)
            .map_or(target.position(), Planet::position),
        target.radius,
    ) {
        orbiting_target_candidates(state, source, target, speed)
    } else {
        static_target_candidates(source, target, speed)
    };
    choose_candidate(state, source, target, candidates).map_or_else(
        || angle_between(source.position(), target.position()),
        |candidate| candidate.angle,
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

fn orbiting_target_candidates(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
) -> Vec<TargetCandidate> {
    let Some(initial_target) = state.initial_planets.get(target.id) else {
        return static_target_candidates(source, target, speed);
    };
    let target_at = |time: f64| {
        if time == 0.0 {
            target.position()
        } else {
            orbit_position(
                initial_target.position(),
                state.angular_velocity,
                state.step as f64 + time,
            )
        }
    };
    moving_target_candidates(
        source,
        target.radius,
        speed,
        ORBIT_TARGET_HORIZON,
        target_at,
    )
}

fn comet_target_candidates(
    state: &State,
    source: &Planet,
    target: &Planet,
    speed: f64,
) -> Vec<TargetCandidate> {
    let Some((group, path_offset)) = state.comets.iter().find_map(|group| {
        group
            .planet_ids
            .iter()
            .position(|planet_id| *planet_id == target.id)
            .map(|path_offset| (group, path_offset))
    }) else {
        return static_target_candidates(source, target, speed);
    };
    let Some(path) = group.paths.get(path_offset) else {
        return static_target_candidates(source, target, speed);
    };
    let path_start = group.path_index.max(0) as usize;
    if path_start >= path.len() {
        return static_target_candidates(source, target, speed);
    }
    let remaining = &path[path_start..];
    let horizon = remaining.len().saturating_sub(1) as f64;
    let target_at = |time: f64| {
        let lower = time.floor() as usize;
        if lower >= remaining.len() - 1 {
            return *remaining.last().expect("remaining path is nonempty");
        }
        let fraction = time - lower as f64;
        lerp(remaining[lower], remaining[lower + 1], fraction)
    };
    let mut candidates =
        moving_target_candidates(source, target.radius, speed, horizon.max(0.0), target_at);
    if candidates.is_empty() {
        if let Some(best) = remaining
            .iter()
            .enumerate()
            .min_by(|(left_index, left), (right_index, right)| {
                let left_time = *left_index as f64;
                let right_time = *right_index as f64;
                let left_angle = angle_between(source.position(), **left);
                let right_angle = angle_between(source.position(), **right);
                let left_gap =
                    (distance(launch_start(source, left_angle), **left) - speed * left_time).abs();
                let right_gap = (distance(launch_start(source, right_angle), **right)
                    - speed * right_time)
                    .abs();
                left_gap.total_cmp(&right_gap)
            })
            .map(|(index, point)| (*point, index as f64))
        {
            let angle = angle_between(source.position(), best.0);
            let start = launch_start(source, angle);
            candidates.push(TargetCandidate {
                angle,
                end: point_along(start, angle, speed * best.1),
                time: best.1,
            });
        }
    }
    candidates
}

fn moving_target_candidates(
    source: &Planet,
    radius: f64,
    speed: f64,
    horizon: f64,
    target_at: impl Fn(f64) -> Point,
) -> Vec<TargetCandidate> {
    if horizon <= 0.0 {
        return moving_candidates_at_time(source, target_at(0.0), radius, speed, 0.0);
    }
    let mut candidates = Vec::new();

    for branch in [
        MovingBranch::Center,
        MovingBranch::LeftEdge,
        MovingBranch::RightEdge,
    ] {
        let roots = find_roots(
            |time| {
                moving_candidate_for_branch(source, target_at(time), radius, speed, time, branch)
                    .map_or(f64::INFINITY, |candidate| candidate.time - time)
            },
            0.0,
            horizon,
        );
        for time in roots {
            if let Some(candidate) =
                moving_candidate_for_branch(source, target_at(time), radius, speed, time, branch)
            {
                candidates.push(candidate);
            }
        }
    }
    candidates.sort_by(|left, right| left.time.total_cmp(&right.time));
    candidates
}

#[derive(Clone, Copy, Debug)]
enum MovingBranch {
    Center,
    LeftEdge,
    RightEdge,
}

fn moving_candidates_at_time(
    source: &Planet,
    target_pos: Point,
    radius: f64,
    speed: f64,
    time: f64,
) -> Vec<TargetCandidate> {
    [
        MovingBranch::Center,
        MovingBranch::LeftEdge,
        MovingBranch::RightEdge,
    ]
    .into_iter()
    .filter_map(|branch| {
        moving_candidate_for_branch(source, target_pos, radius, speed, time, branch)
    })
    .collect()
}

fn moving_candidate_for_branch(
    source: &Planet,
    target_pos: Point,
    radius: f64,
    speed: f64,
    _target_time: f64,
    branch: MovingBranch,
) -> Option<TargetCandidate> {
    let center_angle = angle_between(source.position(), target_pos);
    let angle = match branch {
        MovingBranch::Center => center_angle,
        MovingBranch::LeftEdge | MovingBranch::RightEdge => {
            let radius = (radius - TARGET_EPS).max(0.0);
            if radius == 0.0 {
                return None;
            }
            let source_distance = distance(source.position(), target_pos);
            if source_distance <= radius {
                return None;
            }
            let half_angle = (radius / source_distance).asin();
            match branch {
                MovingBranch::LeftEdge => center_angle + half_angle,
                MovingBranch::RightEdge => center_angle - half_angle,
                MovingBranch::Center => unreachable!(),
            }
        },
    };
    Some(candidate_for_angle(
        source, target_pos, radius, speed, angle,
    ))
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

fn find_roots(f: impl Fn(f64) -> f64, min_time: f64, max_time: f64) -> Vec<f64> {
    let mut roots = Vec::new();
    let mut left_time = min_time;
    let mut left_value = f(left_time);
    let mut right_time = (left_time + ROOT_STEP).min(max_time);
    while left_time < max_time {
        let right_value = f(right_time);
        if left_value.abs() <= ROOT_EPS {
            roots.push(left_time);
        } else if left_value.signum() != right_value.signum() {
            roots.push(bisect_root(&f, left_time, right_time));
        }
        left_time = right_time;
        left_value = right_value;
        right_time = (right_time + ROOT_STEP).min(max_time);
        if right_time == left_time {
            break;
        }
    }
    roots.dedup_by(|left, right| (*left - *right).abs() <= ROOT_STEP);
    roots
}

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
    use crate::rules_engine::state::{CometGroup, SimConfig};

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

        step(&mut state, &decoded);

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

        step(&mut state, &decoded);

        assert!(run_until_planet_changes(&mut state, 1, 100));
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

        step(&mut state, &decoded);

        assert!(run_until_planet_changes(&mut state, 10, 20));
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
