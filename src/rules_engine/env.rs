use std::collections::HashSet;

use super::generation::{
    assign_home_planets, generate_comet_paths, generate_planets, sample_comet_ships,
    spawn_comet_group, RandomSource,
};
use super::state::{
    CometSpawnInjection, Fleet, FleetLossStats, LaunchAction, Planet, PlanetVector, PlayerResult,
    Point, ResetConfig, State, StepInjections, StepResult, BOARD_SIZE, CENTER, COMET_SPAWN_STEPS,
    MAX_PLAYERS, SUN_RADIUS,
};
use super::utils::{
    fleet_speed, is_orbiting, orbit_position, point_to_segment_distance, swept_pair_hit,
};
pub type PlayerAction = Vec<LaunchAction>;

pub fn reset(config: ResetConfig) -> State {
    let mut rng = rand::rng();
    reset_with_rng(config, &mut rng)
}

pub fn reset_with_rng(config: ResetConfig, rng: &mut impl RandomSource) -> State {
    let generated = config.planets.is_none();
    let angular_velocity = config
        .angular_velocity
        .unwrap_or_else(|| rng.uniform(0.025, 0.05));
    let mut planets = config.planets.unwrap_or_else(|| generate_planets(rng));
    let initial_planets = config.initial_planets.unwrap_or_else(|| planets.clone());
    if generated {
        assign_home_planets(&mut planets, config.sim.player_count, rng);
    }

    State {
        config: config.sim,
        step: config.step.unwrap_or(if generated { 1 } else { 0 }),
        angular_velocity,
        planets: planets.into(),
        initial_planets: initial_planets.into(),
        fleets: Vec::new(),
        next_fleet_id: 0,
        comets: Vec::new(),
        comet_planet_ids: Vec::new(),
    }
}

pub fn step(state: &mut State, actions: &[PlayerAction]) -> StepResult {
    let mut rng = rand::rng();
    step_with_rng(state, actions, &mut rng)
}

pub fn step_with_rng(
    state: &mut State,
    actions: &[PlayerAction],
    rng: &mut impl RandomSource,
) -> StepResult {
    step_with_injections(state, actions, rng, StepInjections::default())
}

pub fn step_with_injections(
    state: &mut State,
    actions: &[PlayerAction],
    rng: &mut impl RandomSource,
    injections: StepInjections,
) -> StepResult {
    assert_eq!(
        actions.len(),
        state.config.player_count,
        "step requires actions for every player"
    );

    remove_expired_comets(state);
    spawn_comets(state, rng, injections.comet_spawn);
    process_launches(state, actions);
    produce_ships(state);
    let (planet_paths, expired_comet_planet_ids) = compute_planet_paths(state);
    let (combat_lists, fleet_losses) = move_fleets(state, &planet_paths);
    apply_planet_paths(state, &planet_paths);
    if !expired_comet_planet_ids.is_empty() {
        remove_comet_planets(state, |planet_id| {
            expired_comet_planet_ids.contains(&planet_id)
        });
    }
    let captures = resolve_combats(state, combat_lists);

    let player_results = player_results(state);
    state.step += 1;

    StepResult {
        player_results,
        fleet_losses,
        planets_captured: captures.planets_captured,
        comets_captured: captures.comets_captured,
        fleets_lost_in_combat: captures.fleets_lost_in_combat,
        ships_lost_in_combat: captures.ships_lost_in_combat,
    }
}

fn spawn_comets(
    state: &mut State,
    rng: &mut impl RandomSource,
    comet_spawn: Option<CometSpawnInjection>,
) {
    if !COMET_SPAWN_STEPS.contains(&(state.step + 1)) {
        return;
    }

    let (paths, ships) = match comet_spawn {
        Some(CometSpawnInjection::Spawn { paths, ships }) => (paths, ships),
        Some(CometSpawnInjection::Skip) => return,
        None => {
            let Some(paths) = generate_comet_paths(
                &state.initial_planets,
                state.angular_velocity,
                state.step + 1,
                &state.comet_planet_ids,
                state.config.comet_speed,
                rng,
            ) else {
                return;
            };

            let ships = sample_comet_ships(rng);
            (paths, ships)
        },
    };

    let group = spawn_comet_group(
        &mut state.planets,
        &mut state.initial_planets,
        &mut state.comet_planet_ids,
        paths,
        ships,
    );
    state.comets.push(group);
}

fn remove_expired_comets(state: &mut State) {
    let expired: HashSet<u32> = state
        .comets
        .iter()
        .flat_map(|group| {
            group
                .planet_ids
                .iter()
                .enumerate()
                .filter_map(|(index, planet_id)| {
                    let expired = group.path_index >= group.paths[index].len() as i32;
                    expired.then_some(*planet_id)
                })
        })
        .collect();

    if expired.is_empty() {
        return;
    }

    remove_comet_planets(state, |planet_id| expired.contains(&planet_id));
}

fn remove_comet_planets(state: &mut State, is_expired: impl Fn(u32) -> bool) {
    state.planets.retain(|planet| !is_expired(planet.id));
    state
        .initial_planets
        .retain(|planet| !is_expired(planet.id));
    state
        .comet_planet_ids
        .retain(|planet_id| !is_expired(*planet_id));

    for group in &mut state.comets {
        group.planet_ids.retain(|planet_id| !is_expired(*planet_id));
    }
    state.comets.retain(|group| !group.planet_ids.is_empty());
}

fn process_launches(state: &mut State, actions: &[PlayerAction]) {
    for (player_id, player_actions) in actions.iter().enumerate() {
        for action in player_actions {
            assert!(action.ships > 0, "launch ships must be positive");
            assert!(
                action.angle.is_finite(),
                "launch angle must be a finite f64"
            );

            let from_planet = state
                .planets
                .get_mut(action.from_planet_id)
                .unwrap_or_else(|| panic!("planet {} does not exist", action.from_planet_id));

            assert_eq!(
                from_planet.owner, player_id as i32,
                "player {player_id} cannot launch from planet {} owned by {}",
                from_planet.id, from_planet.owner
            );
            assert!(
                from_planet.ships >= action.ships,
                "planet {} has {} ships, cannot launch {}",
                from_planet.id,
                from_planet.ships,
                action.ships
            );

            from_planet.ships -= action.ships;
            let start_x = from_planet.x + action.angle.cos() * (from_planet.radius + 0.1);
            let start_y = from_planet.y + action.angle.sin() * (from_planet.radius + 0.1);
            state.fleets.push(Fleet {
                id: state.next_fleet_id,
                owner: player_id as i32,
                x: start_x,
                y: start_y,
                angle: action.angle,
                from_planet_id: action.from_planet_id,
                ships: action.ships,
            });
            state.next_fleet_id += 1;
        }
    }
}

fn produce_ships(state: &mut State) {
    for planet in state.planets.iter_mut() {
        if planet.owner != -1 {
            planet.ships += planet.production;
        }
    }
}

struct CombatLists {
    buckets: Vec<Option<Vec<Fleet>>>,
}

impl CombatLists {
    fn for_planets(planets: &PlanetVector) -> Self {
        let mut buckets = vec![None; planets.slot_len()];
        for planet in planets.iter() {
            buckets[planet.id as usize].get_or_insert_with(Vec::new);
        }

        Self { buckets }
    }

    fn push(&mut self, planet_id: u32, fleet: Fleet) {
        self.bucket_mut(planet_id).push(fleet);
    }

    fn bucket_mut(&mut self, planet_id: u32) -> &mut Vec<Fleet> {
        self.buckets
            .get_mut(planet_id as usize)
            .and_then(Option::as_mut)
            .expect("combat list exists for every planet")
    }

    fn into_buckets(self) -> impl Iterator<Item = (u32, Vec<Fleet>)> {
        self.buckets
            .into_iter()
            .enumerate()
            .filter_map(|(planet_id, bucket)| bucket.map(|fleets| (planet_id as u32, fleets)))
    }
}

#[derive(Clone, Copy, Debug)]
struct PlanetPath {
    old_pos: Point,
    new_pos: Point,
    check_collision: bool,
}

type PlanetPaths = Vec<Option<PlanetPath>>;

fn compute_planet_paths(state: &mut State) -> (PlanetPaths, Vec<u32>) {
    let comet_ids: HashSet<u32> = state.comet_planet_ids.iter().copied().collect();
    let mut planet_paths = vec![None; state.planets.slot_len()];
    let mut expired_comet_planet_ids = Vec::new();

    for planet in state.planets.iter() {
        if comet_ids.contains(&planet.id) {
            continue;
        }

        let old_pos = planet.position();
        let mut new_pos = old_pos;
        if let Some(initial_planet) = state.initial_planets.get(planet.id) {
            if is_orbiting(initial_planet.position(), planet.radius) {
                new_pos = orbit_position(
                    initial_planet.position(),
                    state.angular_velocity,
                    state.step.into(),
                );
            }
        }
        planet_paths[planet.id as usize] = Some(PlanetPath {
            old_pos,
            new_pos,
            check_collision: true,
        });
    }

    for group in &mut state.comets {
        group.path_index += 1;
        let path_index = group.path_index;

        for (path_offset, planet_id) in group.planet_ids.iter().enumerate() {
            let Some(planet) = state.planets.get(*planet_id) else {
                continue;
            };
            let old_pos = planet.position();
            let path = &group.paths[path_offset];
            let (new_pos, check_collision) = if path_index >= path.len() as i32 {
                expired_comet_planet_ids.push(*planet_id);
                (old_pos, true)
            } else {
                (path[path_index as usize], old_pos.x >= 0.0)
            };
            planet_paths[*planet_id as usize] = Some(PlanetPath {
                old_pos,
                new_pos,
                check_collision,
            });
        }
    }

    (planet_paths, expired_comet_planet_ids)
}

fn move_fleets(state: &mut State, planet_paths: &PlanetPaths) -> (CombatLists, FleetLossStats) {
    let mut combat_lists = CombatLists::for_planets(&state.planets);
    let mut fleets_to_remove = Vec::new();
    let mut losses = FleetLossStats::default();

    for fleet in &mut state.fleets {
        let old_pos = fleet.position();
        let speed = fleet_speed(fleet.ships, state.config.ship_speed);
        fleet.x += fleet.angle.cos() * speed;
        fleet.y += fleet.angle.sin() * speed;
        let new_pos = fleet.position();

        let mut hit_planet = false;
        for planet in state.planets.iter() {
            let Some(path) = planet_paths.get(planet.id as usize).copied().flatten() else {
                continue;
            };
            if path.check_collision && fleet_hits_planet_path(old_pos, new_pos, path, planet.radius)
            {
                combat_lists.push(planet.id, fleet.clone());
                fleets_to_remove.push(fleet.id);
                hit_planet = true;
                break;
            }
        }
        if hit_planet {
            continue;
        }

        if !(0.0..=BOARD_SIZE).contains(&fleet.x) || !(0.0..=BOARD_SIZE).contains(&fleet.y) {
            fleets_to_remove.push(fleet.id);
            losses.fleets_out_of_bounds += 1;
            losses.ships_out_of_bounds += fleet.ships;
            continue;
        }

        if point_to_segment_distance(Point::new(CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS {
            fleets_to_remove.push(fleet.id);
            losses.fleets_in_sun += 1;
            losses.ships_in_sun += fleet.ships;
            continue;
        }
    }

    state
        .fleets
        .retain(|fleet| !fleets_to_remove.contains(&fleet.id));
    (combat_lists, losses)
}

fn apply_planet_paths(state: &mut State, planet_paths: &PlanetPaths) {
    for planet in state.planets.iter_mut() {
        if let Some(path) = planet_paths.get(planet.id as usize).copied().flatten() {
            planet.x = path.new_pos.x;
            planet.y = path.new_pos.y;
        }
    }
}

fn fleet_hits_planet_path(
    fleet_start: Point,
    fleet_end: Point,
    planet_path: PlanetPath,
    radius: f64,
) -> bool {
    if !swept_aabb_overlaps(
        fleet_start,
        fleet_end,
        planet_path.old_pos,
        planet_path.new_pos,
        radius,
    ) {
        return false;
    }

    if planet_path.old_pos == planet_path.new_pos {
        return point_to_segment_distance(planet_path.old_pos, fleet_start, fleet_end) <= radius;
    }

    swept_pair_hit(
        fleet_start,
        fleet_end,
        planet_path.old_pos,
        planet_path.new_pos,
        radius,
    )
}

fn swept_aabb_overlaps(
    fleet_start: Point,
    fleet_end: Point,
    planet_start: Point,
    planet_end: Point,
    radius: f64,
) -> bool {
    let fleet_min_x = fleet_start.x.min(fleet_end.x);
    let fleet_max_x = fleet_start.x.max(fleet_end.x);
    let fleet_min_y = fleet_start.y.min(fleet_end.y);
    let fleet_max_y = fleet_start.y.max(fleet_end.y);
    let planet_min_x = planet_start.x.min(planet_end.x) - radius;
    let planet_max_x = planet_start.x.max(planet_end.x) + radius;
    let planet_min_y = planet_start.y.min(planet_end.y) - radius;
    let planet_max_y = planet_start.y.max(planet_end.y) + radius;

    !(fleet_max_x < planet_min_x
        || fleet_min_x > planet_max_x
        || fleet_max_y < planet_min_y
        || fleet_min_y > planet_max_y)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CaptureStats {
    planets_captured: u32,
    comets_captured: u32,
    fleets_lost_in_combat: u32,
    ships_lost_in_combat: i64,
}

fn resolve_combats(state: &mut State, combat_lists: CombatLists) -> CaptureStats {
    let mut captures = CaptureStats::default();
    let comet_ids = state
        .comet_planet_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    for (planet_id, planet_fleets) in combat_lists.into_buckets() {
        if planet_fleets.is_empty() {
            continue;
        }

        captures.fleets_lost_in_combat += planet_fleets.len() as u32;
        let Some(planet) = state.planets.get_mut(planet_id) else {
            continue;
        };

        let mut player_ships = [0_i32; MAX_PLAYERS];
        let mut player_present = [false; MAX_PLAYERS];
        for fleet in planet_fleets {
            let owner = valid_fleet_owner(&fleet, state.config.player_count);
            player_ships[owner] += fleet.ships;
            player_present[owner] = true;
        }
        let incoming_ships = player_ships
            .iter()
            .map(|ships| i64::from(*ships))
            .sum::<i64>();

        let mut top_player: Option<(i32, i32)> = None;
        let mut second_player: Option<(i32, i32)> = None;
        for (owner, ships) in player_ships.into_iter().enumerate() {
            if !player_present[owner] {
                continue;
            }

            let player = (owner as i32, ships);
            if match top_player {
                Some(top) => player.1 > top.1,
                None => true,
            } {
                second_player = top_player;
                top_player = Some(player);
            } else if match second_player {
                Some(second) => player.1 > second.1,
                None => true,
            } {
                second_player = Some(player);
            }
        }

        let top = top_player.expect("combat has at least one fleet");
        let (survivor_owner, survivor_ships) = if let Some(second) = second_player {
            if top.1 == second.1 {
                (-1, 0)
            } else {
                (top.0, top.1 - second.1)
            }
        } else {
            top
        };
        captures.ships_lost_in_combat += incoming_ships - i64::from(survivor_ships);

        if survivor_ships <= 0 {
            continue;
        }

        if planet.owner == survivor_owner {
            planet.ships += survivor_ships;
        } else {
            captures.ships_lost_in_combat += i64::from(planet.ships.min(survivor_ships)) * 2;
            planet.ships -= survivor_ships;
            if planet.ships < 0 {
                planet.owner = survivor_owner;
                planet.ships = planet.ships.abs();
                captures.planets_captured += 1;
                if comet_ids.contains(&planet_id) {
                    captures.comets_captured += 1;
                }
            }
        }
    }
    captures
}

fn player_results(state: &State) -> Vec<PlayerResult> {
    let player_count = state.config.player_count;
    let mut alive_players = [false; MAX_PLAYERS];
    for planet in state.planets.iter() {
        if let Some(owner) = valid_planet_owner(planet, player_count) {
            alive_players[owner] = true;
        }
    }
    for fleet in &state.fleets {
        alive_players[valid_fleet_owner(fleet, player_count)] = true;
    }

    let active_alive_players = &alive_players[..player_count];
    let remaining_alive_players = active_alive_players.iter().filter(|alive| **alive).count();
    let terminated = reached_step_limit(state) || remaining_alive_players <= 1;
    if !terminated {
        return active_alive_players
            .iter()
            .map(|alive| {
                if *alive {
                    PlayerResult::Active
                } else {
                    PlayerResult::Lost
                }
            })
            .collect();
    }

    let mut scores = [0_i32; MAX_PLAYERS];
    for planet in state.planets.iter() {
        if let Some(owner) = valid_planet_owner(planet, player_count) {
            scores[owner] += planet.ships;
        }
    }
    for fleet in &state.fleets {
        scores[valid_fleet_owner(fleet, player_count)] += fleet.ships;
    }

    let active_scores = &scores[..player_count];
    let max_score = active_scores.iter().copied().max().unwrap_or(0);
    active_scores
        .iter()
        .map(|score| {
            if *score == max_score && max_score > 0 {
                PlayerResult::Won
            } else {
                PlayerResult::Lost
            }
        })
        .collect()
}

fn reached_step_limit(state: &State) -> bool {
    state.step >= state.config.episode_steps.saturating_sub(2)
}

pub fn player_alive_flags(state: &State) -> Vec<bool> {
    let mut alive_players = vec![false; state.config.player_count];
    for planet in state.planets.iter() {
        if let Some(owner) = valid_planet_owner(planet, state.config.player_count) {
            alive_players[owner] = true;
        }
    }
    for fleet in &state.fleets {
        alive_players[valid_fleet_owner(fleet, state.config.player_count)] = true;
    }
    alive_players
}

fn valid_planet_owner(planet: &Planet, player_count: usize) -> Option<usize> {
    if planet.owner == -1 {
        return None;
    }
    if planet.owner >= 0 && (planet.owner as usize) < player_count {
        return Some(planet.owner as usize);
    }
    panic!(
        "planet {} owner out of bounds for {player_count} players: {}",
        planet.id, planet.owner
    );
}

fn valid_fleet_owner(fleet: &Fleet, player_count: usize) -> usize {
    assert!(
        fleet.owner != -1,
        "fleet {} owner cannot be neutral",
        fleet.id
    );
    if fleet.owner >= 0 && (fleet.owner as usize) < player_count {
        return fleet.owner as usize;
    }
    panic!(
        "fleet {} owner out of bounds for {player_count} players: {}",
        fleet.id, fleet.owner
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::generation::RandomSource;
    use crate::rules_engine::state::{CometGroup, Planet, SimConfig};

    struct RepeatingRandom {
        int: i32,
        float: f64,
    }

    impl RandomSource for RepeatingRandom {
        fn randint(&mut self, _low: i32, _high: i32) -> i32 {
            self.int
        }

        fn uniform(&mut self, low: f64, high: f64) -> f64 {
            low + (high - low) * self.float
        }
    }

    fn base_state(player_count: usize) -> State {
        reset(ResetConfig {
            sim: SimConfig::new(player_count),
            step: None,
            angular_velocity: Some(0.0),
            planets: Some(vec![
                Planet {
                    id: 0,
                    owner: 0,
                    x: 20.0,
                    y: 20.0,
                    radius: 2.0,
                    ships: 50,
                    production: 3,
                },
                Planet {
                    id: 1,
                    owner: -1,
                    x: 80.0,
                    y: 20.0,
                    radius: 2.0,
                    ships: 10,
                    production: 2,
                },
                Planet {
                    id: 2,
                    owner: 1,
                    x: 80.0,
                    y: 80.0,
                    radius: 2.0,
                    ships: 10,
                    production: 1,
                },
            ]),
            initial_planets: None,
        })
    }

    fn planet_with_id(id: u32) -> Planet {
        Planet {
            id,
            owner: -1,
            x: 0.0,
            y: 0.0,
            radius: 1.0,
            ships: 0,
            production: 0,
        }
    }

    #[test]
    #[should_panic(expected = "planet 0 owner out of bounds for 2 players: -2")]
    fn step_rejects_negative_non_neutral_planet_owner() {
        let mut state = base_state(2);
        state.planets[0].owner = -2;

        step(&mut state, &[vec![], vec![]]);
    }

    #[test]
    #[should_panic(expected = "planet 0 owner out of bounds for 2 players: 2")]
    fn step_rejects_planet_owner_outside_player_count() {
        let mut state = base_state(2);
        state.planets[0].owner = 2;

        step(&mut state, &[vec![], vec![]]);
    }

    #[test]
    #[should_panic(expected = "fleet 0 owner out of bounds for 2 players: 2")]
    fn step_rejects_fleet_owner_outside_player_count() {
        let mut state = base_state(2);
        state.fleets = vec![Fleet {
            id: 0,
            owner: 2,
            x: 40.0,
            y: 20.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 1,
        }];

        step(&mut state, &[vec![], vec![]]);
    }

    #[test]
    #[should_panic(expected = "fleet 0 owner cannot be neutral")]
    fn player_alive_flags_rejects_neutral_fleet_owner() {
        let mut state = base_state(2);
        state.fleets = vec![Fleet {
            id: 0,
            owner: -1,
            x: 40.0,
            y: 20.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 1,
        }];

        player_alive_flags(&state);
    }

    #[test]
    fn launch_spends_ships_and_creates_fleet_before_production() {
        let mut state = base_state(2);
        let result = step(
            &mut state,
            &[
                vec![LaunchAction {
                    from_planet_id: 0,
                    angle: 0.0,
                    ships: 20,
                }],
                vec![],
            ],
        );

        assert_eq!(
            result.player_results,
            vec![PlayerResult::Active, PlayerResult::Active]
        );
        assert_eq!(state.planets[0].ships, 33);
        assert_eq!(state.fleets.len(), 1);
        assert_eq!(state.fleets[0].owner, 0);
        assert_eq!(state.fleets[0].ships, 20);
        assert!((state.fleets[0].x - (22.1 + fleet_speed(20, 6.0))).abs() <= 1e-12);
        assert_eq!(state.next_fleet_id, 1);
        assert_eq!(state.step, 1);
    }

    #[test]
    fn nonterminal_eliminated_player_is_lost() {
        let mut state = reset(ResetConfig {
            sim: SimConfig::new(4),
            step: Some(0),
            angular_velocity: Some(0.0),
            planets: Some(vec![
                Planet {
                    id: 0,
                    owner: 0,
                    x: 20.0,
                    y: 20.0,
                    radius: 2.0,
                    ships: 10,
                    production: 1,
                },
                Planet {
                    id: 1,
                    owner: 1,
                    x: 80.0,
                    y: 20.0,
                    radius: 2.0,
                    ships: 10,
                    production: 1,
                },
                Planet {
                    id: 2,
                    owner: 2,
                    x: 20.0,
                    y: 80.0,
                    radius: 2.0,
                    ships: 10,
                    production: 1,
                },
            ]),
            initial_planets: None,
        });

        let result = step(&mut state, &[vec![], vec![], vec![], vec![]]);

        assert_eq!(
            result.player_results,
            vec![
                PlayerResult::Active,
                PlayerResult::Active,
                PlayerResult::Active,
                PlayerResult::Lost,
            ]
        );
    }

    #[test]
    fn nonterminal_player_results_do_not_compute_scores() {
        let state = reset(ResetConfig {
            sim: SimConfig::new(2),
            step: Some(0),
            angular_velocity: Some(0.0),
            planets: Some(vec![
                Planet {
                    id: 0,
                    owner: 0,
                    x: 20.0,
                    y: 20.0,
                    radius: 2.0,
                    ships: i32::MAX,
                    production: 0,
                },
                Planet {
                    id: 1,
                    owner: 0,
                    x: 25.0,
                    y: 20.0,
                    radius: 2.0,
                    ships: i32::MAX,
                    production: 0,
                },
                Planet {
                    id: 2,
                    owner: 1,
                    x: 80.0,
                    y: 80.0,
                    radius: 2.0,
                    ships: 1,
                    production: 0,
                },
            ]),
            initial_planets: None,
        });

        assert_eq!(
            player_results(&state),
            vec![PlayerResult::Active, PlayerResult::Active]
        );
    }

    #[test]
    #[should_panic(expected = "launch ships must be positive")]
    fn launch_rejects_non_positive_ship_count() {
        let mut state = base_state(2);
        step(
            &mut state,
            &[
                vec![LaunchAction {
                    from_planet_id: 0,
                    angle: 0.0,
                    ships: 0,
                }],
                vec![],
            ],
        );
    }

    #[test]
    fn fleet_hitting_planet_queues_combat_and_can_capture() {
        let mut state = base_state(2);
        state.planets[1].x = 25.0;
        state.planets[1].ships = 5;
        state.initial_planets = state.planets.clone();

        let result = step(
            &mut state,
            &[
                vec![LaunchAction {
                    from_planet_id: 0,
                    angle: 0.0,
                    ships: 20,
                }],
                vec![],
            ],
        );

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, 0);
        assert_eq!(state.planets[1].ships, 15);
        assert_eq!(result.planets_captured, 1);
        assert_eq!(result.comets_captured, 0);
        assert_eq!(result.fleets_lost_in_combat, 1);
        assert_eq!(result.ships_lost_in_combat, 10);
    }

    #[test]
    fn fleet_hitting_comet_planet_counts_comet_capture() {
        let mut state = base_state(2);
        state.planets[1].x = 25.0;
        state.planets[1].ships = 5;
        state.comet_planet_ids = vec![1];
        state.comets = vec![CometGroup {
            planet_ids: vec![1],
            paths: vec![vec![Point::new(25.0, 20.0)]],
            path_index: -1,
        }];

        let result = step(
            &mut state,
            &[
                vec![LaunchAction {
                    from_planet_id: 0,
                    angle: 0.0,
                    ships: 20,
                }],
                vec![],
            ],
        );

        assert_eq!(state.planets[1].owner, 0);
        assert_eq!(result.planets_captured, 1);
        assert_eq!(result.comets_captured, 1);
    }

    #[test]
    fn fleet_hits_planet_before_leaving_board() {
        let mut state = base_state(2);
        state.config.ship_speed = 10.0;
        state.planets[1].x = 99.0;
        state.planets[1].y = 20.0;
        state.planets[1].radius = 2.0;
        state.planets[1].ships = 5;
        state.fleets = vec![Fleet {
            id: 0,
            owner: 0,
            x: 95.0,
            y: 20.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 1000,
        }];

        step(&mut state, &[vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, 0);
        assert_eq!(state.planets[1].ships, 995);
    }

    #[test]
    fn fleet_hits_planet_before_crossing_sun() {
        let mut state = base_state(2);
        state.config.ship_speed = 20.0;
        state.planets[1].x = 39.0;
        state.planets[1].y = CENTER;
        state.planets[1].radius = 1.0;
        state.planets[1].ships = 5;
        state.initial_planets = state.planets.clone();
        state.fleets = vec![Fleet {
            id: 0,
            owner: 0,
            x: 35.0,
            y: CENTER,
            angle: 0.0,
            from_planet_id: 0,
            ships: 1000,
        }];

        step(&mut state, &[vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, 0);
        assert_eq!(state.planets[1].ships, 995);
    }

    #[test]
    fn fleet_leaving_board_without_planet_hit_is_removed() {
        let mut state = base_state(2);
        state.config.ship_speed = 10.0;
        state.fleets = vec![Fleet {
            id: 0,
            owner: 0,
            x: 95.0,
            y: 50.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 1000,
        }];

        let result = step(&mut state, &[vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, -1);
        assert_eq!(state.planets[1].ships, 10);
        assert_eq!(
            result.fleet_losses,
            FleetLossStats {
                fleets_out_of_bounds: 1,
                ships_out_of_bounds: 1000,
                ..FleetLossStats::default()
            }
        );
    }

    #[test]
    fn fleet_crossing_sun_without_planet_hit_is_removed() {
        let mut state = base_state(2);
        state.config.ship_speed = 20.0;
        state.fleets = vec![Fleet {
            id: 0,
            owner: 0,
            x: 35.0,
            y: CENTER,
            angle: 0.0,
            from_planet_id: 0,
            ships: 1000,
        }];

        let result = step(&mut state, &[vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, -1);
        assert_eq!(state.planets[1].ships, 10);
        assert_eq!(
            result.fleet_losses,
            FleetLossStats {
                fleets_in_sun: 1,
                ships_in_sun: 1000,
                ..FleetLossStats::default()
            }
        );
    }

    #[test]
    fn equal_attackers_destroy_each_other_without_touching_planet() {
        let mut state = base_state(2);
        state.fleets = vec![
            Fleet {
                id: 0,
                owner: 0,
                x: 27.0,
                y: 20.0,
                angle: 0.0,
                from_planet_id: 0,
                ships: 10,
            },
            Fleet {
                id: 1,
                owner: 1,
                x: 27.0,
                y: 20.0,
                angle: 0.0,
                from_planet_id: 1,
                ships: 10,
            },
        ];
        state.planets[1].x = 28.0;
        state.planets[1].ships = 7;

        let result = step(&mut state, &[vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, -1);
        assert_eq!(state.planets[1].ships, 7);
        assert_eq!(result.fleets_lost_in_combat, 2);
        assert_eq!(result.ships_lost_in_combat, 20);
    }

    #[test]
    fn four_player_top_attacker_tie_leaves_planet_unchanged() {
        let mut state = base_state(4);
        state.fleets = vec![
            Fleet {
                id: 0,
                owner: 0,
                x: 27.0,
                y: 20.0,
                angle: 0.0,
                from_planet_id: 0,
                ships: 5,
            },
            Fleet {
                id: 1,
                owner: 2,
                x: 27.0,
                y: 20.0,
                angle: 0.0,
                from_planet_id: 0,
                ships: 12,
            },
            Fleet {
                id: 2,
                owner: 3,
                x: 27.0,
                y: 20.0,
                angle: 0.0,
                from_planet_id: 0,
                ships: 12,
            },
        ];
        state.planets[1].x = 28.0;
        state.planets[1].ships = 7;

        let result = step(&mut state, &[vec![], vec![], vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, -1);
        assert_eq!(state.planets[1].ships, 7);
        assert_eq!(result.ships_lost_in_combat, 29);
    }

    #[test]
    fn four_player_unique_top_beats_tied_second_place() {
        let mut state = base_state(4);
        state.fleets = vec![
            Fleet {
                id: 0,
                owner: 0,
                x: 27.0,
                y: 20.0,
                angle: 0.0,
                from_planet_id: 0,
                ships: 5,
            },
            Fleet {
                id: 1,
                owner: 1,
                x: 27.0,
                y: 20.0,
                angle: 0.0,
                from_planet_id: 0,
                ships: 5,
            },
            Fleet {
                id: 2,
                owner: 2,
                x: 27.0,
                y: 20.0,
                angle: 0.0,
                from_planet_id: 0,
                ships: 12,
            },
        ];
        state.planets[1].x = 28.0;
        state.planets[1].ships = 3;

        let result = step(&mut state, &[vec![], vec![], vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, 2);
        assert_eq!(state.planets[1].ships, 4);
        assert_eq!(result.ships_lost_in_combat, 21);
    }

    #[test]
    fn same_owner_arrival_reinforces_planet() {
        let mut state = base_state(2);
        state.fleets = vec![Fleet {
            id: 0,
            owner: 0,
            x: 24.0,
            y: 20.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 6,
        }];
        state.planets[1].owner = 0;
        state.planets[1].x = 25.0;
        state.planets[1].ships = 8;

        let result = step(&mut state, &[vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, 0);
        assert_eq!(state.planets[1].ships, 16);
        assert_eq!(result.ships_lost_in_combat, 0);
    }

    #[test]
    fn attacker_tie_with_garrison_leaves_zero_ship_owner_unchanged() {
        let mut state = base_state(2);
        state.fleets = vec![Fleet {
            id: 0,
            owner: 0,
            x: 24.0,
            y: 20.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 10,
        }];
        state.planets[1].owner = 1;
        state.planets[1].x = 25.0;
        state.planets[1].ships = 9;
        state.planets[1].production = 1;

        step(&mut state, &[vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, 1);
        assert_eq!(state.planets[1].ships, 0);
    }

    #[test]
    fn exact_boundary_planet_contact_collides() {
        let mut state = base_state(2);
        state.planets[0].x = 10.0;
        state.planets[0].y = 10.0;
        state.planets[1].x = 20.5;
        state.planets[1].y = 5.0;
        state.planets[1].radius = 1.0;
        state.initial_planets = state.planets.clone();
        state.fleets = vec![Fleet {
            id: 0,
            owner: 0,
            x: 20.0,
            y: 4.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 1,
        }];

        step(&mut state, &[vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, -1);
        assert_eq!(state.planets[1].ships, 9);
    }

    #[test]
    fn swept_aabb_keeps_endpoint_collision_candidates() {
        assert!(swept_aabb_overlaps(
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(-0.5, 0.5),
            Point::new(-0.5, 0.5),
            0.8,
        ));
    }

    #[test]
    fn swept_aabb_rejects_far_paths() {
        assert!(!swept_aabb_overlaps(
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(5.0, 2.0),
            Point::new(5.0, 3.0),
            1.0,
        ));
    }

    #[test]
    fn moving_planet_sweeps_fleet_into_combat() {
        let mut state = reset(ResetConfig {
            sim: SimConfig::new(2),
            step: Some(1),
            angular_velocity: Some(0.2),
            planets: Some(vec![
                Planet {
                    id: 0,
                    owner: 0,
                    x: CENTER + 20.0,
                    y: CENTER,
                    radius: 2.0,
                    ships: 10,
                    production: 0,
                },
                Planet {
                    id: 1,
                    owner: 1,
                    x: 90.0,
                    y: 90.0,
                    radius: 2.0,
                    ships: 10,
                    production: 0,
                },
            ]),
            initial_planets: None,
        });
        state.fleets = vec![Fleet {
            id: 0,
            owner: 1,
            x: CENTER + 19.8,
            y: CENTER + 2.0,
            angle: 0.0,
            from_planet_id: 1,
            ships: 3,
        }];

        step(&mut state, &[vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[0].ships, 7);
    }

    #[test]
    #[should_panic(expected = "planet id must be < 100")]
    fn combat_lists_reject_planet_ids_at_limit() {
        let _combat_lists = CombatLists::for_planets(&vec![planet_with_id(100)].into());
    }

    #[test]
    fn queued_combat_for_removed_planet_still_removes_fleet() {
        let mut state = reset(ResetConfig {
            sim: SimConfig::new(2),
            step: None,
            angular_velocity: Some(0.0),
            planets: Some(vec![Planet {
                id: 10,
                owner: 1,
                x: 50.0,
                y: 50.0,
                radius: 2.0,
                ships: 10,
                production: 0,
            }]),
            initial_planets: None,
        });
        state.fleets = vec![Fleet {
            id: 0,
            owner: 0,
            x: 50.0,
            y: 50.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 4,
        }];
        let planet_paths = vec![
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(PlanetPath {
                old_pos: Point::new(48.0, 50.0),
                new_pos: Point::new(52.0, 50.0),
                check_collision: true,
            }),
        ];

        let (combat_lists, _) = move_fleets(&mut state, &planet_paths);
        remove_comet_planets(&mut state, |planet_id| planet_id == 10);
        resolve_combats(&mut state, combat_lists);

        assert!(state.planets.is_empty());
        assert!(state.fleets.is_empty());
    }

    #[test]
    fn remove_comet_planets_handles_duplicate_expired_ids() {
        let mut state = reset(ResetConfig {
            sim: SimConfig::new(2),
            step: None,
            angular_velocity: Some(0.0),
            planets: Some(vec![
                Planet {
                    id: 10,
                    owner: -1,
                    x: -99.0,
                    y: -99.0,
                    radius: 1.0,
                    ships: 3,
                    production: 1,
                },
                Planet {
                    id: 11,
                    owner: -1,
                    x: -99.0,
                    y: -99.0,
                    radius: 1.0,
                    ships: 3,
                    production: 1,
                },
            ]),
            initial_planets: None,
        });
        state.comet_planet_ids = vec![10, 11];
        state.comets = vec![CometGroup {
            planet_ids: vec![10, 11],
            paths: vec![vec![Point::new(1.0, 1.0)], vec![Point::new(2.0, 2.0)]],
            path_index: 0,
        }];
        let expired = [10, 10];

        remove_comet_planets(&mut state, |planet_id| expired.contains(&planet_id));

        assert_eq!(
            state
                .planets
                .iter()
                .map(|planet| planet.id)
                .collect::<Vec<_>>(),
            vec![11]
        );
        assert_eq!(state.initial_planets.len(), 1);
        assert_eq!(state.comet_planet_ids, vec![11]);
        assert_eq!(state.comets[0].planet_ids, vec![11]);
    }

    #[test]
    #[should_panic(expected = "duplicate planet id 10")]
    fn duplicate_planet_ids_are_rejected() {
        let _state = reset(ResetConfig {
            sim: SimConfig::new(2),
            step: None,
            angular_velocity: Some(0.0),
            planets: Some(vec![
                Planet {
                    id: 10,
                    owner: 1,
                    x: 50.0,
                    y: 50.0,
                    radius: 2.0,
                    ships: 5,
                    production: 0,
                },
                Planet {
                    id: 10,
                    owner: -1,
                    x: 60.0,
                    y: 60.0,
                    radius: 2.0,
                    ships: 20,
                    production: 0,
                },
            ]),
            initial_planets: None,
        });
    }

    #[test]
    #[should_panic(expected = "fleet 0 owner cannot be neutral")]
    fn resolve_combats_rejects_neutral_fleet_owner() {
        let mut state = base_state(2);
        let mut combat_lists = CombatLists::for_planets(&state.planets);
        combat_lists.push(
            1,
            Fleet {
                id: 0,
                owner: -1,
                x: 80.0,
                y: 20.0,
                angle: 0.0,
                from_planet_id: 0,
                ships: 1,
            },
        );

        resolve_combats(&mut state, combat_lists);
    }

    #[test]
    #[should_panic(expected = "fleet 0 owner out of bounds for 2 players: 2")]
    fn resolve_combats_rejects_out_of_bounds_fleet_owner() {
        let mut state = base_state(2);
        let mut combat_lists = CombatLists::for_planets(&state.planets);
        combat_lists.push(
            1,
            Fleet {
                id: 0,
                owner: 2,
                x: 80.0,
                y: 20.0,
                angle: 0.0,
                from_planet_id: 0,
                ships: 1,
            },
        );

        resolve_combats(&mut state, combat_lists);
    }

    #[test]
    fn orbiting_planets_rotate_from_initial_position() {
        let mut state = reset(ResetConfig {
            sim: SimConfig::new(2),
            step: None,
            angular_velocity: Some(0.5),
            planets: Some(vec![Planet {
                id: 0,
                owner: 0,
                x: CENTER + 20.0,
                y: CENTER,
                radius: 2.0,
                ships: 10,
                production: 1,
            }]),
            initial_planets: None,
        });
        state.step = 2;

        step(&mut state, &[vec![], vec![]]);

        assert!((state.planets[0].x - (CENTER + 20.0 * 1.0_f64.cos())).abs() <= 1e-12);
        assert!((state.planets[0].y - (CENTER + 20.0 * 1.0_f64.sin())).abs() <= 1e-12);
    }

    #[test]
    fn comet_moves_along_existing_path_and_expires() {
        let mut state = reset(ResetConfig {
            sim: SimConfig::new(2),
            step: None,
            angular_velocity: Some(0.0),
            planets: Some(vec![Planet {
                id: 10,
                owner: -1,
                x: -99.0,
                y: -99.0,
                radius: 1.0,
                ships: 3,
                production: 1,
            }]),
            initial_planets: None,
        });
        state.comet_planet_ids = vec![10];
        state.comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(12.0, 13.0)]],
            path_index: -1,
        }];

        step(&mut state, &[vec![], vec![]]);
        assert_eq!(
            state.planets.get(10).expect("comet planet").position(),
            Point::new(12.0, 13.0)
        );

        step(&mut state, &[vec![], vec![]]);
        assert!(state.planets.is_empty());
        assert!(state.comets.is_empty());
    }

    #[test]
    fn step_limit_sets_player_results_for_actual_player_count() {
        let mut state = base_state(2);
        state.step = 498;

        let result = step(&mut state, &[vec![], vec![]]);

        assert_eq!(
            result.player_results,
            vec![PlayerResult::Won, PlayerResult::Lost]
        );
    }

    #[test]
    fn no_op_score_tie_marks_all_tied_players_as_winners() {
        let mut state = reset(ResetConfig {
            sim: SimConfig {
                player_count: 2,
                episode_steps: 4,
                ship_speed: 6.0,
                comet_speed: 4.0,
            },
            step: Some(2),
            angular_velocity: Some(0.0),
            planets: Some(vec![
                Planet {
                    id: 0,
                    owner: 0,
                    x: 20.0,
                    y: 20.0,
                    radius: 2.0,
                    ships: 10,
                    production: 0,
                },
                Planet {
                    id: 1,
                    owner: 1,
                    x: 80.0,
                    y: 80.0,
                    radius: 2.0,
                    ships: 10,
                    production: 0,
                },
            ]),
            initial_planets: None,
        });

        let result = step(&mut state, &[vec![], vec![]]);

        assert_eq!(
            result.player_results,
            vec![PlayerResult::Won, PlayerResult::Won]
        );
    }

    #[test]
    fn generated_reset_creates_first_playable_state_with_homes() {
        let mut rng = rand::rng();
        let state = reset_with_rng(ResetConfig::new(4), &mut rng);

        assert_eq!(state.step, 1);
        assert!((0.025..0.05).contains(&state.angular_velocity));
        assert!(!state.planets.is_empty());
        assert_eq!(state.initial_planets.len(), state.planets.len());
        assert_eq!(
            state
                .planets
                .iter()
                .filter(|planet| planet.owner != -1)
                .count(),
            4
        );
        assert!(state
            .initial_planets
            .iter()
            .all(|planet| planet.owner == -1));
    }

    #[test]
    fn step_spawns_comets_before_same_step_comet_movement() {
        let mut state = reset(ResetConfig {
            sim: SimConfig::new(2),
            step: Some(49),
            angular_velocity: Some(0.04),
            planets: Some(Vec::new()),
            initial_planets: Some(Vec::new()),
        });
        let mut rng = RepeatingRandom { int: 7, float: 0.5 };

        step_with_rng(&mut state, &[vec![], vec![]], &mut rng);

        assert_eq!(state.comets.len(), 1);
        assert_eq!(state.comets[0].path_index, 0);
        assert_eq!(state.planets.len(), 4);
        assert_eq!(state.comet_planet_ids, vec![1, 2, 3, 4]);
        assert_eq!(state.planets[1].ships, 7);
        assert_ne!(state.planets[1].position(), Point::new(-99.0, -99.0));
    }
}
