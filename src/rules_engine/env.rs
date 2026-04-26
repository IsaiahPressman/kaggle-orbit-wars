use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

use super::generation::{
    assign_home_planets, generate_comet_paths, generate_planets, sample_comet_ships,
    spawn_comet_group, RandomSource,
};
use super::state::{
    CometSpawnInjection, Fleet, LaunchAction, Planet, PlayerResult, Point, ResetConfig, State,
    StepInjections, StepResult, BOARD_SIZE, CENTER, COMET_SPAWN_STEPS, ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
};
use super::utils::{fleet_speed, point_to_segment_distance};

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
        planets,
        initial_planets,
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
    let mut combat_lists = move_fleets(state);
    move_planets_and_sweep(state, &mut combat_lists);
    move_comets_and_sweep(state, &mut combat_lists);
    remove_marked_fleets(state, &combat_lists);
    resolve_combats(state, combat_lists);

    let player_results = player_results(state);
    state.step += 1;

    StepResult { player_results }
}

fn spawn_comets(
    state: &mut State,
    rng: &mut impl RandomSource,
    comet_spawn: Option<CometSpawnInjection>,
) {
    if !COMET_SPAWN_STEPS.contains(&(state.step + 1)) {
        return;
    }

    let (paths, ships) = if let Some(comet_spawn) = comet_spawn {
        (comet_spawn.paths, comet_spawn.ships)
    } else {
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

    remove_comet_planets(state, &expired);
}

fn remove_comet_planets(state: &mut State, expired: &HashSet<u32>) {
    state.planets.retain(|planet| !expired.contains(&planet.id));
    state
        .initial_planets
        .retain(|planet| !expired.contains(&planet.id));
    state
        .comet_planet_ids
        .retain(|planet_id| !expired.contains(planet_id));

    for group in &mut state.comets {
        group
            .planet_ids
            .retain(|planet_id| !expired.contains(planet_id));
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
                .iter_mut()
                .find(|planet| planet.id == action.from_planet_id)
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
    for planet in &mut state.planets {
        if planet.owner != -1 {
            planet.ships += planet.production;
        }
    }
}

fn move_fleets(state: &mut State) -> HashMap<u32, Vec<Fleet>> {
    let mut combat_lists: HashMap<u32, Vec<Fleet>> = state
        .planets
        .iter()
        .map(|planet| (planet.id, Vec::new()))
        .collect();
    let planets = state.planets.clone();
    let mut fleets_to_remove = HashSet::new();

    for fleet in &mut state.fleets {
        let old_pos = fleet.position();
        let speed = fleet_speed(fleet.ships, state.config.ship_speed);
        fleet.x += fleet.angle.cos() * speed;
        fleet.y += fleet.angle.sin() * speed;
        let new_pos = fleet.position();

        if !(0.0..=BOARD_SIZE).contains(&fleet.x) || !(0.0..=BOARD_SIZE).contains(&fleet.y) {
            fleets_to_remove.insert(fleet.id);
            continue;
        }

        if point_to_segment_distance(Point::new(CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS {
            fleets_to_remove.insert(fleet.id);
            continue;
        }

        for planet in &planets {
            if point_to_segment_distance(planet.position(), old_pos, new_pos) < planet.radius {
                combat_lists
                    .get_mut(&planet.id)
                    .expect("combat list exists for every planet")
                    .push(fleet.clone());
                fleets_to_remove.insert(fleet.id);
                break;
            }
        }
    }

    state
        .fleets
        .retain(|fleet| !fleets_to_remove.contains(&fleet.id));
    combat_lists
}

fn move_planets_and_sweep(state: &mut State, combat_lists: &mut HashMap<u32, Vec<Fleet>>) {
    let comet_ids: HashSet<u32> = state.comet_planet_ids.iter().copied().collect();
    let initial_by_id: HashMap<u32, Planet> = state
        .initial_planets
        .iter()
        .map(|planet| (planet.id, planet.clone()))
        .collect();

    let mut sweep_checks = Vec::new();
    for planet in &mut state.planets {
        if comet_ids.contains(&planet.id) {
            continue;
        }

        let Some(initial_planet) = initial_by_id.get(&planet.id) else {
            continue;
        };

        let dx = initial_planet.x - CENTER;
        let dy = initial_planet.y - CENTER;
        let orbital_radius = (dx.powi(2) + dy.powi(2)).sqrt();
        let old_pos = planet.position();

        if orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT {
            let initial_angle = dy.atan2(dx);
            let current_angle = initial_angle + state.angular_velocity * f64::from(state.step);
            planet.x = CENTER + orbital_radius * current_angle.cos();
            planet.y = CENTER + orbital_radius * current_angle.sin();
        }

        sweep_checks.push((planet.id, planet.radius, old_pos, planet.position()));
    }

    for (planet_id, radius, old_pos, new_pos) in sweep_checks {
        sweep_fleets(state, combat_lists, planet_id, radius, old_pos, new_pos);
    }
}

fn move_comets_and_sweep(state: &mut State, combat_lists: &mut HashMap<u32, Vec<Fleet>>) {
    let mut expired = HashSet::new();
    let mut sweep_checks = Vec::new();

    for group in &mut state.comets {
        group.path_index += 1;
        let path_index = group.path_index;

        for (path_offset, planet_id) in group.planet_ids.iter().enumerate() {
            let Some(planet) = state
                .planets
                .iter_mut()
                .find(|planet| planet.id == *planet_id)
            else {
                continue;
            };
            let path = &group.paths[path_offset];

            if path_index >= path.len() as i32 {
                expired.insert(*planet_id);
                continue;
            }

            let old_pos = planet.position();
            let new_pos = path[path_index as usize];
            planet.x = new_pos.x;
            planet.y = new_pos.y;

            if old_pos.x >= 0.0 {
                sweep_checks.push((planet.id, planet.radius, old_pos, planet.position()));
            }
        }
    }

    if !expired.is_empty() {
        remove_comet_planets(state, &expired);
    }

    for (planet_id, radius, old_pos, new_pos) in sweep_checks {
        sweep_fleets(state, combat_lists, planet_id, radius, old_pos, new_pos);
    }
}

fn sweep_fleets(
    state: &State,
    combat_lists: &mut HashMap<u32, Vec<Fleet>>,
    planet_id: u32,
    planet_radius: f64,
    old_pos: Point,
    new_pos: Point,
) {
    if old_pos == new_pos {
        return;
    }

    let already_removed: HashSet<u32> = combat_lists
        .values()
        .flatten()
        .map(|fleet| fleet.id)
        .collect();
    for fleet in &state.fleets {
        if already_removed.contains(&fleet.id) {
            continue;
        }
        if point_to_segment_distance(fleet.position(), old_pos, new_pos) < planet_radius {
            combat_lists
                .entry(planet_id)
                .or_default()
                .push(fleet.clone());
        }
    }
}

fn remove_marked_fleets(state: &mut State, combat_lists: &HashMap<u32, Vec<Fleet>>) {
    let removed: HashSet<u32> = combat_lists
        .values()
        .flatten()
        .map(|fleet| fleet.id)
        .collect();
    state.fleets.retain(|fleet| !removed.contains(&fleet.id));
}

fn resolve_combats(state: &mut State, combat_lists: HashMap<u32, Vec<Fleet>>) {
    for (planet_id, planet_fleets) in combat_lists {
        if planet_fleets.is_empty() {
            continue;
        }

        let Some(planet) = state
            .planets
            .iter_mut()
            .find(|planet| planet.id == planet_id)
        else {
            continue;
        };

        let mut player_ships: HashMap<i32, i32> = HashMap::new();
        for fleet in planet_fleets {
            *player_ships.entry(fleet.owner).or_default() += fleet.ships;
        }

        let mut sorted_players: Vec<(i32, i32)> = player_ships.into_iter().collect();
        sorted_players.sort_by_key(|player| Reverse(player.1));

        let (survivor_owner, survivor_ships) = if sorted_players.len() > 1 {
            let top = sorted_players[0];
            let second = sorted_players[1];
            if top.1 == second.1 {
                (-1, 0)
            } else {
                (top.0, top.1 - second.1)
            }
        } else {
            sorted_players[0]
        };

        if survivor_ships <= 0 {
            continue;
        }

        if planet.owner == survivor_owner {
            planet.ships += survivor_ships;
        } else {
            planet.ships -= survivor_ships;
            if planet.ships < 0 {
                planet.owner = survivor_owner;
                planet.ships = planet.ships.abs();
            }
        }
    }
}

fn player_results(state: &State) -> Vec<PlayerResult> {
    let terminated = is_game_terminated(state);
    if !terminated {
        return vec![PlayerResult::NotDone; state.config.player_count];
    }

    let scores = player_scores(state);
    let max_score = scores.iter().copied().max().unwrap_or(0);
    scores
        .into_iter()
        .map(|score| {
            if score == max_score && max_score > 0 {
                PlayerResult::Win
            } else {
                PlayerResult::Loss
            }
        })
        .collect()
}

pub fn is_game_terminated(state: &State) -> bool {
    reached_step_limit(state) || remaining_alive_players(state) <= 1
}

fn reached_step_limit(state: &State) -> bool {
    state.step >= state.config.episode_steps.saturating_sub(2)
}

fn remaining_alive_players(state: &State) -> usize {
    player_alive_flags(state)
        .into_iter()
        .filter(|alive| *alive)
        .count()
}

pub fn player_alive_flags(state: &State) -> Vec<bool> {
    let mut alive_players = vec![false; state.config.player_count];
    for planet in &state.planets {
        if planet.owner != -1 {
            alive_players[planet.owner as usize] = true;
        }
    }
    for fleet in &state.fleets {
        alive_players[fleet.owner as usize] = true;
    }
    alive_players
}

fn player_scores(state: &State) -> Vec<i32> {
    let mut scores = vec![0; state.config.player_count];
    for planet in &state.planets {
        if planet.owner != -1 {
            scores[planet.owner as usize] += planet.ships;
        }
    }
    for fleet in &state.fleets {
        scores[fleet.owner as usize] += fleet.ships;
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::generation::RandomSource;
    use crate::rules_engine::state::{CometGroup, SimConfig};

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
            vec![PlayerResult::NotDone, PlayerResult::NotDone]
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

        step(
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

        step(&mut state, &[vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, -1);
        assert_eq!(state.planets[1].ships, 7);
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

        step(&mut state, &[vec![], vec![]]);

        assert!(state.fleets.is_empty());
        assert_eq!(state.planets[1].owner, 0);
        assert_eq!(state.planets[1].ships, 16);
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
    fn exact_boundary_planet_contact_does_not_collide() {
        let mut state = base_state(2);
        state.planets[0].x = 10.0;
        state.planets[0].y = 10.0;
        state.planets[1].x = 20.5;
        state.planets[1].y = 21.0;
        state.planets[1].radius = 1.0;
        state.initial_planets = state.planets.clone();
        state.fleets = vec![Fleet {
            id: 0,
            owner: 0,
            x: 20.0,
            y: 20.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 1,
        }];

        step(&mut state, &[vec![], vec![]]);

        assert_eq!(state.fleets.len(), 1);
        assert_eq!(state.fleets[0].position(), Point::new(21.0, 20.0));
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
        assert_eq!(state.planets[0].position(), Point::new(12.0, 13.0));

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
            vec![PlayerResult::Win, PlayerResult::Loss]
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
            vec![PlayerResult::Win, PlayerResult::Win]
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
        assert_eq!(state.planets[0].ships, 7);
        assert_ne!(state.planets[0].position(), Point::new(-99.0, -99.0));
    }
}
