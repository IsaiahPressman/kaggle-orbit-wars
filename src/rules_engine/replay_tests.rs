use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::env::{step_with_injections, PlayerAction};
use super::generation::RandomSource;
use super::state::{
    CometGroup, CometSpawnInjection, Fleet, LaunchAction, Planet, PlayerResult, Point, SimConfig,
    State, StepInjections, StepResult,
};

const DEFAULT_FIXTURE_DIR: &str = "tests/fixtures/orbit_wars_replays";

#[derive(Debug, Deserialize)]
struct FixtureRow {
    episode_id: u64,
    players: usize,
    step: u32,
    configuration: ConfigFixture,
    before: ObservationFixture,
    actions: Vec<Vec<[f64; 3]>>,
    expected: ObservationFixture,
}

#[derive(Debug, Deserialize)]
struct ConfigFixture {
    #[serde(rename = "episodeSteps")]
    episode_steps: u32,
    #[serde(rename = "shipSpeed")]
    ship_speed: f64,
    #[serde(rename = "cometSpeed")]
    comet_speed: f64,
}

#[derive(Debug, Deserialize)]
struct ObservationFixture {
    planets: Vec<[f64; 7]>,
    fleets: Vec<[f64; 7]>,
    angular_velocity: f64,
    initial_planets: Vec<[f64; 7]>,
    next_fleet_id: u32,
    comets: Vec<CometFixture>,
    comet_planet_ids: Vec<u32>,
    step: u32,
}

#[derive(Debug, Deserialize)]
struct CometFixture {
    planet_ids: Vec<u32>,
    paths: Vec<Vec<[f64; 2]>>,
    path_index: i32,
}

struct PanicRandom;

impl RandomSource for PanicRandom {
    fn randint(&mut self, _low: i32, _high: i32) -> i32 {
        panic!("replay parity test unexpectedly requested an integer random value")
    }

    fn uniform(&mut self, _low: f64, _high: f64) -> f64 {
        panic!("replay parity test unexpectedly requested a float random value")
    }
}

#[test]
fn replay_fixtures_match_reference_transitions() -> Result<(), Box<dyn Error>> {
    let fixture_dir = fixture_dir();
    let fixture_paths = fixture_paths(&fixture_dir)?;
    if fixture_paths.is_empty() {
        warn_or_fail_missing_fixtures(&fixture_dir)?;
        return Ok(());
    }

    let mut checked_rows = 0;
    for fixture_path in fixture_paths {
        let file = File::open(&fixture_path)?;
        for line in BufReader::new(file).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let row: FixtureRow = serde_json::from_str(&line)?;
            check_transition(&row)
                .map_err(|message| format!("{} step {}: {message}", row.episode_id, row.step))?;
            checked_rows += 1;
        }
    }

    assert!(
        checked_rows > 0,
        "replay fixtures contained no transition rows"
    );
    Ok(())
}

fn fixture_dir() -> PathBuf {
    std::env::var("ORBIT_WARS_PARITY_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_FIXTURE_DIR))
}

fn fixture_paths(fixture_dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let Ok(entries) = std::fs::read_dir(fixture_dir) else {
        return Ok(Vec::new());
    };

    let mut fixture_paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("replay-") && name.ends_with(".jsonl"))
        })
        .collect::<Vec<_>>();
    fixture_paths.sort();

    Ok(fixture_paths)
}

fn warn_or_fail_missing_fixtures(fixture_dir: &Path) -> Result<(), Box<dyn Error>> {
    if require_parity_fixtures()? {
        let message = format!(
            "No replay parity fixtures found in {}. \
            Run scripts/regenerate_test_fixtures.sh to enable replay parity, \
            or set REQUIRE_PARITY_FIXTURES=0 to skip replay parity.",
            fixture_dir.display()
        );
        Err(message.into())
    } else {
        Ok(())
    }
}

fn require_parity_fixtures() -> Result<bool, Box<dyn Error>> {
    let Ok(value) = std::env::var("REQUIRE_PARITY_FIXTURES") else {
        return Ok(true);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" => Ok(true),
        "0" | "false" => Ok(false),
        _ => {
            Err(format!("REQUIRE_PARITY_FIXTURES must be 1/true or 0/false, got {value:?}").into())
        },
    }
}

fn check_transition(row: &FixtureRow) -> Result<(), String> {
    let mut state = state_from_observation(row)?;
    let actions = python_validated_actions(&state, &row.actions)?;
    let injections = injections_from_expected(row)?;
    let mut rng = PanicRandom;

    let result = step_with_injections(&mut state, &actions, &mut rng, injections);

    compare_state(&state, &result, row)
}

fn state_from_observation(row: &FixtureRow) -> Result<State, String> {
    let config = SimConfig {
        player_count: row.players,
        episode_steps: row.configuration.episode_steps,
        ship_speed: row.configuration.ship_speed,
        comet_speed: row.configuration.comet_speed,
    };

    Ok(State {
        config,
        step: row.before.step,
        angular_velocity: row.before.angular_velocity,
        planets: row.before.planets.iter().map(planet_from_array).collect(),
        initial_planets: row
            .before
            .initial_planets
            .iter()
            .map(planet_from_array)
            .collect(),
        fleets: row.before.fleets.iter().map(fleet_from_array).collect(),
        next_fleet_id: row.before.next_fleet_id,
        comets: row.before.comets.iter().map(comet_from_fixture).collect(),
        comet_planet_ids: row.before.comet_planet_ids.clone(),
    })
}

fn planet_from_array(raw: &[f64; 7]) -> Planet {
    Planet {
        id: raw[0] as u32,
        owner: raw[1] as i32,
        x: raw[2],
        y: raw[3],
        radius: raw[4],
        ships: raw[5] as i32,
        production: raw[6] as i32,
    }
}

fn fleet_from_array(raw: &[f64; 7]) -> Fleet {
    Fleet {
        id: raw[0] as u32,
        owner: raw[1] as i32,
        x: raw[2],
        y: raw[3],
        angle: raw[4],
        from_planet_id: raw[5] as u32,
        ships: raw[6] as i32,
    }
}

fn comet_from_fixture(raw: &CometFixture) -> CometGroup {
    CometGroup {
        planet_ids: raw.planet_ids.clone(),
        paths: raw
            .paths
            .iter()
            .map(|path| {
                path.iter()
                    .map(|point| Point::new(point[0], point[1]))
                    .collect()
            })
            .collect(),
        path_index: raw.path_index,
    }
}

fn python_validated_actions(
    state: &State,
    raw_actions: &[Vec<[f64; 3]>],
) -> Result<Vec<PlayerAction>, String> {
    if raw_actions.len() != state.config.player_count {
        return Err(format!(
            "expected {} player action lists, got {}",
            state.config.player_count,
            raw_actions.len()
        ));
    }

    let mut planet_ships = state
        .planets
        .iter()
        .map(|planet| (planet.id, (planet.owner, planet.ships)))
        .collect::<std::collections::HashMap<_, _>>();
    let mut actions = vec![Vec::new(); state.config.player_count];

    for (player_id, player_actions) in raw_actions.iter().enumerate() {
        for action in player_actions {
            let from_planet_raw = action[0];
            let angle = action[1];
            let ships = action[2] as i32;
            let Some((&from_planet_id, (owner, available_ships))) = planet_ships
                .iter_mut()
                .find(|(planet_id, _)| f64::from(**planet_id) == from_planet_raw)
            else {
                continue;
            };
            if *owner != player_id as i32 || *available_ships < ships || ships <= 0 {
                continue;
            }

            *available_ships -= ships;
            actions[player_id].push(LaunchAction {
                from_planet_id,
                angle,
                ships,
            });
        }
    }

    Ok(actions)
}

#[test]
fn python_validated_actions_does_not_truncate_planet_ids() -> Result<(), String> {
    let state = State {
        config: SimConfig::new(2),
        step: 1,
        angular_velocity: 0.0,
        planets: vec![Planet {
            id: 0,
            owner: 0,
            x: 20.0,
            y: 20.0,
            radius: 2.0,
            ships: 10,
            production: 1,
        }],
        initial_planets: Vec::new(),
        fleets: Vec::new(),
        next_fleet_id: 0,
        comets: Vec::new(),
        comet_planet_ids: Vec::new(),
    };
    let actions = python_validated_actions(
        &state,
        &[
            vec![
                [0.9, 0.0, 5.0],
                [-0.1, 0.0, 5.0],
                [0.0, 0.25, 5.9],
                [0.0, 0.5, 6.0],
            ],
            vec![],
        ],
    )?;

    assert_eq!(
        actions,
        vec![
            vec![LaunchAction {
                from_planet_id: 0,
                angle: 0.25,
                ships: 5,
            }],
            vec![],
        ]
    );
    Ok(())
}

fn injections_from_expected(row: &FixtureRow) -> Result<StepInjections, String> {
    if row.expected.comets.len() <= row.before.comets.len() {
        return Ok(StepInjections::default());
    }

    let comet = &row.expected.comets[row.before.comets.len()];
    let Some(first_planet_id) = comet.planet_ids.first() else {
        return Err("spawned comet group had no planet ids".to_string());
    };
    let Some(ships) = row
        .expected
        .initial_planets
        .iter()
        .find(|planet| planet[0] as u32 == *first_planet_id)
        .map(|planet| planet[5] as i32)
    else {
        return Err(format!(
            "missing spawned comet initial planet {first_planet_id}"
        ));
    };

    Ok(StepInjections {
        comet_spawn: Some(CometSpawnInjection {
            paths: comet
                .paths
                .iter()
                .map(|path| {
                    path.iter()
                        .map(|point| Point::new(point[0], point[1]))
                        .collect()
                })
                .collect(),
            ships,
        }),
    })
}

fn compare_state(state: &State, result: &StepResult, row: &FixtureRow) -> Result<(), String> {
    if state.step != row.expected.step {
        return Err(format!(
            "step mismatch: {} != {}",
            state.step, row.expected.step
        ));
    }
    let expected_player_results = expected_player_results(row);
    if result.player_results != expected_player_results {
        return Err(format!(
            "player results mismatch: {:?} != {:?}",
            result.player_results, expected_player_results
        ));
    }
    close(
        state.angular_velocity,
        row.expected.angular_velocity,
        "angular_velocity",
    )?;
    compare_planets(
        &state.planets,
        &row.expected
            .planets
            .iter()
            .map(planet_from_array)
            .collect::<Vec<_>>(),
    )?;
    compare_planets(
        &state.initial_planets,
        &row.expected
            .initial_planets
            .iter()
            .map(planet_from_array)
            .collect::<Vec<_>>(),
    )?;
    compare_fleets(
        &state.fleets,
        &row.expected
            .fleets
            .iter()
            .map(fleet_from_array)
            .collect::<Vec<_>>(),
    )?;
    if state.next_fleet_id != row.expected.next_fleet_id {
        return Err(format!(
            "next_fleet_id mismatch: {} != {}",
            state.next_fleet_id, row.expected.next_fleet_id
        ));
    }
    if state.comet_planet_ids != row.expected.comet_planet_ids {
        return Err("comet_planet_ids mismatch".to_string());
    }
    if state.comets.len() != row.expected.comets.len() {
        return Err(format!(
            "comet group count mismatch: {} != {}",
            state.comets.len(),
            row.expected.comets.len()
        ));
    }
    for (actual, expected) in state.comets.iter().zip(&row.expected.comets) {
        if actual.planet_ids != expected.planet_ids {
            return Err("comet group planet_ids mismatch".to_string());
        }
        if actual.path_index != expected.path_index {
            return Err(format!(
                "comet path_index mismatch: {} != {}",
                actual.path_index, expected.path_index
            ));
        }
        compare_paths(&actual.paths, &expected.paths)?;
    }
    Ok(())
}

fn expected_player_results(row: &FixtureRow) -> Vec<PlayerResult> {
    let reached_step_limit =
        row.expected.step.saturating_sub(1) >= row.configuration.episode_steps.saturating_sub(2);
    let alive_flags = player_alive_flags(&row.expected, row.players);
    let terminated = reached_step_limit || alive_flags.iter().filter(|alive| **alive).count() <= 1;
    if !terminated {
        return alive_flags
            .into_iter()
            .map(|alive| {
                if alive {
                    PlayerResult::Active
                } else {
                    PlayerResult::Lost
                }
            })
            .collect();
    }

    let scores = player_scores(row);
    let max_score = scores.iter().copied().max().unwrap_or(0);
    scores
        .into_iter()
        .map(|score| {
            if score == max_score && max_score > 0 {
                PlayerResult::Won
            } else {
                PlayerResult::Lost
            }
        })
        .collect()
}

fn player_alive_flags(observation: &ObservationFixture, player_count: usize) -> Vec<bool> {
    let mut alive_players = vec![false; player_count];
    for planet in &observation.planets {
        if planet[1] != -1.0 {
            alive_players[planet[1] as usize] = true;
        }
    }
    for fleet in &observation.fleets {
        alive_players[fleet[1] as usize] = true;
    }
    alive_players
}

fn player_scores(row: &FixtureRow) -> Vec<i32> {
    let mut scores = vec![0; row.players];
    for planet in &row.expected.planets {
        if planet[1] != -1.0 {
            scores[planet[1] as usize] += planet[5] as i32;
        }
    }
    for fleet in &row.expected.fleets {
        scores[fleet[1] as usize] += fleet[6] as i32;
    }
    scores
}

fn compare_planets(actual: &[Planet], expected: &[Planet]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "planet count mismatch: {} != {}",
            actual.len(),
            expected.len()
        ));
    }

    for (actual, expected) in actual.iter().zip(expected) {
        if actual.id != expected.id
            || actual.owner != expected.owner
            || actual.ships != expected.ships
            || actual.production != expected.production
        {
            return Err(format!(
                "planet discrete mismatch: {actual:?} != {expected:?}"
            ));
        }
        close(actual.x, expected.x, "planet x")?;
        close(actual.y, expected.y, "planet y")?;
        close(actual.radius, expected.radius, "planet radius")?;
    }
    Ok(())
}

fn compare_fleets(actual: &[Fleet], expected: &[Fleet]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "fleet count mismatch: {} != {}",
            actual.len(),
            expected.len()
        ));
    }

    for (actual, expected) in actual.iter().zip(expected) {
        if actual.id != expected.id
            || actual.owner != expected.owner
            || actual.from_planet_id != expected.from_planet_id
            || actual.ships != expected.ships
        {
            return Err(format!(
                "fleet discrete mismatch: {actual:?} != {expected:?}"
            ));
        }
        close(actual.x, expected.x, "fleet x")?;
        close(actual.y, expected.y, "fleet y")?;
        close(actual.angle, expected.angle, "fleet angle")?;
    }
    Ok(())
}

fn compare_paths(actual: &[Vec<Point>], expected: &[Vec<[f64; 2]>]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "comet path count mismatch: {} != {}",
            actual.len(),
            expected.len()
        ));
    }

    for (actual_path, expected_path) in actual.iter().zip(expected) {
        if actual_path.len() != expected_path.len() {
            return Err(format!(
                "comet path length mismatch: {} != {}",
                actual_path.len(),
                expected_path.len()
            ));
        }
        for (actual_point, expected_point) in actual_path.iter().zip(expected_path) {
            close(actual_point.x, expected_point[0], "comet path x")?;
            close(actual_point.y, expected_point[1], "comet path y")?;
        }
    }

    Ok(())
}

fn close(actual: f64, expected: f64, field: &str) -> Result<(), String> {
    let abs_diff = (actual - expected).abs();
    let tolerance = 1e-9_f64.max(1e-9 * actual.abs().max(expected.abs()));
    if abs_diff <= tolerance {
        Ok(())
    } else {
        Err(format!(
            "{field} mismatch: {actual} != {expected} (diff {abs_diff})"
        ))
    }
}
