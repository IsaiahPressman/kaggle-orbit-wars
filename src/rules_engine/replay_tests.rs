use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::env::{step_with_injections, PlayerAction};
use super::generation::RandomSource;
use super::state::{
    CometGroup, CometSpawnInjection, Fleet, LaunchAction, Planet, PlanetVector, PlayerResult,
    Point, SimConfig, State, StepInjections, StepResult, COMET_SPAWN_STEPS,
};

const DEFAULT_FIXTURE_DIR: &str = "tests/fixtures/orbit_wars_replays";
const REQUIRED_REPLAY_COVERAGE: [ReplayCoverageRequirement; 2] = [
    ReplayCoverageRequirement {
        episode_id: 75_598_045,
        players: 2,
        rows: 499,
    },
    ReplayCoverageRequirement {
        episode_id: 75_601_099,
        players: 4,
        rows: 141,
    },
];

struct ReplayCoverageRequirement {
    episode_id: u64,
    players: usize,
    rows: usize,
}

#[derive(Debug, Deserialize)]
struct FixtureRow {
    episode_id: u64,
    players: usize,
    step: u32,
    configuration: ConfigFixture,
    before: ObservationFixture,
    actions: Vec<Vec<[f64; 3]>>,
    results: Vec<KagglePlayerResult>,
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

#[derive(Debug, Deserialize)]
struct KagglePlayerResult {
    status: String,
    reward: Option<f64>,
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
    if !require_parity_fixtures()? {
        return Ok(());
    }

    let fixture_dir = fixture_dir();
    let fixture_paths = fixture_paths(&fixture_dir)?;
    if fixture_paths.is_empty() {
        warn_or_fail_missing_fixtures(&fixture_dir)?;
        return Ok(());
    }

    let mut coverage = REQUIRED_REPLAY_COVERAGE
        .iter()
        .map(|requirement| (requirement.episode_id, (0_usize, None)))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut checked_rows = 0;
    for fixture_path in fixture_paths {
        let file = File::open(&fixture_path)?;
        let mut final_results = None;
        for line in BufReader::new(file).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let row: FixtureRow = serde_json::from_str(&line)?;
            let result = check_transition(&row)
                .map_err(|message| format!("{} step {}: {message}", row.episode_id, row.step))?;
            final_results = Some((
                row.episode_id,
                row.step,
                player_results_from_kaggle(&row).map_err(|message| {
                    format!("{} step {}: {message}", row.episode_id, row.step)
                })?,
                result,
            ));
            if let Some((rows, players)) = coverage.get_mut(&row.episode_id) {
                *rows += 1;
                if players.is_none() {
                    *players = Some(row.players);
                } else if *players != Some(row.players) {
                    return Err(format!(
                        "episode {} mixed player counts in replay fixture",
                        row.episode_id
                    )
                    .into());
                }
            }
            checked_rows += 1;
        }
        validate_final_results(&fixture_path, final_results)?;
    }

    assert!(
        checked_rows > 0,
        "replay fixtures contained no transition rows"
    );
    validate_required_coverage(&coverage)?;
    Ok(())
}

fn validate_final_results(
    fixture_path: &Path,
    final_results: Option<(u64, u32, Vec<PlayerResult>, StepResult)>,
) -> Result<(), Box<dyn Error>> {
    let Some((episode_id, step, kaggle_results, rust_result)) = final_results else {
        return Err(format!("{} contained no transition rows", fixture_path.display()).into());
    };
    if kaggle_results
        .iter()
        .any(|result| matches!(result, PlayerResult::Active))
    {
        return Err(format!(
            "{} final row episode {episode_id} step {step} had nonterminal Kaggle results: {:?}",
            fixture_path.display(),
            kaggle_results
        )
        .into());
    }
    if rust_result
        .player_results
        .iter()
        .any(|result| matches!(result, PlayerResult::Active))
    {
        return Err(format!(
            "{} final row episode {episode_id} step {step} had nonterminal Rust results: {:?}",
            fixture_path.display(),
            rust_result.player_results
        )
        .into());
    }
    Ok(())
}

fn validate_required_coverage(
    coverage: &std::collections::BTreeMap<u64, (usize, Option<usize>)>,
) -> Result<(), Box<dyn Error>> {
    for requirement in REQUIRED_REPLAY_COVERAGE {
        let Some((rows, players)) = coverage.get(&requirement.episode_id) else {
            return Err(format!(
                "required replay fixture for episode {} is missing",
                requirement.episode_id
            )
            .into());
        };
        if *rows != requirement.rows {
            return Err(format!(
                "episode {} row count mismatch: {} != {}",
                requirement.episode_id, rows, requirement.rows
            )
            .into());
        }
        if *players != Some(requirement.players) {
            return Err(format!(
                "episode {} player count mismatch: {:?} != {}",
                requirement.episode_id, players, requirement.players
            )
            .into());
        }
    }
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

fn check_transition(row: &FixtureRow) -> Result<StepResult, String> {
    let mut state = state_from_observation(row)?;
    let actions = python_validated_actions(&state, &row.actions)?;
    let injections = injections_from_expected(row)?;
    let mut rng = PanicRandom;

    let result = step_with_injections(&mut state, &actions, &mut rng, injections);

    compare_state(&state, &result, row)?;
    Ok(result)
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
        planets: row
            .before
            .planets
            .iter()
            .map(planet_from_array)
            .collect::<Vec<_>>()
            .into(),
        initial_planets: row
            .before
            .initial_planets
            .iter()
            .map(planet_from_array)
            .collect::<Vec<_>>()
            .into(),
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
        }]
        .into(),
        initial_planets: Vec::new().into(),
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

#[test]
fn replay_skip_spawn_injection_does_not_request_rng() {
    let mut state = State {
        config: SimConfig::new(2),
        step: 49,
        angular_velocity: 0.0,
        planets: vec![
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
                y: 80.0,
                radius: 2.0,
                ships: 10,
                production: 1,
            },
        ]
        .into(),
        initial_planets: Vec::new().into(),
        fleets: Vec::new(),
        next_fleet_id: 0,
        comets: Vec::new(),
        comet_planet_ids: Vec::new(),
    };
    let mut rng = PanicRandom;

    step_with_injections(
        &mut state,
        &[vec![], vec![]],
        &mut rng,
        StepInjections {
            comet_spawn: Some(CometSpawnInjection::Skip),
        },
    );

    assert!(state.comets.is_empty());
}

fn injections_from_expected(row: &FixtureRow) -> Result<StepInjections, String> {
    if !COMET_SPAWN_STEPS.contains(&row.step) {
        return Ok(StepInjections::default());
    }

    let before_comet_planet_ids = row
        .before
        .comets
        .iter()
        .flat_map(|comet| comet.planet_ids.iter().copied())
        .collect::<std::collections::HashSet<_>>();
    let new_comets = row
        .expected
        .comets
        .iter()
        .filter(|comet| {
            comet
                .planet_ids
                .first()
                .is_some_and(|planet_id| !before_comet_planet_ids.contains(planet_id))
        })
        .collect::<Vec<_>>();

    if new_comets.is_empty() {
        return Ok(StepInjections {
            comet_spawn: Some(CometSpawnInjection::Skip),
        });
    }
    if new_comets.len() > 1 {
        return Err(format!(
            "expected at most one spawned comet group, got {}",
            new_comets.len()
        ));
    }

    let comet = new_comets[0];
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
        comet_spawn: Some(CometSpawnInjection::Spawn {
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
    compare_player_results(result, row)?;
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

fn compare_player_results(result: &StepResult, row: &FixtureRow) -> Result<(), String> {
    let expected_player_results = player_results_from_kaggle(row)?;
    if expected_player_results
        .iter()
        .all(|result| !matches!(result, PlayerResult::Active))
    {
        if result.player_results != expected_player_results {
            return Err(format!(
                "player results mismatch: {:?} != {:?}",
                result.player_results, expected_player_results
            ));
        }
        return Ok(());
    }

    let active_count = result
        .player_results
        .iter()
        .filter(|result| matches!(result, PlayerResult::Active))
        .count();
    if active_count > 1 {
        return Ok(());
    }

    Err(format!(
        "nonterminal replay transition left {active_count} active Rust players: {:?}",
        result.player_results
    ))
}

fn player_results_from_kaggle(row: &FixtureRow) -> Result<Vec<PlayerResult>, String> {
    if row.results.len() != row.players {
        return Err(format!(
            "expected {} Kaggle player results, got {}",
            row.players,
            row.results.len()
        ));
    }

    row.results
        .iter()
        .enumerate()
        .map(|(player_id, result)| match result.status.as_str() {
            "ACTIVE" => {
                if result.reward.unwrap_or(0.0) != 0.0 {
                    return Err(format!(
                        "active player {player_id} had non-zero reward {:?}",
                        result.reward
                    ));
                }
                Ok(PlayerResult::Active)
            },
            "DONE" => {
                if result.reward.unwrap_or(0.0) > 0.0 {
                    Ok(PlayerResult::Won)
                } else {
                    Ok(PlayerResult::Lost)
                }
            },
            status => Err(format!(
                "unsupported Kaggle status for player {player_id}: {status:?}"
            )),
        })
        .collect()
}

fn compare_planets(actual: &PlanetVector, expected: &[Planet]) -> Result<(), String> {
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
