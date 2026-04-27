use rand::RngExt;

use super::state::{
    CometGroup, Planet, Point, BOARD_SIZE, CENTER, COMET_PRODUCTION, COMET_RADIUS,
    MAX_PLANET_GROUPS, MIN_PLANET_GROUPS, MIN_STATIC_GROUPS, PLANET_CLEARANCE,
    ROTATION_RADIUS_LIMIT, SUN_RADIUS,
};
use super::utils::{distance, fourfold_symmetric_points, is_orbiting, orbit_position};

pub trait RandomSource {
    fn randint(&mut self, low: i32, high: i32) -> i32;
    fn uniform(&mut self, low: f64, high: f64) -> f64;
}

impl<T: rand::Rng + ?Sized> RandomSource for T {
    fn randint(&mut self, low: i32, high: i32) -> i32 {
        self.random_range(low..=high)
    }

    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        self.random_range(low..high)
    }
}

pub fn planet_radius(production: i32) -> f64 {
    1.0 + f64::from(production).ln()
}

pub fn generate_planets(rng: &mut impl RandomSource) -> Vec<Planet> {
    let mut planets = Vec::new();
    let num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS) as usize;
    let mut next_id = 0;

    if let Some(group) = place_diagonal_group(&planets, next_id, static_diagonal_orbital_range, rng)
    {
        planets.extend(group);
        next_id += 4;
    }

    if let Some(group) =
        place_diagonal_group(&planets, next_id, orbiting_diagonal_orbital_range, rng)
    {
        planets.extend(group);
        next_id += 4;
    }

    let mut static_groups = 0;
    for _ in 0..5000 {
        if static_groups >= MIN_STATIC_GROUPS {
            break;
        }

        let production = rng.randint(1, 5);
        let radius = planet_radius(production);
        let angle = rng.uniform(0.0, std::f64::consts::FRAC_PI_2);
        let min_orbital = ROTATION_RADIUS_LIMIT - radius;
        let max_orbital = (BOARD_SIZE - CENTER - radius) / angle.cos().max(angle.sin());
        if min_orbital > max_orbital {
            continue;
        }

        let orbital_radius = rng.uniform(min_orbital, max_orbital);
        let x = CENTER + orbital_radius * angle.cos();
        let y = CENTER + orbital_radius * angle.sin();

        if x + radius > BOARD_SIZE
            || x - radius < 0.0
            || y + radius > BOARD_SIZE
            || y - radius < 0.0
        {
            continue;
        }
        if (BOARD_SIZE - x) - radius < 0.0 || (BOARD_SIZE - y) - radius < 0.0 {
            continue;
        }
        if (x - CENTER) < radius + 5.0 || (y - CENTER) < radius + 5.0 {
            continue;
        }

        let ships = rng.randint(5, 99).min(rng.randint(5, 99));
        let group = symmetric_planets(next_id, -1, x, y, radius, ships, production);
        if no_initial_overlap(&group, &planets) {
            planets.extend(group);
            next_id += 4;
            static_groups += 1;
        }
    }

    let mut attempts = 0;
    let mut has_orbiting = false;
    while planets.len() < num_q1 * 4 || (!has_orbiting && attempts < 5000) {
        attempts += 1;
        if attempts >= 5000 {
            break;
        }

        let production = rng.randint(1, 5);
        let radius = planet_radius(production);
        let x = rng.uniform(CENTER + 15.0, BOARD_SIZE - radius - 5.0);
        let y = rng.uniform(CENTER + 15.0, BOARD_SIZE - radius - 5.0);
        let orbital_radius = distance(Point::new(x, y), Point::new(CENTER, CENTER));

        if orbital_radius < SUN_RADIUS + radius + 10.0 {
            continue;
        }

        if !is_orbiting(Point::new(x, y), radius)
            && (x + radius > BOARD_SIZE
                || x - radius < 0.0
                || y + radius > BOARD_SIZE
                || y - radius < 0.0)
        {
            continue;
        }

        let ships = rng.randint(5, 30);
        let group = symmetric_planets(next_id, -1, x, y, radius, ships, production);
        if valid_group(&group, &planets) {
            if is_orbiting(Point::new(x, y), radius) {
                has_orbiting = true;
            }
            planets.extend(group);
            next_id += 4;
        }
    }

    planets
}

pub fn assign_home_planets(
    planets: &mut [Planet],
    player_count: usize,
    rng: &mut impl RandomSource,
) {
    assert!(
        player_count == 2 || player_count == 4,
        "Orbit Wars supports exactly 2 or 4 players"
    );
    let num_groups = planets.len() / 4;
    if num_groups == 0 {
        return;
    }

    let home_group = if player_count == 4 {
        let diagonal_groups = (0..num_groups)
            .filter(|group_index| {
                let planet = &planets[group_index * 4];
                ((planet.x - CENTER) - (planet.y - CENTER)).abs() < 0.01
            })
            .collect::<Vec<_>>();
        assert!(
            !diagonal_groups.is_empty(),
            "4p requires at least one y=x diagonal group"
        );
        let diagonal_index = rng.randint(0, diagonal_groups.len() as i32 - 1) as usize;
        diagonal_groups[diagonal_index]
    } else {
        rng.randint(0, num_groups as i32 - 1) as usize
    };
    let base = home_group * 4;

    if player_count == 2 {
        planets[base].owner = 0;
        planets[base].ships = 10;
        planets[base + 3].owner = 1;
        planets[base + 3].ships = 10;
    } else {
        for player_id in 0..4 {
            planets[base + player_id].owner = player_id as i32;
            planets[base + player_id].ships = 10;
        }
    }
}

fn place_diagonal_group(
    planets: &[Planet],
    next_id: u32,
    orbital_range: fn(f64) -> (f64, f64),
    rng: &mut impl RandomSource,
) -> Option<Vec<Planet>> {
    for _ in 0..1000 {
        let production = rng.randint(1, 5);
        let radius = planet_radius(production);
        let (min_orbital, max_orbital) = orbital_range(radius);
        if min_orbital >= max_orbital {
            continue;
        }

        let orbital_radius = rng.uniform(min_orbital, max_orbital);
        let x = CENTER + orbital_radius * std::f64::consts::FRAC_1_SQRT_2;
        let y = CENTER + orbital_radius * std::f64::consts::FRAC_1_SQRT_2;
        let ships = rng.randint(5, 99).min(rng.randint(5, 99));
        let group = symmetric_planets(next_id, -1, x, y, radius, ships, production);

        if valid_group(&group, planets) {
            return Some(group);
        }
    }

    None
}

fn static_diagonal_orbital_range(radius: f64) -> (f64, f64) {
    (
        (ROTATION_RADIUS_LIMIT - radius).max((radius + 5.0) * std::f64::consts::SQRT_2),
        (BOARD_SIZE - CENTER - radius) * std::f64::consts::SQRT_2,
    )
}

fn orbiting_diagonal_orbital_range(radius: f64) -> (f64, f64) {
    (SUN_RADIUS + radius + 10.0, ROTATION_RADIUS_LIMIT - radius)
}

pub fn generate_comet_paths(
    initial_planets: &[Planet],
    angular_velocity: f64,
    spawn_step: u32,
    comet_planet_ids: &[u32],
    comet_speed: f64,
    rng: &mut impl RandomSource,
) -> Option<Vec<Vec<Point>>> {
    let comet_ids: std::collections::HashSet<u32> = comet_planet_ids.iter().copied().collect();

    for _ in 0..300 {
        let eccentricity = rng.uniform(0.75, 0.93);
        let semi_major = rng.uniform(60.0, 150.0);
        let perihelion = semi_major * (1.0 - eccentricity);
        if perihelion < SUN_RADIUS + COMET_RADIUS {
            continue;
        }

        let semi_minor = semi_major * (1.0 - eccentricity.powi(2)).sqrt();
        let focus_distance = semi_major * eccentricity;
        let phi = rng.uniform(std::f64::consts::PI / 6.0, std::f64::consts::PI / 3.0);

        let mut dense = Vec::with_capacity(5000);
        for i in 0..5000 {
            let t = 0.3 * std::f64::consts::PI + 1.4 * std::f64::consts::PI * f64::from(i) / 4999.0;
            let ellipse_x = focus_distance + semi_major * t.cos();
            let ellipse_y = semi_minor * t.sin();
            dense.push(Point::new(
                CENTER + ellipse_x * phi.cos() - ellipse_y * phi.sin(),
                CENTER + ellipse_x * phi.sin() + ellipse_y * phi.cos(),
            ));
        }

        let mut path = vec![dense[0]];
        let mut cumulative = 0.0;
        let mut target = comet_speed;
        for i in 1..dense.len() {
            cumulative += distance(dense[i], dense[i - 1]);
            if cumulative >= target {
                path.push(dense[i]);
                target += comet_speed;
            }
        }

        let mut board_start = None;
        let mut board_end = None;
        for (index, point) in path.iter().enumerate() {
            if (0.0..=BOARD_SIZE).contains(&point.x) && (0.0..=BOARD_SIZE).contains(&point.y) {
                if board_start.is_none() {
                    board_start = Some(index);
                }
                board_end = Some(index);
            }
        }

        let (Some(board_start), Some(board_end)) = (board_start, board_end) else {
            continue;
        };
        let visible = &path[board_start..=board_end];
        if !(5..=40).contains(&visible.len()) {
            continue;
        }

        if comet_path_is_valid(
            visible,
            initial_planets,
            angular_velocity,
            spawn_step,
            &comet_ids,
        ) {
            return Some(symmetric_paths(visible));
        }
    }

    None
}

pub fn sample_comet_ships(rng: &mut impl RandomSource) -> i32 {
    rng.randint(1, 99)
        .min(rng.randint(1, 99))
        .min(rng.randint(1, 99))
        .min(rng.randint(1, 99))
}

pub fn spawn_comet_group(
    planets: &mut Vec<Planet>,
    initial_planets: &mut Vec<Planet>,
    comet_planet_ids: &mut Vec<u32>,
    paths: Vec<Vec<Point>>,
    ships: i32,
) -> CometGroup {
    let next_id = planets.iter().map(|planet| planet.id).max().unwrap_or(0) + 1;
    let mut group = CometGroup {
        planet_ids: Vec::with_capacity(paths.len()),
        paths,
        path_index: -1,
    };

    for index in 0..group.paths.len() {
        let id = next_id + index as u32;
        group.planet_ids.push(id);
        comet_planet_ids.push(id);
        let planet = Planet {
            id,
            owner: -1,
            x: -99.0,
            y: -99.0,
            radius: COMET_RADIUS,
            ships,
            production: COMET_PRODUCTION,
        };
        planets.push(planet.clone());
        initial_planets.push(planet);
    }

    group
}

fn symmetric_planets(
    id: u32,
    owner: i32,
    x: f64,
    y: f64,
    radius: f64,
    ships: i32,
    production: i32,
) -> Vec<Planet> {
    fourfold_symmetric_points(Point::new(x, y))
        .into_iter()
        .enumerate()
        .map(|(index, point)| {
            planet(
                id + index as u32,
                owner,
                point.x,
                point.y,
                radius,
                ships,
                production,
            )
        })
        .collect()
}

fn planet(id: u32, owner: i32, x: f64, y: f64, radius: f64, ships: i32, production: i32) -> Planet {
    Planet {
        id,
        owner,
        x,
        y,
        radius,
        ships,
        production,
    }
}

fn no_initial_overlap(group: &[Planet], planets: &[Planet]) -> bool {
    group.iter().all(|candidate| {
        planets.iter().all(|planet| {
            distance(candidate.position(), planet.position())
                >= candidate.radius + planet.radius + PLANET_CLEARANCE
        })
    })
}

fn valid_group(group: &[Planet], planets: &[Planet]) -> bool {
    group.iter().all(|candidate| {
        let candidate_orbital = distance(candidate.position(), Point::new(CENTER, CENTER));
        let candidate_rotating = is_orbiting(candidate.position(), candidate.radius);
        planets.iter().all(|planet| {
            let planet_orbital = distance(planet.position(), Point::new(CENTER, CENTER));
            let planet_rotating = is_orbiting(planet.position(), planet.radius);

            distance(candidate.position(), planet.position())
                >= candidate.radius + planet.radius + PLANET_CLEARANCE
                && (candidate_rotating == planet_rotating
                    || (candidate_orbital - planet_orbital).abs()
                        >= candidate.radius + planet.radius + PLANET_CLEARANCE)
        })
    })
}

fn symmetric_paths(visible: &[Point]) -> Vec<Vec<Point>> {
    let mut paths = (0..4)
        .map(|_| Vec::with_capacity(visible.len()))
        .collect::<Vec<_>>();
    for point in visible {
        for (path, symmetric_point) in paths.iter_mut().zip(fourfold_symmetric_points(*point)) {
            path.push(symmetric_point);
        }
    }
    paths
}

fn comet_path_is_valid(
    visible: &[Point],
    initial_planets: &[Planet],
    angular_velocity: f64,
    spawn_step: u32,
    comet_ids: &std::collections::HashSet<u32>,
) -> bool {
    let mut static_planets = Vec::new();
    let mut orbiting_planets = Vec::new();
    for planet in initial_planets {
        if comet_ids.contains(&planet.id) {
            continue;
        }
        if is_orbiting(planet.position(), planet.radius) {
            orbiting_planets.push(planet);
        } else {
            static_planets.push(planet);
        }
    }

    for (index, point) in visible.iter().enumerate() {
        if distance(*point, Point::new(CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS {
            return false;
        }

        let sym_points = fourfold_symmetric_points(*point);

        for planet in &static_planets {
            for sym_point in sym_points {
                if distance(sym_point, planet.position()) < planet.radius + COMET_RADIUS + 0.5 {
                    return false;
                }
            }
        }

        let game_step = f64::from(spawn_step - 1 + index as u32);
        for planet in &orbiting_planets {
            let planet_position = orbit_position(planet.position(), angular_velocity, game_step);
            for sym_point in sym_points {
                if distance(sym_point, planet_position) < planet.radius + COMET_RADIUS {
                    return false;
                }
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::env::{reset_with_rng, step};
    use crate::rules_engine::state::{PlayerResult, ResetConfig, SimConfig, State};

    use serde::Deserialize;

    struct ScriptedRandom {
        ints: std::collections::VecDeque<i32>,
        floats: std::collections::VecDeque<f64>,
    }

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

    impl ScriptedRandom {
        fn new(ints: impl IntoIterator<Item = i32>, floats: impl IntoIterator<Item = f64>) -> Self {
            Self {
                ints: ints.into_iter().collect(),
                floats: floats.into_iter().collect(),
            }
        }
    }

    impl RandomSource for ScriptedRandom {
        fn randint(&mut self, low: i32, high: i32) -> i32 {
            let value = self.ints.pop_front().expect("scripted int");
            assert!(
                (low..=high).contains(&value),
                "scripted int {value} outside {low}..={high}"
            );
            value
        }

        fn uniform(&mut self, low: f64, high: f64) -> f64 {
            let unit = self.floats.pop_front().expect("scripted float");
            low + (high - low) * unit
        }
    }

    #[test]
    fn assign_home_planets_sets_two_player_q1_and_q4() {
        let radius = planet_radius(3);
        let mut planets = symmetric_planets(0, -1, 70.0, 70.0, radius, 20, 3);
        let mut rng = ScriptedRandom::new([0], []);

        assign_home_planets(&mut planets, 2, &mut rng);

        assert_eq!(planets[0].owner, 0);
        assert_eq!(planets[0].ships, 10);
        assert_eq!(planets[3].owner, 1);
        assert_eq!(planets[3].ships, 10);
        assert_eq!(planets[1].owner, -1);
        assert_eq!(planets[2].owner, -1);
    }

    #[test]
    fn assign_home_planets_selects_four_player_home_from_diagonal_groups() {
        let radius = planet_radius(2);
        let mut planets = symmetric_planets(0, -1, 95.0, 95.0, radius, 20, 2);
        planets.extend(symmetric_planets(4, -1, 75.0, 55.0, radius, 20, 2));
        planets.extend(symmetric_planets(8, -1, 70.0, 70.0, radius, 20, 2));
        let mut rng = ScriptedRandom::new([1], []);

        assign_home_planets(&mut planets, 4, &mut rng);

        assert_eq!(
            planets
                .iter()
                .filter(|planet| planet.owner != -1)
                .map(|planet| planet.id)
                .collect::<Vec<_>>(),
            vec![8, 9, 10, 11]
        );
    }

    #[test]
    fn spawn_comet_group_adds_placeholder_planets_and_metadata() {
        let mut planets = vec![planet(7, -1, 20.0, 20.0, 1.0, 5, 1)];
        let mut initial_planets = planets.clone();
        let mut comet_planet_ids = Vec::new();
        let paths = vec![vec![Point::new(1.0, 2.0)]; 4];

        let group = spawn_comet_group(
            &mut planets,
            &mut initial_planets,
            &mut comet_planet_ids,
            paths,
            11,
        );

        assert_eq!(group.planet_ids, vec![8, 9, 10, 11]);
        assert_eq!(comet_planet_ids, vec![8, 9, 10, 11]);
        assert_eq!(planets.last().expect("last comet").x, -99.0);
        assert_eq!(initial_planets.len(), 5);
    }

    #[test]
    fn generate_comet_paths_returns_four_symmetric_paths_without_obstacles() {
        let mut rng = RepeatingRandom { int: 1, float: 0.5 };

        let paths = generate_comet_paths(&[], 0.04, 50, &[], 4.0, &mut rng)
            .expect("deterministic comet path");

        assert_eq!(paths.len(), 4);
        assert!((5..=40).contains(&paths[0].len()));
        assert_eq!(paths[0].len(), paths[1].len());

        for (((q1, q2), q3), q4) in paths[0].iter().zip(&paths[1]).zip(&paths[2]).zip(&paths[3]) {
            assert_eq!(q2.x, BOARD_SIZE - q1.x);
            assert_eq!(q2.y, q1.y);
            assert_eq!(q3.x, q1.x);
            assert_eq!(q3.y, BOARD_SIZE - q1.y);
            assert_eq!(q4.x, BOARD_SIZE - q1.x);
            assert_eq!(q4.y, BOARD_SIZE - q1.y);
        }
    }

    #[test]
    fn generated_planets_have_reference_shape_invariants() {
        let mut rng = rand::rng();
        let planets = generate_planets(&mut rng);

        assert!(planets.len() >= MIN_PLANET_GROUPS as usize * 4);
        assert!(planets.len() <= MAX_PLANET_GROUPS as usize * 4);
        assert_eq!(planets.len() % 4, 0);
        assert!(planets
            .iter()
            .any(|planet| is_orbiting(planet.position(), planet.radius)));

        let static_groups = planets
            .chunks_exact(4)
            .filter(|group| !is_orbiting(group[0].position(), group[0].radius))
            .count();
        assert!(static_groups >= MIN_STATIC_GROUPS);

        for group in planets.chunks_exact(4) {
            let q1 = &group[0];
            assert_eq!(group[1].x, BOARD_SIZE - q1.x);
            assert_eq!(group[1].y, q1.y);
            assert_eq!(group[2].x, q1.x);
            assert_eq!(group[2].y, BOARD_SIZE - q1.y);
            assert_eq!(group[3].x, BOARD_SIZE - q1.x);
            assert_eq!(group[3].y, BOARD_SIZE - q1.y);
        }
    }

    #[derive(Debug, Deserialize)]
    struct GenerationFixture {
        planet_generation: PlanetGenerationFixture,
        comet_path_generation: CometPathGenerationFixture,
        reset_cases: Vec<ResetCaseFixture>,
        comet_path_cases: Vec<CometPathGenerationFixture>,
        comet_ship_cases: Vec<CometShipCaseFixture>,
        terminal_cases: TerminalCasesFixture,
    }

    #[derive(Debug, Deserialize)]
    struct PlanetGenerationFixture {
        random_calls: Vec<RandomCall>,
        planets: Vec<[f64; 7]>,
    }

    #[derive(Debug, Deserialize)]
    struct CometPathGenerationFixture {
        #[serde(default)]
        name: String,
        inputs: CometPathInputs,
        initial_planets: Vec<[f64; 7]>,
        random_calls: Vec<RandomCall>,
        paths: Vec<Vec<[f64; 2]>>,
    }

    #[derive(Debug, Deserialize)]
    struct ResetCaseFixture {
        players: usize,
        random_calls: Vec<RandomCall>,
        state: ResetStateFixture,
    }

    #[derive(Debug, Deserialize)]
    struct ResetStateFixture {
        angular_velocity: f64,
        planets: Vec<[f64; 7]>,
        initial_planets: Vec<[f64; 7]>,
        fleets: Vec<[f64; 7]>,
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
    struct CometShipCaseFixture {
        random_calls: Vec<RandomCall>,
        ships: i32,
    }

    #[derive(Debug, Deserialize)]
    struct TerminalCasesFixture {
        no_op_tie: TerminalCaseFixture,
    }

    #[derive(Debug, Deserialize)]
    struct TerminalCaseFixture {
        players: usize,
        configuration: TerminalConfigFixture,
        before: ResetStateFixture,
        rewards: Vec<i32>,
        statuses: Vec<String>,
    }

    #[derive(Debug, Deserialize)]
    struct TerminalConfigFixture {
        #[serde(rename = "episodeSteps")]
        episode_steps: u32,
        #[serde(rename = "shipSpeed")]
        ship_speed: f64,
        #[serde(rename = "cometSpeed")]
        comet_speed: f64,
    }

    #[derive(Debug, Deserialize)]
    struct CometPathInputs {
        angular_velocity: f64,
        spawn_step: u32,
        comet_planet_ids: Vec<u32>,
        comet_speed: f64,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(tag = "kind")]
    enum RandomCall {
        #[serde(rename = "randint")]
        Randint { low: i32, high: i32, value: i32 },
        #[serde(rename = "uniform")]
        Uniform { low: f64, high: f64, value: f64 },
    }

    struct FixtureRandom {
        calls: std::collections::VecDeque<RandomCall>,
    }

    impl FixtureRandom {
        fn new(calls: Vec<RandomCall>) -> Self {
            Self {
                calls: calls.into_iter().collect(),
            }
        }

        fn assert_finished(&self) {
            assert!(
                self.calls.is_empty(),
                "{} fixture random calls were not consumed",
                self.calls.len()
            );
        }
    }

    impl RandomSource for FixtureRandom {
        fn randint(&mut self, low: i32, high: i32) -> i32 {
            let call = self.calls.pop_front().expect("fixture random call");
            let RandomCall::Randint {
                low: expected_low,
                high: expected_high,
                value,
            } = call
            else {
                panic!("expected randint fixture call");
            };
            assert_eq!(low, expected_low);
            assert_eq!(high, expected_high);
            value
        }

        fn uniform(&mut self, low: f64, high: f64) -> f64 {
            let call = self.calls.pop_front().expect("fixture random call");
            let RandomCall::Uniform {
                low: expected_low,
                high: expected_high,
                value,
            } = call
            else {
                panic!("expected uniform fixture call");
            };
            close(low, expected_low);
            close(high, expected_high);
            value
        }
    }

    const GENERATION_FIXTURE_PATH: &str = "tests/fixtures/generation/reference_generation.json";

    #[test]
    fn generate_planets_matches_python_reference_fixture() -> Result<(), Box<dyn std::error::Error>>
    {
        let Some(fixture) = reference_fixture()? else {
            return Ok(());
        };
        let mut rng = FixtureRandom::new(fixture.planet_generation.random_calls);

        let planets = generate_planets(&mut rng);

        rng.assert_finished();
        compare_planets(&planets, &fixture.planet_generation.planets);
        Ok(())
    }

    #[test]
    fn generate_comet_paths_matches_python_reference_fixture(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(fixture) = reference_fixture()? else {
            return Ok(());
        };
        check_comet_path_case(&fixture.comet_path_generation);
        Ok(())
    }

    #[test]
    fn reset_cases_match_python_reference_fixtures() -> Result<(), Box<dyn std::error::Error>> {
        let Some(fixture) = reference_fixture()? else {
            return Ok(());
        };

        for case in fixture.reset_cases {
            let mut rng = FixtureRandom::new(case.random_calls);
            let state = reset_with_rng(
                ResetConfig {
                    sim: SimConfig::new(case.players),
                    step: None,
                    angular_velocity: None,
                    planets: None,
                    initial_planets: None,
                },
                &mut rng,
            );

            rng.assert_finished();
            compare_reset_state(&state, &case.state);
        }
        Ok(())
    }

    #[test]
    fn comet_path_cases_match_python_reference_fixtures() -> Result<(), Box<dyn std::error::Error>>
    {
        let Some(fixture) = reference_fixture()? else {
            return Ok(());
        };

        for case in &fixture.comet_path_cases {
            check_comet_path_case(case);
        }
        Ok(())
    }

    #[test]
    fn comet_ship_sampling_matches_python_reference_fixtures(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(fixture) = reference_fixture()? else {
            return Ok(());
        };

        for case in fixture.comet_ship_cases {
            let mut rng = FixtureRandom::new(case.random_calls);
            let ships = sample_comet_ships(&mut rng);

            rng.assert_finished();
            assert_eq!(ships, case.ships);
        }
        Ok(())
    }

    #[test]
    fn no_op_terminal_tie_matches_python_reference_fixture(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(fixture) = reference_fixture()? else {
            return Ok(());
        };
        let case = fixture.terminal_cases.no_op_tie;
        let mut state = State {
            config: SimConfig {
                player_count: case.players,
                episode_steps: case.configuration.episode_steps,
                ship_speed: case.configuration.ship_speed,
                comet_speed: case.configuration.comet_speed,
            },
            step: case.before.step,
            angular_velocity: case.before.angular_velocity,
            planets: case.before.planets.iter().map(planet_from_array).collect(),
            initial_planets: case
                .before
                .initial_planets
                .iter()
                .map(planet_from_array)
                .collect(),
            fleets: Vec::new(),
            next_fleet_id: case.before.next_fleet_id,
            comets: Vec::new(),
            comet_planet_ids: case.before.comet_planet_ids,
        };

        let result = step(&mut state, &vec![Vec::new(); case.players]);

        assert_eq!(case.statuses, vec!["DONE"; case.players]);
        assert_eq!(
            result.player_results,
            case.rewards
                .into_iter()
                .map(player_result_from_reward)
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    fn check_comet_path_case(case: &CometPathGenerationFixture) {
        let mut rng = FixtureRandom::new(case.random_calls.clone());
        let initial_planets = case
            .initial_planets
            .iter()
            .map(planet_from_array)
            .collect::<Vec<_>>();
        let inputs = &case.inputs;

        let paths = generate_comet_paths(
            &initial_planets,
            inputs.angular_velocity,
            inputs.spawn_step,
            &inputs.comet_planet_ids,
            inputs.comet_speed,
            &mut rng,
        )
        .unwrap_or_else(|| panic!("{} fixture comet paths", case.name));

        rng.assert_finished();
        compare_paths(&paths, &case.paths);
    }

    fn reference_fixture() -> Result<Option<GenerationFixture>, Box<dyn std::error::Error>> {
        let contents = match std::fs::read_to_string(GENERATION_FIXTURE_PATH) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                warn_or_fail_missing_generation_fixture()?;
                return Ok(None);
            },
            Err(error) => return Err(error.into()),
        };
        Ok(Some(serde_json::from_str(&contents)?))
    }

    fn warn_or_fail_missing_generation_fixture() -> Result<(), Box<dyn std::error::Error>> {
        if require_parity_fixtures()? {
            let message = format!(
                "No generation parity fixture found at {GENERATION_FIXTURE_PATH}. \
                Run scripts/regenerate_test_fixtures.sh to enable generation parity, \
                or set REQUIRE_PARITY_FIXTURES=0 to skip generation parity."
            );
            Err(message.into())
        } else {
            Ok(())
        }
    }

    fn require_parity_fixtures() -> Result<bool, Box<dyn std::error::Error>> {
        let Ok(value) = std::env::var("REQUIRE_PARITY_FIXTURES") else {
            return Ok(true);
        };
        match value.to_ascii_lowercase().as_str() {
            "1" | "true" => Ok(true),
            "0" | "false" => Ok(false),
            _ => Err(
                format!("REQUIRE_PARITY_FIXTURES must be 1/true or 0/false, got {value:?}").into(),
            ),
        }
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

    fn compare_reset_state(actual: &State, expected: &ResetStateFixture) {
        assert_eq!(actual.step, expected.step);
        close(actual.angular_velocity, expected.angular_velocity);
        compare_planets(&actual.planets, &expected.planets);
        compare_planets(&actual.initial_planets, &expected.initial_planets);
        assert_eq!(actual.fleets.len(), expected.fleets.len());
        assert_eq!(actual.next_fleet_id, expected.next_fleet_id);
        assert_eq!(actual.comet_planet_ids, expected.comet_planet_ids);
        assert_eq!(actual.comets.len(), expected.comets.len());
        for (actual, expected) in actual.comets.iter().zip(&expected.comets) {
            assert_eq!(actual.planet_ids, expected.planet_ids);
            assert_eq!(actual.path_index, expected.path_index);
            compare_paths(&actual.paths, &expected.paths);
        }
    }

    fn player_result_from_reward(reward: i32) -> PlayerResult {
        match reward {
            1 => PlayerResult::Win,
            -1 => PlayerResult::Loss,
            _ => PlayerResult::NotDone,
        }
    }

    fn compare_planets(actual: &[Planet], expected: &[[f64; 7]]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.id, expected[0] as u32);
            assert_eq!(actual.owner, expected[1] as i32);
            close(actual.x, expected[2]);
            close(actual.y, expected[3]);
            close(actual.radius, expected[4]);
            assert_eq!(actual.ships, expected[5] as i32);
            assert_eq!(actual.production, expected[6] as i32);
        }
    }

    fn compare_paths(actual: &[Vec<Point>], expected: &[Vec<[f64; 2]>]) {
        assert_eq!(actual.len(), expected.len());
        for (actual_path, expected_path) in actual.iter().zip(expected) {
            assert_eq!(actual_path.len(), expected_path.len());
            for (actual, expected) in actual_path.iter().zip(expected_path) {
                close(actual.x, expected[0]);
                close(actual.y, expected[1]);
            }
        }
    }

    fn close(actual: f64, expected: f64) {
        let tolerance = 1e-9_f64.max(1e-9 * actual.abs().max(expected.abs()));
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}
