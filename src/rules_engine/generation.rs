use rand::RngExt;

use super::state::{
    CometGroup, Planet, Point, BOARD_SIZE, CENTER, COMET_PRODUCTION, COMET_RADIUS,
    MAX_PLANET_GROUPS, MIN_PLANET_GROUPS, MIN_STATIC_GROUPS, PLANET_CLEARANCE,
    ROTATION_RADIUS_LIMIT, SUN_RADIUS,
};
use super::utils::distance;

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

pub fn is_orbiting(planet: &Planet) -> bool {
    distance(planet.position(), Point::new(CENTER, CENTER)) + planet.radius < ROTATION_RADIUS_LIMIT
}

pub fn generate_planets(rng: &mut impl RandomSource) -> Vec<Planet> {
    let mut planets = Vec::new();
    let num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS) as usize;
    let mut next_id = 0;

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

    for _ in 0..1000 {
        let production = rng.randint(1, 5);
        let radius = planet_radius(production);
        let min_orbital = SUN_RADIUS + radius + 10.0;
        let max_orbital = ROTATION_RADIUS_LIMIT - radius;
        if min_orbital >= max_orbital {
            continue;
        }

        let orbital_radius = rng.uniform(min_orbital, max_orbital);
        let x = CENTER + orbital_radius * std::f64::consts::FRAC_1_SQRT_2;
        let y = CENTER + orbital_radius * std::f64::consts::FRAC_1_SQRT_2;
        let ships = rng.randint(5, 99).min(rng.randint(5, 99));
        let group = symmetric_planets(next_id, -1, x, y, radius, ships, production);

        if valid_diagonal_orbiting_group(&group, &planets) {
            planets.extend(group);
            next_id += 4;
            break;
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

        if orbital_radius + radius >= ROTATION_RADIUS_LIMIT
            && (x + radius > BOARD_SIZE
                || x - radius < 0.0
                || y + radius > BOARD_SIZE
                || y - radius < 0.0)
        {
            continue;
        }

        let ships = rng.randint(5, 30);
        let group = symmetric_planets(next_id, -1, x, y, radius, ships, production);
        if valid_fill_group(&group, &planets) {
            if orbital_radius + radius < ROTATION_RADIUS_LIMIT {
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

    let mut home_group = rng.randint(0, num_groups as i32 - 1) as usize;
    let mut base = home_group * 4;

    if player_count == 4 && is_orbiting(&planets[base]) {
        for group_index in 0..num_groups {
            let group_base = group_index * 4;
            let planet = &planets[group_base];
            if is_orbiting(planet) && ((planet.x - CENTER) - (planet.y - CENTER)).abs() < 0.01 {
                home_group = group_index;
                base = home_group * 4;
                break;
            }
        }
    }

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
    vec![
        planet(id, owner, x, y, radius, ships, production),
        planet(id + 1, owner, BOARD_SIZE - x, y, radius, ships, production),
        planet(id + 2, owner, x, BOARD_SIZE - y, radius, ships, production),
        planet(
            id + 3,
            owner,
            BOARD_SIZE - x,
            BOARD_SIZE - y,
            radius,
            ships,
            production,
        ),
    ]
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

fn valid_diagonal_orbiting_group(group: &[Planet], planets: &[Planet]) -> bool {
    group.iter().all(|candidate| {
        let candidate_orbital = distance(candidate.position(), Point::new(CENTER, CENTER));
        planets.iter().all(|planet| {
            let planet_orbital = distance(planet.position(), Point::new(CENTER, CENTER));
            let planet_static = planet_orbital + planet.radius >= ROTATION_RADIUS_LIMIT;

            distance(candidate.position(), planet.position())
                >= candidate.radius + planet.radius + PLANET_CLEARANCE
                && (!planet_static
                    || (candidate_orbital - planet_orbital).abs()
                        >= candidate.radius + planet.radius + PLANET_CLEARANCE)
        })
    })
}

fn valid_fill_group(group: &[Planet], planets: &[Planet]) -> bool {
    group.iter().all(|candidate| {
        let candidate_orbital = distance(candidate.position(), Point::new(CENTER, CENTER));
        let candidate_rotating = candidate_orbital + candidate.radius < ROTATION_RADIUS_LIMIT;

        planets.iter().all(|planet| {
            let planet_orbital = distance(planet.position(), Point::new(CENTER, CENTER));
            let planet_rotating = planet_orbital + planet.radius < ROTATION_RADIUS_LIMIT;

            distance(candidate.position(), planet.position())
                >= candidate.radius + planet.radius + PLANET_CLEARANCE
                && (candidate_rotating == planet_rotating
                    || (candidate_orbital - planet_orbital).abs()
                        >= candidate.radius + planet.radius + PLANET_CLEARANCE)
        })
    })
}

fn symmetric_paths(visible: &[Point]) -> Vec<Vec<Point>> {
    vec![
        visible.to_vec(),
        visible
            .iter()
            .map(|point| Point::new(BOARD_SIZE - point.x, point.y))
            .collect(),
        visible
            .iter()
            .map(|point| Point::new(point.x, BOARD_SIZE - point.y))
            .collect(),
        visible
            .iter()
            .map(|point| Point::new(BOARD_SIZE - point.x, BOARD_SIZE - point.y))
            .collect(),
    ]
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
        if is_orbiting(planet) {
            orbiting_planets.push(planet);
        } else {
            static_planets.push(planet);
        }
    }

    for (index, point) in visible.iter().enumerate() {
        if distance(*point, Point::new(CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS {
            return false;
        }

        let sym_points = [
            *point,
            Point::new(BOARD_SIZE - point.x, point.y),
            Point::new(point.x, BOARD_SIZE - point.y),
            Point::new(BOARD_SIZE - point.x, BOARD_SIZE - point.y),
        ];

        for planet in &static_planets {
            for sym_point in sym_points {
                if distance(sym_point, planet.position()) < planet.radius + COMET_RADIUS + 0.5 {
                    return false;
                }
            }
        }

        let game_step = f64::from(spawn_step - 1 + index as u32);
        for planet in &orbiting_planets {
            let dx = planet.x - CENTER;
            let dy = planet.y - CENTER;
            let orbital_radius = (dx.powi(2) + dy.powi(2)).sqrt();
            let initial_angle = dy.atan2(dx);
            let current_angle = initial_angle + angular_velocity * game_step;
            let planet_position = Point::new(
                CENTER + orbital_radius * current_angle.cos(),
                CENTER + orbital_radius * current_angle.sin(),
            );
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

    struct ScriptedRandom {
        ints: std::collections::VecDeque<i32>,
        floats: std::collections::VecDeque<f64>,
    }

    struct RepeatingRandom {
        int: i32,
        float: f64,
    }

    struct CyclingRandom {
        ints: Vec<i32>,
        floats: Vec<f64>,
        int_index: usize,
        float_index: usize,
    }

    impl CyclingRandom {
        fn new(ints: Vec<i32>, floats: Vec<f64>) -> Self {
            Self {
                ints,
                floats,
                int_index: 0,
                float_index: 0,
            }
        }
    }

    impl RandomSource for CyclingRandom {
        fn randint(&mut self, low: i32, high: i32) -> i32 {
            let raw = self.ints[self.int_index % self.ints.len()];
            self.int_index += 1;
            low + raw.rem_euclid(high - low + 1)
        }

        fn uniform(&mut self, low: f64, high: f64) -> f64 {
            let unit = self.floats[self.float_index % self.floats.len()];
            self.float_index += 1;
            low + (high - low) * unit
        }
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
    fn assign_home_planets_redirects_four_player_orbiting_home_to_diagonal_group() {
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
        assert!(planets.iter().any(is_orbiting));

        let static_groups = planets
            .chunks_exact(4)
            .filter(|group| !is_orbiting(&group[0]))
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

    #[test]
    fn generate_planets_matches_scripted_random_stream_snapshot() {
        let mut rng = CyclingRandom::new(
            vec![5, 3, 11, 17, 4, 23, 29, 2, 31, 37, 5, 13, 19],
            vec![0.13, 0.61, 0.27, 0.79, 0.42, 0.91, 0.18, 0.54],
        );

        let planets = generate_planets(&mut rng);
        let fingerprint = planets
            .chunks_exact(4)
            .map(|group| {
                let q1 = &group[0];
                format!(
                    "{}:{:.6}:{:.6}:{:.6}:{}:{}",
                    q1.id, q1.x, q1.y, q1.radius, q1.ships, q1.production
                )
            })
            .collect::<Vec<_>>()
            .join("|");

        assert_eq!(
            fingerprint,
            "0:97.227887:59.780425:2.386294:16:4|4:96.996720:86.454393:2.098612:36:3|8:97.408089:71.405573:1.693147:9:2|12:73.321659:73.321659:2.386294:7:4"
        );
    }

    #[test]
    fn generate_comet_paths_matches_scripted_random_stream_snapshot() {
        let mut rng = RepeatingRandom { int: 1, float: 0.5 };

        let paths = generate_comet_paths(&[], 0.04, 50, &[], 4.0, &mut rng)
            .expect("deterministic comet path");
        let first = paths[0].first().expect("first point");
        let last = paths[0].last().expect("last point");
        let fingerprint = format!(
            "{}:{:.6}:{:.6}:{:.6}:{:.6}",
            paths[0].len(),
            first.x,
            first.y,
            last.x,
            last.y
        );

        assert_eq!(fingerprint, "33:35.361619:99.499009:97.592785:34.587569");
    }
}
