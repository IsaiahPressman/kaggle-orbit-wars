pub const BOARD_SIZE: f64 = 100.0;
pub const CENTER: f64 = BOARD_SIZE / 2.0;
pub const SUN_RADIUS: f64 = 10.0;
pub const ROTATION_RADIUS_LIMIT: f64 = 50.0;
pub const COMET_RADIUS: f64 = 1.0;
pub const COMET_PRODUCTION: i32 = 1;
pub const PLANET_CLEARANCE: f64 = 7.0;
pub const MIN_PLANET_GROUPS: i32 = 5;
pub const MAX_PLANET_GROUPS: i32 = 10;
pub const MIN_STATIC_GROUPS: usize = 3;
pub const MAX_PLAYERS: usize = 4;
pub const COMET_SPAWN_STEPS: [u32; 5] = [50, 150, 250, 350, 450];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Planet {
    pub id: u32,
    pub owner: i32,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub ships: i32,
    pub production: i32,
}

impl Planet {
    pub const fn position(&self) -> Point {
        Point::new(self.x, self.y)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fleet {
    pub id: u32,
    pub owner: i32,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub from_planet_id: u32,
    pub ships: i32,
}

impl Fleet {
    pub const fn position(&self) -> Point {
        Point::new(self.x, self.y)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CometGroup {
    pub planet_ids: Vec<u32>,
    pub paths: Vec<Vec<Point>>,
    pub path_index: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LaunchAction {
    pub from_planet_id: u32,
    pub angle: f64,
    pub ships: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimConfig {
    pub player_count: usize,
    pub episode_steps: u32,
    pub ship_speed: f64,
    pub comet_speed: f64,
}

impl SimConfig {
    pub fn new(player_count: usize) -> Self {
        assert!(
            player_count == 2 || player_count == MAX_PLAYERS,
            "Orbit Wars supports exactly 2 or 4 players"
        );

        Self {
            player_count,
            episode_steps: 500,
            ship_speed: 6.0,
            comet_speed: 4.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResetConfig {
    pub sim: SimConfig,
    pub step: Option<u32>,
    pub angular_velocity: Option<f64>,
    pub planets: Option<Vec<Planet>>,
    pub initial_planets: Option<Vec<Planet>>,
}

impl ResetConfig {
    pub fn new(player_count: usize) -> Self {
        Self {
            sim: SimConfig::new(player_count),
            step: None,
            angular_velocity: None,
            planets: None,
            initial_planets: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub config: SimConfig,
    pub step: u32,
    pub angular_velocity: f64,
    pub planets: Vec<Planet>,
    pub initial_planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub next_fleet_id: u32,
    pub comets: Vec<CometGroup>,
    pub comet_planet_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepResult {
    pub player_results: Vec<PlayerResult>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerResult {
    Active,
    Won,
    Lost,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepInjections {
    pub comet_spawn: Option<CometSpawnInjection>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CometSpawnInjection {
    Spawn { paths: Vec<Vec<Point>>, ships: i32 },
    Skip,
}
