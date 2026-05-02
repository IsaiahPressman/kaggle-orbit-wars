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
pub const MAX_PLANET_ID: u32 = 100;
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlanetVector {
    slots: Vec<Option<Planet>>,
    len: usize,
}

impl PlanetVector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_planets(planets: Vec<Planet>) -> Self {
        let mut vector = Self::new();
        for planet in planets {
            vector.push(planet);
        }
        vector
    }

    pub fn push(&mut self, planet: Planet) {
        assert!(
            planet.id < MAX_PLANET_ID,
            "planet id must be < {MAX_PLANET_ID}, got {}",
            planet.id
        );
        let index = planet.id as usize;
        if index >= self.slots.len() {
            self.slots.resize_with(index + 1, || None);
        }
        assert!(
            self.slots[index].is_none(),
            "duplicate planet id {}",
            planet.id
        );
        self.slots[index] = Some(planet);
        self.len += 1;
    }

    pub fn get(&self, id: u32) -> Option<&Planet> {
        self.slots.get(id as usize).and_then(Option::as_ref)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut Planet> {
        self.slots.get_mut(id as usize).and_then(Option::as_mut)
    }

    pub fn remove(&mut self, id: u32) -> Option<Planet> {
        let removed = self.slots.get_mut(id as usize)?.take();
        if removed.is_some() {
            self.len -= 1;
        }
        removed
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&Planet) -> bool) {
        for slot in &mut self.slots {
            if slot.as_ref().is_some_and(|planet| !keep(planet)) {
                *slot = None;
                self.len -= 1;
            }
        }
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.len = 0;
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Planet> {
        self.slots.iter().filter_map(Option::as_ref)
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut Planet> {
        self.slots.iter_mut().filter_map(Option::as_mut)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn slot_len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn last(&self) -> Option<&Planet> {
        self.iter().next_back()
    }

    pub fn next_planet_id(&self) -> u32 {
        self.last().map_or(1, |planet| planet.id + 1)
    }
}

impl From<Vec<Planet>> for PlanetVector {
    fn from(planets: Vec<Planet>) -> Self {
        Self::from_planets(planets)
    }
}

impl std::ops::Index<usize> for PlanetVector {
    type Output = Planet;

    fn index(&self, index: usize) -> &Self::Output {
        self.slots
            .get(index)
            .and_then(Option::as_ref)
            .unwrap_or_else(|| panic!("planet id {index} does not exist"))
    }
}

impl std::ops::IndexMut<usize> for PlanetVector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.slots
            .get_mut(index)
            .and_then(Option::as_mut)
            .unwrap_or_else(|| panic!("planet id {index} does not exist"))
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
    pub planets: PlanetVector,
    pub initial_planets: PlanetVector,
    pub fleets: Vec<Fleet>,
    pub next_fleet_id: u32,
    pub comets: Vec<CometGroup>,
    pub comet_planet_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepResult {
    pub player_results: Vec<PlayerResult>,
    pub fleet_losses: FleetLossStats,
    pub planets_captured: u32,
    pub comets_captured: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FleetLossStats {
    pub fleets_in_sun: u32,
    pub fleets_out_of_bounds: u32,
    pub ships_in_sun: i32,
    pub ships_out_of_bounds: i32,
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
