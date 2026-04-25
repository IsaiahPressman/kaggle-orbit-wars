use super::state::{LaunchAction, ResetConfig, State, StepResult};

pub type PlayerAction = Vec<LaunchAction>;

pub fn reset(config: ResetConfig) -> State {
    let planets = config.planets.unwrap_or_default();
    let initial_planets = config.initial_planets.unwrap_or_else(|| planets.clone());

    State {
        config: config.sim,
        step: 0,
        angular_velocity: config.angular_velocity.unwrap_or(0.0),
        planets,
        initial_planets,
        fleets: Vec::new(),
        next_fleet_id: 0,
        comets: Vec::new(),
        comet_planet_ids: Vec::new(),
    }
}

pub fn step(state: &mut State, actions: &[PlayerAction]) -> StepResult {
    assert_eq!(
        actions.len(),
        state.config.player_count,
        "step requires actions for every player"
    );

    StepResult {
        done: vec![false; state.config.player_count],
    }
}
