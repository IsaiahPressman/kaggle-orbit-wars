use std::collections::{HashMap, HashSet};

use crate::rules_engine::env::PlayerAction;
use crate::rules_engine::state::{LaunchAction, Planet, State};

use super::{PlayerMap, ACTION_ENTITY_SLOTS, MAX_COMETS, MAX_PLANETS, OUTER_PLAYER_SLOTS};

pub(super) fn decode_pure_actions(
    state: &State,
    player_map: &PlayerMap,
    launch: &[bool],
    angle: &[f32],
    ships: &[i64],
    max_per_planet_launches: usize,
) -> Result<Vec<PlayerAction>, String> {
    let entities = action_entity_slots(state);
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
                let Some(planet) = planet else {
                    return Err(format!(
                        "player {outer_player} cannot launch from empty action entity slot {entity_index}"
                    ));
                };
                let ship_count = ships[action_index];
                if ship_count < 1 {
                    return Err(format!(
                        "player {outer_player} entity slot {entity_index} launch {launch_index} ships must be >= 1"
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

pub(super) fn encode_action_spec(
    state: &State,
    player_map: &PlayerMap,
    can_act: &mut [bool],
    max_launch: &mut [i64],
) {
    for (entity_index, planet) in action_entity_slots(state).iter().enumerate() {
        let Some(planet) = planet else {
            continue;
        };
        if planet.ships < 1 || planet.owner < 0 {
            continue;
        }
        let player = planet.owner as usize;
        if player >= state.config.player_count {
            continue;
        }
        let index = player_map.internal_to_outer(player) * ACTION_ENTITY_SLOTS + entity_index;
        can_act[index] = true;
        max_launch[index] = i64::from(planet.ships);
    }
}

fn action_entity_slots(state: &State) -> Vec<Option<&Planet>> {
    let comet_ids = state
        .comet_planet_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut entities = vec![None; ACTION_ENTITY_SLOTS];
    for (entity_index, planet) in state
        .planets
        .iter()
        .filter(|planet| !comet_ids.contains(&planet.id))
        .take(MAX_PLANETS)
        .enumerate()
    {
        entities[entity_index] = Some(planet);
    }

    let planets_by_id = state
        .planets
        .iter()
        .map(|planet| (planet.id, planet))
        .collect::<HashMap<_, _>>();
    let mut comet_index = 0;
    for group in &state.comets {
        for planet_id in &group.planet_ids {
            if comet_index >= MAX_COMETS {
                return entities;
            }
            if let Some(planet) = planets_by_id.get(planet_id) {
                entities[MAX_PLANETS + comet_index] = Some(*planet);
                comet_index += 1;
            }
        }
    }
    entities
}

#[cfg(test)]
mod tests {
    use super::*;
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
            initial_planets: planets.clone(),
            planets,
            fleets: Vec::new(),
            next_fleet_id: 0,
            comets: Vec::new(),
            comet_planet_ids: Vec::new(),
        }
    }

    #[test]
    fn pure_launch_errors_when_ship_count_is_zero() {
        let player_map = PlayerMap::identity();
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;

        let err = decode_pure_actions(&one_planet_state(), &player_map, &launch, &angle, &ships, 1)
            .expect_err("zero ships should fail");

        assert!(err.contains("ships must be >= 1"));
    }

    #[test]
    fn pure_launch_errors_when_ship_count_exceeds_i32() {
        let player_map = PlayerMap::identity();
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        ships[0] = i64::from(i32::MAX) + 1;

        let err = decode_pure_actions(&one_planet_state(), &player_map, &launch, &angle, &ships, 1)
            .expect_err("oversized ships should fail");

        assert!(err.contains("ships must fit in i32"));
    }

    #[test]
    fn pure_launch_errors_when_angle_is_not_finite() {
        let player_map = PlayerMap::identity();
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let mut angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;
        angle[0] = f32::INFINITY;
        ships[0] = 1;

        let err = decode_pure_actions(&one_planet_state(), &player_map, &launch, &angle, &ships, 1)
            .expect_err("non-finite angle should fail");

        assert!(err.contains("angle must be finite"));
    }

    #[test]
    fn pure_launch_errors_when_player_does_not_own_source() {
        let player_map = PlayerMap::identity();
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[ACTION_ENTITY_SLOTS] = true;
        ships[ACTION_ENTITY_SLOTS] = 1;

        let err = decode_pure_actions(&one_planet_state(), &player_map, &launch, &angle, &ships, 1)
            .expect_err("wrong owner should fail");

        assert!(err.contains("player 1 cannot launch from planet 7 owned by 0"));
    }

    #[test]
    fn pure_launch_errors_when_total_launches_exceed_source_ships() {
        let player_map = PlayerMap::identity();
        let max_per_planet_launches = 2;
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        launch[0] = true;
        ships[0] = 6;
        launch[1] = true;
        ships[1] = 5;

        let err = decode_pure_actions(
            &one_planet_state(),
            &player_map,
            &launch,
            &angle,
            &ships,
            max_per_planet_launches,
        )
        .expect_err("overspending should fail");

        assert!(err.contains("planet 7 has 10 ships, cannot launch 11"));
    }

    #[test]
    fn pure_launch_emits_multiple_actions_until_first_false_slot() {
        let player_map = PlayerMap::identity();
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
            &one_planet_state(),
            &player_map,
            &launch,
            &angle,
            &ships,
            max_per_planet_launches,
        )
        .expect("valid actions should decode");

        assert_eq!(actions[0].len(), 2);
        assert_eq!(actions[0][0].ships, 2);
        assert_eq!(actions[0][1].ships, 3);
    }

    #[test]
    fn pure_launch_decodes_from_remapped_outer_player_slot() {
        let player_map = PlayerMap::from_outer_slots(2, [3, 1, 0, 2]);
        let max_per_planet_launches = 1;
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let mut ships = vec![0; 4 * ACTION_ENTITY_SLOTS * max_per_planet_launches];
        let outer_player = 3;
        let action_index = outer_player * ACTION_ENTITY_SLOTS * max_per_planet_launches;
        launch[action_index] = true;
        ships[action_index] = 4;

        let actions = decode_pure_actions(
            &one_planet_state(),
            &player_map,
            &launch,
            &angle,
            &ships,
            max_per_planet_launches,
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

        assert_eq!(slots[0].map(|planet| planet.id), Some(7));
        assert!(slots[1..MAX_PLANETS].iter().all(Option::is_none));
        assert_eq!(slots[MAX_PLANETS].map(|planet| planet.id), Some(8));
    }
}
