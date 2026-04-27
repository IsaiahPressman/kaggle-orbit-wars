use std::collections::{HashMap, HashSet};

use crate::rules_engine::env::PlayerAction;
use crate::rules_engine::state::{LaunchAction, Planet, State};

use super::{ACTION_ENTITY_SLOTS, MAX_COMETS, MAX_PLANETS};

pub(super) fn decode_pure_actions(
    state: &State,
    launch: &[bool],
    angle: &[f32],
    ships: &[i64],
    max_per_planet_launches: usize,
) -> Vec<PlayerAction> {
    let entities = action_entity_slots(state);
    let mut actions = vec![Vec::new(); state.config.player_count];
    for (player, player_actions) in actions.iter_mut().enumerate() {
        let player_offset = player * ACTION_ENTITY_SLOTS * max_per_planet_launches;
        for (entity_index, planet) in entities.iter().enumerate() {
            let Some(planet) = planet else {
                continue;
            };
            let entity_offset = player_offset + entity_index * max_per_planet_launches;
            for launch_index in 0..max_per_planet_launches {
                let action_index = entity_offset + launch_index;
                if !launch[action_index] {
                    break;
                }
                let ship_count = ships[action_index];
                assert!(
                    ship_count >= 1,
                    "pure action ships must be >= 1 when launch is true"
                );
                player_actions.push(LaunchAction {
                    from_planet_id: planet.id,
                    angle: f64::from(angle[action_index]),
                    ships: ship_count
                        .try_into()
                        .expect("pure action ships must fit in i32"),
                });
            }
        }
    }
    actions
}

pub(super) fn encode_action_spec(state: &State, can_act: &mut [bool], max_launch: &mut [i64]) {
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
        let index = player * ACTION_ENTITY_SLOTS + entity_index;
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
    #[should_panic(expected = "pure action ships must be >= 1 when launch is true")]
    fn pure_launch_panics_when_ship_count_is_zero() {
        let mut launch = vec![false; 4 * ACTION_ENTITY_SLOTS];
        let angle = vec![0.0; 4 * ACTION_ENTITY_SLOTS];
        let ships = vec![0; 4 * ACTION_ENTITY_SLOTS];
        launch[0] = true;

        decode_pure_actions(&one_planet_state(), &launch, &angle, &ships, 1);
    }

    #[test]
    fn pure_launch_emits_multiple_actions_until_first_false_slot() {
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
            &launch,
            &angle,
            &ships,
            max_per_planet_launches,
        );

        assert_eq!(actions[0].len(), 2);
        assert_eq!(actions[0][0].ships, 2);
        assert_eq!(actions[0][1].ships, 3);
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
