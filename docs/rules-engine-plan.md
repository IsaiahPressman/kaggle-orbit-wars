# Rust Rules Engine Plan

This document is the working map for implementing the Rust Orbit Wars simulator.
The Python reference is the installed
`kaggle_environments.envs.orbit_wars.orbit_wars` module; gameplay prose lives
in `orbit_wars_rules.md`.

## Assumptions

- The Rust simulator is the inner rules API. It receives typed actions and fails
  fast on invalid API inputs.
- Python/Kaggle-compatible action parsing stays outside the simulator.
- Floating-point state uses `f64`. Parity tests compare floats with
  `math.isclose`-style tolerances, while ids, owners, ship counts, removals, and
  termination state must match exactly.
- Procedural generation does not need to match from the same integer seed across
  Python and Rust RNGs. It should match when driven by the same stream of random
  integers/floats.
- The engine supports both 2-player and 4-player games from the start.

## Target API

The first stable Rust API should stay small:

```rust
pub fn reset(config: ResetConfig) -> State;
pub fn step(state: &mut State, actions: &[PlayerAction]) -> StepResult;
```

`State` owns planets, fleets, comet metadata, the current step, the player
count, generation constants, and ids needed for deterministic progression.

`StepResult` should return one terminal result per player: not done, loss, or
win. This matches the actual player count without making 2-player games carry
ignored entries. The Python/RL adapter can widen this to a fixed tensor shape
later if that is more convenient.

## Agentic Workflow

Use focused agents for separable work:

1. Planner/spec writer: keep this plan and parity checklist current.
2. Test writer: add unit and replay tests before simulator implementation.
3. Implementation writer: port one rules component at a time.
4. Reviewer: compare behavior against the Python reference and call out drift.
5. Optimizer: add benchmarks and improve data structures only after parity is
   covered.

Human review should focus on acceptance criteria and rule interpretation. Agents
should own implementation, test updates, and documentation corrections.

## Test Strategy

Start with component tests:

- Geometry: distance and point-to-segment distance.
- Fleet speed curve.
- Planet generation helpers driven by an injectable random source.
- Comet path generation driven by an injectable random source.
- Python-reference generation fixtures for planet and comet generation, so Rust
  must consume the same recorded random calls and match Python's generated
  outputs.
- Action validation and launch side effects.
- Production.
- Fleet movement, out-of-bounds removal, sun collision, planet collision.
- Rotating planet and comet sweep collisions.
- Combat resolution, including tied attackers and same-owner reinforcement.
- Termination and scoring.

Then add replay parity tests:

- Download Kaggle replays directly into compact JSONL fixtures containing typed
  actions and post-step reference observations.
- Use `steps[t - 1][0].observation` as the transition input, actions from
  `steps[t][player].action`, and `steps[t][0].observation` as the canonical
  expected state.
- Keep replay fixture downloads out of Git. Download `replay-<episode-id>.jsonl`
  files to the fixture directory or repo root, where `.gitignore` excludes them.
- `scripts/download_replays.py` writes one JSONL row per transition:
  `episode_id`, `step`, typed per-player actions, the pre-step observation, and
  the canonical post-step player 0 observation.
- `scripts/regenerate_test_fixtures.sh` removes outdated replay fixtures,
  regenerates the Python generation fixture, and downloads the selected replay
  fixture set.
- Keep checked-in fixture files compact. If they become too large, keep them out
  of Git too and make the test print the extraction command when the fixture is
  missing.
- Replay parity tests discover all `replay-*.jsonl` files in
  `ORBIT_WARS_PARITY_FIXTURE_DIR`, or
  `tests/fixtures/orbit_wars_replays` by default, and fail if no fixtures are
  present. When rules change, download new Kaggle episodes as JSONL fixtures,
  update the episode id list below, and leave the test code unchanged unless the
  fixture schema itself changes.

The current downloaded reference episodes are:

- `75373897`: 4-player, 500 recorded steps.
- `75377525`: 2-player, 296 recorded steps.

## Implementation Order

1. Add serializable fixture helpers around the Python reference.
2. Define Rust state/action/config/result types.
3. Port stateless math helpers and unit-test them.
4. Port reset/generation using an injectable RNG trait.
5. Port turn stepping in Python turn-order order.
6. Add replay parity integration tests.
7. Regenerate Python-reference generation fixtures with
   `scripts/regenerate_test_fixtures.sh` when upstream generation changes.
8. Run `just rs-prepare` after Rust edits and `just py-prepare` after Python
   edits.

## Known Risk Areas

- Python `random` parity is intentionally not required from integer seed alone.
- Python silently ignores malformed Kaggle actions; Rust should not mirror this
  at the typed simulator boundary.
- Comets are inserted as planets at off-board placeholder positions and expire
  both before launches and immediately after movement.
- Planet movement uses `initial_planets` as the orbital anchor, not last turn's
  position.
- Combat is queued during fleet movement and sweep, then resolved after all
  movement.
- Termination happens at `episodeSteps - 2`, which is earlier than the prose
  rule's 500-turn wording suggests.
- Player 0 observations are the canonical replay observations. Later player
  observations may omit `step`.
