# Rust Rules Engine Plan

This document is the working map for implementing the Rust Orbit Wars simulator.
The Python reference is `orbit_wars.py`; gameplay prose lives in
`orbit_wars_rules.md`.

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

`StepResult` should use `Vec<bool>` for per-player done flags. This matches the
actual player count without making 2-player games carry ignored entries. The
Python/RL adapter can widen this to `[bool; 4]` later if a fixed tensor shape is
more convenient.

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
- Action validation and launch side effects.
- Production.
- Fleet movement, out-of-bounds removal, sun collision, planet collision.
- Rotating planet and comet sweep collisions.
- Combat resolution, including tied attackers and same-owner reinforcement.
- Termination and scoring.

Then add replay parity tests:

- Convert Kaggle replay JSON into compact JSONL fixtures containing typed
  actions and post-step reference observations.
- Use `steps[t - 1][0].observation` as the transition input, actions from
  `steps[t][player].action`, and `steps[t][0].observation` as the canonical
  expected state.
- Keep replay downloads out of Git. Store only episode ids and fixture extraction
  instructions in docs/README.
- Make fixture paths configurable so updated rule replays can replace old ones
  without rewriting test code.

The current downloaded reference episodes are:

- `75373897`: 4-player, 500 recorded steps.
- `75377525`: 2-player, 296 recorded steps.

## Implementation Order

1. Add serializable fixture extraction helpers around the Python reference.
2. Define Rust state/action/config/result types.
3. Port stateless math helpers and unit-test them.
4. Port reset/generation using an injectable RNG trait.
5. Port turn stepping in Python turn-order order.
6. Add replay parity integration tests.
7. Run `just rs-prepare` after Rust edits and `just py-prepare` after Python
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
