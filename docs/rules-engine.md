# Rust Rules Engine Reference

This document is the current map for the Rust Orbit Wars simulator. The Python
reference is the installed `kaggle_environments.envs.orbit_wars.orbit_wars`
module. Resolve the exact local module path and gameplay prose path with:

```sh
uv run python -c 'from importlib import import_module; from pathlib import Path; m = import_module("kaggle_environments.envs.orbit_wars.orbit_wars"); print(Path(m.__file__).resolve()); print(Path(m.__file__).with_name("README.md").resolve())'
```

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

## Public API

The core Rust API is intentionally small:

```rust
pub fn reset(config: ResetConfig) -> State;
pub fn step(state: &mut State, actions: &[PlayerAction]) -> StepResult;
```

`State` owns planets, fleets, comet metadata, the current step, the player
count, generation constants, and ids needed for deterministic progression.

`StepResult` returns one Kaggle-facing result per actual player: active, won, or
lost. This matches the actual player count without making 2-player games carry
ignored entries. Nonterminal eliminated players remain active here to match the
Python reference `status`; the Python/RL adapter derives early loss/done signals
from post-step alive flags before widening results to fixed outer player slots
for tensor observations, rewards, and dones.

## Current Status

Implemented:

- Rust state/action/config/result types.
- Reset and procedural generation with injectable random sources.
- Turn stepping in Python reference order.
- Focused Rust unit tests for rules components.
- Generation parity over ignored Python-reference fixtures.
- Replay parity over ignored Kaggle JSONL fixtures.
- Python RL observation/action wrappers and vectorized environment.

Open follow-up work:

- Benchmarks and data-structure optimization for training throughput.
- Mechanical doc freshness checks beyond `docs/pr-checklist.md`.
- CI-owned parity fixture cache or checked-in minimal parity fixtures.

## Rules-Change Workflow

For rules changes, work in this order:

1. Update or add parity/unit tests that state the expected behavior.
2. Change the Rust simulator or fixture generator.
3. Update this reference and `docs/rules-parity-coverage.md` in the same change.
4. Run `just rs-prepare` with parity fixtures present. Use
   `REQUIRE_PARITY_FIXTURES=0 just rs-test` only when intentionally skipping
   fixture-backed parity.
5. Use a reviewer pass to compare behavior against the Python reference and call
   out drift.

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
- Planet generation follows the current Python reference phases directly:
  random static groups first, then random fill groups until the target count and
  at least one orbiting group are present.
- Home assignment picks any symmetric group for both 2-player and 4-player
  games; current fixtures no longer require a y=x diagonal group.
- Action validation and launch side effects.
- Production.
- Fleet movement, out-of-bounds removal, sun collision, planet collision.
- Rotating planet and comet sweep collisions.
- Combat resolution, including tied attackers and same-owner reinforcement.
- Termination and scoring.

Replay parity tests:

- Download Kaggle replays directly into compact JSONL fixtures containing
  normalized numeric action triples, per-player Kaggle status/reward, and
  post-step reference observations.
- Use `steps[t - 1][0].observation` as the transition input, actions from
  `steps[t][player].action`, and `steps[t][0].observation` as the canonical
  expected state.
- Keep replay fixture downloads out of Git. Download `replay-<episode-id>.jsonl`
  files to the fixture directory or repo root, where `.gitignore` excludes them.
- `scripts/download_replays.py` writes one JSONL row per transition:
  `episode_id`, `players`, `step`, normalized per-player action triples,
  per-player Kaggle `results`, the pre-step observation, and the canonical
  post-step player 0 observation.
- `scripts/regenerate_test_fixtures.sh` removes outdated replay fixtures,
  regenerates the Python generation fixture, and downloads the selected replay
  fixture set.
- Keep fixture files out of Git and make tests print the regeneration command
  when required fixtures are missing.
- Replay parity tests discover all `replay-*.jsonl` files in
  `ORBIT_WARS_PARITY_FIXTURE_DIR`, or
  `tests/fixtures/orbit_wars_replays` by default. If no fixtures are present,
  the test fails by default. Set `REQUIRE_PARITY_FIXTURES=0` to skip
  fixture-backed parity. When rules change, download new Kaggle episodes as
  JSONL fixtures, update the episode id list below, and leave the test code
  unchanged unless the fixture schema itself changes.
- Replay parity validates the required documented coverage set in
  `src/rules_engine/replay_tests.rs`: episode id, player count, and transition
  row count must match the list below so coverage cannot silently shrink.

The current downloaded reference episodes are:

- `75601099`: 4-player, 141 recorded transitions.
- `75598045`: 2-player, 499 recorded transitions.

## Maintenance Rules

- Treat this file as a current-state map. When a listed implementation step is
  completed, move it into `Current Status` or remove it.
- Any rules-engine change should update this file and
  `docs/rules-parity-coverage.md`, or explicitly state why no docs changed in
  the PR checklist.
- Regenerate Python-reference generation fixtures with
  `scripts/regenerate_test_fixtures.sh` when upstream generation changes.
- Run `just rs-prepare` after Rust edits and `just py-prepare` after Python
  edits.

## Known Risk Areas

- Python `random` parity is intentionally not required from integer seed alone.
- Python silently ignores malformed Kaggle actions; Rust should not mirror this
  at the typed simulator boundary.
- Comets are inserted as planets at off-board placeholder positions and expire
  both before launches and immediately after movement.
- Planet movement uses `initial_planets` as the orbital anchor, not last turn's
  position.
- Four-player home assignment chooses among any symmetric group, using the
  reference RNG stream after planet generation.
- Combat is queued during fleet movement and sweep, then resolved after all
  movement.
- Fleet movement queues planet collisions before checking out-of-bounds or sun
  removal, matching the reference behavior for fast fleets that cross multiple
  collision/removal zones in one step.
- Termination happens at `episodeSteps - 2`, which is earlier than the prose
  rule's 500-turn wording suggests.
- Player 0 observations are the canonical replay observations. Later player
  observations may omit `step`.
