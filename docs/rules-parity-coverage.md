# Rules Parity Coverage

This document is the system of record for Orbit Wars parity coverage. Keep it
updated whenever the Python reference, fixture generators, or Rust rules engine
changes.

## Covered By Replay Fixtures

Replay fixtures are generated with `scripts/download_replays.py` and loaded from
`tests/fixtures/orbit_wars_replays/replay-*.jsonl`.

The parity tests fail by default when required fixtures are missing. Set
`REQUIRE_PARITY_FIXTURES=0` to skip fixture-backed parity on machines that do not
have local fixtures.

Replay coverage is required for:

- `75601099`: 4 players, 141 recorded transitions.
- `75598045`: 2 players, 499 recorded transitions.

The replay parity test checks each transition against the Python reference for:

- step counter
- per-player Kaggle status/reward, mapped to active, won, or lost. The replay
  check requires exact status parity once Kaggle marks every player `DONE`.
  Before that global terminal row, Kaggle can keep eliminated players `ACTIVE`
  while Rust intentionally reports them as `Lost`, so the test only verifies
  that Rust has not ended the game early.
- angular velocity
- planets: id, owner, position, radius, ships, production
- initial planets
- fleets: id, owner, position, angle, source planet, ships
- next fleet id
- comet planet ids
- comet groups: planet ids, full paths, path index

Auxiliary Rust `StepResult` counters for fleets and ships lost in the sun, out
of bounds, or combat resolution, and for planets and comet planets captured,
are not fields in the Kaggle rows. They are covered by focused Rust unit tests
rather than replay parity assertions.

The current local replay set covers both 2-player and 4-player games, launches,
production, fleet movement, collisions, captures, comet movement and expiry,
and step-limit termination.

## Covered By Generation Fixtures

Generation fixtures are produced by `scripts/generate_reference_fixtures.py`
from the installed `kaggle-environments` Orbit Wars implementation and written
to `tests/fixtures/generation/reference_generation.json`. Rust consumes the
recorded random call stream and compares generated output.

The generated fixture currently covers:

- planet generation from seed `42`
- current Python-reference random static and fill phases, including the
  reference fourfold symmetry ordering
- full reset for 2-player and 4-player games, including angular velocity,
  planet generation, initial planets, and current random-group home assignment
- comet paths at spawn steps `50`, `150`, `250`, `350`, and `450`
- comet path generation with existing comet ids excluded
- comet path generation where failed attempts occur before success
- comet ship sampling

## Covered By Unit Tests

Rust unit tests cover focused rules behavior that is hard to isolate from full
replays:

- geometry helpers
- fleet speed curve
- launch validation and side effects
- production order
- fleet movement and removal
- fleet collision priority: planet collisions before out-of-bounds or sun removal
- sun, planet, out-of-bounds, and sweep collisions
- combat resolution, ties, and reinforcement
- comet spawning before same-step movement
- comet movement and expiry
- terminal score ties where all tied players win

## Known Boundaries

Replay parity intentionally injects comet paths and comet ships from the
expected observation. It also explicitly injects "skip spawn" on comet spawn
steps where the replay has no new comet, so replay parity never falls through to
RNG-backed comet generation. This isolates step parity from random generation.
Comet generation parity is covered separately by generation fixtures.

The Rust simulator receives typed actions and fails fast on invalid actions.
Kaggle/Python action parsing is outside the inner simulator API. The replay
harness mirrors Python's accepted-action filtering only to turn historical
Kaggle replay actions into typed Rust actions.

Floating-point parity uses close comparisons rather than bit-for-bit equality.
Discrete ids, owners, ship counts, production, removals, and player statuses
must match exactly.

The Rust state stores planets and initial planets in ID-indexed slots. Parity
comparison iterates live slots in ID order, which matches generated and fixture
planet ordering because fixture IDs are unique and contiguous for live planets.
Manual duplicate planet IDs and IDs at/above `MAX_PLANET_ID` are intentionally
rejected before parity comparison.

Replay and generation fixtures are ignored by Git because full episodes and
recorded reference streams can be large. A fresh checkout must run
`scripts/regenerate_test_fixtures.sh` or restore fixtures from cache before
running required parity. Use `REQUIRE_PARITY_FIXTURES=0 just rs-test` only when
intentionally skipping fixture-backed parity.

RL-only terminal metrics such as `ships_lost_in_combat_per_game` and
`fleets_lost_in_combat_per_game` are derived from simulator step results and
are covered by focused Rust/Python metric tests, not by replay fixture parity
assertions.
