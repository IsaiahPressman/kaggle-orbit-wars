# Rules Parity Coverage

This document is the system of record for Orbit Wars parity coverage. Keep it
updated whenever the Python reference, fixture generators, or Rust rules engine
changes.

## Covered By Replay Fixtures

Replay fixtures are generated with `scripts/download_replays.py` and loaded from
`tests/fixtures/orbit_wars_replays/replay-*.jsonl`.

The replay parity test fails by default when no replay fixtures are present. Set
`ORBIT_WARS_REQUIRE_PARITY_FIXTURES=0` to skip replay parity on machines that do
not have local fixtures.

The replay parity test checks each transition against the Python reference for:

- step counter
- per-player terminal result: not done, loss, or win
- angular velocity
- planets: id, owner, position, radius, ships, production
- initial planets
- fleets: id, owner, position, angle, source planet, ships
- next fleet id
- comet planet ids
- comet groups: planet ids, full paths, path index

The current local replay set covers both 2-player and 4-player games, launches,
production, fleet movement, collisions, captures, comet movement and expiry,
and step-limit termination.

## Covered By Generation Fixtures

Generation fixtures are produced by `scripts/generate_reference_fixtures.py`
from the installed `kaggle-environments` Orbit Wars implementation. Rust
consumes the recorded random call stream and compares generated output.

The checked-in fixture currently covers:

- planet generation from seed `42`
- full reset for 2-player and 4-player games, including angular velocity,
  planet generation, initial planets, and home assignment
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
- sun, planet, and sweep collisions
- combat resolution, ties, and reinforcement
- comet spawning before same-step movement
- comet movement and expiry
- terminal score ties where all tied players win

## Known Boundaries

Replay parity intentionally injects comet paths and comet ships from the
expected observation. This isolates step parity from random generation. Comet
generation parity is covered separately by generation fixtures.

The Rust simulator receives typed actions and fails fast on invalid actions.
Kaggle/Python action parsing is outside the inner simulator API. The replay
harness mirrors Python's accepted-action filtering only to turn historical
Kaggle replay actions into typed Rust actions.

Floating-point parity uses close comparisons rather than bit-for-bit equality.
Discrete ids, owners, ship counts, production, removals, and terminal results
must match exactly.

Replay fixtures are ignored by Git because full episodes can be large. A fresh
checkout must run `scripts/regenerate_test_fixtures.sh` or restore replay
fixtures from cache before running required replay parity. Use
`ORBIT_WARS_REQUIRE_PARITY_FIXTURES=0 just rs-test` only when intentionally
skipping replay parity.
