# OWL: Orbit Wars (reinforcement) Learning

## Setup instructions

1. Install prerequisites:
   - [Rust](https://www.rust-lang.org/tools/install)
   - [uv](https://docs.astral.sh/uv/getting-started/installation/)
   - [just](https://github.com/casey/just)
2. Add nightly toolchain for `rustfmt`:

```sh
rustup update nightly
rustup component add rustfmt --toolchain nightly
```

3. Generate replay fixtures:

```sh
scripts/regenerate_test_fixtures.sh
```

4. `just prepare` should run without errors

## Orbit Wars reference

The Rust rules engine targets the installed `kaggle-environments` Orbit Wars
implementation. Resolve the local module and gameplay prose paths with:

```sh
uv run python -c 'from importlib import import_module; from pathlib import Path; m = import_module("kaggle_environments.envs.orbit_wars.orbit_wars"); print(Path(m.__file__).resolve()); print(Path(m.__file__).with_name("README.md").resolve())'
```

## Orbit Wars replay parity

Replay parity tests use compact Kaggle episode transition fixtures. The
`replay-*.jsonl` files are intentionally ignored by Git because full episodes can
be large.

The current reference episodes are:

- `75373897`
- `75377525`

If fixture files are missing, download them directly into the test fixture
directory:

```sh
scripts/regenerate_test_fixtures.sh
```

The regeneration script requires Kaggle API credentials configured for the local
user.

### Replay parity workflow

The fixture shape is JSONL with one row per transition: episode id, step, typed
player actions from `steps[t][player].action`, the input observation from
`steps[t - 1][0].observation`, and the expected state from
`steps[t][0].observation`. `cargo test` discovers all `replay-*.jsonl` files in
the fixture directory and fails if none are present.

Supported test environment variables:

- `ORBIT_WARS_PARITY_FIXTURE_DIR`: directory containing extracted JSONL parity
  fixtures.
- `ORBIT_WARS_REQUIRE_PARITY_FIXTURES=0`: skip replay parity when replay
  fixtures are missing. Missing fixtures fail by default.

When the upstream rules change, keep the test code stable: download replacement
episodes as JSONL fixtures, move them into the fixture directory if needed,
update the reference episode id list in this README and
`docs/rules-engine-plan.md`, and run the parity tests against the fixture
directory.

## Generation parity

Map generation, reset/home assignment, comet path generation, and comet ship
sampling are checked against fixtures produced by the Python reference
implementation under recorded random streams. Regenerate those fixtures after
intentional upstream rule changes:

```sh
uv run python scripts/generate_reference_fixtures.py
```

The generated fixture is small and checked in at
`tests/fixtures/generation/reference_generation.json`.

## Updating tests after Python rule changes

When the official Orbit Wars environment changes, update the Rust parity tests
in this order:

1. Update the installed `kaggle-environments` package to the latest version.
2. Regenerate all test fixtures. With no arguments, the script uses the current
   reference episodes listed above:

```sh
scripts/regenerate_test_fixtures.sh
```

To switch replay episodes, pass the replacement Kaggle episode IDs:

```sh
scripts/regenerate_test_fixtures.sh NEW_EPISODE_ID_1 NEW_EPISODE_ID_2
```

The script removes outdated `replay-*.jsonl` files before downloading the new
set, and rewrites `tests/fixtures/generation/reference_generation.json` from the
installed Python environment.

3. Update the documented episode IDs in this README and
   `docs/rules-engine-plan.md`.

4. Run the full checks with the new fixtures present:

```sh
just prepare
```

5. Fix any failing Rust parity tests by matching the updated Python behavior,
   then rerun `just prepare`.

The checked-in generation fixture should be committed when it changes. The
replay JSONL fixtures remain ignored by Git; keep only the episode IDs and setup
commands in source control.
