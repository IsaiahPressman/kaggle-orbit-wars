# OWL: Orbit Wars (reinforcement) Learning

## Setup instructions

1. Install prerequisites:
    * [Rust](https://www.rust-lang.org/tools/install)
    * [uv](https://docs.astral.sh/uv/getting-started/installation/)
    * [just](https://github.com/casey/just)
2. Add nightly toolchain for `rustfmt`:

```sh
rustup update nightly
rustup component add rustfmt --toolchain nightly
```

3. `just prepare` should run without errors

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
uv run python scripts/download_replays.py 75373897 75377525 --save-dir tests/fixtures/orbit_wars_replays
```

The downloader requires Kaggle API credentials configured for the local user.

### Replay parity workflow

The fixture shape is JSONL with one row per transition: episode id, step, typed
player actions from `steps[t][player].action`, the input observation from
`steps[t - 1][0].observation`, and the expected state from
`steps[t][0].observation`. `cargo test` skips replay parity when fixtures are
absent and runs it when matching fixture files exist.

If you already have raw Kaggle `replay-*.json` files, convert them with:

```sh
uv run python scripts/extract_replay_fixtures.py --fixture-dir tests/fixtures/orbit_wars_replays
```

Recommended test environment variables:

- `ORBIT_WARS_PARITY_FIXTURE_DIR`: directory containing extracted JSONL parity
  fixtures.
- `ORBIT_WARS_PARITY_EPISODES`: optional comma-separated episode ids to run, for
  example `75373897,75377525`.

When the upstream rules change, keep the test code stable: download replacement
episodes as JSONL fixtures, move them into the fixture directory if needed,
update the reference episode id list in this README and
`docs/rules-engine-plan.md`, and run the parity tests against the new fixture
directory.

## Generation parity

Map generation and comet path generation are checked against fixtures produced
by the Python reference implementation under recorded random streams. Regenerate
those fixtures after intentional upstream rule changes:

```sh
uv run python scripts/generate_reference_fixtures.py
```

The generated fixture is small and checked in at
`tests/fixtures/generation/reference_generation.json`.

## Updating tests after Python rule changes

When `orbit_wars.py` or the official Orbit Wars environment changes, update the
Rust parity tests in this order:

1. Replace the local reference files if needed:

```sh
# update orbit_wars.py and orbit_wars_rules.md first
uv run python scripts/generate_reference_fixtures.py
```

2. Download fresh replay fixtures from games produced by the updated
   environment:

```sh
uv run python scripts/download_replays.py NEW_EPISODE_ID_1 NEW_EPISODE_ID_2 --save-dir tests/fixtures/orbit_wars_replays
```

3. Update the documented episode IDs in this README and
   `docs/rules-engine-plan.md`. If changing the default replay set, also update
   `DEFAULT_EPISODES` in `src/rules_engine/replay_tests.rs`.

4. Run the full checks with the new fixtures present:

```sh
just prepare
```

5. Fix any failing Rust parity tests by matching the updated Python behavior,
   then rerun `just prepare`.

The checked-in generation fixture should be committed when it changes. The
replay JSONL fixtures remain ignored by Git; keep only the episode IDs and setup
commands in source control.
