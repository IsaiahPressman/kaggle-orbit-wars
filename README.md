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

## Orbit Wars replays

Replay parity tests use Kaggle episode downloads as external fixtures. The raw
`replay-*.json` files are intentionally ignored by Git because they can be large.
Store raw downloads in the repo root, for example
`./replay-75373897.json`.

The current reference episodes are:

- `75373897`
- `75377525`

If these files are missing, download them into the repo root:

```sh
uv run python scripts/download_replays.py 75373897 75377525 --save-dir .
```

The downloader requires Kaggle API credentials configured for the local user.

### Replay parity workflow

Use raw downloads only as source material. Extract compact parity fixtures from
them before running simulator parity tests. The intended fixture shape is JSONL
with one row per transition: episode id, step, typed player actions from
`steps[t][player].action`, the input observation from
`steps[t - 1][0].observation`, and the expected state from
`steps[t][0].observation`.

Recommended test environment variables:

- `ORBIT_WARS_REPLAY_DIR`: directory containing raw `replay-*.json` downloads.
  Default recommendation: repo root.
- `ORBIT_WARS_PARITY_FIXTURE_DIR`: directory containing extracted JSONL parity
  fixtures.
- `ORBIT_WARS_PARITY_EPISODES`: optional comma-separated episode ids to run, for
  example `75373897,75377525`.

When the upstream rules change, keep the test code stable: download replacement
episodes, regenerate the JSONL fixtures from those raw replays, update the
reference episode id list in this README and `docs/rules-engine-plan.md`, and
run the parity tests against the new fixture directory.
