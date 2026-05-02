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

4. `just prepare` should run without errors. If mapped docs are genuinely still
   current after a small code change, rerun it as `DOCS_CURRENT=1 just prepare`
   to acknowledge that review.

## Containerized builds

See `docs/containerization.md` for building a Docker image with the locked Python
dependencies, Rust toolchain, compiled extension module, and example Slurm launch
patterns.

## Orbit Wars reference

The Rust rules engine targets the installed `kaggle-environments` Orbit Wars
implementation. Resolve the local module and gameplay prose paths with:

```sh
uv run python -c 'from importlib import import_module; from pathlib import Path; m = import_module("kaggle_environments.envs.orbit_wars.orbit_wars"); print(Path(m.__file__).resolve()); print(Path(m.__file__).with_name("README.md").resolve())'
```

## PPO training configs

Training presets live in `configs/`:

- `baseline.yaml`: vanilla PPO with larger rollout/minibatch sizing, scheduled
  learning rates, disabled periodic checkpoints, `torch.compile` default mode, and
  bfloat16 autocast.
- `pufferish.yaml`: enables Puffer-style V-trace recomputation and
  advantage-prioritized segment sampling without changing the core Python
  defaults.
- `model/stateless_transformer_5m.yaml`: shared stateless transformer model
  config used by the training presets.

The training entrypoint configures PyTorch for TF32 matmul/conv precision and
cuDNN benchmarking before constructing the environment, model, and optimizer.
PPO supports both `pure` and `discrete_targets` action specs when the
`StatelessTransformerV1` actor discriminator matches the environment action
spec. The current discrete-target actor requires `max_per_planet_launches: 1`.

Run a preset with:

```sh
uv run python scripts/run_ppo.py configs/baseline.yaml runs --log-mode debug --max-env-steps 16
```

PPO checkpoints save model, optimizer, scheduler, config, and environment-step
metadata. They do not save the Rust environment state or current observation, so
they are not exact resume snapshots.

Training logs terminal environment metrics under `train/` when episodes finish
during a rollout, including game length, per-player win rates, launch density,
planet occupancy for 2-player and 4-player games, max-entity overflow counts,
terminal ship counts, planet captures, launch and fleet-size statistics,
full-length game rate, and fleet/ship losses in the sun or out of bounds.

## Orbit Wars replay parity

Replay parity tests use compact Kaggle episode transition fixtures. The
`replay-*.jsonl` files are intentionally ignored by Git because full episodes can
be large.

The current reference episodes are:

- `75601099` (4-player game)
- `75598045` (2-player 500-step game)

If fixture files are missing, download them directly into the test fixture
directory:

```sh
scripts/regenerate_test_fixtures.sh
```

The regeneration script requires Kaggle API credentials configured for the local
user.

### Replay parity workflow

The fixture shape is JSONL with one row per transition: episode id, player
count, step, normalized numeric player action triples from
`steps[t][player].action`, per-player Kaggle `status` and `reward`, the input
observation from `steps[t - 1][0].observation`, and the expected state from
`steps[t][0].observation`. `cargo test` discovers all `replay-*.jsonl` files in
the fixture directory and fails if none are present.

Supported test environment variables:

- `ORBIT_WARS_PARITY_FIXTURE_DIR`: directory containing extracted JSONL parity
  fixtures.
- `REQUIRE_PARITY_FIXTURES=0`: skip replay and generation parity when fixtures
  are missing. Missing fixtures fail by default.

When the upstream rules change, keep the test code stable: download replacement
episodes as JSONL fixtures, move them into the fixture directory if needed,
update the reference episode id list in this README and
`docs/rules-engine.md`, and run the parity tests against the fixture
directory.

## Generation parity

Map generation, reset/home assignment, comet path generation, and comet ship
sampling are checked against fixtures produced by the Python reference
implementation under recorded random streams. Regenerate those fixtures after
intentional upstream rule changes:

```sh
uv run python scripts/generate_reference_fixtures.py
```

The generated fixture is written to
`tests/fixtures/generation/reference_generation.json` and ignored by Git.

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
installed Python environment. Replay tests also validate the documented
reference episode player counts and row counts, so update the required coverage
in `src/rules_engine/replay_tests.rs` when replacing the episode set.

3. Update the documented episode IDs in this README and
   `docs/rules-engine.md`.

4. Run the full checks with the new fixtures present:

```sh
just prepare
```

5. Fix any failing Rust parity tests by matching the updated Python behavior,
   then rerun `just prepare`.

Generation and replay fixtures remain ignored by Git; keep only the episode IDs
and setup commands in source control.
