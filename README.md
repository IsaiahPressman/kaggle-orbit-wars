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

## Kaggle submission build

Build a Kaggle-compatible image from the current worktree and create
`artifacts/submission.tar.gz` with:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt
```

Pass a submission name to write `artifacts/<name>.tar.gz` instead:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt my-run
```

Pass a quantization format as the third argument to quantize the packaged
model weights during submission generation:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt my-run fp4
```

Pass an optional fallback checkpoint after the existing arguments to package a
second, faster model:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt my-run fp4 \
  --fallback-checkpoint runs/20260501-090000/checkpoint_last_best.pt
```

The submission recipe runs `just prepare`, rebuilds the `orbit-wars:kaggle`
image with Buildx zstd layer compression, compiles the Rust extension inside
Kaggle's Python image, and packages `python/owl`, `python/main.py` or `main.py`,
the requested model bundle, and its adjacent `config.yaml` under
`models/primary/`. If a fallback checkpoint is provided, its model bundle and
adjacent `config.yaml` are packaged under `models/fallback/`.
Rebuilding the image during submission generation keeps the packaged Python code
aligned with the current checkout. The packaged checkpoint keeps the original
checkpoint contents only for the model weights needed by the Kaggle agent. To
store packaged models below fp32 precision, pass a quantization format such as
`fp8_e4m3fn` or `fp4_e2m1fn_x2_scaled_block16`; unique prefixes such as `fp4`
are accepted. The default `fp32` leaves checkpoint weights unchanged. The Kaggle
agent dequantizes quantized slim checkpoints back to fp32 before loading the
model. Set `fallback_min_overage_time` in `python/owl/agent/agent_config.yaml`
to switch to the fallback model when remaining overage time drops below that
threshold; `null` disables fallback routing even if the fallback model is
packaged.

## Orbit Wars reference

The Rust rules engine targets the installed `kaggle-environments` Orbit Wars
implementation. Resolve the local module and gameplay prose paths with:

```sh
uv run python -c 'from importlib import import_module; from pathlib import Path; m = import_module("kaggle_environments.envs.orbit_wars.orbit_wars"); print(Path(m.__file__).resolve()); print(Path(m.__file__).with_name("README.md").resolve())'
```

## PPO training configs

Training presets live in `configs/`:

- `baseline.yaml`: vanilla PPO with the 20m stateless transformer preset,
  discrete-target actions, `max_entities=256`, one PPO epoch per rollout,
  larger rollout/minibatch sizing, Muon/AdamW optimizer rates, periodic
  checkpoints every 20M environment steps, `torch.compile` default mode, and
  bfloat16 autocast by default.
- `baseline_adam.yaml`: Adam optimizer variant with explicit optimizer
  settings, including `1e-4` learning rate, `(0.9, 0.999)` betas, `1e-5`
  epsilon, no weight decay, and the same warmup/cosine scheduler shape.
- `baseline_adamw.yaml`: AdamW optimizer variant matching `baseline_adam.yaml`
  except for decoupled `0.01` weight decay.
- `model/stateless_transformer_20m_swiglu.yaml`: larger stateless transformer
  model config used by `baseline.yaml`, with an inline discrete-target actor
  override using eight action mixtures.
- `model/stateless_transformer_5m_gelu.yaml` and
  `model/stateless_transformer_20m_gelu.yaml`: GELU variants of the stateless
  transformer presets.
- `model/stateless_transformer_28m.yaml`: larger SwiGLU stateless transformer
  preset with a discrete-target actor.

The training entrypoint configures PyTorch for TF32 matmul/conv precision and
cuDNN benchmarking before constructing the environment, model, and optimizer.
Fresh launches explicitly reset model parameters before optimizer construction.
When `--load-model-weights` is set, the fresh trainer then replaces those
parameters from the checkpoint without loading optimizer state.
resume launches load checkpoint weights and optimizer state without resetting
the model first.
Optimizer configs may set `lr_schedule.schedule` to
`linear_warmup_cosine_decay` for warmup followed by cosine decay, or `cosine`
for a repeating LambdaLR multiplier that moves from `1.0` to `lr_min_ratio`
halfway through `full_cycle_steps` optimizer steps, then back to `1.0` at the
end of the cycle. The
schedule name selects the accepted scheduler fields; unrelated scheduler fields
are rejected.
Training `EnvConfig.n_envs` must be even. PPO updates run
`rl.ppo_epochs` full-shuffle passes over rollout segments, grouped by
`rl.segments_per_minibatch`; set `rl.gradient_accumulation_steps` above `1` to
accumulate multiple minibatches before each optimizer step. `EnvConfig.n_envs`
must be divisible by
`rl.segments_per_minibatch * rl.gradient_accumulation_steps`. In distributed PPO
launches, `EnvConfig.n_envs`, rollout horizon, minibatch segment width, and
gradient accumulation are per GPU. Checkpoint cadence, `--max-env-steps`, W&B
step values, and `train/env_steps` are counted across all ranks. Resume launches
with a different GPU count derive an equivalent per-rank config by scaling
`env.n_envs` and
`rl.segments_per_minibatch * rl.gradient_accumulation_steps` by
`saved_gpus / current_gpus`. The derived config keeps
`rl.segments_per_minibatch` at or below the saved value, so the per-minibatch
training batch does not increase; resume fails if the scaled values are
fractional or config-invalid.
`rl.ppo_clip_mode` defaults to `per_player`, which clips the summed per-player
joint action log-probability. Set it to `per_entity` to clip each controllable
action entity independently before summing those clipped policy-loss terms back
to the player-step.
PPO supports `pure`, `discrete_targets`, and `discrete_target_bins` action specs
when the `StatelessTransformerV1` actor discriminator matches the environment
action spec. The current discrete-target actor requires
`max_per_planet_launches: 1`; the target-bin actor requires matching `n_bins`.
Both discrete target specs default to `targeting_mode: full_mask`; set
`stop_bad_launch` or `anything_goes` to expose loose target masks while
controlling whether sun-crossing decoded launches are replaced with no-ops.

Run a preset with:

```sh
uv run python scripts/run_ppo.py configs/baseline.yaml runs --log-mode debug --max-env-steps 16
```

Fresh launches accept `-o`/`--overrides field.path=value`; when provided, rank 0
prints the flattened override list before loading the config.
Set `rl.dtype=float8` to enable FP8 training with torchao. FP8 mode converts
eligible internal `torch.nn.Linear` layers to torchao `Float8Linear` before
optimizer construction, keeps model input/output projections and shape-ineligible
linears in higher precision, and still wraps forward/eval calls in bfloat16
autocast. FP8 requires CUDA. The default `rl.fp8_recipe: rowwise` is the
stability-oriented starting point; use `tensorwise` for the fastest recipe, or
`rowwise_with_gw_hp` if gradient-weight numerics need a more conservative path.
For example:

```sh
uv run python scripts/run_ppo.py configs/baseline.yaml runs \
  --log-mode debug --max-env-steps 16 -o rl.dtype=float8
```

Fresh launches can also initialize the model from an existing full training
checkpoint without resuming the optimizer, scheduler, config, or W&B run:

```sh
uv run python scripts/run_ppo.py configs/baseline.yaml runs \
  --load-model-weights runs/20260505-120000/checkpoint_last_best.pt
```

This loads only `checkpoint["model"]` plus the `env_steps`,
`player_step_total`, and `total_games_played` logging counters. Optimizer steps,
target-KL counters, optimizer state, scheduler state, and checkpoint config are
fresh for the new run.

PPO run directories save `config.yaml` alongside checkpoints. The saved config
includes `runtime.n_runtime_gpus`; resume uses it to keep the effective rollout
and optimizer-step batch shape equivalent when the current launch uses a
different number of ranks, and fails when no exact derived config exists.
Checkpoints save model, optimizer, scheduler, environment-step metadata,
optimizer-step metadata, player-step metadata, plus the W&B run ID used for
resume. Resume training by passing either a run directory or a checkpoint file
as the only positional path:

```sh
uv run python scripts/run_ppo.py runs/20260505-120000
uv run python scripts/run_ppo.py runs/20260505-120000/checkpoint_00_020_000_000.pt
```

Directory resume loads `checkpoint_final.pt` when present, otherwise the latest
numbered checkpoint, and never treats `checkpoint_last_best.pt` as the primary
training checkpoint. File resume loads `config.yaml` from the checkpoint's
parent directory. Both resume modes require the associated
`checkpoint_last_best.pt` and W&B logging so the saved run ID can be resumed.
Checkpoints do not save the Rust environment state or current observation, so
resumed runs continue from a fresh environment batch rather than acting as exact
simulator snapshots. Periodic checkpoint names use grouped zero-padded
environment-step labels such as `checkpoint_00_022_000_000.pt`. At each
periodic checkpoint, the
current model is evaluated against the last-best model snapshot using
sampled policy actions across `env.n_envs` games using the configured
`env.two_player_weight`, with current and last-best seats randomly shuffled
across active player slots for each eval game, and logs
`eval/win_rate_against_last_best` plus terminal environment metrics under
`eval/`. When the current model reaches at least 70% eval win rate, the
last-best snapshot is replaced and also saved as `checkpoint_last_best.pt`.
Set `rl.eval_replay_games` to a positive count to save random eval replay
samples from the weighted eval game set under
`eval_replays/<checkpoint-name>/` in the run directory. The sampled game
ordinals are selected up front rather than taking the first games to finish.
Each sampled eval game is written as its own JSONL file.

Training logs terminal environment metrics under `train/` when episodes finish
during a rollout, including game length, per-player win rates, launch density,
planet occupancy for 2-player and 4-player games, max-entity overflow counts,
terminal ship counts, completed game counts, planet captures, launch and fleet-size statistics,
neutral planet/comet undershot rates, full-length game rate, cumulative active
player-step totals, and fleet/ship losses in combat, the sun, or out of bounds.
Planet occupancy is reported at terminal as
`train/terminal_planet_occupancy_rate_2p` and
`train/terminal_planet_occupancy_rate_4p`.
Policy logs include total entropy plus policy-specific component means such as
`policy/launch_entropy`, `policy/target_entropy`,
`policy/fleet_size_full_entropy`, and `policy/angle_and_size_entropy`.

## Replay capture

`scripts/benchmark_checkpoints.py` can save replay JSONL samples with
`--save-replay-games N`, split across 2-player and 4-player benchmark games
according to `--two-player-weight`. Files are written under `--replay-dir`,
defaulting to `replays/benchmark_checkpoints`.
Each sampled benchmark game is written as its own JSONL file.
Open `tools/orbit_wars_replay_viewer.html` in a browser and choose
a saved `.jsonl` file or Kaggle episode replay `.json` file to play back a
game.

Replay rows contain raw Rust environment snapshots for one completed game:
board constants, step/config values, outer-slot owner IDs, player maps, action
entity slots, planets, fleets, comets, rewards, dones, and model assignments.
The Python replay recorder samples game ordinals randomly before rollout and
uses Rust terminal snapshots captured before vectorized env auto-reset.

## Orbit Wars replay parity

Replay parity tests use compact Kaggle episode transition fixtures. The
`replay-*.jsonl` files are intentionally ignored by Git because full episodes can
be large.

The current reference episodes are:

- `75930761` (2-player game)
- `75926553` (4-player game)

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
- `REQUIRE_PARITY_FIXTURES=0`: skip replay parity, and skip generation parity
  only when generation fixtures are missing. Missing fixtures fail by default.

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
