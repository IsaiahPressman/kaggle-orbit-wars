# Containerization

This repository can be built into an image that contains Python 3.12, `uv`, the
repo-pinned Rust toolchain, Cargo dependencies, Python dependencies, and the
compiled `maturin` extension.

For Kaggle submission builds, use `Dockerfile.kaggle`. It starts from Kaggle's
CPU Python image, verifies the competition Python/package versions, installs the
repo-pinned Rust toolchain, creates a uv build venv for `maturin`, compiles the
PyO3 extension in release mode with the same native CPU optimization used by
`just prepare-rl`, and packages the importable `owl` package plus the requested
model checkpoint and adjacent model config into `submission.tar.gz`. The
checkpoint is slimmed into a temporary file before packaging so the original
training checkpoint is not overwritten.

## When this helps

Containerizing makes sense for Slurm if the cluster has a supported container
runtime such as Apptainer/Singularity, Enroot/Pyxis, Shifter, or Docker with
NVIDIA Container Toolkit. Use the image as an immutable dependency and build
artifact, then bind-mount run outputs, caches, secrets, and large datasets at job
launch time.

Do not treat the image as the place where training output lives. Slurm jobs
should write checkpoints and logs to a mounted filesystem.

## Build

Build the default CUDA-capable Linux environment for a typical GPU Slurm cluster:

```sh
docker build --platform linux/amd64 -t orbit-wars:dev .
```

The explicit platform matters if you are building from a non-amd64 host. The
Dockerfile intentionally fails early for other platforms because the project
dependency markers select CUDA PyTorch wheels for Linux, and the Slurm target
should match the image architecture.

Building `linux/amd64` from an ARM host runs under emulation and can be slow or
fragile for native linking. Prefer building this image on a native amd64 machine
or in GitHub Actions, then pushing it to GHCR.

## CI image publishing

Pushes to `main` build and publish the image to GHCR with two tags:

- Docker/GHCR form: `ghcr.io/OWNER/REPO:main`
- Docker/GHCR form: `ghcr.io/OWNER/REPO:GIT_SHA`
- Pyxis/Enroot form: `ghcr.io#OWNER/REPO:GIT_SHA`

Use the SHA tag for reproducible Slurm jobs, and use `main` only when you
explicitly want the latest successful `main` image. For private repositories or
private packages, the cluster must have GHCR credentials available to
Enroot/Pyxis.
When substituting `OWNER` and `REPO`, use lowercase values. The GitHub Actions
workflow publishes lowercase GHCR names with `${GITHUB_REPOSITORY,,}`. For this
repo, use `ghcr.io#isaiahpressman/kaggle-orbit-wars:main`.

The CI build frees unused hosted-runner toolchains before Docker starts and runs
`uv sync` without uv's package cache. This avoids keeping a second unpacked copy
of the CUDA PyTorch dependency stack while the virtual environment is installed.

The Linux `uv` markers install the CUDA 12.8 PyTorch wheel from `pyproject.toml`.
At runtime, GPU access still depends on the host NVIDIA driver and the cluster
container runtime exposing the GPU devices into the container.

The image installs the Rust toolchain declared in `rust-toolchain.toml`, so local
and container builds use the same compiler channel.

Create a Kaggle submission tarball on the host:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt
```

The resulting file is `artifacts/submission.tar.gz`. Pass a submission name to
write `artifacts/<name>.tar.gz`, for example:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt my-run
```

Pass a quantization format as the third argument to quantize the packaged model
weights inside the submission image:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt my-run fp4
```

Pass an optional fallback checkpoint after the existing arguments to package a
second, faster model:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt my-run fp4 \
  --fallback-checkpoint runs/20260501-090000/checkpoint_last_best.pt
```

The `kaggle-submission` recipe depends on `just prepare`, then rebuilds
`orbit-wars:kaggle` from the current checkout before running the package script
in that image. The Kaggle image build uses the Buildx docker exporter with zstd
layer compression at the fastest level to reduce time spent in Docker's layer
export step. Rebuilding on each submission avoids packaging stale Python code
from a previous image build.

The package script expects `python/main.py` or `main.py` to exist and copies it
to `main.py` at the archive root. It also extracts `checkpoint["model"]` from
the requested checkpoint into a temporary file, copies that file to
`models/primary/checkpoint.pt`, optionally quantizes those weights when the
recipe quantization argument is not `fp32`, and copies `config.yaml` from the
same directory to `models/primary/config.yaml`. If `--fallback-checkpoint` is
provided, the fallback checkpoint and adjacent config are packaged under
`models/fallback/` using the same fixed filenames. Supported quantization
formats include `fp8_e4m3fn`, `fp4_e2m1fn_x2_scaled_block16`, and
`nf5_g128_lsq_policy_last_fp8`. Lower-bit normal-float formats `nf4_g128_lsq`,
`nf3_nf4_structured_3p5`, and `nf3_g128_lsq` are also supported; unique
quantization prefixes such as `fp4` are accepted. The extraction step validates
that fp32 model states contain only
string keys and tensor values, and custom quantized checkpoint payloads are
checked before packaging. The checked-in
`python/owl/agent/agent_config.yaml` configures `inference_quantization: int8`,
which converts loaded `nn.Linear` layers to PyTorch dynamic int8 CPU inference
while keeping final actor/critic output heads in fp32; `null` disables
serving-time quantization and uses fp32 inference. Quantized slim checkpoints
are stream-dequantized into the live model one tensor at a time, so agent
startup does not hold a complete fp32 dequantized state dict in addition to the
model. Set
`fallback_min_overage_time` in
`python/owl/agent/agent_config.yaml` to switch to the fallback model when
remaining overage time drops below that threshold; `null` disables fallback
routing even when the fallback model is packaged. A packaged fallback config is
validated during startup, but the fallback weights are loaded on the second
observed turn instead of initial agent construction. If remaining overage time
has already fallen below the fallback threshold before fallback weights are
loaded, the agent emits no actions instead of spending the remaining budget on a
first-time fallback load. The image build validates Kaggle-targeted Rust
compilation directly before artifact generation runs with the mounted
checkpoint directory.

The packaged agent's Rust observation encoder filters fleets smaller than the
configured `min_fleet_size` while encoding Kaggle observations. The tradeoff is
deliberate: tiny fleets are usually low-impact, but enough of them can
materially increase inference time and trigger fallback routing. The filter
keeps one safeguard for player liveness: when a player has no current planets
and all of their fleets are below the threshold, the encoder keeps that player's
largest fleet instead of dropping that player from the encoded state entirely.
Use `just kaggle-image` only when you want to rebuild or validate the Kaggle
image without creating a submission tarball.

The `flash-attn` extra is installed by default. Skip it only for machines where
CUDA compiler, PyTorch, or GPU architecture compatibility makes the image build
too brittle:

```sh
docker build --platform linux/amd64 --build-arg SKIP_FLASH_ATTN=1 \
  -t orbit-wars:no-flash-attn .
```

## GHCR access from Slurm

For a private GHCR package, create a GitHub personal access token with
`read:packages`. If the package inherits permissions from a private repository,
GitHub may also require repository access for that token.

Configure Enroot credentials on the cluster login node:

```sh
mkdir -p ~/.config/enroot
chmod 700 ~/.config/enroot
cat > ~/.config/enroot/.credentials <<'EOF'
machine ghcr.io login GITHUB_USERNAME password GITHUB_TOKEN
EOF
chmod 600 ~/.config/enroot/.credentials
```

Replace `GITHUB_USERNAME` and `GITHUB_TOKEN` with your GitHub username and token.
Then test the pull path with Pyxis:

```sh
srun --container-image=ghcr.io#isaiahpressman/kaggle-orbit-wars:main \
  --container-workdir=/workspace/orbit-wars \
  uv run python -c 'import torch, owl, owl.rs; print(torch.__version__)'
```

Pyxis accepts either an image reference or a `.sqsh` path. For non-default
registries, use the Enroot-style `REGISTRY#IMAGE` separator, for example
`ghcr.io#isaiahpressman/kaggle-orbit-wars:main`.

## W&B environment file

Create a private environment file on the cluster:

```sh
mkdir -p ~/.config/orbit-wars
cat > ~/.config/orbit-wars/wandb.env <<'EOF'
WANDB_API_KEY=<API_KEY>
EOF
chmod 600 ~/.config/orbit-wars/wandb.env
```

The Slurm templates source this file on the host, export `WANDB_API_KEY`, and
pass it into the container with Pyxis `--container-env`. W&B run output is still
controlled by the training code's `wandb.init(dir=run_dir)` call, where `run_dir`
is under the mounted `/runs` output directory.

Do not store GHCR tokens in this W&B env file. GHCR pull credentials belong in
`~/.config/enroot/.credentials` so Enroot can authenticate before the container
starts.

## Slurm launch patterns

Clusters differ in how they run OCI images. The important pattern is the same:
request GPUs with Slurm, mount a host output directory, and run the existing
training entrypoint inside the prebuilt image.

### Batch training with Pyxis

Use the checked-in batch template:

```sh
sbatch scripts/slurm/launch-train.sbatch
```

`ORBIT_WARS_OUTPUT_DIR` is mounted as `/runs` inside the container. The training
script creates timestamped subdirectories there and writes checkpoints, configs,
and W&B files under that mounted directory.

The template defaults to `ghcr.io#isaiahpressman/kaggle-orbit-wars:main`.
Override `ORBIT_WARS_IMAGE` to pin a different tag, such as a specific commit
SHA.

`ORBIT_WARS_CONFIG` selects the training config and defaults to
`configs/baseline.yaml`. The batch script expects to be run from the repository
root by default, mounts `./configs` read-only at `/config`, and runs training
with `/config/baseline.yaml`.

To use a different host-edited config without rebuilding the image, set
`ORBIT_WARS_CONFIG` to another host YAML file:

```sh
ORBIT_WARS_CONFIG=/path/to/experiment.yaml \
  sbatch scripts/slurm/launch-train.sbatch
```

The launch script mounts the config file's parent directory read-only at
`/config` and runs training with `/config/experiment.yaml`. If the config uses
subconfig references such as `model: stateless_transformer_6m`, keep the
referenced subconfig directories next to the mounted file, for example
`/path/to/model/stateless_transformer_6m.yaml`.

Override Slurm resources either by editing `scripts/slurm/launch-train.sbatch` or
by passing normal `sbatch` flags, for example:

```sh
sbatch --partition=gpu --account=ACCOUNT --time=24:00:00 \
  scripts/slurm/launch-train.sbatch
```

With no positional arguments, the batch script starts a fresh run from
`ORBIT_WARS_CONFIG` and writes under `/runs`. Arguments beginning with `-` are
forwarded as additional `scripts/run_ppo.py` flags for that default fresh run:

```sh
sbatch --exclude=gpu-node-01 scripts/slurm/launch-train.sbatch \
  --overrides env.n_envs=1024 rl.horizon=256
```

To initialize a fresh Slurm run from existing model weights without resuming the
old optimizer, scheduler, config, or W&B run, set
`ORBIT_WARS_LOAD_MODEL_WEIGHTS` to a host checkpoint path. If the checkpoint is
outside `ORBIT_WARS_OUTPUT_DIR`, the wrapper mounts its parent directory
read-only at `/model-weights` inside the container:

```sh
ORBIT_WARS_LOAD_MODEL_WEIGHTS=/path/to/checkpoints/checkpoint_last_best.pt \
  sbatch scripts/slurm/launch-train.sbatch
```

Set `ORBIT_WARS_LOAD_MODEL_WEIGHTS_MODE=model_and_optimizer` to also reload the
checkpoint optimizer moment/momentum state. Scheduler state and optimizer-step
counters remain fresh, and optimizer hyperparameters such as LR and weight decay
come from the fresh config rather than the checkpoint param groups.

You can also pass `--load-model-weights` after the batch script. Use a container
path under `/runs`, a host path under `ORBIT_WARS_OUTPUT_DIR`, or another
existing host checkpoint path:

```sh
sbatch scripts/slurm/launch-train.sbatch \
  --load-model-weights /runs/20260505-120000/checkpoint_last_best.pt

ORBIT_WARS_OUTPUT_DIR=/path/to/runs \
  sbatch scripts/slurm/launch-train.sbatch \
    --load-model-weights /path/to/runs/20260505-120000/checkpoint_last_best.pt

sbatch scripts/slurm/launch-train.sbatch \
  --load-model-weights /path/to/checkpoints/checkpoint_last_best.pt
```

The new run keeps only the checkpoint model weights plus `env_steps`,
`player_step_total`, `total_games_played`, and `total_active_entities` logging
counters unless `--load-model-weights-mode model_and_optimizer` is set. Host
checkpoint paths under `ORBIT_WARS_OUTPUT_DIR` are mapped into the container's
`/runs` mount before launch; other host checkpoint directories are mounted
read-only at `/model-weights`.

Teacher checkpoints configured with `rl.teacher_init` are loaded by
`scripts/run_ppo.py` itself, not by the Slurm wrapper. The checkpoint path must
therefore be visible inside the container, and the checkpoint's parent
directory must also contain the teacher `config.yaml`. Relative
`rl.teacher_init` values are resolved against the training config path; with
the default `/config` mount, prefer container-visible paths such as
`/runs/<run>/checkpoint_last_best.pt` or add an explicit mount for any external
teacher directory.

When the first batch-script argument is a positional target, the arguments after
the batch script use the same shape as `scripts/run_ppo.py`, with default
`--log-mode` and `--max-runtime-hours` supplied by the wrapper:

```sh
sbatch scripts/slurm/launch-train.sbatch /runs/20260505-120000
sbatch scripts/slurm/launch-train.sbatch \
  /runs/20260505-120000/checkpoint_00_020_000_000.pt --max-runtime-hours 12
```

Resume targets may also use a host path under `ORBIT_WARS_OUTPUT_DIR`; the
wrapper maps that first path into the container's `/runs` mount before launching
training.

For production jobs, point at `configs/baseline.yaml` or another checked-in
config, and mount any external data or required settings.

### Multi-GPU PPO

The batch launcher runs PPO through `torchrun` with one process per GPU on a
single node. Override Slurm GPU resources in the normal `sbatch` position:

```sh
sbatch --gres=gpu:b200:4 scripts/slurm/launch-train.sbatch
```

`EnvConfig.n_envs`, `rl.horizon`, and `rl.segments_per_minibatch` are per rank.
The effective rollout width is:

```text
world_size * env.n_envs * rl.horizon
```

Checkpoint frequency, `--max-env-steps`, W&B step values, and logged
`train/env_steps` are global across ranks. Rank 0 owns W&B, checkpoints,
last-best evaluation replay files, and the saved `config.yaml`. The saved config
records `runtime.n_runtime_gpus`; resume launches with a different GPU count
derive an equivalent per-rank `env.n_envs` and minibatch/accumulation shape when
those values scale exactly, and fail otherwise. The derived resume config never
increases `rl.segments_per_minibatch`.

## Interactive debugging

Use the interactive helper to request a Slurm allocation and open a shell inside
the container:

```sh
ORBIT_WARS_OUTPUT_DIR=/path/to/debug-runs \
ORBIT_WARS_PARTITION=gpu \
ORBIT_WARS_GPUS=1 \
ORBIT_WARS_TIME=01:00:00 \
scripts/slurm/launch-interactive.sh
```

The helper uses `ghcr.io#isaiahpressman/kaggle-orbit-wars:main` by default.
Set `ORBIT_WARS_IMAGE` to debug a different tag.
It requests GPUs from Slurm with `--gres=gpu:b200:$ORBIT_WARS_GPUS`; Docker's
`--gpus all` flag is not used with Pyxis.
The image and helper also set `NVIDIA_VISIBLE_DEVICES=all` and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility` so Enroot's NVIDIA hook exposes
CUDA devices and driver utilities.

Inside the shell, run focused commands before scheduling a full job:

```sh
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi
uv run python -c 'import torch; print(torch.cuda.is_available())'
uv run python scripts/run_ppo.py configs/baseline.yaml /runs \
  --log-mode debug \
  -o env.n_envs=1 env.pin_memory=false model.embed_dim=32 model.depth=1 \
    model.n_heads=4 rl.horizon=16 rl.segments_per_minibatch=1 \
  --max-env-steps 16
```

The helper mounts `ORBIT_WARS_OUTPUT_DIR` as `/runs`, sources the optional W&B
env file, and passes `WANDB_API_KEY` into the container when present. It also
mounts `ORBIT_WARS_CONFIG_DIR`, defaulting to `./configs`, read-only at
`/config` and exports `ORBIT_WARS_CONFIG_DIR=/config` inside the shell:

```sh
ORBIT_WARS_CONFIG_DIR=/path/to/configs \
  scripts/slurm/launch-interactive.sh

uv run python scripts/run_ppo.py "$ORBIT_WARS_CONFIG_DIR/experiment.yaml" /runs \
  --log-mode debug
```

## References

- GitHub Container Registry authentication:
  <https://docs.github.com/packages/getting-started-with-github-container-registry/about-github-container-registry>
- Pyxis usage and `--container-image`, `--container-mounts`, and
  `--container-env` options: <https://github.com/NVIDIA/pyxis>
