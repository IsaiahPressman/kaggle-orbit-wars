# Containerization

This repository can be built into an image that contains Python 3.12, `uv`, the
repo-pinned Rust toolchain, Cargo dependencies, Python dependencies, and the
compiled `maturin` extension.

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
ORBIT_WARS_CONFIG=/sw/isaiah/orbit-wars/configs/experiment.yaml \
  sbatch scripts/slurm/launch-train.sbatch
```

The launch script mounts the config file's parent directory read-only at
`/config` and runs training with `/config/experiment.yaml`. If the config uses
subconfig references such as `model: stateless_transformer_5m`, keep the
referenced subconfig directories next to the mounted file, for example
`/sw/isaiah/orbit-wars/configs/model/stateless_transformer_5m.yaml`.

Override Slurm resources either by editing `scripts/slurm/launch-train.sbatch` or
by passing normal `sbatch` flags, for example:

```sh
sbatch --partition=gpu --account=ACCOUNT --time=24:00:00 \
  scripts/slurm/launch-train.sbatch
```

For production jobs, point at `configs/baseline.yaml` or another checked-in
config, and mount any external data or required settings.

## Interactive debugging

Use the interactive helper to request a Slurm allocation and open a shell inside
the container:

```sh
ORBIT_WARS_OUTPUT_DIR=/sw/isaiah/orbit-wars/debug \
ORBIT_WARS_PARTITION=gpu \
ORBIT_WARS_GPUS=1 \
ORBIT_WARS_TIME=01:00:00 \
scripts/slurm/launch-interactive.sh
```

The helper uses `ghcr.io#isaiahpressman/kaggle-orbit-wars:main` by default.
Set `ORBIT_WARS_IMAGE` to debug a different tag.
It requests GPUs from Slurm with `--gpus-per-node=$ORBIT_WARS_GPUS`; Docker's
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
    model.n_heads=4 rl.horizon=16 \
    rl.segment_sampling.segments_per_minibatch=1 \
  --max-env-steps 16
```

The helper mounts `ORBIT_WARS_OUTPUT_DIR` as `/runs`, sources the optional W&B
env file, and passes `WANDB_API_KEY` into the container when present. It also
mounts `ORBIT_WARS_CONFIG_DIR`, defaulting to `./configs`, read-only at
`/config` and exports `ORBIT_WARS_CONFIG_DIR=/config` inside the shell:

```sh
ORBIT_WARS_CONFIG_DIR=/sw/isaiah/orbit-wars/configs \
  scripts/slurm/launch-interactive.sh

uv run python scripts/run_ppo.py "$ORBIT_WARS_CONFIG_DIR/experiment.yaml" /runs \
  --log-mode debug
```

## References

- GitHub Container Registry authentication:
  <https://docs.github.com/packages/getting-started-with-github-container-registry/about-github-container-registry>
- Pyxis usage and `--container-image`, `--container-mounts`, and
  `--container-env` options: <https://github.com/NVIDIA/pyxis>
