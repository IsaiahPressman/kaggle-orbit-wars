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

## Local smoke test

CPU-only smoke test:

```sh
docker run --rm orbit-wars:dev \
  uv run python scripts/run_ppo.py configs/train/debug.yaml /tmp/runs \
    --log-mode debug \
    --max-env-steps 16
```

GPU smoke test on a Docker host with NVIDIA Container Toolkit:

```sh
docker run --rm --gpus all orbit-wars:dev \
  uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Slurm launch patterns

Clusters differ in how they run OCI images. The important pattern is the same:
request GPUs with Slurm, mount a host output directory, and run the existing
training entrypoint inside the prebuilt image.

Example with Enroot/Pyxis:

```sh
#!/usr/bin/env bash
#SBATCH --job-name=orbit-wars-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

IMAGE="registry.example.com/orbit-wars:dev"
RUNS_DIR="$PWD/runs"
mkdir -p "$RUNS_DIR" logs

srun --container-image="$IMAGE" \
  --container-mounts="$RUNS_DIR:/runs" \
  uv run python scripts/run_ppo.py configs/train/debug.yaml /runs \
    --log-mode debug \
    --max-runtime-hours 0.9
```

Example with Apptainer:

```sh
#!/usr/bin/env bash
#SBATCH --job-name=orbit-wars-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

IMAGE="orbit-wars_dev.sif"
RUNS_DIR="$PWD/runs"
mkdir -p "$RUNS_DIR" logs

srun apptainer exec --nv --bind "$RUNS_DIR:/runs" "$IMAGE" \
  uv run python scripts/run_ppo.py configs/train/debug.yaml /runs \
    --log-mode debug \
    --max-runtime-hours 0.9
```

For production jobs, point at `configs/train/baseline.yaml` or another checked-in
config, and mount any external data, Kaggle credentials, or W&B settings needed
by the logging mode.
