#!/usr/bin/env bash
#SBATCH --job-name=owl
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem-per-gpu=128G
#SBATCH --partition=llm
#SBATCH --time=24:00:00

set -euo pipefail

IMAGE="${ORBIT_WARS_IMAGE:-ghcr.io#isaiahpressman/kaggle-orbit-wars:main}"
WANDB_ENV_FILE="${ORBIT_WARS_ENV_FILE:-$HOME/.config/orbit-wars/wandb.env}"
OUTPUT_DIR="${ORBIT_WARS_OUTPUT_DIR:-/sw/isaiah/orbit-wars/runs}"
CONFIG="${ORBIT_WARS_CONFIG:-configs/baseline.yaml}"
LOG_MODE="${ORBIT_WARS_LOG_MODE:-wandb}"
MAX_RUNTIME_HOURS="${ORBIT_WARS_MAX_RUNTIME_HOURS:-23.5}"
CONFIG_MOUNT_TARGET="/config"
CONFIG_MOUNT_DIR=""

if [[ ! -f "$WANDB_ENV_FILE" ]]; then
    echo "Missing environment file: $WANDB_ENV_FILE" >&2
    echo "Create it with: WANDB_API_KEY=<API_KEY>" >&2
    exit 1
fi

if [[ -f "$CONFIG" ]]; then
    CONFIG_MOUNT_DIR="$(cd "$(dirname "$CONFIG")" && pwd)"
    CONFIG="$CONFIG_MOUNT_TARGET/$(basename "$CONFIG")"
elif [[ "$CONFIG" = /* ]]; then
    if [[ ! -f "$CONFIG" ]]; then
        echo "Missing config file: $CONFIG" >&2
        exit 1
    fi
else
    echo "Missing config file: $CONFIG" >&2
    echo "Run from the repository root or set ORBIT_WARS_CONFIG to a host config file." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

set -a
# shellcheck disable=SC1090
source "$WANDB_ENV_FILE"
set +a

: "${NVIDIA_VISIBLE_DEVICES:=all}"
: "${NVIDIA_DRIVER_CAPABILITIES:=compute,utility}"
export NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES ORBIT_WARS_CONFIG="$CONFIG"

if [[ "$LOG_MODE" == "wandb" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY must be set in $WANDB_ENV_FILE when ORBIT_WARS_LOG_MODE=wandb." >&2
    exit 1
fi

train_cmd=(
    uv run torchrun
    --standalone
    --nnodes=1
    --nproc-per-node="${ORBIT_WARS_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
    scripts/run_ppo.py
    "$CONFIG"
    /runs
    --log-mode "$LOG_MODE"
)

if [[ -n "$MAX_RUNTIME_HOURS" ]]; then
    train_cmd+=(--max-runtime-hours "$MAX_RUNTIME_HOURS")
fi

if [[ "$#" -gt 0 ]]; then
    train_cmd+=("$@")
fi

CONTAINER_MOUNTS="$OUTPUT_DIR:/runs"
if [[ -n "$CONFIG_MOUNT_DIR" ]]; then
    CONTAINER_MOUNTS+=",$CONFIG_MOUNT_DIR:$CONFIG_MOUNT_TARGET:ro"
fi

srun \
    --container-image="$IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir=/workspace/orbit-wars \
    --container-env=WANDB_API_KEY,NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,ORBIT_WARS_CONFIG \
    "${train_cmd[@]}"
