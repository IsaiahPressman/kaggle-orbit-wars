#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ORBIT_WARS_IMAGE:-ghcr.io#isaiahpressman/kaggle-orbit-wars:main}"
ENV_FILE="${ORBIT_WARS_ENV_FILE:-$HOME/.config/orbit-wars/wandb.env}"
OUTPUT_DIR="${ORBIT_WARS_OUTPUT_DIR:-/sw/isaiah/orbit-wars/runs}"
CONFIG_DIR="${ORBIT_WARS_CONFIG_DIR:-configs}"
GPUS="${ORBIT_WARS_GPUS:-1}"
CPUS="${ORBIT_WARS_CPUS:-32}"
MEM="${ORBIT_WARS_MEM:-128G}"
TIME="${ORBIT_WARS_TIME:-06:00:00}"
CONFIG_MOUNT_TARGET="/config"
SRUN_EXTRA_ARGS=()

if [[ $# -gt 0 ]]; then
    if [[ "$1" != "--" ]]; then
        echo "Usage: $0 [-- SRUN_ARGS...]" >&2
        exit 2
    fi

    shift
    SRUN_EXTRA_ARGS=("$@")
fi

mkdir -p "$OUTPUT_DIR"

if [[ -n "$CONFIG_DIR" ]]; then
    if [[ ! -d "$CONFIG_DIR" ]]; then
        echo "Missing config directory: $CONFIG_DIR" >&2
        exit 1
    fi

    CONFIG_DIR="$(cd "$CONFIG_DIR" && pwd)"
    export ORBIT_WARS_CONFIG_DIR="$CONFIG_MOUNT_TARGET"
fi

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "Warning: environment file not found: $ENV_FILE" >&2
    echo "Continuing without W&B environment variables." >&2
fi

: "${NVIDIA_VISIBLE_DEVICES:=all}"
: "${NVIDIA_DRIVER_CAPABILITIES:=compute,utility}"
export NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES

CONTAINER_MOUNTS="$OUTPUT_DIR:/runs"
if [[ -n "$CONFIG_DIR" ]]; then
    CONTAINER_MOUNTS+=",$CONFIG_DIR:$CONFIG_MOUNT_TARGET:ro"
fi

srun_args=(
    --ntasks=1
    --nodes=1
    --gpus-per-node="$GPUS"
    --cpus-per-gpu="$CPUS"
    --mem-per-gpu="$MEM"
    --time="$TIME"
    --partition=interactive
)

if [[ -n "${ORBIT_WARS_ACCOUNT:-}" ]]; then
    srun_args+=(--account="$ORBIT_WARS_ACCOUNT")
fi

if [[ -n "${ORBIT_WARS_PARTITION:-}" ]]; then
    srun_args+=(--partition="$ORBIT_WARS_PARTITION")
fi

srun_args+=("${SRUN_EXTRA_ARGS[@]}")

srun "${srun_args[@]}" \
    --pty \
    --container-image="$IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir=/workspace/orbit-wars \
    --container-env=WANDB_API_KEY,NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,ORBIT_WARS_CONFIG_DIR \
    bash
