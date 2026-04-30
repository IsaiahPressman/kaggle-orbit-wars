#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ORBIT_WARS_IMAGE:-ghcr.io#isaiahpressman/kaggle-orbit-wars:main}"
ENV_FILE="${ORBIT_WARS_ENV_FILE:-$HOME/.config/orbit-wars/wandb.env}"
OUTPUT_DIR="${ORBIT_WARS_OUTPUT_DIR:-/sw/isaiah/orbit-wars/runs}"
GPUS="${ORBIT_WARS_GPUS:-1}"
CPUS="${ORBIT_WARS_CPUS:-32}"
MEM="${ORBIT_WARS_MEM:-128G}"
TIME="${ORBIT_WARS_TIME:-06:00:00}"

mkdir -p "$OUTPUT_DIR"

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

srun "${srun_args[@]}" \
    --pty \
    --container-image="$IMAGE" \
    --container-mounts="$OUTPUT_DIR:/runs" \
    --container-workdir=/workspace/orbit-wars \
    --container-env=WANDB_API_KEY,NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES \
    bash
