#!/usr/bin/env bash
#SBATCH --job-name=owl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem-per-gpu=128G
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --time=24:00:00

set -euo pipefail

IMAGE="${ORBIT_WARS_IMAGE:-ghcr.io#isaiahpressman/kaggle-orbit-wars:main}"
WANDB_ENV_FILE="${ORBIT_WARS_ENV_FILE:-$HOME/.config/orbit-wars/wandb.env}"
OUTPUT_DIR="${ORBIT_WARS_OUTPUT_DIR:-/data/personal/isaiah/orbit-wars/runs}"
CONFIG="${ORBIT_WARS_CONFIG:-configs/baseline.yaml}"
LOG_MODE="${ORBIT_WARS_LOG_MODE:-wandb}"
MAX_RUNTIME_HOURS="${ORBIT_WARS_MAX_RUNTIME_HOURS:-23.5}"
LOAD_MODEL_WEIGHTS="${ORBIT_WARS_LOAD_MODEL_WEIGHTS:-}"
LOAD_MODEL_WEIGHTS_MODE="${ORBIT_WARS_LOAD_MODEL_WEIGHTS_MODE:-model_only}"
MASTER_ADDR="${ORBIT_WARS_MASTER_ADDR:-}"
MASTER_PORT="${ORBIT_WARS_MASTER_PORT:-29500}"
EXTRA_CONTAINER_ENV="${ORBIT_WARS_CONTAINER_ENV:-}"
CONFIG_MOUNT_TARGET="/config"
CONFIG_MOUNT_DIR=""
MODEL_WEIGHTS_MOUNT_TARGET="/model-weights"
MODEL_WEIGHTS_MOUNT_DIR=""
MAPPED_MODEL_WEIGHTS_PATH=""
RUN_ARGS=("$@")
USE_DEFAULT_FRESH_LAUNCH=0

if [[ "${#RUN_ARGS[@]}" -eq 0 ]]; then
    USE_DEFAULT_FRESH_LAUNCH=1
elif [[ "${RUN_ARGS[0]}" == -* ]]; then
    USE_DEFAULT_FRESH_LAUNCH=1
fi

if [[ ! -f "$WANDB_ENV_FILE" ]]; then
    echo "Missing environment file: $WANDB_ENV_FILE" >&2
    echo "Create it with: WANDB_API_KEY=<API_KEY>" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$USE_DEFAULT_FRESH_LAUNCH" -eq 1 ]]; then
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
fi

map_output_path() {
    local path="$1"
    local output_dir_abs
    local path_abs

    if [[ "$path" == "/runs" || "$path" == /runs/* ]]; then
        printf '%s\n' "$path"
        return
    fi

    output_dir_abs="$(cd "$OUTPUT_DIR" && pwd)"
    if [[ "$path" = /* ]]; then
        path_abs="$path"
    elif [[ -e "$path" ]]; then
        path_abs="$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
    else
        printf '%s\n' "$path"
        return
    fi

    if [[ "$path_abs" == "$output_dir_abs" ]]; then
        printf '/runs\n'
    elif [[ "$path_abs" == "$output_dir_abs"/* ]]; then
        printf '/runs/%s\n' "${path_abs#"$output_dir_abs"/}"
    else
        printf '%s\n' "$path"
    fi
}

if [[ "$USE_DEFAULT_FRESH_LAUNCH" -eq 0 ]]; then
    RUN_ARGS[0]="$(map_output_path "${RUN_ARGS[0]}")"
fi

has_load_model_weights_arg() {
    local arg
    for arg in "${RUN_ARGS[@]}"; do
        if [[ "$arg" == "--load-model-weights" || "$arg" == --load-model-weights=* ]]; then
            return 0
        fi
    done
    return 1
}

has_load_model_weights_mode_arg() {
    local arg
    for arg in "${RUN_ARGS[@]}"; do
        if [[ "$arg" == "--load-model-weights-mode" || \
            "$arg" == --load-model-weights-mode=* ]]; then
            return 0
        fi
    done
    return 1
}

map_model_weights_path() {
    local path="$1"
    local mapped_path
    local host_dir

    mapped_path="$(map_output_path "$path")"
    if [[ "$mapped_path" != "$path" || "$mapped_path" == "/runs" || "$mapped_path" == /runs/* ]]; then
        MAPPED_MODEL_WEIGHTS_PATH="$mapped_path"
        return
    fi

    if [[ ! -f "$path" ]]; then
        echo "Missing model-weights checkpoint: $path" >&2
        exit 1
    fi

    host_dir="$(cd "$(dirname "$path")" && pwd)"
    MODEL_WEIGHTS_MOUNT_DIR="$host_dir"
    MAPPED_MODEL_WEIGHTS_PATH="$MODEL_WEIGHTS_MOUNT_TARGET/$(basename "$path")"
}

map_load_model_weights_args() {
    local index=0
    local arg
    while [[ "$index" -lt "${#RUN_ARGS[@]}" ]]; do
        arg="${RUN_ARGS[$index]}"
        if [[ "$arg" == "--load-model-weights" ]]; then
            if [[ "$((index + 1))" -ge "${#RUN_ARGS[@]}" ]]; then
                echo "--load-model-weights requires a checkpoint path" >&2
                exit 1
            fi
            map_model_weights_path "${RUN_ARGS[$((index + 1))]}"
            RUN_ARGS[$((index + 1))]="$MAPPED_MODEL_WEIGHTS_PATH"
            index=$((index + 2))
        elif [[ "$arg" == --load-model-weights=* ]]; then
            map_model_weights_path "${arg#--load-model-weights=}"
            RUN_ARGS[$index]="--load-model-weights=$MAPPED_MODEL_WEIGHTS_PATH"
            index=$((index + 1))
        else
            index=$((index + 1))
        fi
    done
}

if [[ -n "$LOAD_MODEL_WEIGHTS" && "$USE_DEFAULT_FRESH_LAUNCH" -eq 0 ]]; then
    echo "ORBIT_WARS_LOAD_MODEL_WEIGHTS is only supported for default fresh launches." >&2
    echo "Pass --load-model-weights explicitly when providing run_ppo.py positional arguments." >&2
    exit 1
fi
if [[ -n "$LOAD_MODEL_WEIGHTS" ]] && has_load_model_weights_arg; then
    echo "Use either ORBIT_WARS_LOAD_MODEL_WEIGHTS or --load-model-weights, not both." >&2
    exit 1
fi
if [[ -n "$LOAD_MODEL_WEIGHTS" ]]; then
    map_model_weights_path "$LOAD_MODEL_WEIGHTS"
    LOAD_MODEL_WEIGHTS="$MAPPED_MODEL_WEIGHTS_PATH"
fi
map_load_model_weights_args
if [[ "$LOAD_MODEL_WEIGHTS_MODE" != "model_only" && \
    "$LOAD_MODEL_WEIGHTS_MODE" != "model_and_optimizer" ]]; then
    echo \
        "ORBIT_WARS_LOAD_MODEL_WEIGHTS_MODE must be model_only or model_and_optimizer." \
        >&2
    exit 1
fi
FORWARD_LOAD_MODEL_WEIGHTS_MODE=0
if [[ "$LOAD_MODEL_WEIGHTS_MODE" != "model_only" ]]; then
    if has_load_model_weights_mode_arg; then
        echo \
            "Use either ORBIT_WARS_LOAD_MODEL_WEIGHTS_MODE or --load-model-weights-mode, not both." \
            >&2
        exit 1
    fi
    if [[ -z "$LOAD_MODEL_WEIGHTS" ]]; then
        echo \
            "ORBIT_WARS_LOAD_MODEL_WEIGHTS_MODE requires ORBIT_WARS_LOAD_MODEL_WEIGHTS." \
            >&2
        exit 1
    fi
    FORWARD_LOAD_MODEL_WEIGHTS_MODE=1
fi

set -a
# shellcheck disable=SC1090
source "$WANDB_ENV_FILE"
set +a

: "${NVIDIA_VISIBLE_DEVICES:=all}"
: "${NVIDIA_DRIVER_CAPABILITIES:=compute,utility}"
: "${UV_COMPILE_BYTECODE:=0}"
export NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES UV_COMPILE_BYTECODE ORBIT_WARS_CONFIG="$CONFIG"

if [[ "$LOG_MODE" == "wandb" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY must be set in $WANDB_ENV_FILE when ORBIT_WARS_LOG_MODE=wandb." >&2
    exit 1
fi

NNODES="${SLURM_NNODES:-1}"
if [[ ! "$NNODES" =~ ^[0-9]+$ ]]; then
    echo "SLURM_NNODES must be a positive integer, got: $NNODES" >&2
    exit 1
fi
if ((NNODES < 1)); then
    echo "SLURM_NNODES must be positive, got: $NNODES" >&2
    exit 1
fi

GPUS_PER_NODE="${ORBIT_WARS_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
torchrun_cmd=(
    uv run torchrun
)
if ((NNODES > 1)); then
    if [[ -z "$MASTER_ADDR" ]]; then
        if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
            echo "SLURM_JOB_NODELIST is required for multi-node torchrun." >&2
            exit 1
        fi
        if ! command -v scontrol >/dev/null 2>&1; then
            echo "scontrol is required to resolve the multi-node torchrun master address." >&2
            exit 1
        fi
        read -r MASTER_ADDR < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    fi
    if [[ -z "$MASTER_ADDR" ]]; then
        echo "Failed to resolve the multi-node torchrun master address." >&2
        exit 1
    fi
    torchrun_cmd+=(
        --nnodes="$NNODES"
        --nproc-per-node="$GPUS_PER_NODE"
        --rdzv-backend=c10d
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT"
        --rdzv-id="${ORBIT_WARS_RDZV_ID:-${SLURM_JOB_ID:-orbit-wars}}"
    )
else
    torchrun_cmd+=(
        --standalone
        --nnodes=1
        --nproc-per-node="$GPUS_PER_NODE"
    )
fi

train_cmd=(
    "${torchrun_cmd[@]}"
    scripts/run_ppo.py
    --log-mode "$LOG_MODE"
)

if [[ "$USE_DEFAULT_FRESH_LAUNCH" -eq 1 ]]; then
    train_cmd+=("$CONFIG" /runs)
    if [[ -n "$LOAD_MODEL_WEIGHTS" ]]; then
        train_cmd+=(--load-model-weights "$LOAD_MODEL_WEIGHTS")
    fi
    if [[ "$FORWARD_LOAD_MODEL_WEIGHTS_MODE" -eq 1 ]]; then
        train_cmd+=(--load-model-weights-mode "$LOAD_MODEL_WEIGHTS_MODE")
    fi
fi

if [[ -n "$MAX_RUNTIME_HOURS" ]]; then
    train_cmd+=(--max-runtime-hours "$MAX_RUNTIME_HOURS")
fi

if [[ "${#RUN_ARGS[@]}" -gt 0 ]]; then
    train_cmd+=("${RUN_ARGS[@]}")
fi

CONTAINER_MOUNTS="$OUTPUT_DIR:/runs"
if [[ -n "$CONFIG_MOUNT_DIR" ]]; then
    CONTAINER_MOUNTS+=",$CONFIG_MOUNT_DIR:$CONFIG_MOUNT_TARGET:ro"
fi
if [[ -n "$MODEL_WEIGHTS_MOUNT_DIR" ]]; then
    CONTAINER_MOUNTS+=",$MODEL_WEIGHTS_MOUNT_DIR:$MODEL_WEIGHTS_MOUNT_TARGET:ro"
fi

CONTAINER_ENV="WANDB_API_KEY,NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,UV_COMPILE_BYTECODE,ORBIT_WARS_CONFIG"
if [[ -n "$EXTRA_CONTAINER_ENV" ]]; then
    CONTAINER_ENV+=",$EXTRA_CONTAINER_ENV"
fi

srun \
    --nodes="$NNODES" \
    --ntasks="$NNODES" \
    --ntasks-per-node=1 \
    --container-image="$IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir=/workspace/orbit-wars \
    --container-env="$CONTAINER_ENV" \
    "${train_cmd[@]}"
