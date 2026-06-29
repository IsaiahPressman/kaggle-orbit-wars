#!/usr/bin/env bash
# Pre-launch memory + throughput smoke for the scaling-law configs.
#
# Runs each model on ONE GPU for a few iterations at the config's
# segments_per_minibatch/grad_accum (16/1 by default) with the last_best teacher
# active, sampling peak GPU memory and per-GPU throughput. Per-GPU memory is
# independent of world size (DDP replicates the model per rank), so a single-GPU
# smoke determines the gradient-accumulation split we need for every tier, and
# the per-GPU throughput lets us size the chained-job count.
#
# Intended to run inside the training container via a single interactive srun:
#   srun --partition=interactive --gres=gpu:b200:1 ... \
#     --container-image=<img> --container-workdir=/workspace/orbit-wars \
#     --container-mounts=<python:ro,scripts:ro,configs:/config:ro> \
#     bash /workspace/orbit-wars/scripts/slurm/smoke_scaling.sh
set -uo pipefail
cd /workspace/orbit-wars

CONFIGS="${SMOKE_CONFIGS:-scaling_6m scaling_12m scaling_25m scaling_50m}"
STEPS="${SMOKE_MAX_ENV_STEPS:-80000}"
PER_MODEL_TIMEOUT="${SMOKE_TIMEOUT:-1500}"

echo "smoke configs: $CONFIGS | max_env_steps=$STEPS | per-model timeout=${PER_MODEL_TIMEOUT}s"
for cfg in $CONFIGS; do
    echo "================ SMOKE $cfg ================"
    memlog="/tmp/mem_$cfg.log"; outlog="/tmp/out_$cfg.log"
    : > "$memlog"
    ( while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 >> "$memlog"
        sleep 2
      done ) &
    sampler=$!
    start=$(date +%s)
    UV_NO_SYNC=1 timeout "$PER_MODEL_TIMEOUT" uv run python scripts/run_ppo.py \
        "/config/$cfg.yaml" "/tmp/smoke_$cfg" \
        --log-mode debug --max-env-steps "$STEPS" > "$outlog" 2>&1
    rc=$?
    end=$(date +%s)
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    peak=$(sort -n "$memlog" 2>/dev/null | tail -1)
    sps=$(grep -oE "'perf/steps_per_second': [0-9.]+" "$outlog" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
    iters=$(grep -c "'train/env_steps'" "$outlog" 2>/dev/null)
    oom=$(grep -ciE "out of memory|CUDA out of memory|OutOfMemoryError" "$outlog" 2>/dev/null)
    echo "RESULT $cfg: exit=$rc elapsed=$((end - start))s iters=$iters peak_gpu_MiB=${peak:-NA} per_gpu_sps=${sps:-NA} oom_lines=$oom"
    if [ "$rc" -ne 0 ]; then
        echo "---- tail of $cfg output (exit=$rc) ----"
        tail -20 "$outlog"
    fi
done
echo "================ SMOKE COMPLETE ================"
