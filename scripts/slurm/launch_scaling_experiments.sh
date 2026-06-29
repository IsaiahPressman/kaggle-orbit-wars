#!/usr/bin/env bash
# Queue the scaling-law sweep on kander. Each model trains to its global env-step
# target across a chain of <=12h jobs: training is capped at 11.75h, resume jobs
# pick up the latest run dir (ORBIT_WARS_RESUME_LATEST), and ORBIT_WARS_MAX_ENV_STEPS
# is the hard stop, so trailing jobs in a chain no-op once the target is reached.
#
# GPUs and env-step targets double per tier; n_envs stays 256 per rank, so global
# envs and batch double with the GPU count. UV_NO_SYNC keeps the image-built Rust
# extension importable under the read-only python/ mount (this branch's Python
# differs from the image; no Rust/dep changes, so skipping the sync is correct).
#
# Chain lengths are sized from the pre-launch throughput smoke (per-GPU sps drops
# with model size, so wall-clock grows across tiers). Run from the repo root on a
# kander login node:  scripts/slurm/launch_scaling_experiments.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_BASE="${ORBIT_WARS_OUTPUT_DIR:-/data/personal/isaiah/orbit-wars/runs}"
MAX_RUNTIME_HOURS="${ORBIT_WARS_MAX_RUNTIME_HOURS:-11.75}"
TIME_LIMIT="${ORBIT_WARS_SLURM_TIME:-12:00:00}"
SBATCH="scripts/slurm/launch-train.sbatch"
COMMON="UV_NO_SYNC=1,ORBIT_WARS_CONTAINER_ENV=UV_NO_SYNC,ORBIT_WARS_MAX_RUNTIME_HOURS=$MAX_RUNTIME_HOURS"

# name | config | gres | max_env_steps | n_chained_jobs
experiments=(
    "scaling_6m|configs/scaling_6m.yaml|gpu:b200:1|500000000|2"
    "scaling_12m|configs/scaling_12m.yaml|gpu:b200:2|1000000000|3"
    "scaling_25m|configs/scaling_25m.yaml|gpu:b200:4|2000000000|4"
    "scaling_50m|configs/scaling_50m.yaml|gpu:b200:8|4000000000|5"
)

for entry in "${experiments[@]}"; do
    IFS='|' read -r name config gres steps njobs <<< "$entry"
    [ -f "$config" ] || { echo "missing config: $config" >&2; exit 1; }
    out_dir="$OUTPUT_BASE/$name"
    mkdir -p "$out_dir"
    prev=""
    chain=""
    for i in $(seq 1 "$njobs"); do
        if [ "$i" -eq 1 ]; then
            jid=$(sbatch --parsable --job-name="${name}-j${i}" --time="$TIME_LIMIT" --gres="$gres" \
                --export="ALL,$COMMON,ORBIT_WARS_CONFIG=$config,ORBIT_WARS_MAX_ENV_STEPS=$steps,ORBIT_WARS_OUTPUT_DIR=$out_dir" \
                "$SBATCH")
        else
            jid=$(sbatch --parsable --job-name="${name}-j${i}" --time="$TIME_LIMIT" --gres="$gres" \
                --dependency="afterany:$prev" \
                --export="ALL,$COMMON,ORBIT_WARS_RESUME_LATEST=1,ORBIT_WARS_MAX_ENV_STEPS=$steps,ORBIT_WARS_OUTPUT_DIR=$out_dir" \
                "$SBATCH")
        fi
        chain="$chain $jid"
        prev="$jid"
    done
    echo "$name ($gres x $njobs jobs, target $steps env steps): chain =$chain | out=$out_dir"
done
