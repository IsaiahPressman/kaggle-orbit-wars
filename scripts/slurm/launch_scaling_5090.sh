#!/usr/bin/env bash
# Down-tier scaling sweep on consumer 5090s (3M @ 250M steps, 1.5M @ 125M steps).
# Same chain-of-jobs pattern as launch_scaling_experiments.sh, adapted for 5090s:
#   - no NVLink -> NCCL_P2P_DISABLE/NCCL_SHM_DISABLE, forwarded into the container
#     via ORBIT_WARS_CONTAINER_ENV (exported so --export=ALL carries the comma list).
#   - rtx nodes can't satisfy the sbatch default --cpus-per-gpu/--mem-per-gpu, so
#     override to 8 cpus / 40G per GPU.
#   - dev 5090 jobs cap at 4h -> 3h50m runtime stop, chained to the step target.
# EFFECTIVE (global) n_envs/batch = per-rank n_envs (in the config) * n_gpus; the
# configs are written for the GRES below (2x5090; see each config header). Keep them
# in sync. Throughput smoke: these tiny models are comms-bound past 2 GPUs (2->4 was
# +0.8%), so 2x5090/run is the throughput/GPU sweet spot. Real co-located sps on
# shared 5090 nodes is ~2.7k (vs ~5k solo -> CPU/bandwidth contention when two runs
# share a node), so chains are sized generously; trailing jobs no-op once the
# ORBIT_WARS_MAX_ENV_STEPS target is hit: 3M (250M) -> 8 jobs; 1.5M (125M) -> 5 jobs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_BASE="${ORBIT_WARS_OUTPUT_DIR:-/data/personal/isaiah/orbit-wars/runs}"
MAX_RUNTIME_HOURS="${ORBIT_WARS_MAX_RUNTIME_HOURS:-3.83}"
TIME_LIMIT="${ORBIT_WARS_SLURM_TIME:-03:55:00}"
PARTITION="${ORBIT_WARS_PARTITION:-dev}"
SBATCH="scripts/slurm/launch-train.sbatch"

# Consumer 5090s: no NVLink. Disable P2P/SHM and forward the (comma-containing)
# container-env list through the shell so --export=ALL carries it intact.
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export ORBIT_WARS_CONTAINER_ENV="UV_NO_SYNC,NCCL_P2P_DISABLE,NCCL_SHM_DISABLE"
COMMON="UV_NO_SYNC=1,ORBIT_WARS_MAX_RUNTIME_HOURS=$MAX_RUNTIME_HOURS"

# name | config | gres | max_env_steps | n_chained_jobs   (njobs sized from the smoke)
experiments=(
    "scaling_3m|configs/scaling_3m.yaml|gpu:5090:2|250000000|8"
    "scaling_1p5m|configs/scaling_1p5m.yaml|gpu:5090:2|125000000|5"
)

for entry in "${experiments[@]}"; do
    IFS='|' read -r name config gres steps njobs <<< "$entry"
    [ -f "$config" ] || { echo "missing config: $config" >&2; exit 1; }
    experiment_root="$OUTPUT_BASE/$name"
    mkdir -p "$experiment_root"
    out_dir="$(mktemp -d \
        "$experiment_root/chain-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
    prev=""
    chain=""
    for i in $(seq 1 "$njobs"); do
        if [ "$i" -eq 1 ]; then
            jid=$(sbatch --parsable --job-name="${name}-j${i}" --time="$TIME_LIMIT" \
                --partition="$PARTITION" --gres="$gres" --cpus-per-gpu=8 --mem-per-gpu=40G \
                --export="ALL,$COMMON,ORBIT_WARS_CONFIG=$config,ORBIT_WARS_MAX_ENV_STEPS=$steps,ORBIT_WARS_OUTPUT_DIR=$out_dir" \
                "$SBATCH")
        else
            jid=$(sbatch --parsable --job-name="${name}-j${i}" --time="$TIME_LIMIT" \
                --partition="$PARTITION" --gres="$gres" --cpus-per-gpu=8 --mem-per-gpu=40G \
                --dependency="afterany:$prev" \
                --export="ALL,$COMMON,ORBIT_WARS_RESUME_LATEST=1,ORBIT_WARS_MAX_ENV_STEPS=$steps,ORBIT_WARS_OUTPUT_DIR=$out_dir" \
                "$SBATCH")
        fi
        chain="$chain $jid"
        prev="$jid"
    done
    echo "$name ($gres x $njobs jobs, target $steps env steps): chain =$chain | out=$out_dir"
done
