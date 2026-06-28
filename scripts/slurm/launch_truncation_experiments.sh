#!/usr/bin/env bash
# Submit the four truncation experiments to Slurm (kander), one B200 each, as two
# consecutive jobs per experiment so each trains ~23.5h total.
#
# Each job uses scripts/slurm/launch-train.sbatch with a 12h Slurm wall-clock
# limit (the partition max) and caps training at 11.75h (--max-runtime-hours),
# leaving ~15 min to save the final checkpoint and exit cleanly. The second job
# resumes the first job's run directory (ORBIT_WARS_RESUME_LATEST=1) and is queued
# with an `afterany` dependency so it starts once the first job finishes. Each
# experiment writes to its own run directory under ORBIT_WARS_OUTPUT_DIR/<name>.
#
# Submission relies on launch-train.sbatch's defaults (--account=research,
# --partition=dev) and only overrides the wall-clock limit and GPU count. Run from
# the repository root on a kander login node:
#   scripts/slurm/launch_truncation_experiments.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_BASE="${ORBIT_WARS_OUTPUT_DIR:-/data/personal/isaiah/orbit-wars/runs}"
MAX_RUNTIME_HOURS="${ORBIT_WARS_MAX_RUNTIME_HOURS:-11.75}"
TIME_LIMIT="${ORBIT_WARS_SLURM_TIME:-12:00:00}"
GRES="${ORBIT_WARS_GRES:-gpu:b200:1}"
SBATCH_SCRIPT="scripts/slurm/launch-train.sbatch"

# name:config — name doubles as the Slurm job-name prefix and run output subdir.
experiments=(
    "trunc_exp1_200_shipratio:configs/trunc_exp1_200_shipratio.yaml"
    "trunc_exp2_200:configs/trunc_exp2_200.yaml"
    "trunc_exp3_300:configs/trunc_exp3_300.yaml"
    "trunc_exp4_shipratio_full:configs/trunc_exp4_shipratio_full.yaml"
)

for entry in "${experiments[@]}"; do
    name="${entry%%:*}"
    config="${entry#*:}"
    if [[ ! -f "$config" ]]; then
        echo "Missing config: $config" >&2
        exit 1
    fi
    out_dir="$OUTPUT_BASE/$name"
    mkdir -p "$out_dir"

    # UV_NO_SYNC=1 (forwarded into the container via ORBIT_WARS_CONTAINER_ENV)
    # stops `uv run` from rebuilding owl at job start. This branch's mounted
    # python/ differs from the image, which would otherwise trigger a maturin
    # editable rebuild that fails writing rs.abi3.so into the read-only python/
    # mount. The image-built Rust extension stays importable via
    # OWL_NATIVE_MODULE_DIR, and this branch only changes Python (no Rust/dep
    # changes vs the image), so skipping the sync is correct.
    jid1=$(sbatch --parsable \
        --job-name="${name}-j1" \
        --time="$TIME_LIMIT" \
        --gres="$GRES" \
        --export="ALL,UV_NO_SYNC=1,ORBIT_WARS_CONTAINER_ENV=UV_NO_SYNC,ORBIT_WARS_CONFIG=$config,ORBIT_WARS_MAX_RUNTIME_HOURS=$MAX_RUNTIME_HOURS,ORBIT_WARS_OUTPUT_DIR=$out_dir" \
        "$SBATCH_SCRIPT")

    jid2=$(sbatch --parsable \
        --job-name="${name}-j2" \
        --time="$TIME_LIMIT" \
        --gres="$GRES" \
        --dependency="afterany:$jid1" \
        --export="ALL,UV_NO_SYNC=1,ORBIT_WARS_CONTAINER_ENV=UV_NO_SYNC,ORBIT_WARS_RESUME_LATEST=1,ORBIT_WARS_MAX_RUNTIME_HOURS=$MAX_RUNTIME_HOURS,ORBIT_WARS_OUTPUT_DIR=$out_dir" \
        "$SBATCH_SCRIPT")

    echo "$name: job1=$jid1 (fresh) -> job2=$jid2 (resume after $jid1) | out=$out_dir"
done
