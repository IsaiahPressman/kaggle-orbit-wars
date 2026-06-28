#!/usr/bin/env bash
# Submit the four truncation experiments to Slurm (kander), one B200 each.
#
# Each job uses scripts/slurm/launch-train.sbatch with a 12h wall-clock limit and
# caps training at 11h45m (--max-runtime-hours 11.75) so the run checkpoints and
# exits cleanly before Slurm kills it. Each experiment writes to its own run
# directory under ORBIT_WARS_OUTPUT_DIR/<name>.
#
# Run from the repository root:
#   scripts/slurm/launch_truncation_experiments.sh
#
# Honors the same environment overrides as launch-train.sbatch
# (ORBIT_WARS_OUTPUT_DIR, ORBIT_WARS_IMAGE, ORBIT_WARS_ENV_FILE, ...).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_BASE="${ORBIT_WARS_OUTPUT_DIR:-/data/personal/isaiah/orbit-wars/runs}"
MAX_RUNTIME_HOURS="${ORBIT_WARS_MAX_RUNTIME_HOURS:-11.75}"
TIME_LIMIT="${ORBIT_WARS_SLURM_TIME:-12:00:00}"
SBATCH_SCRIPT="scripts/slurm/launch-train.sbatch"

# name:config — name doubles as the Slurm job name and the run output subdir.
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
    echo "Submitting $name (config=$config, time=$TIME_LIMIT, out=$out_dir)"
    sbatch \
        --job-name="$name" \
        --time="$TIME_LIMIT" \
        --export="ALL,ORBIT_WARS_CONFIG=$config,ORBIT_WARS_MAX_RUNTIME_HOURS=$MAX_RUNTIME_HOURS,ORBIT_WARS_OUTPUT_DIR=$out_dir" \
        "$SBATCH_SCRIPT"
done
