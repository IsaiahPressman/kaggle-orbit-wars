#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/build_kaggle_submission.sh CHECKPOINT_PATH [OUTPUT_PATH]

Build the release Rust extension and write a Kaggle submission tarball.

Arguments:
  CHECKPOINT_PATH  Checkpoint/model file to include at submission archive root.
                   The script also requires config.yaml from the same directory.
  OUTPUT_PATH      Tarball path to write. Defaults to submission.tar.gz.

Options:
  -o, --output PATH  Tarball path to write.
  -h, --help         Show this help.
EOF
}

checkpoint_path=""
output_path="submission.tar.gz"
output_path_set=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    -o | --output)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a path argument" >&2
        exit 2
      fi
      output_path="$2"
      output_path_set=1
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -z "$checkpoint_path" ]]; then
        checkpoint_path="$1"
      elif [[ "$output_path_set" -eq 0 ]]; then
        output_path="$1"
        output_path_set=1
      else
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 2
      fi
      shift
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage_dir="$(mktemp -d)"

cleanup() {
  rm -rf "$stage_dir"
}
trap cleanup EXIT

cd "$repo_root"

if [[ -z "$checkpoint_path" ]]; then
  echo "CHECKPOINT_PATH is required" >&2
  usage >&2
  exit 2
fi

checkpoint_path="$(cd "$(dirname "$checkpoint_path")" && pwd)/$(basename "$checkpoint_path")"
model_config_path="$(dirname "$checkpoint_path")/config.yaml"

if [[ ! -f "$checkpoint_path" ]]; then
  echo "Checkpoint path does not exist: $checkpoint_path" >&2
  exit 1
fi

if [[ ! -f "$model_config_path" ]]; then
  echo "Adjacent model config does not exist: $model_config_path" >&2
  exit 1
fi

if [[ -f python/main.py ]]; then
  entrypoint_path="python/main.py"
elif [[ -f main.py ]]; then
  entrypoint_path="main.py"
else
  cat >&2 <<'EOF'
No Kaggle entrypoint found.

Expected python/main.py or main.py so the submission tarball can expose
main.py at archive root. Add the agent entrypoint, then rerun this script.
EOF
  exit 1
fi

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
uv run python - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(f"uv must use Python 3.11, found {sys.version}")
PY
just build-release
uv run python - <<'PY'
import owl.rs

owl.rs.assert_release_build()
PY

mkdir -p "$stage_dir/submission"
cp -R python/owl "$stage_dir/submission/owl"
find "$stage_dir/submission" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$stage_dir/submission" -type f -name "*.pyc" -delete

slim_checkpoint_path="$stage_dir/$(basename "$checkpoint_path")"
python scripts/extract_model_weights.py "$checkpoint_path" "$slim_checkpoint_path"
cp "$slim_checkpoint_path" "$stage_dir/submission/$(basename "$checkpoint_path")"
cp "$model_config_path" "$stage_dir/submission/config.yaml"
cp "$entrypoint_path" "$stage_dir/submission/main.py"

mkdir -p "$(dirname "$output_path")"
tar -C "$stage_dir/submission" -czf "$output_path" .
echo "Wrote $output_path"
