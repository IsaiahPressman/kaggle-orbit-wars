#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/build_kaggle_submission.sh CHECKPOINT_PATH [OUTPUT_PATH]

Build the release Rust extension and write a Kaggle submission tarball.

Arguments:
  CHECKPOINT_PATH  Primary checkpoint/model file to include in the submission.
                   The script also requires config.yaml from the same directory.
  OUTPUT_PATH      Tarball path to write. Defaults to submission.tar.gz.

Options:
  -o, --output PATH  Tarball path to write.
  --fallback-checkpoint PATH
                     Optional fallback checkpoint/model file to include in the
                     submission. The script also requires config.yaml from the
                     same directory.
  --quantization FORMAT
                     Optional slim-checkpoint quantization format:
                     fp32, fp8_e4m3fn, fp4_e2m1fn_x2_scaled_block16, or
                     nf5_g128_lsq_policy_last_fp8, nf4_g128_lsq,
                     nf3_nf4_structured_3p5, or nf3_g128_lsq.
                     Unique prefixes such as fp4 are accepted.
  --lora-quantization FORMAT
                     Optional LoRA adapter quantization format: fp32, fp16,
                     bf16, or any --quantization format. Defaults to bf16 when
                     --quantization is set and the checkpoint has LoRA tensors.
  -h, --help         Show this help.
EOF
}

checkpoint_path=""
fallback_checkpoint_path=""
output_path="submission.tar.gz"
output_path_set=0
quantization=""
lora_quantization=""

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
    --fallback-checkpoint)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a path argument" >&2
        exit 2
      fi
      fallback_checkpoint_path="$2"
      shift 2
      ;;
    --quantization)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a format argument" >&2
        exit 2
      fi
      quantization="$2"
      shift 2
      ;;
    --lora-quantization)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a format argument" >&2
        exit 2
      fi
      lora_quantization="$2"
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
fallback_model_config_path=""
if [[ -n "$fallback_checkpoint_path" ]]; then
  fallback_checkpoint_path="$(cd "$(dirname "$fallback_checkpoint_path")" && pwd)/$(basename "$fallback_checkpoint_path")"
  fallback_model_config_path="$(dirname "$fallback_checkpoint_path")/config.yaml"
fi

if [[ ! -f "$checkpoint_path" ]]; then
  echo "Checkpoint path does not exist: $checkpoint_path" >&2
  exit 1
fi

if [[ ! -f "$model_config_path" ]]; then
  echo "Adjacent model config does not exist: $model_config_path" >&2
  exit 1
fi
if [[ -n "$fallback_checkpoint_path" && ! -f "$fallback_checkpoint_path" ]]; then
  echo "Fallback checkpoint path does not exist: $fallback_checkpoint_path" >&2
  exit 1
fi
if [[ -n "$fallback_model_config_path" && ! -f "$fallback_model_config_path" ]]; then
  echo "Adjacent fallback model config does not exist: $fallback_model_config_path" >&2
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

uv_run=(uv run)
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  export UV_PYTHON="${UV_PYTHON:-$VIRTUAL_ENV/bin/python}"
  export UV_NO_MANAGED_PYTHON="${UV_NO_MANAGED_PYTHON:-1}"
  export PYO3_PYTHON="${PYO3_PYTHON:-$VIRTUAL_ENV/bin/python}"
  uv_run=(uv run --active)
fi

"${uv_run[@]}" python - <<'PY'
import os
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(
        f"uv must use Python 3.11, found {sys.version} at {sys.executable}"
    )

if virtual_env := os.environ.get("VIRTUAL_ENV"):
    expected_python = Path(virtual_env, "bin", "python").resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise RuntimeError(
            "uv must use the active Kaggle venv Python, "
            f"expected {expected_python}, found {actual_python}"
        )
PY
"${uv_run[@]}" maturin develop --release
"${uv_run[@]}" python - <<'PY'
import owl.rs

owl.rs.assert_release_build()
PY

mkdir -p "$stage_dir/submission"
cp -R python/owl "$stage_dir/submission/owl"
find "$stage_dir/submission" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$stage_dir/submission" -type f -name "*.pyc" -delete

copy_model_bundle() {
  local source_checkpoint_path="$1"
  local source_config_path="$2"
  local bundle_name="$3"
  local bundle_dir="$stage_dir/submission/models/$bundle_name"
  local slim_checkpoint_path="$stage_dir/${bundle_name}_checkpoint.pt"
  local extract_args

  mkdir -p "$bundle_dir"
  extract_args=(scripts/extract_model_weights.py "$source_checkpoint_path" "$slim_checkpoint_path")
  if [[ -n "$quantization" ]]; then
    extract_args+=(--quantization "$quantization")
  fi
  if [[ -n "$lora_quantization" ]]; then
    extract_args+=(--lora-quantization "$lora_quantization")
  fi
  "${uv_run[@]}" python "${extract_args[@]}"
  cp "$slim_checkpoint_path" "$bundle_dir/checkpoint.pt"
  cp "$source_config_path" "$bundle_dir/config.yaml"
}

copy_model_bundle "$checkpoint_path" "$model_config_path" primary
if [[ -n "$fallback_checkpoint_path" ]]; then
  copy_model_bundle "$fallback_checkpoint_path" "$fallback_model_config_path" fallback
fi
cp "$entrypoint_path" "$stage_dir/submission/main.py"

mkdir -p "$(dirname "$output_path")"
tar -C "$stage_dir/submission" -czf "$output_path" .
echo "Wrote $output_path"
