#!/usr/bin/env bash
set -euo pipefail

REPLAY_FIXTURE_DIR="${ORBIT_WARS_PARITY_FIXTURE_DIR:-tests/fixtures/orbit_wars_replays}"
GENERATION_FIXTURE="tests/fixtures/generation/reference_generation.json"
REPLAY_FIXTURE_PARENT="$(dirname "$REPLAY_FIXTURE_DIR")"

if [ "$#" -eq 0 ]; then
  EPISODE_IDS=(75373897 75377525)
else
  EPISODE_IDS=("$@")
fi

mkdir -p "$REPLAY_FIXTURE_DIR"
mkdir -p "$(dirname "$GENERATION_FIXTURE")"
mkdir -p "$REPLAY_FIXTURE_PARENT"

GENERATION_TMP="$(mktemp "${GENERATION_FIXTURE}.tmp.XXXXXX")"
REPLAY_TMP_DIR="$(mktemp -d "${REPLAY_FIXTURE_DIR}.tmp.XXXXXX")"

cleanup() {
  rm -f "$GENERATION_TMP"
  rm -rf "$REPLAY_TMP_DIR"
}
trap cleanup EXIT

uv run python scripts/generate_reference_fixtures.py --outfile "$GENERATION_TMP"
uv run python scripts/download_replays.py "${EPISODE_IDS[@]}" --save-dir "$REPLAY_TMP_DIR"

mv "$GENERATION_TMP" "$GENERATION_FIXTURE"
rm -f "$REPLAY_FIXTURE_DIR"/replay-*.jsonl
mv "$REPLAY_TMP_DIR"/replay-*.jsonl "$REPLAY_FIXTURE_DIR"/

echo "Regenerated Orbit Wars test fixtures."
echo "Next: just prepare"
