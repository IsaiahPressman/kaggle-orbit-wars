#!/usr/bin/env bash
set -euo pipefail

REPLAY_FIXTURE_DIR="${ORBIT_WARS_PARITY_FIXTURE_DIR:-tests/fixtures/orbit_wars_replays}"
GENERATION_FIXTURE="tests/fixtures/generation/reference_generation.json"

if [ "$#" -eq 0 ]; then
  EPISODE_IDS=(75373897 75377525)
else
  EPISODE_IDS=("$@")
fi

mkdir -p "$REPLAY_FIXTURE_DIR"
mkdir -p "$(dirname "$GENERATION_FIXTURE")"

rm -f "$REPLAY_FIXTURE_DIR"/replay-*.jsonl
rm -f "$GENERATION_FIXTURE"

uv run python scripts/generate_reference_fixtures.py --outfile "$GENERATION_FIXTURE"
uv run python scripts/download_replays.py "${EPISODE_IDS[@]}" --save-dir "$REPLAY_FIXTURE_DIR"

echo "Regenerated Orbit Wars test fixtures."
echo "Next: just prepare"
