#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_EPISODES = (75373897, 75377525)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract compact Orbit Wars replay transition fixtures."
    )
    parser.add_argument(
        "episode_ids",
        type=int,
        nargs="*",
        default=DEFAULT_EPISODES,
        help="Kaggle episode IDs to extract; defaults to the documented references.",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing raw replay-<episode-id>.json files.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("tests/fixtures/orbit_wars_replays"),
        help="Directory to write compact JSONL fixtures.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Optional comma-separated transition step numbers to extract.",
    )
    return parser.parse_args()


def parse_steps(raw_steps: str | None) -> set[int] | None:
    if raw_steps is None:
        return None
    return {int(step.strip()) for step in raw_steps.split(",") if step.strip()}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as replay_file:
        data: Any = json.load(replay_file)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return data


def extract_rows(
    replay: dict[str, Any], selected_steps: set[int] | None
) -> Iterable[dict[str, Any]]:
    episode_id = int(replay["info"]["EpisodeId"])
    configuration = replay["configuration"]
    steps = replay["steps"]
    player_count = len(steps[0])

    for step in range(1, len(steps)):
        if selected_steps is not None and step not in selected_steps:
            continue

        before = steps[step - 1][0]["observation"]
        expected = steps[step][0]["observation"]
        if not isinstance(before, dict) or not before.get("planets"):
            continue
        if not isinstance(expected, dict) or not expected.get("planets"):
            continue

        yield {
            "episode_id": episode_id,
            "players": player_count,
            "step": step,
            "configuration": configuration,
            "before": before,
            "actions": [
                steps[step][player]["action"] for player in range(player_count)
            ],
            "expected": expected,
        }


def write_fixture(
    replay_dir: Path,
    fixture_dir: Path,
    episode_id: int,
    selected_steps: set[int] | None,
) -> Path:
    replay_path = replay_dir / f"replay-{episode_id}.json"
    replay = load_json(replay_path)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    outfile = fixture_dir / f"replay-{episode_id}.jsonl"

    with outfile.open("w", encoding="utf-8") as fixture_file:
        for row in extract_rows(replay, selected_steps):
            fixture_file.write(json.dumps(row, separators=(",", ":")))
            fixture_file.write("\n")

    return outfile


def main() -> None:
    args = parse_args()
    selected_steps = parse_steps(args.steps)

    for episode_id in args.episode_ids:
        outfile = write_fixture(
            replay_dir=args.replay_dir,
            fixture_dir=args.fixture_dir,
            episode_id=episode_id,
            selected_steps=selected_steps,
        )
        print(f"Fixture written to: {outfile}")


if __name__ == "__main__":
    main()
