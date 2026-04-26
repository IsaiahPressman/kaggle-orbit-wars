#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import (
    ApiGetEpisodeReplayRequest,
    KaggleApi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Kaggle Orbit Wars replay parity fixtures."
    )
    parser.add_argument(
        "episode_ids",
        type=int,
        nargs="+",
        help="Kaggle episode ID(s) to download",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to save replay-<episode-id>.jsonl fixtures in",
    )
    return parser.parse_args()


def extract_rows(replay: dict[str, Any]) -> Iterable[dict[str, Any]]:
    episode_id = int(replay["info"]["EpisodeId"])
    configuration = replay["configuration"]
    steps = replay["steps"]
    player_count = len(steps[0])

    for step in range(1, len(steps)):
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


def download_replay_fixture(
    kaggle: Any,
    episode_id: int,
    save_dir: Path,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    outfile = save_dir / f"replay-{episode_id}.jsonl"

    request = ApiGetEpisodeReplayRequest()
    request.episode_id = episode_id
    response: Any = kaggle.competitions.competition_api_client.get_episode_replay(
        request
    )
    response.raise_for_status()
    replay = response.json()
    if not isinstance(replay, dict):
        raise TypeError(f"Replay {episode_id} response was not a JSON object")

    with outfile.open("w", encoding="utf-8") as replay_file:
        for row in extract_rows(replay):
            replay_file.write(json.dumps(row, separators=(",", ":")))
            replay_file.write("\n")

    return outfile


def main() -> None:
    args = parse_args()

    api = KaggleApi()
    api.authenticate()

    with api.build_kaggle_client() as kaggle:
        for episode_id in args.episode_ids:
            outfile = download_replay_fixture(
                kaggle,
                episode_id,
                args.save_dir,
            )
            print(f"Replay fixture downloaded to: {outfile}")


if __name__ == "__main__":
    main()
