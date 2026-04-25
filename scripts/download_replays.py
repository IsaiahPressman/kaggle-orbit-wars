#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from extract_replay_fixtures import extract_rows, parse_steps
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
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Optional comma-separated transition step numbers to save",
    )
    return parser.parse_args()


def download_replay_fixture(
    kaggle: Any,
    episode_id: int,
    save_dir: Path,
    selected_steps: set[int] | None,
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
        for row in extract_rows(replay, selected_steps):
            replay_file.write(json.dumps(row, separators=(",", ":")))
            replay_file.write("\n")

    return outfile


def main() -> None:
    args = parse_args()
    selected_steps = parse_steps(args.steps)

    api = KaggleApi()
    api.authenticate()

    with api.build_kaggle_client() as kaggle:
        for episode_id in args.episode_ids:
            outfile = download_replay_fixture(
                kaggle,
                episode_id,
                args.save_dir,
                selected_steps,
            )
            print(f"Replay fixture downloaded to: {outfile}")


if __name__ == "__main__":
    main()
