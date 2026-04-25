#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import (
    ApiGetEpisodeReplayRequest,
    KaggleApi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Kaggle competition episode replays."
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
        help="Directory to save the replay file in; defaults to the current directory",
    )
    return parser.parse_args()


def download_replay(kaggle: Any, episode_id: int, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    outfile = save_dir / f"replay-{episode_id}.json"

    request = ApiGetEpisodeReplayRequest()
    request.episode_id = episode_id
    response: Any = kaggle.competitions.competition_api_client.get_episode_replay(
        request
    )
    response.raise_for_status()

    with outfile.open("wb") as replay_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                replay_file.write(chunk)

    return outfile


def main() -> None:
    args = parse_args()

    api = KaggleApi()
    api.authenticate()

    with api.build_kaggle_client() as kaggle:
        for episode_id in args.episode_ids:
            outfile = download_replay(kaggle, episode_id, args.save_dir)
            print(f"Replay downloaded to: {outfile}")


if __name__ == "__main__":
    main()
