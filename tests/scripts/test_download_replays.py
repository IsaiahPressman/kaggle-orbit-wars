from __future__ import annotations

import importlib.util
from pathlib import Path

_DOWNLOAD_REPLAYS_PATH = Path(__file__).parents[2] / "scripts" / "download_replays.py"
_DOWNLOAD_REPLAYS_SPEC = importlib.util.spec_from_file_location(
    "download_replays",
    _DOWNLOAD_REPLAYS_PATH,
)
assert _DOWNLOAD_REPLAYS_SPEC is not None
assert _DOWNLOAD_REPLAYS_SPEC.loader is not None
download_replays = importlib.util.module_from_spec(_DOWNLOAD_REPLAYS_SPEC)
_DOWNLOAD_REPLAYS_SPEC.loader.exec_module(download_replays)


def test_normalize_actions_keeps_numeric_triples() -> None:
    assert download_replays.normalize_actions(
        [
            [1, 0.25, 3],
            [2.0, -0.5, 4.0],
        ]
    ) == [
        [1.0, 0.25, 3.0],
        [2.0, -0.5, 4.0],
    ]


def test_normalize_actions_drops_malformed_entries() -> None:
    assert download_replays.normalize_actions(
        [
            [1, 0.25],
            [2, 0.5, 3, 4],
            ["2", 0.5, 3],
            [True, 0.5, 3],
            [3, float("nan"), 3],
            {"from": 1},
            [4, 1.0, 5],
        ]
    ) == [[4.0, 1.0, 5.0]]


def test_normalize_actions_rejects_non_list_action() -> None:
    assert download_replays.normalize_actions(None) == []
    assert download_replays.normalize_actions({"action": [1, 2, 3]}) == []


def test_extract_rows_stores_normalized_actions_and_player_results() -> None:
    replay = {
        "info": {"EpisodeId": 123},
        "configuration": {"episodeSteps": 500, "shipSpeed": 6.0, "cometSpeed": 4.0},
        "steps": [
            [
                {
                    "observation": {
                        "planets": [[0, 0, 1, 2, 3, 4, 5]],
                        "fleets": [],
                        "comets": [],
                        "step": 0,
                    },
                },
                {"observation": {}},
            ],
            [
                {
                    "action": [[0, 1.5, 2], ["bad", 0, 1]],
                    "status": "ACTIVE",
                    "reward": 0,
                    "observation": {
                        "planets": [[0, 0, 1, 2, 3, 4, 5]],
                        "fleets": [],
                        "comets": [],
                        "step": 1,
                    },
                },
                {
                    "action": None,
                    "status": "DONE",
                    "reward": 1,
                    "observation": {},
                },
            ],
        ],
    }

    rows = list(download_replays.extract_rows(replay))

    assert rows == [
        {
            "episode_id": 123,
            "players": 2,
            "step": 1,
            "configuration": replay["configuration"],
            "before": replay["steps"][0][0]["observation"],
            "actions": [[[0.0, 1.5, 2.0]], []],
            "results": [
                {"status": "ACTIVE", "reward": 0},
                {"status": "DONE", "reward": 1},
            ],
            "expected": replay["steps"][1][0]["observation"],
        }
    ]
