import json
import random
from pathlib import Path

import torch
from owl.replay import ReplayRecorder


class _FakeEnv:
    def state_snapshot(self, env_index: int) -> dict[str, object]:
        return {"env_index": env_index, "step": 0}

    def terminal_snapshot(self, env_index: int) -> dict[str, object]:
        return {"env_index": env_index, "step": 3}


def test_replay_recorder_samples_game_ordinals_and_writes_jsonl(tmp_path: Path) -> None:
    recorder = ReplayRecorder(
        output_path=tmp_path / "sample.jsonl",
        source="test",
        player_count=2,
        total_games=4,
        sample_games=2,
        metadata={"checkpoint": "x"},
        rng=random.Random(4),
    )
    env = _FakeEnv()
    assignments = torch.tensor([0, 1, -1, -1])
    start_mask = torch.tensor([True, True, False, False])

    for game_ordinal in range(4):
        recorder.start_episode(
            env,
            0,
            game_ordinal=game_ordinal,
            assignments=assignments,
            start_mask=start_mask,
        )
        recorder.record_step(
            env,
            terminal_envs={0},
            rewards=torch.tensor([[1.0, -1.0, 0.0, 0.0]]),
            dones=torch.tensor([[True, True, True, True]]),
        )

    rows = [
        json.loads(line)
        for line in (tmp_path / "sample.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["game_ordinal"] for row in rows] == sorted(
        recorder.sampled_game_ordinals
    )
    assert len(rows) == 2
    assert rows[0]["frames"][-1]["terminal"]
    assert rows[0]["metadata"] == {"checkpoint": "x"}
