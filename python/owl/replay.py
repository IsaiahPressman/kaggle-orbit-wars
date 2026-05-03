from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from owl.rl import VectorizedEnv


@dataclass
class _ActiveReplay:
    game_ordinal: int
    env_index: int
    model_assignments: list[int]
    start_mask: list[bool]
    frames: list[dict[str, Any]] = field(default_factory=list)


class ReplayRecorder:
    def __init__(
        self,
        *,
        output_path: Path,
        source: str,
        player_count: int,
        total_games: int,
        sample_games: int,
        metadata: dict[str, Any],
        rng: random.Random,
    ) -> None:
        if sample_games < 0:
            raise ValueError("sample_games must be non-negative")
        if sample_games > total_games:
            raise ValueError("sample_games must be <= total_games")

        self.output_path = output_path
        self.source = source
        self.player_count = player_count
        self.total_games = total_games
        self.metadata = metadata
        self.sampled_game_ordinals = set(rng.sample(range(total_games), sample_games))
        self._active: dict[int, _ActiveReplay] = {}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    def start_episode(
        self,
        env: VectorizedEnv,
        env_index: int,
        *,
        game_ordinal: int,
        assignments: torch.Tensor,
        start_mask: torch.Tensor,
    ) -> None:
        self._active.pop(env_index, None)
        if game_ordinal not in self.sampled_game_ordinals:
            return

        replay = _ActiveReplay(
            game_ordinal=game_ordinal,
            env_index=env_index,
            model_assignments=_tensor_list(assignments),
            start_mask=_tensor_list(start_mask),
        )
        replay.frames.append(
            self._frame(
                env.state_snapshot(env_index),
                terminal=False,
                rewards=None,
                dones=None,
            )
        )
        self._active[env_index] = replay

    def record_step(
        self,
        env: VectorizedEnv,
        *,
        terminal_envs: set[int],
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        for env_index, replay in list(self._active.items()):
            if env_index in terminal_envs:
                snapshot = env.terminal_snapshot(env_index)
                if snapshot is None:
                    raise RuntimeError(f"missing terminal snapshot for env {env_index}")
                replay.frames.append(
                    self._frame(
                        snapshot,
                        terminal=True,
                        rewards=_tensor_list(rewards[env_index]),
                        dones=_tensor_list(dones[env_index]),
                    )
                )
                self._write_episode(replay)
                del self._active[env_index]
            else:
                replay.frames.append(
                    self._frame(
                        env.state_snapshot(env_index),
                        terminal=False,
                        rewards=_tensor_list(rewards[env_index]),
                        dones=_tensor_list(dones[env_index]),
                    )
                )

    def _frame(
        self,
        snapshot: dict[str, Any],
        *,
        terminal: bool,
        rewards: list[float] | None,
        dones: list[bool] | None,
    ) -> dict[str, Any]:
        frame = dict(snapshot)
        frame["terminal"] = terminal
        if rewards is not None:
            frame["rewards"] = rewards
        if dones is not None:
            frame["dones"] = dones
        return frame

    def _write_episode(self, replay: _ActiveReplay) -> None:
        row = {
            "schema_version": 1,
            "source": self.source,
            "player_count": self.player_count,
            "game_ordinal": replay.game_ordinal,
            "env_index": replay.env_index,
            "model_assignments": replay.model_assignments,
            "start_mask": replay.start_mask,
            "metadata": self.metadata,
            "frames": replay.frames,
        }
        with self.output_path.open("a", encoding="utf-8") as replay_file:
            replay_file.write(json.dumps(row, separators=(",", ":")))
            replay_file.write("\n")


def _tensor_list(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()
