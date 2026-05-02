from __future__ import annotations

import importlib.util
import re
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

_BENCHMARK_PATH = Path(__file__).parents[2] / "scripts" / "benchmark_checkpoints.py"
_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "benchmark_checkpoints",
    _BENCHMARK_PATH,
)
assert _BENCHMARK_SPEC is not None
assert _BENCHMARK_SPEC.loader is not None
benchmark_checkpoints = importlib.util.module_from_spec(_BENCHMARK_SPEC)
sys.modules["benchmark_checkpoints"] = benchmark_checkpoints
_BENCHMARK_SPEC.loader.exec_module(benchmark_checkpoints)


def test_assignment_pattern_assigns_one_model_per_two_player_game() -> None:
    assignments = torch.full((1, 4), -1, dtype=torch.int64)
    active_slots = torch.tensor([True, False, True, False])

    benchmark_checkpoints._assign_episode_models(
        assignments,
        0,
        active_slots=active_slots,
        player_count=2,
    )

    assert assignments.tolist() == [[0, -1, 1, -1]]


def test_assignment_pattern_assigns_two_models_per_four_player_game() -> None:
    pattern = benchmark_checkpoints._assignment_pattern(player_count=4)

    assert pattern.count(0) == 2
    assert pattern.count(1) == 2


def test_validate_args_requires_even_game_count() -> None:
    benchmark_checkpoints._validate_args(Namespace(n_games=10, n_envs=1))

    with pytest.raises(ValueError, match="must be even"):
        benchmark_checkpoints._validate_args(Namespace(n_games=9, n_envs=1))


def test_record_terminal_result_counts_winners_by_checkpoint() -> None:
    stats = benchmark_checkpoints.MatchupStats.empty()
    assignment = torch.tensor([0, 0, 1, 1])
    start_mask = torch.tensor([True, True, True, True])
    returns = torch.tensor([0.0, -1.0, 0.0, -1.0])

    benchmark_checkpoints._record_terminal_result(
        stats,
        assignment,
        start_mask,
        returns,
    )

    assert stats.player_games == [2, 2]
    assert stats.wins == [1, 1]


def test_record_terminal_result_ignores_inactive_two_player_slots() -> None:
    stats = benchmark_checkpoints.MatchupStats.empty()
    assignment = torch.tensor([-1, 0, -1, 1])
    start_mask = torch.tensor([False, True, False, True])
    returns = torch.tensor([0.0, 1.0, 0.0, -1.0])

    benchmark_checkpoints._record_terminal_result(
        stats,
        assignment,
        start_mask,
        returns,
    )

    assert stats.player_games == [1, 1]
    assert stats.wins == [1, 0]


def test_checkpoint_config_path_requires_checkpoint_parent_config(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")

    assert benchmark_checkpoints._checkpoint_config_path(checkpoint_path) == config_path


def test_checkpoint_config_path_rejects_missing_config(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    expected = f"expected checkpoint config at {tmp_path / 'config.yaml'}"

    with pytest.raises(ValueError, match=re.escape(expected)):
        benchmark_checkpoints._checkpoint_config_path(checkpoint_path)
