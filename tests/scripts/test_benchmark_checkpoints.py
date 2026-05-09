from __future__ import annotations

import importlib.util
import re
import sys
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from owl.rl import ACTION_ENTITY_SLOTS, ObsBatch

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

    assert pattern == (0, 1, 1, 0)
    assert pattern.count(0) == 2
    assert pattern.count(1) == 2


def test_validate_args_requires_even_game_count() -> None:
    benchmark_checkpoints._validate_args(
        Namespace(n_games=10, n_envs=5, save_replay_games=0)
    )

    with pytest.raises(ValueError, match="must be even"):
        benchmark_checkpoints._validate_args(
            Namespace(n_games=9, n_envs=1, save_replay_games=0)
        )


def test_validate_args_requires_games_per_player_count_divisible_by_envs() -> None:
    with pytest.raises(ValueError, match="must be divisible by --n-envs"):
        benchmark_checkpoints._validate_args(
            Namespace(n_games=10, n_envs=3, save_replay_games=0)
        )


def test_validate_args_requires_even_replay_count() -> None:
    with pytest.raises(ValueError, match="--save-replay-games must be even"):
        benchmark_checkpoints._validate_args(
            Namespace(n_games=10, n_envs=5, save_replay_games=1)
        )


def test_record_terminal_result_counts_model_winner_by_game() -> None:
    stats = benchmark_checkpoints.MatchupStats.empty()
    assignment = torch.tensor([0, 0, 1, 1])
    start_mask = torch.tensor([True, True, True, True])
    returns = torch.tensor([1.0, 1.0, -1.0, -1.0])

    benchmark_checkpoints._record_terminal_result(
        stats,
        assignment,
        start_mask,
        returns,
    )

    assert stats.model_games == [1, 1]
    assert stats.wins == [1.0, 0.0]


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

    assert stats.model_games == [1, 1]
    assert stats.wins == [1.0, 0.0]


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


def test_actions_for_assignments_uses_checkpoint_autocast_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def __init__(self, *, launch_value: bool, ship_value: int) -> None:
            self.launch_value = launch_value
            self.ship_value = ship_value

        def __call__(
            self,
            obs: ObsBatch,  # noqa: ARG002
            *,
            deterministic: bool = False,
        ) -> SimpleNamespace:
            shape = (1, 4, ACTION_ENTITY_SLOTS, 1)
            actions = benchmark_checkpoints.PureActions(
                launch=torch.full(shape, self.launch_value, dtype=torch.bool),
                ships=torch.full(shape, self.ship_value, dtype=torch.int64),
                angle=torch.zeros(shape, dtype=torch.float32),
            )
            assert deterministic
            return SimpleNamespace(actions=actions)

    seen: list[tuple[str, torch.device]] = []

    @contextmanager
    def fake_autocast_context(
        cfg: Namespace,
        device: torch.device,
    ) -> Iterator[None]:
        seen.append((cfg.dtype, device))
        yield

    monkeypatch.setattr(
        benchmark_checkpoints,
        "autocast_context",
        fake_autocast_context,
    )
    obs = ObsBatch(
        planets=torch.zeros((1, 1, 1)),
        orbiting_planets=torch.zeros((1, 1), dtype=torch.bool),
        fleets=torch.zeros((1, 1, 1)),
        comets=torch.zeros((1, 1, 1)),
        entity_mask=torch.zeros((1, 1), dtype=torch.bool),
        still_playing=torch.ones((1, 4), dtype=torch.bool),
        global_features=torch.zeros((1, 1)),
        can_act=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
        max_launch=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64),
    )
    assignments = torch.tensor([[0, 1, 0, 1]])

    actions = benchmark_checkpoints._actions_for_assignments(
        obs,
        assignments,
        model_a=FakeModel(launch_value=True, ship_value=3),
        model_b=FakeModel(launch_value=False, ship_value=7),
        config_a=Namespace(dtype="bfloat16"),
        config_b=Namespace(dtype="float32"),
        device=torch.device("cpu"),
        deterministic=True,
    )

    assert seen == [
        ("bfloat16", torch.device("cpu")),
        ("float32", torch.device("cpu")),
    ]
    assert actions.launch[0, :, 0, 0].tolist() == [True, False, True, False]
    assert actions.ships[0, :, 0, 0].tolist() == [3, 7, 3, 7]


def test_select_actions_handles_discrete_target_bundles() -> None:
    shape = (1, 4, ACTION_ENTITY_SLOTS, 1)
    actions_a = benchmark_checkpoints.DiscreteTargetActions(
        launch=torch.full(shape, True, dtype=torch.bool),
        target=torch.full(shape, 3, dtype=torch.int64),
        ships=torch.full(shape, 5, dtype=torch.int64),
    )
    actions_b = benchmark_checkpoints.DiscreteTargetActions(
        launch=torch.full(shape, False, dtype=torch.bool),
        target=torch.full(shape, 7, dtype=torch.int64),
        ships=torch.full(shape, 11, dtype=torch.int64),
    )

    selected = benchmark_checkpoints._select_actions(
        actions_a,
        actions_b,
        torch.tensor([[True, False, True, False]]),
    )

    assert isinstance(selected, benchmark_checkpoints.DiscreteTargetActions)
    assert selected.target[0, :, 0, 0].tolist() == [3, 7, 3, 7]
    assert selected.ships[0, :, 0, 0].tolist() == [5, 11, 5, 11]


def test_select_actions_handles_discrete_target_bin_bundles() -> None:
    shape = (1, 4, ACTION_ENTITY_SLOTS)
    actions_a = benchmark_checkpoints.DiscreteTargetBinActions(
        target=torch.full(shape, 2, dtype=torch.int64),
        fleet_bin=torch.full(shape, 4, dtype=torch.int64),
    )
    actions_b = benchmark_checkpoints.DiscreteTargetBinActions(
        target=torch.full(shape, 6, dtype=torch.int64),
        fleet_bin=torch.full(shape, 8, dtype=torch.int64),
    )

    selected = benchmark_checkpoints._select_actions(
        actions_a,
        actions_b,
        torch.tensor([[True, False, True, False]]),
    )

    assert isinstance(selected, benchmark_checkpoints.DiscreteTargetBinActions)
    assert selected.target[0, :, 0].tolist() == [2, 6, 2, 6]
    assert selected.fleet_bin[0, :, 0].tolist() == [4, 8, 4, 8]
