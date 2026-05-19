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
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    ActionDiscreteTargetsConfig,
    ActionPureConfig,
    DecodedLaunchActions,
    DiscreteTargetActionMask,
    EntityBasedConfig,
    EntityBasedExtV1Config,
    ObsBatch,
    PureActionMask,
)

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

    class FakeEnv:
        def __init__(self) -> None:
            self.observed_specs: list[tuple[str, str]] = []
            self.decoded_specs: list[str] = []

        def observation_for_spec(
            self,
            obs_spec: benchmark_checkpoints.ObsConfig,
            action_spec: ActionPureConfig,
        ) -> ObsBatch:
            self.observed_specs.append((obs_spec.obs_spec, action_spec.action_spec))
            return ObsBatch(
                planets=torch.zeros((1, 1, 1)),
                orbiting_planets=torch.zeros((1, 1), dtype=torch.bool),
                fleets=torch.zeros((1, 1, 1)),
                comets=torch.zeros((1, 1, 1)),
                entity_mask=torch.zeros((1, 1), dtype=torch.bool),
                still_playing=torch.ones((1, 4), dtype=torch.bool),
                global_features=torch.zeros((1, 1)),
                action_mask=PureActionMask(
                    can_act=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
                    max_launch=torch.zeros(
                        (1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64
                    ),
                ),
            )

        def decode_actions(
            self,
            actions: benchmark_checkpoints.ActionBundle,
            *,
            action_spec: ActionPureConfig,
        ) -> DecodedLaunchActions:
            self.decoded_specs.append(action_spec.action_spec)
            valid = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool)
            ships = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
            valid[:, :, 0] = actions.launch[:, :, 0, 0]
            ships[:, :, 0] = torch.where(
                valid[:, :, 0],
                actions.ships[:, :, 0, 0],
                torch.zeros_like(actions.ships[:, :, 0, 0]),
            )
            return DecodedLaunchActions(
                valid=valid,
                from_planet_id=torch.zeros_like(ships),
                angle=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.float32),
                ships=ships,
            )

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
    env = FakeEnv()
    assignments = torch.tensor([[0, 1, 0, 1]])

    actions = benchmark_checkpoints._actions_for_assignments(
        env,
        assignments,
        model_a=FakeModel(launch_value=True, ship_value=3),
        model_b=FakeModel(launch_value=False, ship_value=7),
        obs_spec_a=EntityBasedConfig(),
        obs_spec_b=EntityBasedConfig(),
        action_spec_a=ActionPureConfig(max_per_planet_launches=1),
        action_spec_b=ActionPureConfig(max_per_planet_launches=1),
        config_a=Namespace(dtype="bfloat16"),
        config_b=Namespace(dtype="float32"),
        device=torch.device("cpu"),
        deterministic=True,
    )

    assert env.observed_specs == [
        ("entity_based", "pure"),
        ("entity_based", "pure"),
    ]
    assert env.decoded_specs == ["pure", "pure"]
    assert seen == [
        ("bfloat16", torch.device("cpu")),
        ("float32", torch.device("cpu")),
    ]
    assert actions.valid[0, :, 0].tolist() == [True, False, True, False]
    assert actions.ships[0, :, 0].tolist() == [3, 0, 3, 0]


def test_actions_for_assignments_uses_each_checkpoint_action_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.observed_specs: list[tuple[str, str]] = []
            self.decoded_specs: list[str] = []

        def observation_for_spec(
            self,
            obs_spec: benchmark_checkpoints.ObsConfig,
            action_spec: benchmark_checkpoints.ActionConfig,
        ) -> ObsBatch:
            self.observed_specs.append((obs_spec.obs_spec, action_spec.action_spec))
            if isinstance(action_spec, ActionDiscreteTargetsConfig):
                action_mask = DiscreteTargetActionMask(
                    can_act=torch.zeros(
                        (1, 4, ACTION_ENTITY_SLOTS, ACTION_ENTITY_SLOTS),
                        dtype=torch.bool,
                    ),
                    max_launch=torch.zeros(
                        (1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64
                    ),
                )
            else:
                action_mask = PureActionMask(
                    can_act=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
                    max_launch=torch.zeros(
                        (1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64
                    ),
                )
            return ObsBatch(
                planets=torch.zeros((1, 1, obs_spec.planet_channels)),
                orbiting_planets=torch.zeros((1, 1), dtype=torch.bool),
                fleets=torch.zeros((1, 1, obs_spec.fleet_channels)),
                comets=torch.zeros((1, 1, obs_spec.comet_channels)),
                entity_mask=torch.zeros((1, obs_spec.max_entities), dtype=torch.bool),
                still_playing=torch.ones((1, 4), dtype=torch.bool),
                global_features=torch.zeros((1, obs_spec.global_channels)),
                action_mask=action_mask,
            )

        def decode_actions(
            self,
            actions: benchmark_checkpoints.ActionBundle,  # noqa: ARG002
            *,
            action_spec: benchmark_checkpoints.ActionConfig,
        ) -> DecodedLaunchActions:
            self.decoded_specs.append(action_spec.action_spec)
            valid = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool)
            valid[:, :, 0] = True
            ships = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
            ships[:, :, 0] = (
                7 if isinstance(action_spec, ActionDiscreteTargetsConfig) else 3
            )
            return DecodedLaunchActions(
                valid=valid,
                from_planet_id=torch.zeros_like(ships),
                angle=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.float32),
                ships=ships,
            )

    class FakeModel:
        def __call__(
            self,
            obs: ObsBatch,
            *,
            deterministic: bool = False,
        ) -> SimpleNamespace:
            assert deterministic
            shape = (1, 4, ACTION_ENTITY_SLOTS, 1)
            if isinstance(obs.action_mask, DiscreteTargetActionMask):
                actions = benchmark_checkpoints.DiscreteTargetActions(
                    launch=torch.ones(shape, dtype=torch.bool),
                    target=torch.zeros(shape, dtype=torch.int64),
                    ships=torch.full(shape, 7, dtype=torch.int64),
                )
            else:
                actions = benchmark_checkpoints.PureActions(
                    launch=torch.ones(shape, dtype=torch.bool),
                    angle=torch.zeros(shape, dtype=torch.float32),
                    ships=torch.full(shape, 3, dtype=torch.int64),
                )
            return SimpleNamespace(actions=actions)

    @contextmanager
    def fake_autocast_context(
        cfg: Namespace,  # noqa: ARG001
        device: torch.device,  # noqa: ARG001
    ) -> Iterator[None]:
        yield

    monkeypatch.setattr(
        benchmark_checkpoints,
        "autocast_context",
        fake_autocast_context,
    )
    env = FakeEnv()

    actions = benchmark_checkpoints._actions_for_assignments(
        env,
        torch.tensor([[0, 1, 0, 1]]),
        model_a=FakeModel(),
        model_b=FakeModel(),
        obs_spec_a=EntityBasedConfig(max_entities=128),
        obs_spec_b=EntityBasedExtV1Config(max_entities=256, ship_count_one_hot_max=5),
        action_spec_a=ActionPureConfig(max_per_planet_launches=1),
        action_spec_b=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
        config_a=Namespace(dtype="float32"),
        config_b=Namespace(dtype="float32"),
        device=torch.device("cpu"),
        deterministic=True,
    )

    assert env.observed_specs == [
        ("entity_based", "pure"),
        ("entity_based_ext_v1", "discrete_targets"),
    ]
    assert env.decoded_specs == ["pure", "discrete_targets"]
    assert actions.ships[0, :, 0].tolist() == [3, 7, 3, 7]


def test_actions_for_assignments_and_hidden_returns_next_hidden_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def __init__(self, *, launch_value: bool, initial_seen: float) -> None:
            self.launch_value = launch_value
            self.initial_seen = initial_seen
            self.seen_hidden: list[torch.Tensor] = []

        def __call__(
            self,
            obs: ObsBatch,
            *,
            deterministic: bool = False,
            hidden_state: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            assert deterministic
            assert hidden_state is not None
            self.seen_hidden.append(hidden_state.detach().cpu().clone())
            shape = (obs.global_features.shape[0], 4, ACTION_ENTITY_SLOTS, 1)
            actions = benchmark_checkpoints.PureActions(
                launch=torch.full(shape, self.launch_value, dtype=torch.bool),
                angle=torch.zeros(shape, dtype=torch.float32),
                ships=torch.ones(shape, dtype=torch.int64),
            )
            return SimpleNamespace(
                actions=actions,
                next_hidden_state=hidden_state + self.initial_seen,
            )

    class FakeEnv:
        def observation_for_spec(
            self,
            obs_spec: benchmark_checkpoints.ObsConfig,
            action_spec: ActionPureConfig,  # noqa: ARG002
        ) -> ObsBatch:
            return ObsBatch(
                planets=torch.zeros((2, 1, obs_spec.planet_channels)),
                orbiting_planets=torch.zeros((2, 1), dtype=torch.bool),
                fleets=torch.zeros((2, 1, obs_spec.fleet_channels)),
                comets=torch.zeros((2, 1, obs_spec.comet_channels)),
                entity_mask=torch.zeros((2, obs_spec.max_entities), dtype=torch.bool),
                still_playing=torch.ones((2, 4), dtype=torch.bool),
                global_features=torch.zeros((2, obs_spec.global_channels)),
                action_mask=PureActionMask(
                    can_act=torch.zeros((2, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
                    max_launch=torch.zeros(
                        (2, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64
                    ),
                ),
            )

        def decode_actions(
            self,
            actions: benchmark_checkpoints.ActionBundle,
            *,
            action_spec: ActionPureConfig,  # noqa: ARG002
        ) -> DecodedLaunchActions:
            valid = torch.zeros((2, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool)
            valid[:, :, 0] = actions.launch[:, :, 0, 0]
            ships = torch.zeros((2, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
            ships[:, :, 0] = actions.ships[:, :, 0, 0]
            return DecodedLaunchActions(
                valid=valid,
                from_planet_id=torch.zeros_like(ships),
                angle=torch.zeros((2, 4, ACTION_ENTITY_SLOTS), dtype=torch.float32),
                ships=ships,
            )

    @contextmanager
    def fake_autocast_context(
        cfg: Namespace,  # noqa: ARG001
        device: torch.device,  # noqa: ARG001
    ) -> Iterator[None]:
        yield

    monkeypatch.setattr(
        benchmark_checkpoints,
        "autocast_context",
        fake_autocast_context,
    )
    model_a = FakeModel(launch_value=True, initial_seen=1.0)
    model_b = FakeModel(launch_value=False, initial_seen=10.0)

    actions, hidden_a, hidden_b = (
        benchmark_checkpoints._actions_for_assignments_and_hidden(
            FakeEnv(),
            torch.tensor(
                [
                    [0, 1, 0, 1],
                    [1, 0, 1, 0],
                ]
            ),
            model_a=model_a,
            model_b=model_b,
            hidden_a=torch.tensor([1.0, 2.0]),
            hidden_b=torch.tensor([3.0, 4.0]),
            obs_spec_a=EntityBasedConfig(),
            obs_spec_b=EntityBasedConfig(),
            action_spec_a=ActionPureConfig(max_per_planet_launches=1),
            action_spec_b=ActionPureConfig(max_per_planet_launches=1),
            config_a=Namespace(dtype="float32"),
            config_b=Namespace(dtype="float32"),
            device=torch.device("cpu"),
            deterministic=True,
        )
    )

    assert [hidden.tolist() for hidden in model_a.seen_hidden] == [[1.0, 2.0]]
    assert [hidden.tolist() for hidden in model_b.seen_hidden] == [[3.0, 4.0]]
    assert hidden_a.tolist() == [2.0, 3.0]
    assert hidden_b.tolist() == [13.0, 14.0]
    assert actions.valid[:, :, 0].tolist() == [
        [True, False, True, False],
        [False, True, False, True],
    ]


def test_select_decoded_actions_pads_to_larger_action_capacity() -> None:
    actions_a = DecodedLaunchActions(
        valid=torch.ones((1, 4, 1), dtype=torch.bool),
        from_planet_id=torch.zeros((1, 4, 1), dtype=torch.int64),
        angle=torch.zeros((1, 4, 1), dtype=torch.float32),
        ships=torch.full((1, 4, 1), 3, dtype=torch.int64),
    )
    actions_b = DecodedLaunchActions(
        valid=torch.ones((1, 4, 2), dtype=torch.bool),
        from_planet_id=torch.zeros((1, 4, 2), dtype=torch.int64),
        angle=torch.zeros((1, 4, 2), dtype=torch.float32),
        ships=torch.full((1, 4, 2), 7, dtype=torch.int64),
    )

    selected = benchmark_checkpoints._select_decoded_actions(
        actions_a,
        actions_b,
        torch.tensor([[True, False, True, False]]),
    )

    assert selected.valid.shape == (1, 4, 2)
    assert selected.valid[0, :, 0].tolist() == [True, True, True, True]
    assert selected.valid[0, :, 1].tolist() == [False, True, False, True]
    assert selected.ships[0, :, 0].tolist() == [3, 7, 3, 7]
