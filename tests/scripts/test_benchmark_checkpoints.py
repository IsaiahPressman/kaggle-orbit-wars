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
from owl.int8_emulation import Int8EmulatedLinear
from owl.model import (
    LoRAConfig,
    LoRALinear,
    StatelessTransformerV1,
    StatelessTransformerV1Config,
    apply_lora_to_stateless_transformer,
    create_model,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    ActionBundle,
    ActionConfig,
    ActionDiscreteTargetsConfig,
    ActionPureConfig,
    DecodedLaunchActions,
    DiscreteTargetActionMask,
    EntityBasedConfig,
    EntityBasedExtV1Config,
    ObsBatch,
    ObsConfig,
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


def _loaded_checkpoint(
    model: object,
    *,
    obs_spec: object | None = None,
    action_spec: object | None = None,
    dtype: str = "float32",
) -> benchmark_checkpoints.LoadedCheckpoint:
    return benchmark_checkpoints.LoadedCheckpoint(
        path=Path("checkpoint.pt"),
        config=SimpleNamespace(
            env=SimpleNamespace(
                obs_spec=obs_spec or EntityBasedConfig(),
                action_spec=action_spec or ActionPureConfig(max_per_planet_launches=1),
            ),
            rl=Namespace(dtype=dtype),
        ),
        model=model,
        env_steps=0,
    )


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


def test_player_count_counts_uses_two_player_weight() -> None:
    assert benchmark_checkpoints._player_count_counts(9, 0.25) == {2: 2, 4: 7}
    assert benchmark_checkpoints._player_count_counts(9, 0.75) == {2: 7, 4: 2}


def test_validate_args_allows_weighted_odd_game_count() -> None:
    benchmark_checkpoints._validate_args(
        Namespace(
            n_games=9,
            n_envs=5,
            save_replay_games=0,
            two_player_weight=0.25,
        )
    )

    benchmark_checkpoints._validate_args(
        Namespace(
            n_games=9,
            n_envs=5,
            save_replay_games=1,
            two_player_weight=0.25,
        )
    )


def test_validate_args_requires_two_player_weight_in_range() -> None:
    with pytest.raises(ValueError, match="--two-player-weight must be in"):
        benchmark_checkpoints._validate_args(
            Namespace(
                n_games=10,
                n_envs=5,
                save_replay_games=0,
                two_player_weight=1.1,
            )
        )


def test_validate_args_allows_games_per_player_count_not_divisible_by_envs() -> None:
    benchmark_checkpoints._validate_args(
        Namespace(n_games=10, n_envs=3, save_replay_games=0, two_player_weight=0.5)
    )


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], (False, False)),
        (["--deterministic"], (True, True)),
        (["--deterministic", "a"], (True, False)),
        (["--deterministic", "b"], (False, True)),
        (["-d"], (True, True)),
        (["-d", "a"], (True, False)),
        (["-d", "b"], (False, True)),
    ],
)
def test_parse_args_accepts_optional_deterministic_target(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected: tuple[bool, bool],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_checkpoints.py",
            "checkpoint_a.pt",
            "checkpoint_b.pt",
            *extra_args,
        ],
    )

    args = benchmark_checkpoints._parse_args()
    determinism = benchmark_checkpoints._deterministic_flags(args.deterministic)

    assert (determinism.a, determinism.b) == expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["benchmark_checkpoints.py", "-d", "checkpoint_a.pt", "checkpoint_b.pt"],
            (True, True),
        ),
        (
            [
                "benchmark_checkpoints.py",
                "-d",
                "a",
                "checkpoint_a.pt",
                "checkpoint_b.pt",
            ],
            (True, False),
        ),
    ],
)
def test_parse_args_accepts_deterministic_before_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: tuple[bool, bool],
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    args = benchmark_checkpoints._parse_args()
    determinism = benchmark_checkpoints._deterministic_flags(args.deterministic)

    assert (args.checkpoint_a, args.checkpoint_b) == (
        Path("checkpoint_a.pt"),
        Path("checkpoint_b.pt"),
    )
    assert (determinism.a, determinism.b) == expected


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], (False, False)),
        (["--int8-emulation", "none"], (False, False)),
        (["--int8-emulation"], (True, True)),
        (["--int8-emulation", "a"], (True, False)),
        (["--int8-emulation", "b"], (False, True)),
    ],
)
def test_parse_args_accepts_optional_int8_emulation_target(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected: tuple[bool, bool],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_checkpoints.py",
            "checkpoint_a.pt",
            "checkpoint_b.pt",
            *extra_args,
        ],
    )

    args = benchmark_checkpoints._parse_args()
    int8_emulation = benchmark_checkpoints._int8_emulation_flags(args.int8_emulation)

    assert (int8_emulation.a, int8_emulation.b) == expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            [
                "benchmark_checkpoints.py",
                "--int8-emulation",
                "checkpoint_a.pt",
                "checkpoint_b.pt",
            ],
            (True, True),
        ),
        (
            [
                "benchmark_checkpoints.py",
                "--int8-emulation",
                "none",
                "checkpoint_a.pt",
                "checkpoint_b.pt",
            ],
            (False, False),
        ),
        (
            [
                "benchmark_checkpoints.py",
                "--int8-emulation",
                "b",
                "checkpoint_a.pt",
                "checkpoint_b.pt",
            ],
            (False, True),
        ),
    ],
)
def test_parse_args_accepts_int8_emulation_before_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: tuple[bool, bool],
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    args = benchmark_checkpoints._parse_args()
    int8_emulation = benchmark_checkpoints._int8_emulation_flags(args.int8_emulation)

    assert (args.checkpoint_a, args.checkpoint_b) == (
        Path("checkpoint_a.pt"),
        Path("checkpoint_b.pt"),
    )
    assert (int8_emulation.a, int8_emulation.b) == expected


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


def test_load_checkpoint_int8_emulation_replaces_non_output_linears(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = torch.nn.Linear(2, 3)
            self.output = torch.nn.Linear(3, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.output(torch.relu(self.body(x)))

        def get_output_layers(self) -> tuple[torch.nn.Module, ...]:
            return (self.output,)

    model = FakeModel()
    checkpoint_path = tmp_path / "checkpoint.pt"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused\n")
    torch.save({"model": model.state_dict(), "env_steps": 12}, checkpoint_path)
    config = SimpleNamespace(
        model=SimpleNamespace(),
        env=SimpleNamespace(
            obs_spec=EntityBasedConfig(), action_spec=ActionPureConfig()
        ),
        rl=Namespace(dtype="float32"),
    )

    monkeypatch.setattr(
        benchmark_checkpoints.FullConfig,
        "from_file",
        classmethod(lambda _cls, _path: config),
    )
    monkeypatch.setattr(
        benchmark_checkpoints,
        "create_model",
        lambda _model_cfg, **_kwargs: model,
    )

    loaded = benchmark_checkpoints._load_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
        int8_emulation=True,
    )

    assert loaded.int8_emulation is True
    assert loaded.env_steps == 12
    assert loaded.model.training is False
    assert isinstance(model.body, Int8EmulatedLinear)
    assert isinstance(model.output, torch.nn.Linear)


def test_load_checkpoint_folds_lora_before_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_config = StatelessTransformerV1Config(
        embed_dim=8,
        depth=1,
        n_heads=1,
        lora=LoRAConfig(rank=1, target_modules=("q",), target_block_count=1),
    )
    env_config = SimpleNamespace(
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(),
    )
    source = create_model(
        model_config,
        obs_spec=env_config.obs_spec,
        action_spec=env_config.action_spec,
    )
    assert isinstance(source, StatelessTransformerV1)
    source.reset_parameters()
    assert model_config.lora is not None
    apply_lora_to_stateless_transformer(source, model_config.lora)
    assert isinstance(source.blocks[0].attn.q, LoRALinear)
    with torch.no_grad():
        source.blocks[0].attn.q.lora_down.fill_(0.25)
        source.blocks[0].attn.q.lora_up.fill_(0.5)
    expected_q_weight = (
        source.blocks[0].attn.q.weight
        + (source.blocks[0].attn.q.lora_up @ source.blocks[0].attn.q.lora_down)
        * source.blocks[0].attn.q.scaling
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused\n")
    torch.save({"model": source.state_dict(), "env_steps": 12}, checkpoint_path)
    config = SimpleNamespace(
        model=model_config,
        env=env_config,
        rl=Namespace(dtype="float32"),
    )
    monkeypatch.setattr(
        benchmark_checkpoints.FullConfig,
        "from_file",
        classmethod(lambda _cls, _path: config),
    )

    loaded = benchmark_checkpoints._load_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded.model, StatelessTransformerV1)
    assert isinstance(loaded.model.blocks[0].attn.q, torch.nn.Linear)
    assert not isinstance(loaded.model.blocks[0].attn.q, LoRALinear)
    assert not any(
        name.endswith((".lora_down", ".lora_up")) for name in loaded.model.state_dict()
    )
    assert torch.allclose(loaded.model.blocks[0].attn.q.weight, expected_q_weight)


def test_actions_for_checkpoints_uses_checkpoint_autocast_context(
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
            return SimpleNamespace(actions=actions, next_hidden_state=None)

    class FakeEnv:
        def __init__(self) -> None:
            self.observed_specs: list[tuple[str, str]] = []
            self.decoded_specs: list[str] = []

        def observation_for_spec(
            self,
            obs_spec: ObsConfig,
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
            actions: ActionBundle,
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

    actions, _hidden_a, _hidden_b = benchmark_checkpoints._actions_for_checkpoints(
        env,
        assignments,
        checkpoint_a=_loaded_checkpoint(
            FakeModel(launch_value=True, ship_value=3),
            dtype="bfloat16",
        ),
        checkpoint_b=_loaded_checkpoint(
            FakeModel(launch_value=False, ship_value=7),
            dtype="float32",
        ),
        hidden_a=None,
        hidden_b=None,
        device=torch.device("cpu"),
        determinism=benchmark_checkpoints.CheckpointDeterminism(a=True, b=True),
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


def test_actions_for_checkpoints_uses_each_checkpoint_action_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.observed_specs: list[tuple[str, str]] = []
            self.decoded_specs: list[str] = []

        def observation_for_spec(
            self,
            obs_spec: ObsConfig,
            action_spec: ActionConfig,
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
            actions: ActionBundle,  # noqa: ARG002
            *,
            action_spec: ActionConfig,
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
            return SimpleNamespace(actions=actions, next_hidden_state=None)

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

    actions, _hidden_a, _hidden_b = benchmark_checkpoints._actions_for_checkpoints(
        env,
        torch.tensor([[0, 1, 0, 1]]),
        checkpoint_a=_loaded_checkpoint(
            FakeModel(),
            obs_spec=EntityBasedConfig(max_entities=128),
            action_spec=ActionPureConfig(max_per_planet_launches=1),
        ),
        checkpoint_b=_loaded_checkpoint(
            FakeModel(),
            obs_spec=EntityBasedExtV1Config(
                max_entities=256,
                ship_count_one_hot_max=5,
            ),
            action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
        ),
        hidden_a=None,
        hidden_b=None,
        device=torch.device("cpu"),
        determinism=benchmark_checkpoints.CheckpointDeterminism(a=True, b=True),
    )

    assert env.observed_specs == [
        ("entity_based", "pure"),
        ("entity_based_ext_v1", "discrete_targets"),
    ]
    assert env.decoded_specs == ["pure", "discrete_targets"]
    assert actions.ships[0, :, 0].tolist() == [3, 7, 3, 7]


def test_actions_for_checkpoints_returns_next_hidden_state(
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
            obs_spec: ObsConfig,
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
            actions: ActionBundle,
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

    actions, hidden_a, hidden_b = benchmark_checkpoints._actions_for_checkpoints(
        FakeEnv(),
        torch.tensor(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ]
        ),
        checkpoint_a=_loaded_checkpoint(model_a),
        checkpoint_b=_loaded_checkpoint(model_b),
        hidden_a=torch.tensor([1.0, 2.0]),
        hidden_b=torch.tensor([3.0, 4.0]),
        device=torch.device("cpu"),
        determinism=benchmark_checkpoints.CheckpointDeterminism(a=True, b=True),
    )

    assert [hidden.tolist() for hidden in model_a.seen_hidden] == [[1.0, 2.0]]
    assert [hidden.tolist() for hidden in model_b.seen_hidden] == [[3.0, 4.0]]
    assert hidden_a.tolist() == [2.0, 3.0]
    assert hidden_b.tolist() == [13.0, 14.0]
    assert actions.valid[:, :, 0].tolist() == [
        [True, False, True, False],
        [False, True, False, True],
    ]


def test_actions_for_checkpoints_passes_per_checkpoint_deterministic_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.seen_deterministic: list[bool] = []

        def __call__(
            self,
            obs: ObsBatch,
            *,
            deterministic: bool = False,
        ) -> SimpleNamespace:
            self.seen_deterministic.append(deterministic)
            shape = (obs.global_features.shape[0], 4, ACTION_ENTITY_SLOTS, 1)
            actions = benchmark_checkpoints.PureActions(
                launch=torch.ones(shape, dtype=torch.bool),
                angle=torch.zeros(shape, dtype=torch.float32),
                ships=torch.ones(shape, dtype=torch.int64),
            )
            return SimpleNamespace(actions=actions, next_hidden_state=None)

    class FakeEnv:
        def observation_for_spec(
            self,
            obs_spec: ObsConfig,
            action_spec: ActionPureConfig,  # noqa: ARG002
        ) -> ObsBatch:
            return ObsBatch(
                planets=torch.zeros((1, 1, obs_spec.planet_channels)),
                orbiting_planets=torch.zeros((1, 1), dtype=torch.bool),
                fleets=torch.zeros((1, 1, obs_spec.fleet_channels)),
                comets=torch.zeros((1, 1, obs_spec.comet_channels)),
                entity_mask=torch.zeros((1, obs_spec.max_entities), dtype=torch.bool),
                still_playing=torch.ones((1, 4), dtype=torch.bool),
                global_features=torch.zeros((1, obs_spec.global_channels)),
                action_mask=PureActionMask(
                    can_act=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
                    max_launch=torch.zeros(
                        (1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64
                    ),
                ),
            )

        def decode_actions(
            self,
            actions: ActionBundle,
            *,
            action_spec: ActionPureConfig,  # noqa: ARG002
        ) -> DecodedLaunchActions:
            valid = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool)
            valid[:, :, 0] = actions.launch[:, :, 0, 0]
            ships = torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64)
            ships[:, :, 0] = actions.ships[:, :, 0, 0]
            return DecodedLaunchActions(
                valid=valid,
                from_planet_id=torch.zeros_like(ships),
                angle=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.float32),
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
    model_a = FakeModel()
    model_b = FakeModel()

    benchmark_checkpoints._actions_for_checkpoints(
        FakeEnv(),
        torch.tensor([[0, 1, 0, 1]]),
        checkpoint_a=_loaded_checkpoint(model_a),
        checkpoint_b=_loaded_checkpoint(model_b),
        hidden_a=None,
        hidden_b=None,
        device=torch.device("cpu"),
        determinism=benchmark_checkpoints.CheckpointDeterminism(a=True, b=False),
    )

    assert model_a.seen_deterministic == [True]
    assert model_b.seen_deterministic == [False]


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
