from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
from owl.model import (
    BaseModelAPI,
    LoRAConfig,
    ModelActionEntropies,
    ModelActionLogProbs,
    ModelEvaluation,
    ModelOutput,
    RecurrentTransformerV1,
    RecurrentTransformerV1Config,
    StatelessTransformerV1,
    StatelessTransformerV1Config,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    ActionBundle,
    ActionDiscreteTargetsConfig,
    ActionPureConfig,
    EntityBasedConfig,
    ObsBatch,
    PureActionMask,
    PureActions,
)
from owl.train import distributed as distributed_module
from owl.train.distributed import (
    DistributedContext,
    DistributedModelAdapter,
    all_gather_object,
    all_reduce_any,
    all_reduce_max,
    all_reduce_sum,
    broadcast_object,
    distributed_session,
    model_no_sync_context,
    unwrap_model,
    wrap_model_for_distributed,
)


class _TinyModel(BaseModelAPI):
    def __init__(self) -> None:
        super().__init__()
        self.action_spec = ActionPureConfig()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,  # noqa: ARG002
    ) -> ModelOutput:
        n_envs = obs.global_features.shape[0]
        actions = _actions(n_envs)
        values = self.weight.expand(n_envs, 4)
        log_probs = _log_probs(torch.zeros_like(values))
        entropies = _entropies(torch.zeros_like(values))
        return ModelOutput(
            actions=actions,
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=torch.softmax(values, dim=-1),
        )

    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ActionBundle,  # noqa: ARG002
    ) -> ModelEvaluation:
        values = self.weight.expand(obs.global_features.shape[0], 4)
        log_probs = _log_probs(torch.zeros_like(values))
        entropies = _entropies(torch.zeros_like(values))
        return ModelEvaluation(
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=torch.softmax(values, dim=-1),
        )

    def compute_value(self, obs: ObsBatch) -> torch.Tensor:
        return self.weight.expand(obs.global_features.shape[0], 4)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def get_input_layers(self) -> tuple[torch.nn.Module, ...]:
        return ()

    def get_output_layers(self) -> tuple[torch.nn.Module, ...]:
        return ()


class _FakeDDP(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        *,
        device_ids: list[int],
        output_device: int,
        find_unused_parameters: bool = False,
    ) -> None:
        super().__init__()
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.find_unused_parameters = find_unused_parameters
        self.no_sync_entries = 0
        self.no_sync_active = False

    def forward(self, *args: object) -> object:
        return self.module(*args)

    @contextmanager
    def no_sync(self) -> object:
        self.no_sync_entries += 1
        self.no_sync_active = True
        try:
            yield
        finally:
            self.no_sync_active = False


def _actions(n_envs: int) -> PureActions:
    shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
    return PureActions(
        launch=torch.zeros(shape, dtype=torch.bool),
        angle=torch.zeros(shape, dtype=torch.float32),
        ships=torch.zeros(shape, dtype=torch.int64),
    )


def _obs(n_envs: int) -> ObsBatch:
    return ObsBatch(
        planets=torch.zeros((n_envs, 1, 1)),
        orbiting_planets=torch.zeros((n_envs, 1), dtype=torch.bool),
        fleets=torch.zeros((n_envs, 1, 1)),
        comets=torch.zeros((n_envs, 1, 1)),
        entity_mask=torch.ones((n_envs, 1), dtype=torch.bool),
        still_playing=torch.ones((n_envs, 4), dtype=torch.bool),
        global_features=torch.zeros((n_envs, 1)),
        action_mask=PureActionMask(
            can_act=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
            max_launch=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64),
        ),
    )


def _log_probs(per_player: torch.Tensor) -> ModelActionLogProbs:
    n_envs = per_player.shape[0]
    action_shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
    return ModelActionLogProbs(
        launch=torch.zeros(action_shape),
        event=torch.zeros(action_shape),
        per_player_entity=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS)),
    )


def _entropies(per_player: torch.Tensor) -> ModelActionEntropies:
    n_envs = per_player.shape[0]
    action_shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
    return ModelActionEntropies(
        launch=torch.zeros(action_shape),
        event=torch.zeros(action_shape),
        per_player_entity=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS)),
        components={"launch": per_player},
    )


def test_distributed_context_defaults_to_cpu_without_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(distributed_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: False)

    context = DistributedContext.from_runtime()

    assert context == DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=1,
        initialized=False,
    )
    assert context.is_main_process


def test_distributed_context_rejects_nonzero_local_rank_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(distributed_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(
        RuntimeError,
        match="CUDA is not available - can't create distributed context",
    ):
        DistributedContext.from_runtime()


def test_distributed_context_reads_initialized_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(distributed_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "get_rank", lambda: 5)
    monkeypatch.setattr(distributed_module.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: True)

    context = DistributedContext.from_runtime()

    assert context == DistributedContext(
        device=torch.device("cuda", 2),
        rank=5,
        local_rank=2,
        world_size=8,
        initialized=True,
    )
    assert not context.is_main_process


def test_distributed_session_initializes_and_destroys_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object | None]] = []

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        distributed_module.torch.cuda,
        "set_device",
        lambda local_rank: calls.append(("set_device", local_rank)),
    )
    monkeypatch.setattr(distributed_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        distributed_module.dist,
        "init_process_group",
        lambda backend, device_id: calls.append(
            ("init_process_group", (backend, device_id))
        ),
    )
    monkeypatch.setattr(
        distributed_module.dist,
        "destroy_process_group",
        lambda: calls.append(("destroy_process_group", None)),
    )
    monkeypatch.setattr(
        DistributedContext,
        "from_runtime",
        classmethod(
            lambda cls: cls(
                device=torch.device("cuda", 1),
                rank=1,
                local_rank=1,
                world_size=2,
                initialized=True,
            )
        ),
    )

    with distributed_session() as context:
        assert context.local_rank == 1

    assert calls == [
        ("set_device", torch.device("cuda:1")),
        ("init_process_group", ("nccl", torch.device("cuda:1"))),
        ("destroy_process_group", None),
    ]


def test_distributed_session_rejects_distributed_cpu_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: False)

    with (
        pytest.raises(
            RuntimeError,
            match="distributed session requires CUDA, but CUDA is not available",
        ),
        distributed_session(),
    ):
        pass


def test_collective_helpers_return_input_without_process_group() -> None:
    context = DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=1,
        initialized=False,
    )
    tensor = torch.tensor([1.0])

    assert all_reduce_sum(tensor, context) is tensor
    assert all_reduce_max(tensor, context) is tensor
    assert all_reduce_any(True, context) is True
    assert all_gather_object({"rank": 0}, context) == [{"rank": 0}]
    assert broadcast_object({"run": "abc"}, context) == {"run": "abc"}


def test_collective_helpers_delegate_to_torch_distributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    context = DistributedContext(
        device=torch.device("cpu"),
        rank=1,
        local_rank=0,
        world_size=2,
        initialized=True,
    )

    def fake_all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        calls.append(op)
        tensor.add_(1)

    def fake_broadcast_object_list(values: list[object | None], *, src: int) -> None:
        calls.append(("broadcast", src))
        values[0] = "from-main"

    def fake_all_gather_object(values: list[object | None], value: object) -> None:
        calls.append(("gather", value))
        values[:] = ["rank-0", "rank-1"]

    monkeypatch.setattr(distributed_module.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        distributed_module.dist,
        "all_gather_object",
        fake_all_gather_object,
    )
    monkeypatch.setattr(
        distributed_module.dist,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )

    assert torch.equal(
        all_reduce_sum(torch.tensor([1.0]), context),
        torch.tensor([2.0]),
    )
    assert torch.equal(
        all_reduce_max(torch.tensor([2.0]), context),
        torch.tensor([3.0]),
    )
    assert all_reduce_any(False, context) is True
    assert all_gather_object("local", context) == ["rank-0", "rank-1"]
    assert broadcast_object(None, context) == "from-main"
    assert calls == [
        distributed_module.dist.ReduceOp.SUM,
        distributed_module.dist.ReduceOp.MAX,
        distributed_module.dist.ReduceOp.MAX,
        ("gather", "local"),
        ("broadcast", 0),
    ]


def test_wrap_model_for_distributed_returns_model_without_process_group() -> None:
    model = _TinyModel()
    context = DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=1,
        initialized=False,
    )

    assert wrap_model_for_distributed(model, context) is model


def test_wrap_model_for_distributed_uses_ddp_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(distributed_module, "DistributedDataParallel", _FakeDDP)
    model = _TinyModel()
    context = DistributedContext(
        device=torch.device("cuda", 3),
        rank=3,
        local_rank=3,
        world_size=4,
        initialized=True,
    )

    wrapped = wrap_model_for_distributed(model, context)

    assert isinstance(wrapped, DistributedModelAdapter)
    assert unwrap_model(wrapped) is model
    assert isinstance(wrapped._ddp, _FakeDDP)
    assert wrapped._ddp.device_ids == [3]
    assert wrapped._ddp.output_device == 3
    assert not wrapped._ddp.find_unused_parameters


def test_model_no_sync_context_delegates_for_distributed_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(distributed_module, "DistributedDataParallel", _FakeDDP)
    model = _TinyModel()
    wrapped = wrap_model_for_distributed(
        model,
        DistributedContext(
            device=torch.device("cuda", 0),
            rank=0,
            local_rank=0,
            world_size=2,
            initialized=True,
        ),
    )
    assert isinstance(wrapped, DistributedModelAdapter)
    assert isinstance(wrapped._ddp, _FakeDDP)

    with model_no_sync_context(wrapped, enabled=True):
        assert wrapped._ddp.no_sync_active

    assert wrapped._ddp.no_sync_entries == 1
    assert not wrapped._ddp.no_sync_active


def test_model_no_sync_context_is_noop_when_disabled_or_not_distributed() -> None:
    model = _TinyModel()

    with model_no_sync_context(model, enabled=True):
        pass
    with model_no_sync_context(model, enabled=False):
        pass


def test_wrap_model_for_distributed_finds_unused_player_count_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(distributed_module, "DistributedDataParallel", _FakeDDP)
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(
            embed_dim=16,
            depth=1,
            n_heads=4,
            player_count_adapters_enabled=True,
        ),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    context = DistributedContext(
        device=torch.device("cuda", 0),
        rank=0,
        local_rank=0,
        world_size=2,
        initialized=True,
    )

    wrapped = wrap_model_for_distributed(model, context)

    assert isinstance(wrapped, DistributedModelAdapter)
    assert isinstance(wrapped._ddp, _FakeDDP)
    assert wrapped._ddp.find_unused_parameters
    assert wrapped.action_spec == model.action_spec


def test_wrap_model_for_distributed_disables_unused_detection_for_lora_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LoRA adapters are on shared forward/evaluate paths. Leaving
    # find_unused_parameters=False avoids DDP's extra graph traversal.
    monkeypatch.setattr(distributed_module, "DistributedDataParallel", _FakeDDP)
    model = StatelessTransformerV1(
        StatelessTransformerV1Config(
            embed_dim=16,
            depth=1,
            n_heads=4,
            lora=LoRAConfig(rank=2, target_modules=("q", "v")),
        ),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )
    context = DistributedContext(
        device=torch.device("cuda", 0),
        rank=0,
        local_rank=0,
        world_size=2,
        initialized=True,
    )

    wrapped = wrap_model_for_distributed(model, context)

    assert isinstance(wrapped, DistributedModelAdapter)
    assert isinstance(wrapped._ddp, _FakeDDP)
    assert not wrapped._ddp.find_unused_parameters


def test_requires_unused_parameter_detection_handles_recurrent_model() -> None:
    # RecurrentTransformerV1 subclasses StatelessTransformerV1 but does not use
    # player-count adapters, so it should keep the cheaper DDP reducer path.
    model = RecurrentTransformerV1(
        RecurrentTransformerV1Config(embed_dim=16, depth=1, n_heads=4),
        obs_spec=EntityBasedConfig(max_entities=ACTION_ENTITY_SLOTS + 2),
        action_spec=ActionDiscreteTargetsConfig(max_per_planet_launches=1),
    )

    assert distributed_module._requires_unused_parameter_detection(model) is False


def test_distributed_evaluate_actions_forwards_dones_without_hidden_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RecordingDonesModel(_TinyModel):
        def __init__(self) -> None:
            super().__init__()
            self.seen_dones: torch.Tensor | None = None

        def evaluate_actions(
            self,
            obs: ObsBatch,
            actions: ActionBundle,
            *,
            hidden_state: object | None = None,  # noqa: ARG002
            dones: torch.Tensor | None = None,
        ) -> ModelEvaluation:
            self.seen_dones = dones
            return super().evaluate_actions(obs, actions)

    monkeypatch.setattr(distributed_module, "DistributedDataParallel", _FakeDDP)
    model = _RecordingDonesModel()
    wrapped = wrap_model_for_distributed(
        model,
        DistributedContext(
            device=torch.device("cuda", 0),
            rank=0,
            local_rank=0,
            world_size=2,
            initialized=True,
        ),
    )
    dones = torch.tensor(
        [
            [False, True, False, False],
            [True, True, True, True],
        ]
    )

    wrapped.evaluate_actions(_obs(2), _actions(2), dones=dones)

    assert model.seen_dones is dones


def test_wrap_model_for_distributed_requires_cuda_device() -> None:
    model = _TinyModel()
    context = DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=2,
        initialized=True,
    )

    with pytest.raises(
        RuntimeError,
        match="distributed model wrapping requires a CUDA device",
    ):
        wrap_model_for_distributed(model, context)
