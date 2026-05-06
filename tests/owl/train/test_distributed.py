from __future__ import annotations

import pytest
import torch
from owl.model import (
    BaseModelAPI,
    ModelActionEntropies,
    ModelActionLogProbs,
    ModelActions,
    ModelEvaluation,
    ModelOutput,
)
from owl.rl import ACTION_ENTITY_SLOTS, ActionPureConfig, ObsBatch
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
        actions: ModelActions,  # noqa: ARG002
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
    ) -> None:
        super().__init__()
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device

    def forward(self, *args: object) -> object:
        return self.module(*args)


def _actions(n_envs: int) -> ModelActions:
    shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
    return ModelActions(
        launch=torch.zeros(shape, dtype=torch.bool),
        ships=torch.zeros(shape, dtype=torch.int64),
        angle=torch.zeros(shape, dtype=torch.float32),
    )


def _log_probs(per_player: torch.Tensor) -> ModelActionLogProbs:
    n_envs = per_player.shape[0]
    action_shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
    return ModelActionLogProbs(
        launch=torch.zeros(action_shape),
        angle_and_size=torch.zeros(action_shape),
        per_player_entity=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS)),
    )


def _entropies(per_player: torch.Tensor) -> ModelActionEntropies:
    n_envs = per_player.shape[0]
    action_shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
    return ModelActionEntropies(
        launch=torch.zeros(action_shape),
        angle_and_size=torch.zeros(action_shape),
        per_player_entity=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS)),
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
    assert wrapped.action_spec == model.action_spec


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
