from __future__ import annotations

import pytest
import torch
from owl.train import distributed as distributed_module
from owl.train.distributed import (
    DistributedContext,
    all_reduce_any,
    all_reduce_max,
    all_reduce_sum,
    broadcast_object,
    distributed_session,
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
        lambda backend: calls.append(("init_process_group", backend)),
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
        ("set_device", 1),
        ("init_process_group", "nccl"),
        ("destroy_process_group", None),
    ]


def test_distributed_session_rejects_distributed_cpu_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(
        RuntimeError,
        match="distributed session requires CUDA, but CUDA is not available",
    ), distributed_session():
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

    monkeypatch.setattr(distributed_module.dist, "all_reduce", fake_all_reduce)
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
    assert broadcast_object(None, context) == "from-main"
    assert calls == [
        distributed_module.dist.ReduceOp.SUM,
        distributed_module.dist.ReduceOp.MAX,
        distributed_module.dist.ReduceOp.MAX,
        ("broadcast", 0),
    ]
