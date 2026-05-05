from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    initialized: bool

    @classmethod
    def from_runtime(cls) -> DistributedContext:
        initialized = dist.is_available() and dist.is_initialized()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        elif local_rank != 0:
            raise RuntimeError(
                "CUDA is not available - can't create distributed context"
            )
        else:
            device = torch.device("cpu")

        if initialized:
            return cls(
                device=device,
                rank=dist.get_rank(),
                local_rank=local_rank,
                world_size=dist.get_world_size(),
                initialized=True,
            )

        return cls(
            device=device,
            rank=0,
            local_rank=local_rank,
            world_size=1,
            initialized=False,
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.initialized:
            dist.barrier()


@contextmanager
def distributed_session() -> Iterator[DistributedContext]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    manage_process_group = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if manage_process_group:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "distributed session requires CUDA, but CUDA is not available"
            )

        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available")

        if dist.is_initialized():
            raise RuntimeError("torch.distributed is already initialized")

        dist.init_process_group(backend="nccl")

    try:
        yield DistributedContext.from_runtime()
    finally:
        if manage_process_group:
            dist.destroy_process_group()


def broadcast_object[T](
    value: T | None,
    context: DistributedContext,
    *,
    src: int = 0,
) -> T:
    values: list[object | None] = [value if context.rank == src else None]
    if context.initialized:
        dist.broadcast_object_list(values, src=src)
    return cast(T, values[0])


def all_reduce_sum(
    tensor: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    if not context.initialized:
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def all_reduce_max(
    tensor: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    if not context.initialized:
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return reduced


def all_reduce_any(value: bool, context: DistributedContext) -> bool:
    if not context.initialized:
        return value
    flag = torch.tensor(int(value), device=context.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())
