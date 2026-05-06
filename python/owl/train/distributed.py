from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Self, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from owl.model import (
    BaseModelAPI,
    InputLayer,
    ModelActions,
    ModelEvaluation,
    ModelOutput,
)
from owl.rl import ActionConfig, ObsBatch


@dataclass(frozen=True)
class DistributedContext:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    initialized: bool

    @classmethod
    def from_runtime(cls) -> Self:
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

    @classmethod
    def single_process_cpu(cls) -> Self:
        return cls(
            device=torch.device("cpu"),
            rank=0,
            local_rank=0,
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
        torch.cuda.set_device(torch.device(f"cuda:{local_rank}"))

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

        dist.init_process_group(
            backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
        )

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


def all_gather_object[T](value: T, context: DistributedContext) -> list[T]:
    if not context.initialized:
        return [value]
    values: list[object | None] = [None for _ in range(context.world_size)]
    dist.all_gather_object(values, value)
    return cast(list[T], values)


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


class _DistributedModelDispatch(nn.Module):
    def __init__(self, model: BaseModelAPI) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        mode: str,
        obs: object,
        actions: object | None = None,
        deterministic: bool = False,
    ) -> object:
        if mode == "forward":
            return self.model(cast(ObsBatch, obs), deterministic=deterministic)
        if mode == "evaluate_actions":
            if actions is None:
                raise ValueError("actions are required for evaluate_actions")
            return self.model.evaluate_actions(
                cast(ObsBatch, obs),
                cast(ModelActions, actions),
            )
        if mode == "compute_value":
            return self.model.compute_value(cast(ObsBatch, obs))
        raise ValueError(f"unknown distributed model mode: {mode}")


class DistributedModelAdapter(BaseModelAPI):
    def __init__(
        self,
        model: BaseModelAPI,
        context: DistributedContext,
    ) -> None:
        super().__init__()
        if not context.initialized:
            raise ValueError("DistributedModelAdapter requires an initialized context")
        if context.device.type != "cuda":
            raise RuntimeError("distributed model wrapping requires a CUDA device")
        self._ddp = DistributedDataParallel(
            _DistributedModelDispatch(model),
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )

    @property
    def wrapped_model(self) -> BaseModelAPI:
        return self._ddp.module.model

    @property
    def action_spec(self) -> ActionConfig:
        return cast(ActionConfig, self.wrapped_model.action_spec)

    def forward(
        self,
        obs: ObsBatch,
        *,
        deterministic: bool = False,
    ) -> ModelOutput:
        return cast(ModelOutput, self._ddp("forward", obs, None, deterministic))

    def evaluate_actions(
        self,
        obs: ObsBatch,
        actions: ModelActions,
    ) -> ModelEvaluation:
        return cast(
            ModelEvaluation,
            self._ddp("evaluate_actions", obs, actions, False),
        )

    def compute_value(self, obs: ObsBatch) -> torch.Tensor:
        return cast(torch.Tensor, self._ddp("compute_value", obs, None, False))

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        return self.wrapped_model.get_input_layers()

    def get_output_layers(self) -> tuple[nn.Module, ...]:
        return self.wrapped_model.get_output_layers()


def wrap_model_for_distributed(
    model: BaseModelAPI,
    context: DistributedContext,
) -> BaseModelAPI:
    if not context.initialized:
        return model
    return DistributedModelAdapter(model, context)


def unwrap_model(model: BaseModelAPI) -> BaseModelAPI:
    if isinstance(model, DistributedModelAdapter):
        return model.wrapped_model
    return model
