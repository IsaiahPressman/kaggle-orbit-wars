from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Literal, Protocol, assert_never

import torch

TrainingDType = Literal["float32", "bfloat16"]


class DTypeConfig(Protocol):
    @property
    def dtype(self) -> TrainingDType: ...


def configure_torch() -> None:
    torch.backends.fp32_precision = "tf32"  # type: ignore[attr-defined]
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = True


def autocast_context(
    cfg: DTypeConfig,
    device: torch.device,
) -> AbstractContextManager[None]:
    match cfg.dtype:
        case "bfloat16":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        case "float32":
            return nullcontext()
        case _:
            assert_never(cfg.dtype)


def assert_finite(tensor: torch.Tensor, name: str) -> None:
    if torch.isfinite(tensor).all():
        return
    raise ValueError(f"{name} must contain only finite values")


def require_same_shape(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if left.shape == right.shape:
        return
    raise ValueError(
        f"{right_name} must match {left_name} shape {left.shape}, got {right.shape}"
    )


def require_2d(tensor: torch.Tensor, name: str) -> None:
    """Require a segment-major/time-second tensor with shape [N, T]."""
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [N, T], got {tensor.shape}")


def require_segment_time_major(tensor: torch.Tensor, name: str) -> None:
    """Require segment-major/time-second layout: [N, T, ...]."""
    if tensor.ndim < 2:
        raise ValueError(f"{name} must have shape [N, T, ...], got {tensor.shape}")


def require_probability_range(value: float, name: str) -> None:
    if 0.0 <= value <= 1.0:
        return

    raise ValueError(f"{name} must be between 0 and 1")
