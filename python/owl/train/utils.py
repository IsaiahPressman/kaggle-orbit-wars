from __future__ import annotations

import warnings
from contextlib import AbstractContextManager, nullcontext
from typing import Literal, Protocol, assert_never

import torch

from owl.model import BaseModelAPI

ModelCompileTarget = Literal["none", "mlp"]
ModelCompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
TrainingDType = Literal["float32", "bfloat16"]


class DTypeConfig(Protocol):
    @property
    def dtype(self) -> TrainingDType: ...


class ModelCompileConfig(Protocol):
    @property
    def model_compile(self) -> ModelCompileTarget: ...

    @property
    def model_compile_mode(self) -> ModelCompileMode: ...


def configure_torch() -> None:
    # PyTorch 2.9 Inductor still reads the legacy matmul allow_tf32 flag during
    # lowering. Mixing the new fp32_precision setters with that read raises at
    # compile time, so keep this on one API family until Inductor moves over.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Please use the new API settings to control TF32 behavior.*",
            category=UserWarning,
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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


def configure_model_compile(model: BaseModelAPI, cfg: ModelCompileConfig) -> int:
    match cfg.model_compile:
        case "none":
            return 0
        case "mlp":
            return _compile_transformer_mlp_modules(
                model,
                mode=cfg.model_compile_mode,
            )
        case _:
            assert_never(cfg.model_compile)


def _compile_transformer_mlp_modules(
    model: BaseModelAPI,
    *,
    mode: ModelCompileMode,
) -> int:
    compiled = 0
    for name, module in model.named_modules():
        if not _is_transformer_mlp_module_name(name):
            continue
        module.compile(mode=mode, dynamic=True)
        compiled += 1
    if compiled == 0:
        raise RuntimeError(
            "rl.model_compile='mlp' found no transformer MLP modules to compile"
        )
    return compiled


def _is_transformer_mlp_module_name(name: str) -> bool:
    return name.startswith("blocks.") and name.endswith(".mlp")


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


def require_segment_time_major(tensor: torch.Tensor, name: str) -> None:
    """Require segment-major/time-second layout: [N, T, ...]."""
    if tensor.ndim < 2:
        raise ValueError(f"{name} must have shape [N, T, ...], got {tensor.shape}")


def require_probability_range(value: float, name: str) -> None:
    if 0.0 <= value <= 1.0:
        return

    raise ValueError(f"{name} must be between 0 and 1")
