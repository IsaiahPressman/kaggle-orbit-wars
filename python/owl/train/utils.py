from __future__ import annotations

import warnings
from contextlib import AbstractContextManager, nullcontext
from typing import Literal, Protocol, assert_never

import torch
from torch import nn

from owl.model import BaseModelAPI

Float8Recipe = Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"]
ModelCompileTarget = Literal["none", "mlp"]
ModelCompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
TrainingDType = Literal["float32", "bfloat16", "float8"]


class DTypeConfig(Protocol):
    @property
    def dtype(self) -> TrainingDType: ...


class Float8TrainingConfig(DTypeConfig, Protocol):
    @property
    def fp8_recipe(self) -> Float8Recipe: ...


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
        case "float8":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        case "float32":
            return nullcontext()
        case _:
            assert_never(cfg.dtype)


def configure_model_for_training_dtype(
    model: BaseModelAPI,
    cfg: Float8TrainingConfig,
    *,
    device: torch.device,
) -> int:
    if cfg.dtype != "float8":
        return 0
    if device.type != "cuda":
        raise RuntimeError("rl.dtype='float8' requires a CUDA device")

    try:
        from torchao.float8 import Float8LinearConfig
        from torchao.float8 import convert_to_float8_training as torchao_float8
    except ImportError as exc:
        raise RuntimeError("rl.dtype='float8' requires the torchao package") from exc

    excluded_linear_ids = _fp8_excluded_linear_module_ids(model)
    converted_names: list[str] = []

    def module_filter_fn(
        module: nn.Module,
        fqn: str,
        *,
        uses_packed_token_rows: bool,
    ) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if id(module) in excluded_linear_ids:
            return False
        if _linear_uses_packed_token_rows(fqn) != uses_packed_token_rows:
            return False
        if not _linear_shape_supports_fp8(module):
            return False
        converted_names.append(fqn)
        return True

    torchao_float8(
        model,
        config=Float8LinearConfig.from_recipe_name(cfg.fp8_recipe),
        module_filter_fn=lambda module, fqn: module_filter_fn(
            module,
            fqn,
            uses_packed_token_rows=False,
        ),
    )
    torchao_float8(
        model,
        config=Float8LinearConfig.from_recipe_name("rowwise_with_gw_hp"),
        module_filter_fn=lambda module, fqn: module_filter_fn(
            module,
            fqn,
            uses_packed_token_rows=True,
        ),
    )
    if not converted_names:
        raise RuntimeError(
            "rl.dtype='float8' found no eligible Linear layers; FP8 training "
            "requires non-input/output Linear dimensions divisible by 16"
        )
    return len(converted_names)


def _fp8_excluded_linear_module_ids(model: BaseModelAPI) -> set[int]:
    excluded: set[int] = set()
    for layer in (*model.get_input_layers(), *model.get_output_layers()):
        if isinstance(layer, nn.Linear):
            excluded.add(id(layer))
    return excluded


def _linear_shape_supports_fp8(module: nn.Linear) -> bool:
    return module.in_features % 16 == 0 and module.out_features % 16 == 0


def _linear_uses_packed_token_rows(name: str) -> bool:
    return name.startswith("blocks.") or (
        name.startswith("player_count_adapters.") and ".blocks." in name
    )


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
