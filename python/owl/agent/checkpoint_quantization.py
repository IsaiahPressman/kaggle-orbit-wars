from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias, TypeGuard

import torch

QuantizationFormat: TypeAlias = Literal["fp8_e4m3fn", "fp4_e2m1fn_x2"]

FP8_E4M3FN: QuantizationFormat = "fp8_e4m3fn"
FP4_E2M1FN_X2: QuantizationFormat = "fp4_e2m1fn_x2"
SUPPORTED_QUANTIZATION_FORMATS: tuple[QuantizationFormat, ...] = (
    FP8_E4M3FN,
    FP4_E2M1FN_X2,
)

_QUANTIZED_STATE_MARKER = "__owl_quantized_model_state_dict__"
_QUANTIZED_STATE_VERSION = 1
_TENSOR_QUANTIZED_KEY = "quantized"
_TENSOR_FORMAT_KEY = "format"
_TENSOR_SHAPE_KEY = "shape"
_TENSOR_DATA_KEY = "data"
_TENSOR_SOURCE_DTYPE_KEY = "source_dtype"

_FP4_E2M1FN_VALUES = torch.tensor(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=torch.float32,
)
_FP4_E2M1FN_ROUNDING_THRESHOLDS = torch.tensor(
    (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
    dtype=torch.float32,
)


def quantize_model_state_dict(
    model_state: Mapping[str, torch.Tensor],
    quantization: QuantizationFormat,
) -> dict[str, object]:
    if quantization not in SUPPORTED_QUANTIZATION_FORMATS:
        raise ValueError(f"unsupported quantization format: {quantization}")

    return {
        _QUANTIZED_STATE_MARKER: _QUANTIZED_STATE_VERSION,
        "format": quantization,
        "tensors": {
            name: _quantize_state_tensor(name, tensor, quantization)
            for name, tensor in model_state.items()
        },
    }


def dequantize_model_state_dict(model_state: object) -> dict[str, torch.Tensor]:
    if not _is_quantized_model_state_dict(model_state):
        if not isinstance(model_state, Mapping):
            raise ValueError("checkpoint['model'] must be a mapping")
        return _validate_unquantized_model_state(model_state)

    version = model_state[_QUANTIZED_STATE_MARKER]
    if version != _QUANTIZED_STATE_VERSION:
        raise ValueError(f"unsupported quantized model state version: {version}")

    tensors = model_state["tensors"]
    if not isinstance(tensors, Mapping):
        raise ValueError("quantized model state 'tensors' must be a mapping")

    return {
        _validate_state_key(name): _dequantize_state_tensor(name, payload)
        for name, payload in tensors.items()
    }


def _quantize_state_tensor(
    name: str,
    tensor: torch.Tensor,
    quantization: QuantizationFormat,
) -> dict[str, object]:
    _validate_state_key(name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"model state '{name}' must be a tensor")

    tensor = tensor.detach().cpu().contiguous()
    if not tensor.dtype.is_floating_point:
        return {
            _TENSOR_QUANTIZED_KEY: False,
            _TENSOR_DATA_KEY: tensor,
        }

    if quantization == FP8_E4M3FN:
        data = _quantize_fp8_e4m3fn(tensor)
    elif quantization == FP4_E2M1FN_X2:
        data = _pack_fp4_e2m1fn(_quantize_fp4_e2m1fn_codes(tensor))
    else:
        raise ValueError(f"unsupported quantization format: {quantization}")

    return {
        _TENSOR_QUANTIZED_KEY: True,
        _TENSOR_FORMAT_KEY: quantization,
        _TENSOR_SHAPE_KEY: tuple(tensor.shape),
        _TENSOR_SOURCE_DTYPE_KEY: str(tensor.dtype),
        _TENSOR_DATA_KEY: data,
    }


def _dequantize_state_tensor(name: object, payload: object) -> torch.Tensor:
    state_key = _validate_state_key(name)
    if not isinstance(payload, Mapping):
        raise ValueError(f"quantized model state '{state_key}' must be a mapping")

    quantized = payload.get(_TENSOR_QUANTIZED_KEY)
    data = payload.get(_TENSOR_DATA_KEY)
    if not isinstance(data, torch.Tensor):
        raise ValueError(f"quantized model state '{state_key}' data must be a tensor")

    if quantized is False:
        return data
    if quantized is not True:
        raise ValueError(
            f"quantized model state '{state_key}' must have boolean 'quantized'"
        )

    shape = _validate_shape(payload.get(_TENSOR_SHAPE_KEY), state_key)
    quantization = payload.get(_TENSOR_FORMAT_KEY)
    if quantization == FP8_E4M3FN:
        return _dequantize_fp8_e4m3fn(data, shape)
    if quantization == FP4_E2M1FN_X2:
        return _dequantize_fp4_e2m1fn(data, shape)
    raise ValueError(
        f"quantized model state '{state_key}' has unsupported format: {quantization}"
    )


def _quantize_fp8_e4m3fn(tensor: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required for fp8 quantization")
    return tensor.to(torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)


def _dequantize_fp8_e4m3fn(data: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required for fp8 dequantization")
    return data.contiguous().view(torch.float8_e4m3fn).to(torch.float32).reshape(shape)


def _quantize_fp4_e2m1fn_codes(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.to(torch.float32)
    if not torch.isfinite(tensor).all().item():
        raise ValueError("fp4_e2m1fn_x2 quantization requires finite tensors")

    abs_tensor = tensor.abs()
    tie_values = _FP4_E2M1FN_ROUNDING_THRESHOLDS.to(device=tensor.device)
    codes = torch.bucketize(abs_tensor, tie_values, right=False).to(torch.uint8)
    flat_abs = abs_tensor.reshape(-1)
    flat_codes = codes.reshape(-1)
    is_tie = (flat_abs.unsqueeze(-1) == tie_values).any(dim=-1)
    round_up_to_even = is_tie & (flat_codes % 2 == 1)
    flat_codes[round_up_to_even] += 1

    sign = torch.signbit(tensor).to(torch.uint8) << 3
    return codes | sign


def _pack_fp4_e2m1fn(codes: torch.Tensor) -> torch.Tensor:
    flat_codes = codes.reshape(-1).to(torch.uint8)
    packed = torch.zeros(
        (flat_codes.numel() + 1) // 2,
        dtype=torch.uint8,
        device=flat_codes.device,
    )
    packed |= flat_codes[0::2] & 0x0F
    if flat_codes.numel() > 1:
        packed[: flat_codes[1::2].numel()] |= (flat_codes[1::2] & 0x0F) << 4
    return packed


def _dequantize_fp4_e2m1fn(
    data: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    packed = data.contiguous().view(torch.uint8).reshape(-1)
    codes = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    codes[0::2] = packed & 0x0F
    codes[1::2] = (packed >> 4) & 0x0F
    numel = _shape_numel(shape)
    if codes.numel() < numel:
        raise ValueError(
            f"fp4 payload has {codes.numel()} unpacked values, expected {numel}"
        )
    values = _FP4_E2M1FN_VALUES.to(device=packed.device)
    return values[codes[:numel].long()].reshape(shape)


def _is_quantized_model_state_dict(
    model_state: object,
) -> TypeGuard[Mapping[object, object]]:
    return (
        isinstance(model_state, Mapping)
        and model_state.get(_QUANTIZED_STATE_MARKER) is not None
    )


def _validate_unquantized_model_state(
    model_state: Mapping[object, object],
) -> dict[str, torch.Tensor]:
    validated: dict[str, torch.Tensor] = {}
    for name, tensor in model_state.items():
        state_key = _validate_state_key(name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"model state '{state_key}' must be a tensor")
        validated[state_key] = tensor
    return validated


def _validate_state_key(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"model state keys must be non-empty strings: {name}")
    return name


def _validate_shape(shape: object, name: str) -> tuple[int, ...]:
    if not isinstance(shape, (tuple, list)):
        raise ValueError(f"quantized model state '{name}' shape must be a sequence")

    validated: list[int] = []
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise ValueError(
                f"quantized model state '{name}' shape must contain "
                "non-negative integers"
            )
        validated.append(dim)
    return tuple(validated)


def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel
