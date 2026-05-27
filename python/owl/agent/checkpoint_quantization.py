from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias, TypeGuard

import torch

QuantizationFormat: TypeAlias = Literal[
    "fp8_e4m3fn",
    "fp4_e2m1fn_x2_scaled_block16",
    "nf5_g128_lsq_policy_last_fp8",
    "nf5_g128_lsq_policy_final4_fp8",
]

FP8_E4M3FN: QuantizationFormat = "fp8_e4m3fn"
FP4_E2M1FN_X2_SCALED_BLOCK16: QuantizationFormat = "fp4_e2m1fn_x2_scaled_block16"
NF5_G128_LSQ_POLICY_LAST_FP8: QuantizationFormat = "nf5_g128_lsq_policy_last_fp8"
NF5_G128_LSQ_POLICY_FINAL4_FP8: QuantizationFormat = "nf5_g128_lsq_policy_final4_fp8"
SUPPORTED_QUANTIZATION_FORMATS: tuple[QuantizationFormat, ...] = (
    FP8_E4M3FN,
    FP4_E2M1FN_X2_SCALED_BLOCK16,
    NF5_G128_LSQ_POLICY_LAST_FP8,
    NF5_G128_LSQ_POLICY_FINAL4_FP8,
)

_QUANTIZED_STATE_MARKER = "__owl_quantized_model_state_dict__"
_QUANTIZED_STATE_VERSION = 1
_TENSOR_QUANTIZED_KEY = "quantized"
_TENSOR_FORMAT_KEY = "format"
_TENSOR_SHAPE_KEY = "shape"
_TENSOR_DATA_KEY = "data"
_TENSOR_SCALE_KEY = "scale"
_TENSOR_SOURCE_DTYPE_KEY = "source_dtype"
_TENSOR_COLS_KEY = "cols"

_FP4_BLOCK_SIZE = 16
_NF5_GROUP_SIZE = 128
_FP16: Literal["fp16"] = "fp16"

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
_NF5_NORMAL_VALUES = torch.erfinv(
    2.0 * ((torch.arange(32, dtype=torch.float32) + 0.5) / 32.0) - 1.0
) * (2.0**0.5)
_NF5_NORMAL_VALUES = _NF5_NORMAL_VALUES / _NF5_NORMAL_VALUES.abs().max()
_NF5_NORMAL_THRESHOLDS = (_NF5_NORMAL_VALUES[:-1] + _NF5_NORMAL_VALUES[1:]) / 2.0
_NF5_POLICY_FP8_TENSOR_NAMES = frozenset(
    (
        "source_actor_input_proj.weight",
        "target_actor_input_proj.weight",
        "critic_head.up.weight",
        "actor.continue_source_proj.weight",
        "actor.size_pair_proj.weight",
        "actor.mix_head.out.weight",
        "actor.mean_head.out.weight",
        "actor.scale_head.out.weight",
    )
)
_NF5_POLICY_LAST_FP8_TENSOR_NAMES = _NF5_POLICY_FP8_TENSOR_NAMES | frozenset(
    (
        "blocks.27.attn.out.weight",
        "blocks.27.mlp.down.weight",
    )
)
_NF5_POLICY_FINAL4_FP8_TENSOR_NAMES = _NF5_POLICY_FP8_TENSOR_NAMES | frozenset(
    (
        "blocks.24.attn.out.weight",
        "blocks.24.mlp.down.weight",
        "blocks.25.attn.out.weight",
        "blocks.25.mlp.down.weight",
        "blocks.26.attn.out.weight",
        "blocks.26.mlp.down.weight",
        "blocks.27.attn.out.weight",
        "blocks.27.mlp.down.weight",
    )
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
        return {
            _TENSOR_QUANTIZED_KEY: True,
            _TENSOR_FORMAT_KEY: quantization,
            _TENSOR_SHAPE_KEY: tuple(tensor.shape),
            _TENSOR_SOURCE_DTYPE_KEY: str(tensor.dtype),
            _TENSOR_DATA_KEY: _quantize_fp8_e4m3fn(tensor),
        }
    if quantization == FP4_E2M1FN_X2_SCALED_BLOCK16:
        data, scale = _quantize_fp4_e2m1fn_scaled_block16(tensor)
        return {
            _TENSOR_QUANTIZED_KEY: True,
            _TENSOR_FORMAT_KEY: quantization,
            _TENSOR_SHAPE_KEY: tuple(tensor.shape),
            _TENSOR_SOURCE_DTYPE_KEY: str(tensor.dtype),
            _TENSOR_DATA_KEY: data,
            _TENSOR_SCALE_KEY: scale,
        }
    if quantization == NF5_G128_LSQ_POLICY_LAST_FP8:
        return _quantize_nf5_g128_policy_fp8_state_tensor(
            name,
            tensor,
            quantization,
            _NF5_POLICY_LAST_FP8_TENSOR_NAMES,
        )
    if quantization == NF5_G128_LSQ_POLICY_FINAL4_FP8:
        return _quantize_nf5_g128_policy_fp8_state_tensor(
            name,
            tensor,
            quantization,
            _NF5_POLICY_FINAL4_FP8_TENSOR_NAMES,
        )
    raise ValueError(f"unsupported quantization format: {quantization}")


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
    if quantization == FP4_E2M1FN_X2_SCALED_BLOCK16:
        scale = payload.get(_TENSOR_SCALE_KEY)
        if not isinstance(scale, torch.Tensor):
            raise ValueError(
                f"quantized model state '{state_key}' scale must be a tensor"
            )
        return _dequantize_fp4_e2m1fn_scaled_block16(data, scale, shape)
    if quantization == _FP16:
        return data.to(torch.float16).to(torch.float32).reshape(shape)
    if quantization in (
        NF5_G128_LSQ_POLICY_LAST_FP8,
        NF5_G128_LSQ_POLICY_FINAL4_FP8,
    ):
        scale = payload.get(_TENSOR_SCALE_KEY)
        if not isinstance(scale, torch.Tensor):
            raise ValueError(
                f"quantized model state '{state_key}' scale must be a tensor"
            )
        cols = payload.get(_TENSOR_COLS_KEY)
        if not isinstance(cols, int) or cols < 0:
            raise ValueError(
                f"quantized model state '{state_key}' cols must be a "
                "non-negative integer"
            )
        return _dequantize_nf5_g128_lsq(data, scale, shape, cols)
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
        raise ValueError(
            "fp4_e2m1fn_x2_scaled_block16 quantization requires finite tensors"
        )

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


def _quantize_fp4_e2m1fn_scaled_block16(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = tensor.to(torch.float32)
    flat = tensor.reshape(-1)
    numel = flat.numel()
    if numel == 0:
        return torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=torch.float16)

    block_count = (numel + _FP4_BLOCK_SIZE - 1) // _FP4_BLOCK_SIZE
    padded = torch.zeros(
        block_count * _FP4_BLOCK_SIZE,
        dtype=torch.float32,
        device=flat.device,
    )
    padded[:numel] = flat
    blocks = padded.reshape(block_count, _FP4_BLOCK_SIZE)
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    scale = max_abs / 6.0
    safe_scale = torch.where(max_abs == 0, torch.ones_like(scale), scale)
    codes = _quantize_fp4_e2m1fn_codes(blocks / safe_scale).reshape(-1)[:numel]
    return _pack_fp4_e2m1fn(codes), scale.reshape(-1).to(torch.float16)


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


def _dequantize_fp4_e2m1fn_scaled_block16(
    data: torch.Tensor,
    scale: torch.Tensor,
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
    expected_blocks = (numel + _FP4_BLOCK_SIZE - 1) // _FP4_BLOCK_SIZE
    flat_scale = scale.contiguous().to(torch.float32).reshape(-1)
    if flat_scale.numel() != expected_blocks:
        raise ValueError(
            f"fp4 payload has {flat_scale.numel()} scale values, "
            f"expected {expected_blocks}"
        )
    values = _FP4_E2M1FN_VALUES.to(device=packed.device)
    block_scale = flat_scale.to(device=packed.device).repeat_interleave(
        _FP4_BLOCK_SIZE
    )[:numel]
    return (values[codes[:numel].long()] * block_scale).reshape(shape)


def _quantize_nf5_g128_policy_fp8_state_tensor(
    name: str,
    tensor: torch.Tensor,
    quantization: QuantizationFormat,
    fp8_tensor_names: frozenset[str],
) -> dict[str, object]:
    if name in fp8_tensor_names:
        return {
            _TENSOR_QUANTIZED_KEY: True,
            _TENSOR_FORMAT_KEY: FP8_E4M3FN,
            _TENSOR_SHAPE_KEY: tuple(tensor.shape),
            _TENSOR_SOURCE_DTYPE_KEY: str(tensor.dtype),
            _TENSOR_DATA_KEY: _quantize_fp8_e4m3fn(tensor),
        }
    if tensor.ndim != 2:
        return {
            _TENSOR_QUANTIZED_KEY: True,
            _TENSOR_FORMAT_KEY: _FP16,
            _TENSOR_SHAPE_KEY: tuple(tensor.shape),
            _TENSOR_SOURCE_DTYPE_KEY: str(tensor.dtype),
            _TENSOR_DATA_KEY: tensor.to(torch.float16),
        }

    data, scale = _quantize_nf5_g128_lsq(tensor)
    return {
        _TENSOR_QUANTIZED_KEY: True,
        _TENSOR_FORMAT_KEY: quantization,
        _TENSOR_SHAPE_KEY: tuple(tensor.shape),
        _TENSOR_COLS_KEY: tensor.shape[1],
        _TENSOR_SOURCE_DTYPE_KEY: str(tensor.dtype),
        _TENSOR_DATA_KEY: data,
        _TENSOR_SCALE_KEY: scale,
    }


def _quantize_nf5_g128_lsq(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = tensor.to(torch.float32)
    if tensor.ndim != 2:
        raise ValueError("nf5_g128_lsq quantization requires 2D tensors")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("nf5_g128_lsq quantization requires finite tensors")

    rows, cols = tensor.shape
    group_count = (cols + _NF5_GROUP_SIZE - 1) // _NF5_GROUP_SIZE
    padded_cols = group_count * _NF5_GROUP_SIZE
    padded = torch.zeros((rows, padded_cols), dtype=torch.float32, device=tensor.device)
    padded[:, :cols] = tensor
    groups = padded.reshape(rows * group_count, _NF5_GROUP_SIZE)

    scale = groups.abs().amax(dim=1, keepdim=True)
    values = _NF5_NORMAL_VALUES.to(device=tensor.device)
    thresholds = _NF5_NORMAL_THRESHOLDS.to(device=tensor.device)
    for _ in range(2):
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes = torch.bucketize(groups / safe_scale, thresholds).to(torch.long)
        quantized = values[codes]
        denominator = (quantized * quantized).sum(dim=1, keepdim=True)
        improved = torch.where(
            denominator > 0,
            ((groups * quantized).sum(dim=1, keepdim=True) / denominator).clamp_min(
                0.0
            ),
            torch.zeros_like(scale),
        )
        scale = torch.where(scale == 0, scale, improved)

    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.bucketize(groups / safe_scale, thresholds).to(torch.uint8).reshape(-1)
    return _pack_5bit_codes(codes), scale.reshape(-1).to(torch.float16)


def _dequantize_nf5_g128_lsq(
    data: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, ...],
    cols: int,
) -> torch.Tensor:
    if len(shape) != 2:
        raise ValueError(f"nf5 payload shape must be 2D, got {shape}")
    rows = shape[0]
    if cols != shape[1]:
        raise ValueError(f"nf5 payload cols {cols} does not match shape {shape}")
    group_count = (cols + _NF5_GROUP_SIZE - 1) // _NF5_GROUP_SIZE
    expected_codes = rows * group_count * _NF5_GROUP_SIZE
    codes = _unpack_5bit_codes(data, expected_codes)
    flat_scale = scale.contiguous().to(torch.float32).reshape(-1)
    expected_scales = rows * group_count
    if flat_scale.numel() != expected_scales:
        raise ValueError(
            f"nf5 payload has {flat_scale.numel()} scale values, "
            f"expected {expected_scales}"
        )

    values = _NF5_NORMAL_VALUES.to(device=codes.device)
    group_scale = flat_scale.to(device=codes.device).repeat_interleave(_NF5_GROUP_SIZE)
    dequantized = values[codes.long()] * group_scale
    return dequantized.reshape(rows, group_count * _NF5_GROUP_SIZE)[:, :cols].reshape(
        shape
    )


def _pack_5bit_codes(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.reshape(-1).to(torch.uint8)
    if ((flat & 0xE0) != 0).any().item():
        raise ValueError("5-bit packing requires codes in [0, 32)")
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.cat(
            (
                flat,
                torch.zeros(padding, dtype=torch.uint8, device=flat.device),
            )
        )
    values = flat.reshape(-1, 8).to(torch.int64)
    packed = torch.empty((values.shape[0], 5), dtype=torch.uint8, device=flat.device)
    packed[:, 0] = (values[:, 0] | (values[:, 1] << 5)).to(torch.uint8)
    packed[:, 1] = ((values[:, 1] >> 3) | (values[:, 2] << 2) | (values[:, 3] << 7)).to(
        torch.uint8
    )
    packed[:, 2] = ((values[:, 3] >> 1) | (values[:, 4] << 4)).to(torch.uint8)
    packed[:, 3] = ((values[:, 4] >> 4) | (values[:, 5] << 1) | (values[:, 6] << 6)).to(
        torch.uint8
    )
    packed[:, 4] = ((values[:, 6] >> 2) | (values[:, 7] << 3)).to(torch.uint8)
    return packed.reshape(-1)


def _unpack_5bit_codes(data: torch.Tensor, count: int) -> torch.Tensor:
    packed = data.contiguous().view(torch.uint8).reshape(-1)
    expected_bytes = ((count + 7) // 8) * 5
    if packed.numel() < expected_bytes:
        raise ValueError(
            f"5-bit payload has {packed.numel()} bytes, expected {expected_bytes}"
        )
    packed = packed[:expected_bytes].reshape(-1, 5).to(torch.int64)
    codes = torch.empty((packed.shape[0], 8), dtype=torch.uint8, device=packed.device)
    codes[:, 0] = (packed[:, 0] & 0x1F).to(torch.uint8)
    codes[:, 1] = (((packed[:, 0] >> 5) | (packed[:, 1] << 3)) & 0x1F).to(torch.uint8)
    codes[:, 2] = ((packed[:, 1] >> 2) & 0x1F).to(torch.uint8)
    codes[:, 3] = (((packed[:, 1] >> 7) | (packed[:, 2] << 1)) & 0x1F).to(torch.uint8)
    codes[:, 4] = (((packed[:, 2] >> 4) | (packed[:, 3] << 4)) & 0x1F).to(torch.uint8)
    codes[:, 5] = ((packed[:, 3] >> 1) & 0x1F).to(torch.uint8)
    codes[:, 6] = (((packed[:, 3] >> 6) | (packed[:, 4] << 2)) & 0x1F).to(torch.uint8)
    codes[:, 7] = ((packed[:, 4] >> 3) & 0x1F).to(torch.uint8)
    return codes.reshape(-1)[:count]


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
