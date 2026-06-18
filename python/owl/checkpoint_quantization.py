from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeGuard

import torch

from owl.quantization_formats import (
    BF16,
    FP4_E2M1FN_X2_SCALED_BLOCK16,
    FP8_E4M3FN,
    FP16,
    FP32,
    NF3_G128_LSQ,
    NF3_NF4_STRUCTURED_3P5,
    NF4_G128_LSQ,
    NF5_G128_LSQ_POLICY_LAST_FP8,
    SUPPORTED_QUANTIZATION_FORMATS,
    SUPPORTED_TENSOR_QUANTIZATION_FORMATS,
    QuantizationFormat,
    TensorQuantizationFormat,
)

__all__ = [
    "BF16",
    "FP4_E2M1FN_X2_SCALED_BLOCK16",
    "FP8_E4M3FN",
    "FP16",
    "FP32",
    "NF3_G128_LSQ",
    "NF3_NF4_STRUCTURED_3P5",
    "NF4_G128_LSQ",
    "NF5_G128_LSQ_POLICY_LAST_FP8",
    "SUPPORTED_QUANTIZATION_FORMATS",
    "SUPPORTED_TENSOR_QUANTIZATION_FORMATS",
    "QuantizationFormat",
    "TensorQuantizationFormat",
    "dequantize_model_state_dict",
    "is_lora_adapter_state_key",
    "load_model_state_dict_streaming",
    "quantize_model_state_dict",
]

_QUANTIZED_STATE_MARKER = "__owl_quantized_model_state_dict__"
_QUANTIZED_STATE_VERSION = 1
_TENSOR_QUANTIZED_KEY = "quantized"
_TENSOR_FORMAT_KEY = "format"
_TENSOR_SHAPE_KEY = "shape"
_TENSOR_DATA_KEY = "data"
_TENSOR_SCALE_KEY = "scale"
_TENSOR_SOURCE_DTYPE_KEY = "source_dtype"
_TENSOR_COLS_KEY = "cols"
_TENSOR_BITS_KEY = "bits"
_TENSOR_CODEBOOK_KEY = "codebook"

_FP4_BLOCK_SIZE = 16
_NORMALFLOAT_GROUP_SIZE = 128
_NORMALFLOAT_CODEBOOK: Literal["nf"] = "nf"


def _normal_float_values(bits: int) -> torch.Tensor:
    levels = 1 << bits
    values = torch.erfinv(
        2.0 * ((torch.arange(levels, dtype=torch.float32) + 0.5) / levels) - 1.0
    ) * (2.0**0.5)
    return values / values.abs().max()


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
_NF3_NORMAL_VALUES = _normal_float_values(3)
_NF4_NORMAL_VALUES = _normal_float_values(4)
_NF5_NORMAL_VALUES = _normal_float_values(5)
_NF3_NORMAL_THRESHOLDS = (_NF3_NORMAL_VALUES[:-1] + _NF3_NORMAL_VALUES[1:]) / 2.0
_NF4_NORMAL_THRESHOLDS = (_NF4_NORMAL_VALUES[:-1] + _NF4_NORMAL_VALUES[1:]) / 2.0
_NF5_NORMAL_THRESHOLDS = (_NF5_NORMAL_VALUES[:-1] + _NF5_NORMAL_VALUES[1:]) / 2.0
_ACTOR_FP8_TENSOR_NAMES = frozenset(
    (
        "source_actor_input_proj.weight",
        "target_actor_input_proj.weight",
        "actor.continue_source_proj.weight",
        "actor.size_pair_proj.weight",
        "actor.mix_head.out.weight",
        "actor.mean_head.out.weight",
        "actor.scale_head.out.weight",
    )
)
_TRUNK_FP8_TENSOR_SUFFIXES = ("attn.out.weight", "mlp.down.weight")


def _normalfloat_tensor_bits(
    model_state: Mapping[str, torch.Tensor],
    quantization: QuantizationFormat,
) -> dict[str, int]:
    if quantization == NF4_G128_LSQ:
        return {
            name: 4
            for name, tensor in model_state.items()
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
        }
    if quantization == NF3_G128_LSQ:
        return {
            name: 3
            for name, tensor in model_state.items()
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
        }
    if quantization != NF3_NF4_STRUCTURED_3P5:
        return {}

    candidates: list[tuple[float, str, int]] = []
    total_codes = 0
    priority_names = _policy_fp8_tensor_names(model_state, NF5_G128_LSQ_POLICY_LAST_FP8)
    for name, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not tensor.is_floating_point() or tensor.ndim != 2:
            continue
        if _is_critic_tensor_name(name):
            continue
        code_count = _padded_normalfloat_code_count(tensor)
        total_codes += code_count
        candidates.append(
            (_normalfloat_sensitivity_score(name, priority_names), name, code_count)
        )

    high_code_budget = total_codes // 2
    selected: set[str] = set()
    used_codes = 0
    for _score, name, code_count in sorted(candidates, reverse=True):
        if used_codes + code_count > high_code_budget and selected:
            continue
        selected.add(name)
        used_codes += code_count
        if used_codes >= high_code_budget:
            break

    return {
        name: 4 if name in selected else 3
        for name, tensor in model_state.items()
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
    }


def _padded_normalfloat_code_count(tensor: torch.Tensor) -> int:
    if tensor.ndim != 2:
        return 0
    rows, cols = tensor.shape
    group_count = (cols + _NORMALFLOAT_GROUP_SIZE - 1) // _NORMALFLOAT_GROUP_SIZE
    return rows * group_count * _NORMALFLOAT_GROUP_SIZE


def _normalfloat_sensitivity_score(
    name: str,
    priority_names: frozenset[str],
) -> float:
    score = 0.0
    if name in priority_names:
        score += 1000.0
    if name.startswith("actor."):
        score += 500.0
    if "source_actor_input_proj" in name or "target_actor_input_proj" in name:
        score += 500.0
    if name.startswith("static_") or name.endswith("_tokens"):
        score += 300.0
    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            block = int(parts[1])
            score += block * 10.0
            if ".attn.out." in name or ".mlp.down." in name:
                score += 100.0
    return score


def _policy_fp8_tensor_names(
    model_state: Mapping[str, torch.Tensor],
    quantization: QuantizationFormat,
) -> frozenset[str]:
    if quantization != NF5_G128_LSQ_POLICY_LAST_FP8:
        return frozenset()

    names = set(_ACTOR_FP8_TENSOR_NAMES)
    block_index = _last_transformer_block_index(model_state)
    if block_index is not None:
        for suffix in _TRUNK_FP8_TENSOR_SUFFIXES:
            name = f"blocks.{block_index}.{suffix}"
            if name in model_state:
                names.add(name)
    return frozenset(names)


def _last_transformer_block_index(
    model_state: Mapping[str, torch.Tensor],
) -> int | None:
    indices: set[int] = set()
    for name in model_state:
        if not name.startswith("blocks."):
            continue
        parts = name.split(".")
        if len(parts) > 2 and parts[1].isdigit():
            indices.add(int(parts[1]))
    return max(indices) if indices else None


def _is_critic_tensor_name(name: str) -> bool:
    return name.startswith("critic_head.") or ".critic_head." in name


def is_lora_adapter_state_key(name: str) -> bool:
    return name.endswith((".lora_down", ".lora_up"))


def quantize_model_state_dict(
    model_state: Mapping[str, torch.Tensor],
    quantization: QuantizationFormat | None,
    lora_quantization: TensorQuantizationFormat | None = None,
) -> dict[str, object]:
    if quantization is not None and quantization not in SUPPORTED_QUANTIZATION_FORMATS:
        raise ValueError(f"unsupported quantization format: {quantization}")

    validated_model_state = _validate_unquantized_model_state(model_state)
    lora_state = {
        name: tensor
        for name, tensor in validated_model_state.items()
        if is_lora_adapter_state_key(name)
    }
    base_state = {
        name: tensor
        for name, tensor in validated_model_state.items()
        if not is_lora_adapter_state_key(name)
    }
    effective_lora_quantization = _effective_lora_quantization(
        has_lora=bool(lora_state),
        quantization=quantization,
        lora_quantization=lora_quantization,
    )
    if quantization is None and effective_lora_quantization is None:
        raise ValueError("quantization requires a model or LoRA quantization format")

    normalfloat_bits: dict[str, int] = {}
    fp8_tensor_names: frozenset[str] = frozenset()
    if quantization is not None:
        normalfloat_bits.update(_normalfloat_tensor_bits(base_state, quantization))
        fp8_tensor_names = fp8_tensor_names | _policy_fp8_tensor_names(
            base_state,
            quantization,
        )
    if _is_supported_model_quantization(effective_lora_quantization):
        normalfloat_bits.update(
            _normalfloat_tensor_bits(lora_state, effective_lora_quantization)
        )
        fp8_tensor_names = fp8_tensor_names | _policy_fp8_tensor_names(
            lora_state,
            effective_lora_quantization,
        )

    quantized_state: dict[str, object] = {
        _QUANTIZED_STATE_MARKER: _QUANTIZED_STATE_VERSION,
        "format": FP32 if quantization is None else quantization,
        "tensors": {
            name: _quantize_state_tensor(
                name,
                tensor,
                (
                    effective_lora_quantization
                    if is_lora_adapter_state_key(name)
                    else quantization
                ),
                normalfloat_bits.get(name),
                fp8_tensor_names,
            )
            for name, tensor in validated_model_state.items()
        },
    }
    if lora_state:
        quantized_state["lora_format"] = (
            FP32 if effective_lora_quantization is None else effective_lora_quantization
        )
    return quantized_state


def _effective_lora_quantization(
    *,
    has_lora: bool,
    quantization: QuantizationFormat | None,
    lora_quantization: TensorQuantizationFormat | None,
) -> TensorQuantizationFormat | None:
    if not has_lora:
        return None
    if lora_quantization is None:
        return BF16 if quantization is not None else None
    if lora_quantization not in SUPPORTED_TENSOR_QUANTIZATION_FORMATS:
        raise ValueError(f"unsupported LoRA quantization format: {lora_quantization}")
    return lora_quantization


def _is_supported_model_quantization(
    quantization: TensorQuantizationFormat | None,
) -> TypeGuard[QuantizationFormat]:
    return quantization in SUPPORTED_QUANTIZATION_FORMATS


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


def load_model_state_dict_streaming(
    module: torch.nn.Module,
    model_state: object,
    *,
    allow_missing_lora_adapters: bool = False,
    ignore_unexpected_lora_adapters: bool = False,
) -> None:
    if not _is_quantized_model_state_dict(model_state):
        if not isinstance(model_state, Mapping):
            raise ValueError("checkpoint['model'] must be a mapping")
        _load_unquantized_model_state_dict(
            module,
            model_state,
            allow_missing_lora_adapters=allow_missing_lora_adapters,
            ignore_unexpected_lora_adapters=ignore_unexpected_lora_adapters,
        )
        return

    version = model_state[_QUANTIZED_STATE_MARKER]
    if version != _QUANTIZED_STATE_VERSION:
        raise ValueError(f"unsupported quantized model state version: {version}")

    tensors = model_state["tensors"]
    if not isinstance(tensors, Mapping):
        raise ValueError("quantized model state 'tensors' must be a mapping")

    destination = module.state_dict()
    loaded_keys: set[str] = set()
    unexpected_keys: list[str] = []
    error_messages: list[str] = []
    with torch.no_grad():
        for raw_name, payload in tensors.items():
            name = _validate_state_key(raw_name)
            target = destination.get(name)
            if target is None:
                if not (
                    ignore_unexpected_lora_adapters and is_lora_adapter_state_key(name)
                ):
                    unexpected_keys.append(name)
                continue

            loaded_keys.add(name)
            tensor = _dequantize_state_tensor(name, payload)
            if tensor.shape != target.shape:
                error_messages.append(
                    f"size mismatch for {name}: copying a param with shape "
                    f"{tuple(tensor.shape)} from checkpoint, the shape in current "
                    f"model is {tuple(target.shape)}."
                )
                del tensor
                continue
            target.copy_(tensor.to(device=target.device))
            del tensor

    missing_keys = set(destination) - loaded_keys
    _extend_state_dict_error_messages(
        module,
        error_messages,
        missing_keys=missing_keys,
        unexpected_keys=set(unexpected_keys),
        allow_missing_lora_adapters=allow_missing_lora_adapters,
        ignore_unexpected_lora_adapters=ignore_unexpected_lora_adapters,
    )
    if error_messages:
        _raise_state_dict_errors(module, error_messages)


def _load_unquantized_model_state_dict(
    module: torch.nn.Module,
    model_state: Mapping[object, object],
    *,
    allow_missing_lora_adapters: bool,
    ignore_unexpected_lora_adapters: bool,
) -> None:
    validated = _validate_unquantized_model_state(model_state)
    result = module.load_state_dict(validated, strict=False)
    error_messages: list[str] = []
    _extend_state_dict_error_messages(
        module,
        error_messages,
        missing_keys=set(result.missing_keys),
        unexpected_keys=set(result.unexpected_keys),
        allow_missing_lora_adapters=allow_missing_lora_adapters,
        ignore_unexpected_lora_adapters=ignore_unexpected_lora_adapters,
    )
    if error_messages:
        _raise_state_dict_errors(module, error_messages)


def _extend_state_dict_error_messages(
    module: torch.nn.Module,
    error_messages: list[str],
    *,
    missing_keys: set[str],
    unexpected_keys: set[str],
    allow_missing_lora_adapters: bool,
    ignore_unexpected_lora_adapters: bool,
) -> None:
    if ignore_unexpected_lora_adapters:
        unexpected_keys = {
            key for key in unexpected_keys if not is_lora_adapter_state_key(key)
        }

    if allow_missing_lora_adapters:
        destination_lora_keys = {
            name for name in module.state_dict() if is_lora_adapter_state_key(name)
        }
        missing_lora_keys = missing_keys & destination_lora_keys
        provided_lora_keys = destination_lora_keys - missing_lora_keys
        if not provided_lora_keys:
            missing_keys = missing_keys - missing_lora_keys

    if missing_keys:
        error_messages.append(
            "Missing key(s) in state_dict: "
            + ", ".join(repr(key) for key in sorted(missing_keys))
            + "."
        )
    if unexpected_keys:
        error_messages.append(
            "Unexpected key(s) in state_dict: "
            + ", ".join(repr(key) for key in sorted(unexpected_keys))
            + "."
        )


def _raise_state_dict_errors(
    module: torch.nn.Module,
    error_messages: list[str],
) -> None:
    module_name = module.__class__.__name__
    joined = "\n\t".join(error_messages)
    raise RuntimeError(f"Error(s) in loading state_dict for {module_name}:\n\t{joined}")


def _quantize_state_tensor(
    name: str,
    tensor: torch.Tensor,
    quantization: TensorQuantizationFormat | None,
    normalfloat_bits: int | None = None,
    fp8_tensor_names: frozenset[str] | None = None,
) -> dict[str, object]:
    _validate_state_key(name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"model state '{name}' must be a tensor")

    tensor = tensor.detach().cpu().contiguous()
    if quantization is None or quantization == FP32:
        return {
            _TENSOR_QUANTIZED_KEY: False,
            _TENSOR_DATA_KEY: tensor,
        }
    if not tensor.dtype.is_floating_point:
        return {
            _TENSOR_QUANTIZED_KEY: False,
            _TENSOR_DATA_KEY: tensor,
        }

    if quantization == FP16:
        return _quantize_dtype_state_tensor(tensor, "fp16")
    if quantization == BF16:
        return _quantize_dtype_state_tensor(tensor, "bf16")

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
            fp8_tensor_names or frozenset(),
        )
    if quantization in (
        NF4_G128_LSQ,
        NF3_NF4_STRUCTURED_3P5,
        NF3_G128_LSQ,
    ):
        if normalfloat_bits is None:
            raise ValueError(f"missing normal-float bit width for {name}")
        return _quantize_normalfloat_g128_lsq_state_tensor(
            tensor,
            quantization,
            normalfloat_bits,
        )
    raise ValueError(f"unsupported quantization format: {quantization}")


def _quantize_dtype_state_tensor(
    tensor: torch.Tensor,
    quantization: Literal["fp16", "bf16"],
) -> dict[str, object]:
    dtype = _dtype_for_precision_format(quantization)
    return {
        _TENSOR_QUANTIZED_KEY: True,
        _TENSOR_FORMAT_KEY: quantization,
        _TENSOR_SHAPE_KEY: tuple(tensor.shape),
        _TENSOR_SOURCE_DTYPE_KEY: str(tensor.dtype),
        _TENSOR_DATA_KEY: tensor.to(dtype),
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
    if quantization == FP4_E2M1FN_X2_SCALED_BLOCK16:
        scale = payload.get(_TENSOR_SCALE_KEY)
        if not isinstance(scale, torch.Tensor):
            raise ValueError(
                f"quantized model state '{state_key}' scale must be a tensor"
            )
        return _dequantize_fp4_e2m1fn_scaled_block16(data, scale, shape)
    if quantization in (FP16, BF16):
        return (
            data.to(_dtype_for_precision_format(quantization))
            .to(torch.float32)
            .reshape(shape)
        )
    if quantization == NF5_G128_LSQ_POLICY_LAST_FP8:
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
        return _dequantize_normalfloat_g128_lsq(
            data,
            scale,
            shape,
            cols,
            _validate_normalfloat_format_bits(
                payload.get(_TENSOR_BITS_KEY),
                state_key,
                quantization,
            ),
        )
    if quantization in (
        NF4_G128_LSQ,
        NF3_NF4_STRUCTURED_3P5,
        NF3_G128_LSQ,
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
        codebook = payload.get(_TENSOR_CODEBOOK_KEY)
        if codebook != _NORMALFLOAT_CODEBOOK:
            raise ValueError(
                f"quantized model state '{state_key}' codebook must be "
                f"{_NORMALFLOAT_CODEBOOK!r}"
            )
        return _dequantize_normalfloat_g128_lsq(
            data,
            scale,
            shape,
            cols,
            _validate_normalfloat_format_bits(
                payload.get(_TENSOR_BITS_KEY),
                state_key,
                quantization,
            ),
        )
    raise ValueError(
        f"quantized model state '{state_key}' has unsupported format: {quantization}"
    )


def _dtype_for_precision_format(quantization: Literal["fp16", "bf16"]) -> torch.dtype:
    if quantization == FP16:
        return torch.float16
    if quantization == BF16:
        return torch.bfloat16
    raise AssertionError(f"unsupported precision format: {quantization}")


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
    expected_bytes = (numel + 1) // 2
    if packed.numel() != expected_bytes:
        raise ValueError(
            f"fp4 payload has {packed.numel()} bytes, expected {expected_bytes}"
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
            _TENSOR_FORMAT_KEY: FP16,
            _TENSOR_SHAPE_KEY: tuple(tensor.shape),
            _TENSOR_SOURCE_DTYPE_KEY: str(tensor.dtype),
            _TENSOR_DATA_KEY: tensor.to(torch.float16),
        }

    data, scale = _quantize_normalfloat_g128_lsq(tensor, 5)
    return {
        _TENSOR_QUANTIZED_KEY: True,
        _TENSOR_FORMAT_KEY: quantization,
        _TENSOR_SHAPE_KEY: tuple(tensor.shape),
        _TENSOR_COLS_KEY: tensor.shape[1],
        _TENSOR_BITS_KEY: 5,
        _TENSOR_CODEBOOK_KEY: _NORMALFLOAT_CODEBOOK,
        _TENSOR_SOURCE_DTYPE_KEY: str(tensor.dtype),
        _TENSOR_DATA_KEY: data,
        _TENSOR_SCALE_KEY: scale,
    }


def _quantize_normalfloat_g128_lsq_state_tensor(
    tensor: torch.Tensor,
    quantization: QuantizationFormat,
    bits: int,
) -> dict[str, object]:
    if tensor.ndim != 2:
        return {
            _TENSOR_QUANTIZED_KEY: True,
            _TENSOR_FORMAT_KEY: FP16,
            _TENSOR_SHAPE_KEY: tuple(tensor.shape),
            _TENSOR_SOURCE_DTYPE_KEY: str(tensor.dtype),
            _TENSOR_DATA_KEY: tensor.to(torch.float16),
        }

    data, scale = _quantize_normalfloat_g128_lsq(tensor, bits)
    return {
        _TENSOR_QUANTIZED_KEY: True,
        _TENSOR_FORMAT_KEY: quantization,
        _TENSOR_SHAPE_KEY: tuple(tensor.shape),
        _TENSOR_COLS_KEY: tensor.shape[1],
        _TENSOR_BITS_KEY: bits,
        _TENSOR_CODEBOOK_KEY: _NORMALFLOAT_CODEBOOK,
        _TENSOR_SOURCE_DTYPE_KEY: str(tensor.dtype),
        _TENSOR_DATA_KEY: data,
        _TENSOR_SCALE_KEY: scale,
    }


def _quantize_normalfloat_g128_lsq(
    tensor: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_normalfloat_bits(bits, "normalfloat", None)
    tensor = tensor.to(torch.float32)
    if tensor.ndim != 2:
        raise ValueError("normal-float groupwise quantization requires 2D tensors")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("normal-float groupwise quantization requires finite tensors")

    rows, cols = tensor.shape
    group_count = (cols + _NORMALFLOAT_GROUP_SIZE - 1) // _NORMALFLOAT_GROUP_SIZE
    padded_cols = group_count * _NORMALFLOAT_GROUP_SIZE
    padded = torch.zeros((rows, padded_cols), dtype=torch.float32, device=tensor.device)
    padded[:, :cols] = tensor
    groups = padded.reshape(rows * group_count, _NORMALFLOAT_GROUP_SIZE)
    valid = torch.zeros((rows, padded_cols), dtype=torch.bool, device=tensor.device)
    valid[:, :cols] = True
    valid_groups = valid.reshape(rows * group_count, _NORMALFLOAT_GROUP_SIZE)

    valid_abs_groups = torch.where(valid_groups, groups.abs(), torch.zeros_like(groups))
    max_abs = valid_abs_groups.amax(dim=1, keepdim=True)
    scale = max_abs
    values = _normalfloat_values_for_bits(bits).to(device=tensor.device)
    thresholds = _normalfloat_thresholds_for_bits(bits).to(device=tensor.device)
    for _ in range(2):
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes = torch.bucketize(groups / safe_scale, thresholds).to(torch.long)
        quantized = values[codes]
        valid_quantized = torch.where(
            valid_groups,
            quantized,
            torch.zeros_like(quantized),
        )
        valid_group_values = torch.where(valid_groups, groups, torch.zeros_like(groups))
        denominator = (valid_quantized * valid_quantized).sum(dim=1, keepdim=True)
        raw_improved = (valid_group_values * valid_quantized).sum(
            dim=1, keepdim=True
        ) / denominator
        improved = torch.where(
            denominator > 0,
            torch.minimum(raw_improved.clamp_min(0.0), max_abs),
            torch.zeros_like(scale),
        )
        scale = torch.where(scale == 0, scale, improved)

    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.bucketize(groups / safe_scale, thresholds).to(torch.uint8).reshape(-1)
    return _pack_lowbit_codes(codes, bits), scale.reshape(-1).to(torch.float16)


def _dequantize_normalfloat_g128_lsq(
    data: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, ...],
    cols: int,
    bits: int,
) -> torch.Tensor:
    bits = _validate_normalfloat_bits(bits, "normalfloat", None)
    if len(shape) != 2:
        raise ValueError(f"normal-float payload shape must be 2D, got {shape}")
    rows = shape[0]
    if cols != shape[1]:
        raise ValueError(
            f"normal-float payload cols {cols} does not match shape {shape}"
        )
    group_count = (cols + _NORMALFLOAT_GROUP_SIZE - 1) // _NORMALFLOAT_GROUP_SIZE
    expected_codes = rows * group_count * _NORMALFLOAT_GROUP_SIZE
    codes = _unpack_lowbit_codes(data, expected_codes, bits)
    flat_scale = scale.contiguous().to(torch.float32).reshape(-1)
    expected_scales = rows * group_count
    if flat_scale.numel() != expected_scales:
        raise ValueError(
            f"normal-float payload has {flat_scale.numel()} scale values, "
            f"expected {expected_scales}"
        )

    values = _normalfloat_values_for_bits(bits).to(device=codes.device)
    dequantized = values[codes.long()].reshape(
        rows * group_count,
        _NORMALFLOAT_GROUP_SIZE,
    )
    dequantized *= flat_scale.to(device=codes.device).reshape(-1, 1)
    return dequantized.reshape(rows, group_count * _NORMALFLOAT_GROUP_SIZE)[
        :, :cols
    ].reshape(shape)


def _pack_lowbit_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    bits = _validate_normalfloat_bits(bits, "normalfloat", None)
    flat = codes.reshape(-1).to(torch.uint8)
    if ((flat >> bits) != 0).any().item():
        raise ValueError(f"{bits}-bit packing requires codes in [0, {1 << bits})")
    if bits == 3:
        flat = _pad_lowbit_codes(flat, 8)
        values = flat.reshape(-1, 8).to(torch.int64)
        packed = torch.empty(
            (values.shape[0], 3), dtype=torch.uint8, device=flat.device
        )
        packed[:, 0] = (
            values[:, 0] | (values[:, 1] << 3) | ((values[:, 2] & 0x03) << 6)
        ).to(torch.uint8)
        packed[:, 1] = (
            (values[:, 2] >> 2)
            | (values[:, 3] << 1)
            | (values[:, 4] << 4)
            | ((values[:, 5] & 0x01) << 7)
        ).to(torch.uint8)
        packed[:, 2] = (
            (values[:, 5] >> 1) | (values[:, 6] << 2) | (values[:, 7] << 5)
        ).to(torch.uint8)
        return packed.reshape(-1)
    if bits == 4:
        flat = _pad_lowbit_codes(flat, 2)
        values = flat.reshape(-1, 2).to(torch.int64)
        return (values[:, 0] | (values[:, 1] << 4)).to(torch.uint8)

    flat = _pad_lowbit_codes(flat, 8)
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


def _unpack_lowbit_codes(data: torch.Tensor, count: int, bits: int) -> torch.Tensor:
    bits = _validate_normalfloat_bits(bits, "normalfloat", None)
    packed = data.contiguous().view(torch.uint8).reshape(-1)
    expected_bytes = _lowbit_packed_bytes(count, bits)
    if packed.numel() != expected_bytes:
        raise ValueError(
            f"{bits}-bit payload has {packed.numel()} bytes, expected {expected_bytes}"
        )
    if bits == 3:
        packed = packed.reshape(-1, 3).to(torch.int64)
        codes = torch.empty(
            (packed.shape[0], 8), dtype=torch.uint8, device=packed.device
        )
        codes[:, 0] = (packed[:, 0] & 0x07).to(torch.uint8)
        codes[:, 1] = ((packed[:, 0] >> 3) & 0x07).to(torch.uint8)
        codes[:, 2] = (((packed[:, 0] >> 6) | (packed[:, 1] << 2)) & 0x07).to(
            torch.uint8
        )
        codes[:, 3] = ((packed[:, 1] >> 1) & 0x07).to(torch.uint8)
        codes[:, 4] = ((packed[:, 1] >> 4) & 0x07).to(torch.uint8)
        codes[:, 5] = (((packed[:, 1] >> 7) | (packed[:, 2] << 1)) & 0x07).to(
            torch.uint8
        )
        codes[:, 6] = ((packed[:, 2] >> 2) & 0x07).to(torch.uint8)
        codes[:, 7] = ((packed[:, 2] >> 5) & 0x07).to(torch.uint8)
        return codes.reshape(-1)[:count]
    if bits == 4:
        codes = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
        codes[0::2] = packed & 0x0F
        codes[1::2] = (packed >> 4) & 0x0F
        return codes[:count]

    packed = packed.reshape(-1, 5).to(torch.int64)
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


def _pad_lowbit_codes(codes: torch.Tensor, multiple: int) -> torch.Tensor:
    padding = (-codes.numel()) % multiple
    if padding == 0:
        return codes
    return torch.cat(
        (
            codes,
            torch.zeros(padding, dtype=torch.uint8, device=codes.device),
        )
    )


def _lowbit_packed_bytes(count: int, bits: int) -> int:
    if bits == 3:
        return ((count + 7) // 8) * 3
    if bits == 4:
        return (count + 1) // 2
    if bits == 5:
        return ((count + 7) // 8) * 5
    raise ValueError(f"unsupported normal-float bit width: {bits}")


def _normalfloat_values_for_bits(bits: int) -> torch.Tensor:
    bits = _validate_normalfloat_bits(bits, "normalfloat", None)
    if bits == 3:
        return _NF3_NORMAL_VALUES
    if bits == 4:
        return _NF4_NORMAL_VALUES
    return _NF5_NORMAL_VALUES


def _normalfloat_thresholds_for_bits(bits: int) -> torch.Tensor:
    bits = _validate_normalfloat_bits(bits, "normalfloat", None)
    if bits == 3:
        return _NF3_NORMAL_THRESHOLDS
    if bits == 4:
        return _NF4_NORMAL_THRESHOLDS
    return _NF5_NORMAL_THRESHOLDS


def _validate_normalfloat_bits(
    bits: object,
    name: str,
    default: int | None,
) -> int:
    if bits is None and default is not None:
        return default
    if not isinstance(bits, int) or bits not in (3, 4, 5):
        raise ValueError(
            f"quantized model state '{name}' bits must be one of 3, 4, or 5"
        )
    return bits


def _validate_normalfloat_format_bits(
    bits: object,
    name: str,
    quantization: QuantizationFormat,
) -> int:
    if quantization == NF5_G128_LSQ_POLICY_LAST_FP8:
        return _validate_exact_normalfloat_bits(bits, name, quantization, 5, 5)
    if quantization == NF4_G128_LSQ:
        return _validate_exact_normalfloat_bits(bits, name, quantization, 4, None)
    if quantization == NF3_G128_LSQ:
        return _validate_exact_normalfloat_bits(bits, name, quantization, 3, None)
    if quantization == NF3_NF4_STRUCTURED_3P5:
        validated = _validate_normalfloat_bits(bits, name, None)
        if validated in (3, 4):
            return validated
        raise ValueError(
            f"quantized model state '{name}' bits must be 3 or 4 for "
            f"format {quantization!r}"
        )
    raise ValueError(f"unsupported normal-float quantization format: {quantization}")


def _validate_exact_normalfloat_bits(
    bits: object,
    name: str,
    quantization: QuantizationFormat,
    expected: int,
    default: int | None,
) -> int:
    validated = _validate_normalfloat_bits(bits, name, default)
    if validated != expected:
        raise ValueError(
            f"quantized model state '{name}' bits must be {expected} for "
            f"format {quantization!r}"
        )
    return validated


def _is_quantized_model_state_dict(
    model_state: object,
) -> TypeGuard[Mapping[object, object]]:
    return (
        isinstance(model_state, Mapping)
        and model_state.get(_QUANTIZED_STATE_MARKER) is not None
    )


def _validate_unquantized_model_state(
    model_state: Mapping[Any, Any],
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
