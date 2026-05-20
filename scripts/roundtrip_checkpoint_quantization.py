#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from owl.agent.checkpoint_quantization import (
    SUPPORTED_QUANTIZATION_FORMATS,
    QuantizationFormat,
    dequantize_model_state_dict,
    quantize_model_state_dict,
)

TARGET_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
TARGET_FORMATS = tuple(sorted((*TARGET_DTYPES, *SUPPORTED_QUANTIZATION_FORMATS)))


@dataclass(frozen=True)
class RoundTripStats:
    converted_tensors: int
    unchanged_tensors: int
    original_dtypes: tuple[str, ...]


@dataclass(frozen=True)
class RoundTripResult:
    stats: RoundTripStats
    pre_quantization_model_size_bytes: int
    post_quantization_model_size_bytes: int


@dataclass(frozen=True)
class ModelRoundTripResult:
    model_state: MutableMapping[str, Any]
    stats: RoundTripStats
    post_quantization_model_size_bytes: int


@dataclass(frozen=True)
class ResolvedTargetFormat:
    value: str
    inferred_from: str | None = None


def roundtrip_checkpoint_model_dtype(
    checkpoint_path: Path,
    target_format: str,
) -> RoundTripStats:
    return _roundtrip_checkpoint_model_dtype(checkpoint_path, target_format).stats


def _roundtrip_checkpoint_model_dtype(
    checkpoint_path: Path,
    target_format: str,
) -> RoundTripResult:
    checkpoint_path = checkpoint_path.resolve()
    target_format = _target_format(target_format)
    output_path = _roundtrip_output_path(checkpoint_path, target_format)

    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
    if output_path.exists():
        raise ValueError(f"output path already exists: {output_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, MutableMapping):
        raise ValueError(f"checkpoint must be a mutable mapping: {checkpoint_path}")
    if "model" not in checkpoint:
        raise ValueError(f"checkpoint is missing 'model': {checkpoint_path}")

    model_state = checkpoint["model"]
    if not isinstance(model_state, MutableMapping):
        raise ValueError(
            f"checkpoint['model'] must be a mutable mapping: {checkpoint_path}"
        )

    pre_quantization_model_size_bytes = _model_state_storage_bytes(model_state)
    roundtrip_result = _roundtrip_model_state(
        model_state,
        target_format,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted_checkpoint = copy.copy(checkpoint)
    converted_checkpoint["model"] = roundtrip_result.model_state
    torch.save(converted_checkpoint, output_path)
    return RoundTripResult(
        stats=roundtrip_result.stats,
        pre_quantization_model_size_bytes=pre_quantization_model_size_bytes,
        post_quantization_model_size_bytes=(
            roundtrip_result.post_quantization_model_size_bytes
        ),
    )


def _roundtrip_output_path(checkpoint_path: Path, target_format: str) -> Path:
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_{target_format}_roundtrip{checkpoint_path.suffix}"
    )


def _roundtrip_model_state(
    model_state: MutableMapping[str, Any],
    target_format: str,
) -> ModelRoundTripResult:
    if target_format in TARGET_DTYPES:
        return _roundtrip_model_state_dtype(model_state, TARGET_DTYPES[target_format])
    if target_format in SUPPORTED_QUANTIZATION_FORMATS:
        return _roundtrip_model_state_quantized(
            model_state,
            _quantization_format(target_format),
        )
    allowed = ", ".join(TARGET_FORMATS)
    raise ValueError(f"target format must be one of: {allowed}")


def _roundtrip_model_state_dtype(
    model_state: MutableMapping[str, Any],
    target_dtype: torch.dtype,
) -> ModelRoundTripResult:
    converted_model_state = copy.copy(model_state)
    converted_tensors = 0
    unchanged_tensors = 0
    original_dtypes: set[str] = set()
    post_quantization_model_size_bytes = 0

    for name, value in model_state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint['model'][{name!r}] must be a tensor")

        if value.is_floating_point():
            original_dtypes.add(str(value.dtype))
            quantized_value = value.to(target_dtype)
            post_quantization_model_size_bytes += _tensor_storage_bytes(quantized_value)
            converted_model_state[name] = quantized_value.to(value.dtype)
            converted_tensors += 1
        else:
            converted_model_state[name] = value
            post_quantization_model_size_bytes += _tensor_storage_bytes(value)
            unchanged_tensors += 1

    return ModelRoundTripResult(
        model_state=converted_model_state,
        stats=RoundTripStats(
            converted_tensors=converted_tensors,
            unchanged_tensors=unchanged_tensors,
            original_dtypes=tuple(sorted(original_dtypes)),
        ),
        post_quantization_model_size_bytes=post_quantization_model_size_bytes,
    )


def _roundtrip_model_state_quantized(
    model_state: MutableMapping[str, Any],
    quantization: QuantizationFormat,
) -> ModelRoundTripResult:
    tensor_state: dict[str, torch.Tensor] = {}
    original_dtypes: dict[str, torch.dtype] = {}
    converted_tensors = 0
    unchanged_tensors = 0
    original_dtype_names: set[str] = set()

    for name, value in model_state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint['model'][{name!r}] must be a tensor")
        tensor_state[name] = value
        if value.is_floating_point():
            original_dtypes[name] = value.dtype
            original_dtype_names.add(str(value.dtype))
            converted_tensors += 1
        else:
            unchanged_tensors += 1

    quantized_state = quantize_model_state_dict(tensor_state, quantization)
    post_quantization_model_size_bytes = _tensor_payload_storage_bytes(quantized_state)
    dequantized_state = dequantize_model_state_dict(quantized_state)
    converted_model_state = copy.copy(model_state)
    for name, value in dequantized_state.items():
        if name in original_dtypes:
            converted_model_state[name] = value.to(original_dtypes[name])
        else:
            converted_model_state[name] = value

    return ModelRoundTripResult(
        model_state=converted_model_state,
        stats=RoundTripStats(
            converted_tensors=converted_tensors,
            unchanged_tensors=unchanged_tensors,
            original_dtypes=tuple(sorted(original_dtype_names)),
        ),
        post_quantization_model_size_bytes=post_quantization_model_size_bytes,
    )


def _model_state_storage_bytes(model_state: Mapping[str, Any]) -> int:
    total_size_bytes = 0
    for name, value in model_state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint['model'][{name!r}] must be a tensor")
        total_size_bytes += _tensor_storage_bytes(value)
    return total_size_bytes


def _tensor_payload_storage_bytes(payload: object) -> int:
    if isinstance(payload, torch.Tensor):
        return _tensor_storage_bytes(payload)
    if isinstance(payload, Mapping):
        return sum(_tensor_payload_storage_bytes(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(_tensor_payload_storage_bytes(value) for value in payload)
    return 0


def _tensor_storage_bytes(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().nbytes()


def _target_format(target_format: str) -> str:
    if target_format in TARGET_FORMATS:
        return target_format
    allowed = ", ".join(TARGET_FORMATS)
    raise ValueError(f"target format must be one of: {allowed}")


def _parse_target_format_arg(target_format: str) -> ResolvedTargetFormat:
    if target_format in TARGET_FORMATS:
        return ResolvedTargetFormat(value=target_format)

    matches = tuple(fmt for fmt in TARGET_FORMATS if fmt.startswith(target_format))
    if len(matches) == 1:
        return ResolvedTargetFormat(value=matches[0], inferred_from=target_format)

    allowed = ", ".join(TARGET_FORMATS)
    if len(matches) > 1:
        match_list = ", ".join(matches)
        raise argparse.ArgumentTypeError(
            f"target format prefix {target_format!r} is ambiguous; "
            f"matches: {match_list}"
        )
    raise argparse.ArgumentTypeError(f"target format must be one of: {allowed}")


def _quantization_format(target_format: str) -> QuantizationFormat:
    if target_format in SUPPORTED_QUANTIZATION_FORMATS:
        return target_format
    allowed = ", ".join(SUPPORTED_QUANTIZATION_FORMATS)
    raise ValueError(f"target quantization must be one of: {allowed}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round-trip checkpoint model weights through a lower-precision "
            "format and save the resulting checkpoint."
        ),
    )
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("target_format", type=_parse_target_format_arg)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    resolved_target = args.target_format
    if resolved_target.inferred_from is not None:
        print(
            f"Inferred target format {resolved_target.value!r} "
            f"from prefix {resolved_target.inferred_from!r}"
        )

    result = _roundtrip_checkpoint_model_dtype(
        checkpoint_path=args.checkpoint_path,
        target_format=resolved_target.value,
    )
    output_path = _roundtrip_output_path(
        args.checkpoint_path.resolve(),
        resolved_target.value,
    )
    stats = result.stats
    print(
        f"Converted {stats.converted_tensors} floating tensors and "
        f"{stats.unchanged_tensors} non-floating tensors unchanged; "
        f"original dtype(s): {_format_original_dtypes(stats.original_dtypes)}"
    )
    print(
        "Model weights size: "
        f"{_format_size_mib(result.pre_quantization_model_size_bytes)} "
        "before quantization, "
        f"{_format_size_mib(result.post_quantization_model_size_bytes)} "
        "after quantization"
    )
    print(f"Saved round-tripped checkpoint to {output_path}")


def _format_original_dtypes(original_dtypes: tuple[str, ...]) -> str:
    if not original_dtypes:
        return "none"
    return ", ".join(original_dtypes)


def _format_size_mib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MiB"


if __name__ == "__main__":
    main()
