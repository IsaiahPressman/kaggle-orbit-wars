from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, TypeAlias, cast

import torch
from owl.checkpoint_quantization import (
    FP32,
    SUPPORTED_QUANTIZATION_FORMATS,
    SUPPORTED_TENSOR_QUANTIZATION_FORMATS,
    QuantizationFormat,
    TensorQuantizationFormat,
    dequantize_model_state_dict,
    effective_lora_quantization,
    is_lora_adapter_state_key,
    quantize_model_state_dict,
)
from owl.utils import ResolvedFormat, parse_format_prefix_arg

OutputModelFormat: TypeAlias = QuantizationFormat | Literal["fp32"]
SUPPORTED_OUTPUT_MODEL_FORMATS: tuple[OutputModelFormat, ...] = (
    "fp32",
    *SUPPORTED_QUANTIZATION_FORMATS,
)
LoRAOutputModelFormat: TypeAlias = TensorQuantizationFormat
SUPPORTED_LORA_OUTPUT_MODEL_FORMATS: tuple[LoRAOutputModelFormat, ...] = (
    *SUPPORTED_TENSOR_QUANTIZATION_FORMATS,
)


def extract_model_weights(
    checkpoint_path: Path,
    output_path: Path,
    *,
    quantization: OutputModelFormat | None = None,
    lora_quantization: LoRAOutputModelFormat | None = None,
) -> None:
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()

    if checkpoint_path == output_path:
        raise ValueError("output path must not overwrite the input checkpoint")
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
    if output_path.exists():
        raise ValueError(f"output path already exists: {output_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must be a dictionary: {checkpoint_path}")

    if "model" not in checkpoint:
        raise ValueError(f"checkpoint is missing 'model': {checkpoint_path}")

    model_state = checkpoint["model"]
    if not isinstance(model_state, dict):
        raise ValueError(f"checkpoint['model'] must be a dictionary: {checkpoint_path}")
    model_state = dequantize_model_state_dict(model_state)
    has_lora_adapters = any(is_lora_adapter_state_key(name) for name in model_state)
    base_quantization = _base_output_quantization(quantization)
    # Resolve the adapter format the same way quantize_model_state_dict will, so
    # the pass-through gate below matches what quantization would actually emit.
    effective_lora = effective_lora_quantization(
        has_lora=has_lora_adapters,
        quantization=base_quantization,
        lora_quantization=lora_quantization,
    )

    output_model_state: dict[str, torch.Tensor] | dict[str, object]
    if base_quantization is None and effective_lora is None:
        output_model_state = model_state
    else:
        output_model_state = quantize_model_state_dict(
            model_state,
            base_quantization,
            lora_quantization=lora_quantization,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": output_model_state}, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract only model weights from an OWL training checkpoint.",
    )
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--quantization",
        type=_parse_quantization_arg,
        default=ResolvedFormat(value=FP32),
        help=(
            "Optional model-weight quantization format for the slim checkpoint. "
            f"Choices: {', '.join(SUPPORTED_OUTPUT_MODEL_FORMATS)}. Unique "
            "prefixes are accepted when unambiguous."
        ),
    )
    parser.add_argument(
        "--lora-quantization",
        type=_parse_lora_quantization_arg,
        default=None,
        help=(
            "Optional LoRA adapter quantization format for adapter tensors. "
            "Choices: "
            f"{', '.join(SUPPORTED_LORA_OUTPUT_MODEL_FORMATS)}. Unique prefixes "
            "are accepted when unambiguous. Defaults to bf16 when --quantization "
            "is set to a non-fp32 format."
        ),
    )
    return parser.parse_args()


def _parse_quantization_arg(quantization: str) -> ResolvedFormat:
    return parse_format_prefix_arg(
        quantization,
        allowed_formats=SUPPORTED_OUTPUT_MODEL_FORMATS,
        label="quantization format",
    )


def _parse_lora_quantization_arg(quantization: str) -> ResolvedFormat:
    return parse_format_prefix_arg(
        quantization,
        allowed_formats=SUPPORTED_LORA_OUTPUT_MODEL_FORMATS,
        label="LoRA quantization format",
    )


def _base_output_quantization(
    quantization: OutputModelFormat | None,
) -> QuantizationFormat | None:
    if quantization is None or quantization == FP32:
        return None
    return cast(QuantizationFormat, quantization)


def main() -> None:
    args = _parse_args()
    resolved_quantization = args.quantization
    resolved_lora_quantization = args.lora_quantization
    if resolved_quantization.inferred_from is not None:
        print(
            f"Inferred quantization format {resolved_quantization.value!r} "
            f"from prefix {resolved_quantization.inferred_from!r}"
        )
    if (
        resolved_lora_quantization is not None
        and resolved_lora_quantization.inferred_from is not None
    ):
        print(
            "Inferred LoRA quantization format "
            f"{resolved_lora_quantization.value!r} from prefix "
            f"{resolved_lora_quantization.inferred_from!r}"
        )

    extract_model_weights(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        quantization=resolved_quantization.value,
        lora_quantization=(
            None
            if resolved_lora_quantization is None
            else resolved_lora_quantization.value
        ),
    )


if __name__ == "__main__":
    main()
