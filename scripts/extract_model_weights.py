from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, TypeAlias

import torch
from owl.agent.checkpoint_quantization import (
    SUPPORTED_QUANTIZATION_FORMATS,
    QuantizationFormat,
    quantize_model_state_dict,
)
from owl.utils import ResolvedFormat, parse_format_prefix_arg

FP32: Literal["fp32"] = "fp32"
OutputModelFormat: TypeAlias = QuantizationFormat | Literal["fp32"]
SUPPORTED_OUTPUT_MODEL_FORMATS: tuple[OutputModelFormat, ...] = (
    FP32,
    *SUPPORTED_QUANTIZATION_FORMATS,
)


def extract_model_weights(
    checkpoint_path: Path,
    output_path: Path,
    *,
    quantization: OutputModelFormat | None = None,
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

    if quantization is None or quantization == FP32:
        output_model_state = model_state
    else:
        output_model_state = quantize_model_state_dict(
            model_state,
            quantization,
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
        help="Optional model-weight quantization format for the slim checkpoint.",
    )
    return parser.parse_args()


def _parse_quantization_arg(quantization: str) -> ResolvedFormat:
    return parse_format_prefix_arg(
        quantization,
        allowed_formats=SUPPORTED_OUTPUT_MODEL_FORMATS,
        label="quantization format",
    )


def main() -> None:
    args = _parse_args()
    resolved_quantization = args.quantization
    if resolved_quantization.inferred_from is not None:
        print(
            f"Inferred quantization format {resolved_quantization.value!r} "
            f"from prefix {resolved_quantization.inferred_from!r}"
        )

    extract_model_weights(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        quantization=resolved_quantization.value,
    )


if __name__ == "__main__":
    main()
