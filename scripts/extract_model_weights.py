from __future__ import annotations

import argparse
from pathlib import Path

import torch
from owl.agent.checkpoint_quantization import (
    SUPPORTED_QUANTIZATION_FORMATS,
    QuantizationFormat,
    quantize_model_state_dict,
)


def extract_model_weights(
    checkpoint_path: Path,
    output_path: Path,
    *,
    quantization: QuantizationFormat | None = None,
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

    output_model_state = (
        model_state
        if quantization is None
        else quantize_model_state_dict(model_state, quantization)
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
        choices=SUPPORTED_QUANTIZATION_FORMATS,
        default=None,
        help="Optional model-weight quantization format for the slim checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    extract_model_weights(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()
