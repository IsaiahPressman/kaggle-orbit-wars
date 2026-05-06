from __future__ import annotations

import argparse
from pathlib import Path

import torch


def extract_model_weights(checkpoint_path: Path, output_path: Path) -> None:
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_state}, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract only model weights from an OWL training checkpoint.",
    )
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("output_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    extract_model_weights(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
