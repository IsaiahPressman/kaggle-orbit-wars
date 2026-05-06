from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "extract_model_weights.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "extract_model_weights",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None
assert _SCRIPT_SPEC.loader is not None
extract_model_weights = importlib.util.module_from_spec(_SCRIPT_SPEC)
sys.modules["extract_model_weights"] = extract_model_weights
_SCRIPT_SPEC.loader.exec_module(extract_model_weights)


def test_extract_model_weights_writes_only_model_key(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint_last_best.pt"
    output_path = tmp_path / "slim.pt"
    model_state = {"linear.weight": torch.ones((2, 3))}
    torch.save(
        {
            "model": model_state,
            "optimizer": {"state": {"large": torch.zeros((100, 100))}},
            "env_steps": 123,
        },
        checkpoint_path,
    )

    extract_model_weights.extract_model_weights(checkpoint_path, output_path)

    slim_checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    assert slim_checkpoint.keys() == {"model"}
    assert torch.equal(
        slim_checkpoint["model"]["linear.weight"], model_state["linear.weight"]
    )


def test_extract_model_weights_rejects_overwriting_input(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="must not overwrite"):
        extract_model_weights.extract_model_weights(checkpoint_path, checkpoint_path)


def test_extract_model_weights_rejects_missing_model_key(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / "slim.pt"
    torch.save({"optimizer": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="missing 'model'"):
        extract_model_weights.extract_model_weights(checkpoint_path, output_path)
