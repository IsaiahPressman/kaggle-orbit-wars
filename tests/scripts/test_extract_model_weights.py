from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from owl.agent.checkpoint_quantization import (
    FP4_E2M1FN_X2_SCALED_BLOCK16,
    FP8_E4M3FN,
    NF3_G128_LSQ,
    NF3_NF4_STRUCTURED_3P5,
    NF4_G128_LSQ,
    NF5_G128_LSQ_POLICY_LAST_FP8,
    dequantize_model_state_dict,
)

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


def test_extract_model_weights_can_quantize_model_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint_last_best.pt"
    output_path = tmp_path / "slim.pt"
    model_state = {"linear.weight": torch.tensor([[0.375, 1.125]])}
    torch.save(
        {
            "model": model_state,
            "optimizer": {"state": {"large": torch.zeros((100, 100))}},
        },
        checkpoint_path,
    )

    extract_model_weights.extract_model_weights(
        checkpoint_path,
        output_path,
        quantization=FP8_E4M3FN,
    )

    slim_checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    assert slim_checkpoint.keys() == {"model"}
    dequantized = dequantize_model_state_dict(slim_checkpoint["model"])
    expected = model_state["linear.weight"].to(torch.float8_e4m3fn).to(torch.float32)
    assert torch.equal(
        dequantized["linear.weight"].view(torch.int32),
        expected.view(torch.int32),
    )


def test_extract_model_weights_accepts_explicit_fp32_output(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint_last_best.pt"
    output_path = tmp_path / "slim.pt"
    model_state = {"linear.weight": torch.tensor([[0.375, 1.125]])}
    torch.save(
        {
            "model": model_state,
            "optimizer": {"state": {"large": torch.zeros((100, 100))}},
        },
        checkpoint_path,
    )

    extract_model_weights.extract_model_weights(
        checkpoint_path,
        output_path,
        quantization=extract_model_weights.FP32,
    )

    slim_checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    assert slim_checkpoint.keys() == {"model"}
    assert torch.equal(
        slim_checkpoint["model"]["linear.weight"], model_state["linear.weight"]
    )


def test_extract_model_weights_can_quantize_model_state_to_fp4(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint_last_best.pt"
    output_path = tmp_path / "slim.pt"
    model_state = {"linear.weight": torch.tensor([[0.375, 1.125, 2.5]])}
    torch.save({"model": model_state}, checkpoint_path)

    extract_model_weights.extract_model_weights(
        checkpoint_path,
        output_path,
        quantization=FP4_E2M1FN_X2_SCALED_BLOCK16,
    )

    slim_checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    assert slim_checkpoint.keys() == {"model"}
    assert slim_checkpoint["model"]["format"] == FP4_E2M1FN_X2_SCALED_BLOCK16
    dequantized = dequantize_model_state_dict(slim_checkpoint["model"])
    assert dequantized["linear.weight"].shape == model_state["linear.weight"].shape


@pytest.mark.parametrize(
    "quantization",
    [
        NF5_G128_LSQ_POLICY_LAST_FP8,
        NF4_G128_LSQ,
        NF3_NF4_STRUCTURED_3P5,
        NF3_G128_LSQ,
    ],
)
def test_extract_model_weights_can_quantize_model_state_to_grouped_normalfloat(
    tmp_path: Path,
    quantization: str,
) -> None:
    checkpoint_path = tmp_path / "checkpoint_last_best.pt"
    output_path = tmp_path / "slim.pt"
    model_state = {"linear.weight": torch.tensor([[0.375, 1.125, 2.5]])}
    torch.save({"model": model_state}, checkpoint_path)

    extract_model_weights.extract_model_weights(
        checkpoint_path,
        output_path,
        quantization=quantization,
    )

    slim_checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    assert slim_checkpoint.keys() == {"model"}
    assert slim_checkpoint["model"]["format"] == quantization
    dequantized = dequantize_model_state_dict(slim_checkpoint["model"])
    assert dequantized["linear.weight"].shape == model_state["linear.weight"].shape


def test_main_infers_quantization_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "checkpoint_last_best.pt"
    output_path = tmp_path / "slim.pt"
    model_state = {"linear.weight": torch.tensor([[0.375, 1.125, 2.5]])}
    torch.save({"model": model_state}, checkpoint_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_model_weights.py",
            str(checkpoint_path),
            str(output_path),
            "--quantization",
            "fp4",
        ],
    )

    extract_model_weights.main()

    captured = capsys.readouterr()
    assert (
        "Inferred quantization format 'fp4_e2m1fn_x2_scaled_block16' from prefix 'fp4'"
    ) in captured.out
    slim_checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    assert slim_checkpoint["model"]["format"] == FP4_E2M1FN_X2_SCALED_BLOCK16


def test_main_rejects_ambiguous_quantization_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "checkpoint_last_best.pt"
    output_path = tmp_path / "slim.pt"
    torch.save({"model": {"linear.weight": torch.ones(1)}}, checkpoint_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_model_weights.py",
            str(checkpoint_path),
            str(output_path),
            "--quantization",
            "fp",
        ],
    )

    with pytest.raises(SystemExit):
        extract_model_weights.main()

    captured = capsys.readouterr()
    assert "quantization format prefix 'fp' is ambiguous" in captured.err
    assert "fp4_e2m1fn_x2_scaled_block16" in captured.err


def test_main_help_lists_quantization_choices(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["extract_model_weights.py", "-h"])

    with pytest.raises(SystemExit) as exc_info:
        extract_model_weights.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--quantization" in captured.out
    assert extract_model_weights.FP32 in captured.out
    assert FP8_E4M3FN in captured.out
    assert FP4_E2M1FN_X2_SCALED_BLOCK16 in captured.out
    assert NF5_G128_LSQ_POLICY_LAST_FP8 in captured.out
    assert NF4_G128_LSQ in captured.out
    assert NF3_NF4_STRUCTURED_3P5 in captured.out
    assert NF3_G128_LSQ in captured.out


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


@pytest.mark.parametrize(
    ("model_state", "message"),
    [
        ({"linear.weight": "bad"}, "must be a tensor"),
        ({1: torch.ones(1)}, "model state keys must be non-empty strings"),
    ],
)
def test_extract_model_weights_rejects_malformed_fp32_model_state(
    tmp_path: Path,
    model_state: dict[object, object],
    message: str,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / "slim.pt"
    torch.save({"model": model_state}, checkpoint_path)

    with pytest.raises(ValueError, match=message):
        extract_model_weights.extract_model_weights(checkpoint_path, output_path)
