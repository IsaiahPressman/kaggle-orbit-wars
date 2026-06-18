from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from owl.checkpoint_quantization import (
    FP4_E2M1FN_X2_SCALED_BLOCK16,
    FP8_E4M3FN,
    NF3_G128_LSQ,
    NF3_NF4_STRUCTURED_3P5,
    NF4_G128_LSQ,
    NF5_G128_LSQ_POLICY_LAST_FP8,
    dequantize_model_state_dict,
    quantize_model_state_dict,
)

_SCRIPT_PATH = (
    Path(__file__).parents[2] / "scripts" / "roundtrip_checkpoint_quantization.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "roundtrip_checkpoint_quantization",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None
assert _SCRIPT_SPEC.loader is not None
roundtrip_checkpoint_quantization = importlib.util.module_from_spec(_SCRIPT_SPEC)
sys.modules["roundtrip_checkpoint_quantization"] = roundtrip_checkpoint_quantization
_SCRIPT_SPEC.loader.exec_module(roundtrip_checkpoint_quantization)


def test_roundtrip_checkpoint_model_dtype_preserves_non_model_keys(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / "checkpoint_fp16_roundtrip.pt"
    model_state = {
        "linear.weight": torch.tensor([[0.1, 1.5]], dtype=torch.float32),
        "step": torch.tensor(7, dtype=torch.int64),
    }
    checkpoint_extra = {"lr": torch.tensor(0.001)}
    torch.save(
        {
            "model": model_state,
            "env_steps": 123,
            "optimizer": checkpoint_extra,
        },
        checkpoint_path,
    )

    stats = roundtrip_checkpoint_quantization.roundtrip_checkpoint_model_dtype(
        checkpoint_path,
        "fp16",
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    assert checkpoint.keys() == {"model", "env_steps", "optimizer"}
    assert checkpoint["env_steps"] == 123
    assert torch.equal(checkpoint["optimizer"]["lr"], checkpoint_extra["lr"])
    expected = model_state["linear.weight"].to(torch.float16).to(torch.float32)
    _assert_float32_bits_equal(checkpoint["model"]["linear.weight"], expected)
    assert torch.equal(checkpoint["model"]["step"], model_state["step"])
    assert stats == roundtrip_checkpoint_quantization.RoundTripStats(
        converted_tensors=1,
        unchanged_tensors=1,
        original_dtypes=("torch.float32",),
    )


@pytest.mark.parametrize(
    "quantization",
    [
        FP8_E4M3FN,
        FP4_E2M1FN_X2_SCALED_BLOCK16,
        NF5_G128_LSQ_POLICY_LAST_FP8,
        NF4_G128_LSQ,
        NF3_NF4_STRUCTURED_3P5,
        NF3_G128_LSQ,
    ],
)
def test_roundtrip_checkpoint_model_dtype_supports_agent_quantization_formats(
    tmp_path: Path,
    quantization: str,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / f"checkpoint_{quantization}_roundtrip.pt"
    model_state = {
        "linear.weight": torch.tensor(
            [[-3.0, -1.25, -0.0], [0.25, 1.75, 5.0]],
            dtype=torch.float32,
        ),
        "step": torch.tensor(7, dtype=torch.int64),
    }
    torch.save({"model": model_state, "optimizer": {"state": {}}}, checkpoint_path)

    stats = roundtrip_checkpoint_quantization.roundtrip_checkpoint_model_dtype(
        checkpoint_path,
        quantization,
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    expected = dequantize_model_state_dict(
        quantize_model_state_dict(
            {"linear.weight": model_state["linear.weight"]},
            quantization,
        )
    )["linear.weight"]
    _assert_float32_bits_equal(checkpoint["model"]["linear.weight"], expected)
    assert torch.equal(checkpoint["model"]["step"], model_state["step"])
    assert checkpoint["optimizer"] == {"state": {}}
    assert stats == roundtrip_checkpoint_quantization.RoundTripStats(
        converted_tensors=1,
        unchanged_tensors=1,
        original_dtypes=("torch.float32",),
    )


def test_roundtrip_checkpoint_model_dtype_defaults_lora_to_bf16(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / f"checkpoint_{FP4_E2M1FN_X2_SCALED_BLOCK16}_roundtrip.pt"
    model_state = {
        "linear.weight": torch.tensor([[0.375, 1.125, 2.5]], dtype=torch.float32),
        "linear.lora_down": torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32),
    }
    torch.save({"model": model_state}, checkpoint_path)

    stats = roundtrip_checkpoint_quantization.roundtrip_checkpoint_model_dtype(
        checkpoint_path,
        FP4_E2M1FN_X2_SCALED_BLOCK16,
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    expected_base = dequantize_model_state_dict(
        quantize_model_state_dict(
            {"linear.weight": model_state["linear.weight"]},
            FP4_E2M1FN_X2_SCALED_BLOCK16,
        )
    )["linear.weight"]
    expected_lora = model_state["linear.lora_down"].to(torch.bfloat16).to(torch.float32)
    _assert_float32_bits_equal(checkpoint["model"]["linear.weight"], expected_base)
    _assert_float32_bits_equal(checkpoint["model"]["linear.lora_down"], expected_lora)
    assert stats == roundtrip_checkpoint_quantization.RoundTripStats(
        converted_tensors=2,
        unchanged_tensors=0,
        original_dtypes=("torch.float32",),
    )


def test_roundtrip_checkpoint_model_dtype_rejects_unique_prefix(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    model_state = {
        "linear.weight": torch.tensor([[0.25, 1.75, 5.0]], dtype=torch.float32),
    }
    torch.save({"model": model_state}, checkpoint_path)

    with pytest.raises(ValueError, match="target format must be one of"):
        roundtrip_checkpoint_quantization.roundtrip_checkpoint_model_dtype(
            checkpoint_path,
            "fp4",
        )


def test_roundtrip_checkpoint_model_dtype_rejects_unknown_target(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="target format must be one of"):
        roundtrip_checkpoint_quantization.roundtrip_checkpoint_model_dtype(
            checkpoint_path,
            "int8",
        )


def test_roundtrip_checkpoint_model_dtype_rejects_ambiguous_prefix(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="target format must be one of"):
        roundtrip_checkpoint_quantization.roundtrip_checkpoint_model_dtype(
            checkpoint_path,
            "fp",
        )


def test_roundtrip_output_path_uses_input_stem_format_and_suffix(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint_01_380_057_088.pt"

    output_path = roundtrip_checkpoint_quantization._roundtrip_output_path(
        checkpoint_path,
        FP4_E2M1FN_X2_SCALED_BLOCK16,
    )

    assert (
        output_path
        == tmp_path
        / "checkpoint_01_380_057_088_fp4_e2m1fn_x2_scaled_block16_roundtrip.pt"
    )


def test_main_prints_saved_roundtrip_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": {"weight": torch.ones(1)}}, checkpoint_path)
    output_path = tmp_path / "checkpoint_fp16_roundtrip.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "roundtrip_checkpoint_quantization.py",
            str(checkpoint_path),
            "fp16",
        ],
    )

    roundtrip_checkpoint_quantization.main()

    assert output_path.is_file()
    assert f"Saved round-tripped checkpoint to {output_path}" in capsys.readouterr().out


def test_main_prints_inferred_target_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": {"weight": torch.ones(1)}}, checkpoint_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "roundtrip_checkpoint_quantization.py",
            str(checkpoint_path),
            "fp4",
        ],
    )

    roundtrip_checkpoint_quantization.main()

    assert (
        "Inferred target format 'fp4_e2m1fn_x2_scaled_block16' from prefix 'fp4'"
        in capsys.readouterr().out
    )


def test_main_prints_multiline_diagnostics_with_model_only_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / ("checkpoint_fp4_e2m1fn_x2_scaled_block16_roundtrip.pt")
    model_tensor = torch.zeros(262_144, dtype=torch.float32)
    torch.save(
        {
            "model": {"weight": model_tensor},
            "optimizer": {"large_extra": torch.zeros(262_144, dtype=torch.float32)},
        },
        checkpoint_path,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "roundtrip_checkpoint_quantization.py",
            str(checkpoint_path),
            FP4_E2M1FN_X2_SCALED_BLOCK16,
        ],
    )

    roundtrip_checkpoint_quantization.main()

    assert capsys.readouterr().out.splitlines() == [
        (
            "Converted 1 floating tensors and 0 non-floating tensors unchanged; "
            "original dtype(s): torch.float32"
        ),
        (
            "Model weights size: 1.00 MiB before quantization, "
            "0.16 MiB after quantization"
        ),
        f"Saved round-tripped checkpoint to {output_path}",
    ]


def test_main_rejects_ambiguous_target_format_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": {"weight": torch.ones(1)}}, checkpoint_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "roundtrip_checkpoint_quantization.py",
            str(checkpoint_path),
            "fp",
        ],
    )

    with pytest.raises(SystemExit):
        roundtrip_checkpoint_quantization.main()

    captured = capsys.readouterr()
    assert "target format prefix 'fp' is ambiguous" in captured.err
    assert "fp4_e2m1fn_x2_scaled_block16" in captured.err


def test_main_help_lists_target_format_choices(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["roundtrip_checkpoint_quantization.py", "-h"])

    with pytest.raises(SystemExit) as exc_info:
        roundtrip_checkpoint_quantization.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "target_format" in captured.out
    assert "--lora-target-format" in captured.out
    assert "fp32" in captured.out
    assert "fp16" in captured.out
    assert "bf16" in captured.out
    assert FP8_E4M3FN in captured.out
    assert FP4_E2M1FN_X2_SCALED_BLOCK16 in captured.out
    assert NF5_G128_LSQ_POLICY_LAST_FP8 in captured.out
    assert NF4_G128_LSQ in captured.out
    assert NF3_NF4_STRUCTURED_3P5 in captured.out
    assert NF3_G128_LSQ in captured.out


def _assert_float32_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == torch.float32
    assert expected.dtype == torch.float32
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.int32),
        expected.contiguous().view(torch.int32),
    )
