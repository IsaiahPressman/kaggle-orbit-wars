from pathlib import Path

import pytest
import torch
from owl.agent.agent import Agent, AgentCheckpointConfig, AgentConfig
from owl.agent.checkpoint_quantization import (
    FP4_E2M1FN_X2_SCALED_BLOCK16,
    FP8_E4M3FN,
    dequantize_model_state_dict,
    quantize_model_state_dict,
)
from owl.model import StatelessTransformerV1Config
from owl.rl import EntityBasedConfig, EnvConfig

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
_FP4_BLOCK_SIZE = 16


def test_fp8_quantization_matches_torch_cast_bits() -> None:
    values = torch.tensor(
        [[-3.0, -2.25, -0.0, 0.0], [0.375, 1.125, 2.5, 3.0]],
        dtype=torch.float32,
    )

    quantized = quantize_model_state_dict({"weight": values}, FP8_E4M3FN)

    payload = quantized["tensors"]["weight"]
    expected_lowp = values.to(torch.float8_e4m3fn)
    assert torch.equal(payload["data"], expected_lowp.view(torch.uint8))

    dequantized = dequantize_model_state_dict(quantized)["weight"]
    _assert_float32_bits_equal(dequantized, expected_lowp.to(torch.float32))


def test_fp8_quantized_payload_uses_one_storage_byte_per_value() -> None:
    values = torch.arange(15, dtype=torch.float32).reshape(3, 5)

    quantized = quantize_model_state_dict({"weight": values}, FP8_E4M3FN)

    data = quantized["tensors"]["weight"]["data"]
    assert data.dtype == torch.uint8
    assert data.numel() == values.numel()
    assert _tensor_storage_nbytes(data) == values.numel()


def test_fp4_quantization_matches_scaled_block16_code_table_bits() -> None:
    values = torch.tensor(
        [
            -10.0,
            -5.0,
            -3.5,
            -2.5,
            -1.75,
            -1.25,
            -0.75,
            -0.25,
            -0.0,
            0.0,
            0.25,
            0.5,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            10.0,
            -0.12,
            -0.04,
            0.0,
            0.04,
            0.12,
        ],
        dtype=torch.float32,
    )

    quantized = quantize_model_state_dict(
        {"weight": values},
        FP4_E2M1FN_X2_SCALED_BLOCK16,
    )

    expected_data, expected_scale, expected_dequantized = (
        _reference_fp4_e2m1fn_scaled_block16(values)
    )
    payload = quantized["tensors"]["weight"]
    assert torch.equal(payload["data"], expected_data)
    assert torch.equal(payload["scale"], expected_scale)

    dequantized = dequantize_model_state_dict(quantized)["weight"]
    _assert_float32_bits_equal(dequantized, expected_dequantized)


def test_fp4_quantized_payload_uses_packed_bytes_and_fp16_block_scales() -> None:
    values = torch.arange(17, dtype=torch.float32)

    quantized = quantize_model_state_dict(
        {"weight": values},
        FP4_E2M1FN_X2_SCALED_BLOCK16,
    )

    data = quantized["tensors"]["weight"]["data"]
    scale = quantized["tensors"]["weight"]["scale"]
    expected_data_bytes = (values.numel() + 1) // 2
    expected_scale_values = (values.numel() + _FP4_BLOCK_SIZE - 1) // _FP4_BLOCK_SIZE
    assert data.dtype == torch.uint8
    assert data.numel() == expected_data_bytes
    assert _tensor_storage_nbytes(data) == expected_data_bytes
    assert scale.dtype == torch.float16
    assert scale.numel() == expected_scale_values
    assert _tensor_storage_nbytes(scale) == expected_scale_values * 2


def test_fp4_dequantization_unpacks_low_nibble_then_high_nibble() -> None:
    codes = torch.arange(16, dtype=torch.uint8)
    quantized = {
        "__owl_quantized_model_state_dict__": 1,
        "format": FP4_E2M1FN_X2_SCALED_BLOCK16,
        "tensors": {
            "weight": {
                "quantized": True,
                "format": FP4_E2M1FN_X2_SCALED_BLOCK16,
                "shape": (16,),
                "source_dtype": "torch.float32",
                "data": _pack_reference_fp4_codes(codes),
                "scale": torch.tensor([1.0], dtype=torch.float16),
            },
        },
    }

    dequantized = dequantize_model_state_dict(quantized)["weight"]

    _assert_float32_bits_equal(dequantized, _FP4_E2M1FN_VALUES)


@pytest.mark.parametrize("quantization", [FP8_E4M3FN, FP4_E2M1FN_X2_SCALED_BLOCK16])
def test_quantized_checkpoint_file_round_trips_to_expected_fp32(
    tmp_path: Path,
    quantization: str,
) -> None:
    values = torch.tensor(
        [[-3.0, -1.25, -0.0], [0.25, 1.75, 5.0]],
        dtype=torch.float32,
    )
    expected = _expected_dequantized(values, quantization)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": quantize_model_state_dict({"weight": values}, quantization)},
        path,
    )

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    dequantized = dequantize_model_state_dict(loaded["model"])

    assert dequantized.keys() == {"weight"}
    _assert_float32_bits_equal(dequantized["weight"], expected)


def test_unquantized_model_state_still_loads_unchanged() -> None:
    values = torch.ones((2, 3), dtype=torch.float32)

    dequantized = dequantize_model_state_dict({"weight": values})

    assert dequantized.keys() == {"weight"}
    assert dequantized["weight"] is values


def test_agent_init_loads_quantized_checkpoint_as_fp32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_weight = torch.tensor([[0.25, 1.75, 5.0]], dtype=torch.float32)
    expected_weight = _expected_dequantized(
        source_weight,
        FP4_E2M1FN_X2_SCALED_BLOCK16,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_config_path = tmp_path / "config.yaml"
    checkpoint_config_path.write_text("unused\n")
    torch.save(
        {
            "model": quantize_model_state_dict(
                {"weight": source_weight},
                FP4_E2M1FN_X2_SCALED_BLOCK16,
            ),
        },
        checkpoint_path,
    )
    model = torch.nn.Linear(3, 1, bias=False)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        AgentConfig,
        "from_file",
        classmethod(lambda _cls, _path: AgentConfig(deterministic=True)),
    )
    monkeypatch.setattr(
        AgentCheckpointConfig,
        "from_file",
        classmethod(
            lambda _cls, _path: AgentCheckpointConfig(
                env=EnvConfig(obs_spec=EntityBasedConfig(max_entities=64)),
                model=StatelessTransformerV1Config(),
            )
        ),
    )
    monkeypatch.setattr(
        "owl.agent.agent.create_model",
        lambda *_args, **_kwargs: model,
    )

    agent = Agent(
        checkpoint_config_path=checkpoint_config_path,
        checkpoint_path=checkpoint_path,
    )

    assert agent.model.weight.dtype == torch.float32
    _assert_float32_bits_equal(agent.model.weight.detach(), expected_weight)


def _expected_dequantized(values: torch.Tensor, quantization: str) -> torch.Tensor:
    if quantization == FP8_E4M3FN:
        return values.to(torch.float8_e4m3fn).to(torch.float32)
    if quantization == FP4_E2M1FN_X2_SCALED_BLOCK16:
        return _reference_fp4_e2m1fn_scaled_block16(values)[2]
    raise AssertionError(f"unexpected quantization: {quantization}")


def _reference_fp4_e2m1fn_scaled_block16(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_values = values.to(torch.float32).reshape(-1)
    codes = torch.empty_like(flat_values, dtype=torch.uint8)
    scale = torch.empty(
        (flat_values.numel() + _FP4_BLOCK_SIZE - 1) // _FP4_BLOCK_SIZE,
        dtype=torch.float16,
    )
    dequantized = torch.empty_like(flat_values, dtype=torch.float32)

    for block_index, start in enumerate(range(0, flat_values.numel(), _FP4_BLOCK_SIZE)):
        block = flat_values[start : start + _FP4_BLOCK_SIZE]
        max_abs = block.abs().max()
        block_scale = max_abs / 6.0
        stored_scale = block_scale.to(torch.float16)
        scale[block_index] = stored_scale
        normalized = block if max_abs == 0 else block / block_scale
        block_codes = _reference_fp4_e2m1fn_codes(normalized)
        codes[start : start + block.numel()] = block_codes
        dequantized[start : start + block.numel()] = _FP4_E2M1FN_VALUES[
            block_codes.long()
        ] * stored_scale.to(torch.float32)

    return (
        _pack_reference_fp4_codes(codes),
        scale,
        dequantized.reshape(values.shape),
    )


def _reference_fp4_e2m1fn_codes(values: torch.Tensor) -> torch.Tensor:
    positive_values = _FP4_E2M1FN_VALUES[:8]
    abs_values = values.abs().to(torch.float32)
    distances = (abs_values.unsqueeze(-1) - positive_values).abs()
    codes = distances.argmin(dim=-1).to(torch.uint8)

    tie_values = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        dtype=torch.float32,
    )
    flat_abs = abs_values.reshape(-1)
    flat_codes = codes.reshape(-1)
    is_tie = (flat_abs.unsqueeze(-1) == tie_values).any(dim=-1)
    lower_tie_code = torch.searchsorted(tie_values, flat_abs, right=False).to(
        torch.uint8
    )
    flat_codes[is_tie & (lower_tie_code % 2 == 1)] += 1

    sign = torch.signbit(values).to(torch.uint8) << 3
    return codes | sign


def _pack_reference_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    flat_codes = codes.reshape(-1).to(torch.uint8)
    packed = torch.zeros((flat_codes.numel() + 1) // 2, dtype=torch.uint8)
    packed |= flat_codes[0::2] & 0x0F
    if flat_codes.numel() > 1:
        packed[: flat_codes[1::2].numel()] |= (flat_codes[1::2] & 0x0F) << 4
    return packed


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().nbytes()


def _assert_float32_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == torch.float32
    assert expected.dtype == torch.float32
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.int32),
        expected.contiguous().view(torch.int32),
    )
