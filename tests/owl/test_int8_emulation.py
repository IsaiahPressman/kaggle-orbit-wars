from __future__ import annotations

import copy
import warnings

import pytest
import torch
from owl import int8_emulation

_QUANTIZED_ENGINE_PREFERENCE = ("x86", "fbgemm", "qnnpack", "onednn")


def _ensure_quantized_engine_for_test() -> None:
    if torch.backends.quantized.engine != "none":
        return

    supported_engines = tuple(torch.backends.quantized.supported_engines)
    for engine in _QUANTIZED_ENGINE_PREFERENCE:
        if engine in supported_engines:
            torch.backends.quantized.engine = engine
            return

    supported = ", ".join(supported_engines) or "none"
    pytest.skip(f"no supported torch quantized backend available: {supported}")


def test_int8_emulated_linear_fake_quantizes_weights_and_activations() -> None:
    source = torch.nn.Linear(2, 2)
    source.weight.data.copy_(torch.tensor([[0.55, -0.20], [0.07, 0.33]]))
    source.bias.data.copy_(torch.tensor([0.10, -0.15]))
    emulated_linear = int8_emulation.Int8EmulatedLinear(source)
    x = torch.tensor([[0.13, -1.70], [2.40, 0.60]], dtype=torch.bfloat16)

    actual = emulated_linear(x)

    expected_x = int8_emulation._fake_quantize_quint8_affine_per_tensor(
        x.to(dtype=torch.float32)
    )
    expected_weight = int8_emulation._fake_quantize_qint8_symmetric_per_tensor(
        source.weight.detach()
    )
    expected = torch.nn.functional.linear(expected_x, expected_weight, source.bias)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_int8_emulated_linear_restores_autocast_dtype() -> None:
    source = torch.nn.Linear(2, 2)
    source.weight.data.copy_(torch.tensor([[0.55, -0.20], [0.07, 0.33]]))
    source.bias.data.copy_(torch.tensor([0.10, -0.15]))
    emulated_linear = int8_emulation.Int8EmulatedLinear(source)
    x = torch.tensor([[0.13, -1.70], [2.40, 0.60]], dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = emulated_linear(x)

    expected_x = int8_emulation._fake_quantize_quint8_affine_per_tensor(x)
    expected_weight = int8_emulation._fake_quantize_qint8_symmetric_per_tensor(
        source.weight.detach()
    )
    expected = torch.nn.functional.linear(expected_x, expected_weight, source.bias)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected.to(dtype=torch.bfloat16))


def test_int8_emulated_model_matches_torch_dynamic_quantization_except_output() -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = torch.nn.Linear(5, 4)
            self.output = torch.nn.Linear(4, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.output(torch.relu(self.body(x)))

        def get_output_layers(self) -> tuple[torch.nn.Module, ...]:
            return (self.output,)

    _ensure_quantized_engine_for_test()
    torch.manual_seed(99)
    base_model = FakeModel().eval()
    base_model.body.weight.data.uniform_(-0.9, 0.9)
    base_model.body.bias.data.uniform_(-0.3, 0.3)
    base_model.output.weight.data.uniform_(-0.7, 0.7)
    base_model.output.bias.data.uniform_(-0.2, 0.2)
    x = torch.randn(13, 5)
    emulated_model = copy.deepcopy(base_model).eval()
    real_model = copy.deepcopy(base_model).eval()

    int8_emulation.apply_int8_emulation(emulated_model)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.ao.quantization is deprecated.*",
            category=DeprecationWarning,
        )
        real_model = torch.quantization.quantize_dynamic(
            real_model,
            {
                torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
                "output": None,
            },
            dtype=torch.qint8,
            inplace=False,
        ).eval()

    assert isinstance(emulated_model.body, int8_emulation.Int8EmulatedLinear)
    assert not isinstance(real_model.body, torch.nn.Linear)
    assert isinstance(emulated_model.output, torch.nn.Linear)
    assert isinstance(real_model.output, torch.nn.Linear)
    torch.testing.assert_close(emulated_model(x), real_model(x), rtol=1e-6, atol=1e-6)
