from __future__ import annotations

import torch
import torch.nn.functional as F

from owl.model import BaseModelAPI


class Int8EmulatedLinear(torch.nn.Module):
    weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, source: torch.nn.Linear) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_buffer(
            "weight",
            _fake_quantize_qint8_symmetric_per_tensor(source.weight.detach()),
        )
        self.register_buffer(
            "bias",
            None if source.bias is None else source.bias.detach().clone(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        autocast_dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else None
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            fake_quantized_x = _fake_quantize_quint8_affine_per_tensor(
                x.to(dtype=torch.float32)
            )
            weight = self.weight.to(dtype=torch.float32)
            bias = None if self.bias is None else self.bias.to(dtype=torch.float32)
            output = F.linear(fake_quantized_x, weight, bias)
        if autocast_dtype is None:
            return output
        return output.to(dtype=autocast_dtype)


def apply_int8_emulation(model: BaseModelAPI) -> int:
    output_layer_ids = {id(layer) for layer in model.get_output_layers()}
    return _replace_int8_emulated_linear_children(model, output_layer_ids)


def _replace_int8_emulated_linear_children(
    module: torch.nn.Module,
    output_layer_ids: set[int],
) -> int:
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            if id(child) in output_layer_ids:
                continue
            module.add_module(name, Int8EmulatedLinear(child))
            replaced += 1
            continue

        replaced += _replace_int8_emulated_linear_children(child, output_layer_ids)
    return replaced


def _fake_quantize_qint8_symmetric_per_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor.detach().clone()

    source_dtype = tensor.dtype
    tensor_f = tensor.detach().to(dtype=torch.float32)
    max_abs = torch.amax(torch.abs(tensor_f))
    scale = torch.clamp(max_abs * (2.0 / 255.0), min=torch.finfo(tensor_f.dtype).eps)
    quantized = torch.round(tensor_f / scale).clamp(-128, 127)
    return (quantized * scale).to(dtype=source_dtype)


def _fake_quantize_quint8_affine_per_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor.clone()

    source_dtype = tensor.dtype
    tensor_f = tensor.to(dtype=torch.float32)
    zero = tensor_f.new_zeros(())
    min_value = torch.minimum(torch.amin(tensor_f), zero)
    max_value = torch.maximum(torch.amax(tensor_f), zero)
    scale = torch.clamp(
        (max_value - min_value) / 255.0,
        min=torch.finfo(tensor_f.dtype).eps,
    )
    zero_point = torch.round(-min_value / scale).clamp(0, 255)
    quantized = torch.round(tensor_f / scale + zero_point).clamp(0, 255)
    return ((quantized - zero_point) * scale).to(dtype=source_dtype)
