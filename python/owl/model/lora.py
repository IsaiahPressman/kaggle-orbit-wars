from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from owl.model.base import BaseModelAPI
from owl.model.lora_config import LoRAConfig, LoRATargetModule
from owl.model.stateless_transformer_v1 import (
    StatelessTransformerV1,
    TransformerBlock,
)


@dataclass(frozen=True)
class LoRAApplication:
    module_count: int
    trainable_parameters: int


class LoRALinear(nn.Linear):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.weight = base.weight
        self.bias = base.bias
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_down = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.lora_up = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        update = self.lora_up(self.lora_dropout(self.lora_down(x))) * self.scaling
        return base + update


def apply_lora_to_stateless_transformer(
    model: BaseModelAPI,
    config: LoRAConfig,
) -> LoRAApplication:
    if not isinstance(model, StatelessTransformerV1):
        raise ValueError("LoRA fine-tuning only supports stateless_transformer_v1")
    if model.player_count_adapters:
        raise ValueError("LoRA fine-tuning does not support player-count adapters")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    block_indices = _target_block_indices(model, config)
    module_count = 0
    for block_index in block_indices:
        block = cast(TransformerBlock, model.blocks[block_index])
        for target_module in config.target_modules:
            _replace_target_linear(block, target_module, config)
            module_count += 1

    trainable_parameters = sum(
        parameter.numel() for parameter in lora_parameters(model)
    )
    return LoRAApplication(
        module_count=module_count,
        trainable_parameters=trainable_parameters,
    )


def lora_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    return tuple(
        parameter
        for name, parameter in model.named_parameters()
        if _is_lora_parameter_name(name)
    )


def load_model_state_dict_allowing_lora(
    model: nn.Module,
    state_dict: object,
) -> None:
    if not isinstance(state_dict, Mapping):
        raise ValueError("model state_dict must be a mapping")
    result = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = set(result.unexpected_keys)
    if unexpected_keys:
        unexpected = ", ".join(sorted(unexpected_keys))
        raise RuntimeError(f"unexpected model state_dict keys: {unexpected}")

    missing_keys = set(result.missing_keys)
    expected_missing = {name for name in model.state_dict() if _is_lora_state_key(name)}
    source_lora_keys = {
        key for key in state_dict if isinstance(key, str) and _is_lora_state_key(key)
    }
    if source_lora_keys:
        missing_lora_keys = missing_keys & expected_missing
        if missing_lora_keys:
            missing = ", ".join(sorted(missing_lora_keys))
            raise RuntimeError(f"missing LoRA model state_dict keys: {missing}")
    invalid_missing = missing_keys - expected_missing
    if invalid_missing:
        missing = ", ".join(sorted(invalid_missing))
        raise RuntimeError(f"missing non-LoRA model state_dict keys: {missing}")


def _target_block_indices(
    model: StatelessTransformerV1,
    config: LoRAConfig,
) -> range:
    block_count = len(model.blocks)
    if block_count == 0:
        raise ValueError("LoRA fine-tuning requires at least one shared block")
    if config.target_block_count is None:
        return range(block_count)
    if config.target_block_count > block_count:
        raise ValueError(
            "lora.target_block_count must be <= shared transformer block count "
            f"{block_count}, got {config.target_block_count}"
        )
    return range(block_count - config.target_block_count, block_count)


def _replace_target_linear(
    block: TransformerBlock,
    target_module: LoRATargetModule,
    config: LoRAConfig,
) -> None:
    # This dynamic replacement is deliberately narrow: the config names the fixed
    # transformer-block projections that can receive LoRA adapters.
    parent: nn.Module
    attribute: str
    match target_module:
        case "q" | "k" | "v" | "out":
            parent = block.attn
            attribute = target_module
        case "up" | "down" | "gate" | "value":
            parent = block.mlp
            attribute = target_module
        case _:
            raise AssertionError(f"unhandled LoRA target module: {target_module}")

    module = getattr(parent, attribute, None)
    if module is None:
        raise ValueError(f"LoRA target module does not exist: {target_module}")
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"LoRA target module must be nn.Linear: {target_module} "
            f"({type(module).__name__})"
        )
    if isinstance(module, LoRALinear):
        raise ValueError(f"LoRA target module is already wrapped: {target_module}")
    wrapped = LoRALinear(
        module,
        rank=config.rank,
        alpha=config.scaling_alpha,
        dropout=config.dropout,
    )
    setattr(parent, attribute, wrapped)


def _is_lora_parameter_name(name: str) -> bool:
    return ".lora_down." in name or ".lora_up." in name


def _is_lora_state_key(name: str) -> bool:
    return _is_lora_parameter_name(name)
