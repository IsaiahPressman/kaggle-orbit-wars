from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from owl.model.base import BaseModelAPI
from owl.model.config import ModelConfig
from owl.model.lora_config import LoRAConfig, LoRATargetModule
from owl.model.lora_linear import LoRALinear
from owl.model.stateless_transformer_v1 import (
    StatelessTransformerV1,
    StatelessTransformerV1Config,
    TransformerBlock,
)


@dataclass(frozen=True)
class LoRAApplication:
    module_count: int
    trainable_parameters: int


def lora_config_for_model(config: ModelConfig) -> LoRAConfig | None:
    """Return the LoRA config for models that support LoRA, else ``None``.

    LoRA fine-tuning is only supported for stateless transformers; this is the
    single place that encodes which model architectures can carry a LoRA config.
    """
    if isinstance(config, StatelessTransformerV1Config):
        return config.lora
    return None


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

    module_count = 0
    if config.target_modules:
        for block_index in _target_block_indices(model, config):
            block = cast(TransformerBlock, model.blocks[block_index])
            for target_module in config.target_modules:
                _replace_target_linear(block, target_module, config)
                module_count += 1
    if config.target_value_head:
        module_count += _wrap_value_head(model, config)
    if config.target_policy_head:
        module_count += _wrap_policy_head(model, config)

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
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in (module.lora_down, module.lora_up)
    )


def roundtrip_lora_base_quantization(
    model: nn.Module,
    config: LoRAConfig,
) -> None:
    quantization = config.roundtrip_quantization
    if quantization is None:
        return

    from owl.checkpoint_quantization import (
        dequantize_model_state_dict,
        quantize_model_state_dict,
    )

    model_state = model.state_dict()
    base_state = {
        name: tensor
        for name, tensor in model_state.items()
        if not _is_lora_adapter_state_key(name)
    }
    dequantized = dequantize_model_state_dict(
        quantize_model_state_dict(base_state, quantization)
    )
    if dequantized.keys() != base_state.keys():
        raise RuntimeError("LoRA base quantization changed model state keys")

    destination = model.state_dict()
    with torch.no_grad():
        for name, tensor in dequantized.items():
            target = destination[name]
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))


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
    lora_keys = _lora_state_keys(model)
    # A source that supplies any adapter tensor is treated as a LoRA checkpoint
    # and must supply all of them; a source with no adapter tensors is a base
    # checkpoint and may leave every adapter to its config-initialized value.
    provided_lora_keys = lora_keys - missing_keys
    if provided_lora_keys:
        missing_lora_keys = lora_keys & missing_keys
        if missing_lora_keys:
            missing = ", ".join(sorted(missing_lora_keys))
            raise RuntimeError(f"missing LoRA model state_dict keys: {missing}")
    invalid_missing = missing_keys - lora_keys
    if invalid_missing:
        missing = ", ".join(sorted(invalid_missing))
        raise RuntimeError(f"missing non-LoRA model state_dict keys: {missing}")


def _lora_state_keys(model: nn.Module) -> set[str]:
    keys: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = f"{name}." if name else ""
            keys.add(f"{prefix}lora_down")
            keys.add(f"{prefix}lora_up")
    return keys


def _is_lora_adapter_state_key(name: str) -> bool:
    return name.endswith((".lora_down", ".lora_up"))


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
    if isinstance(module, LoRALinear):
        raise ValueError(f"LoRA target module is already wrapped: {target_module}")
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"LoRA target module must be nn.Linear: {target_module} "
            f"({type(module).__name__})"
        )
    _set_lora_linear(parent, attribute, module, config)


def _wrap_value_head(model: StatelessTransformerV1, config: LoRAConfig) -> int:
    if model.critic_head is None:
        raise ValueError("LoRA target_value_head requires a shared critic head")
    return _wrap_subtree_linears(model.critic_head, config)


def _wrap_policy_head(model: StatelessTransformerV1, config: LoRAConfig) -> int:
    actor = model.actor
    if actor is None:
        raise ValueError("LoRA target_policy_head requires a shared actor head")
    count = 0
    # The actor consumes source/target projections of the trunk hidden state;
    # wrap them alongside the actor body so the whole policy head adapts.
    input_projections = (
        ("source_actor_input_proj", model.source_actor_input_proj),
        ("target_actor_input_proj", model.target_actor_input_proj),
    )
    for attribute, projection in input_projections:
        if projection is None or isinstance(projection, LoRALinear):
            continue
        _set_lora_linear(model, attribute, projection, config)
        count += 1
    if model.pairwise_bias_mlp is not None:
        count += _wrap_subtree_linears(model.pairwise_bias_mlp, config)
    count += _wrap_subtree_linears(actor, config)
    return count


def _wrap_subtree_linears(root: nn.Module, config: LoRAConfig) -> int:
    # Recursively replace every plain nn.Linear leaf in this subtree with a LoRA
    # wrapper. Head modules are heterogeneous, so this generic surgery (rather
    # than a fixed list of named projections) is what lets a single flag adapt
    # an entire actor/critic head regardless of its internal structure.
    count = 0
    for name, child in list(root.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            _set_lora_linear(root, name, child, config)
            count += 1
        else:
            count += _wrap_subtree_linears(child, config)
    return count


def _set_lora_linear(
    parent: nn.Module,
    attribute: str,
    base: nn.Linear,
    config: LoRAConfig,
) -> None:
    # Module surgery: swap a frozen nn.Linear leaf for its LoRA-wrapped form.
    # setattr is required here because LoRALinear is intentionally not an
    # nn.Linear subclass, so the assignment cannot be statically typed.
    wrapped = LoRALinear(base, rank=config.rank, alpha=config.scaling_alpha)
    setattr(parent, attribute, wrapped)
