from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen base ``nn.Linear``.

    The base weight/bias are shared (not copied) and stay frozen; only the
    ``lora_down``/``lora_up`` factors are trainable. ``lora_up`` is zero-init so
    the wrapped module starts as an exact no-op. This is a plain ``nn.Module``
    rather than an ``nn.Linear`` subclass so that generic ``nn.Linear`` sweeps
    (e.g. ``reset_parameters``) do not touch the shared base weight or the
    adapter factors.

    This lives in its own module (rather than ``owl.model.lora``) so that
    ``owl.model.stateless_transformer_v1`` can import it at module scope without
    forming an import cycle with the surgery helpers in ``owl.model.lora``.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        # Clamp the adapter rank to min(in, out): a rank above that cannot raise
        # the update's rank, so it would only add redundant parameters (e.g. the
        # critic's embed_dim -> 1 output). The adapter stays separate from the
        # frozen base weight, and the scaling keeps the configured alpha / rank so
        # every adapter shares the same update scale regardless of clamping.
        self.rank = min(rank, base.in_features, base.out_features)
        self.scaling = alpha / rank
        self.weight = base.weight
        self.bias = base.bias
        self.lora_down = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_up = nn.Parameter(
            torch.empty(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        # Fold the constant LoRA scale into the rank-sized intermediate instead of
        # the full out_features-wide update. F.linear is linear, so scaling the
        # low-rank activation is exactly equivalent to scaling the final update
        # while avoiding an out_features-wide multiply/allocation every forward.
        down = F.linear(x, self.lora_down) * self.scaling
        update = F.linear(down, self.lora_up)
        return base + update
