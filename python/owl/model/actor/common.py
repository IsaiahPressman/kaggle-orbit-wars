from __future__ import annotations

from typing import Any, Literal, assert_never

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli


class FeedForward(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.activation: Literal["gelu", "silu", "swiglu"] = config.activation
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        match self.activation:
            case "gelu" | "silu":
                self.up = nn.Linear(config.embed_dim, hidden_dim)
                self.down = nn.Linear(hidden_dim, config.embed_dim)
            case "swiglu":
                self.gate = nn.Linear(config.embed_dim, hidden_dim)
                self.value = nn.Linear(config.embed_dim, hidden_dim)
                self.down = nn.Linear(hidden_dim, config.embed_dim)
            case _:
                assert_never(self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        match self.activation:
            case "gelu":
                return self.down(F.gelu(self.up(x)))
            case "silu":
                return self.down(F.silu(self.up(x)))
            case "swiglu":
                return self.down(F.silu(self.gate(x)) * self.value(x))
            case _:
                assert_never(self.activation)


def sample_launch(
    logits: torch.Tensor,
    active: torch.Tensor,
    *,
    deterministic: bool,
) -> torch.Tensor:
    logits = logits.float()
    if deterministic:
        launch = logits.sigmoid() > 0.5
    else:
        launch = Bernoulli(logits=logits).sample().bool()

    return launch & active


def binary_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    return F.binary_cross_entropy_with_logits(logits, probability, reduction="none")
