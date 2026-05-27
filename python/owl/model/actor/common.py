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


class OutputProjectionMLP(nn.Module):
    def __init__(self, config: Any, output_dim: int) -> None:
        super().__init__()
        self.activation: Literal["gelu", "silu", "swiglu"] = config.activation
        match self.activation:
            case "gelu" | "silu":
                self.up = nn.Linear(config.embed_dim, config.embed_dim)
            case "swiglu":
                self.gate = nn.Linear(config.embed_dim, config.embed_dim)
                self.value = nn.Linear(config.embed_dim, config.embed_dim)
            case _:
                assert_never(self.activation)
        self.out = nn.Linear(config.embed_dim, output_dim)

    @property
    def weight(self) -> torch.Tensor:
        return self.out.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.out.bias

    @property
    def out_features(self) -> int:
        return self.out.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        match self.activation:
            case "gelu":
                return self.out(F.gelu(self.up(x)))
            case "silu":
                return self.out(F.silu(self.up(x)))
            case "swiglu":
                return self.out(F.silu(self.gate(x)) * self.value(x))
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


def binary_kl_from_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    teacher_logits = teacher_logits.float()
    student_logits = student_logits.float()
    teacher_prob = torch.sigmoid(teacher_logits)
    teacher_log_prob = F.logsigmoid(teacher_logits)
    teacher_log_not_prob = F.logsigmoid(-teacher_logits)
    student_log_prob = F.logsigmoid(student_logits)
    student_log_not_prob = F.logsigmoid(-student_logits)
    return teacher_prob * (teacher_log_prob - student_log_prob) + (
        1.0 - teacher_prob
    ) * (teacher_log_not_prob - student_log_not_prob)


def categorical_kl_from_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    teacher_prob = teacher_log_prob.exp()
    kl = teacher_prob * (teacher_log_prob - student_log_prob)
    return kl.masked_fill(~mask, 0.0).sum(dim=-1)
