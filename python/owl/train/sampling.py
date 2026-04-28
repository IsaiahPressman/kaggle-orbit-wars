from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

import torch
from pydantic import Field

from owl.config import BaseConfig
from owl.train.utils import assert_finite, require_2d

SegmentSampling = Literal["uniform", "advantage_priority"]


class SegmentSamplingConfig(BaseConfig):
    sampling: SegmentSampling = "uniform"
    segments_per_minibatch: int = Field(default=1, ge=1)
    prio_alpha: float = Field(default=0.0, ge=0.0)
    prio_beta: float = Field(default=0.2, ge=0.0)
    prio_eps: float = Field(default=1e-6, gt=0.0)


@dataclass(frozen=True)
class SegmentSample:
    indices: torch.Tensor
    importance: torch.Tensor
    probabilities: torch.Tensor


@dataclass(frozen=True)
class SegmentSamplingMetrics:
    priority_min: torch.Tensor
    priority_mean: torch.Tensor
    priority_max: torch.Tensor
    probability_entropy: torch.Tensor
    duplicate_fraction: torch.Tensor
    importance_mean: torch.Tensor
    importance_max: torch.Tensor


def sample_segments(
    advantages: torch.Tensor,
    config: SegmentSamplingConfig,
) -> SegmentSample:
    require_2d(advantages, "advantages")
    if config.segments_per_minibatch <= 0:
        raise ValueError("segments_per_minibatch must be positive")
    if config.sampling == "uniform":
        return sample_segments_uniform(
            n_segments=advantages.shape[0],
            segments_per_minibatch=config.segments_per_minibatch,
            device=advantages.device,
        )
    if config.sampling == "advantage_priority":
        return sample_segments_by_advantage(
            advantages,
            segments_per_minibatch=config.segments_per_minibatch,
            alpha=config.prio_alpha,
            beta=config.prio_beta,
            eps=config.prio_eps,
        )
    assert_never(config.sampling)


def sample_segments_uniform(
    *,
    n_segments: int,
    segments_per_minibatch: int,
    device: torch.device | None = None,
) -> SegmentSample:
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    if segments_per_minibatch <= 0:
        raise ValueError("segments_per_minibatch must be positive")

    indices = torch.randint(0, n_segments, (segments_per_minibatch,), device=device)
    probabilities = torch.full(
        (n_segments,),
        1.0 / n_segments,
        dtype=torch.float32,
        device=device,
    )
    importance = torch.ones((segments_per_minibatch, 1), device=device)
    return SegmentSample(
        indices=indices,
        importance=importance,
        probabilities=probabilities,
    )


def sample_segments_uniform_single_pass(
    *,
    n_segments: int,
    segments_per_minibatch: int,
    device: torch.device | None = None,
) -> list[SegmentSample]:
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    if segments_per_minibatch <= 0:
        raise ValueError("segments_per_minibatch must be positive")

    probabilities = torch.full(
        (n_segments,),
        1.0 / n_segments,
        dtype=torch.float32,
        device=device,
    )
    permutation = torch.randperm(n_segments, device=device)
    samples: list[SegmentSample] = []
    for indices in permutation.split(segments_per_minibatch):
        importance = torch.ones((indices.shape[0], 1), device=device)
        samples.append(
            SegmentSample(
                indices=indices,
                importance=importance,
                probabilities=probabilities,
            )
        )
    return samples


def sample_segments_by_advantage(
    advantages: torch.Tensor,
    *,
    segments_per_minibatch: int,
    alpha: float,
    beta: float,
    eps: float = 1e-6,
) -> SegmentSample:
    require_2d(advantages, "advantages")
    assert_finite(advantages, "advantages")
    if segments_per_minibatch <= 0:
        raise ValueError("segments_per_minibatch must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")

    n_segments = advantages.shape[0]
    priority = advantages.detach().abs().sum(dim=1)
    weights = torch.nan_to_num(
        priority.pow(alpha),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    probabilities = (weights + eps) / (weights.sum() + eps * n_segments)
    indices = torch.multinomial(
        probabilities,
        segments_per_minibatch,
        replacement=True,
    )
    importance = (n_segments * probabilities[indices]).pow(-beta).unsqueeze(-1)
    importance = importance / importance.max().clamp_min(eps)
    return SegmentSample(
        indices=indices,
        importance=importance,
        probabilities=probabilities,
    )


def segment_sampling_metrics(
    advantages: torch.Tensor,
    sample: SegmentSample,
) -> SegmentSamplingMetrics:
    require_2d(advantages, "advantages")
    if sample.indices.ndim != 1:
        raise ValueError("sample.indices must be 1D")
    if sample.importance.shape != (sample.indices.shape[0], 1):
        raise ValueError(
            "sample.importance must have shape "
            f"({sample.indices.shape[0]}, 1), got {sample.importance.shape}"
        )
    if sample.probabilities.shape != (advantages.shape[0],):
        raise ValueError(
            f"sample.probabilities must have shape ({advantages.shape[0]},), "
            f"got {sample.probabilities.shape}"
        )

    priority = advantages.detach().abs().sum(dim=1)
    probabilities = sample.probabilities.clamp_min(torch.finfo(torch.float32).eps)
    probability_entropy = -(sample.probabilities * probabilities.log()).sum()
    unique_count = sample.indices.unique().numel()
    duplicate_fraction = 1.0 - unique_count / sample.indices.numel()
    return SegmentSamplingMetrics(
        priority_min=priority.min(),
        priority_mean=priority.mean(),
        priority_max=priority.max(),
        probability_entropy=probability_entropy,
        duplicate_fraction=torch.as_tensor(
            duplicate_fraction,
            dtype=sample.importance.dtype,
            device=sample.importance.device,
        ),
        importance_mean=sample.importance.mean(),
        importance_max=sample.importance.max(),
    )
