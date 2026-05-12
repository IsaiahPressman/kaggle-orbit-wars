from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SegmentSample:
    indices: torch.Tensor
    importance: torch.Tensor


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
    importance = torch.ones((segments_per_minibatch, 1), device=device)
    return SegmentSample(
        indices=indices,
        importance=importance,
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

    permutation = torch.randperm(n_segments, device=device)
    samples: list[SegmentSample] = []
    for indices in permutation.split(segments_per_minibatch):
        importance = torch.ones((indices.shape[0], 1), device=device)
        samples.append(
            SegmentSample(
                indices=indices,
                importance=importance,
            )
        )
    return samples
