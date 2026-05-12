from __future__ import annotations

import torch


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return weighted_mean(values, weights)


def masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(values, mask)
    weights = mask.to(dtype=values.dtype)
    variance = weighted_mean((values - mean).pow(2), weights)
    return variance.sqrt()


def explained_variance(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    values = target[valid_mask]
    errors = (target - predicted)[valid_mask]
    target_variance = torch.var(values, unbiased=False)
    if target_variance == 0:
        return torch.zeros((), dtype=predicted.dtype, device=predicted.device)
    return 1.0 - torch.var(errors, unbiased=False) / target_variance
