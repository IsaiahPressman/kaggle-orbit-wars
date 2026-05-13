from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def log_interpolate(
    low: float,
    high: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    log_low = math.log(low)
    high = high.float()
    weight = weight.float()
    return (log_low + weight * (high.log() - log_low)).exp()


def logsubexp(log_x: torch.Tensor, log_y: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(log_x.dtype).eps
    ratio = (log_y - log_x).exp().clamp_max(1.0 - eps)
    return log_x + torch.log1p(-ratio)


def logistic_cdf_diff_logprob(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    log_cdf_hi = F.logsigmoid(hi)
    log_cdf_lo = F.logsigmoid(lo)
    left = logsubexp(log_cdf_hi, log_cdf_lo)

    log_sf_lo = F.logsigmoid(-lo)
    log_sf_hi = F.logsigmoid(-hi)
    right = logsubexp(log_sf_lo, log_sf_hi)

    return torch.where((lo + hi) > 0.0, right, left)


def discretized_logistic_mixture_log_prob(
    ships: torch.Tensor,
    residual_budget: torch.Tensor,
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    *,
    min_fleet_size: int,
) -> torch.Tensor:
    mix_logits = mix_logits.float()
    mu = mu.float()
    scale = scale.float()
    n = ships.to(torch.float32).unsqueeze(-1)
    safe_residual_budget = residual_budget.clamp_min(min_fleet_size)
    residual = safe_residual_budget.to(torch.float32).unsqueeze(-1)

    valid = (
        (ships >= min_fleet_size)
        & (ships <= residual_budget)
        & (residual_budget >= min_fleet_size)
    )

    lo = (n - 0.5 - mu) / scale
    hi = (n + 0.5 - mu) / scale

    support_lo = (float(min_fleet_size) - 0.5 - mu) / scale
    support_hi = (residual + 0.5 - mu) / scale

    log_bin_mass = logistic_cdf_diff_logprob(lo, hi)
    log_support_mass = logistic_cdf_diff_logprob(support_lo, support_hi)
    log_w = F.log_softmax(mix_logits, dim=-1)
    log_comp = log_w + log_bin_mass - log_support_mass
    log_comp = torch.where(
        valid.unsqueeze(-1),
        log_comp,
        torch.full_like(log_comp, -torch.inf),
    )
    logp = torch.logsumexp(log_comp, dim=-1)
    return torch.where(valid, logp, torch.full_like(logp, -torch.inf))


def sample_discretized_logistic_mixture(
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    residual_budget: torch.Tensor,
    *,
    min_fleet_size: int,
    deterministic: bool,
    deterministic_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not deterministic:
        mix_index = Categorical(logits=mix_logits.float()).sample()
        gather_index = mix_index.unsqueeze(-1)
        selected_mu = mu.gather(-1, gather_index).squeeze(-1).float()
        selected_scale = scale.gather(-1, gather_index).squeeze(-1).float()

        residual = residual_budget.float()
        lo_count = float(min_fleet_size) - 0.5
        hi_count = residual + 0.5
        cdf_lo = torch.sigmoid((lo_count - selected_mu) / selected_scale)
        cdf_hi = torch.sigmoid((hi_count - selected_mu) / selected_scale)
        support_mass = (cdf_hi - cdf_lo).clamp_min(1e-12)
        u = cdf_lo + torch.rand_like(cdf_lo) * support_mass
        y = selected_mu + selected_scale * torch.logit(u.clamp(1e-6, 1.0 - 1e-6))
        ships = y.round().to(dtype=torch.int64).clamp_min(min_fleet_size)
        return torch.minimum(ships, residual_budget.clamp_min(min_fleet_size))

    if (
        deterministic_mask is not None
        and deterministic_mask.shape != residual_budget.shape
    ):
        raise ValueError(
            "deterministic_mask must match residual_budget shape, "
            f"got {deterministic_mask.shape} and {residual_budget.shape}"
        )

    if residual_budget.device.type == "cpu":
        return _deterministic_logistic_map_cpu_loop(
            mix_logits,
            mu,
            scale,
            residual_budget,
            min_fleet_size=min_fleet_size,
            deterministic_mask=deterministic_mask,
        )

    return _deterministic_logistic_map_indexed(
        mix_logits,
        mu,
        scale,
        residual_budget,
        min_fleet_size=min_fleet_size,
        deterministic_mask=deterministic_mask,
    )


def _deterministic_logistic_map_cpu_loop(
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    residual_budget: torch.Tensor,
    *,
    min_fleet_size: int,
    deterministic_mask: torch.Tensor | None,
) -> torch.Tensor:
    output = torch.zeros_like(residual_budget, dtype=torch.int64)
    if deterministic_mask is None:
        selected = torch.arange(residual_budget.numel(), device=residual_budget.device)
    else:
        selected = torch.nonzero(deterministic_mask.reshape(-1), as_tuple=True)[0]
    if selected.numel() == 0:
        return output

    flat_residual = residual_budget.reshape(-1)
    flat_mix_logits = mix_logits.reshape(-1, mix_logits.shape[-1])
    flat_mu = mu.reshape(-1, mu.shape[-1])
    flat_scale = scale.reshape(-1, scale.shape[-1])
    flat_output = output.reshape(-1)
    for index in selected.tolist():
        residual = int(flat_residual[index].item())
        if residual < min_fleet_size:
            flat_output[index] = min_fleet_size
            continue

        support = torch.arange(
            min_fleet_size,
            residual + 1,
            device=residual_budget.device,
            dtype=residual_budget.dtype,
        )
        support_residual = torch.full_like(support, residual)
        log_probs = discretized_logistic_mixture_log_prob(
            support,
            support_residual,
            flat_mix_logits[index],
            flat_mu[index],
            flat_scale[index],
            min_fleet_size=min_fleet_size,
        )
        flat_output[index] = support[log_probs.argmax()]
    return output


def _deterministic_logistic_map_indexed(
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    residual_budget: torch.Tensor,
    *,
    min_fleet_size: int,
    deterministic_mask: torch.Tensor | None,
) -> torch.Tensor:
    output = torch.zeros_like(residual_budget, dtype=torch.int64)
    if deterministic_mask is None:
        selected = torch.arange(residual_budget.numel(), device=residual_budget.device)
    else:
        selected = torch.nonzero(deterministic_mask.reshape(-1), as_tuple=True)[0]
    if selected.numel() == 0:
        return output

    row_shape = mix_logits.shape[-1]
    flat_residual = residual_budget.reshape(-1)
    selected_residual = flat_residual.index_select(0, selected)
    selected_support = ship_support(
        selected_residual,
        min_fleet_size=min_fleet_size,
        max_ship_support=int(
            selected_residual.max().clamp_min(min_fleet_size).item()
            - min_fleet_size
            + 1
        ),
    )
    log_probs = discretized_logistic_mixture_log_prob(
        selected_support,
        selected_residual.unsqueeze(-1),
        mix_logits.reshape(-1, row_shape).index_select(0, selected).unsqueeze(-2),
        mu.reshape(-1, row_shape).index_select(0, selected).unsqueeze(-2),
        scale.reshape(-1, row_shape).index_select(0, selected).unsqueeze(-2),
        min_fleet_size=min_fleet_size,
    )
    valid = selected_support <= selected_residual.unsqueeze(-1)
    log_probs = log_probs.masked_fill(~valid, torch.finfo(log_probs.dtype).min)
    support_index = log_probs.argmax(dim=-1)
    selected_support = selected_support.expand_as(log_probs)
    selected_ships = selected_support.gather(
        dim=-1,
        index=support_index.unsqueeze(-1),
    ).squeeze(-1)
    return output.reshape(-1).scatter(0, selected, selected_ships).view_as(output)


def ship_support(
    residual_budget: torch.Tensor,
    *,
    min_fleet_size: int,
    max_ship_support: int,
) -> torch.Tensor:
    max_residual = int(residual_budget.max().item())
    support_count = max(max_residual - min_fleet_size + 1, 1)
    support_count = min(support_count, max_ship_support)
    support_count = max(support_count, 1)
    offsets = torch.arange(
        support_count,
        device=residual_budget.device,
        dtype=residual_budget.dtype,
    )
    return min_fleet_size + offsets.view(
        *((1,) * residual_budget.ndim),
        support_count,
    )


def truncated_logistic_mixture_entropy(
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    residual_budget: torch.Tensor,
    *,
    min_fleet_size: int,
    entropy_ship_quantiles: int,
) -> torch.Tensor:
    mix_logits = mix_logits.float()
    mu = mu.float()
    scale = scale.float()
    safe_residual_budget = residual_budget.clamp_min(min_fleet_size).float()
    support_lo = float(min_fleet_size) - 0.5
    support_hi = safe_residual_budget.unsqueeze(-1) + 0.5

    cdf_lo = torch.sigmoid((support_lo - mu) / scale)
    cdf_hi = torch.sigmoid((support_hi - mu) / scale)
    support_mass = (cdf_hi - cdf_lo).clamp_min(1e-12)
    quantiles = (
        torch.arange(
            entropy_ship_quantiles,
            device=mu.device,
            dtype=mu.dtype,
        )
        + 0.5
    ) / entropy_ship_quantiles
    quantiles = quantiles.view(*((1,) * mu.ndim), entropy_ship_quantiles)
    cdf_samples = cdf_lo.unsqueeze(-1) + support_mass.unsqueeze(-1) * quantiles
    cdf_samples = cdf_samples.clamp(1e-6, 1.0 - 1e-6)
    samples = mu.unsqueeze(-1) + scale.unsqueeze(-1) * torch.logit(cdf_samples)

    z = (samples.unsqueeze(-1) - mu.unsqueeze(-2).unsqueeze(-2)) / scale.unsqueeze(
        -2,
    ).unsqueeze(-2)
    log_pdf = (
        F.logsigmoid(z) + F.logsigmoid(-z) - scale.log().unsqueeze(-2).unsqueeze(-2)
    )
    component_log_prob = (
        F.log_softmax(mix_logits, dim=-1).unsqueeze(-2).unsqueeze(-2)
        + log_pdf
        - support_mass.log().unsqueeze(-2).unsqueeze(-2)
    )
    log_prob = torch.logsumexp(component_log_prob, dim=-1)
    component_entropy = -log_prob.mean(dim=-1)
    mix_prob = torch.softmax(mix_logits, dim=-1)
    return (mix_prob * component_entropy).sum(dim=-1)
