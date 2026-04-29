from __future__ import annotations

from typing import Literal, assert_never

import torch

from owl.train.utils import (
    assert_finite,
    require_probability_range,
    require_same_shape,
    require_segment_time_major,
)

AdvantageMode = Literal["gae", "gae_vtrace"]


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    ratios: torch.Tensor | None = None,
    mode: AdvantageMode = "gae",
    vtrace_rho_clip: float = 1.0,
    vtrace_c_clip: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = compute_advantages(
        values=values,
        rewards=rewards,
        dones=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
        bootstrap_values=last_values,
        ratios=ratios,
        mode=mode,
        vtrace_rho_clip=vtrace_rho_clip,
        vtrace_c_clip=vtrace_c_clip,
    )
    return advantages, advantages + values


def compute_advantages(
    *,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    bootstrap_values: torch.Tensor | None = None,
    ratios: torch.Tensor | None = None,
    mode: AdvantageMode = "gae",
    vtrace_rho_clip: float = 1.0,
    vtrace_c_clip: float = 1.0,
) -> torch.Tensor:
    """Compute advantages for segment-major/time-second tensors [N, T, ...]."""
    require_segment_time_major(values, "values")
    require_same_shape(values, rewards, left_name="values", right_name="rewards")
    require_same_shape(values, dones, left_name="values", right_name="dones")
    require_probability_range(gamma, "gamma")
    require_probability_range(gae_lambda, "gae_lambda")
    assert_finite(values, "values")
    assert_finite(rewards, "rewards")

    if bootstrap_values is None:
        next_values = torch.zeros_like(values[:, -1])
    else:
        expected_shape = values[:, -1].shape
        if bootstrap_values.shape != expected_shape:
            raise ValueError(
                f"bootstrap_values must have shape {expected_shape}, "
                f"got {bootstrap_values.shape}"
            )
        assert_finite(bootstrap_values, "bootstrap_values")
        next_values = bootstrap_values

    if mode == "gae":
        rho = torch.ones_like(values)
        c = torch.ones_like(values)
    elif mode == "gae_vtrace":
        if ratios is None:
            raise ValueError("ratios are required when mode='gae_vtrace'")
        require_same_shape(values, ratios, left_name="values", right_name="ratios")
        assert_finite(ratios, "ratios")
        if vtrace_rho_clip <= 0:
            raise ValueError("vtrace_rho_clip must be positive")
        if vtrace_c_clip <= 0:
            raise ValueError("vtrace_c_clip must be positive")
        rho = torch.clamp(ratios, max=vtrace_rho_clip)
        c = torch.clamp(ratios, max=vtrace_c_clip)
    else:
        assert_never(mode)

    dones_float = dones.to(dtype=values.dtype)
    advantages = torch.zeros_like(values)
    last_advantage = torch.zeros_like(values[:, -1])

    for step in range(values.shape[1] - 1, -1, -1):
        next_value = next_values if step == values.shape[1] - 1 else values[:, step + 1]
        next_nonterminal = 1.0 - dones_float[:, step]
        delta = rho[:, step] * (
            rewards[:, step] + gamma * next_value * next_nonterminal - values[:, step]
        )
        last_advantage = (
            delta + gamma * gae_lambda * c[:, step] * next_nonterminal * last_advantage
        )
        advantages[:, step] = last_advantage

    return advantages
