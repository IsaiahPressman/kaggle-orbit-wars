from __future__ import annotations

from collections.abc import Callable

import torch

from owl.train.utils import (
    assert_finite,
    require_probability_range,
    require_same_shape,
    require_segment_time_major,
)


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    truncated: torch.Tensor | None = None,
    bootstrap_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_values = _advantage_tensor_inputs(
        values=values,
        rewards=rewards,
        dones=dones,
        bootstrap_values=last_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    truncated, bootstrap_values = _resolve_truncation_inputs(
        values=values,
        dones=dones,
        truncated=truncated,
        bootstrap_values=bootstrap_values,
    )
    return _compute_gae_tensors(
        rewards=rewards,
        values=values,
        dones=dones,
        next_values=next_values,
        truncated=truncated,
        bootstrap_values=bootstrap_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )


def compile_compute_gae(
    compile_mode: str | None,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    if compile_mode is None:
        return compute_gae
    compiled_compute_gae_tensors = torch.compile(
        _compute_gae_tensors,
        mode=compile_mode,
    )

    def compiled_compute_gae(
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        truncated: torch.Tensor | None = None,
        bootstrap_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_values = _advantage_tensor_inputs(
            values=values,
            rewards=rewards,
            dones=dones,
            bootstrap_values=last_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        truncated, bootstrap_values = _resolve_truncation_inputs(
            values=values,
            dones=dones,
            truncated=truncated,
            bootstrap_values=bootstrap_values,
        )
        return compiled_compute_gae_tensors(
            rewards=rewards,
            values=values,
            dones=dones,
            next_values=next_values,
            truncated=truncated,
            bootstrap_values=bootstrap_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    return compiled_compute_gae


def _advantage_tensor_inputs(
    *,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_values: torch.Tensor | None,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    require_segment_time_major(values, "values")
    require_same_shape(values, rewards, left_name="values", right_name="rewards")
    require_same_shape(values, dones, left_name="values", right_name="dones")
    require_probability_range(gamma, "gamma")
    require_probability_range(gae_lambda, "gae_lambda")
    assert_finite(values, "values")
    assert_finite(rewards, "rewards")

    if bootstrap_values is None:
        return torch.zeros_like(values[:, -1])

    expected_shape = values[:, -1].shape
    if bootstrap_values.shape != expected_shape:
        raise ValueError(
            f"bootstrap_values must have shape {expected_shape}, "
            f"got {bootstrap_values.shape}"
        )
    assert_finite(bootstrap_values, "bootstrap_values")
    return bootstrap_values


def _resolve_truncation_inputs(
    *,
    values: torch.Tensor,
    dones: torch.Tensor,
    truncated: torch.Tensor | None,
    bootstrap_values: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Default truncation inputs to all-false / zero, preserving plain GAE.

    Resolving ``None`` to concrete tensors here keeps the compiled
    ``_compute_gae_tensors`` graph free of optional arguments.
    """
    if truncated is None:
        truncated = torch.zeros_like(dones)
    else:
        require_same_shape(
            values, truncated, left_name="values", right_name="truncated"
        )
    if bootstrap_values is None:
        bootstrap_values = torch.zeros_like(values)
    else:
        require_same_shape(
            values,
            bootstrap_values,
            left_name="values",
            right_name="bootstrap_values",
        )
        assert_finite(bootstrap_values, "bootstrap_values")
    return truncated, bootstrap_values


def _compute_gae_tensors(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    truncated: torch.Tensor,
    bootstrap_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = _compute_gae_advantages_tensors(
        values=values,
        rewards=rewards,
        dones=dones,
        next_values=next_values,
        truncated=truncated,
        bootstrap_values=bootstrap_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return advantages, advantages + values


def _compute_gae_advantages_tensors(
    *,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    truncated: torch.Tensor,
    bootstrap_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    dones_float = dones.to(dtype=values.dtype)
    truncated_float = truncated.to(dtype=values.dtype)
    advantages = torch.zeros_like(values)
    last_advantage = torch.zeros_like(values[:, -1])

    for step in range(values.shape[1] - 1, -1, -1):
        next_value = next_values if step == values.shape[1] - 1 else values[:, step + 1]
        next_nonterminal = 1.0 - dones_float[:, step]
        # On a truncation step `dones` is true (so `next_value` drops out) and we
        # bootstrap from the critic's value of the truncated final state instead
        # of a real terminal return. On a genuine terminal both terms are zero.
        bootstrap = (
            next_value * next_nonterminal
            + truncated_float[:, step] * bootstrap_values[:, step]
        )
        delta = rewards[:, step] + gamma * bootstrap - values[:, step]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[:, step] = last_advantage

    return advantages
