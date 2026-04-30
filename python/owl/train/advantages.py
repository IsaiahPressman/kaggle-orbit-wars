from __future__ import annotations

from typing import Literal, Protocol, assert_never

import torch

from owl.train.utils import (
    assert_finite,
    require_probability_range,
    require_same_shape,
    require_segment_time_major,
)

AdvantageMode = Literal["gae", "puffer_vtrace"]
type BootstrappedAdvantageMode = Literal["gae"]


class PufferVTraceFn(Protocol):
    def __call__(
        self,
        *,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        ratios: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        vtrace_rho_clip: float,
        vtrace_c_clip: float,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class ComputeGAEFn(Protocol):
    def __call__(
        self,
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
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


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
    if mode == "puffer_vtrace":
        if ratios is None:
            raise ValueError("ratios are required when mode='puffer_vtrace'")
        return compute_puffer_vtrace_action_aligned(
            values=values,
            rewards=rewards,
            dones=dones,
            ratios=ratios,
            gamma=gamma,
            gae_lambda=gae_lambda,
            vtrace_rho_clip=vtrace_rho_clip,
            vtrace_c_clip=vtrace_c_clip,
        )

    next_values, rho, c = _advantage_tensor_inputs(
        values=values,
        rewards=rewards,
        dones=dones,
        bootstrap_values=last_values,
        mode=mode,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return _compute_gae_tensors(
        rewards=rewards,
        values=values,
        dones=dones,
        next_values=next_values,
        rho=rho,
        c=c,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )


def compile_compute_gae(compile_mode: str | None) -> ComputeGAEFn:
    if compile_mode is None:
        return compute_gae
    compiled_compute_gae_tensors = torch.compile(
        _compute_gae_tensors,
        mode=compile_mode,
    )
    compiled_puffer_vtrace_tensors: PufferVTraceFn | None = None

    def compiled_compute_gae(
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
        if mode == "puffer_vtrace":
            if ratios is None:
                raise ValueError("ratios are required when mode='puffer_vtrace'")
            nonlocal compiled_puffer_vtrace_tensors
            if compiled_puffer_vtrace_tensors is None:
                compiled_puffer_vtrace_tensors = torch.compile(
                    _compute_puffer_vtrace_action_aligned_tensors,
                    mode=compile_mode,
                )
            _validate_puffer_vtrace_inputs(
                values=values,
                rewards=rewards,
                dones=dones,
                ratios=ratios,
                gamma=gamma,
                gae_lambda=gae_lambda,
                vtrace_rho_clip=vtrace_rho_clip,
                vtrace_c_clip=vtrace_c_clip,
            )
            return compiled_puffer_vtrace_tensors(
                values=values,
                rewards=rewards,
                dones=dones,
                ratios=ratios,
                gamma=gamma,
                gae_lambda=gae_lambda,
                vtrace_rho_clip=vtrace_rho_clip,
                vtrace_c_clip=vtrace_c_clip,
            )

        next_values, rho, c = _advantage_tensor_inputs(
            values=values,
            rewards=rewards,
            dones=dones,
            bootstrap_values=last_values,
            mode=mode,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        return compiled_compute_gae_tensors(
            rewards=rewards,
            values=values,
            dones=dones,
            next_values=next_values,
            rho=rho,
            c=c,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    return compiled_compute_gae


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
    if mode == "puffer_vtrace":
        if ratios is None:
            raise ValueError("ratios are required when mode='puffer_vtrace'")
        advantages, _returns = compute_puffer_vtrace_action_aligned(
            values=values,
            rewards=rewards,
            dones=dones,
            ratios=ratios,
            gamma=gamma,
            gae_lambda=gae_lambda,
            vtrace_rho_clip=vtrace_rho_clip,
            vtrace_c_clip=vtrace_c_clip,
        )
        return advantages

    next_values, rho, c = _advantage_tensor_inputs(
        values=values,
        rewards=rewards,
        dones=dones,
        bootstrap_values=bootstrap_values,
        mode=mode,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return _compute_advantages_tensors(
        values=values,
        rewards=rewards,
        dones=dones,
        next_values=next_values,
        rho=rho,
        c=c,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )


def compute_puffer_vtrace_action_aligned(
    *,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    ratios: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Puffer-style V-trace for action-aligned rollout tensors."""
    _validate_puffer_vtrace_inputs(
        values=values,
        rewards=rewards,
        dones=dones,
        ratios=ratios,
        gamma=gamma,
        gae_lambda=gae_lambda,
        vtrace_rho_clip=vtrace_rho_clip,
        vtrace_c_clip=vtrace_c_clip,
    )
    return _compute_puffer_vtrace_action_aligned_tensors(
        values=values,
        rewards=rewards,
        dones=dones,
        ratios=ratios,
        gamma=gamma,
        gae_lambda=gae_lambda,
        vtrace_rho_clip=vtrace_rho_clip,
        vtrace_c_clip=vtrace_c_clip,
    )


def _validate_puffer_vtrace_inputs(
    *,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    ratios: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
) -> None:
    require_segment_time_major(values, "values")
    require_same_shape(values, rewards, left_name="values", right_name="rewards")
    require_same_shape(values, dones, left_name="values", right_name="dones")
    require_same_shape(values, ratios, left_name="values", right_name="ratios")
    require_probability_range(gamma, "gamma")
    require_probability_range(gae_lambda, "gae_lambda")
    assert_finite(values, "values")
    assert_finite(rewards, "rewards")
    assert_finite(ratios, "ratios")
    if vtrace_rho_clip <= 0:
        raise ValueError("vtrace_rho_clip must be positive")
    if vtrace_c_clip <= 0:
        raise ValueError("vtrace_c_clip must be positive")


def _compute_puffer_vtrace_action_aligned_tensors(
    *,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    ratios: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(values)
    last_advantage = torch.zeros_like(values[:, 0])

    rho = torch.clamp(ratios, max=vtrace_rho_clip)
    c = torch.clamp(ratios, max=vtrace_c_clip)
    dones_float = dones.to(dtype=values.dtype)

    for step in range(values.shape[1] - 2, -1, -1):
        next_nonterminal = 1.0 - dones_float[:, step]
        delta = rho[:, step] * (
            rewards[:, step]
            + gamma * values[:, step + 1] * next_nonterminal
            - values[:, step]
        )
        last_advantage = (
            delta + gamma * gae_lambda * c[:, step] * next_nonterminal * last_advantage
        )
        advantages[:, step] = last_advantage

    return advantages, advantages + values


def _advantage_tensor_inputs(
    *,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_values: torch.Tensor | None,
    mode: BootstrappedAdvantageMode,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    else:
        assert_never(mode)

    return next_values, rho, c


def _compute_gae_tensors(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    rho: torch.Tensor,
    c: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = _compute_advantages_tensors(
        values=values,
        rewards=rewards,
        dones=dones,
        next_values=next_values,
        rho=rho,
        c=c,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return advantages, advantages + values


def _compute_advantages_tensors(
    *,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    rho: torch.Tensor,
    c: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
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
