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


def compute_winner_lambda_targets(
    *,
    winner_probabilities: torch.Tensor,
    terminal_winner: torch.Tensor,
    game_done: torch.Tensor,
    game_truncated: torch.Tensor,
    last_winner_probabilities: torch.Tensor,
    bootstrap_winner_probabilities: torch.Tensor,
    gae_lambda: float,
) -> torch.Tensor:
    """GAE(lambda) target distribution for the winner-probability critic.

    With ``gamma == 1`` and a terminal-only ``win_only`` reward (whose per-step
    reward vector is itself the winner distribution), the scalar lambda-return is
    a convex, lambda-weighted average of the future critic win-probabilities and
    the terminal winner distribution. Carrying that exact recursion on the
    per-player distribution instead of the scalar value yields a valid target
    distribution whose ``value_mode='win_only'`` scalar value (``value = p``)
    equals the scalar lambda-return:

        q_t = (1 - lambda) * p(s_{t+1}) + lambda * q_{t+1}    (non-terminal)
        q_t = terminal winner distribution                    (whole-game done)
        q_t = critic win-probabilities at the cut state        (time-limit trunc)

    ``lambda == 1`` collapses to the Monte-Carlo winner distribution; ``lambda
    == 0`` is the one-step bootstrap ``p(s_{t+1})``.

    Unlike :func:`compute_gae`, the transition masks are whole-game and per-env
    (``game_done``, ``game_truncated`` have shape ``(n_envs, horizon)``): the
    winner softmax is a joint distribution over players, so its target is a
    single per-state distribution that only resolves when the whole game ends.
    ``terminal_winner`` is the winner distribution at whole-game terminals (the
    ``win_only`` reward vector); it is ignored on non-terminal steps.
    """
    require_segment_time_major(winner_probabilities, "winner_probabilities")
    require_probability_range(gae_lambda, "gae_lambda")
    require_same_shape(
        winner_probabilities,
        terminal_winner,
        left_name="winner_probabilities",
        right_name="terminal_winner",
    )
    require_same_shape(
        winner_probabilities,
        bootstrap_winner_probabilities,
        left_name="winner_probabilities",
        right_name="bootstrap_winner_probabilities",
    )
    assert_finite(winner_probabilities, "winner_probabilities")
    expected_mask_shape = winner_probabilities.shape[:2]
    if game_done.shape != expected_mask_shape:
        raise ValueError(
            f"game_done must have shape {tuple(expected_mask_shape)}, "
            f"got {tuple(game_done.shape)}"
        )
    if game_truncated.shape != expected_mask_shape:
        raise ValueError(
            f"game_truncated must have shape {tuple(expected_mask_shape)}, "
            f"got {tuple(game_truncated.shape)}"
        )
    expected_boundary_shape = winner_probabilities[:, -1].shape
    if last_winner_probabilities.shape != expected_boundary_shape:
        raise ValueError(
            f"last_winner_probabilities must have shape "
            f"{tuple(expected_boundary_shape)}, "
            f"got {tuple(last_winner_probabilities.shape)}"
        )

    horizon = winner_probabilities.shape[1]
    targets = torch.zeros_like(winner_probabilities)
    carry = last_winner_probabilities
    done = game_done.unsqueeze(-1)
    truncated = game_truncated.unsqueeze(-1)
    for step in range(horizon - 1, -1, -1):
        next_dist = (
            last_winner_probabilities
            if step == horizon - 1
            else winner_probabilities[:, step + 1]
        )
        blended = (1.0 - gae_lambda) * next_dist + gae_lambda * carry
        # On a whole-game terminal the target is the realized winner
        # distribution; on a time-limit truncation it bootstraps from the
        # critic's win-probabilities at the cut state; both reset the recursion
        # so a later game's distribution never leaks across the boundary.
        carry = torch.where(
            truncated[:, step],
            bootstrap_winner_probabilities[:, step],
            torch.where(done[:, step], terminal_winner[:, step], blended),
        )
        targets[:, step] = carry
    return targets
