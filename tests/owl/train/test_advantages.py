import pytest
import torch
from owl.train import compute_advantages


def test_compute_advantages_matches_gae_recurrence() -> None:
    advantages = compute_advantages(
        values=torch.tensor([[0.5, 0.25, 0.0]]),
        rewards=torch.tensor([[1.0, 1.0, 1.0]]),
        dones=torch.tensor([[False, False, True]]),
        gamma=0.9,
        gae_lambda=0.8,
    )

    assert torch.allclose(advantages, torch.tensor([[1.7834, 1.47, 1.0]]))


def test_compute_advantages_uses_bootstrap_value() -> None:
    advantages = compute_advantages(
        values=torch.tensor([[0.0, 0.0]]),
        rewards=torch.tensor([[0.0, 1.0]]),
        dones=torch.tensor([[False, False]]),
        bootstrap_values=torch.tensor([2.0]),
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, torch.tensor([[1.0, 2.0]]))


def test_compute_advantages_vtrace_clips_ratios() -> None:
    advantages = compute_advantages(
        values=torch.zeros((1, 2)),
        rewards=torch.ones((1, 2)),
        dones=torch.zeros((1, 2), dtype=torch.bool),
        ratios=torch.tensor([[2.0, 0.5]]),
        gamma=1.0,
        gae_lambda=1.0,
        mode="gae_vtrace",
        vtrace_rho_clip=1.0,
        vtrace_c_clip=0.25,
    )

    assert torch.allclose(advantages, torch.tensor([[1.125, 0.5]]))


def test_compute_advantages_resets_recursion_at_terminal_dones() -> None:
    rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 3.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[False, True, False], [True, False, False]])

    advantages = compute_advantages(
        values=values,
        rewards=rewards,
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.equal(advantages, torch.tensor([[2.0, 1.0, 1.0], [0.0, 5.0, 3.0]]))


def test_compute_advantages_bootstraps_only_nonterminal_last_steps() -> None:
    rewards = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[False, False], [False, True]])

    advantages = compute_advantages(
        values=values,
        rewards=rewards,
        dones=dones,
        bootstrap_values=torch.tensor([10.0, 10.0]),
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert torch.equal(advantages, torch.tensor([[3.0, 6.0], [0.5, 1.0]]))


def test_compute_advantages_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="rewards must match values shape"):
        compute_advantages(
            values=torch.zeros((2, 3)),
            rewards=torch.zeros((3, 2)),
            dones=torch.zeros((2, 3), dtype=torch.bool),
            gamma=0.99,
            gae_lambda=0.95,
        )
