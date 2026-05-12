import owl.train.advantages as advantages_module
import pytest
import torch
from owl.train import compute_gae


def test_compute_gae_matches_recurrence() -> None:
    advantages, returns = compute_gae(
        values=torch.tensor([[0.5, 0.25, 0.0]]),
        rewards=torch.tensor([[1.0, 1.0, 1.0]]),
        dones=torch.tensor([[False, False, True]]),
        last_values=torch.tensor([10.0]),
        gamma=0.9,
        gae_lambda=0.8,
    )

    assert torch.allclose(advantages, torch.tensor([[1.7834, 1.47, 1.0]]))
    assert torch.allclose(returns, advantages + torch.tensor([[0.5, 0.25, 0.0]]))


def test_compute_gae_uses_bootstrap_value() -> None:
    advantages, _returns = compute_gae(
        values=torch.tensor([[0.0, 0.0]]),
        rewards=torch.tensor([[0.0, 1.0]]),
        dones=torch.tensor([[False, False]]),
        last_values=torch.tensor([2.0]),
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, torch.tensor([[1.0, 2.0]]))


def test_compute_gae_resets_recursion_at_terminal_dones() -> None:
    rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 3.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[False, True, False], [True, False, False]])

    advantages, _returns = compute_gae(
        values=values,
        rewards=rewards,
        dones=dones,
        last_values=torch.zeros((2,)),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.equal(advantages, torch.tensor([[2.0, 1.0, 1.0], [0.0, 5.0, 3.0]]))


def test_compute_gae_bootstraps_only_nonterminal_last_steps() -> None:
    rewards = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[False, False], [False, True]])

    advantages, _returns = compute_gae(
        values=values,
        rewards=rewards,
        dones=dones,
        last_values=torch.tensor([10.0, 10.0]),
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert torch.equal(advantages, torch.tensor([[3.0, 6.0], [0.5, 1.0]]))


def test_compute_gae_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="rewards must match values shape"):
        compute_gae(
            values=torch.zeros((2, 3)),
            rewards=torch.zeros((3, 2)),
            dones=torch.zeros((2, 3), dtype=torch.bool),
            last_values=torch.zeros((2,)),
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_compiled_compute_gae_validates_before_tensor_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_compile(target: object, *, mode: str) -> object:
        assert mode == "default"
        calls.append(target.__name__)
        return target

    monkeypatch.setattr(advantages_module.torch, "compile", fake_compile)
    compute_gae = advantages_module.compile_compute_gae("default")

    with pytest.raises(ValueError, match="rewards must match values shape"):
        compute_gae(
            rewards=torch.zeros((3, 2)),
            values=torch.zeros((2, 3)),
            dones=torch.zeros((2, 3), dtype=torch.bool),
            last_values=torch.zeros((2,)),
            gamma=0.99,
            gae_lambda=0.95,
        )

    assert calls == ["_compute_gae_tensors"]


def test_compiled_compute_gae_matches_uncompiled_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_compile(target: object, *, mode: str) -> object:
        assert mode == "default"
        calls.append(target.__name__)
        return target

    monkeypatch.setattr(advantages_module.torch, "compile", fake_compile)
    compute_gae = advantages_module.compile_compute_gae("default")

    assert calls == ["_compute_gae_tensors"]

    advantages, returns = compute_gae(
        rewards=torch.tensor([[1.0, 2.0, 3.0]]),
        values=torch.zeros((1, 3)),
        dones=torch.zeros((1, 3), dtype=torch.bool),
        last_values=torch.tensor([10.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert calls == ["_compute_gae_tensors"]
    assert torch.equal(advantages, torch.tensor([[16.0, 15.0, 13.0]]))
    assert torch.equal(returns, torch.tensor([[16.0, 15.0, 13.0]]))
