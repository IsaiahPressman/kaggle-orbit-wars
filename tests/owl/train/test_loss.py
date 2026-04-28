import pytest
import torch
from owl.train import PPOConfig, ppo_loss, validate_ppo_loss_inputs


def test_ppo_loss_matches_clipped_objectives() -> None:
    new_logp = torch.log(torch.tensor([[1.3, 0.7]]))
    old_logp = torch.zeros((1, 2))
    old_values = torch.tensor([[0.0, 1.0]])
    new_values = torch.tensor([[0.5, 0.0]])
    returns = torch.tensor([[1.0, -1.0]])
    advantages = torch.tensor([[1.0, -1.0]])
    entropy = torch.tensor([[0.2, 0.4]])

    metrics = ppo_loss(
        new_logp=new_logp,
        entropy=entropy,
        new_values=new_values,
        old_logp=old_logp,
        old_values=old_values,
        returns=returns,
        advantages=advantages,
        loss_weight=torch.ones_like(advantages),
        config=PPOConfig(
            clip_coef=0.2,
            vf_clip_coef=0.25,
            vf_coef=0.5,
            ent_coef=0.1,
            normalize_advantages=False,
        ),
    )

    ratio = torch.tensor([[1.3, 0.7]])
    expected_policy = torch.max(
        -advantages * ratio,
        -advantages * torch.clamp(ratio, 0.8, 1.2),
    ).mean()
    value_clipped = old_values + torch.clamp(new_values - old_values, -0.25, 0.25)
    expected_value = (
        0.5
        * torch.max(
            (new_values - returns).pow(2),
            (value_clipped - returns).pow(2),
        ).mean()
    )
    expected_entropy = entropy.mean()
    expected_loss = expected_policy + 0.5 * expected_value - 0.1 * expected_entropy

    assert torch.allclose(metrics.policy_loss, expected_policy)
    assert torch.allclose(metrics.value_loss, expected_value)
    assert torch.allclose(metrics.entropy, expected_entropy)
    assert torch.allclose(metrics.loss, expected_loss)
    assert torch.allclose(metrics.clipfrac, torch.tensor(1.0))


def test_validate_ppo_loss_inputs_checks_cold_path_invariants() -> None:
    with pytest.raises(ValueError, match="loss_weight must match new_logp shape"):
        validate_ppo_loss_inputs(
            new_logp=torch.zeros((2, 3)),
            entropy=torch.zeros((2, 3)),
            new_values=torch.zeros((2, 3)),
            old_logp=torch.zeros((2, 3)),
            old_values=torch.zeros((2, 3)),
            returns=torch.zeros((2, 3)),
            advantages=torch.zeros((2, 3)),
            loss_weight=torch.zeros((3, 2)),
        )
