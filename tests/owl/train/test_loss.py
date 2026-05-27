from math import prod

import owl.train.ppo as ppo
import pytest
import torch
from owl.train import PPOConfig
from owl.train.ppo import _ppo_loss


def test_ppo_loss_matches_clipped_objectives() -> None:
    new_logp = torch.log(torch.tensor([[1.3, 0.7]]))
    old_logp = torch.zeros((1, 2))
    old_values = torch.tensor([[0.0, 1.0]])
    new_values = torch.tensor([[0.5, 0.0]])
    returns = torch.tensor([[1.0, -1.0]])
    advantages = torch.tensor([[1.0, -1.0]])
    entropy = torch.tensor([[0.2, 0.4]])

    metrics, backward_loss = _ppo_loss(
        new_logp=new_logp,
        entropy=entropy,
        new_values=new_values,
        old_logp=old_logp,
        old_values=old_values,
        returns=returns,
        advantages=advantages,
        policy_weight=torch.ones_like(advantages),
        value_weight=torch.ones_like(advantages),
        config=PPOConfig(
            clip_coef=0.2,
            vf_clip_coef=0.25,
            vf_coef=0.5,
            ent_coef=0.1,
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
    expected_entropy_loss = -0.1 * expected_entropy
    expected_loss = expected_policy + 0.5 * expected_value - 0.1 * expected_entropy

    assert torch.allclose(metrics.policy_loss, expected_policy)
    assert torch.allclose(metrics.value_loss, expected_value)
    assert torch.allclose(metrics.entropy_loss, expected_entropy_loss)
    assert torch.allclose(metrics.entropy, expected_entropy)
    assert torch.allclose(metrics.loss, expected_loss)
    assert backward_loss is metrics.loss
    assert torch.allclose(metrics.clipfrac, torch.tensor(1.0))


def test_ppo_loss_can_clip_per_entity_before_summing() -> None:
    new_logp = torch.log(torch.tensor([[[1.3, 1.01]]]))
    old_logp = torch.zeros((1, 1, 2))
    old_values = torch.zeros((1, 1))
    new_values = torch.zeros((1, 1))
    returns = torch.zeros((1, 1))
    advantages = torch.tensor([[2.0]])
    entropy = torch.tensor([[[0.2, 0.4]]])
    entity_policy_weight = torch.ones_like(new_logp)

    metrics, _backward_loss = _ppo_loss(
        new_logp=new_logp,
        entropy=entropy,
        new_values=new_values,
        old_logp=old_logp,
        old_values=old_values,
        returns=returns,
        advantages=advantages,
        policy_weight=torch.ones_like(advantages),
        value_weight=torch.ones_like(advantages),
        config=PPOConfig(
            clip_coef=0.2,
            vf_coef=0.0,
            ent_coef=0.0,
            ppo_clip_mode="per_entity",
        ),
        entity_policy_weight=entity_policy_weight,
    )

    expected_policy = torch.tensor(-2.0 * 1.2 - 2.0 * 1.01)
    expected_kl = ((new_logp.exp() - 1.0) - new_logp).sum()
    assert torch.allclose(metrics.policy_loss, expected_policy)
    assert torch.allclose(metrics.entropy, torch.tensor(0.6))
    assert torch.allclose(metrics.approx_kl, expected_kl)
    assert torch.allclose(metrics.clipfrac, torch.tensor(0.5))


def test_ppo_loss_uses_policy_and_value_weights_separately() -> None:
    new_logp = torch.log(torch.tensor([[1.1, 10.0]]))
    old_logp = torch.zeros((1, 2))
    old_values = torch.tensor([[0.0, 0.0]])
    new_values = torch.tensor([[0.0, 0.0]])
    returns = torch.tensor([[2.0, 4.0]])
    advantages = torch.tensor([[3.0, 100.0]])
    entropy = torch.tensor([[0.25, 99.0]])
    policy_weight = torch.tensor([[1.0, 0.0]])
    value_weight = torch.ones_like(policy_weight)

    metrics, _backward_loss = _ppo_loss(
        new_logp=new_logp,
        entropy=entropy,
        new_values=new_values,
        old_logp=old_logp,
        old_values=old_values,
        returns=returns,
        advantages=advantages,
        policy_weight=policy_weight,
        value_weight=value_weight,
        config=PPOConfig(
            clip_coef=0.2,
            vf_clip_coef=10.0,
            vf_coef=0.5,
            ent_coef=0.1,
        ),
    )

    assert torch.allclose(metrics.policy_loss, torch.tensor(-3.3))
    assert torch.allclose(metrics.value_loss, torch.tensor(5.0))
    assert torch.allclose(metrics.entropy_loss, torch.tensor(-0.025))
    assert torch.allclose(metrics.entropy, torch.tensor(0.25))
    assert torch.allclose(metrics.ratio_mean, torch.tensor(1.1))
    assert torch.allclose(metrics.ratio_max, torch.tensor(1.1))
    assert torch.allclose(metrics.clipfrac, torch.tensor(0.0))
    assert torch.allclose(
        metrics.approx_kl,
        (torch.tensor(1.1) - 1.0) - torch.log(torch.tensor(1.1)),
    )


def test_ppo_loss_adds_teacher_terms_with_separate_weights() -> None:
    shape = (1, 2)
    policy_weight = torch.tensor([[1.0, 0.0]])
    value_weight = torch.ones(shape)
    teacher_kl = torch.tensor([[0.25, 99.0]])
    teacher_value_loss_values = torch.tensor([[0.2, 0.6]])

    metrics, _backward_loss = _ppo_loss(
        new_logp=torch.zeros(shape),
        entropy=torch.zeros(shape),
        new_values=torch.zeros(shape),
        old_logp=torch.zeros(shape),
        old_values=torch.zeros(shape),
        returns=torch.zeros(shape),
        advantages=torch.zeros(shape),
        policy_weight=policy_weight,
        value_weight=value_weight,
        teacher_kl=teacher_kl,
        teacher_value_loss_values=teacher_value_loss_values,
        config=PPOConfig(
            vf_coef=0.0,
            ent_coef=0.0,
            teacher_kl_coef=0.5,
            teacher_value_coef=0.25,
        ),
    )

    assert torch.allclose(metrics.teacher_kl, torch.tensor(0.25))
    assert torch.allclose(metrics.teacher_kl_loss, torch.tensor(0.125))
    assert torch.allclose(
        metrics.teacher_value_cross_entropy,
        torch.tensor(0.4),
    )
    assert torch.allclose(metrics.teacher_value_loss, torch.tensor(0.1))
    assert torch.allclose(metrics.loss, torch.tensor(0.225))


def test_ppo_loss_uses_raw_advantages() -> None:
    new_logp = torch.log(torch.tensor([[1.1, 0.9, 9.0]]))
    old_logp = torch.zeros((1, 3))
    values = torch.zeros((1, 3))
    returns = torch.zeros((1, 3))
    advantages = torch.tensor([[1.0, 3.0, 100.0]])
    entropy = torch.zeros((1, 3))
    policy_weight = torch.tensor([[1.0, 1.0, 0.0]])

    metrics, _backward_loss = _ppo_loss(
        new_logp=new_logp,
        entropy=entropy,
        new_values=values,
        old_logp=old_logp,
        old_values=values,
        returns=returns,
        advantages=advantages,
        policy_weight=policy_weight,
        value_weight=torch.ones_like(policy_weight),
        config=PPOConfig(
            clip_coef=10.0,
            vf_clip_coef=10.0,
            vf_coef=0.0,
            ent_coef=0.0,
        ),
    )

    expected_policy = (-(advantages[:, :2]) * torch.tensor([[1.1, 0.9]])).mean()
    assert torch.allclose(metrics.policy_loss, expected_policy)


def test_ppo_loss_handles_all_policy_invalid_minibatch() -> None:
    shape = (1, 2)
    policy_weight = torch.zeros(shape)
    metrics, _backward_loss = _ppo_loss(
        new_logp=torch.zeros(shape),
        entropy=torch.ones(shape),
        new_values=torch.zeros(shape),
        old_logp=torch.zeros(shape),
        old_values=torch.zeros(shape),
        returns=torch.ones(shape),
        advantages=torch.ones(shape),
        policy_weight=policy_weight,
        value_weight=torch.ones(shape),
        config=PPOConfig(),
    )

    for tensor in (
        metrics.loss,
        metrics.policy_loss,
        metrics.value_loss,
        metrics.entropy_loss,
        metrics.entropy,
        metrics.approx_kl,
        metrics.clipfrac,
        metrics.ratio_mean,
        metrics.ratio_max,
    ):
        assert torch.isfinite(tensor)
    assert torch.allclose(metrics.policy_loss, torch.tensor(0.0))
    assert torch.allclose(metrics.value_loss, torch.tensor(0.5))
    assert torch.allclose(metrics.ratio_max, torch.tensor(0.0))


def test_distributed_ppo_loss_only_reduces_scalar_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=2,
        initialized=True,
    )
    reduced_shapes: list[tuple[int, ...]] = []

    def fake_all_reduce_sum(
        tensor: torch.Tensor,
        _context: ppo.DistributedContext,
    ) -> torch.Tensor:
        assert _context is context
        reduced_shapes.append(tuple(tensor.shape))
        return tensor.clone()

    def fake_all_reduce_max(
        tensor: torch.Tensor,
        _context: ppo.DistributedContext,
    ) -> torch.Tensor:
        assert _context is context
        reduced_shapes.append(tuple(tensor.shape))
        return tensor.clone()

    monkeypatch.setattr(ppo, "all_reduce_sum", fake_all_reduce_sum)
    monkeypatch.setattr(ppo, "all_reduce_max", fake_all_reduce_max)
    shape = (2, 3)
    new_logp = torch.zeros(shape, requires_grad=True)

    metrics, backward_loss = _ppo_loss(
        new_logp=new_logp,
        entropy=torch.ones(shape),
        new_values=torch.zeros(shape, requires_grad=True),
        old_logp=torch.zeros(shape),
        old_values=torch.zeros(shape),
        returns=torch.ones(shape),
        advantages=torch.ones(shape),
        policy_weight=torch.ones(shape),
        value_weight=torch.ones(shape),
        config=PPOConfig(),
        context=context,
    )

    assert backward_loss is not metrics.loss
    backward_loss.backward()
    assert reduced_shapes
    assert all(prod(shape) <= 2 for shape in reduced_shapes)
