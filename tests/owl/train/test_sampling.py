import pytest
import torch
from owl.train import sample_segments_by_advantage, sample_segments_uniform


def test_sample_segments_uniform_shapes() -> None:
    sample = sample_segments_uniform(n_segments=4, segments_per_minibatch=3)

    assert sample.indices.shape == (3,)
    assert sample.importance.shape == (3, 1)
    assert sample.probabilities.shape == (4,)
    assert torch.all(sample.indices >= 0)
    assert torch.all(sample.indices < 4)
    assert torch.allclose(sample.probabilities, torch.full((4,), 0.25))
    assert torch.allclose(sample.importance, torch.ones((3, 1)))


def test_sample_segments_by_advantage_uses_absolute_advantage_priority() -> None:
    torch.manual_seed(0)
    sample = sample_segments_by_advantage(
        torch.tensor([[1.0, -1.0], [0.0, 0.0], [3.0, 1.0]]),
        segments_per_minibatch=5,
        alpha=1.0,
        beta=0.5,
        eps=1e-6,
    )

    expected = torch.tensor([2.0, 0.0, 4.0])
    expected = (expected + 1e-6) / (expected.sum() + 3e-6)
    assert torch.allclose(sample.probabilities, expected)
    assert sample.importance.shape == (5, 1)
    assert sample.importance.max() == pytest.approx(1.0)
