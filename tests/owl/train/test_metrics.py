import pytest
import torch
from owl.train import (
    SegmentSample,
    assert_finite,
    explained_variance,
    segment_sampling_metrics,
)


def test_segment_sampling_metrics_reports_duplicates_and_priority() -> None:
    sample = SegmentSample(
        indices=torch.tensor([0, 0, 1, 2]),
        importance=torch.tensor([[1.0], [0.5], [0.5], [0.25]]),
        probabilities=torch.full((3,), 1 / 3),
    )

    metrics = segment_sampling_metrics(
        torch.tensor([[1.0, -2.0], [0.5, 0.5], [4.0, 0.0]]),
        sample,
    )

    assert metrics.priority_min == pytest.approx(1.0)
    assert metrics.priority_mean == pytest.approx(8.0 / 3.0)
    assert metrics.priority_max == pytest.approx(4.0)
    assert metrics.duplicate_fraction == pytest.approx(0.25)
    assert metrics.importance_mean == pytest.approx(0.5625)
    assert metrics.importance_max == pytest.approx(1.0)


def test_explained_variance_returns_zero_for_constant_target() -> None:
    actual = explained_variance(
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 3.0]),
        valid_mask=torch.tensor([True, True]),
    )

    assert actual == pytest.approx(0.0)


def test_assert_finite_rejects_nan() -> None:
    with pytest.raises(ValueError, match="loss must contain only finite values"):
        assert_finite(torch.tensor([1.0, torch.nan]), "loss")
