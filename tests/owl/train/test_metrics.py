import pytest
import torch
from owl.train.metrics import explained_variance
from owl.train.utils import assert_finite


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
