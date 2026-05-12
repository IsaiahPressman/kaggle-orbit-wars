import torch
from owl.train import sample_segments_uniform_single_pass


def test_sample_segments_uniform_single_pass_shuffles_without_replacement() -> None:
    torch.manual_seed(0)
    samples = sample_segments_uniform_single_pass(
        n_segments=5,
        segments_per_minibatch=2,
    )

    assert [sample.indices.shape for sample in samples] == [(2,), (2,), (1,)]
    indices = torch.cat([sample.indices for sample in samples])
    assert indices.unique().numel() == 5
    assert torch.equal(indices.sort().values, torch.arange(5))
    for sample in samples:
        assert torch.allclose(
            sample.importance,
            torch.ones((sample.indices.shape[0], 1)),
        )
