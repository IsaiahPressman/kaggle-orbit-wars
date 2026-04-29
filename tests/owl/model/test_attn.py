import pytest
import torch
import torch.nn.functional as F
from owl.model.attn import (
    _should_use_flash_attn,
    flash_attn_available,
    varlen_attention,
)


def test_varlen_attention_cpu_matches_torch_sdpa_per_sequence() -> None:
    cu_seqlens = torch.tensor([0, 1, 4, 6], dtype=torch.int32)
    q = torch.zeros((6, 2, 2), dtype=torch.float32)
    k = torch.zeros((6, 2, 2), dtype=torch.float32)
    v = torch.tensor(
        [
            [[1.0, 3.0], [2.0, 4.0]],
            [[4.0, 6.0], [8.0, 10.0]],
            [[10.0, 20.0], [30.0, 40.0]],
            [[16.0, 22.0], [36.0, 44.0]],
            [[100.0, 200.0], [300.0, 400.0]],
            [[300.0, 400.0], [500.0, 600.0]],
        ],
        dtype=torch.float32,
    )

    actual = varlen_attention(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=3)
    expected = torch.stack(
        (
            v[0],
            v[1:4].mean(dim=0),
            v[1:4].mean(dim=0),
            v[1:4].mean(dim=0),
            v[4:6].mean(dim=0),
            v[4:6].mean(dim=0),
        )
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    ("device_type", "dtype", "has_flash_attn", "expected"),
    [
        ("cuda", torch.float16, True, True),
        ("cuda", torch.bfloat16, True, True),
        ("cuda", torch.float32, True, False),
        ("cuda", torch.float16, False, False),
        ("cpu", torch.float16, True, False),
    ],
)
def test_flash_attention_backend_selection(
    device_type: str,
    dtype: torch.dtype,
    has_flash_attn: bool,
    expected: bool,
) -> None:
    assert (
        _should_use_flash_attn(
            device_type=device_type,
            dtype=dtype,
            has_flash_attn=has_flash_attn,
        )
        is expected
    )


def _sdpa_varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    for start, end in zip(
        cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True
    ):
        outputs.append(
            F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
    return torch.cat(outputs, dim=0)


@pytest.mark.skipif(
    not flash_attn_available() or not torch.cuda.is_available(),
    reason="flash-attn CUDA backend is not available",
)
def test_varlen_attention_flash_backend() -> None:
    torch.manual_seed(0)
    cu_seqlens = torch.tensor([0, 1, 4, 6], dtype=torch.int32, device="cuda")
    q = torch.randn((6, 2, 4), dtype=torch.float16, device="cuda")
    k = torch.randn((6, 2, 4), dtype=torch.float16, device="cuda")
    v = torch.randn((6, 2, 4), dtype=torch.float16, device="cuda")

    actual = varlen_attention(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=3)
    expected = _sdpa_varlen_reference(q, k, v, cu_seqlens)

    assert actual.shape == q.shape
    assert actual.dtype == q.dtype
    assert actual.device.type == "cuda"
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)
