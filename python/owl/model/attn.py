from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _load_flash_attn_varlen() -> Any | None:
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        return None

    return flash_attn_varlen_func


_FLASH_ATTN_VARLEN_FUNC = _load_flash_attn_varlen()


def flash_attn_available() -> bool:
    return _FLASH_ATTN_VARLEN_FUNC is not None


def _should_use_flash_attn(
    *,
    device_type: str,
    dtype: torch.dtype,
    has_flash_attn: bool,
) -> bool:
    return (
        device_type == "cuda"
        and dtype in (torch.float16, torch.bfloat16)
        and has_flash_attn
    )


def use_flash_attn(q: torch.Tensor) -> bool:
    return _should_use_flash_attn(
        device_type=q.device.type,
        dtype=q.dtype,
        has_flash_attn=_FLASH_ATTN_VARLEN_FUNC is not None,
    )


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    if use_flash_attn(q):
        if _FLASH_ATTN_VARLEN_FUNC is None:
            raise RuntimeError("internal error: flash-attn backend is unavailable")
        return _FLASH_ATTN_VARLEN_FUNC(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=False,
        )

    return _varlen_attention_sdpa(q, k, v, cu_seqlens=cu_seqlens)


def _varlen_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    for start, end in zip(
        cu_seqlens[:-1].tolist(),
        cu_seqlens[1:].tolist(),
        strict=True,
    ):
        attn = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1).unsqueeze(0),
            k[start:end].transpose(0, 1).unsqueeze(0),
            v[start:end].transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
        )
        outputs.append(attn.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)
