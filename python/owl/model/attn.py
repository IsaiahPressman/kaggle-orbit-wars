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


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    if q.device.type == "cuda":
        if _FLASH_ATTN_VARLEN_FUNC is None:
            raise RuntimeError("flash-attn is required for CUDA varlen attention")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError("flash-attn requires float16 or bfloat16 CUDA tensors")
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
