# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import Optional

import torch

from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_enabled = get_bool_env_var("SGLANG_ROCM_USE_WVSPLITK", "True")

# Benchmarked on gfx1151 with amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16.
# Keep this deliberately narrow: wvSplitK has severe regressions on some
# unlisted RDNA shapes.
_QWEN3_5_BF16_WVSPLITK_MAX_ROWS = {
    (14336, 5120): 8,  # MTP full-attention fused Q/K/V/gate projection
    (5120, 6144): 8,  # MTP full-attention output projection
    (34816, 5120): 5,  # MTP gate/up projection
    (5120, 17408): 5,  # MTP down projection
    (5120, 10240): 8,  # MTP input fusion projection
    (248320, 5120): 5,  # target and MTP language-model heads
}


@functools.cache
def _gfx1151_cu_count() -> Optional[int]:
    if not (_is_hip and _enabled and torch.cuda.is_available()):
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if not getattr(props, "gcnArchName", "").split(":", 1)[0].startswith(
        "gfx1151"
    ):
        return None

    # Importing sgl_kernel loads the torch.library registrations.
    try:
        import sgl_kernel  # noqa: F401
    except ImportError:
        return None
    if not hasattr(torch.ops.sgl_kernel, "wvSplitK"):
        return None
    return props.multi_processor_count


def should_use_rocm_wv_split_k(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> bool:
    rows = x.numel() // x.shape[-1]
    max_rows = _QWEN3_5_BF16_WVSPLITK_MAX_ROWS.get(tuple(weight.shape), 0)
    return (
        _gfx1151_cu_count() is not None
        and 1 <= rows <= max_rows
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.dtype == weight.dtype
        and weight.ndim == 2
        and weight.shape[1] == x.shape[-1]
        and weight.shape[1] % 8 == 0
        and weight.is_contiguous()
        and (bias is None or (bias.dtype == x.dtype and bias.is_contiguous()))
    )


def rocm_wv_split_k(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cu_count = _gfx1151_cu_count()
    if cu_count is None:
        raise RuntimeError("ROCm wvSplitK is unavailable on the current device")

    input_2d = x.reshape(-1, x.shape[-1]).contiguous()
    output = torch.ops.sgl_kernel.wvSplitK.default(
        weight,
        input_2d,
        bias,
        cu_count,
    )
    return output.reshape(*x.shape[:-1], weight.shape[0])


__all__ = ["rocm_wv_split_k", "should_use_rocm_wv_split_k"]
