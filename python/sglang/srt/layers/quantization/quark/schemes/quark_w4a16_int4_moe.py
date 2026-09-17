# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Parameter

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = ["QuarkW4A16Int4MoE"]

# AWQ maps logical lanes to physical nibbles as [0, 4, 1, 5, 2, 6, 3, 7].
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

# Flips the sign bit of both nibbles in a byte, converting a pair of signed
# int4 codes to the unsigned zero-point-8 convention.
_SIGN_FLIP = 0x88


def _awq_to_kpacked(tensor: torch.Tensor, is_weight: bool) -> torch.Tensor:
    """Move Quark's AWQ-style N-packing to the K-packing the Triton kernel reads.

    weight:     (K, N // 8) int32     -> (N, K // 2) uint8
    zero point: (K // G, N // 8) int32 -> (N // 2, K // G) uint8

    The kernel selects a nibble with ``(k % 2) * 4``, so even K lands in the
    low nibble and odd K in the high nibble.
    """
    size0 = tensor.size(0)
    tensor = tensor.contiguous().view(torch.uint8)

    shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
    tensor = (tensor[:, :, None] >> shifter) & 0xF
    tensor = tensor.view(-1, 8)[:, _REVERSE_AWQ_PACK_ORDER]
    tensor = tensor.view(size0, -1).T.contiguous()

    if is_weight:
        return tensor[:, 1::2] * 16 + tensor[:, ::2]
    return tensor[1::2, :] * 16 + tensor[::2, :]


class QuarkW4A16Int4MoE(QuarkMoEScheme):
    """Quark weight-only INT4 fused MoE on the Triton W4A16 kernel."""

    def __init__(
        self, weight_config: dict[str, Any], input_config: Optional[dict[str, Any]]
    ):
        self.group_size = weight_config.get("group_size")
        self.is_sym = bool(weight_config.get("symmetric", False))
        # Quark stores signed nibbles for int4 and unsigned ones for uint4; the
        # Triton kernel only consumes the unsigned zero-point-8 convention.
        self.is_signed = weight_config.get("dtype") == "int4"
        self.pack_factor = 2

        if not self.is_sym:
            raise NotImplementedError(
                "Quark INT4 W4A16 MoE currently requires symmetric weights; "
                "asymmetric checkpoints need per-group zero points threaded "
                "through the fused MoE kernel."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        group_size = self.group_size
        for name, size in (
            ("hidden_size", hidden_size),
            ("intermediate_size_per_partition", intermediate_size_per_partition),
        ):
            if size % group_size != 0:
                raise ValueError(
                    f"Quark INT4 W4A16 MoE requires {name} ({size}) to be "
                    f"divisible by the group size ({group_size})."
                )

        extra_weight_attrs.update(
            {
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
                "is_transposed": False,
            }
        )
        extra_weight_attrs["weight_loader"] = self._wrap_weight_loader(
            layer, extra_weight_attrs["weight_loader"]
        )

        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Symmetric checkpoints still ship constant zero points. Register them
        # so the expert key remaps onto a real parameter instead of raising a
        # KeyError, then drop them once loading is done.
        w13_weight_zero_point = Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition // self.pack_factor,
                hidden_size // group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_zero_point", w13_weight_zero_point)
        set_weight_attrs(w13_weight_zero_point, extra_weight_attrs)

        w2_weight_zero_point = Parameter(
            torch.zeros(
                num_experts,
                hidden_size // self.pack_factor,
                intermediate_size_per_partition // group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_zero_point", w2_weight_zero_point)
        set_weight_attrs(w2_weight_zero_point, extra_weight_attrs)

    def _wrap_weight_loader(self, layer: torch.nn.Module, weight_loader):
        """Convert each expert tensor into the kernel layout before it is stored.

        The fused MoE loader shards and copies but never repacks, so the
        AWQ -> K-packed conversion has to happen on the loaded tensor.
        """

        def quark_w4a16_moe_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            if "weight_zero_point" in weight_name:
                return

            if "weight_scale" in weight_name:
                loaded_weight = loaded_weight.t().contiguous()
            else:
                loaded_weight = _awq_to_kpacked(loaded_weight, is_weight=True)
                if self.is_signed:
                    loaded_weight ^= _SIGN_FLIP

            weight_loader(param, loaded_weight, weight_name, shard_id, expert_id)

        return quark_w4a16_moe_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        empty = torch.empty(0, dtype=torch.uint8, device=layer.w13_weight.device)
        layer.w13_weight_zero_point = Parameter(empty, requires_grad=False)
        layer.w2_weight_zero_point = Parameter(empty, requires_grad=False)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int4_w4a16=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            block_shape=[0, self.group_size],
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        assert self.moe_runner_config.activation == "silu", (
            "Only SiLU activation is supported."
        )

        return self.runner.run(dispatch_output, self.get_triton_quant_info(layer))
