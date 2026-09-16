# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional

import torch
from torch.nn import Parameter

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme
from sglang.srt.utils import is_hip

__all__ = ["QuarkW4A16Int4"]


class QuarkW4A16Int4(QuarkLinearScheme):
    """Quark weight-only INT4 using hybrid ROCm skinny/Triton GEMMs."""

    _SKINNY_MAX_BATCH_SIZE = 5
    _SKINNY_LDS_ELEMENTS = 64 * 1024 // 2
    _SKINNY_GROUP_SIZES = {32, 64, 128}

    def __init__(
        self, weight_config: dict[str, Any], input_config: Optional[dict[str, Any]]
    ):
        self.group_size = weight_config.get("group_size")
        self.is_sym = bool(weight_config.get("symmetric", False))
        self.pack_factor = 8
        self.use_hybrid_rocm = (
            is_hip() and self.group_size in self._SKINNY_GROUP_SIZES
        )

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)

        if input_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"Input size {input_size_per_partition} is not divisible by "
                f"group size {self.group_size}. This can be caused by too "
                "large a tensor parallel size."
            )
        if output_size_per_partition % self.pack_factor != 0:
            raise ValueError(
                f"Output size {output_size_per_partition} is not divisible by "
                f"the INT4 pack factor {self.pack_factor}. This can be caused "
                "by too large a tensor parallel size."
            )

        layer.logical_widths = output_partition_sizes
        weight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        weight_zero_point = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.group_size,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                input_size_per_partition // self.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_zero_point", weight_zero_point)
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.use_hybrid_rocm:
            from sglang.kernels.ops.quantization.gptq_triton import (
                repack_awq_qzeros_to_skinny,
                repack_awq_w4_to_skinny,
            )

            # Quark stores signed INT4 in AWQ's N-packed/interleaved layout.
            # Convert once to unsigned zero-point-8 values in the shared
            # ExLlama K-packed layout used by both fused ROCm kernels.
            weight = repack_awq_w4_to_skinny(
                layer.weight.data.contiguous(), signed=True
            )
            if self.is_sym:
                weight_zero_point = torch.empty(
                    0, dtype=torch.int32, device=layer.weight_zero_point.device
                )
            else:
                weight_zero_point = repack_awq_qzeros_to_skinny(
                    layer.weight_zero_point.data.contiguous(), signed=True
                )
            weight_scale = layer.weight_scale.data.t().contiguous()
        else:
            # Preserve the branch's CUDA path: convert signed Quark nibbles to
            # standard unsigned AWQ values without changing their layout.
            sign_flip = torch.tensor(
                -2004318072,  # 0x88888888 as signed int32
                dtype=torch.int32,
                device=layer.weight.device,
            )
            weight = torch.bitwise_xor(layer.weight.data, sign_flip)
            weight_zero_point = torch.bitwise_xor(
                layer.weight_zero_point.data, sign_flip
            )
            weight_scale = layer.weight_scale.data

        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_zero_point = Parameter(
            weight_zero_point, requires_grad=False
        )
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1]).contiguous()

        if self.use_hybrid_rocm:
            from sglang.kernels.ops.quantization.gptq_triton import (
                gptq_w4a16_skinny_gemm,
            )

            m, k = reshaped_x.shape
            out_shape = x.shape[:-1] + (layer.weight.shape[0],)
            use_native_skinny = (
                m <= self._SKINNY_MAX_BATCH_SIZE
                and k * m <= self._SKINNY_LDS_ELEMENTS
                and hasattr(torch.ops.sgl_kernel, "wvSplitK_int4_g")
            )
            zero_points = None if self.is_sym else layer.weight_zero_point

            if use_native_skinny:
                cu_count = torch.cuda.get_device_properties(
                    reshaped_x.device
                ).multi_processor_count
                out = torch.ops.sgl_kernel.wvSplitK_int4_g.default(
                    layer.weight,
                    reshaped_x,
                    layer.weight_scale,
                    zero_points,
                    bias,
                    cu_count,
                    self.group_size,
                )
            else:
                out = gptq_w4a16_skinny_gemm(
                    input=reshaped_x,
                    qweight=layer.weight,
                    scales=layer.weight_scale,
                    group_size=self.group_size,
                    qzeros=zero_points,
                )
                if bias is not None:
                    out.add_(bias)
            return out.reshape(out_shape)

        from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import (
            awq_dequantize,
        )

        out_shape = x.shape[:-1] + (layer.weight.shape[-1] * self.pack_factor,)
        weight = awq_dequantize(
            layer.weight, layer.weight_scale, layer.weight_zero_point
        )
        out = torch.matmul(reshaped_x, weight)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
