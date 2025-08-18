# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/openai/gpt-oss/blob/main/gpt_oss/triton/moe.py
from fastvideo.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
import torch
from torch.nn.parameter import Parameter

from typing import Any, Tuple
from fastvideo.models.utils import set_weight_attrs
import time 

import triton_kernels
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.matmul_ogs import PrecisionConfig, FlexCtx
from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.numerics import InFlexData
from triton_kernels.tensor import convert_layout
from triton_kernels.tensor_details.layout import StridedLayout, HopperMXValueLayout
from triton_kernels.tensor import wrap_torch_tensor, FP4

def quantize_mx4(w):
    w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), HopperMXValueLayout, mx_axis=1)
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    return w, w_scale

class MXFP4QuantizeMethod(QuantizeMethodBase):
    def __init__(self):
        super().__init__()

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        """Create weights for a linear layer. Note the corrected signature to match LinearMethodBase."""
        weight = Parameter(torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype,
        ),
        requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        w, w_mx = quantize_mx4(layer.weight)
        pc = PrecisionConfig(weight_scale=w_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
        out = matmul_ogs(x, w, bias, precision_config=pc)
        return out
        

class MXFP4Config(QuantizationConfig):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "mxfp4"
    
    def get_supported_act_dtypes(self):
        #TODO: Confirm this
        return [torch.bfloat16]
    
    @classmethod
    def get_min_capability(cls):
        return 90
    
    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MXFP4Config":
        return cls()
    
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # Apply MXFP4 quantization to all linear layers
        from fastvideo.layers.linear import LinearBase
        if isinstance(layer, LinearBase):
            return MXFP4QuantizeMethod()
        return None
