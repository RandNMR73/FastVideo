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
    # Convert back to PyTorch tensors for buffer registration
    return w.to_torch(), w_scale.to_torch()

class MXFP4QuantizeMethod(QuantizeMethodBase):
    def __init__(self):
        super().__init__()
        self.weight_mxfp4 = None
        self.weight_scale = None

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

    # @torch.compile
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Apply MXFP4 quantized computation."""
        out_dim = layer.weight.shape[0]
        # Need contiguous tensors for collectives.
        assert x.dtype == torch.bfloat16, f"only allow bf16 inputs to mxfp4 linear, got {x.dtype}"
        
        weight_mxfp4 = layer._mxfp4_weight
        weight_scale = layer._mxfp4_weight_scale
        
        # Convert PyTorch tensors back to Triton tensors for matmul_ogs
        weight_mxfp4_triton = wrap_torch_tensor(weight_mxfp4, dtype=FP4)
        weight_scale_triton = wrap_torch_tensor(weight_scale)
        
        original_shape = x.shape
        x_reshaped = x.view(-1, x.shape[-1])
        
        pc = PrecisionConfig(weight_scale=weight_scale_triton, flex_ctx=FlexCtx(rhs_data=InFlexData()))
        out = matmul_ogs(x_reshaped, weight_mxfp4_triton, bias, precision_config=pc)
            
        if bias is not None:
            if bias.device != out.device or bias.dtype != out.dtype:
                bias = bias.to(device=out.device, dtype=out.dtype)
            out = out + bias
        
        if len(original_shape) == 3:
            out = out.view(original_shape[0], original_shape[1], out_dim)
        
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
        mxfp4_layers = ["ffn.fc_in", "ffn.fc_out", "to_q", "to_k", "to_v", "to_out"]
        if isinstance(layer, LinearBase) and any(layer_name in prefix for layer_name in mxfp4_layers):
            return MXFP4QuantizeMethod()
        return None

# @torch.compile
def convert_model_to_mxfp4(model: torch.nn.Module):
    from torch.distributed.tensor import DTensor  # type: ignore
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, MXFP4QuantizeMethod):
            weight = getattr(mod, "weight", None)
            if weight is None:
                continue
            if isinstance(weight, DTensor):  # type: ignore
                weight_local = weight.to_local()
            else:
                weight_local = weight
            mxfp4_w, mxfp4_s = quantize_mx4(weight_local)
            mod.register_buffer("_mxfp4_weight", mxfp4_w, persistent=False)
            mod.register_buffer("_mxfp4_weight_scale", mxfp4_s, persistent=False)
