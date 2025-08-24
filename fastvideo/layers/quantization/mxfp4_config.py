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
    """Quantize weights to MXFP4 format."""
    w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=0)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), HopperMXValueLayout, mx_axis=0)
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    return w, w_scale

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

        # Override bias creation to ensure bfloat16 dtype for MXFP4 compatibility
        if hasattr(layer, 'bias') and layer.bias is not None:
            # Replace the existing bias with bfloat16 version
            bias_data = layer.bias.data.clone()
            layer.bias = Parameter(bias_data.to(torch.bfloat16), requires_grad=False)
            set_weight_attrs(layer.bias, {
                "output_dim": 0,
                "weight_loader": getattr(layer, 'weight_loader', None),
            })


    # @torch.compile
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Apply MXFP4 quantized computation."""
        out_dim = layer.weight.shape[0]
        # Need contiguous tensors for collectives.
        assert x.dtype == torch.bfloat16, f"only allow bf16 inputs to mxfp4 linear, got {x.dtype}"
        
        print(f"Layer weight shape: {layer.weight.shape}")
        weight, scale = quantize_mx4(layer.weight) 
        print(f"Quantized weight shape: {weight.shape}")
        print(f"Scale shape: {scale.shape}")
        
        original_shape = x.shape
        x = x.view(-1, x.shape[-1])
        print(f"Input shape after view: {x.shape}")
        
        # Don't pass bias to the kernel - handle it manually afterward to avoid type conflicts
        pc = PrecisionConfig(weight_scale=scale, flex_ctx=FlexCtx(rhs_data=InFlexData()))
        out = matmul_ogs(x, weight, None, precision_config=pc)
        
        # Add bias manually if it exists
        if bias is not None:
            if bias.dtype != out.dtype:
                bias = bias.to(out.dtype)
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
        mxfp4_layers = ["fc_in", "fc_out", "to_q", "to_k", "to_v", "to_out"]
        
        # Debug logging to see what prefixes we're getting
        if isinstance(layer, LinearBase):
            print(f"Checking layer with prefix: '{prefix}' for MXFP4 quantization")
            if any(layer_name in prefix for layer_name in mxfp4_layers):
                print(f"  -> Applying MXFP4 quantization to '{prefix}'")
                return MXFP4QuantizeMethod()
            else:
                print(f"  -> No match found for '{prefix}'")
                # Fallback: quantize all linear layers if specific matching fails
                print(f"  -> Fallback: Applying MXFP4 quantization to all linear layers")
                return MXFP4QuantizeMethod()
        
        return None

