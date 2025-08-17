# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/character-ai/pipelining-sft/blob/main/models/deepseek_v3/fp8_layers.py
from fastvideo.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
import torch
from torch.nn.parameter import Parameter

from typing import Any, Tuple
import deep_gemm
from deep_gemm import ceil_div
from fastvideo.models.utils import set_weight_attrs

block_size = 128

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)

def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

class FP8QuantizeMethod(QuantizeMethodBase):
    def __init__(self):
        super().__init__()
        self.weight_fp8 = None
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert weights to FP8 after loading from checkpoint."""
        if hasattr(layer, 'weight') and layer.weight is not None:
            # Convert the loaded weights to FP8
            self.weight_fp8, self.weight_scale = per_block_cast_to_fp8(layer.weight.data)
            # Store on the layer for later use
            layer._fp8_weight = self.weight_fp8
            layer._fp8_weight_scale = self.weight_scale

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Apply FP8 quantized computation."""
        # Ensure we have FP8 weights
        if not hasattr(layer, '_fp8_weight') or layer._fp8_weight is None:
            # Fallback to converting on-the-fly if not already converted
            self.weight_fp8, self.weight_scale = per_block_cast_to_fp8(layer.weight.data)
            layer._fp8_weight = self.weight_fp8
            layer._fp8_weight_scale = self.weight_scale
        
        out_dim = layer.weight.shape[0]
        # Need contiguous tensors for collectives.
        assert x.dtype == torch.bfloat16, f"only allow bf16 inputs to fp8 linear, got {x.dtype}"
        
        # Convert input to FP8
        x_fp8, x_scale = per_token_cast_to_fp8(x.view(-1, x.shape[-1]))
        original_shape = x.shape
        
        # Perform FP8 GEMM
        out = torch.zeros((x_fp8.shape[0], out_dim), device=x.device, dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale), (layer._fp8_weight, layer._fp8_weight_scale), out)
        
        # Restore original shape
        if len(original_shape) == 3:
            out = out.view(original_shape[0], original_shape[1], out_dim)
        
        return out
        

class FP8Config(QuantizationConfig):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "fp8"
    
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
    def from_config(cls, config: dict[str, Any]) -> "FP8Config":
        return cls()
    
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # Apply FP8 quantization to all linear layers
        from fastvideo.layers.linear import LinearBase
        if isinstance(layer, LinearBase):
            return FP8QuantizeMethod()
        return None


def convert_model_to_fp8(model: torch.nn.Module) -> torch.nn.Module:
    """
    Convert an existing model to use FP8 quantization.
    
    Args:
        model: The model to convert
        
    Returns:
        The model with FP8 quantization applied
    """
    from fastvideo.layers.linear import LinearBase
    
    fp8_config = FP8Config()
    
    # Convert all linear layers to use FP8 quantization
    def convert_layer_recursive(module: torch.nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, LinearBase):
                # Replace the quantization method
                quant_method = fp8_config.get_quant_method(child, child_prefix)
                if quant_method is not None:
                    child.quant_method = quant_method
                    child.quant_config = fp8_config
                    # Process weights to convert to FP8
                    quant_method.process_weights_after_loading(child)
            else:
                # Recursively process child modules
                convert_layer_recursive(child, child_prefix)
    
    convert_layer_recursive(model)
    return model
    