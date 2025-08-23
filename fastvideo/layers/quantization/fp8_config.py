# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/character-ai/pipelining-sft/blob/main/models/deepseek_v3/fp8_layers.py
from fastvideo.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
import torch
from torch.nn.parameter import Parameter

from typing import Any, Tuple
import deep_gemm
from deep_gemm import ceil_div, get_mn_major_tma_aligned_tensor
from fastvideo.models.utils import set_weight_attrs
# import time 

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

    @torch.compile
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Apply FP8 quantized computation."""
        # if not hasattr(layer, '_fp8_weight') or layer._fp8_weight is None:
        #     # start_time = time.time()
        #     self.weight_fp8, self.weight_scale = per_block_cast_to_fp8(layer.weight)
        #     torch.cuda.synchronize()
        #     # end_time = time.time()
        #     # print(f"Time taken to cast weight to FP8: {end_time - start_time} seconds")
        #     layer._fp8_weight = self.weight_fp8
        #     layer._fp8_weight_scale = self.weight_scale
        
        out_dim = layer.weight.shape[0]
        # Need contiguous tensors for collectives.
        assert x.dtype == torch.bfloat16, f"only allow bf16 inputs to fp8 linear, got {x.dtype}"
        
        # start_time = time.time()
        x_fp8, x_scale = per_token_cast_to_fp8(x.view(-1, x.shape[-1]))
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"Time taken to cast input to FP8: {end_time - start_time} seconds")
        # print(f"x_scale.dtype: {x_scale.dtype}")
        x_scale = get_mn_major_tma_aligned_tensor(x_scale)
        weight_fp8 = layer._fp8_weight
        weight_scale = layer._fp8_weight_scale
        
        original_shape = x.shape
        out = torch.zeros((x_fp8.shape[0], out_dim), device=x.device, dtype=x.dtype)
        # start_time = time.time()
        
        deep_gemm.fp8_gemm_nt(
            (x_fp8, x_scale),
            (weight_fp8, weight_scale),
            out,
            # disable_ue8m0_cast=False  # TODO: need to set flag based on sm90/sm100
        )   
            
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"Time taken to perform FP8 GEMM: {end_time - start_time} seconds")
        
        if bias is not None:
            if bias.device != out.device or bias.dtype != out.dtype:
                bias = bias.to(device=out.device, dtype=out.dtype)
            out = out + bias
        
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
        # Apply FP8 quantization only to linear layers with "ffn.fc_in" in their prefix
        from fastvideo.layers.linear import LinearBase
        if isinstance(layer, LinearBase) and "ffn.fc_in" in prefix:
            return FP8QuantizeMethod()
        return None

@torch.compile
def convert_model_to_fp8(model: torch.nn.Module):
    from torch.distributed.tensor import DTensor  # type: ignore
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, FP8QuantizeMethod):
            weight = getattr(mod, "weight", None)
            if weight is None:
                continue
            if isinstance(weight, DTensor):  # type: ignore
                weight_local = weight.to_local()
            else:
                weight_local = weight
            fp8_w, fp8_s = per_block_cast_to_fp8(weight_local)
            mod.register_buffer("_fp8_weight", fp8_w, persistent=False)
            mod.register_buffer("_fp8_weight_scale", fp8_s, persistent=False)