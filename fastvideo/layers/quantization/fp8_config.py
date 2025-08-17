# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/character-ai/pipelining-sft/blob/main/models/deepseek_v3/fp8_layers_triton.py
from fastvideo.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
import torch
from torch.nn.parameter import Parameter

from typing import Any, Tuple
import deep_gemm
from deep_gemm import ceil_div, get_col_major_tma_aligned_tensor
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

    def create_weights(self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs):
        _, input_size_per_partition, output_partition_sizes, _, _, params_dtype, _ = weight_args
        weight = Parameter(torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype,
        ),
        requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        self.weight_fp8 = per_block_cast_to_fp8(layer.weight)

    def apply(self, layer: torch.nn.Module, *args, **kwargs):
        out_dim = layer.weight.shape[0]
        x = args[0]
        # Need contiguous tensors for collectives.
        assert x.dtype == torch.bfloat16, f"only allow bf16 inputs to fp8 linear"
        x_fp8 = per_token_cast_to_fp8(x)
        shape = x.shape
        # flattened
        out = torch.zeros((shape[0], out_dim), device=x.device, dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, self.weight_fp8, out)
        if len(shape) == 3:
            out = out.view(shape[0], shape[1], out_dim)
        return out
        

class FP8Config(QuantizationConfig):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "fp8"
    
    def get_supported_act_dtypes(self):
        #TODO: Confirm this
        return [torch.bfloat16]
    
    def get_min_capability(self):
        return 90
    
    def get_config_filenames(self):
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FP8Config":
        return cls()
    
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, torch.nn.Linear):
            return FP8QuantizeMethod()
        return None
    