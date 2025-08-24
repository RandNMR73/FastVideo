"""
MXFP4 Quantization Module for GPT-OSS

This module provides MXFP4 quantized linear layers and utilities for testing
quantization performance in the gpt-oss repository.
"""

import torch
import torch.nn as nn
from torch.profiler import record_function
import time

import triton_kernels
import triton_kernels.swiglu
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.matmul_ogs import PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
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


class MXFP4Linear(nn.Module):
    """
    MXFP4 quantized linear layer that can be used as a drop-in replacement
    for torch.nn.Linear in the gpt-oss codebase.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight parameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features, 
                                             dtype=torch.bfloat16, device=device))
        
        # Create bias parameter if requested
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device=device))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
        
        # Quantized weight and scale (will be set during forward pass)
        self.quantized_weight = None
        self.weight_scale = None
        self._quantized = False
    
    def reset_parameters(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _quantize_weights(self):
        """Quantize weights to MXFP4 format."""
        if not self._quantized:
            self.quantized_weight, self.weight_scale = quantize_mx4(self.weight)
            self._quantized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MXFP4 quantized weights."""
        if not self._quantized:
            self._quantize_weights()
        
        # Ensure input is bfloat16
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        
        # Create precision config
        pc = PrecisionConfig(weight_scale=self.weight_scale, 
                           flex_ctx=FlexCtx(rhs_data=InFlexData()))
        
        # Perform quantized matrix multiplication
        output = matmul_ogs(x, self.quantized_weight, self.bias, precision_config=pc)
        
        return output


class MXFP4LinearWithActivation(nn.Module):
    """
    MXFP4 quantized linear layer with optional SwiGLU activation.
    This mimics the behavior of the MLP blocks in the gpt-oss model.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: str = "swiglu", swiglu_limit: float = 7.0,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.swiglu_limit = swiglu_limit
        
        # Create weight parameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features, 
                                             dtype=torch.bfloat16, device=device))
        
        # Create bias parameter if requested
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device=device))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
        
        # Quantized weight and scale (will be set during forward pass)
        self.quantized_weight = None
        self.weight_scale = None
        self._quantized = False
    
    def reset_parameters(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _quantize_weights(self):
        """Quantize weights to MXFP4 format."""
        if not self._quantized:
            self.quantized_weight, self.weight_scale = quantize_mx4(self.weight)
            print(f"quantized weight tensor stride: {self.quantized_weight.stride(-2)}")
            self._quantized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MXFP4 quantized weights and activation."""
        if not self._quantized:
            self._quantize_weights()
        
        # Ensure input is bfloat16
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        
        # Create precision config
        pc = PrecisionConfig(weight_scale=self.weight_scale, 
                           flex_ctx=FlexCtx(rhs_data=InFlexData()))
        
        if self.activation == "swiglu":
            # Use fused SwiGLU activation
            act = FusedActivation(
                FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")), 
                (1.702, self.swiglu_limit), 2
            )
            output = matmul_ogs(x, self.quantized_weight, self.bias, 
                               precision_config=pc, fused_activation=act)
        else:
            # No activation
            output = matmul_ogs(x, self.quantized_weight, self.bias, precision_config=pc)
        
        return output


def create_quantized_mlp_block(hidden_size: int, intermediate_size: int, 
                              num_experts: int = 1, experts_per_token: int = 1,
                              swiglu_limit: float = 7.0, device=None):
    """
    Create a quantized MLP block similar to the one used in gpt-oss.
    
    Args:
        hidden_size: Size of hidden dimension
        intermediate_size: Size of intermediate dimension
        num_experts: Number of experts (for MoE)
        experts_per_token: Number of experts per token
        swiglu_limit: Limit for SwiGLU activation
        device: Device to place the model on
    
    Returns:
        Tuple of (gate_layer, mlp1_layer, mlp2_layer)
    """
    # Gate layer (no quantization, used for routing)
    gate_layer = nn.Linear(hidden_size, num_experts, bias=True, 
                          device=device, dtype=torch.bfloat16)
    
    # MLP1 layer with SwiGLU activation (quantized)
    mlp1_layer = MXFP4LinearWithActivation(
        in_features=hidden_size,
        out_features=intermediate_size * 2,  # *2 for SwiGLU
        bias=True,
        activation="swiglu",
        swiglu_limit=swiglu_limit,
        device=device
    )
    
    # MLP2 layer (quantized)
    mlp2_layer = MXFP4LinearWithActivation(
        in_features=intermediate_size,
        out_features=hidden_size,
        bias=True,
        activation=None,  # No activation
        device=device
    )
    
    return gate_layer, mlp1_layer, mlp2_layer


def benchmark_linear_layers(input_size: int, hidden_size: int, output_size: int,
                           batch_size: int, num_iterations: int = 100,
                           warmup_iterations: int = 10, device=None):
    """
    Benchmark performance of standard vs MXFP4 quantized linear layers.
    
    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension
        batch_size: Batch size for testing
        num_iterations: Number of iterations for benchmarking
        warmup_iterations: Number of warmup iterations
        device: Device to run benchmarks on
    
    Returns:
        Dictionary with benchmark results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Benchmarking on device: {device}")
    print(f"Input: {batch_size}x{input_size}, Hidden: {hidden_size}, Output: {output_size}")
    
    # Create test input
    x = torch.randn(batch_size, input_size, dtype=torch.bfloat16, device=device)
    
    # Create standard linear layer
    standard_linear = nn.Linear(input_size, output_size, bias=True, 
                               device=device, dtype=torch.bfloat16)
    
    # Create MXFP4 quantized linear layer
    mxfp4_linear = MXFP4Linear(input_size, output_size, bias=True, 
                               device=device, dtype=torch.bfloat16)
    
    # Copy weights for fair comparison
    with torch.no_grad():
        mxfp4_linear.weight.copy_(standard_linear.weight)
        if standard_linear.bias is not None:
            mxfp4_linear.bias.copy_(standard_linear.bias.to(torch.float32))
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = standard_linear(x)
            _ = mxfp4_linear(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark standard linear layer
    print("Benchmarking standard linear layer...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = standard_linear(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    standard_time = time.time() - start_time
    
    # Benchmark MXFP4 linear layer
    print("Benchmarking MXFP4 quantized linear layer...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = mxfp4_linear(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mxfp4_time = time.time() - start_time
    
    # Calculate performance metrics
    theoretical_flops = 2 * batch_size * input_size * output_size
    theoretical_tflops = theoretical_flops / 1e12
    
    standard_tflops = (theoretical_tflops * num_iterations) / standard_time
    mxfp4_tflops = (theoretical_tflops * num_iterations) / mxfp4_time
    
    speedup = standard_time / mxfp4_time
    tflops_speedup = mxfp4_tflops / standard_tflops
    
    # Memory usage (if CUDA available)
    if torch.cuda.is_available():
        standard_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        torch.cuda.reset_peak_memory_stats()
        
        _ = mxfp4_linear(x)  # Trigger quantization
        mxfp4_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    else:
        standard_memory = 0
        mxfp4_memory = 0
    
    results = {
        'standard_time': standard_time,
        'standard_tflops': standard_tflops,
        'standard_memory_gb': standard_memory,
        'mxfp4_time': mxfp4_time,
        'mxfp4_tflops': mxfp4_tflops,
        'mxfp4_memory_gb': mxfp4_memory,
        'speedup': speedup,
        'tflops_speedup': tflops_speedup,
        'theoretical_tflops': theoretical_tflops,
        'num_iterations': num_iterations
    }
    
    print(f"\nBenchmark Results:")
    print(f"  Standard:  {standard_tflops:.3f} TFLOPs, {standard_time:.4f}s, {standard_memory:.2f}GB")
    print(f"  MXFP4:     {mxfp4_tflops:.3f} TFLOPs, {mxfp4_time:.4f}s, {mxfp4_memory:.2f}GB")
    print(f"  Speedup:   {speedup:.2f}x (time), {tflops_speedup:.2f}x (TFLOPs)")
    
    return results


def test_quantization_accuracy(input_size: int, hidden_size: int, output_size: int,
                              batch_size: int, num_tests: int = 10, device=None):
    """
    Test the accuracy of MXFP4 quantization compared to standard linear layers.
    
    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension
        batch_size: Batch size for testing
        num_tests: Number of random tests to run
        device: Device to run tests on
    
    Returns:
        Dictionary with accuracy results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing quantization accuracy on device: {device}")
    
    # Create layers
    standard_linear = nn.Linear(input_size, output_size, bias=True, 
                               device=device, dtype=torch.bfloat16)
    mxfp4_linear = MXFP4Linear(input_size, output_size, bias=True, 
                               device=device, dtype=torch.bfloat16)
    
    # Copy weights for fair comparison
    with torch.no_grad():
        mxfp4_linear.weight.copy_(standard_linear.weight)
        if standard_linear.bias is not None:
            mxfp4_linear.bias.copy_(standard_linear.bias.to(torch.float32))
    
    # Test with random inputs
    max_abs_error = 0.0
    max_rel_error = 0.0
    total_abs_error = 0.0
    total_rel_error = 0.0
    
    for i in range(num_tests):
        # Generate random input
        x = torch.randn(batch_size, input_size, dtype=torch.bfloat16, device=device)
        
        # Get outputs
        with torch.no_grad():
            standard_output = standard_linear(x)
            mxfp4_output = mxfp4_linear(x)
        
        # Calculate errors
        abs_error = (standard_output - mxfp4_output).abs()
        rel_error = abs_error / (standard_output.abs().clamp_min(1e-30))
        
        max_abs_error = max(max_abs_error, abs_error.max().item())
        max_rel_error = max(max_rel_error, rel_error.max().item())
        total_abs_error += abs_error.mean().item()
        total_rel_error += rel_error.mean().item()
        
        if i < 3:  # Show first few results
            print(f"  Test {i+1}: Max abs error: {abs_error.max().item():.6f}, "
                  f"Max rel error: {rel_error.max().item():.6f}")
    
    avg_abs_error = total_abs_error / num_tests
    avg_rel_error = total_rel_error / num_tests
    
    results = {
        'max_abs_error': max_abs_error,
        'max_rel_error': max_rel_error,
        'avg_abs_error': avg_abs_error,
        'avg_rel_error': avg_rel_error,
        'num_tests': num_tests
    }
    
    print(f"\nAccuracy Results:")
    print(f"  Max absolute error: {max_abs_error:.6f}")
    print(f"  Max relative error: {max_rel_error:.6f}")
    print(f"  Avg absolute error: {avg_abs_error:.6f}")
    print(f"  Avg relative error: {avg_rel_error:.6f}")
    
    return results 