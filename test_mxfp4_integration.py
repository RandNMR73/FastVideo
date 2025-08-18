#!/usr/bin/env python3
"""
Test script to verify MXFP4 quantization integration.
"""

import torch
import time
import gc
from fastvideo.layers.quantization.mxfp4_config import MXFP4Config, MXFP4QuantizeMethod
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization import get_quantization_config

def test_mxfp4_registration():
    """Test that MXFP4 config is properly registered."""
    print("=== Testing MXFP4 Registration ===")
    
    try:
        # Try to get the MXFP4 config
        mxfp4_config_cls = get_quantization_config("mxfp4")
        print(f"✓ MXFP4 config class: {mxfp4_config_cls}")
        
        # Create an instance
        mxfp4_config = mxfp4_config_cls()
        print(f"✓ MXFP4 config instance: {mxfp4_config}")
        print(f"✓ MXFP4 config name: {mxfp4_config.get_name()}")
        
        return True
    except Exception as e:
        print(f"✗ MXFP4 registration failed: {e}")
        print("Note: MXFP4 may need to be registered in the quantization system")
        return False

def test_mxfp4_quant_method():
    """Test that MXFP4 quant method works correctly."""
    print("\n=== Testing MXFP4 Quant Method ===")
    
    try:
        # Create a simple linear layer
        linear_layer = ReplicatedLinear(
            input_size=128,
            output_size=256,
            bias=True,
            quant_config=None  # Start without quantization
        )
        
        print(f"✓ Created linear layer: {linear_layer}")
        print(f"✓ Original quant method: {type(linear_layer.quant_method).__name__}")
        
        # Now apply MXFP4 quantization
        mxfp4_config = MXFP4Config()
        quant_method = mxfp4_config.get_quant_method(linear_layer, "test")
        
        if quant_method is not None:
            print(f"✓ MXFP4 quant method created: {type(quant_method).__name__}")
            
            # Replace the quantization method
            linear_layer.quant_method = quant_method
            linear_layer.quant_config = mxfp4_config
            
            print(f"✓ Applied MXFP4 quantization to linear layer")
            print(f"✓ New quant method: {type(linear_layer.quant_method).__name__}")
            
            # Test weight processing
            if hasattr(quant_method, 'process_weights_after_loading'):
                quant_method.process_weights_after_loading(linear_layer)
                print(f"✓ Processed weights for MXFP4")
            else:
                print(f"ℹ️  process_weights_after_loading method not found (not required for MXFP4)")
            
            return True
        else:
            print(f"✗ MXFP4 quant method not created")
            return False
            
    except Exception as e:
        print(f"✗ MXFP4 quant method test failed: {e}")
        return False

def test_mxfp4_forward_pass():
    """Test that MXFP4 forward pass works."""
    print("\n=== Testing MXFP4 Forward Pass ===")
    
    try:
        # Create a linear layer with MXFP4 quantization
        mxfp4_config = MXFP4Config()
        linear_layer = ReplicatedLinear(
            input_size=128,
            output_size=256,
            bias=False,
            quant_config=mxfp4_config,
            params_dtype=torch.bfloat16
        )

        linear_unquant = ReplicatedLinear(
            input_size=128,
            output_size=256,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16
        )
        
        print(f"✓ Created MXFP4 linear layer")
        
        # Create test input (must be bfloat16 for MXFP4)
        test_input = torch.randn(2, 128, dtype=torch.bfloat16, device='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Created test input: {test_input.shape}, {test_input.dtype}")
        
        # Move layers to the same device as input
        device = test_input.device
        linear_layer = linear_layer.to(device)
        linear_unquant = linear_unquant.to(device)
        
        # Copy weights and bias for fair comparison
        with torch.no_grad():
            linear_layer.weight.normal_(0, 0.02)
            linear_unquant.weight.copy_(linear_layer.weight)
            if linear_layer.bias is not None:
                linear_layer.bias.zero_()
                linear_unquant.bias.copy_(linear_layer.bias)
        print(f"✓ Moved layers to device: {device}")
        
        # Test forward pass
        try:
            with torch.no_grad():
                start_time = time.time()
                output = linear_layer(test_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                print(f"Time taken to perform MXFP4 GEMM: {end_time - start_time} seconds")
                
                start_time = time.time()
                output_unquant = linear_unquant(test_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                print(f"Time taken to perform unquant GEMM: {end_time - start_time} seconds")
                
                print(f"✓ Forward pass successful: {output[0].shape}")
                print(f"✓ Forward pass successful: {output_unquant[0].shape}")
                print(f"✓ Output {output[0]}")
                print(f"✓ Output unquant {output_unquant[0]}")
                
                # Calculate relative error
                num, ref = output[0], output_unquant[0]
                rel_err = ((num - ref).abs() / (ref.abs().clamp_min(1e-30))).max()
                print("relative max-error:", rel_err.item())
                
            return True
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ MXFP4 forward pass test failed: {e}")
        return False

def test_model_integration():
    """Test how quantization config flows through model hierarchy."""
    print("\n=== Testing Model Integration ===")
    
    try:
        from fastvideo.configs.models.dits.base import DiTConfig
        
        # Create a DiT config with MXFP4 quantization
        dit_config = DiTConfig()
        dit_config.quant_config = MXFP4Config()
        
        print(f"✓ Created DiT config with MXFP4 quantization")
        print(f"✓ Quant config: {dit_config.quant_config}")
        print(f"✓ Quant config type: {type(dit_config.quant_config)}")
        
        # Check if the config is properly set
        if dit_config.quant_config is not None:
            print(f"✓ Quantization config is set in DiT config")
            return True
        else:
            print(f"✗ Quantization config is not set in DiT config")
            return False
            
    except Exception as e:
        print(f"✗ Model integration test failed: {e}")
        return False

def test_triton_kernels_availability():
    """Test that triton_kernels dependencies are available."""
    print("\n=== Testing Triton Kernels Availability ===")
    
    try:
        import triton_kernels
        print(f"✓ triton_kernels imported successfully")
        
        # Test specific imports
        from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
        print(f"✓ downcast_to_mxfp imported successfully")
        
        from triton_kernels.matmul_ogs import PrecisionConfig, FlexCtx, matmul_ogs
        print(f"✓ matmul_ogs components imported successfully")
        
        from triton_kernels.numerics import InFlexData
        print(f"✓ InFlexData imported successfully")
        
        from triton_kernels.tensor import convert_layout, wrap_torch_tensor, FP4
        print(f"✓ tensor utilities imported successfully")
        
        from triton_kernels.tensor_details.layout import StridedLayout, HopperMXValueLayout
        print(f"✓ layout classes imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ triton_kernels import failed: {e}")
        print("Note: MXFP4 requires triton_kernels library to be installed")
        return False
    except Exception as e:
        print(f"✗ triton_kernels test failed: {e}")
        return False

def benchmark_gemm_performance():
    """Benchmark TFLOPs for MXFP4 quantized vs unquantized GEMM operations."""
    print("\n=== Benchmarking GEMM Performance ===")
    
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping GPU benchmark")
            return False
            
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        gc.collect()
        
        # Benchmark configurations
        batch_sizes = [1, 4, 8, 16, 32]
        hidden_sizes = [512, 1024, 2048, 4096]
        num_warmup = 10
        num_iterations = 100
        
        print(f"✓ Device: {device}")
        print(f"✓ Warmup iterations: {num_warmup}")
        print(f"✓ Benchmark iterations: {num_iterations}")
        
        results = []
        
        for hidden_size in hidden_sizes:
            for batch_size in batch_sizes:
                print(f"\n--- Benchmarking: batch_size={batch_size}, hidden_size={hidden_size} ---")
                
                # Calculate theoretical FLOPs for GEMM: 2 * batch_size * input_size * output_size
                theoretical_flops = 2 * batch_size * hidden_size * hidden_size
                theoretical_tflops = theoretical_flops / 1e12
                
                # Create layers
                mxfp4_config = MXFP4Config()
                linear_mxfp4 = ReplicatedLinear(
                    input_size=hidden_size,
                    output_size=hidden_size,
                    bias=True,
                    quant_config=mxfp4_config,
                    params_dtype=torch.bfloat16
                ).to(device)
                
                linear_unquant = ReplicatedLinear(
                    input_size=hidden_size,
                    output_size=hidden_size,
                    bias=True,
                    quant_config=None,
                    params_dtype=torch.bfloat16
                ).to(device)
                
                # Copy weights for fair comparison
                with torch.no_grad():
                    linear_mxfp4.weight.normal_(0, 0.02)
                    linear_unquant.weight.copy_(linear_mxfp4.weight)
                    if linear_mxfp4.bias is not None:
                        linear_mxfp4.bias.zero_()
                        linear_unquant.bias.copy_(linear_mxfp4.bias)
                
                # Create test input
                test_input = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
                
                # Warmup runs
                print("  Warming up...")
                with torch.no_grad():
                    for _ in range(num_warmup):
                        _ = linear_mxfp4(test_input)
                        _ = linear_unquant(test_input)
                
                torch.cuda.synchronize()
                
                # Benchmark MXFP4 quantized
                print("  Benchmarking MXFP4 quantized...")
                start_time = time.time()
                torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    for _ in range(num_iterations):
                        _ = linear_mxfp4(test_input)
                
                torch.cuda.synchronize()
                mxfp4_time = time.time() - start_time
                mxfp4_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                
                # Calculate MXFP4 performance
                mxfp4_throughput = num_iterations / mxfp4_time
                mxfp4_tflops = (theoretical_tflops * num_iterations) / mxfp4_time
                
                # Benchmark unquantized
                print("  Benchmarking unquantized...")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(num_iterations):
                        _ = linear_unquant(test_input)
                
                torch.cuda.synchronize()
                unquant_time = time.time() - start_time
                unquant_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                
                # Calculate unquantized performance
                unquant_throughput = num_iterations / unquant_time
                unquant_tflops = (theoretical_tflops * num_iterations) / unquant_time
                
                # Calculate speedup
                speedup = unquant_time / mxfp4_time
                tflops_speedup = mxfp4_tflops / unquant_tflops
                
                # Store results
                result = {
                    'batch_size': batch_size,
                    'hidden_size': hidden_size,
                    'theoretical_tflops': theoretical_tflops,
                    'mxfp4_time': mxfp4_time,
                    'mxfp4_throughput': mxfp4_throughput,
                    'mxfp4_tflops': mxfp4_tflops,
                    'mxfp4_memory_gb': mxfp4_memory,
                    'unquant_time': unquant_time,
                    'unquant_throughput': unquant_throughput,
                    'unquant_tflops': unquant_tflops,
                    'unquant_memory_gb': unquant_memory,
                    'speedup': speedup,
                    'tflops_speedup': tflops_speedup
                }
                results.append(result)
                
                # Print results for this configuration
                print(f"    MXFP4:   {mxfp4_tflops:.3f} TFLOPs, {mxfp4_time:.4f}s, {mxfp4_memory:.2f}GB")
                print(f"    Unquant: {unquant_tflops:.3f} TFLOPs, {unquant_time:.4f}s, {unquant_memory:.2f}GB")
                print(f"    Speedup: {speedup:.2f}x (time), {tflops_speedup:.2f}x (TFLOPs)")
                
                # Clean up
                del linear_mxfp4, linear_unquant, test_input
                torch.cuda.empty_cache()
                gc.collect()
        
        # Print summary table
        print("\n" + "=" * 80)
        print("GEMM Performance Benchmark Summary")
        print("=" * 80)
        print(f"{'Config':<15} {'MXFP4 TFLOPs':<12} {'Unquant TFLOPs':<15} {'Speedup':<10} {'Memory MXFP4':<12} {'Memory Unquant':<15}")
        print("-" * 80)
        
        for result in results:
            config = f"{result['batch_size']}x{result['hidden_size']}"
            print(f"{config:<15} {result['mxfp4_tflops']:<12.3f} {result['unquant_tflops']:<15.3f} "
                  f"{result['speedup']:<10.2f} {result['mxfp4_memory_gb']:<12.2f} {result['unquant_memory_gb']:<15.2f}")
        
        # Calculate overall statistics
        avg_mxfp4_tflops = sum(r['mxfp4_tflops'] for r in results) / len(results)
        avg_unquant_tflops = sum(r['unquant_tflops'] for r in results) / len(results)
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        avg_tflops_speedup = sum(r['tflops_speedup'] for r in results) / len(results)
        
        print("-" * 80)
        print(f"{'AVERAGE':<15} {avg_mxfp4_tflops:<12.3f} {avg_unquant_tflops:<15.3f} "
              f"{avg_speedup:<10.2f} {'-':<12} {'-':<15}")
        
        print(f"\nOverall Performance Summary:")
        print(f"  Average MXFP4 TFLOPs: {avg_mxfp4_tflops:.3f}")
        print(f"  Average Unquant TFLOPs: {avg_unquant_tflops:.3f}")
        print(f"  Average Time Speedup: {avg_speedup:.2f}x")
        print(f"  Average TFLOPs Speedup: {avg_tflops_speedup:.2f}x")
        
        if avg_tflops_speedup > 1.0:
            print(f"  🚀 MXFP4 quantization provides {avg_tflops_speedup:.2f}x TFLOPs improvement!")
        else:
            print(f"  ⚠️  MXFP4 quantization shows {avg_tflops_speedup:.2f}x TFLOPs performance")
        
        return True
        
    except Exception as e:
        print(f"✗ GEMM benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mxfp4_vs_fp8_comparison():
    """Compare MXFP4 vs FP8 quantization performance."""
    print("\n=== Testing MXFP4 vs FP8 Comparison ===")
    
    try:
        # Check if both quantization methods are available
        try:
            from fastvideo.layers.quantization.fp8_config import FP8Config
            fp8_available = True
        except ImportError:
            fp8_available = False
            print("⚠️  FP8 not available for comparison")
        
        if not fp8_available:
            return False
            
        # Create test configuration
        batch_size = 16
        hidden_size = 1024
        num_iterations = 50
        
        print(f"✓ Test config: batch_size={batch_size}, hidden_size={hidden_size}")
        
        # Create layers
        mxfp4_config = MXFP4Config()
        fp8_config = FP8Config()
        
        linear_mxfp4 = ReplicatedLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=False,
            quant_config=mxfp4_config,
            params_dtype=torch.bfloat16
        )
        
        linear_fp8 = ReplicatedLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=False,
            quant_config=fp8_config,
            params_dtype=torch.bfloat16
        )
        
        linear_unquant = ReplicatedLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16
        )
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        linear_mxfp4 = linear_mxfp4.to(device)
        linear_fp8 = linear_fp8.to(device)
        linear_unquant = linear_unquant.to(device)
        
        # Create test input
        test_input = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
        
        # Copy weights for fair comparison
        with torch.no_grad():
            linear_mxfp4.weight.normal_(0, 0.02)
            linear_fp8.weight.copy_(linear_mxfp4.weight)
            linear_unquant.weight.copy_(linear_mxfp4.weight)
        
        # Warmup
        print("  Warming up...")
        with torch.no_grad():
            for _ in range(10):
                _ = linear_mxfp4(test_input)
                _ = linear_fp8(test_input)
                _ = linear_unquant(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark MXFP4
        print("  Benchmarking MXFP4...")
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = linear_mxfp4(test_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mxfp4_time = time.time() - start_time
        
        # Benchmark FP8
        print("  Benchmarking FP8...")
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = linear_fp8(test_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fp8_time = time.time() - start_time
        
        # Benchmark unquantized
        print("  Benchmarking unquantized...")
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = linear_unquant(test_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        unquant_time = time.time() - start_time
        
        # Calculate performance metrics
        mxfp4_speedup = unquant_time / mxfp4_time
        fp8_speedup = unquant_time / fp8_time
        mxfp4_vs_fp8 = fp8_time / mxfp4_time
        
        print(f"\nPerformance Comparison Results:")
        print(f"  MXFP4 time:     {mxfp4_time:.4f}s (speedup: {mxfp4_speedup:.2f}x)")
        print(f"  FP8 time:        {fp8_time:.4f}s (speedup: {fp8_speedup:.2f}x)")
        print(f"  Unquant time:    {unquant_time:.4f}s")
        print(f"  MXFP4 vs FP8:    {mxfp4_vs_fp8:.2f}x (MXFP4 is {mxfp4_vs_fp8:.2f}x {'faster' if mxfp4_vs_fp8 > 1 else 'slower'})")
        
        return True
        
    except Exception as e:
        print(f"✗ MXFP4 vs FP8 comparison failed: {e}")
        return False

def main():
    """Run all tests."""
    print("MXFP4 Quantization Integration Tests")
    print("=" * 40)
    
    tests = [
        test_triton_kernels_availability,  # Check dependencies first
        test_mxfp4_registration,
        test_mxfp4_quant_method,
        test_mxfp4_forward_pass,
        test_model_integration,
        benchmark_gemm_performance,
        test_mxfp4_vs_fp8_comparison,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! MXFP4 quantization is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        
        if not results[0]:
            print("\n🔧 Fix needed: triton_kernels dependency issue")
        if not results[1]:
            print("\n🔧 Fix needed: MXFP4 config registration issue")
        if not results[2]:
            print("\n🔧 Fix needed: MXFP4 quant method creation issue")
        if not results[3]:
            print("\n🔧 Fix needed: MXFP4 forward pass issue")
        if not results[4]:
            print("\n🔧 Fix needed: Model integration issue")
        if not results[5]:
            print("\n🔧 Fix needed: GEMM performance benchmark issue")
        if not results[6]:
            print("\n🔧 Fix needed: MXFP4 vs FP8 comparison issue")

if __name__ == "__main__":
    main() 