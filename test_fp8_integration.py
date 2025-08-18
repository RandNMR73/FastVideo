#!/usr/bin/env python3
"""
Test script to verify FP8 quantization integration.
"""

import torch
import torch.profiler
import time
import gc
from fastvideo.layers.quantization.fp8_config import FP8Config, FP8QuantizeMethod
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization import get_quantization_config

def test_fp8_registration():
    """Test that FP8 config is properly registered."""
    print("=== Testing FP8 Registration ===")
    
    try:
        # Try to get the FP8 config
        fp8_config_cls = get_quantization_config("fp8")
        print(f"✓ FP8 config class: {fp8_config_cls}")
        
        # Create an instance
        fp8_config = fp8_config_cls()
        print(f"✓ FP8 config instance: {fp8_config}")
        print(f"✓ FP8 config name: {fp8_config.get_name()}")
        
        return True
    except Exception as e:
        print(f"✗ FP8 registration failed: {e}")
        return False

def test_fp8_quant_method():
    """Test that FP8 quant method works correctly."""
    print("\n=== Testing FP8 Quant Method ===")
    
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
        
        # Now apply FP8 quantization
        fp8_config = FP8Config()
        quant_method = fp8_config.get_quant_method(linear_layer, "test")
        
        if quant_method is not None:
            print(f"✓ FP8 quant method created: {type(quant_method).__name__}")
            
            # Replace the quantization method
            linear_layer.quant_method = quant_method
            linear_layer.quant_config = fp8_config
            
            print(f"✓ Applied FP8 quantization to linear layer")
            print(f"✓ New quant method: {type(linear_layer.quant_method).__name__}")
            
            # Test weight processing
            if hasattr(quant_method, 'process_weights_after_loading'):
                quant_method.process_weights_after_loading(linear_layer)
                print(f"✓ Processed weights for FP8")
                
                if hasattr(linear_layer, '_fp8_weight'):
                    print(f"✓ FP8 weights created: {linear_layer._fp8_weight.shape}")
                else:
                    print(f"✗ FP8 weights not created")
            else:
                print(f"✗ process_weights_after_loading method not found")
            
            return True
        else:
            print(f"✗ FP8 quant method not created")
            return False
            
    except Exception as e:
        print(f"✗ FP8 quant method test failed: {e}")
        return False

def test_fp8_forward_pass():
    """Test that FP8 forward pass works."""
    print("\n=== Testing FP8 Forward Pass ===")
    
    try:
        # Create a linear layer with FP8 quantization
        fp8_config = FP8Config()
        linear_layer = ReplicatedLinear(
            input_size=128,
            output_size=256,
            bias=False,
            quant_config=fp8_config,
            params_dtype=torch.bfloat16
        )

        linear_unquant = ReplicatedLinear(
            input_size=128,
            output_size=256,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16
        )
        
        print(f"✓ Created FP8 linear layer")
        
        # Create test input (must be bfloat16 for FP8)
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
                torch.cuda.synchronize()
                end_time = time.time()
                print(f"Time taken to perform FP8 GEMM: {end_time - start_time} seconds")
                start_time = time.time()
                output_unquant = linear_unquant(test_input)
                torch.cuda.synchronize()
                end_time = time.time()
                print(f"Time taken to perform unquant GEMM: {end_time - start_time} seconds")
                print(f"✓ Forward pass successful: {output[0].shape}")
                print(f"✓ Forward pass successful: {output_unquant[0].shape}")
                print(f"✓ Output {output[0]}")
                print(f"✓ Output unquant {output_unquant[0]}")
                num, ref = output[0], output_unquant[0]
                rel_err = ((num - ref).abs() / (ref.abs().clamp_min(1e-30))).max()
                print("relative max-error:", rel_err.item())
            print(f"✓ Forward pass successful: {output[0].shape}")
            return True
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ FP8 forward pass test failed: {e}")
        return False

def test_model_integration():
    """Test how quantization config flows through model hierarchy."""
    print("\n=== Testing Model Integration ===")
    
    try:
        from fastvideo.configs.models.dits.base import DiTConfig
        
        # Create a DiT config with FP8 quantization
        dit_config = DiTConfig()
        dit_config.quant_config = FP8Config()
        
        print(f"✓ Created DiT config with FP8 quantization")
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

def benchmark_gemm_performance():
    """Benchmark TFLOPs for FP8 quantized vs unquantized GEMM operations."""
    print("\n=== Benchmarking GEMM Performance ===")
    
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping GPU benchmark")
            return False
            
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create traces directory if it doesn't exist
        import os
        os.makedirs("./traces", exist_ok=True)
        
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
                fp8_config = FP8Config()
                linear_fp8 = ReplicatedLinear(
                    input_size=hidden_size,
                    output_size=hidden_size,
                    bias=True,
                    quant_config=fp8_config,
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
                    linear_fp8.weight.normal_(0, 0.02)
                    linear_unquant.weight.copy_(linear_fp8.weight)
                    if linear_fp8.bias is not None:
                        linear_fp8.bias.zero_()
                        linear_unquant.bias.copy_(linear_fp8.bias)
                
                # Create test input
                test_input = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
                
                # Warmup runs
                print("  Warming up...")
                with torch.no_grad():
                    for _ in range(num_warmup):
                        _ = linear_fp8(test_input)
                        _ = linear_unquant(test_input)
                
                torch.cuda.synchronize()
                
                # Benchmark FP8 quantized with PyTorch profiler
                print("  Benchmarking FP8 quantized with profiler...")
                start_time = time.time()
                torch.cuda.reset_peak_memory_stats()
                
                # Create trace filename for FP8
                trace_filename_fp8 = f"fp8_trace_b{batch_size}_h{hidden_size}.json"
                
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=0,
                        warmup=5,
                        active=num_iterations-5,
                        repeat=1
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        dir_name="./traces",
                        worker_name=f"fp8_b{batch_size}_h{hidden_size}"
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    with_flops=True,
                    with_modules=True
                ) as prof:
                    with torch.no_grad():
                        for i in range(num_iterations):
                            _ = linear_fp8(test_input)
                            prof.step()
                
                # The trace is automatically saved by tensorboard_trace_handler
                chrome_trace_fp8 = f"./traces/fp8_b{batch_size}_h{hidden_size}.pt.trace.json"
                print(f"    ✓ FP8 trace automatically saved to: {chrome_trace_fp8}")
                
                torch.cuda.synchronize()
                fp8_time = time.time() - start_time
                fp8_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                
                # Calculate FP8 performance
                fp8_throughput = num_iterations / fp8_time
                fp8_tflops = (theoretical_tflops * num_iterations) / fp8_time
                
                # Benchmark unquantized with PyTorch profiler
                print("  Benchmarking unquantized with profiler...")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Create trace filename for unquantized
                trace_filename_unquant = f"unquant_trace_b{batch_size}_h{hidden_size}.json"
                
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=0,
                        warmup=5,
                        active=num_iterations-5,
                        repeat=1
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        dir_name="./traces",
                        worker_name=f"unquant_b{batch_size}_h{hidden_size}"
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    with_flops=True,
                    with_modules=True
                ) as prof:
                    with torch.no_grad():
                        for i in range(num_iterations):
                            _ = linear_unquant(test_input)
                            prof.step()
                
                # The trace is automatically saved by tensorboard_trace_handler
                chrome_trace_unquant = f"./traces/unquant_b{batch_size}_h{hidden_size}.pt.trace.json"
                print(f"    ✓ Unquantized trace automatically saved to: {chrome_trace_unquant}")
                
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
                speedup = unquant_time / fp8_time
                tflops_speedup = fp8_tflops / unquant_tflops
                
                # Store results
                result = {
                    'batch_size': batch_size,
                    'hidden_size': hidden_size,
                    'theoretical_tflops': theoretical_tflops,
                    'fp8_time': fp8_time,
                    'fp8_throughput': fp8_throughput,
                    'fp8_tflops': fp8_tflops,
                    'fp8_memory_gb': fp8_memory,
                    'unquant_time': unquant_time,
                    'unquant_throughput': unquant_throughput,
                    'unquant_tflops': unquant_tflops,
                    'unquant_memory_gb': unquant_memory,
                    'speedup': speedup,
                    'tflops_speedup': tflops_speedup,
                    'fp8_trace_file': chrome_trace_fp8,
                    'unquant_trace_file': chrome_trace_unquant
                }
                results.append(result)
                
                # Print results for this configuration
                print(f"    FP8:     {fp8_tflops:.3f} TFLOPs, {fp8_time:.4f}s, {fp8_memory:.2f}GB")
                print(f"    Unquant: {unquant_tflops:.3f} TFLOPs, {unquant_time:.4f}s, {unquant_memory:.2f}GB")
                print(f"    Speedup: {speedup:.2f}x (time), {tflops_speedup:.2f}x (TFLOPs)")
                
                # Clean up
                del linear_fp8, linear_unquant, test_input
                torch.cuda.empty_cache()
                gc.collect()
        
        # Print summary table
        print("\n" + "=" * 80)
        print("GEMM Performance Benchmark Summary")
        print("=" * 80)
        print(f"{'Config':<15} {'FP8 TFLOPs':<12} {'Unquant TFLOPs':<15} {'Speedup':<10} {'Memory FP8':<12} {'Memory Unquant':<15}")
        print("-" * 80)
        
        for result in results:
            config = f"{result['batch_size']}x{result['hidden_size']}"
            print(f"{config:<15} {result['fp8_tflops']:<12.3f} {result['unquant_tflops']:<15.3f} "
                  f"{result['speedup']:<10.2f} {result['fp8_memory_gb']:<12.2f} {result['unquant_memory_gb']:<15.2f}")
        
        # Calculate overall statistics
        avg_fp8_tflops = sum(r['fp8_tflops'] for r in results) / len(results)
        avg_unquant_tflops = sum(r['unquant_tflops'] for r in results) / len(results)
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        avg_tflops_speedup = sum(r['tflops_speedup'] for r in results) / len(results)
        
        print("-" * 80)
        print(f"{'AVERAGE':<15} {avg_fp8_tflops:<12.3f} {avg_unquant_tflops:<15.3f} "
              f"{avg_speedup:<10.2f} {'-':<12} {'-':<15}")
        
        print(f"\nOverall Performance Summary:")
        print(f"  Average FP8 TFLOPs: {avg_fp8_tflops:.3f}")
        print(f"  Average Unquant TFLOPs: {avg_unquant_tflops:.3f}")
        print(f"  Average Time Speedup: {avg_speedup:.2f}x")
        print(f"  Average TFLOPs Speedup: {avg_tflops_speedup:.2f}x")
        
        if avg_tflops_speedup > 1.0:
            print(f"  🚀 FP8 quantization provides {avg_tflops_speedup:.2f}x TFLOPs improvement!")
        else:
            print(f"  ⚠️  FP8 quantization shows {avg_tflops_speedup:.2f}x TFLOPs performance")
        
        # Print trace file information
        print(f"\n📊 Profiler Trace Files Generated:")
        print(f"  All Chrome trace files have been saved to the './traces/' directory")
        print(f"  To analyze bottlenecks:")
        print(f"    1. SCP the trace files to your local machine:")
        print(f"       scp -r user@remote:/path/to/fastvideo/traces/ ./local_traces/")
        print(f"    2. Open Chrome and navigate to: chrome://tracing/")
        print(f"    3. Load the .json trace files to view detailed performance analysis")
        print(f"    4. Look for:")
        print(f"       - CUDA kernel execution times")
        print(f"       - Memory allocation patterns")
        print(f"       - CPU-GPU synchronization overhead")
        print(f"       - Quantization vs dequantization costs")
        
        return True
        
    except Exception as e:
        print(f"✗ GEMM benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("FP8 Quantization Integration Tests")
    print("=" * 40)
    
    tests = [
        test_fp8_registration,
        test_fp8_quant_method,
        test_fp8_forward_pass,
        test_model_integration,
        benchmark_gemm_performance,
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
        print("🎉 All tests passed! FP8 quantization is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        
        if not results[0]:
            print("\n🔧 Fix needed: FP8 config registration issue")
        if not results[1]:
            print("\n🔧 Fix needed: FP8 quant method creation issue")
        if not results[2]:
            print("\n🔧 Fix needed: FP8 forward pass issue")
        if not results[3]:
            print("\n🔧 Fix needed: Model integration issue")
        if not results[4]:
            print("\n🔧 Fix needed: GEMM performance benchmark issue")

if __name__ == "__main__":
    main() 