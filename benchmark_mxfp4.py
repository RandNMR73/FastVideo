#!/usr/bin/env python3
"""
Simple benchmark script for MXFP4 vs standard linear layers in GPT-OSS.

This script provides a quick way to test if MXFP4 quantization provides
performance benefits for linear layers while maintaining correctness.
"""

import torch
import time
import gc
import argparse
import numpy as np

# Import our MXFP4 quantization module
from gpt_oss.triton.mxfp4_quantization import MXFP4Linear


def test_correctness(input_size: int, output_size: int, batch_size: int, 
                    num_tests: int = 10, device=None, tolerance: float = 1e-3):
    """
    Test correctness of MXFP4 quantization vs standard linear layers.
    
    Args:
        input_size: Input dimension
        output_size: Output dimension  
        batch_size: Batch size
        num_tests: Number of random test cases
        device: Device to run on
        tolerance: Tolerance for numerical differences
    
    Returns:
        dict: Correctness test results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing correctness on {device}")
    print(f"Input: {batch_size}x{input_size} -> {output_size}")
    print(f"Tests: {num_tests}, Tolerance: {tolerance}")
    
    # Create layers
    standard_linear = torch.nn.Linear(input_size, output_size, bias=True, 
                                     device=device, dtype=torch.bfloat16)
    mxfp4_linear = MXFP4Linear(input_size, output_size, bias=True, 
                               device=device, dtype=torch.bfloat16)
    
    # Copy weights for fair comparison
    with torch.no_grad():
        mxfp4_linear.weight.copy_(standard_linear.weight)
        if standard_linear.bias is not None:
            mxfp4_linear.bias.copy_(standard_linear.bias)
    
    # Test with random inputs
    max_diff = 0.0
    max_rel_diff = 0.0
    all_passed = True
    test_results = []
    
    for test_idx in range(num_tests):
        # Generate random input
        x = torch.randn(batch_size, input_size, dtype=torch.bfloat16, device=device)
        
        # Forward pass
        with torch.no_grad():
            standard_output = standard_linear(x)
            mxfp4_output = mxfp4_linear(x)
        
        # Calculate differences
        abs_diff = torch.abs(standard_output - mxfp4_output)
        rel_diff = abs_diff / (torch.abs(standard_output) + 1e-8)
        
        max_abs_diff = torch.max(abs_diff).item()
        max_rel_diff_val = torch.max(rel_diff).item()
        
        # Check if within tolerance
        passed = max_abs_diff <= tolerance
        
        test_results.append({
            'test_idx': test_idx,
            'max_abs_diff': max_abs_diff,
            'max_rel_diff': max_rel_diff_val,
            'passed': passed
        })
        
        if not passed:
            all_passed = False
        
        max_diff = max(max_diff, max_abs_diff)
        max_rel_diff = max(max_rel_diff, max_rel_diff_val)
    
    # Print results
    print(f"\nCorrectness Test Results:")
    print(f"  All tests passed: {'✓' if all_passed else '✗'}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Max relative difference: {max_rel_diff:.6f}")
    print(f"  Tolerance: {tolerance}")
    
    if all_passed:
        print("🎯 All correctness tests passed!")
    else:
        print("⚠️  Some correctness tests failed!")
    
    return {
        'all_passed': all_passed,
        'max_abs_diff': max_diff,
        'max_rel_diff': max_rel_diff,
        'test_results': test_results,
        'tolerance': tolerance
    }


def test_edge_cases(device=None):
    """
    Test edge cases for correctness.
    
    Args:
        device: Device to run on
    
    Returns:
        dict: Edge case test results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nTesting edge cases on {device}")
    
    edge_cases = [
        # Zero input
        torch.zeros(1, 64, dtype=torch.bfloat16, device=device),
        # Small values
        torch.full((1, 64), 1e-6, dtype=torch.bfloat16, device=device),
        # Large values
        torch.full((1, 64), 1e6, dtype=torch.bfloat16, device=device),
        # Mixed signs
        torch.cat([
            torch.full((1, 32), 1.0, dtype=torch.bfloat16, device=device),
            torch.full((1, 32), -1.0, dtype=torch.bfloat16, device=device)
        ], dim=1),
        # NaN/Inf handling
        torch.tensor([[float('nan'), float('inf'), float('-inf'), 1.0]], 
                    dtype=torch.bfloat16, device=device)
    ]
    
    edge_case_names = [
        "Zero input",
        "Small values (1e-6)",
        "Large values (1e6)",
        "Mixed signs",
        "NaN/Inf values"
    ]
    
    results = []
    
    for i, (x, name) in enumerate(zip(edge_cases, edge_case_names)):
        print(f"  Testing: {name}")
        
        # Create layers
        standard_linear = torch.nn.Linear(64, 32, bias=True, 
                                         device=device, dtype=torch.bfloat16)
        mxfp4_linear = MXFP4Linear(64, 32, bias=True, 
                                   device=device, dtype=torch.bfloat16)
        
        # Copy weights
        with torch.no_grad():
            mxfp4_linear.weight.copy_(standard_linear.weight)
            mxfp4_linear.bias.copy_(standard_linear.bias)
        
        # Forward pass
        with torch.no_grad():
            try:
                standard_output = standard_linear(x)
                mxfp4_output = mxfp4_linear(x)
                
                # Check for NaN/Inf in outputs
                standard_has_nan = torch.isnan(standard_output).any().item()
                standard_has_inf = torch.isinf(standard_output).any().item()
                mxfp4_has_nan = torch.isnan(mxfp4_output).any().item()
                mxfp4_has_inf = torch.isinf(mxfp4_output).any().item()
                
                # Calculate difference (handle NaN/Inf carefully)
                if standard_has_nan or mxfp4_has_nan:
                    diff = float('nan')
                    passed = standard_has_nan == mxfp4_has_nan
                elif standard_has_inf or mxfp4_has_inf:
                    diff = float('nan')
                    passed = standard_has_inf == mxfp4_has_inf
                else:
                    diff = torch.max(torch.abs(standard_output - mxfp4_output)).item()
                    passed = diff <= 1e-3
                
                results.append({
                    'name': name,
                    'passed': passed,
                    'diff': diff,
                    'standard_has_nan': standard_has_nan,
                    'standard_has_inf': standard_has_inf,
                    'mxfp4_has_nan': mxfp4_has_nan,
                    'mxfp4_has_inf': mxfp4_has_inf
                })
                
                status = "✓" if passed else "✗"
                print(f"    {status} Diff: {diff:.6f}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                results.append({
                    'name': name,
                    'passed': False,
                    'error': str(e)
                })
    
    # Summary
    passed_count = sum(1 for r in results if r.get('passed', False))
    total_count = len(results)
    
    print(f"\n  Edge case summary: {passed_count}/{total_count} passed")
    
    return results


def simple_benchmark(input_size: int, output_size: int, batch_size: int, 
                    num_iterations: int = 100, device=None):
    """
    Simple benchmark comparing standard vs MXFP4 linear layers.
    
    Args:
        input_size: Input dimension
        output_size: Output dimension  
        batch_size: Batch size
        num_iterations: Number of iterations for benchmarking
        device: Device to run on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Benchmarking on {device}")
    print(f"Input: {batch_size}x{input_size} -> {output_size}")
    print(f"Iterations: {num_iterations}")
    
    # Create test input
    x = torch.randn(batch_size, input_size, dtype=torch.bfloat16, device=device)
    
    # Create layers
    standard_linear = torch.nn.Linear(input_size, output_size, bias=True, 
                                     device=device, dtype=torch.bfloat16)
    mxfp4_linear = MXFP4Linear(input_size, output_size, bias=True, 
                               device=device, dtype=torch.bfloat16)
    
    # Copy weights for fair comparison
    with torch.no_grad():
        mxfp4_linear.weight.copy_(standard_linear.weight)
        if standard_linear.bias is not None:
            mxfp4_linear.bias.copy_(standard_linear.bias)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = standard_linear(x)
            _ = mxfp4_linear(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark standard
    print("Benchmarking standard linear layer...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = standard_linear(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    standard_time = time.time() - start_time
    
    # Benchmark MXFP4
    print("Benchmarking MXFP4 linear layer...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = mxfp4_linear(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mxfp4_time = time.time() - start_time
    
    # Calculate metrics
    theoretical_flops = 2 * batch_size * input_size * output_size
    theoretical_tflops = theoretical_flops / 1e12
    
    standard_tflops = (theoretical_tflops * num_iterations) / standard_time
    mxfp4_tflops = (theoretical_tflops * num_iterations) / mxfp4_time
    
    speedup = standard_time / mxfp4_time
    
    # Print results
    print(f"\nResults:")
    print(f"  Standard: {standard_time:.4f}s, {standard_tflops:.3f} TFLOPs")
    print(f"  MXFP4:    {mxfp4_time:.4f}s, {mxfp4_tflops:.3f} TFLOPs")
    print(f"  Speedup:  {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"🚀 MXFP4 is {speedup:.2f}x faster!")
    else:
        print(f"⚠️  MXFP4 is {1/speedup:.2f}x slower")
    
    return {
        'standard_time': standard_time,
        'mxfp4_time': mxfp4_time,
        'speedup': speedup,
        'standard_tflops': standard_tflops,
        'mxfp4_tflops': mxfp4_tflops
    }


def benchmark_different_sizes():
    """Benchmark different layer sizes."""
    print("=== Benchmarking Different Layer Sizes ===")
    
    # Test configurations: (input_size, output_size, batch_size)
    configs = [
        (256, 5120, 1),
        (5120, 5120, 1),
        (5120, 30720, 1),
        (4096, 5120, 1),
        (5120, 5120, 1),
        (5120, 5120, 1),
        (5120, 13824, 1),
        (13824, 5120, 1)
    ]
    
    results = []
    
    for input_size, output_size, batch_size in configs:
        print(f"\n--- {input_size}x{output_size}, batch={batch_size} ---")
        try:
            result = simple_benchmark(input_size, output_size, batch_size, num_iterations=50)
            result.update({
                'input_size': input_size,
                'output_size': output_size,
                'batch_size': batch_size
            })
            results.append(result)
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    # Print summary
    if results:
        print(f"\n=== Summary ===")
        print(f"{'Config':<20} {'Speedup':<10}")
        print("-" * 30)
        
        total_speedup = 0
        for result in results:
            config = f"{result['input_size']}x{result['output_size']}x{result['batch_size']}"
            speedup = result['speedup']
            print(f"{config:<20} {speedup:<10.2f}")
            total_speedup += speedup
        
        avg_speedup = total_speedup / len(results)
        print("-" * 30)
        print(f"{'AVERAGE':<20} {avg_speedup:<10.2f}")
        
        if avg_speedup > 1.0:
            print(f"\n🚀 MXFP4 provides {avg_speedup:.2f}x average speedup!")
        else:
            print(f"\n⚠️  MXFP4 shows {avg_speedup:.2f}x average performance")


def run_full_test_suite(device=None):
    """
    Run the full test suite including correctness and performance tests.
    
    Args:
        device: Device to run on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("MXFP4 FULL TEST SUITE")
    print("=" * 60)
    
    # 1. Correctness tests
    print("\n1. CORRECTNESS TESTS")
    print("-" * 30)
    
    # Test basic correctness
    correctness_result = test_correctness(512, 512, 16, num_tests=20, device=device)
    
    # Test edge cases
    edge_case_results = test_edge_cases(device)
    
    # 2. Performance tests
    print("\n2. PERFORMANCE TESTS")
    print("-" * 30)
    
    # Test different sizes
    benchmark_different_sizes()
    
    # 3. Summary
    print("\n3. FINAL SUMMARY")
    print("-" * 30)
    
    correctness_passed = correctness_result['all_passed']
    edge_cases_passed = all(r.get('passed', False) for r in edge_case_results)
    
    print(f"Correctness tests: {'✓ PASSED' if correctness_passed else '✗ FAILED'}")
    print(f"Edge case tests:  {'✓ PASSED' if edge_cases_passed else '✗ FAILED'}")
    
    if correctness_passed and edge_cases_passed:
        print("\n🎉 ALL TESTS PASSED! MXFP4 is working correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED! Check the results above.")
    
    return {
        'correctness_passed': correctness_passed,
        'edge_cases_passed': edge_cases_passed,
        'correctness_result': correctness_result,
        'edge_case_results': edge_case_results
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MXFP4 vs standard linear layers")
    parser.add_argument("--input-size", type=int, default=2880, help="Input dimension")
    parser.add_argument("--output-size", type=int, default=2880, help="Output dimension")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--all-sizes", action="store_true", help="Benchmark all sizes")
    parser.add_argument("--correctness-only", action="store_true", help="Run only correctness tests")
    parser.add_argument("--full-suite", action="store_true", help="Run full test suite")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Tolerance for correctness tests")
    
    args = parser.parse_args()
    
    print("MXFP4 vs Standard Linear Layer Benchmark")
    print("=" * 50)
    
    # Check dependencies
    try:
        import triton_kernels
        print("✓ triton_kernels available")
    except ImportError:
        print("✗ triton_kernels not available. Please install it first.")
        return
    
    if args.full_suite:
        run_full_test_suite()
    elif args.correctness_only:
        # Run correctness tests only
        test_correctness(args.input_size, args.output_size, args.batch_size, 
                        num_tests=20, tolerance=args.tolerance)
        test_edge_cases()
    elif args.all_sizes:
        benchmark_different_sizes()
    else:
        # Run single benchmark with correctness check
        print("Running performance benchmark...")
        simple_benchmark(
            input_size=args.input_size,
            output_size=args.output_size,
            batch_size=args.batch_size,
            num_iterations=args.iterations
        )
        
        print("\nRunning correctness check...")
        test_correctness(
            input_size=args.input_size,
            output_size=args.output_size,
            batch_size=args.batch_size,
            tolerance=args.tolerance
        )


if __name__ == "__main__":
    main() 