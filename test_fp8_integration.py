#!/usr/bin/env python3
"""
Test script to verify FP8 quantization integration.
"""

import torch
from fastvideo.layers.quantization.fp8_config import FP8Config, FP8QuantizeMethod
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization import get_quantization_config
import time

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
            bias=True,
            quant_config=fp8_config,
            params_dtype=torch.bfloat16
        )

        linear_unquant = ReplicatedLinear(
            input_size=128,
            output_size=256,
            bias=True,
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

def main():
    """Run all tests."""
    print("FP8 Quantization Integration Tests")
    print("=" * 40)
    
    tests = [
        test_fp8_registration,
        test_fp8_quant_method,
        test_fp8_forward_pass,
        test_model_integration,
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

if __name__ == "__main__":
    main() 