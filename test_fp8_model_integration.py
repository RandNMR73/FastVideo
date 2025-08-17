#!/usr/bin/env python3
"""
Test script to verify FP8 quantization integration with updated model structure.
"""

import torch
from fastvideo.layers.quantization.fp8_config import FP8Config
from fastvideo.configs.models.dits.base import DiTConfig

def test_config_flow():
    """Test that quantization config flows through the config hierarchy."""
    print("=== Testing Config Flow ===")
    
    try:
        # Create FP8 config
        fp8_config = FP8Config()
        print(f"✓ Created FP8 config: {fp8_config}")
        
        # Create DiT config with FP8 quantization
        dit_config = DiTConfig()
        dit_config.quant_config = fp8_config
        
        print(f"✓ Set quant_config in DiT config: {dit_config.quant_config}")
        print(f"✓ Quant config type: {type(dit_config.quant_config)}")
        
        # Verify the config is properly set
        if dit_config.quant_config is not None and dit_config.quant_config.get_name() == "fp8":
            print("✓ FP8 quantization config is properly set in DiT config")
            return True
        else:
            print("✗ FP8 quantization config is not properly set")
            return False
            
    except Exception as e:
        print(f"✗ Config flow test failed: {e}")
        return False

def test_quantization_registration():
    """Test that FP8 quantization is properly registered."""
    print("\n=== Testing Quantization Registration ===")
    
    try:
        from fastvideo.layers.quantization import get_quantization_config
        
        # Try to get the FP8 config
        fp8_config_cls = get_quantization_config("fp8")
        print(f"✓ FP8 config class: {fp8_config_cls}")
        
        # Create an instance
        fp8_config = fp8_config_cls()
        print(f"✓ FP8 config instance: {fp8_config}")
        print(f"✓ FP8 config name: {fp8_config.get_name()}")
        
        return True
    except Exception as e:
        print(f"✗ Quantization registration test failed: {e}")
        return False

def test_linear_layer_quantization():
    """Test that linear layers can use FP8 quantization."""
    print("\n=== Testing Linear Layer Quantization ===")
    
    try:
        from fastvideo.layers.linear import ReplicatedLinear
        
        # Create FP8 config
        fp8_config = FP8Config()
        
        # Create a linear layer with FP8 quantization
        linear_layer = ReplicatedLinear(
            input_size=128,
            output_size=256,
            bias=True,
            quant_config=fp8_config
        )
        
        print(f"✓ Created linear layer with FP8 quantization")
        print(f"✓ Quant method type: {type(linear_layer.quant_method).__name__}")
        
        # Check if the quantization method is FP8
        if "FP8" in linear_layer.quant_method.__class__.__name__:
            print("✓ Linear layer is using FP8 quantization")
            return True
        else:
            print(f"✗ Linear layer is not using FP8 quantization: {linear_layer.quant_method.__class__.__name__}")
            return False
            
    except Exception as e:
        print(f"✗ Linear layer quantization test failed: {e}")
        return False

def test_model_structure():
    """Test that the model structure supports quantization config."""
    print("\n=== Testing Model Structure ===")
    
    try:
        # Import the model classes to check their signatures
        from fastvideo.models.dits.wanvideo import WanTransformerBlock, WanTransformerBlock_VSA
        
        # Check if the classes accept quant_config parameter
        import inspect
        
        # Check WanTransformerBlock
        sig = inspect.signature(WanTransformerBlock.__init__)
        if 'quant_config' in sig.parameters:
            print("✓ WanTransformerBlock accepts quant_config parameter")
        else:
            print("✗ WanTransformerBlock does not accept quant_config parameter")
            return False
        
        # Check WanTransformerBlock_VSA
        sig = inspect.signature(WanTransformerBlock_VSA.__init__)
        if 'quant_config' in sig.parameters:
            print("✓ WanTransformerBlock_VSA accepts quant_config parameter")
        else:
            print("✗ WanTransformerBlock_VSA does not accept quant_config parameter")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Model structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("FP8 Quantization Model Integration Tests")
    print("=" * 50)
    
    tests = [
        test_config_flow,
        test_quantization_registration,
        test_linear_layer_quantization,
        test_model_structure,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! FP8 quantization integration is working correctly.")
        print("\nNext steps:")
        print("1. Run your video generation script with FP8 quantization")
        print("2. The quantization config should now flow through the model hierarchy")
        print("3. All linear layers should use FP8 quantization")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        
        if not results[0]:
            print("\n🔧 Fix needed: Config flow issue")
        if not results[1]:
            print("\n🔧 Fix needed: Quantization registration issue")
        if not results[2]:
            print("\n🔧 Fix needed: Linear layer quantization issue")
        if not results[3]:
            print("\n🔧 Fix needed: Model structure issue")

if __name__ == "__main__":
    main() 