#!/usr/bin/env python3
"""
Simple test runner for FP8 quantization tests.
This script can be run directly to test the FP8 quantization components.
"""

import sys
import os

# Add the parent directory to the path so we can import fastvideo modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

def main():
    """Run the FP8 quantization tests."""
    print("Running FP8 Quantization Tests...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    print("\n" + "="*50)
    
    # Run the tests
    test_file = os.path.join(os.path.dirname(__file__), "test_fp8_quantization.py")
    
    # Run with verbose output and show local variables on failures
    exit_code = pytest.main([
        test_file,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-s",  # Show print statements
        "--maxfail=5",  # Stop after 5 failures
    ])
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 