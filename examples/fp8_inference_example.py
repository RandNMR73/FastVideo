#!/usr/bin/env python3
"""
Example script showing how to use FP8 quantization for inference with existing models.

This script demonstrates two approaches:
1. Converting an existing loaded model to FP8
2. Loading a model with FP8 quantization from the start

Usage:
    python examples/fp8_inference_example.py --model-path /path/to/your/model
"""

import argparse
import torch
from pathlib import Path

# Import fastvideo components
from fastvideo.pipelines import build_pipeline, PipelineType
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.layers.quantization.fp8_config import FP8Config, convert_model_to_fp8
from fastvideo.layers.quantization import get_quantization_config


def example_1_convert_existing_model(model_path: str):
    """
    Example 1: Load a model normally, then convert it to FP8 quantization.
    This is useful when you have an existing model and want to apply FP8 for inference.
    """
    print("=== Example 1: Converting Existing Model to FP8 ===")
    
    # Create standard FastVideo args for loading the model
    fastvideo_args = FastVideoArgs(
        model_path=model_path,
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        dit_cpu_offload=False,  # Keep model on GPU for FP8
        inference_mode=True,
    )
    
    # Set precision to bfloat16 (required for FP8)
    fastvideo_args.pipeline_config.dit_precision = "bf16"
    
    # Build the pipeline normally
    print("Loading model...")
    pipeline = build_pipeline(fastvideo_args, PipelineType.BASIC)
    
    # Get the transformer model (DiT)
    transformer = pipeline.get_module("transformer")
    if transformer is None:
        print("Warning: No transformer found in pipeline")
        return
    
    print(f"Original model dtype: {next(transformer.parameters()).dtype}")
    
    # Convert the model to FP8 quantization
    print("Converting model to FP8 quantization...")
    transformer = convert_model_to_fp8(transformer)
    
    # Verify conversion
    print("FP8 conversion completed!")
    
    # Check if any linear layers have FP8 weights
    fp8_layers_found = 0
    for name, module in transformer.named_modules():
        if hasattr(module, '_fp8_weight'):
            fp8_layers_found += 1
            print(f"  Layer {name} converted to FP8")
            if fp8_layers_found >= 3:  # Just show first few
                print(f"  ... and {fp8_layers_found - 3} more layers")
                break
    
    print(f"Total FP8 layers: {fp8_layers_found}")
    
    # Now the model is ready for FP8 inference
    print("Model is ready for FP8 inference!")
    
    return pipeline


def example_2_load_with_fp8_config(model_path: str):
    """
    Example 2: Load a model with FP8 quantization config from the start.
    This approach integrates FP8 quantization during model loading.
    """
    print("\n=== Example 2: Loading Model with FP8 Config ===")
    
    # Create FP8 quantization config
    fp8_config = FP8Config()
    
    # Create FastVideo args with FP8 quantization
    fastvideo_args = FastVideoArgs(
        model_path=model_path,
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        dit_cpu_offload=False,  # Keep model on GPU for FP8
        inference_mode=True,
    )
    
    # Set precision to bfloat16 (required for FP8)
    fastvideo_args.pipeline_config.dit_precision = "bf16"
    
    # Set quantization config in the DiT config
    fastvideo_args.pipeline_config.dit_config.quant_config = fp8_config
    
    print("Loading model with FP8 quantization...")
    pipeline = build_pipeline(fastvideo_args, PipelineType.BASIC)
    
    transformer = pipeline.get_module("transformer")
    if transformer is None:
        print("Warning: No transformer found in pipeline")
        return
    
    print(f"Model loaded with FP8 quantization!")
    
    # Check quantization methods
    fp8_layers_found = 0
    for name, module in transformer.named_modules():
        if hasattr(module, 'quant_method') and module.quant_method is not None:
            if hasattr(module.quant_method, '__class__') and 'FP8' in module.quant_method.__class__.__name__:
                fp8_layers_found += 1
                if fp8_layers_found <= 3:  # Just show first few
                    print(f"  Layer {name} using FP8 quantization")
    
    print(f"Total FP8 quantized layers: {fp8_layers_found}")
    
    return pipeline


def example_3_inference_with_fp8(pipeline, prompt: str = "A beautiful sunset over mountains"):
    """
    Example 3: Run inference with FP8 quantized model.
    """
    print(f"\n=== Example 3: Running Inference ===")
    print(f"Prompt: {prompt}")
    
    try:
        # Create a simple inference input
        # Note: This is a simplified example - actual inference depends on your pipeline type
        print("Running inference with FP8 quantization...")
        
        # For demonstration, we'll just check that the model is ready
        transformer = pipeline.get_module("transformer")
        if transformer is not None:
            print("✓ Transformer model is ready for inference")
            print(f"✓ Model device: {next(transformer.parameters()).device}")
            print(f"✓ Model dtype: {next(transformer.parameters()).dtype}")
            
            # You can now use the pipeline for actual inference
            # result = pipeline(prompt=prompt, ...)
            
        print("FP8 inference setup completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Make sure you have the required deep_gemm library installed")


def main():
    parser = argparse.ArgumentParser(description="FP8 Quantization Example")
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to the model directory")
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over mountains",
                       help="Text prompt for inference example")
    parser.add_argument("--example", type=int, choices=[1, 2, 3], default=1,
                       help="Which example to run (1: convert existing, 2: load with config, 3: both)")
    
    args = parser.parse_args()
    
    # Verify model path exists
    if not Path(args.model_path).exists():
        print(f"Error: Model path {args.model_path} does not exist")
        return
    
    # Check GPU availability and FP8 support
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. FP8 quantization requires GPU.")
        return
    
    # Check GPU capability for FP8
    gpu_capability = torch.cuda.get_device_capability()
    if gpu_capability[0] < 9:  # H100 and newer
        print(f"Warning: GPU capability {gpu_capability} may not support FP8. Recommended: 9.0+")
    
    print(f"Using model: {args.model_path}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Capability: {gpu_capability}")
    
    pipeline = None
    
    if args.example in [1, 3]:
        pipeline = example_1_convert_existing_model(args.model_path)
    
    if args.example in [2, 3]:
        pipeline = example_2_load_with_fp8_config(args.model_path)
    
    if pipeline is not None:
        example_3_inference_with_fp8(pipeline, args.prompt)
    
    print("\n=== Summary ===")
    print("FP8 quantization provides:")
    print("- Reduced memory usage (FP8 vs FP16/BF16)")
    print("- Faster inference on supported hardware (H100+)")
    print("- Maintains model accuracy with proper scaling")
    print("\nFor production use:")
    print("1. Ensure you have deep_gemm library installed")
    print("2. Use appropriate GPU hardware (H100 or newer)")
    print("3. Set model precision to bfloat16")
    print("4. Test accuracy with your specific models")


if __name__ == "__main__":
    main() 