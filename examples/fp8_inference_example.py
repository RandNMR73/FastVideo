#!/usr/bin/env python3
"""
Example script showing how to use FP8 quantization for inference with existing models.

This script demonstrates how to generate videos using FP8 quantization for improved
memory efficiency and faster inference on supported hardware (H100+).

Usage:
    python examples/fp8_inference_example.py
"""

import torch
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.layers.quantization.fp8_config import FP8Config

OUTPUT_PATH = "fp8_video_samples"
# OUTPUT_PATH = "video_samples"

def main():
    print("=== FP8 Quantization Video Generation Example ===")
    
    # Check GPU availability and FP8 support
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. FP8 quantization requires GPU.")
        return
    
    # Check GPU capability for FP8
    gpu_capability = torch.cuda.get_device_capability()
    if gpu_capability[0] < 9:  # H100 and newer
        print(f"Warning: GPU capability {gpu_capability} may not support FP8. Recommended: 9.0+")
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Capability: {gpu_capability}")
    
    # Prepare pipeline config with FP8 quantization
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    pipeline_config = PipelineConfig.from_pretrained(model_id)
    pipeline_config.dit_precision = "bf16"  # required for FP8
    pipeline_config.dit_config.quant_config = FP8Config()
    
    # Create VideoGenerator with FP8-compatible settings
    print("\nLoading model with FP8 quantization...")
    generator = VideoGenerator.from_pretrained(
        model_id,
        pipeline_config=pipeline_config,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,  # Keep DiT on GPU for FP8
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )
    
    print("FP8 configuration applied. Generating videos...")
    
    # Generate video with FP8 quantization
    print("\n=== Generating Video with FP8 Quantization ===")
    
    prompt1 = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    
    print(f"Prompt: {prompt1}")
    print("Generating video...")
    
    try:
        video1 = generator.generate_video(
            prompt1,
            output_path=OUTPUT_PATH,
            save_video=True,
        )
        print("✓ First video generated successfully with FP8 quantization!")
        
        # # Generate a second video to show the model can be reused
        # prompt2 = (
        #     "A majestic lion strides across the golden savanna, its powerful frame "
        #     "glistening under the warm afternoon sun. The tall grass ripples gently in "
        #     "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        #     "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        #     "cinematic."
        # )
        
        # print(f"\nGenerating second video...")
        # print(f"Prompt: {prompt2}")
        
        # video2 = generator.generate_video(
        #     prompt2,
        #     output_path=OUTPUT_PATH,
        #     save_video=True,
        # )
        # print("✓ Second video generated successfully with FP8 quantization!")
        
    except Exception as e:
        print(f"Error during video generation: {e}")
        print("Make sure you have the required deep_gemm library installed for FP8 support")
        return
    
    print(f"\n=== FP8 Quantization Summary ===")
    print(f"Videos saved to: {OUTPUT_PATH}")
    print("FP8 quantization benefits:")
    print("- ✓ Reduced memory usage compared to FP16/BF16")
    print("- ✓ Faster inference on supported hardware (H100+)")
    print("- ✓ Maintains model quality with proper scaling")
    print("\nFor optimal FP8 performance:")
    print("1. Use H100 or newer GPUs")
    print("2. Ensure deep_gemm library is installed")
    print("3. Keep models on GPU (avoid CPU offloading for quantized layers)")


if __name__ == "__main__":
    main() 