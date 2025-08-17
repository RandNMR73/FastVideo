# FP8 Quantization Guide

This guide explains how to use FP8 quantization for inference with your existing models in FastVideo.

## Overview

FP8 quantization reduces memory usage and can accelerate inference on supported hardware (H100 and newer GPUs) while maintaining model accuracy through proper scaling.

## Prerequisites

1. **Hardware**: NVIDIA H100 or newer GPU (compute capability 9.0+)
2. **Software**:
   - PyTorch with FP8 support
   - `deep_gemm` library for FP8 GEMM operations
3. **Model precision**: Models should use bfloat16 precision

## Installation

Make sure you have the `deep_gemm` library installed:

```bash
# Install deep_gemm (replace with actual installation method)
pip install deep_gemm
```

## Usage Methods

### Method 1: Convert Existing Model (Recommended)

This approach loads your model normally, then converts it to FP8:

```python
from fastvideo.pipelines import build_pipeline, PipelineType
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.layers.quantization.fp8_config import convert_model_to_fp8

# Load model normally
fastvideo_args = FastVideoArgs(
    model_path="/path/to/your/model",
    num_gpus=1,
    dit_cpu_offload=False,  # Keep on GPU for FP8
    inference_mode=True,
)
fastvideo_args.pipeline_config.dit_precision = "bf16"  # Required for FP8

pipeline = build_pipeline(fastvideo_args, PipelineType.BASIC)

# Convert to FP8
transformer = pipeline.get_module("transformer")
transformer = convert_model_to_fp8(transformer)

# Now ready for FP8 inference
```

### Method 2: Load with FP8 Config

This approach integrates FP8 quantization during model loading:

```python
from fastvideo.layers.quantization.fp8_config import FP8Config

# Create FP8 config
fp8_config = FP8Config()

# Set up args with quantization
fastvideo_args = FastVideoArgs(
    model_path="/path/to/your/model",
    num_gpus=1,
    dit_cpu_offload=False,
    inference_mode=True,
)
fastvideo_args.pipeline_config.dit_precision = "bf16"
fastvideo_args.pipeline_config.dit_config.quant_config = fp8_config

pipeline = build_pipeline(fastvideo_args, PipelineType.BASIC)
```

## Example Script

Run the provided example script:

```bash
python examples/fp8_inference_example.py --model-path /path/to/your/model --example 1
```

Options:

- `--example 1`: Convert existing model to FP8
- `--example 2`: Load model with FP8 config
- `--example 3`: Run both examples

## Technical Details

### How It Works

1. **Weight Quantization**: Model weights are converted from BF16 to FP8 using block-wise quantization
2. **Activation Quantization**: Input activations are converted to FP8 per-token during forward pass
3. **Scaling**: Proper scaling factors maintain numerical accuracy
4. **GEMM**: FP8 matrix multiplications use optimized kernels from `deep_gemm`

### Memory Savings

- **Weights**: ~50% reduction (BF16 → FP8)
- **Activations**: ~50% reduction during computation
- **Overall**: Significant memory savings, especially for large models

### Performance

- **Speed**: Faster inference on H100+ hardware
- **Accuracy**: Minimal accuracy loss with proper scaling
- **Compatibility**: Works with existing model architectures

## Troubleshooting

### Common Issues

1. **GPU Compatibility Error**

   ```
   GPU capability X.X may not support FP8. Recommended: 9.0+
   ```

   **Solution**: Use H100 or newer GPU

2. **deep_gemm Not Found**

   ```
   ModuleNotFoundError: No module named 'deep_gemm'
   ```

   **Solution**: Install the deep_gemm library

3. **Wrong Input Dtype**

   ```
   only allow bf16 inputs to fp8 linear, got torch.float16
   ```

   **Solution**: Set model precision to bfloat16:

   ```python
   fastvideo_args.pipeline_config.dit_precision = "bf16"
   ```

4. **CUDA Out of Memory**
   - Ensure `dit_cpu_offload=False` when using FP8
   - FP8 quantization should reduce memory usage

### Debugging

Enable verbose logging to see which layers are converted:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Run conversion
transformer = convert_model_to_fp8(transformer)
```

## Performance Tuning

### Best Practices

1. **Keep Models on GPU**: Set `dit_cpu_offload=False`
2. **Use BF16 Precision**: Required for FP8 compatibility
3. **Batch Operations**: FP8 benefits from larger batch sizes
4. **Profile Memory**: Monitor GPU memory usage before/after conversion

### Benchmarking

Compare performance with and without FP8:

```python
import time
import torch

# Benchmark function
def benchmark_inference(model, input_tensor, num_runs=10):
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            output = model(input_tensor)
        torch.cuda.synchronize()

    end = time.time()
    return (end - start) / num_runs

# Compare models
fp16_time = benchmark_inference(original_model, test_input)
fp8_time = benchmark_inference(fp8_model, test_input)

print(f"FP16 time: {fp16_time:.4f}s")
print(f"FP8 time: {fp8_time:.4f}s")
print(f"Speedup: {fp16_time / fp8_time:.2f}x")
```

## Limitations

1. **Hardware**: Requires H100 or newer GPUs
2. **Precision**: Only works with bfloat16 input precision
3. **Library Dependency**: Requires `deep_gemm` for optimal performance
4. **Model Support**: Currently supports linear layers only

## Future Improvements

- Support for additional layer types
- Automatic mixed precision with FP8
- Model-specific quantization strategies
- Integration with model compilation

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Verify hardware and software requirements
3. Review the example script for proper usage patterns
