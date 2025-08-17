# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/character-ai/pipelining-sft/blob/main/models/deepseek_v3/fp8_layers_triton.py

import triton
import triton.language as tl
import torch
import deep_gemm
from deep_gemm import ceil_div
from typing import Tuple

@triton.jit
def _fp8_cast_kernel(
    x_ptr,                      # Input pointer
    out_ptr,                    # Output FP8 pointer  
    scale_ptr,                  # Scale pointer
    M,                          # Number of rows
    N,                          # Number of columns (original, unpadded)
    N_padded,                   # Number of columns after padding
    stride_x_m,                 # Stride for x in M dimension
    stride_x_n,                 # Stride for x in N dimension
    stride_out_m,               # Stride for output in M dimension
    stride_out_n,               # Stride for output in N dimension
    stride_scale_m,             # Stride for scale in M dimension
    stride_scale_n,             # Stride for scale in N dimension (between chunks)
    BLOCK_SIZE: tl.constexpr,   # Size of each chunk (128)
):
    # Get the row we're processing
    row_idx = tl.program_id(0)
    
    # Calculate number of chunks
    num_chunks = (N_padded + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Process each chunk in the row
    for chunk_idx in range(num_chunks):
        # Calculate the starting column for this chunk
        col_start = chunk_idx * BLOCK_SIZE
        
        # Create the column indices for this chunk
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid elements (not padding)
        mask = cols < N
        
        # Calculate pointers for this chunk
        x_ptrs = x_ptr + row_idx * stride_x_m + cols * stride_x_n
        out_ptrs = out_ptr + row_idx * stride_out_m + cols * stride_out_n
        
        # Load the chunk (use 0.0 for padding)
        x_chunk = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Compute absolute values
        x_abs = tl.abs(x_chunk)
        
        # Find maximum in this chunk
        # Note: tl.max returns a scalar when axis=0 is used on a 1D tensor
        amax = tl.max(x_abs, axis=0)
        
        # Clamp to avoid division by zero
        amax = tl.maximum(amax, 1e-4)
        
        # Compute and store scale (this is what gets stored, not used for scaling)
        scale = amax / 448.0
        scale_ptr_loc = scale_ptr + row_idx * stride_scale_m + chunk_idx * stride_scale_n
        tl.store(scale_ptr_loc, scale)
        
        # Scale the values (multiply by 448.0 / amax, not divide by scale!)
        scale_factor = 448.0 / amax
        x_scaled = x_chunk * scale_factor
        
        # Cast to FP8 (correct way in Triton)
        x_fp8 = x_scaled.to(tl.float8e4nv)
        
        # Store the FP8 values
        tl.store(out_ptrs, x_fp8, mask=mask)


def per_token_cast_to_fp8_triton(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Correct implementation of per-token FP8 casting that matches the original PyTorch version.
    
    Args:
        x: Input tensor of shape (M, N)
    
    Returns:
        x_fp8: FP8 quantized tensor of shape (M, N)
        scales: Scale factors of shape (M, num_chunks) where num_chunks = ceil(N/128)
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, N = x.shape
    
    # Calculate padding
    BLOCK_SIZE = 128
    pad_size = (BLOCK_SIZE - (N % BLOCK_SIZE)) % BLOCK_SIZE
    N_padded = N + pad_size
    num_chunks = N_padded // BLOCK_SIZE
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Allocate PADDED output tensor (like PyTorch version)
    x_fp8_padded = torch.empty((M, N_padded), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((M, num_chunks), dtype=torch.float32, device=x.device)
    
    # Launch kernel - one thread block per row
    grid = (M,)
    _fp8_cast_kernel[grid](
        x,
        x_fp8_padded,  # Use padded tensor
        scales,
        M,
        N,
        N_padded,
        x.stride(0),
        x.stride(1),
        x_fp8_padded.stride(0),
        x_fp8_padded.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,  # Tune based on your GPU
    )
    return x_fp8_padded[:, :N], scales


@triton.jit
def _fp8_per_block_cast_kernel(
    x_ptr,                      # Input pointer
    out_ptr,                    # Output FP8 pointer  
    scale_ptr,                  # Scale pointer
    M,                          # Number of rows (original)
    N,                          # Number of columns (original)
    M_padded,                   # Number of rows after padding to multiple of 128
    N_padded,                   # Number of columns after padding to multiple of 128
    stride_x_m,                 # Stride for x in M dimension
    stride_x_n,                 # Stride for x in N dimension
    stride_out_m,               # Stride for output in M dimension
    stride_out_n,               # Stride for output in N dimension
    stride_scale_m,             # Stride for scale in M dimension (block row)
    stride_scale_n,             # Stride for scale in N dimension (block col)
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    # Get the 128x128 block we're processing
    block_id_m = tl.program_id(0)
    block_id_n = tl.program_id(1)
    
    # Calculate the starting indices for this block
    m_start = block_id_m * BLOCK_M
    n_start = block_id_n * BLOCK_N
    
    # Initialize max value for this block
    block_max = 0.0
    
    # Process the block in smaller chunks to find the maximum
    # We'll process in 32x32 tiles for efficiency
    TILE_M: tl.constexpr = 32
    TILE_N: tl.constexpr = 32
    
    # First pass: find the maximum absolute value in the entire 128x128 block
    for tile_m in range(0, BLOCK_M, TILE_M):
        for tile_n in range(0, BLOCK_N, TILE_N):
            # Create row and column offsets for this tile
            rm = tl.arange(0, TILE_M)
            rn = tl.arange(0, TILE_N)
            
            # Calculate actual indices
            rows = m_start + tile_m + rm[:, None]
            cols = n_start + tile_n + rn[None, :]
            
            # Create mask for valid elements
            mask = (rows < M) & (cols < N)
            
            # Calculate pointers
            ptrs = x_ptr + rows * stride_x_m + cols * stride_x_n
            
            # Load tile data
            tile_data = tl.load(ptrs, mask=mask, other=0.0)
            
            # Find max absolute value in this tile
            tile_abs = tl.abs(tile_data)
            tile_max = tl.max(tile_abs)
            
            # Update block maximum
            block_max = tl.maximum(block_max, tile_max)
    
    # Clamp to avoid division by zero
    block_max = tl.maximum(block_max, 1e-4)
    
    # Compute scale factor
    scale = block_max / 448.0
    scale_factor = 448.0 / block_max
    
    # Store the scale for this block
    scale_ptr_loc = scale_ptr + block_id_m * stride_scale_m + block_id_n * stride_scale_n
    tl.store(scale_ptr_loc, scale)
    
    # Second pass: scale and convert to FP8
    for tile_m in range(0, BLOCK_M, TILE_M):
        for tile_n in range(0, BLOCK_N, TILE_N):
            # Create row and column offsets for this tile
            rm = tl.arange(0, TILE_M)
            rn = tl.arange(0, TILE_N)
            
            # Calculate actual indices
            rows = m_start + tile_m + rm[:, None]
            cols = n_start + tile_n + rn[None, :]
            
            # Create mask for valid elements
            mask = (rows < M) & (cols < N)
            
            # Calculate pointers
            in_ptrs = x_ptr + rows * stride_x_m + cols * stride_x_n
            out_ptrs = out_ptr + rows * stride_out_m + cols * stride_out_n
            
            # Load, scale, and convert tile data
            tile_data = tl.load(in_ptrs, mask=mask, other=0.0)
            tile_scaled = tile_data * scale_factor
            tile_fp8 = tile_scaled.to(tl.float8e4nv)
            
            # Store FP8 data
            tl.store(out_ptrs, tile_fp8, mask=mask)


def per_block_cast_to_fp8_triton(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton implementation of per-block FP8 casting that matches the original PyTorch version.
    
    Each 128x128 block gets its own scale factor based on the maximum absolute value
    in that block.
    
    Args:
        x: Input tensor of shape (M, N)
    
    Returns:
        x_fp8: FP8 quantized tensor of shape (M, N)
        scales: Scale factors of shape (num_block_rows, num_block_cols)
                where num_block_rows = ceil(M/128), num_block_cols = ceil(N/128)
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, N = x.shape
    
    # Calculate padded dimensions
    BLOCK_SIZE = 128
    M_padded = ceil_div(M, BLOCK_SIZE) * BLOCK_SIZE
    N_padded = ceil_div(N, BLOCK_SIZE) * BLOCK_SIZE
    num_block_rows = M_padded // BLOCK_SIZE
    num_block_cols = N_padded // BLOCK_SIZE
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Allocate output tensors
    x_fp8 = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((num_block_rows, num_block_cols), dtype=torch.float32, device=x.device)
    
    # Launch kernel - one thread block per 128x128 block
    grid = (num_block_rows, num_block_cols)
    _fp8_per_block_cast_kernel[grid](
        x,
        x_fp8,
        scales,
        M,
        N,
        M_padded,
        N_padded,
        x.stride(0),
        x.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        scales.stride(0),
        scales.stride(1),
        num_warps=8,
    )
    
    return x_fp8, scales
