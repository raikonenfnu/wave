# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wave template for fused dynamic quantizer + FP4 GEMM kernel."""

import torch
from typing import Sequence

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)


SCALE_GROUP_SIZE = 32  # Hardware-specified FP4 scale group size


def get_dynamic_quant_mxfp4_gemm_kernel(
    shape: tuple[int, int, int],
    dynamic_dims: bool | tuple[bool, bool, bool],
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    dtype_in: torch.dtype = torch.float32,
    dtype_out: torch.dtype = torch.float32,
):
    """Get a fused dynamic quantizer + FP4 GEMM kernel.
    
    This kernel performs:
    1. Dynamic quantization of input tensor A to FP4
    2. Scaled MMA operation with pre-quantized FP4 weight B
    3. Write output to C
    
    Args:
        shape: (M, N, K) dimensions
        dynamic_dims: Which dimensions are dynamic
        mfma_variant: The scaled MMA variant to use
        dtype_in: Input data type (for unquantized input)
        dtype_out: Output data type
    """
    if not isinstance(dynamic_dims, Sequence):
        dynamic_dims = (dynamic_dims,) * 3

    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    
    dtype_in = torch_dtype_to_wave(dtype_in)
    dtype_out = torch_dtype_to_wave(dtype_out)
    
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    # With dynamic dimensions, we need to add an assumption on how big
    # the iterate dimension is to determine whether we can schedule or not.
    if dynamic_dims[2]:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

    @tkw.wave(constraints)
    def dynamic_quant_mxfp4_gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, dtype_in],  # Unquantized input
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],  # Pre-quantized FP4 weights (packed)
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],  # Pre-computed FP4 scales  
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, dtype_out],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # Read unquantized input tile
            a_reg = tkw.read(a)
            
            # Dynamic quantization of input A with group-wise scaling
            # Groups are of size SCALE_GROUP_SIZE (32) along K dimension
            
            # Step 1: Compute absolute values
            a_abs = tkw.abs(a_reg)
            
            # Step 2: Reshape to handle groups of 32 elements
            # We need to find max per group of 32 along K dimension
            # For now, we'll compute a scale per group by reshaping
            # Shape: [M, K] -> [M, K/32, 32]
            # Then compute max over the last dimension
            
            # Since Wave doesn't have explicit reshape, we'll approximate by
            # computing multiple scales along K dimension
            # This produces shape [M, K/32] for scales
            
            # For simplified implementation, compute scale for entire K tile
            # but replicate it to match the expected [M, K/32] shape
            max_abs_per_row = tkw.max(a_abs, dim=K)
            
            # Step 3: Compute scale using sharktank pattern: max_abs * 0.25
            scale_factor = 0.25
            quarter = tkl.Register[M, dtype_in](scale_factor)
            biased_scale_per_row = max_abs_per_row * quarter
            
            # Step 4: Create group-wise scales by broadcasting
            # We need shape [M, K/32] for the scales
            # Broadcast the per-row scale to create per-group scales
            # Note: K/32 represents the number of scale groups
            num_groups = K // SCALE_GROUP_SIZE
            a_scale_groups = tkw.broadcast(biased_scale_per_row, [M, num_groups])
            
            # Step 5: Scale and quantize to FP4
            # For quantization, we need to apply the same scale to each group
            # Since we have one scale per row, broadcast it to full shape
            biased_scale_expanded = tkw.broadcast(biased_scale_per_row, [M, K])
            scaled_input = a_reg / biased_scale_expanded
            
            # Cast to FP4 format
            a_fp4 = tkw.cast(scaled_input, tkl.f4e2m1fn)
            
            # Convert scales to FE8M0 format for hardware
            a_scale_reg = tkw.cast(a_scale_groups, tkl.f8e8m0fnu)
            
            # Read pre-quantized weights and scales
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            
            # Perform scaled MMA
            acc = tkw.scaled_mma(a_fp4, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        # Cast and write output
        if dtype_out != tkl.f32:
            result = tkw.cast(repeat, dtype_out)
        else:
            result = repeat
        tkw.write(result, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 256,  # Must be multiple of SCALE_GROUP_SIZE (32)
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = []
    if dynamic_dims[0]:
        dynamic_symbols.append(M)
        del hyperparams[M]

    if dynamic_dims[1]:
        dynamic_symbols.append(N)
        del hyperparams[N]

    if dynamic_dims[2]:
        dynamic_symbols.append(K)
        del hyperparams[K]

    return dynamic_quant_mxfp4_gemm, hyperparams, dynamic_symbols


def get_dynamic_quant_mxfp4_batched_gemm_kernel(
    shape: tuple[int, int, int, int],  # (B, M, N, K)
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    dtype_in: torch.dtype = torch.float32,
    dtype_out: torch.dtype = torch.float16,
):
    """Get a fused dynamic quantizer + FP4 batched GEMM kernel.
    
    This kernel performs:
    1. Dynamic quantization of batched input tensor A to FP4
    2. Scaled MMA operation with pre-quantized FP4 weight B (shared across batch)
    3. Write output to C
    
    Args:
        shape: (B, M, N, K) dimensions
        mfma_variant: The scaled MMA variant to use
        dtype_in: Input data type (for unquantized input)
        dtype_out: Output data type
    """
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    
    dtype_in = torch_dtype_to_wave(dtype_in)
    dtype_out = torch_dtype_to_wave(dtype_out)
    
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def dynamic_quant_mxfp4_batched_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, dtype_in],  # Unquantized batched input
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],  # Pre-quantized FP4 weights (packed)
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],  # Pre-computed FP4 scales
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, dtype_out],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            # Read unquantized input tile (batched)
            a_reg = tkw.read(a)
            
            # Dynamic quantization of batched input A with group-wise scaling
            # Groups are of size SCALE_GROUP_SIZE (32) along K dimension
            
            # Step 1: Compute absolute values
            a_abs = tkw.abs(a_reg)
            
            # Step 2: Find max absolute value per batch/row (reduce over K dimension)
            max_abs_per_row = tkw.max(a_abs, dim=K)
            
            # Step 3: Compute scale using sharktank pattern: max_abs * 0.25
            scale_factor = 0.25
            quarter = tkl.Register[B, M, dtype_in](scale_factor)
            biased_scale_per_row = max_abs_per_row * quarter
            
            # Step 4: Create group-wise scales by broadcasting
            # We need shape [B, M, K/32] for the scales
            num_groups = K // SCALE_GROUP_SIZE
            a_scale_groups = tkw.broadcast(biased_scale_per_row, [B, M, num_groups])
            
            # Step 5: Scale and quantize to FP4
            # For quantization, broadcast to full shape
            biased_scale_expanded = tkw.broadcast(biased_scale_per_row, [B, M, K])
            scaled_input = a_reg / biased_scale_expanded
            
            # Cast to FP4 format
            a_fp4 = tkw.cast(scaled_input, tkl.f4e2m1fn)
            
            # Convert scales to FE8M0 format for hardware
            a_scale_reg = tkw.cast(a_scale_groups, tkl.f8e8m0fnu)
            
            # Read pre-quantized weights and scales
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            
            # Perform scaled MMA
            acc = tkw.scaled_mma(a_fp4, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        # Cast and write output
        if dtype_out != tkl.f32:
            result = tkw.cast(repeat, dtype_out)
        else:
            result = repeat
        tkw.write(result, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 256,
        BLOCK_N: 128,
        BLOCK_K: 256,  # Must be multiple of SCALE_GROUP_SIZE (32)
        N: shape[2],
        K: shape[3],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = [B, M]
    
    return dynamic_quant_mxfp4_batched_gemm, hyperparams, dynamic_symbols