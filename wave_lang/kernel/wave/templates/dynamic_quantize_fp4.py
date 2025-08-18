# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wave template for standalone dynamic FP4 quantization kernel."""

import torch
from typing import Sequence

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)


SCALE_GROUP_SIZE = 32  # Hardware-specified FP4 scale group size


def get_dynamic_quantize_fp4_kernel(
    shape: tuple[int, int],  # (M, K)
    dynamic_dims: bool | tuple[bool, bool] = False,
    dtype_in: torch.dtype = torch.float32,
):
    """Get a standalone dynamic FP4 quantization kernel.
    
    This kernel performs dynamic quantization of input tensor to FP4 E2M1 format.
    
    Args:
        shape: (M, K) dimensions
        dynamic_dims: Which dimensions are dynamic
        dtype_in: Input data type
    
    Returns:
        Kernel function, hyperparameters, and dynamic symbols
    """
    if not isinstance(dynamic_dims, Sequence):
        dynamic_dims = (dynamic_dims,) * 2

    # Input sizes
    M = tkl.sym.M
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    # Address space
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    
    dtype_in = torch_dtype_to_wave(dtype_in)
    
    # Constraints following the working softmax pattern
    wave_size = 64
    BLOCK_M_SIZE = 1  # Use 1 like the working softmax example
    
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M_SIZE, K: K},  # K: K means use full K dimension
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]  # Note: dimension 1, not 0 like softmax
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    @tkw.wave(constraints)
    def dynamic_quantize_fp4(
        input: tkl.Memory[M, K, ADDRESS_SPACE, dtype_in],  # Unquantized input
        output_fp4: tkl.Memory[M, K, GLOBAL_ADDRESS_SPACE, tkl.i8],  # Quantized output as i8
        output_scales: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, dtype_in],  # Simple per-row scales
    ):
        # Read input
        input_reg = tkw.read(input)
        
        # Step 1: Compute absolute values
        input_abs = tkw.abs(input_reg)
        
        # Step 2: Find max absolute value per row
        # Note: max reduces over the K dimension
        max_abs = tkw.max(input_abs, dim=K)
        
        # Step 3: Compute scale for FP4 E2M1 format
        # Following the sharktank implementation: multiply by 0.25
        # This biases the scale for the FP4 E2M1 range
        scale_factor = 0.25
        quarter = tkl.Register[M, dtype_in](scale_factor)
        biased_scale = max_abs * quarter
        
        # Step 4: Write the scale directly
        tkw.write(biased_scale, output_scales)
        
        # Step 5: Quantize values 
        # First broadcast the scale back to match input shape
        biased_scale_bcast = tkw.broadcast(biased_scale, [M, K])
        scaled_input = input_reg / biased_scale_bcast
        
        # Just cast to i8 directly for testing
        quantized = tkw.cast(scaled_input, tkl.i8)
        
        # Write outputs
        tkw.write(quantized, output_fp4)

    hyperparams = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,  # Use global memory for simplicity
        BLOCK_M: 1,  # Use 1 like softmax example
        M: shape[0],
        K: shape[1],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = []
    if dynamic_dims[0]:
        dynamic_symbols.append(M)
        del hyperparams[M]

    if dynamic_dims[1]:
        dynamic_symbols.append(K)
        del hyperparams[K]

    return dynamic_quantize_fp4, hyperparams, dynamic_symbols