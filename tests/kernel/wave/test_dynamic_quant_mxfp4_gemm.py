"""Test for fused dynamic quantizer + FP4 GEMM kernel."""

import torch
import pytest

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_randint,
    device_tensor,
    device_zeros,
)
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.templates.dynamic_quant_mxfp4_gemm import (
    get_dynamic_quant_mxfp4_gemm_kernel,
    get_dynamic_quant_mxfp4_batched_gemm_kernel,
    SCALE_GROUP_SIZE,
)

try:
    from .common.utils import param_bool, require_e2e, require_cdna4
except ImportError:
    # Running as standalone script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    param_bool = lambda *args, **kwargs: lambda x: x  # No-op decorator
    require_e2e = lambda x: x  # No-op decorator when running standalone
    require_cdna4 = lambda x: x  # No-op decorator when running standalone


def mxfp4_to_f32(x):
    """Convert packed FP4 values to float32."""
    # 2 because we pack fp4 in uint8
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = device_tensor(mxfp4_list, dtype=torch.float32)
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    """Convert E8M0 values to float32."""
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def dynamic_quantize_to_fp4_torch(values):
    """Reference implementation for dynamic FP4 quantization.
    
    Simplified version using per-row quantization with max_abs * 0.25 scaling.
    """
    # Get dimensions
    orig_shape = values.shape
    values = values.view(-1, orig_shape[-1])  # Flatten all but last dim
    batch_size, k = values.shape
    
    # Find max absolute value per row
    max_abs = torch.max(torch.abs(values), dim=1, keepdim=True)[0]
    
    # Compute scale using sharktank pattern: max_abs * 0.25
    scales_float = max_abs * 0.25
    scales_float = scales_float.clamp(min=torch.finfo(torch.float32).eps)
    
    # Quantize to FP4
    scaled_values = values / scales_float
    
    # Clamp to reasonable range before FP4 conversion
    scaled_values = scaled_values.clamp(-6.0, 6.0)
    
    # Convert to FP4 indices (simplified - just find nearest)
    fp4_values = torch.zeros_like(scaled_values, dtype=torch.uint8)
    fp4_lookup = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    
    for i in range(scaled_values.numel()):
        val = scaled_values.view(-1)[i]
        sign = 1 if val >= 0 else -1
        abs_val = abs(val)
        
        # Find closest FP4 value
        diffs = torch.abs(fp4_lookup - abs_val)
        idx = torch.argmin(diffs)
        
        # Add sign bit if negative
        if sign < 0:
            idx = idx + 8
            
        fp4_values.view(-1)[i] = idx
    
    # Pack two FP4 values into one uint8
    fp4_packed = torch.zeros(batch_size, k // 2, dtype=torch.uint8)
    fp4_packed = fp4_values[:, ::2] | (fp4_values[:, 1::2] << 4)
    
    # Convert scales to E8M0 format (log2(scale) + 127) 
    scales_e8m0 = (torch.log2(scales_float.squeeze()) + 127).clamp(0, 255).to(torch.uint8)
    
    # Reshape outputs
    fp4_packed = fp4_packed.view(*orig_shape[:-1], k // 2)
    scales_e8m0 = scales_e8m0.view(*orig_shape[:-1])
    
    return fp4_packed, scales_e8m0


def torch_dynamic_quant_gemm_reference(a, w, w_scales):
    """Reference implementation for dynamic quantization + GEMM."""
    # Dynamic quantize input A (simplified per-row)
    a_fp4, a_scales = dynamic_quantize_to_fp4_torch(a)
    
    # Convert to float32 for computation
    a_f32 = mxfp4_to_f32(a_fp4)
    w_f32 = mxfp4_to_f32(w.T).T
    
    # Apply scales
    a_scales_f32 = e8m0_to_f32(a_scales)
    a_scales_f32 = a_scales_f32.unsqueeze(-1).expand(-1, a.shape[-1])
    a_f32 = a_f32 * a_scales_f32
    
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_scales_f32 = w_scales_f32.repeat_interleave(SCALE_GROUP_SIZE, dim=-1)
    w_f32 = w_f32 * w_scales_f32.T
    
    # Compute GEMM
    return torch.mm(a_f32, w_f32)


@require_e2e
@require_cdna4
@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (512, 768, 1024)])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.F32_16x16x128_F8F6F4,
    ],
)
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.NONE,
        SchedulingType.PREFETCH,
    ],
)
def test_dynamic_quant_mxfp4_gemm(
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
):
    """Test fused dynamic quantizer + FP4 GEMM kernel."""
    M, N, K = shape
    
    # Generate test data
    torch.manual_seed(42)
    
    # Unquantized input A (will be dynamically quantized)
    a = device_randn((M, K), dtype=torch.float32)
    
    # Pre-quantized weight B (packed FP4)
    w_low = device_randint(0, 16, (N, K // 2), dtype=torch.uint8)
    w_high = device_randint(0, 16, (N, K // 2), dtype=torch.uint8)
    w = w_low | (w_high << 4)
    
    # Weight scales (E8M0 format)
    w_scales = device_randint(124, 128, (N, K // SCALE_GROUP_SIZE), dtype=torch.uint8)
    
    # Get kernel
    kernel_func, hyperparams, dynamic_symbols = get_dynamic_quant_mxfp4_gemm_kernel(
        shape, 
        dynamic_dims=False,
        mfma_variant=mfma_variant,
        dtype_in=torch.float32,
        dtype_out=torch.float32,
    )
    
    # Compile kernel
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
        dynamic_symbols=dynamic_symbols,
    )
    options = set_default_run_config(options)
    compiled_kernel = wave_compile(options, kernel_func)
    
    # Prepare output buffer
    out = device_zeros(M, N, dtype=torch.float32)
    
    # Run kernel
    w_t = w.contiguous()
    w_scales_t = w_scales.contiguous()
    compiled_kernel(a, w_t, w_scales_t, out)
    
    # Compute reference
    torch_out = torch_dynamic_quant_gemm_reference(a, w, w_scales)
    
    # Compare results (with tolerance for quantization)
    torch.testing.assert_close(torch_out, out, rtol=0.1, atol=0.1, check_dtype=False)


@require_e2e
@require_cdna4
@pytest.mark.parametrize("batch", [4, 8])
@pytest.mark.parametrize("shape", [(512, 768, 1024)])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.F32_16x16x128_F8F6F4,
    ],
)
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.PREFETCH])
def test_dynamic_quant_mxfp4_batched_gemm(
    batch: int,
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
):
    """Test fused dynamic quantizer + FP4 batched GEMM kernel."""
    M, N, K = shape
    full_shape = (batch, M, N, K)
    
    # Generate test data
    torch.manual_seed(42)
    
    # Unquantized batched input A (will be dynamically quantized)
    a = device_randn((batch, M, K), dtype=torch.float32)
    
    # Pre-quantized weight B (packed FP4, shared across batch)
    w_low = device_randint(0, 16, (N, K // 2), dtype=torch.uint8)
    w_high = device_randint(0, 16, (N, K // 2), dtype=torch.uint8)
    w = w_low | (w_high << 4)
    
    # Weight scales (E8M0 format)
    w_scales = device_randint(124, 128, (N, K // SCALE_GROUP_SIZE), dtype=torch.uint8)
    
    # Get kernel
    kernel_func, hyperparams, dynamic_symbols = get_dynamic_quant_mxfp4_batched_gemm_kernel(
        full_shape,
        mfma_variant=mfma_variant,
        dtype_in=torch.float32,
        dtype_out=torch.float16,
    )
    
    # Compile kernel
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
        dynamic_symbols=dynamic_symbols,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        use_stride_cache_swizzle=True,
    )
    options = set_default_run_config(options)
    compiled_kernel = wave_compile(options, kernel_func)
    
    # Prepare output buffer
    out = device_zeros(batch, M, N, dtype=torch.float16)
    
    # Run kernel
    w_t = w.contiguous()
    w_scales_t = w_scales.contiguous()
    compiled_kernel(a, w_t, w_scales_t, out)
    
    # Compute reference (batch-wise)
    torch_out = torch.zeros(batch, M, N, dtype=torch.float32, device=a.device)
    for b in range(batch):
        torch_out[b] = torch_dynamic_quant_gemm_reference(a[b], w, w_scales)
    torch_out = torch_out.to(torch.float16)
    
    # Compare results (with tolerance for quantization)
    torch.testing.assert_close(torch_out, out, rtol=0.15, atol=0.15, check_dtype=False)
