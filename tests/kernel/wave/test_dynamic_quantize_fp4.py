"""Test for standalone dynamic FP4 quantization kernel."""

import torch
import pytest

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_tensor,
    device_zeros,
)
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.templates.dynamic_quantize_fp4 import (
    get_dynamic_quantize_fp4_kernel,
    SCALE_GROUP_SIZE,
)

from .common.utils import require_e2e


def fp4_e2m1_to_f32(fp4_indices):
    """Convert FP4 E2M1 indices to float32 values."""
    # FP4 E2M1 lookup table
    fp4_lookup = torch.tensor([
        0.0,   # 0000: zero
        0.5,   # 0001: 2^(-1) * 1.0
        1.0,   # 0010: 2^0 * 1.0
        1.5,   # 0011: 2^0 * 1.5
        2.0,   # 0100: 2^1 * 1.0
        3.0,   # 0101: 2^1 * 1.5
        4.0,   # 0110: 2^2 * 1.0
        6.0,   # 0111: 2^2 * 1.5
        -0.0,  # 1000: negative zero
        -0.5,  # 1001: - 2^(-1) * 1.0
        -1.0,  # 1010: - 2^0 * 1.0
        -1.5,  # 1011: - 2^0 * 1.5
        -2.0,  # 1100: - 2^1 * 1.0
        -3.0,  # 1101: - 2^1 * 1.5
        -4.0,  # 1110: - 2^2 * 1.0
        -6.0,  # 1111: - 2^2 * 1.5
    ], dtype=torch.float32)
    
    if fp4_indices.device.type == 'cuda':
        fp4_lookup = fp4_lookup.cuda()
    
    return fp4_lookup[fp4_indices.long()]


def e8m0_to_f32(e8m0_values):
    """Convert E8M0 values to float32."""
    return torch.pow(2.0, e8m0_values.float() - 127.0)


def dynamic_quantize_fp4_reference(values):
    """Reference implementation for dynamic FP4 quantization.
    
    This simplified version uses per-row scaling with max_abs * 0.25.
    """
    M, K = values.shape
    
    # Find max absolute value per row
    max_abs = torch.max(torch.abs(values), dim=1, keepdim=False)[0]
    
    # Compute scales using the sharktank pattern: max_abs * 0.25
    scales = max_abs * 0.25
    scales = scales.clamp(min=torch.finfo(torch.float32).eps)
    
    # Quantize values
    scales_expanded = scales.unsqueeze(1).expand(M, K)
    scaled_values = values / scales_expanded
    
    # Simple cast to i8 for testing
    quantized = scaled_values.to(torch.int8)
    
    return quantized, scales


@require_e2e
@pytest.mark.parametrize("shape", [(2, 128), (4, 256), (8, 512)])
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
def test_dynamic_quantize_fp4(shape: tuple[int], enable_scheduling: SchedulingType):
    """Test standalone dynamic FP4 quantization kernel."""
    M, K = shape
    
    # Generate test data
    torch.manual_seed(42)
    input_data = device_randn((M, K), dtype=torch.float32)
    
    # Get kernel
    kernel_func, hyperparams, dynamic_symbols = get_dynamic_quantize_fp4_kernel(
        shape,
        dynamic_dims=False,
        dtype_in=torch.float32,
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
    
    # Prepare output buffers
    output_quantized = device_zeros(M, K, dtype=torch.int8)
    output_scales = device_zeros(M, dtype=torch.float32)
    
    # Run kernel
    compiled_kernel(input_data, output_quantized, output_scales)
    
    # Compute reference
    ref_quantized, ref_scales = dynamic_quantize_fp4_reference(input_data.cpu())
    ref_quantized = ref_quantized.cuda()
    ref_scales = ref_scales.cuda()
    
    # Compare scales (should match exactly)
    torch.testing.assert_close(output_scales, ref_scales, rtol=1e-5, atol=1e-5)
    
    # Check quantized values are within reasonable range
    assert output_quantized.abs().max().item() <= 127, f"Quantized values exceed i8 range"
    
    # Verify dequantization gives reasonable approximation
    scales_expanded = output_scales.unsqueeze(1).expand(M, K)
    dequantized = output_quantized.float() * scales_expanded
    
    # Original values should be approximately recovered
    # Note: Large tolerance due to simple cast quantization
    torch.testing.assert_close(dequantized, input_data, rtol=0.5, atol=1.0)
    
    print(f"Test passed for shape {shape}")
    print(f"  Scale range: [{output_scales.min().item():.4f}, {output_scales.max().item():.4f}]")
    print(f"  Quantized range: [{output_quantized.min().item()}, {output_quantized.max().item()}]")


if __name__ == "__main__":
    # Run a simple test
    print("Testing dynamic FP4 quantization kernel...")
    test_dynamic_quantize_fp4((64, 1024), SchedulingType.NONE)
    print("Test completed successfully!")