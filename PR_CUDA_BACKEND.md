# CUDA Backend Support for flux2.c

## Summary

This PR adds NVIDIA CUDA GPU acceleration to flux2.c, following the same pattern as the existing Metal/MPS backend. It enables `make cuda` for Linux users with NVIDIA GPUs.

## Changes

### New Files
- `flux_cuda.h` - C header with CUDA function declarations (matches flux_metal.h API)
- `flux_cuda.cu` - CUDA implementation with cuBLAS and custom kernels

### Modified Files
- `Makefile` - Added `make cuda` target with auto-detection
- `flux_kernels.c` - Added CUDA dispatch in matrix operations

## Features

### cuBLAS Matrix Operations
- `flux_cuda_sgemm` - General matrix multiplication via cuBLAS
- `flux_cuda_sgemm_bf16` - BF16 weight support (converts to F32)
- `flux_cuda_sgemm_batch` - Batched matrix multiplication

### Custom CUDA Kernels
- `k_silu` / `k_silu_mul` - SiLU activation (SwiGLU)
- `k_gelu` - GELU activation
- `k_rms_norm` - RMSNorm with parallel reduction
- `k_softmax` - Row-wise softmax with shared memory
- `k_qk_rms_norm` - QK normalization for attention
- `k_adaln_norm` - AdaLN modulation
- `k_rope_2d` - 2D Rotary Position Embeddings
- Element-wise: `k_add`, `k_mul`, `k_scale`

### Makefile Integration
```makefile
make cuda       # Build with CUDA backend (auto-detects nvcc)
make generic    # Pure C, no dependencies
make blas       # BLAS acceleration
make mps        # Apple Silicon Metal (macOS only)
```

## Requirements

- CUDA Toolkit 11.0+ (tested path: `/usr/local/cuda`)
- cuBLAS library
- OpenBLAS (for CPU fallback)
- NVIDIA GPU with compute capability 5.0+

## Architecture Support

The Makefile auto-detects GPU architecture. Override with:
```bash
CUDA_ARCH=sm_86 make cuda   # For RTX 30xx
CUDA_ARCH=sm_89 make cuda   # For RTX 40xx
CUDA_ARCH=sm_120 make cuda  # For Blackwell (RTX 50xx)
```

## Design Decisions

1. **Standalone implementation** - No GGML dependency, following antirez's philosophy
2. **Same API as Metal** - `flux_cuda_*` mirrors `flux_metal_*` functions
3. **Conditional compilation** - `#ifdef USE_CUDA` guards all CUDA code
4. **Graceful fallback** - Returns 0 from init if no GPU, falls back to CPU/BLAS
5. **TF32 enabled** - Uses Tensor Cores on Ampere+ for ~2x matmul speedup

## TODO (Future Improvements)

- [ ] Flash Attention kernel for memory efficiency
- [ ] im2col + cuBLAS for conv2d
- [ ] Persistent GPU memory pool (reduce alloc overhead)
- [ ] Multi-GPU support
- [ ] cuBLAS batched GEMM for attention

## Testing

```bash
# Build
make cuda

# Run inference
./flux -d flux-klein-model -p "a fluffy cat" -o cat.png -v
```

## Credits

- Inspired by [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) GGML CUDA backend
- Following the minimalist philosophy of flux2.c by @antirez
