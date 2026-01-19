# FLUX.2-klein C Implementation Notes

This file documents the implementation details of the FLUX.2-klein 4B image generation model in pure C.

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Inference Flow](#inference-flow)
3. [Performance Optimizations](#performance-optimizations)
4. [API Reference](#api-reference)
5. [Debugging and Verification](#debugging-and-verification)
6. [Work Log](#work-log)

---

## Model Architecture

### Transformer Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_size | 3072 | Main hidden dimension |
| num_heads | 24 | Attention heads |
| head_dim | 128 | Per-head dimension |
| mlp_hidden | 9216 | 3 × hidden |
| num_double_layers | 5 | Two-stream blocks |
| num_single_layers | 20 | Single-stream blocks |
| text_dim | 7680 | Text embedding dimension |
| latent_channels | 128 | VAE latent channels |
| rope_theta | 2000.0 | RoPE frequency base |
| axes_dim_rope | [32,32,32,32] | 4 axes, 128 total |

### Qwen3 Text Encoder Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_size | 2560 | 3 × 2560 = 7680 = text_dim |
| num_attention_heads | 32 | Query heads |
| num_key_value_heads | 8 | KV heads (GQA 4:1) |
| head_dim | 128 | Per-head dimension |
| intermediate_size | 9728 | FFN hidden |
| num_hidden_layers | 36 | Transformer layers |
| vocab_size | 151936 | Vocabulary size |
| rms_norm_eps | 1e-6 | RMSNorm epsilon |
| rope_theta | 1000000.0 | RoPE frequency base |

### RoPE Implementation

The model uses 4-axis RoPE (Rotary Position Embedding):
- **Axis 0 (T)**: dims 0-31, temporal
- **Axis 1 (H)**: dims 32-63, height
- **Axis 2 (W)**: dims 64-95, width
- **Axis 3 (L)**: dims 96-127, sequence length

**Image tokens**: Position IDs = (T=0, H=y, W=x, L=0)
- Only axes 1,2 rotate (based on spatial coordinates)

**Text tokens**: Position IDs = (T=0, H=0, W=0, L=seq_idx)
- Only axis 3 rotates (based on sequence position)

**Rotation formula** (per pair):
```c
out[0] = cos * x0 - sin * x1
out[1] = sin * x0 + cos * x1  // NOT cos*x1 + sin*x0
```

### Concatenation Order

Official Python concatenates as **[TEXT, IMAGE]** for Q, K, V:
```python
cat_k = [txt_k, img_k]
cat_v = [txt_v, img_v]
pe = [pe_txt, pe_img]
```

### Timestep Embedding

1. Input timestep scaled by 1000 (t=1.0 → 1000.0)
2. Sinusoidal embedding: 128 frequencies → 256 dims
3. Two-layer MLP: linear(256→3072) + SiLU + linear(3072→3072)

### AdaLN Modulation

- SiLU applied to t_emb **before** modulation projection
- Order: shift first, then scale: `out = (1 + scale) * norm(x) + shift`
- Double block: 6 params each for img/txt (shift1, scale1, gate1, shift2, scale2, gate2)
- Single block: 3 params (shift, scale, gate)

### Final Layer

- Uses AdaLayerNormContinuous (not RMSNorm)
- LayerNorm with elementwise_affine=False
- **Critical**: Projection output splits as (scale, shift) NOT (shift, scale)
- Formula: `out = (1 + scale) * LayerNorm(x) + shift`
- Linear projection to latent_channels

### VAE (AutoencoderKLFlux2)

- latent_channels: 32 (VAE internal)
- patch_size: 2×2
- Transformer outputs 128 channels = 32 latent × 2×2 patch
- **Unpacking**: [B, 128, H, W] → [B, 32, H×2, W×2]

---

## Inference Flow

### End-to-End Pipeline

```
1. Text Encoding (Qwen3)
   ├── Tokenize prompt with chat template
   ├── Run through 36 transformer layers
   └── Extract hidden states from layers [8, 17, 26]
       └── Output: [512, 7680] embeddings

2. Latent Initialization
   ├── Generate noise: [channels, H/8, W/8]
   └── Flatten to sequence: [H*W/64, 128]

3. Denoising Loop (4 steps for klein)
   ├── Compute timestep embedding
   ├── Apply RoPE to image positions
   └── For each step:
       ├── 5 double-stream blocks (parallel img/txt)
       ├── Concatenate streams
       ├── 20 single-stream blocks
       └── Final layer → predict noise

4. VAE Decode
   ├── Unpack: [128, H, W] → [32, H*2, W*2]
   └── Decode to RGB: [3, H*16, W*16]
```

### Text Encoder Details

**Tokenization**:
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

**Layer Extraction**:
- Python's `hidden_states[i]` = output after layer `i-1`
- Extract `hidden_states[9]`, `[18]`, `[27]` = after layers 8, 17, 26
- Stack and reshape: [3, 512, 2560] → [512, 7680]

### Double-Stream Block

```
Input: img_hidden [img_seq, 3072], txt_hidden [txt_seq, 3072]

1. AdaLN normalize both streams
2. QKV projection (separate for img/txt)
3. QK normalization (RMSNorm per-head)
4. Apply RoPE
5. Joint attention: Q_img attends to [K_txt, K_img]
6. Output projection + gating
7. FFN with SiLU-gated MLP
8. Residual connections

Output: img_hidden', txt_hidden'
```

### Single-Stream Block

```
Input: concatenated [txt_hidden, img_hidden]

1. AdaLN normalize
2. Fused QKV + MLP projection
3. Self-attention over full sequence
4. RoPE: text uses axis 3, image uses axes 1,2
5. Output projection + gating + FFN
6. Residual connection

Output: updated hidden states
```

---

## Performance Optimizations

### Current Performance (2024-01-19)

**Benchmark: 256×256, 4 steps, Apple M3 Max**

| Component | Time | Notes |
|-----------|------|-------|
| Model loading | ~5.2s | Disk I/O + weight caching |
| Text encoding | ~3.1s | GPU causal attention |
| Denoising | ~11-12s | GPU attention, GPU softmax |
| VAE decode | ~2.0s | CPU BLAS |
| **Total** | ~24s | |

**Comparison with PyTorch (bf16, MPS):**

| Size | C Implementation | PyTorch | Gap |
|------|-----------------|---------|-----|
| 256×256 | ~24s | ~3s | 8x |
| 512×512 | ~50s | ~5.4s | 9x |

### Optimization History

#### 1. BLAS for Linear Layers
- Replaced naive O(n³) loops with `cblas_sgemm`
- Uses Apple Accelerate (macOS) or OpenBLAS (Linux)
- **Speedup: ~30x** (32 GFLOPS → 989 GFLOPS)

#### 2. BLAS for Convolutions (VAE)
- Replaced 6-nested-loop conv with im2col + BLAS
- **Speedup: ~180x** (VAE decode: 18.6s → 0.1s)

#### 3. GPU-Accelerated Attention
- Batched all 24 heads into 2 GPU submissions
- Phase 1: Q @ K^T for all heads (single command buffer)
- Phase 2: Softmax (CPU or GPU)
- Phase 3: scores @ V for all heads (single command buffer)
- **Impact: ~50% denoising speedup**

#### 4. GPU Softmax Fusion
- Added Metal compute shader for softmax
- Eliminated CPU roundtrip in attention
- GPU Q@K → GPU softmax → GPU @V in single command buffer
- **Impact: ~39% total improvement**

#### 5. Text Encoder GPU FFN
- Lowered GPU threshold from 10M to 1M elements
- FFN projections (~3.5M elements) now use GPU
- **Impact: Text encoding 5.4s → 4.4s**

#### 6. Pre-allocated Attention Buffers
- Moved per-call mallocs to model struct
- Eliminated 32 heads × 36 layers = 1152 malloc/free pairs
- **Impact: Text encoding 4.4s → 4.1s**

#### 7. Strided BLAS Access
- Eliminated Q and output copies in text encoder attention
- BLAS `lda` parameter enables strided memory access
- K still needs transpose (different layout requirement)
- **Impact: Text encoding 4.1s → 3.9s**

#### 8. Fused GPU Causal Attention (Latest)
- New Metal kernel: `causal_attention_fused`
- Processes all 32 heads in parallel on GPU
- Supports GQA (32 Q heads / 8 KV heads)
- Fuses Q@K^T, causal mask, attention mask, softmax, @V
- Uses threadgroup shared memory for scores
- **Impact: Text encoding 3.9s → 3.1s (21% faster)**

**Correctness**: Max pixel diff = 1, mean = 0.003 (identical to CPU)

### GPU Infrastructure

#### Buffer Pool
- 64-entry activation pool, 512-entry weight cache
- Shared memory mode (zero-copy on Apple Silicon)
- Automatic buffer reuse for same-size allocations

#### Batch Command Buffers
- `flux_gpu_batch_begin()` / `flux_gpu_batch_end()`
- Batches independent operations into single GPU submission
- Used for parallel QKV projections, output projections

#### Operation Chains
- `flux_gpu_chain_begin()` / `flux_gpu_chain_end()`
- Operations share command buffer, sync only at chain end
- Keeps intermediate data on GPU

#### Persistent Tensors
- `flux_gpu_tensor_alloc_persistent()` - don't return to pool
- `flux_gpu_tensor_set_persistent()` - mark existing tensor
- For activations that stay on GPU between operations

### Remaining Optimization Opportunities

1. **VAE Decoder on GPU**: Currently CPU-only, ~2s overhead
2. **bf16 Inference**: Keep weights in bf16, compute in f16 for 2x bandwidth
3. **Flash Attention**: Tiled attention for O(n) memory instead of O(n²)
4. **Fused Kernels**: Combine RMSNorm + linear, reduce sync points

---

## API Reference

### GPU Tensor API

```c
// Create/allocate GPU tensors
flux_gpu_tensor_t flux_gpu_tensor_create(const float *data, size_t num_elements);
flux_gpu_tensor_t flux_gpu_tensor_alloc(size_t num_elements);
flux_gpu_tensor_t flux_gpu_tensor_alloc_persistent(size_t num_elements);

// Control persistence
void flux_gpu_tensor_set_persistent(flux_gpu_tensor_t tensor, int persistent);

// Access data
float *flux_gpu_tensor_data(flux_gpu_tensor_t tensor);
void flux_gpu_tensor_read(flux_gpu_tensor_t tensor, float *out, size_t num_elements);
void flux_gpu_tensor_free(flux_gpu_tensor_t tensor);

// Batching and chains
void flux_gpu_batch_begin(void);
void flux_gpu_batch_end(void);
void flux_gpu_chain_begin(void);
void flux_gpu_chain_end(void);
int flux_gpu_in_chain(void);
void flux_gpu_sync(void);
```

### GPU Attention APIs

```c
// Transformer attention (batched heads, GPU softmax)
void flux_metal_attention(float *out,
                          const float *Q, const float *K, const float *V,
                          float *scores_scratch,
                          int heads, int seq_q, int seq_k, int head_dim,
                          float scale);

// Text encoder causal attention (fused kernel, all heads parallel)
int flux_metal_causal_attention(float *out,
                                 const float *Q, const float *K, const float *V,
                                 const int *attention_mask,
                                 int seq, int num_q_heads, int num_kv_heads,
                                 int head_dim, float scale);
```

### Text Encoder API

```c
// Load text encoder
flux_text_encoder_t *flux_text_encoder_load(const char *model_dir);
void flux_text_encoder_free(flux_text_encoder_t *enc);

// Encode text to embeddings
float *flux_encode_text(flux_text_encoder_t *enc, const char *prompt, int *out_seq_len);

// Memory management
void flux_release_text_encoder(flux_ctx *ctx);  // Auto-release after encoding
```

### Progress Callbacks

```c
typedef enum {
    FLUX_SUBSTEP_DOUBLE_BLOCK,
    FLUX_SUBSTEP_SINGLE_BLOCK,
    FLUX_SUBSTEP_FINAL_LAYER,
} flux_substep_type_t;

extern flux_substep_callback_t flux_substep_callback;  // (type, index, total)
extern flux_step_callback_t flux_step_callback;        // (step, total)
```

Progress display: `d` = double block, `s` = 5 single blocks, `F` = final layer

---

## Debugging and Verification

### Verified Matching Values (Python vs C)

All values verified to match within FP32 precision:

| Component | Status | Notes |
|-----------|--------|-------|
| t_emb | ✓ MATCH | Timestep embedding |
| img_proj | ✓ MATCH | x_embedder output |
| txt_proj | ✓ MATCH | context_embedder output |
| RoPE cos/sin | ✓ MATCH | Frequency tables |
| AdaLN | ✓ MATCH | Normalization + modulation |
| Q after proj+norm | ✓ MATCH | Query projection |
| Q after RoPE | ✓ MATCH | After rotation |
| Attention scores | ✓ MATCH | Q @ K^T |
| Attention output | ✓ MATCH | softmax(scores) @ V |
| All 5 double blocks | ✓ MATCH | Full blocks |
| All 20 single blocks | ✓ MATCH | Full blocks |
| Final layer | ✓ MATCH | Output projection |

### Test Commands

```bash
# Verify output matches reference
./flux -d flux-klein-model -p "test" -o /tmp/test.png -W 64 -H 64 -S 42

python3 -c "
import numpy as np
from PIL import Image
ref = np.array(Image.open('test_vectors/reference_1step_64x64_seed42.png'))
test = np.array(Image.open('/tmp/test.png'))
diff = np.abs(ref.astype(float) - test.astype(float))
print(f'Max diff: {diff.max()}, Mean: {diff.mean():.2f}')
print('PASS' if diff.max() < 2 else 'FAIL')
"
```

### Compilation Flags

```bash
# Optimized build
CFLAGS="-O3 -ffast-math -march=native"

# Debug with transformer output
gcc $CFLAGS -DDEBUG_TRANSFORMER -DDEBUG_DOUBLE_BLOCK -c flux_transformer.c
```

### Bugs Fixed

1. **Final Layer scale/shift order**: Changed (shift, scale) → (scale, shift)
2. **K/V concatenation order**: Changed [IMAGE, TEXT] → [TEXT, IMAGE]
3. **Text sequence length**: Aligned Python and C to 512 tokens

---

## Work Log

### 2024-01-17
- Initial transformer implementation
- Verified all components match Python reference

### 2024-01-18
- Implemented Qwen3 text encoder (tokenizer + model)
- Added BLAS optimizations for linear and conv
- Verified text embeddings match within FP32 precision
- Added automatic text encoder memory release

### 2024-01-19
- Added GPU batch command buffer infrastructure
- Implemented GPU-accelerated attention (batched heads)
- Added GPU tensor API for persistent buffers
- Implemented fused GPU softmax in attention
- Added batch input caching to reduce copies
- Moved text encoder FFN to GPU
- Pre-allocated attention work buffers
- Added strided BLAS access (eliminated Q/output copies)
- **Implemented fused GPU causal attention kernel**
  - All 32 heads processed in parallel
  - Supports GQA and attention mask
  - 21% faster text encoding, 33% faster total
- Added operation chain API
- Updated flux_metal_attention to use buffer pool

**Current commits:**
- `fc419de` - Strided BLAS access
- `828125c` - Fused GPU softmax
- `551f8fe` - GPU for text encoder FFN
- `050421f` - Pre-allocated attention buffers

---

## File Structure

```
flux.c                  - Main library (generation, img2img)
flux_transformer.c      - Diffusion transformer
flux_qwen3.c            - Qwen3 text encoder
flux_qwen3_tokenizer.c  - BPE tokenizer
flux_vae.c              - VAE encoder/decoder
flux_kernels.c          - CPU kernels (softmax, RMSNorm, etc.)
flux_metal.m            - Metal GPU acceleration
flux_shaders.metal      - Metal compute kernels
flux_safetensors.c      - Weight loading
main.c                  - CLI interface
```
