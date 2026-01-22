/*
 * FLUX CUDA Acceleration
 *
 * GPU-accelerated matrix operations using NVIDIA CUDA and cuBLAS.
 * Provides significant speedup on NVIDIA GPUs.
 *
 * Inspired by stable-diffusion.cpp's GGML CUDA backend, but standalone.
 */

#ifndef FLUX_CUDA_H
#define FLUX_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize CUDA acceleration.
 * Returns 1 on success, 0 if CUDA is not available.
 * Safe to call multiple times.
 */
int flux_cuda_init(void);

/*
 * Check if CUDA acceleration is available and initialized.
 */
int flux_cuda_available(void);

/*
 * Cleanup CUDA resources.
 */
void flux_cuda_cleanup(void);

/*
 * Reset all GPU state (caches, pools, pending commands).
 * Call this between independent inference phases.
 */
void flux_cuda_reset(void);

/*
 * GPU-accelerated matrix multiplication using cuBLAS.
 * C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
 *
 * transpose_a: if non-zero, use A^T
 * transpose_b: if non-zero, use B^T
 */
void flux_cuda_sgemm(int transpose_a, int transpose_b,
                     int M, int N, int K,
                     float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float beta,
                     float *C, int ldc);

/*
 * GPU-accelerated matrix multiplication with bf16 weights.
 * C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
 *
 * A is f32, B is bf16 (weights), C is f32
 * This provides 2x memory bandwidth improvement for weight-bound operations.
 */
void flux_cuda_sgemm_bf16(int transpose_a, int transpose_b,
                          int M, int N, int K,
                          float alpha,
                          const float *A, int lda,
                          const uint16_t *B_bf16, int ldb,
                          float beta,
                          float *C, int ldc);

/*
 * 2D convolution using cuDNN (if available) or im2col+cuBLAS.
 * Returns 1 on success, 0 on failure.
 */
int flux_cuda_conv2d(float *out, const float *in,
                     const float *weight, const float *bias,
                     int batch, int in_ch, int out_ch,
                     int H, int W, int kH, int kW,
                     int stride, int padding);

/*
 * Batch matrix multiplication on GPU.
 * Performs batch_count independent matrix multiplications.
 */
void flux_cuda_sgemm_batch(int transpose_a, int transpose_b,
                           int M, int N, int K,
                           float alpha,
                           const float *A, int lda, int stride_a,
                           const float *B, int ldb, int stride_b,
                           float beta,
                           float *C, int ldc, int stride_c,
                           int batch_count);

/*
 * Synchronize GPU operations (wait for completion).
 */
void flux_cuda_sync(void);

/*
 * Begin a batch of GPU operations.
 * Operations after this call are queued but not executed until flux_cuda_end_batch().
 */
void flux_cuda_begin_batch(void);

/*
 * End a batch of GPU operations.
 * Executes all queued operations and waits for completion.
 */
void flux_cuda_end_batch(void);

/*
 * Check if currently in batch mode.
 */
int flux_cuda_in_batch(void);

/*
 * Get GPU memory usage info (for debugging).
 */
size_t flux_cuda_memory_used(void);

/* ========================================================================
 * GPU Compute Kernels - Element-wise operations on GPU
 * ======================================================================== */

/*
 * GPU-accelerated RMSNorm.
 * out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
 * x: [seq_len, hidden], weight: [hidden], out: [seq_len, hidden]
 */
void flux_cuda_rms_norm(float *out, const float *x, const float *weight,
                        int seq_len, int hidden, float eps);

/*
 * GPU-accelerated QK RMSNorm (in-place).
 * Normalizes Q and K separately for each head.
 * q, k: [seq, heads*head_dim] (modified in-place)
 * q_weight, k_weight: [head_dim]
 */
void flux_cuda_qk_rms_norm(float *q, float *k,
                           const float *q_weight, const float *k_weight,
                           int seq, int heads, int head_dim, float eps);

/*
 * GPU-accelerated LayerNorm + AdaLN modulation.
 * out = (1 + scale) * layernorm(x) + shift
 * x: [seq_len, hidden], shift/scale: [hidden]
 */
void flux_cuda_adaln_norm(float *out, const float *x,
                          const float *shift, const float *scale,
                          int seq_len, int hidden, float eps);

/*
 * GPU-accelerated SiLU activation (in-place).
 * x = x * sigmoid(x)
 */
void flux_cuda_silu(float *x, int n);

/*
 * GPU-accelerated SiLU with multiply (SwiGLU style, in-place).
 * gate = silu(gate) * up
 */
void flux_cuda_silu_mul(float *gate, const float *up, int n);

/*
 * GPU-accelerated softmax (row-wise, in-place).
 * x: [rows, cols], softmax applied to each row
 */
void flux_cuda_softmax(float *x, int rows, int cols);

/*
 * GPU-accelerated 2D RoPE (in-place).
 * x: [seq, heads*head_dim]
 * cos_freq, sin_freq: [seq, head_dim]
 */
void flux_cuda_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                       int seq, int heads, int head_dim, int axis_dim);

/*
 * GPU-accelerated GELU activation (in-place).
 * x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
void flux_cuda_gelu(float *x, int n);

/*
 * GPU-accelerated element-wise add (in-place).
 * a += b
 */
void flux_cuda_add_inplace(float *a, const float *b, int n);

/*
 * GPU-accelerated element-wise multiply (in-place).
 * a *= b
 */
void flux_cuda_mul_inplace(float *a, const float *b, int n);

/*
 * GPU-accelerated scale (in-place).
 * a *= s
 */
void flux_cuda_scale_inplace(float *a, float s, int n);

/*
 * Fused attention on GPU.
 * Computes attention for all heads in a single GPU batch.
 *
 * Q, K, V are in [seq, heads*head_dim] layout
 * out: [seq_q, heads * head_dim]
 *
 * This does: out = softmax(Q @ K^T * scale) @ V
 * Returns 1 on success, 0 on failure (falls back to CPU).
 */
int flux_cuda_attention_fused(float *out,
                              const float *Q, const float *K, const float *V,
                              int seq_q, int seq_k, int num_heads, int head_dim,
                              float scale);

/*
 * GPU-accelerated causal attention for text encoder.
 * Supports GQA (Grouped Query Attention).
 * Returns 1 on success, 0 on failure.
 */
int flux_cuda_causal_attention(float *out,
                               const float *Q, const float *K, const float *V,
                               const int *attention_mask,
                               int seq, int num_q_heads, int num_kv_heads,
                               int head_dim, float scale);

/*
 * Check if compute kernels are available.
 */
int flux_cuda_kernels_available(void);

/*
 * Get CUDA device name for display.
 */
const char* flux_cuda_device_name(void);

/*
 * Get CUDA compute capability.
 */
int flux_cuda_compute_capability(void);

#ifdef __cplusplus
}
#endif

#endif /* FLUX_CUDA_H */
