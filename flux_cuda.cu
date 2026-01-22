/*
 * FLUX CUDA Acceleration - Implementation
 *
 * GPU-accelerated operations using NVIDIA CUDA and cuBLAS.
 * Inspired by ggml-cuda from stable-diffusion.cpp, but standalone.
 */

#include "flux_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Error Handling Macros
 * ======================================================================== */

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_RET(err, ret) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return ret; \
    } \
} while(0)

#define CUBLAS_CHECK(err) do { \
    cublasStatus_t e = (err); \
    if (e != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)e); \
        return; \
    } \
} while(0)

/* ========================================================================
 * Global State
 * ======================================================================== */

static int g_initialized = 0;
static int g_available = 0;
static cublasHandle_t g_cublas = NULL;
static cudaStream_t g_stream = NULL;
static int g_batch_mode = 0;
static char g_device_name[256] = "Unknown";
static int g_compute_cap = 0;

/* ========================================================================
 * Kernel Constants
 * ======================================================================== */

#define WARP_SIZE 32
#define BLOCK_1D 256
#define BLOCK_NORM 256

/* ========================================================================
 * Initialization
 * ======================================================================== */

int flux_cuda_init(void) {
    if (g_initialized) return g_available;
    g_initialized = 1;

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        fprintf(stderr, "CUDA: No devices found\n");
        return 0;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 0;

    snprintf(g_device_name, sizeof(g_device_name), "%s", prop.name);
    g_compute_cap = prop.major * 10 + prop.minor;

    printf("CUDA: %s (SM %d.%d, %zu MB)\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024 * 1024));

    if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) return 0;
    if (cudaStreamCreate(&g_stream) != cudaSuccess) {
        cublasDestroy(g_cublas);
        return 0;
    }

    cublasSetStream(g_cublas, g_stream);
    if (g_compute_cap >= 70) cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    g_available = 1;
    return 1;
}

int flux_cuda_available(void) { return g_available; }
const char* flux_cuda_device_name(void) { return g_device_name; }
int flux_cuda_compute_capability(void) { return g_compute_cap; }
int flux_cuda_kernels_available(void) { return g_available; }

void flux_cuda_cleanup(void) {
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = NULL; }
    if (g_cublas) { cublasDestroy(g_cublas); g_cublas = NULL; }
    g_available = 0;
    g_initialized = 0;
}

void flux_cuda_reset(void) {
    if (g_available) cudaStreamSynchronize(g_stream);
}

void flux_cuda_sync(void) {
    if (g_available) cudaStreamSynchronize(g_stream);
}

void flux_cuda_begin_batch(void) { g_batch_mode = 1; }
void flux_cuda_end_batch(void) { g_batch_mode = 0; flux_cuda_sync(); }
int flux_cuda_in_batch(void) { return g_batch_mode; }
size_t flux_cuda_memory_used(void) { return 0; }

/* ========================================================================
 * CUDA Kernels
 * ======================================================================== */

__global__ void k_silu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

__global__ void k_silu_mul(float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void k_gelu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

__global__ void k_add(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void k_mul(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= b[i];
}

__global__ void k_scale(float *a, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= s;
}

__global__ void k_rms_norm(float *out, const float *x, const float *w,
                            int seq, int hid, float eps) {
    int row = blockIdx.x;
    if (row >= seq) return;

    const float *xr = x + row * hid;
    float *outr = out + row * hid;

    __shared__ float ssum[BLOCK_NORM];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hid; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    ssum[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }

    float rms = rsqrtf(ssum[0] / hid + eps);
    for (int i = threadIdx.x; i < hid; i += blockDim.x) {
        outr[i] = xr[i] * rms * w[i];
    }
}

__global__ void k_softmax(float *x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float *xr = x + row * cols;
    __shared__ float smax[BLOCK_NORM], ssum[BLOCK_NORM];

    float mx = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        mx = fmaxf(mx, xr[i]);
    smax[threadIdx.x] = mx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        __syncthreads();
    }
    mx = smax[0];

    float sm = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(xr[i] - mx);
        xr[i] = e;
        sm += e;
    }
    ssum[threadIdx.x] = sm;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    sm = ssum[0];

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        xr[i] /= sm;
}

__global__ void k_qk_rms_norm(float *q, float *k, const float *qw, const float *kw,
                               int seq, int heads, int hdim, float eps) {
    int idx = blockIdx.x;
    int s = idx / heads, h = idx % heads;
    if (s >= seq) return;

    float *qh = q + s * heads * hdim + h * hdim;
    float *kh = k + s * heads * hdim + h * hdim;

    __shared__ float sq[BLOCK_NORM], sk[BLOCK_NORM];
    float sumq = 0, sumk = 0;
    for (int i = threadIdx.x; i < hdim; i += blockDim.x) {
        sumq += qh[i] * qh[i];
        sumk += kh[i] * kh[i];
    }
    sq[threadIdx.x] = sumq;
    sk[threadIdx.x] = sumk;
    __syncthreads();

    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            sq[threadIdx.x] += sq[threadIdx.x + st];
            sk[threadIdx.x] += sk[threadIdx.x + st];
        }
        __syncthreads();
    }

    float rmsq = rsqrtf(sq[0] / hdim + eps);
    float rmsk = rsqrtf(sk[0] / hdim + eps);

    for (int i = threadIdx.x; i < hdim; i += blockDim.x) {
        qh[i] = qh[i] * rmsq * qw[i];
        kh[i] = kh[i] * rmsk * kw[i];
    }
}

__global__ void k_adaln_norm(float *out, const float *x, const float *shift,
                              const float *scale, int seq, int hid, float eps) {
    int row = blockIdx.x;
    if (row >= seq) return;

    const float *xr = x + row * hid;
    float *outr = out + row * hid;

    __shared__ float smean[BLOCK_NORM], svar[BLOCK_NORM];
    float sm = 0, sv = 0;
    for (int i = threadIdx.x; i < hid; i += blockDim.x) sm += xr[i];
    smean[threadIdx.x] = sm;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smean[threadIdx.x] += smean[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smean[0] / hid;

    for (int i = threadIdx.x; i < hid; i += blockDim.x) {
        float d = xr[i] - mean;
        sv += d * d;
    }
    svar[threadIdx.x] = sv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) svar[threadIdx.x] += svar[threadIdx.x + s];
        __syncthreads();
    }
    float rstd = rsqrtf(svar[0] / hid + eps);

    for (int i = threadIdx.x; i < hid; i += blockDim.x) {
        float norm = (xr[i] - mean) * rstd;
        outr[i] = (1.0f + scale[i]) * norm + shift[i];
    }
}

__global__ void k_rope_2d(float *x, const float *cos_f, const float *sin_f,
                           int seq, int heads, int hdim, int axis_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * (axis_dim / 2);
    if (idx >= total) return;

    int s = idx / (heads * (axis_dim / 2));
    int rem = idx % (heads * (axis_dim / 2));
    int h = rem / (axis_dim / 2);
    int p = rem % (axis_dim / 2);

    int freq_idx = s * (axis_dim / 2) + p;
    float c = cos_f[freq_idx], sn = sin_f[freq_idx];

    int base = s * heads * hdim + h * hdim + p * 2;
    float x0 = x[base], x1 = x[base + 1];
    x[base] = x0 * c - x1 * sn;
    x[base + 1] = x0 * sn + x1 * c;
}

/* ========================================================================
 * cuBLAS Matrix Multiplication
 * ======================================================================== */

void flux_cuda_sgemm(int ta, int tb, int M, int N, int K,
                     float alpha, const float *A, int lda,
                     const float *B, int ldb, float beta, float *C, int ldc) {
    if (!g_available) return;

    size_t szA = (size_t)(ta ? K * M : M * K) * sizeof(float);
    size_t szB = (size_t)(tb ? N * K : K * N) * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, szA));
    CUDA_CHECK(cudaMalloc(&dB, szB));
    CUDA_CHECK(cudaMalloc(&dC, szC));

    CUDA_CHECK(cudaMemcpyAsync(dA, A, szA, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dB, B, szB, cudaMemcpyHostToDevice, g_stream));
    if (beta != 0.0f) CUDA_CHECK(cudaMemcpyAsync(dC, C, szC, cudaMemcpyHostToDevice, g_stream));

    /*
     * Row-major to column-major trick for cuBLAS:
     * We want: C[M,N] = A[M,K] @ B[K,N] (row-major)
     * cuBLAS sees row-major data as transposed column-major.
     *
     * So we call: cublasSgemm(op_B, op_A, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc)
     * This computes C^T = B^T @ A^T which gives us C in row-major.
     */
    cublasOperation_t opA = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = tb ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemm(g_cublas, opB, opA, N, M, K, &alpha,
                             dB, ldb, dA, lda, &beta, dC, ldc));

    CUDA_CHECK(cudaMemcpyAsync(C, dC, szC, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

void flux_cuda_sgemm_bf16(int ta, int tb, int M, int N, int K,
                          float alpha, const float *A, int lda,
                          const uint16_t *B_bf16, int ldb,
                          float beta, float *C, int ldc) {
    if (!g_available) return;

    /* Convert bf16 to f32 */
    size_t szB = (size_t)(tb ? N * K : K * N);
    float *B_f32 = (float *)malloc(szB * sizeof(float));
    if (!B_f32) return;

    for (size_t i = 0; i < szB; i++) {
        uint32_t bits = ((uint32_t)B_bf16[i]) << 16;
        memcpy(&B_f32[i], &bits, sizeof(float));
    }

    flux_cuda_sgemm(ta, tb, M, N, K, alpha, A, lda, B_f32, ldb, beta, C, ldc);
    free(B_f32);
}

void flux_cuda_sgemm_batch(int ta, int tb, int M, int N, int K,
                           float alpha, const float *A, int lda, int strideA,
                           const float *B, int ldb, int strideB,
                           float beta, float *C, int ldc, int strideC, int batch) {
    for (int b = 0; b < batch; b++) {
        flux_cuda_sgemm(ta, tb, M, N, K, alpha,
                        A + b * strideA, lda, B + b * strideB, ldb,
                        beta, C + b * strideC, ldc);
    }
}

/* ========================================================================
 * C API Wrappers for Kernels
 * ======================================================================== */

void flux_cuda_silu(float *x, int n) {
    if (!g_available) return;
    float *dx; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dx, sz));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, sz, cudaMemcpyHostToDevice, g_stream));
    int blk = (n + BLOCK_1D - 1) / BLOCK_1D;
    k_silu<<<blk, BLOCK_1D, 0, g_stream>>>(dx, n);
    CUDA_CHECK(cudaMemcpyAsync(x, dx, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dx);
}

void flux_cuda_gelu(float *x, int n) {
    if (!g_available) return;
    float *dx; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dx, sz));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, sz, cudaMemcpyHostToDevice, g_stream));
    int blk = (n + BLOCK_1D - 1) / BLOCK_1D;
    k_gelu<<<blk, BLOCK_1D, 0, g_stream>>>(dx, n);
    CUDA_CHECK(cudaMemcpyAsync(x, dx, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dx);
}

void flux_cuda_silu_mul(float *gate, const float *up, int n) {
    if (!g_available) return;
    float *dg, *du; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dg, sz)); CUDA_CHECK(cudaMalloc(&du, sz));
    CUDA_CHECK(cudaMemcpyAsync(dg, gate, sz, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(du, up, sz, cudaMemcpyHostToDevice, g_stream));
    int blk = (n + BLOCK_1D - 1) / BLOCK_1D;
    k_silu_mul<<<blk, BLOCK_1D, 0, g_stream>>>(dg, du, n);
    CUDA_CHECK(cudaMemcpyAsync(gate, dg, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dg); cudaFree(du);
}

void flux_cuda_add_inplace(float *a, const float *b, int n) {
    if (!g_available) return;
    float *da, *db; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&da, sz)); CUDA_CHECK(cudaMalloc(&db, sz));
    CUDA_CHECK(cudaMemcpyAsync(da, a, sz, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(db, b, sz, cudaMemcpyHostToDevice, g_stream));
    k_add<<<(n + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(da, db, n);
    CUDA_CHECK(cudaMemcpyAsync(a, da, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(da); cudaFree(db);
}

void flux_cuda_mul_inplace(float *a, const float *b, int n) {
    if (!g_available) return;
    float *da, *db; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&da, sz)); CUDA_CHECK(cudaMalloc(&db, sz));
    CUDA_CHECK(cudaMemcpyAsync(da, a, sz, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(db, b, sz, cudaMemcpyHostToDevice, g_stream));
    k_mul<<<(n + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(da, db, n);
    CUDA_CHECK(cudaMemcpyAsync(a, da, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(da); cudaFree(db);
}

void flux_cuda_scale_inplace(float *a, float s, int n) {
    if (!g_available) return;
    float *da; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&da, sz));
    CUDA_CHECK(cudaMemcpyAsync(da, a, sz, cudaMemcpyHostToDevice, g_stream));
    k_scale<<<(n + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(da, s, n);
    CUDA_CHECK(cudaMemcpyAsync(a, da, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(da);
}

void flux_cuda_rms_norm(float *out, const float *x, const float *w,
                        int seq, int hid, float eps) {
    if (!g_available) return;
    float *dout, *dx, *dw;
    size_t szx = (size_t)seq * hid * sizeof(float), szw = hid * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dout, szx)); CUDA_CHECK(cudaMalloc(&dx, szx)); CUDA_CHECK(cudaMalloc(&dw, szw));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, szx, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dw, w, szw, cudaMemcpyHostToDevice, g_stream));
    k_rms_norm<<<seq, BLOCK_NORM, 0, g_stream>>>(dout, dx, dw, seq, hid, eps);
    CUDA_CHECK(cudaMemcpyAsync(out, dout, szx, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dout); cudaFree(dx); cudaFree(dw);
}

void flux_cuda_softmax(float *x, int rows, int cols) {
    if (!g_available) return;
    float *dx; size_t sz = (size_t)rows * cols * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dx, sz));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, sz, cudaMemcpyHostToDevice, g_stream));
    k_softmax<<<rows, BLOCK_NORM, 0, g_stream>>>(dx, rows, cols);
    CUDA_CHECK(cudaMemcpyAsync(x, dx, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dx);
}

void flux_cuda_qk_rms_norm(float *q, float *k, const float *qw, const float *kw,
                           int seq, int heads, int hdim, float eps) {
    if (!g_available) return;
    float *dq, *dk, *dqw, *dkw;
    size_t szqk = (size_t)seq * heads * hdim * sizeof(float), szw = hdim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dq, szqk)); CUDA_CHECK(cudaMalloc(&dk, szqk));
    CUDA_CHECK(cudaMalloc(&dqw, szw)); CUDA_CHECK(cudaMalloc(&dkw, szw));
    CUDA_CHECK(cudaMemcpyAsync(dq, q, szqk, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dk, k, szqk, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dqw, qw, szw, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dkw, kw, szw, cudaMemcpyHostToDevice, g_stream));
    k_qk_rms_norm<<<seq * heads, BLOCK_NORM, 0, g_stream>>>(dq, dk, dqw, dkw, seq, heads, hdim, eps);
    CUDA_CHECK(cudaMemcpyAsync(q, dq, szqk, cudaMemcpyDeviceToHost, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(k, dk, szqk, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dq); cudaFree(dk); cudaFree(dqw); cudaFree(dkw);
}

void flux_cuda_adaln_norm(float *out, const float *x, const float *shift,
                          const float *scale, int seq, int hid, float eps) {
    if (!g_available) return;
    float *dout, *dx, *dsh, *dsc;
    size_t szx = (size_t)seq * hid * sizeof(float), szm = hid * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dout, szx)); CUDA_CHECK(cudaMalloc(&dx, szx));
    CUDA_CHECK(cudaMalloc(&dsh, szm)); CUDA_CHECK(cudaMalloc(&dsc, szm));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, szx, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dsh, shift, szm, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dsc, scale, szm, cudaMemcpyHostToDevice, g_stream));
    k_adaln_norm<<<seq, BLOCK_NORM, 0, g_stream>>>(dout, dx, dsh, dsc, seq, hid, eps);
    CUDA_CHECK(cudaMemcpyAsync(out, dout, szx, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dout); cudaFree(dx); cudaFree(dsh); cudaFree(dsc);
}

void flux_cuda_rope_2d(float *x, const float *cos_f, const float *sin_f,
                       int seq, int heads, int hdim, int axis_dim) {
    if (!g_available) return;
    float *dx, *dc, *ds;
    size_t szx = (size_t)seq * heads * hdim * sizeof(float);
    size_t szf = (size_t)seq * (axis_dim / 2) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dx, szx)); CUDA_CHECK(cudaMalloc(&dc, szf)); CUDA_CHECK(cudaMalloc(&ds, szf));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, szx, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dc, cos_f, szf, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(ds, sin_f, szf, cudaMemcpyHostToDevice, g_stream));
    int total = seq * heads * (axis_dim / 2);
    k_rope_2d<<<(total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(dx, dc, ds, seq, heads, hdim, axis_dim);
    CUDA_CHECK(cudaMemcpyAsync(x, dx, szx, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dx); cudaFree(dc); cudaFree(ds);
}

/* ========================================================================
 * Attention and Conv2D - Fall back to CPU for now
 * ======================================================================== */

int flux_cuda_conv2d(float *out, const float *in, const float *weight, const float *bias,
                     int batch, int in_ch, int out_ch, int H, int W, int kH, int kW,
                     int stride, int padding) {
    (void)out; (void)in; (void)weight; (void)bias;
    (void)batch; (void)in_ch; (void)out_ch; (void)H; (void)W; (void)kH; (void)kW;
    (void)stride; (void)padding;
    return 0;  /* Fall back to CPU */
}

int flux_cuda_attention_fused(float *out, const float *Q, const float *K, const float *V,
                              int seq_q, int seq_k, int num_heads, int head_dim, float scale) {
    (void)out; (void)Q; (void)K; (void)V;
    (void)seq_q; (void)seq_k; (void)num_heads; (void)head_dim; (void)scale;
    return 0;  /* Fall back to CPU */
}

int flux_cuda_causal_attention(float *out, const float *Q, const float *K, const float *V,
                               const int *attention_mask, int seq, int num_q_heads,
                               int num_kv_heads, int head_dim, float scale) {
    (void)out; (void)Q; (void)K; (void)V; (void)attention_mask;
    (void)seq; (void)num_q_heads; (void)num_kv_heads; (void)head_dim; (void)scale;
    return 0;  /* Fall back to CPU */
}
