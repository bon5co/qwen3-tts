/*
 * qwen_tts_cuda.c — NVIDIA CUDA backend (G3), cuBLAS-first v1.
 *
 * Built only by `make cuda` (-DQWEN_HAVE_CUDA -lcublas -lcudart). Without the
 * define, compiles to no-op stubs so `make blas` / M1 builds are unaffected.
 *
 * Row-major mapping for cuBLAS (column-major): to get
 *   Y[rows,B] = W[rows,cols] @ X[cols,B]   (all row-major)
 * we compute, in cuBLAS column-major terms,
 *   Y_cm[B,rows] = X_cm[B,cols] * W_cm[cols,rows]
 * i.e. cublasSgemm(N, N, m=B, n=rows, k=cols, X(lda=B), Wf32(ldb=cols), Y(ldc=B)).
 *
 * v1 keeps it deliberately simple + correct (cuBLAS only, no custom kernels, no
 * nvcc): W is converted bf16→f32 on the host and uploaded per call. That is NOT
 * the shipping-fast path — resident bf16 weights + cublasGemmEx + a custom
 * decode matvec + CUDA Graphs are G3b, to be built and RTF-measured on the DGX.
 * The value of v1: the seam compiles + runs on the DGX and gives a real cuBLAS
 * RTF baseline to compare against the audio.cpp/ggml oracle.
 */

#include "qwen_tts_cuda.h"

#ifndef QWEN_HAVE_CUDA

int   qwen_cuda_available(void) { return 0; }
void *qwen_cuda_init(void) { return 0; }
void  qwen_cuda_free(void *ctx) { (void)ctx; }
void  qwen_cuda_matvec_bf16(void *ctx, float *y, const uint16_t *W,
                            const float *x, int rows, int cols) {
    (void)ctx; (void)y; (void)W; (void)x; (void)rows; (void)cols;
}
void  qwen_cuda_matmat_bf16(void *ctx, float *Y, const uint16_t *W,
                            const float *X, int rows, int cols, int B) {
    (void)ctx; (void)Y; (void)W; (void)X; (void)rows; (void)cols; (void)B;
}

#else /* QWEN_HAVE_CUDA */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    cublasHandle_t handle;
} qwen_cuda_ctx;

static inline float bf16_to_f32_host(uint16_t b) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)b << 16;
    return v.f;
}

int qwen_cuda_available(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n > 0;
}

void *qwen_cuda_init(void) {
    qwen_cuda_ctx *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    if (cublasCreate(&c->handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUDA: cublasCreate failed\n");
        free(c);
        return NULL;
    }
    return c;
}

void qwen_cuda_free(void *ctx) {
    if (!ctx) return;
    qwen_cuda_ctx *c = ctx;
    if (c->handle) cublasDestroy(c->handle);
    free(c);
}

/* Convert W (bf16, host) → f32 host scratch, upload, run Sgemm, download. */
void qwen_cuda_matmat_bf16(void *ctx, float *Y, const uint16_t *W,
                           const float *X, int rows, int cols, int B) {
    qwen_cuda_ctx *c = ctx;
    const size_t nW = (size_t)rows * cols;
    float *Wf = (float *)malloc(nW * sizeof(float));
    if (!Wf) return;
    for (size_t i = 0; i < nW; ++i) Wf[i] = bf16_to_f32_host(W[i]);

    float *dW = NULL, *dX = NULL, *dY = NULL;
    cudaMalloc((void **)&dW, nW * sizeof(float));
    cudaMalloc((void **)&dX, (size_t)cols * B * sizeof(float));
    cudaMalloc((void **)&dY, (size_t)rows * B * sizeof(float));
    cudaMemcpy(dW, Wf, nW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X, (size_t)cols * B * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;
    /* Y_cm[B,rows] = X_cm[B,cols] * W_cm[cols,rows] */
    cublasSgemm(c->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                /*m=*/B, /*n=*/rows, /*k=*/cols,
                &alpha,
                dX, /*lda=*/B,
                dW, /*ldb=*/cols,
                &beta,
                dY, /*ldc=*/B);

    cudaMemcpy(Y, dY, (size_t)rows * B * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dW); cudaFree(dX); cudaFree(dY);
    free(Wf);
}

void qwen_cuda_matvec_bf16(void *ctx, float *y, const uint16_t *W,
                           const float *x, int rows, int cols) {
    qwen_cuda_matmat_bf16(ctx, y, W, x, rows, cols, 1);
}

#endif /* QWEN_HAVE_CUDA */
