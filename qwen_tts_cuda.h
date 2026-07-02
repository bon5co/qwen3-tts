/*
 * qwen_tts_cuda.h — NVIDIA CUDA backend (G3), C-callable surface.
 *
 * Implemented in qwen_tts_cuda.c, built only by `make cuda` (defines
 * QWEN_HAVE_CUDA, links -lcublas -lcudart). When the define is absent the TU
 * compiles to no-op stubs (available()→0), so the file is safe to build on M1.
 *
 * v1 = cuBLAS-first (no nvcc): matmat/matvec via cublasSgemm on device f32.
 * Weights are bf16→f32-converted on upload for now; the resident-bf16 +
 * cublasGemmEx (sm_80+) and custom decode matvec + CUDA Graphs are G3b
 * (plan_v4 §E4.ter) — validated on the DGX/5090, not here.
 */

#ifndef QWEN_TTS_CUDA_H
#define QWEN_TTS_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int   qwen_cuda_available(void);
void *qwen_cuda_init(void);
void  qwen_cuda_free(void *ctx);

/* y[rows] = W[rows,cols] @ x[cols]   (W bf16, x/y f32). */
void  qwen_cuda_matvec_bf16(void *ctx, float *y,
                            const uint16_t *W, const float *x,
                            int rows, int cols);

/* Y[rows,B] = W[rows,cols] @ X[cols,B]  (row-major f32; B<=64). */
void  qwen_cuda_matmat_bf16(void *ctx, float *Y,
                            const uint16_t *W, const float *X,
                            int rows, int cols, int B);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_CUDA_H */
