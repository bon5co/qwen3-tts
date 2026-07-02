/*
 * qwen_tts_metal.h — Apple Metal backend (G2), C-callable surface.
 *
 * Implemented in qwen_tts_metal.m (Objective-C, clang -fobjc-arc). Only built
 * by `make metal` (defines QWEN_HAVE_METAL). The functions are plain C linkage
 * so the rest of the engine (gcc-compiled) can call them.
 *
 * v1 scope: matvec_bf16 (correctness vehicle) + matmat_bf16 (the compute-bound
 * primitive that can actually beat CPU on M1). Weights are read as raw bf16
 * uint16 and reconstructed to f32 in-shader via a bit-shift, so no dependency
 * on MSL `bfloat` type availability.
 */

#ifndef QWEN_TTS_METAL_H
#define QWEN_TTS_METAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 if a Metal device exists on this machine (and this TU is compiled in). */
int   qwen_metal_available(void);

/* Create the Metal context (device, queue, compiled pipeline states).
 * Returns an opaque handle, or NULL on failure (no device / shader compile). */
void *qwen_metal_init(void);
void  qwen_metal_free(void *ctx);

/* y[rows] = W[rows,cols] @ x[cols]   (W bf16, x/y f32). */
void  qwen_metal_matvec_bf16(void *ctx, float *y,
                             const uint16_t *W, const float *x,
                             int rows, int cols);

/* Y[rows,B] = W[rows,cols] @ X[cols,B]  (row-major f32; B<=64). */
void  qwen_metal_matmat_bf16(void *ctx, float *Y,
                             const uint16_t *W, const float *X,
                             int rows, int cols, int B);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_METAL_H */
