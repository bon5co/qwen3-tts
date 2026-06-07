/* batching_bench.c — premise test for text-chunk BATCHING.
 *
 * Single-stream TTS re-reads the bf16 Talker weights from DRAM for EVERY token
 * (matrix-VECTOR / GEMV). The batching idea: step B text-chunks together so each
 * weight element is read ONCE and reused across all B chunks (matrix-MATRIX / GEMM).
 * If memory-bound, GEMM(B) costs ~the same as ONE GEMV -> up to B x throughput.
 * If compute-bound (NEON FMA throughput, or weights fit cache), batching buys little.
 *
 * Both kernels use the SAME efficient NEON bf16 decode; the GEMM keeps the B
 * accumulators register-resident across the k loop (W streamed once). A printed
 * checksum stops the compiler dead-code-eliminating the work.
 *
 * Build/run:  make batching-bench
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static inline float bf16_to_f32(uint16_t bf) {
    uint32_t u = (uint32_t)bf << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16);
}
static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* GEMV over rows [r0,r1): y = W @ x. Returns a checksum (sink). */
static float gemv_bf16(const uint16_t *W, const float *x, int r0, int r1, int C) {
    float sink = 0.0f;
    for (int r = r0; r < r1; r++) {
        const uint16_t *w = W + (size_t)r * C; int k = 0; float s = 0.0f;
#ifdef __ARM_NEON
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        for (; k + 8 <= C; k += 8) {
            uint16x8_t bf = vld1q_u16(w + k);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)), vld1q_f32(x + k));
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)), vld1q_f32(x + k + 4));
        }
        s = vaddvq_f32(vaddq_f32(a0, a1));
#endif
        for (; k < C; k++) s += bf16_to_f32(w[k]) * x[k];
        sink += s;
    }
    return sink;
}

/* GEMM B=16: Y[R x 16] = W[R x C] @ X[C x 16]. X row-major [C][16]. The 16 batch
 * accumulators stay in 4 NEON registers across the whole k loop -> W is streamed
 * from DRAM exactly once. Returns a checksum. */
static float gemm16_bf16(const uint16_t *W, const float *X, int r0, int r1, int C) {
    float sink = 0.0f;
#ifdef __ARM_NEON
    for (int r = r0; r < r1; r++) {
        const uint16_t *w = W + (size_t)r * C;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0),
                    a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        for (int k = 0; k < C; k++) {
            float32x4_t wq = vdupq_n_f32(bf16_to_f32(w[k]));
            const float *xk = X + (size_t)k * 16;
            a0 = vfmaq_f32(a0, wq, vld1q_f32(xk));
            a1 = vfmaq_f32(a1, wq, vld1q_f32(xk + 4));
            a2 = vfmaq_f32(a2, wq, vld1q_f32(xk + 8));
            a3 = vfmaq_f32(a3, wq, vld1q_f32(xk + 12));
        }
        sink += vaddvq_f32(a0) + vaddvq_f32(a1) + vaddvq_f32(a2) + vaddvq_f32(a3);
    }
#else
    (void)W; (void)X; (void)r0; (void)r1; (void)C;
#endif
    return sink;
}

/* ---- threaded harness: split ROWS across T threads (as the real matvec does),
 * so all cores contend for DRAM bandwidth — the realistic single-stream regime. ---- */
typedef struct { const uint16_t *W; const float *X; int r0, r1, C, B, mode; float sink; } job_t;
static void *worker(void *a) {
    job_t *j = (job_t *)a;
    if (j->mode == 0)                                   /* B sequential GEMV over the slice */
        for (int b = 0; b < j->B; b++) j->sink += gemv_bf16(j->W, j->X + (size_t)b, j->r0, j->r1, j->C);
    else                                                /* one batched GEMM(16) over the slice */
        j->sink += gemm16_bf16(j->W, j->X, j->r0, j->r1, j->C);
    return NULL;
}
static double timed_threaded(const uint16_t *W, const float *X, int R, int C, int B,
                             int T, int mode, int reps, volatile float *sink) {
    double t0 = now_s();
    for (int it = 0; it < reps; it++) {
        pthread_t th[64]; job_t jb[64];
        for (int t = 0; t < T; t++) {
            jb[t] = (job_t){ W, X, (int)((long)t * R / T), (int)((long)(t + 1) * R / T), C, B, mode, 0 };
            pthread_create(&th[t], NULL, worker, &jb[t]);
        }
        for (int t = 0; t < T; t++) { pthread_join(th[t], NULL); *sink += jb[t].sink; }
    }
    return (now_s() - t0) / reps * 1e3;
}

static void run_shape(const char *name, int R, int C, int T) {
    const int B = 16;
    size_t wn = (size_t)R * C;
    uint16_t *W = malloc(wn * sizeof(uint16_t));
    float *X = malloc((size_t)C * B * sizeof(float));
    for (size_t i = 0; i < wn; i++) W[i] = f32_to_bf16(((float)(i % 17) - 8.0f) * 0.01f);
    for (int k = 0; k < C; k++) for (int b = 0; b < B; b++)
        X[(size_t)k * B + b] = ((float)(k % 13) - 6.0f) * 0.02f + b * 1e-4f;

    double wMB = wn * 2.0 / (1024 * 1024);
    int reps = wMB > 8 ? 30 : 300;
    volatile float sink = 0;
    sink += gemv_bf16(W, X, 0, R, C); sink += gemm16_bf16(W, X, 0, R, C);  /* warm */

    /* single-thread */
    double t0 = now_s();
    for (int it = 0; it < reps; it++) for (int b = 0; b < B; b++) sink += gemv_bf16(W, X + (size_t)b, 0, R, C);
    double tv1 = (now_s() - t0) / reps * 1e3;
    t0 = now_s();
    for (int it = 0; it < reps; it++) sink += gemm16_bf16(W, X, 0, R, C);
    double tg1 = (now_s() - t0) / reps * 1e3;

    /* threaded (realistic: cores contend for DRAM) */
    double tvT = timed_threaded(W, X, R, C, B, T, 0, reps, &sink);
    double tgT = timed_threaded(W, X, R, C, B, T, 1, reps, &sink);

    printf("  %-13s %5.1fMB | 1T: 16xGEMV %6.2f GEMM %6.2f = %4.2fx | %dT: 16xGEMV %6.2f GEMM %6.2f = %4.2fx\n",
           name, wMB, tv1, tg1, tv1 / tg1, T, tvT, tgT, tvT / tgT);
    free(W); free(X);
}

int main(void) {
    int T = 4;
    printf("=== BATCHING premise: 16x GEMV (re-read weights) vs GEMM(16) (read once) ===\n");
    printf("speedup>1 => batching amortizes weight reads (memory-bound). ~1 => compute-bound.\n");
    printf("The %dT column is the realistic regime (cores contend for M1 DRAM bandwidth).\n\n", T);
    run_shape("0.6B gate_up", 2816, 1024, T);
    run_shape("1.7B gate_up", 5632, 2048, T);
    run_shape("1.7B down",    2048, 5632, T);
    run_shape("BIG DRAM",     8192, 4096, T);
    return 0;
}
