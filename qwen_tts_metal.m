/*
 * qwen_tts_metal.m — Apple Metal backend (G2). Objective-C, build with
 * clang -fobjc-arc (see `make metal`). gcc cannot compile ObjC, so this is the
 * one TU compiled by clang; everything else stays gcc/plain-C.
 *
 * v1: two compute kernels (matvec_bf16, matmat_bf16), MSL compiled at runtime
 * from the embedded source below (llama.cpp's ggml-metal newLibraryWithSource
 * pattern). bf16 weights are reconstructed to f32 in-shader via a 16-bit left
 * shift into the float mantissa — correct for all finite bf16 and portable
 * across every Metal version (no reliance on the MSL `bfloat` type).
 *
 * Honest M1 note (plan_v4 §E4.ter): single-stream matvec on the shared-memory
 * M1 GPU is ~parity-or-worse vs CPU (bandwidth-bound); the win is the batched
 * matmat (compute-bound) and, later, decoder offload + CPU/GPU overlap. v1 uses
 * matvec only as the correctness vehicle for --gpu-selftest.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "qwen_tts_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- embedded Metal Shading Language ------------------------------------ */
static const char *QWEN_METAL_SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"inline float bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16); }\n"
"\n"
"kernel void matvec_bf16(\n"
"    device const ushort *W [[buffer(0)]],\n"
"    device const float  *x [[buffer(1)]],\n"
"    device float        *y [[buffer(2)]],\n"
"    constant uint       &cols [[buffer(3)]],\n"
"    uint row [[thread_position_in_grid]])\n"
"{\n"
"    float acc = 0.0f;\n"
"    device const ushort *w = W + (ulong)row * cols;\n"
"    for (uint c = 0; c < cols; ++c) acc += bf16_to_f32(w[c]) * x[c];\n"
"    y[row] = acc;\n"
"}\n"
"\n"
"kernel void matmat_bf16(\n"
"    device const ushort *W [[buffer(0)]],\n"
"    device const float  *X [[buffer(1)]],\n"   /* [cols, B] row-major */
"    device float        *Y [[buffer(2)]],\n"   /* [rows, B] row-major */
"    constant uint       &cols [[buffer(3)]],\n"
"    constant uint       &B    [[buffer(4)]],\n"
"    uint row [[thread_position_in_grid]])\n"
"{\n"
"    float acc[64];\n"
"    for (uint b = 0; b < B; ++b) acc[b] = 0.0f;\n"
"    device const ushort *w = W + (ulong)row * cols;\n"
"    for (uint c = 0; c < cols; ++c) {\n"
"        float wv = bf16_to_f32(w[c]);\n"
"        device const float *xc = X + (ulong)c * B;\n"
"        for (uint b = 0; b < B; ++b) acc[b] += wv * xc[b];\n"
"    }\n"
"    device float *yr = Y + (ulong)row * B;\n"
"    for (uint b = 0; b < B; ++b) yr[b] = acc[b];\n"
"}\n";

typedef struct {
    void *device;       /* id<MTLDevice>              (bridge-retained) */
    void *queue;        /* id<MTLCommandQueue>        (bridge-retained) */
    void *pso_matvec;   /* id<MTLComputePipelineState>(bridge-retained) */
    void *pso_matmat;   /* id<MTLComputePipelineState>(bridge-retained) */
} qwen_metal_ctx;

int qwen_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return dev != nil;
    }
}

void *qwen_metal_init(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "Metal: no system default device\n"); return NULL; }

        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:QWEN_METAL_SRC];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "Metal: shader compile failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "(unknown)");
            return NULL;
        }
        id<MTLFunction> fmv = [lib newFunctionWithName:@"matvec_bf16"];
        id<MTLFunction> fmm = [lib newFunctionWithName:@"matmat_bf16"];
        if (!fmv || !fmm) { fprintf(stderr, "Metal: kernel function missing\n"); return NULL; }

        id<MTLComputePipelineState> pmv = [dev newComputePipelineStateWithFunction:fmv error:&err];
        if (!pmv) { fprintf(stderr, "Metal: pso matvec failed: %s\n",
                            err ? err.localizedDescription.UTF8String : "(unknown)"); return NULL; }
        id<MTLComputePipelineState> pmm = [dev newComputePipelineStateWithFunction:fmm error:&err];
        if (!pmm) { fprintf(stderr, "Metal: pso matmat failed: %s\n",
                            err ? err.localizedDescription.UTF8String : "(unknown)"); return NULL; }

        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!q) { fprintf(stderr, "Metal: command queue failed\n"); return NULL; }

        qwen_metal_ctx *c = calloc(1, sizeof(*c));
        if (!c) return NULL;
        c->device     = (__bridge_retained void *)dev;
        c->queue      = (__bridge_retained void *)q;
        c->pso_matvec = (__bridge_retained void *)pmv;
        c->pso_matmat = (__bridge_retained void *)pmm;
        return c;
    }
}

void qwen_metal_free(void *ctx) {
    if (!ctx) return;
    qwen_metal_ctx *c = ctx;
    /* transfer ownership back to ARC so the objects are released */
    if (c->device)     { id o = (__bridge_transfer id)c->device;     (void)o; }
    if (c->queue)      { id o = (__bridge_transfer id)c->queue;      (void)o; }
    if (c->pso_matvec) { id o = (__bridge_transfer id)c->pso_matvec; (void)o; }
    if (c->pso_matmat) { id o = (__bridge_transfer id)c->pso_matmat; (void)o; }
    free(c);
}

static NSUInteger tpt_for(id<MTLComputePipelineState> pso) {
    NSUInteger t = pso.maxTotalThreadsPerThreadgroup;
    return t > 256 ? 256 : t;   /* one thread per output row; small groups fine */
}

void qwen_metal_matvec_bf16(void *ctx, float *y,
                            const uint16_t *W, const float *x,
                            int rows, int cols) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLDevice> dev = (__bridge id<MTLDevice>)c->device;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_matvec;

        id<MTLBuffer> bW = [dev newBufferWithBytes:W length:(NSUInteger)rows * cols * sizeof(uint16_t)
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bx = [dev newBufferWithBytes:x length:(NSUInteger)cols * sizeof(float)
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [dev newBufferWithLength:(NSUInteger)rows * sizeof(float)
                                            options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bx offset:0 atIndex:1];
        [enc setBuffer:by offset:0 atIndex:2];
        uint32_t ccols = (uint32_t)cols;
        [enc setBytes:&ccols length:sizeof(ccols) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpt_for(pso), 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(y, by.contents, (NSUInteger)rows * sizeof(float));
    }
}

void qwen_metal_matmat_bf16(void *ctx, float *Y,
                            const uint16_t *W, const float *X,
                            int rows, int cols, int B) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLDevice> dev = (__bridge id<MTLDevice>)c->device;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_matmat;

        id<MTLBuffer> bW = [dev newBufferWithBytes:W length:(NSUInteger)rows * cols * sizeof(uint16_t)
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bX = [dev newBufferWithBytes:X length:(NSUInteger)cols * B * sizeof(float)
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bY = [dev newBufferWithLength:(NSUInteger)rows * B * sizeof(float)
                                            options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:bY offset:0 atIndex:2];
        uint32_t ccols = (uint32_t)cols, cB = (uint32_t)B;
        [enc setBytes:&ccols length:sizeof(ccols) atIndex:3];
        [enc setBytes:&cB    length:sizeof(cB)    atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpt_for(pso), 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(Y, bY.contents, (NSUInteger)rows * B * sizeof(float));
    }
}
