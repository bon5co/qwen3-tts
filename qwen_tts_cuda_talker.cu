/*
 * qwen_tts_cuda_talker.cu — GPU-RESIDENT fused Talker step (CUDA fused-forward epic, M1).
 *
 * The per-op matvec hook (qwen_tts_cuda.c) uploads/computes/downloads PER matvec → a
 * device sync every op. On the autoregressive decode that is ~140 syncs/frame and the
 * un-offloaded CPU ops (attn/norm/rope/swiglu) dominate anyway. This TU keeps WEIGHTS +
 * KV + activations RESIDENT on the device and runs the WHOLE Talker step as a chain of
 * kernels + cuBLAS calls with a SINGLE sync at the end (download the final hidden). Only
 * the input embed goes in and the hidden comes out.
 *
 * M1 = correctness-first: resident fp32 weights + cublasSgemm (row-major→col-major mapping
 * VALIDATED by the matmat selftest, rel 1.9e-4). bf16 tensor-core (cublasGemmEx) is M4.
 * Kernels match the CPU semantics EXACTLY (qwen_tts_talker.c / qwen_tts_kernels.c):
 *   - SwiGLU: interleaved gate/up (gate_up[2i], gate_up[2i+1])         [k_swiglu_il]
 *   - RoPE: NeoX SPLIT-HALF (xh[i],xh[i+half]), NOT interleaved         [k_rope_neox]
 *   - per-head RMSNorm on Q/K with a shared [head_dim] weight           [k_rmsnorm_ph]
 *   - causal GQA attention, online softmax (flash-style)               [k_attn]
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" {
#include "qwen_tts.h"
}

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));} } while(0)
#define TPB 256
#define CEIL(a,b) (((a)+(b)-1)/(b))

/* ---- kernels (device pointers, NO per-op copy) -------------------------- */

__global__ void k_rmsnorm_full(const float *x, const float *w, float *y, int dim, float eps) {
    extern __shared__ float part[];
    int tid = threadIdx.x, tc = blockDim.x;
    float s = 0.f; for (int i=tid;i<dim;i+=tc) s += x[i]*x[i];
    part[tid]=s; __syncthreads();
    for (int st=tc/2; st>0; st>>=1){ if(tid<st) part[tid]+=part[tid+st]; __syncthreads(); }
    float inv = rsqrtf(part[0]/(float)dim + eps);
    for (int i=tid;i<dim;i+=tc) y[i] = x[i]*inv*w[i];
}

/* per-head RMSNorm: one block per head, weight w[head_dim] shared across heads. */
__global__ void k_rmsnorm_ph(float *x, const float *w, int head_dim, float eps) {
    extern __shared__ float part[];
    int h = blockIdx.x, tid = threadIdx.x, tc = blockDim.x;
    float *xh = x + (size_t)h*head_dim;
    float s = 0.f; for (int i=tid;i<head_dim;i+=tc) s += xh[i]*xh[i];
    part[tid]=s; __syncthreads();
    for (int st=tc/2; st>0; st>>=1){ if(tid<st) part[tid]+=part[tid+st]; __syncthreads(); }
    float inv = rsqrtf(part[0]/(float)head_dim + eps);
    for (int i=tid;i<head_dim;i+=tc) xh[i] = xh[i]*inv*w[i];
}

/* NeoX split-half RoPE, cos/sin at position pos (half = head_dim/2). */
__global__ void k_rope_neox(float *x, const float *cosp, const float *sinp,
                            int n_heads, int head_dim) {
    int half = head_dim/2;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= n_heads*half) return;
    int h = gid/half, i = gid%half;
    float *xh = x + (size_t)h*head_dim;
    float c = cosp[i], s = sinp[i];
    float x1 = xh[i], x2 = xh[i+half];
    xh[i]      = x1*c - x2*s;
    xh[i+half] = x2*c + x1*s;
}

__global__ void k_swiglu_il(const float *in, float *out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=n) return;
    float g = in[2*i], u = in[2*i+1];
    out[i] = g/(1.f+expf(-g))*u;
}

__global__ void k_add_ip(float *a, const float *b, int n) {  /* a += b */
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i<n) a[i]+=b[i];
}

/* round-trip f32→bf16→f32 (truncate mantissa) to MATCH the CPU's bf16 KV cache. */
__global__ void k_trunc_bf16(float *x, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=n) return;
    uint32_t u = __float_as_uint(x[i]);
    x[i] = __uint_as_float(u & 0xFFFF0000u);   /* truncate low 16 bits (CPU uses truncation) */
}

/* causal GQA attention, online softmax (flash-style). Q[1,n_heads,hd], K/V[seq_k,n_kv,hd]. */
__global__ void k_attn(const float *Q, const float *K, const float *V, float *O,
                       int seq_k, int n_heads, int n_kv, int hd, float scale, int qpos) {
    int h = blockIdx.x; if (h>=n_heads) return;
    int kvh = h/(n_heads/n_kv);
    int valid = qpos+1; if (valid>seq_k) valid=seq_k;
    const float *q = Q + (size_t)h*hd;
    float *o = O + (size_t)h*hd;
    float m=-1e30f;
    for (int j=0;j<valid;++j){ const float *k=K+((size_t)j*n_kv+kvh)*hd; float d=0;
        for(int t=0;t<hd;++t) d+=q[t]*k[t]; d*=scale; if(d>m)m=d; }
    for(int t=0;t<hd;++t) o[t]=0; float den=0;
    for (int j=0;j<valid;++j){ const float *k=K+((size_t)j*n_kv+kvh)*hd; float d=0;
        for(int t=0;t<hd;++t) d+=q[t]*k[t]; d=expf(d*scale-m); den+=d;
        const float *v=V+((size_t)j*n_kv+kvh)*hd; for(int t=0;t<hd;++t) o[t]+=d*v[t]; }
    float inv=1.f/den; for(int t=0;t<hd;++t) o[t]*=inv;
}

/* ---- resident state ----------------------------------------------------- */

typedef struct {
    cublasHandle_t handle;
    int hidden, q_dim, kv_dim, inter, n_heads, n_kv, head_dim, n_layers, kv_max;
    float eps;
    /* resident weights (device fp32), per layer */
    float **wq,**wk,**wv,**wo,**wgu,**wdn,**inorm,**pnorm,**qn,**kn;
    float *tnorm;                            /* final talker RMSNorm weight [hidden] */
    float *rope_cos,*rope_sin;               /* [kv_max*half] */
    float *kcache,*vcache;                   /* [n_layers*kv_max*kv_dim] */
    float *x,*xn,*q,*k,*v,*attn,*proj,*gate,*gu; /* work buffers (gu = [2*inter]) */
} cuda_talker_t;

static float *up_bf16(const uint16_t *w, size_t n) {
    float *h=(float*)malloc(n*sizeof(float));
    for(size_t i=0;i<n;++i){ union{uint32_t u;float f;}v; v.u=(uint32_t)w[i]<<16; h[i]=v.f; }
    float *d=NULL; CK(cudaMalloc(&d,n*sizeof(float)));
    CK(cudaMemcpy(d,h,n*sizeof(float),cudaMemcpyHostToDevice)); free(h); return d;
}
static float *up_f32(const float *w, size_t n) {
    float *d=NULL; CK(cudaMalloc(&d,n*sizeof(float)));
    CK(cudaMemcpy(d,w,n*sizeof(float),cudaMemcpyHostToDevice)); return d;
}

extern "C" void *qwen_cuda_talker_init(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c=&ctx->config;
    cuda_talker_t *s=(cuda_talker_t*)calloc(1,sizeof(*s));
    if (cublasCreate(&s->handle)!=CUBLAS_STATUS_SUCCESS){ free(s); return NULL; }
    /* Force TRUE fp32 accumulate (no TF32 tensor-core rounding) so the fused step matches the
     * CPU bf16-weight/f32-accumulate matvec. TF32 (10-bit mantissa) drifts ~5e-3 over 28 layers.
     * bf16 tensor-core is a SEPARATE opt-in speed path (M4), validated against this fp32 ref. */
    cublasSetMathMode(s->handle, CUBLAS_PEDANTIC_MATH);
    s->hidden=c->hidden_size; s->n_heads=c->num_heads; s->n_kv=c->num_kv_heads;
    s->head_dim=c->head_dim; s->inter=c->intermediate_size; s->n_layers=c->num_layers;
    s->q_dim=c->num_heads*c->head_dim; s->kv_dim=c->num_kv_heads*c->head_dim;
    s->eps=c->rms_norm_eps; s->kv_max=ctx->kv_max;
    int L=s->n_layers, H=s->hidden, hd=s->head_dim, half=hd/2;
    s->wq=(float**)calloc(L,sizeof(float*)); s->wk=(float**)calloc(L,sizeof(float*));
    s->wv=(float**)calloc(L,sizeof(float*)); s->wo=(float**)calloc(L,sizeof(float*));
    s->wgu=(float**)calloc(L,sizeof(float*)); s->wdn=(float**)calloc(L,sizeof(float*));
    s->inorm=(float**)calloc(L,sizeof(float*)); s->pnorm=(float**)calloc(L,sizeof(float*));
    s->qn=(float**)calloc(L,sizeof(float*)); s->kn=(float**)calloc(L,sizeof(float*));
    for (int l=0;l<L;++l){
        qwen_talker_layer_t *ly=&ctx->layers[l];
        if (!ly->wq_bf16 || !ly->gate_up_fused_bf16){ fprintf(stderr,"CUDA talker: layer %d not bf16 (int8/q4 unsupported in M1)\n",l); return NULL; }
        s->wq[l]=up_bf16(ly->wq_bf16,(size_t)s->q_dim*H);
        s->wk[l]=up_bf16(ly->wk_bf16,(size_t)s->kv_dim*H);
        s->wv[l]=up_bf16(ly->wv_bf16,(size_t)s->kv_dim*H);
        s->wo[l]=up_bf16(ly->wo_bf16,(size_t)H*s->q_dim);
        s->wgu[l]=up_bf16(ly->gate_up_fused_bf16,(size_t)2*s->inter*H);
        s->wdn[l]=up_bf16(ly->down_bf16,(size_t)H*s->inter);
        s->inorm[l]=up_f32(ly->input_norm,H); s->pnorm[l]=up_f32(ly->post_attn_norm,H);
        s->qn[l]=up_f32(ly->q_norm,hd); s->kn[l]=up_f32(ly->k_norm,hd);
    }
    s->tnorm=up_f32(ctx->talker_norm,H);
    s->rope_cos=up_f32(ctx->rope_cos,(size_t)s->kv_max*half);
    s->rope_sin=up_f32(ctx->rope_sin,(size_t)s->kv_max*half);
    CK(cudaMalloc(&s->kcache,(size_t)L*s->kv_max*s->kv_dim*sizeof(float)));
    CK(cudaMalloc(&s->vcache,(size_t)L*s->kv_max*s->kv_dim*sizeof(float)));
    CK(cudaMalloc(&s->x,H*sizeof(float)));  CK(cudaMalloc(&s->xn,H*sizeof(float)));
    CK(cudaMalloc(&s->q,s->q_dim*sizeof(float))); CK(cudaMalloc(&s->k,s->kv_dim*sizeof(float)));
    CK(cudaMalloc(&s->v,s->kv_dim*sizeof(float))); CK(cudaMalloc(&s->attn,s->q_dim*sizeof(float)));
    CK(cudaMalloc(&s->proj,H*sizeof(float))); CK(cudaMalloc(&s->gate,s->inter*sizeof(float)));
    CK(cudaMalloc(&s->gu,(size_t)2*s->inter*sizeof(float)));
    fprintf(stderr,"CUDA talker: resident fused step ready (%d layers, hidden=%d, fp32 weights)\n",L,H);
    return s;
}

/* dY[rows] = W[rows,cols] @ dX[cols], W row-major (col-major [cols,rows]); B=1 mapping
 * matches the VALIDATED matmat selftest. */
static inline void mv(cuda_talker_t *s, float *dY, const float *dW, const float *dX, int rows, int cols) {
    const float a=1.f,b=0.f;
    cublasSgemm(s->handle,CUBLAS_OP_N,CUBLAS_OP_N,1,rows,cols,&a,dX,1,dW,cols,&b,dY,1);
}

/* Run one resident Talker step. embed[hidden] in, hidden_out[hidden] out. pos = KV position. */
extern "C" void qwen_cuda_talker_step(void *st, const float *embed, float *hidden_out, int pos) {
    cuda_talker_t *s=(cuda_talker_t*)st;
    int H=s->hidden, qd=s->q_dim, kvd=s->kv_dim, hd=s->head_dim, half=hd/2;
    int nh=s->n_heads, nkv=s->n_kv, inter=s->inter;
    float scale=1.f/sqrtf((float)hd);
    CK(cudaMemcpy(s->x,embed,H*sizeof(float),cudaMemcpyHostToDevice));
    for (int l=0;l<s->n_layers;++l){
        /* 1. input RMSNorm */
        k_rmsnorm_full<<<1,TPB,TPB*sizeof(float)>>>(s->x,s->inorm[l],s->xn,H,s->eps);
        /* 2. QKV */
        mv(s,s->q,s->wq[l],s->xn,qd,H);
        mv(s,s->k,s->wk[l],s->xn,kvd,H);
        mv(s,s->v,s->wv[l],s->xn,kvd,H);
        /* 3. per-head Q/K RMSNorm */
        k_rmsnorm_ph<<<nh, TPB, TPB*sizeof(float)>>>(s->q,s->qn[l],hd,s->eps);
        k_rmsnorm_ph<<<nkv,TPB, TPB*sizeof(float)>>>(s->k,s->kn[l],hd,s->eps);
        /* 4. NeoX RoPE */
        const float *cosp=s->rope_cos+(size_t)pos*half, *sinp=s->rope_sin+(size_t)pos*half;
        k_rope_neox<<<CEIL(nh*half,TPB),TPB>>>(s->q,cosp,sinp,nh,hd);
        k_rope_neox<<<CEIL(nkv*half,TPB),TPB>>>(s->k,cosp,sinp,nkv,hd);
        /* 5. append K,V to resident cache at pos — TRUNCATE to bf16 first to MATCH the CPU's
         *    bf16 KV cache (f32_to_bf16 = bits>>16 = truncation), else attention diverges ~1e-2. */
        k_trunc_bf16<<<CEIL(kvd,TPB),TPB>>>(s->k,kvd);
        k_trunc_bf16<<<CEIL(kvd,TPB),TPB>>>(s->v,kvd);
        float *kc=s->kcache+((size_t)l*s->kv_max+pos)*kvd;
        float *vc=s->vcache+((size_t)l*s->kv_max+pos)*kvd;
        CK(cudaMemcpyAsync(kc,s->k,kvd*sizeof(float),cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(vc,s->v,kvd*sizeof(float),cudaMemcpyDeviceToDevice));
        /* 6. causal GQA attention over [0..pos] */
        float *Kl=s->kcache+(size_t)l*s->kv_max*kvd, *Vl=s->vcache+(size_t)l*s->kv_max*kvd;
        k_attn<<<nh,1>>>(s->q,Kl,Vl,s->attn,pos+1,nh,nkv,hd,scale,pos);
        /* 7. O proj */
        mv(s,s->proj,s->wo[l],s->attn,H,qd);
        /* 8. residual + post-attn RMSNorm */
        k_add_ip<<<CEIL(H,TPB),TPB>>>(s->x,s->proj,H);
        k_rmsnorm_full<<<1,TPB,TPB*sizeof(float)>>>(s->x,s->pnorm[l],s->xn,H,s->eps);
        /* 9. fused gate_up (2*inter) → SwiGLU (interleaved) → gate (inter) */
        mv(s,s->gu,s->wgu[l],s->xn,2*inter,H);
        k_swiglu_il<<<CEIL(inter,TPB),TPB>>>(s->gu,s->gate,inter);
        /* 10. down proj → proj (hidden) */
        mv(s,s->proj,s->wdn[l],s->gate,H,inter);
        /* 11. residual add into x */
        k_add_ip<<<CEIL(H,TPB),TPB>>>(s->x,s->proj,H);
    }
    /* Final talker RMSNorm → xn (matches CPU qwen_talker_step tail) */
    k_rmsnorm_full<<<1,TPB,TPB*sizeof(float)>>>(s->x,s->tnorm,s->xn,H,s->eps);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hidden_out,s->xn,H*sizeof(float),cudaMemcpyDeviceToHost));
}

extern "C" void qwen_cuda_talker_free(void *st) {
    cuda_talker_t *s=(cuda_talker_t*)st; if(!s) return;
    for(int l=0;l<s->n_layers;++l){ cudaFree(s->wq[l]);cudaFree(s->wk[l]);cudaFree(s->wv[l]);
        cudaFree(s->wo[l]);cudaFree(s->wgu[l]);cudaFree(s->wdn[l]);cudaFree(s->inorm[l]);
        cudaFree(s->pnorm[l]);cudaFree(s->qn[l]);cudaFree(s->kn[l]); }
    free(s->wq);free(s->wk);free(s->wv);free(s->wo);free(s->wgu);free(s->wdn);
    free(s->inorm);free(s->pnorm);free(s->qn);free(s->kn);
    cudaFree(s->tnorm);cudaFree(s->rope_cos);cudaFree(s->rope_sin);cudaFree(s->kcache);cudaFree(s->vcache);
    cudaFree(s->x);cudaFree(s->xn);cudaFree(s->q);cudaFree(s->k);cudaFree(s->v);
    cudaFree(s->attn);cudaFree(s->proj);cudaFree(s->gate);cudaFree(s->gu);
    if(s->handle) cublasDestroy(s->handle);
    free(s);
}
