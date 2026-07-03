/*
 * qwen_tts_cuda_talker.cu — GPU-RESIDENT fused Talker step (CUDA fused-forward epic, M1).
 *
 * The per-op matvec hook (qwen_tts_cuda.c) uploads/computes/downloads PER matvec → a device
 * sync every op. This TU keeps WEIGHTS + KV + activations RESIDENT on the device and runs the
 * WHOLE Talker step as a chain of kernels with a SINGLE sync at the end. Only the input embed
 * goes in and the hidden comes out.
 *
 * KEY perf insight (measured on GB10): single-token decode is BANDWIDTH-BOUND on the weight
 * reads (1.7B ≈ 3.4 GB bf16 / step). fp32-resident weights read 2× the bytes → no speedup vs the
 * per-op path. So weights stay **bf16** on the device and a custom bf16 matvec reads them at 2
 * bytes/elem while keeping the ACTIVATION in fp32 — exactly the CPU's bf16-weight × f32-act
 * semantics (no extra precision loss, unlike cublasGemmEx which needs bf16 activations too).
 *
 * Kernels match the CPU semantics EXACTLY (qwen_tts_talker.c / qwen_tts_kernels.c):
 *   - matvec: bf16 weight (row-major [rows,cols]) × f32 activation, f32 accumulate  [k_matvec_bf16]
 *   - SwiGLU: interleaved gate/up (gate_up[2i], gate_up[2i+1])                        [k_swiglu_il]
 *   - RoPE: NeoX SPLIT-HALF (xh[i],xh[i+half]), NOT interleaved                       [k_rope_neox]
 *   - per-head RMSNorm on Q/K with a shared [head_dim] weight                         [k_rmsnorm_ph]
 *   - causal GQA attention, online softmax (flash-style)                             [k_attn]
 *   - bf16-TRUNCATED KV cache (CPU f32_to_bf16 = bits>>16 = truncation)              [k_trunc_bf16]
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

/* y[rows] = W[rows,cols] @ x[cols].  W row-major bf16, x/y f32.  One block per output row;
 * threads stride over cols, warp/block reduce. Reads bf16 weights (half the bytes of fp32). */
__global__ void k_matvec_bf16(const __nv_bfloat16 *W, const float *x, float *y, int rows, int cols) {
    int row = blockIdx.x; if (row >= rows) return;
    const __nv_bfloat16 *wr = W + (size_t)row * cols;
    extern __shared__ float red[];
    float s = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) s += __bfloat162float(wr[i]) * x[i];
    red[threadIdx.x] = s; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) { if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x+st]; __syncthreads(); }
    if (threadIdx.x == 0) y[row] = red[0];
}

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
__global__ void k_rope_neox(float *x, const float *cosp, const float *sinp, int n_heads, int head_dim) {
    int half = head_dim/2;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= n_heads*half) return;
    int h = gid/half, i = gid%half;
    float *xh = x + (size_t)h*head_dim;
    float c = cosp[i], sn = sinp[i];
    float x1 = xh[i], x2 = xh[i+half];
    xh[i] = x1*c - x2*sn; xh[i+half] = x2*c + x1*sn;
}

__global__ void k_swiglu_il(const float *in, float *out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=n) return;
    float g = in[2*i], u = in[2*i+1];
    out[i] = g/(1.f+expf(-g))*u;
}

__global__ void k_add_ip(float *a, const float *b, int n) {  /* a += b */
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i<n) a[i]+=b[i];
}

/* round-trip f32→bf16→f32 (truncate mantissa) to MATCH the CPU's bf16 KV cache (bits>>16). */
__global__ void k_trunc_bf16(float *x, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=n) return;
    uint32_t u = __float_as_uint(x[i]);
    x[i] = __uint_as_float(u & 0xFFFF0000u);
}

/* causal GQA attention, online softmax (flash-style). Q[1,n_heads,hd], K/V[seq_k,n_kv,hd].
 * One block per (head), threads split head_dim. */
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
    int hidden, q_dim, kv_dim, inter, n_heads, n_kv, head_dim, n_layers, kv_max;
    float eps;
    __nv_bfloat16 **wq,**wk,**wv,**wo,**wgu,**wdn;   /* resident bf16 weights, per layer */
    float **inorm,**pnorm,**qn,**kn;                 /* f32 norms, per layer */
    float *tnorm, *rope_cos, *rope_sin;
    float *kcache,*vcache;                           /* [n_layers*kv_max*kv_dim] f32 */
    float *x,*xn,*q,*k,*v,*attn,*proj,*gate,*gu;     /* work buffers */
} cuda_talker_t;

static __nv_bfloat16 *up_bf16(const uint16_t *w, size_t n) {
    __nv_bfloat16 *d=NULL; CK(cudaMalloc(&d,n*sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(d,w,n*sizeof(uint16_t),cudaMemcpyHostToDevice));  /* bf16 bits == uint16 bits */
    return d;
}
static float *up_f32(const float *w, size_t n) {
    float *d=NULL; CK(cudaMalloc(&d,n*sizeof(float)));
    CK(cudaMemcpy(d,w,n*sizeof(float),cudaMemcpyHostToDevice)); return d;
}

extern "C" void *qwen_cuda_talker_init(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c=&ctx->config;
    cuda_talker_t *s=(cuda_talker_t*)calloc(1,sizeof(*s));
    s->hidden=c->hidden_size; s->n_heads=c->num_heads; s->n_kv=c->num_kv_heads;
    s->head_dim=c->head_dim; s->inter=c->intermediate_size; s->n_layers=c->num_layers;
    s->q_dim=c->num_heads*c->head_dim; s->kv_dim=c->num_kv_heads*c->head_dim;
    s->eps=c->rms_norm_eps; s->kv_max=ctx->kv_max;
    int L=s->n_layers, H=s->hidden, hd=s->head_dim, half=hd/2;
    s->wq=(__nv_bfloat16**)calloc(L,sizeof(void*)); s->wk=(__nv_bfloat16**)calloc(L,sizeof(void*));
    s->wv=(__nv_bfloat16**)calloc(L,sizeof(void*)); s->wo=(__nv_bfloat16**)calloc(L,sizeof(void*));
    s->wgu=(__nv_bfloat16**)calloc(L,sizeof(void*)); s->wdn=(__nv_bfloat16**)calloc(L,sizeof(void*));
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
    fprintf(stderr,"CUDA talker: resident fused step ready (%d layers, hidden=%d, bf16 weights)\n",L,H);
    return s;
}

static inline void mv(const __nv_bfloat16 *dW, const float *dX, float *dY, int rows, int cols) {
    k_matvec_bf16<<<rows, TPB, TPB*sizeof(float)>>>(dW, dX, dY, rows, cols);
}

extern "C" void qwen_cuda_talker_step(void *st, const float *embed, float *hidden_out, int pos) {
    cuda_talker_t *s=(cuda_talker_t*)st;
    int H=s->hidden, qd=s->q_dim, kvd=s->kv_dim, hd=s->head_dim, half=hd/2;
    int nh=s->n_heads, nkv=s->n_kv, inter=s->inter;
    float scale=1.f/sqrtf((float)hd);
    CK(cudaMemcpy(s->x,embed,H*sizeof(float),cudaMemcpyHostToDevice));
    for (int l=0;l<s->n_layers;++l){
        k_rmsnorm_full<<<1,TPB,TPB*sizeof(float)>>>(s->x,s->inorm[l],s->xn,H,s->eps);
        mv(s->wq[l],s->xn,s->q,qd,H);
        mv(s->wk[l],s->xn,s->k,kvd,H);
        mv(s->wv[l],s->xn,s->v,kvd,H);
        k_rmsnorm_ph<<<nh, TPB, TPB*sizeof(float)>>>(s->q,s->qn[l],hd,s->eps);
        k_rmsnorm_ph<<<nkv,TPB, TPB*sizeof(float)>>>(s->k,s->kn[l],hd,s->eps);
        const float *cosp=s->rope_cos+(size_t)pos*half, *sinp=s->rope_sin+(size_t)pos*half;
        k_rope_neox<<<CEIL(nh*half,TPB),TPB>>>(s->q,cosp,sinp,nh,hd);
        k_rope_neox<<<CEIL(nkv*half,TPB),TPB>>>(s->k,cosp,sinp,nkv,hd);
        k_trunc_bf16<<<CEIL(kvd,TPB),TPB>>>(s->k,kvd);
        k_trunc_bf16<<<CEIL(kvd,TPB),TPB>>>(s->v,kvd);
        float *kc=s->kcache+((size_t)l*s->kv_max+pos)*kvd;
        float *vc=s->vcache+((size_t)l*s->kv_max+pos)*kvd;
        CK(cudaMemcpyAsync(kc,s->k,kvd*sizeof(float),cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(vc,s->v,kvd*sizeof(float),cudaMemcpyDeviceToDevice));
        float *Kl=s->kcache+(size_t)l*s->kv_max*kvd, *Vl=s->vcache+(size_t)l*s->kv_max*kvd;
        k_attn<<<nh,1>>>(s->q,Kl,Vl,s->attn,pos+1,nh,nkv,hd,scale,pos);
        mv(s->wo[l],s->attn,s->proj,H,qd);
        k_add_ip<<<CEIL(H,TPB),TPB>>>(s->x,s->proj,H);
        k_rmsnorm_full<<<1,TPB,TPB*sizeof(float)>>>(s->x,s->pnorm[l],s->xn,H,s->eps);
        mv(s->wgu[l],s->xn,s->gu,2*inter,H);
        k_swiglu_il<<<CEIL(inter,TPB),TPB>>>(s->gu,s->gate,inter);
        mv(s->wdn[l],s->gate,s->proj,H,inter);
        k_add_ip<<<CEIL(H,TPB),TPB>>>(s->x,s->proj,H);
    }
    k_rmsnorm_full<<<1,TPB,TPB*sizeof(float)>>>(s->x,s->tnorm,s->xn,H,s->eps);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hidden_out,s->xn,H*sizeof(float),cudaMemcpyDeviceToHost));
}

extern "C" void qwen_cuda_talker_upload_kv(void *state, qwen_tts_ctx_t *ctx, int prefill_len) {
    cuda_talker_t *s=(cuda_talker_t*)state; if(!s||prefill_len<=0) return;
    int kvd=s->kv_dim, L=s->n_layers, kvm=s->kv_max;
    size_t nper=(size_t)prefill_len*kvd;
    float *hk=(float*)malloc(nper*sizeof(float)), *hv=(float*)malloc(nper*sizeof(float));
    for (int l=0;l<L;++l){
        const uint16_t *ck=ctx->kv_cache_k+(size_t)l*kvm*kvd;
        const uint16_t *cv=ctx->kv_cache_v+(size_t)l*kvm*kvd;
        for (size_t i=0;i<nper;++i){ union{uint32_t u;float f;}a,b;
            a.u=(uint32_t)ck[i]<<16; hk[i]=a.f; b.u=(uint32_t)cv[i]<<16; hv[i]=b.f; }
        CK(cudaMemcpy(s->kcache+(size_t)l*kvm*kvd, hk, nper*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(s->vcache+(size_t)l*kvm*kvd, hv, nper*sizeof(float), cudaMemcpyHostToDevice));
    }
    free(hk); free(hv);
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
    free(s);
}
