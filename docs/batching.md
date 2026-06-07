# Text-chunk batching — premise test (feat/batching)

**Question:** if we split a long text into B chunks and step them *together* (so each bf16
weight is read from DRAM once and reused across all B chunks — matrix-MATRIX instead of
matrix-VECTOR), do we get throughput? This is the only lever the earlier analysis left open
(worker-pool concurrency was a dead-end: +8%, M1 bandwidth saturated by ONE synthesis).

## Microbench (`make batching-bench`, M1, single-thread + 4-thread)

`16× GEMV` (weights re-read 16×, as today) vs `GEMM(16)` (weights read once, FMA'd into 16
register-resident accumulators) on representative Talker weight shapes:

| shape | size | 1T speedup | 4T speedup (realistic) |
|---|---|---|---|
| 0.6B gate_up | 5.5 MB | 1.34× | 1.44× |
| 1.7B gate_up | 22 MB | 1.75× | 1.90× |
| 1.7B down | 22 MB | 2.01× | 2.04× |
| big (64 MB) | 64 MB | 1.98× | 1.88× |

## Verdict: the idea HOLDS — but the ceiling is ~1.5–2×, not B×

- A single bf16 GEMV reaches only ~12–16 GB/s effective on one M1 core — **well under** the
  ~60–100 GB/s the core can pull. So single-stream is **compute-bound on the NEON bf16 path**,
  not purely memory-bound. Batching amortizes the weight-read (~40–50% of the work), giving
  ~2×; the other ~50% is FMA compute that B chunks still have to do.
- The 4-thread (realistic) column is ~the same as 1-thread → threading already scales well and
  doesn't push us hard into the memory-bound regime where batching would pay 4–16×.
- This still **beats the worker-pool** dead-end (+8%): batched-GEMM is the right mechanism,
  confirming the prior conclusion. Just don't expect more than ~2× on M1.

## If we build it (the real prototype)

A ~2× throughput win requires a real rewrite — scope before committing:
1. **B independent sequences** in flight, each with its own Talker + CP KV cache.
2. **GEMM step kernels** (replace the per-token GEMV in `qwen_tts_kernels.c` with a
   register-blocked [out × B] matmul) for Talker QKV/O/gate_up/down AND the Code Predictor
   (CP is 90% matvec and runs 15×/frame — batching it matters most).
3. **Batched sampling / EOS** — chunks finish at different lengths; need ragged-batch handling
   (drop a chunk when it hits EOS, compact the batch).
4. **Chunk scheduler** — split long text on sentence boundaries, keep the batch full.
5. Output: re-stitch chunk audio in order (the `--compose`/`render_spans` concat already does
   seamless joins; reuse it).

**Recommendation:** worth it only for a *throughput/serving* goal (many requests, or one very
long document where 2× wall-time matters). For single short utterances it does nothing. The
gain is ~2× on M1; a higher-bandwidth box (where single-stream is more memory-bound) could see
more. Decide based on whether 2× justifies the rewrite + the per-sequence complexity.

Bench: `tests/batching_bench.c` (`make batching-bench`).
