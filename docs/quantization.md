# Weight Quantization

The `--int8` and `--int4` flags quantize Talker and Code Predictor (CP) weights at load time,
reducing memory usage and (for INT8) improving speed.

> **Updated 2026-06-03 — int8 now helps BOTH models.** The older claim that int8 had "no effect on
> 0.6B" was **wrong**: it only quantized the Talker, and the 0.6B Talker (hidden=1024) is too small
> to benefit. Once CP quantization was enabled (the CP is hidden=1024 and the bottleneck on **both**
> models), `--int8` wins big on **both**. With native int8 SDOT on Apple Silicon the 0.6B model goes
> **sub-realtime (RTF < 1.0) in every mode**: CLI ~0.90 short / **~0.80 long**, streaming ~0.81–0.89
> (first audio ~0.5 s), HTTP server warm ~0.88, cloned `.qvoice` ~0.93 — at near-bf16 quality. On 1.7B:
> **RTF 2.66 → 1.79 (−33%)**, Talker −23%, CP −29%. Full table in [Performance](performance.md).

> **x86 (updated 2026-06-05):** the int8 matvec now has **AVX2 + AVX-512/VNNI** twins (native
> `_mm512_dpbusd_epi32`) and decode runs on a cross-OS pthread pool — validated on a Ryzen 7 6800H
> (AVX2) and an EPYC 9555P / Zen5 (AVX-512/VNNI), where the int8 kernel is a ~1.85× win at equal core
> count. x86 single-stream RTF is memory/cache-bound (so it won't reach Apple's sub-1.0 without a
> cache-rich chip), but `--int8`/`--int4` are the right levers there too. Toggle SDOT/VNNI off with
> `QWEN_NO_SDOT=1` / `QWEN_NO_VNNI=1`. Measure your box: `bash tests/x86_bench.sh`. See PLAN.md 21.3.

## INT8 (Recommended on Apple Silicon, both models)

```bash
./qwen_tts -d qwen3-tts-1.7b --text "Hello world" --int8 -o hello.wav   # 1.7B: Talker+CP
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --int8 -o hello.wav   # 0.6B: CP (now a real win)
```

- Talker −23% (1.7B) + CP −29% (both models) with SDOT — reduced memory bandwidth + native int8 dot
- Good audio quality — minimal perceptual difference from BF16 (validated by ear, preset + custom voice)
- Halves Talker RAM usage on 1.7B (2.8 GB → 1.4 GB)
- Works with all features: server, streaming, custom voices (`.qvoice` re-quantized after override), instruct

## INT4 (Experimental)

```bash
./qwen_tts -d qwen3-tts-1.7b --text "Hello world" --int4 -o hello.wav
```

- Q4_0 format (4-bit with per-block scale factors)
- Smallest memory footprint (0.7 GB Talker RAM)
- Slightly **slower** than BF16 due to nibble unpacking overhead
- Audio quality may degrade on some inputs

## Comparison

**1.7B, Italian, seed=42, Apple M1 16 GB, 4 threads** (these rows predate SDOT — with SDOT the
INT8 Talker is ~46 ms/f, see the validated figures in the box above):

| Config | Talker ms/f | Total time | RTF | Talker RAM |
|--------|-------------|------------|-----|------------|
| BF16 (default) | ~80 ms/f | ~13s | ~4.3 | 2.8 GB (mmap) |
| **INT8 (recommended)** | **~67 ms/f** | **~11s** | **~3.6** | **1.4 GB** |
| INT4 (experimental) | ~83 ms/f | ~14s | ~4.5 | 0.7 GB |

## Recommendation

On Apple Silicon, use `--int8` for **both** models — Talker −23% (1.7B) and CP −29% (both) with
SDOT, and **0.6B goes sub-realtime (RTF < 1.0) in CLI/stream/server**. On x86 the int8 matvec now has
AVX2 + AVX-512/VNNI (validated on Ryzen 6800H and EPYC 9555P/Zen5) — `--int8` is the right default
there too; on a memory-starved CPU with a small L3, `--int4` can edge ahead multi-threaded
(see [x86 optimization](x86-optimization.md)).

INT4 is the lever on **memory-starved x86** (small L3 → fewer weight bytes wins, e.g. Ryzen 6800H
3.9→2.02). On **cache-rich / bandwidth-rich chips (Apple M1)** INT4 is *slower* than INT8 (nibble
unpacking dominates) — there INT8 is the quality/speed floor. Per-block-32 int4 scales are also a
touch coarse for the CP's fine residuals (slight timbre shift), so INT8 stays the quality reference.
For maximum speed, use the 0.6B model (RTF ~1.3–1.7 vs 3.6 for 1.7B INT8).

On systems with 16+ GB free RAM, expected performance is better than shown above
(our test machine had high system memory pressure from other applications).
Projected RTF with free RAM: **0.6B ~1.3, 1.7B BF16 ~3.0, 1.7B INT8 ~2.5**.

## Testing

```bash
make test-large-int8  # 1.7B INT8 tests (Italian + English, seed 42)
make test-large-int4  # 1.7B INT4 tests (Italian + English, seed 42)
make test-large-quant # All 1.7B quantization tests (INT8 + INT4)
```
