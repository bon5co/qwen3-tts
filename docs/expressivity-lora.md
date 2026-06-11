# Expressivity `.expr` micro-files — technical reference

How a tiny, composable **expressivity delta** (`<lang>.expr`) makes any voice — preset or
cloned — speak a target language more naturally and emote, *without* a second multi-GB model.

> Status: technical/dev reference (captured 2026-06-11). An end-user-friendly version and a
> dev.to writeup are tracked as a final-phase task. This file is the substance.

Related: [`docs/expressivity.md`](expressivity.md) (the instruct + temperature recipe),
[`docs/custom-voices.md`](custom-voices.md), [`docs/icl-graft-portability.md`](icl-graft-portability.md)
(the `.qvoice-lite` reusable-clone format).

---

## 1. The idea: voice and expressivity are decoupled

```
a voice   =  .qvoice-lite  (ICL ref_codes / x-vector + metadata, ~24 MB)   ← WHO it is
          +  <lang>.expr   (an expressivity weight delta, 16–63 MB)        ← HOW it speaks
```

The two compose at load. The `.qvoice-lite` carries the **identity** (timbre); the `.expr`
carries **language-prosody + emotion competence**. Either can be swapped independently:
N cloned voices × M language `.expr` packs.

## 2. Which layers, and why (L16-26)

A weight-diff of the official models (Base vs CustomVoice, 1.7B) showed the instruct/expressivity
fine-tune is **not** spread across the network — it concentrates in **mid-late Talker layers
L16-26** (peak L18-20), specifically `mlp.gate_proj` + `self_attn.{q,k,v,o}_proj`, plus a touch of
`text_projection`. A per-layer activation map further localized emotion *identity* to **L21-25**.
The decoder (speech tokenizer) and the early/late layers were essentially untouched.

So we fine-tune **exactly those layers** and nothing else. Measured on our Italian fine-tune
(EMOVO), the changed set vs the base CustomVoice is:

| tensors | what | dtype in engine |
|---|---|---|
| 55 | `layers.{16..26}.self_attn.{q,k,v,o}_proj` + `mlp.gate_proj` | bf16 weight matrices |
| 17 | `layers.{16..26}.self_attn.{q,k}_norm` | f32 RMSNorm (route-a only) |

Everything else (404 talker tensors total, the decoder, embeddings, code-predictor) is **identical**
to the base model → it does not belong in a `.expr`.

**Crucially, `.expr` does not change *who* the voice is.** The decoder (the literal timbre engine)
is byte-identical; identity comes from the x-vector/qvoice. `.expr` changes only *how well/expressively*
the Talker drives that voice. That is why it composes with any voice without altering identity.

## 3. Two routes to the file — and why route-b is small

The full fine-tune is a 3.8 GB checkpoint. We never ship that. Two ways to extract just the change:

### Route A — dense delta (186 MB, bit-exact)
Store only the changed tensors as the **integer difference of their bf16 bit-patterns** + LZ4
(the same encoding as the `.qvoice` WDELTA section). Reconstruct `weight = base + delta` at load.
This is **bit-exact** to the full fine-tune (verified: 0/276.8M mismatched). 20× smaller than the
3.8 GB checkpoint — but still not tiny, because the full-fine-tune delta is **high-rank / diffuse**:
an SVD keeps only ~15-20 % of its energy at rank 16, so you *cannot* shrink it post-hoc by low-rank
truncation without losing the effect.

### Route B — LoRA factors (16–63 MB, the real win)
Instead of extracting a delta from a full fine-tune, **train a LoRA** on the same L16-26 modules.
A LoRA is *born* low-rank: for each weight `W [out,in]` it learns `A [r,in]` and `B [out,r]`, and the
update is `ΔW = (α/r)·B·A`. We store **only A and B** (+ the scale), and the engine reconstructs
`ΔW = scale·(B·A)` at load.

Why that's small — a `gate_proj` is `[6144,2048] = 12.6 M` numbers; at `r=16` its LoRA is
`(2048+6144)·16 = 131 K` numbers, ~96× fewer. Across the 55 matrices:

| file | rank | params | size | train loss (EMOVO, 8 ep) |
|---|---|---|---|---|
| full checkpoint | — | 1.92 B | 3.8 GB | (reference) |
| route-a dense `.expr` | (full) | 276.8 M | **186 MB** | = full FT, bit-exact |
| `italian_lora.expr` | 16 | 3.96 M (0.2 %) | **16 MB** | 5.5 |
| `italian_lora_r32.expr` | 32 | 7.9 M | **32 MB** | 4.9 |
| `italian_lora_r64.expr` | 64 | 15.9 M | **63 MB** | 4.34 |
| (r128 — overfits) | 128 | 31.7 M | 127 MB | 3.70 |

A trained low-rank LoRA reaches the objective *directly* in few parameters — and in practice the
rank constraint **focuses** the change into the dominant expressive direction (vs the full FT spreading
it thinly across 277 M params), so a good LoRA is often *more* expressive than the dense delta at a
fraction of the size.

## 4. Per-voice rank: rank compensates the clone's damping

A **preset** voice responds strongly to the delta; a **cloned** voice emotes ~3-4× weaker at the same
delta, because its x-vector identity signal damps the expressive response (a structural cap). The
fix is **rank**, not magnitude:

- **Preset → r32.** Plenty; r64 over-acts.
- **Clone → r64.** r32 is too mild on a clone; r64 makes a calm cloned voice genuinely raise its
  voice in anger. r128 *overfits* — it rushes/eats words on part of the emotion palette.

`--expr-weight` (below) scales magnitude, which on a clone only adds energy/speed (and runs away
above ~2), **not** expressivity — so rank is the right lever for the clone gap, weight is for fine taste.

## 5. File format and loader

`.expr` = a small header + a `WDLT` tensor stream:

```
"QEXP" | u32 version | char lang[16] | u32 reserved
"WDLT" | u32 hidden_size | u32 n_tensors
  per tensor: u16 name_len; name; u32 data_bytes; u8 dtype; u32 comp_size; payload
    dtype 0 = raw f32 replacement   (RMSNorm, route-a)
    dtype 4 = int16 bf16-bit delta + LZ4   (dense weights, route-a)
    dtype 5 = LoRA factors: u32 r,in,out + f32 scale + A[r·in] f32 + B[out·r] f32   (route-b)
```

The loader (`apply_expr_file()` in `main.c`) runs **after** the voice-load block and before any
generation/serve dispatch, so it covers single / batch / streaming / server. For each tensor it
resolves the name to the in-memory Talker weight pointer and applies:
- dtype 4: `new_bits = cur_bits + delta16` (mod 2^16, bit-exact on bf16);
- dtype 5: `new = f32_to_bf16( bf16_to_f32(cur) + scale·(B·A) )`, reconstructing `B·A` row-by-row.

Then it rebuilds the fused `gate_up` buffer (gate_proj changed) and re-quantizes INT8 if active.
The `B·A` reconstruction is a one-time ~sub-second cost at load.

## 6. CLI

```bash
# preset
./qwen_tts -d qwen3-tts-1.7b -s vivian -l Italian -T 1.1 \
  --expr presets/expr/italian_lora_r32.expr \
  --instruct "<vivid English instruction>" --text "<Italian text>" -o out.wav

# cloned voice (graft) — higher rank for the clone
./qwen_tts -d qwen3-tts-1.7b --load-voice voices/myvoice.qvoice --icl-only -l Italian -T 1.1 \
  --expr presets/expr/italian_lora_r64.expr \
  --instruct "<vivid English instruction>" --text "<Italian text>" -o out.wav
```

- `--expr <file>` — apply the expressivity delta on top of the loaded voice (1.7B only).
- `--expr-weight <m>` — scale a **factored-LoRA** `.expr` at load (1.0 = as trained, 0.6 = subtler;
  useful range ~0.5–1.1; dense `.expr` can't be scaled meaningfully). No retraining needed.

Recipe note: the spoken text is in the target language (`-l`), but the **instruct stays in English/
Chinese** (the model's instruct-following is EN/ZH-centric) — see `docs/expressivity.md`.
