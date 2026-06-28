---
license: cc-by-4.0
language:
  - it
  - de
  - fr
tags:
  - text-to-speech
  - qwen3-tts
  - emotion
  - paralinguistics
  - expressivity
---

# qwen3-tts expressivity assets (emotion + paralinguistics)

Composable expressivity add-ons for **[qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts)**
(the pure-C inference engine for Qwen3-TTS). These are **plugins layered on top of normal synthesis** —
they do not replace the base model. All for the **1.7B** model.

## What's here — organized by purpose

| folder | what it does | type | size |
|---|---|---|---|
| `expr/` | **emotion fine-tunes** (per-language CSP weight-deltas) — apply with `--expr` | `.expr` weight-delta | 30–203 MB each |
| `steer/emotion/` | **emotion steering vectors** — apply with `--ml-steer` | `.qlsteer` activation dir | ~232 KB each |
| `steer/paraling/` | **paralinguistic vectors** (laugh / sigh) — apply with `--ml-steer` | `.qlsteer` | ~232 KB each |

> The tiny `steer/` vectors also ship inside the GitHub repo (`presets/steer/`). The big `expr/`
> files live here on HF because they are too large for git.

### `expr/` — emotion fine-tunes
- `italian_csp_topk6.expr` — **Italian emotion default** (cleanest on presets + clones).
- `german_csp_k6.expr`, `french_csp_k6.expr` — native German / French emotion.
- `italian_csp_topk4.expr`, `italian_l1626_dense.expr` (+ `_r32`/`_r64`, `multi`/`multitag`) — earlier Italian variants (A/B / research).

### `steer/emotion/` — `ryan_{ang,sad,joy,fear,disgust,surprise}`, `galatea_{ang,sad}_ft`, `vivian_{ang,sad}_ft`
### `steer/paraling/` — `laugh_vs_cry`, `sigh_vs_laugh` (+ `.qamp` source captures to rebuild)

## How to download
```bash
# from the qwen3-tts repo:
bash download_assets.sh            # fetches expr/ into presets/expr/ (sha256-verified)
# or grab a single file:
curl -L -o presets/expr/italian_csp_topk6.expr \
  https://huggingface.co/gabrione/qwen3-tts-italian-expr/resolve/main/expr/italian_csp_topk6.expr
```
Disk: the full `expr/` set ≈ 1.4 GB; Italian-only emotion needs just `italian_csp_topk6.expr` (203 MB).

## How to activate
See **[docs/expressivity-assets.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/expressivity-assets.md)**
for full recipes. Short version:

**Emotion** (steer is the main lever; `w8` sweet spot, `w12` over-steers):
```bash
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian -T 1.1 --text "..." \
  --ml-steer presets/steer/emotion/ryan_sad.qlsteer --ml-range 21-25 --ml-weight 8
# far language / language-drifting voice → add  --expr presets/expr/italian_csp_topk6.expr --expr-weight 1.0
```

**Paralinguistics (laugh / sigh)** — needs the onomatopoeia inline in the text **plus** the vector:
```bash
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian -T 1.1 \
  --text "Haaah... che giornata, haaah." \
  --ml-steer presets/steer/paraling/sigh_vs_laugh.qlsteer --ml-range 21-25 --ml-weight 6
# laugh: put 'ahah' in the text + sigh_vs_laugh → laugh_vs_cry
```
Per-voice paralinguistic weight: galatea 8 · vivian 8 · ryan 6.

## License / attribution
**CC-BY 4.0.** The Italian emotion fine-tune uses the **Emozionalmente** dataset — please cite:
F. Catania, J. W. Wilke, F. Garzotto (Politecnico di Milano), *IEEE TASLP* 33:1142-1155, 2025,
doi:10.1109/TASLPRO.2025.3540662.
