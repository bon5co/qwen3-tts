# Expressivity recipes — validated 2026-06-06 (ear-confirmed)

Practical, ear-validated recipes for emotional/expressive delivery on top of the
control-vector engine (`--emotion`, `--roughness`, `--steer-weight`). These extend
[expressivity.md](expressivity.md) with what actually works per mood and per language,
plus the dead-ends (so nobody re-walks them).

> **Golden rule learned the hard way:** judge emotion BY EAR. Acoustic proxies (ΔF0, F0std,
> spectral flatness) measure *how much* delivery moves, NOT *which* emotion it reads as — they
> mislabeled basins 3×. Use the numbers to scan, the ear to decide.

## The reliable palette: `presets/emotions/it_centered/`

The shipped IT palette was highly collinear (tones sounded alike: mean pairwise cosine **+0.57**).
The **mean-centered** palette (`tests/steer_center.py`, cosine → **−0.09**, ~2× contrast) is the one
to use. Point at it with `QWEN_EMOTION_DIR=presets/emotions/it_centered`.

## Per-mood recipes (the wins)

| mood | recipe | why |
|---|---|---|
| **joy** | `--emotion excited --steer-weight 2.6` + faster (`atempo 1.10`) + louder (`volume 1.10`) | `happy` LOSES energy when pushed (neutral is already upbeat) → **use `excited` for joy, never `happy`** |
| **sad** | `--emotion sad --steer-weight ~2.0` + **slower** (`atempo 0.82–0.85`) + **pauses** (`...`/commas) + quieter | down-moods don't respond to more steering weight; sadness = tempo + pauses + lower energy |
| **excited / proud / eager** | `--emotion <name> --steer-weight 1.8–2.6` | clean & on-manifold up to ~2.6 (no clipping/harshness) |
| **annoyed / stern** | `--emotion angry --steer-weight 2.6` `--roughness 0.25–0.40` + faster + louder | "tired prof: hey, meeting tomorrow!" — credible irritated/authoritative. **Full furious rage is OUT OF REACH** (model converts it to forceful/proud) |
| **news / announcer** | `--emotion proud` (clean on IT) | reads as authoritative anchor |

**Pauses are a first-class lever** and FREE: ellipsis/commas in the text produce real, natural
pauses in-model (VoiceDesign's sad is mostly pauses+slow). Prefer punctuation over DSP silence
insertion (DSP cuts click — abandoned).

## Per-language notes (steering does NOT transfer uniformly)

The IT/EN-captured palette is **cross-model** (0.6B↔1.7B↔.qvoice) but only **partially cross-language**:

| language | voice | status | use |
|---|---|---|---|
| Italian | ryan / qvoice | ✅ reliable, on-manifold | full palette + recipes |
| Japanese | ono_anna | ✅ very good | full palette |
| English | ryan | ✅ ok | full palette |
| German | ryan | ⚠️ tones similar; angry needs `--roughness 0.40` | excited=irritated, roughness for anger |
| Korean | sohee | ⚠️ partial (sad/happy weak) | use `excited` for joy |
| Spanish | ryan | ❌ inverts: angry→sad, `happy`→sultry | **use `excited` (not happy)**; ES may warrant a dedicated capture |
| French | ryan | ⚠️ `happy`→sultry; `excited` = TOP | **use `excited`**; angry+`roughness 0.40` |

**Cross-language joy fix: always `excited`, never `happy`** (happy inverts/goes sultry in ES/FR/KO; excited transfers).

## Levers & flags

- `--emotion <name[:scale,...]>` — centered palette directions (blendable).
- `--steer-weight <f>` — global scale. **It's a mood crossfade, not just intensity**: push far and
  you land on a neighbour, and at high weight (≳2.6) you push OFF-manifold → language-specific
  emergent deliveries (see below).
- `--roughness <0..1>` — TIMBRE knob (gravel/worn voice), NOT an emotion. Pairs with `--emotion` for grit.
- `QWEN_SPK_SCALE` (env, default 1.0) — diagnostic; scaling the speaker embedding does NOT free pitch
  range (the register clamp is in the WDELTA weights) — relax-identity is a dead end.
- volume / rate — currently DSP post (`ffmpeg volume=`, `atempo=`); `--volume`/`--rate` flags are a
  trivial future add (volume = pure PCM gain, no obstacle unlike pitch).

## Dead-ends (proven — do not re-attempt)

- **Paralinguistic tags** (`<laugh>`, `<sigh>`, `(happy)`): do NOT exist in Qwen3-TTS (verified in
  added_tokens; community feature-requests open). Real laughs/sighs need a 2nd-stage edit model
  (Step Audio EditX), which is ZH/EN-only.
- **Real-breath splice** from the reference: the clone refs have no isolated breaths (glued to words);
  phase-vocoder stretch = metallic. Dead end.
- **DSP silence-insertion**: clicks. Use punctuation pauses instead.
- **relax-identity** (`QWEN_SPK_SCALE<1`): removes drive, doesn't free pitch; at 0.3 = a different voice.
- **`think` / hidden tokens**: `think` is just the language-conditioning slot (verified vs official
  source); no generative reasoning step, no hidden emotion lever. Instruct is Chinese-tuned and weak
  on EU + not even exposed for cloned voices → **CP-steering is the only emotion path for cloned EU voices.**

## Emergent-delivery map (high weight, off-manifold — fragile, ear-curate)

At `--steer-weight ~2.8` the vector becomes a language-specific perturbation that triggers learned
vocal "basins" — a discovery menu, but knife-edge (a 0.2 weight change flips it), so curate by ear:
FR/DE `excited` → irritated; ES `proud` → weary/fed-up; ES `excited` @**2.6** → a bored "eeem"/sigh
(gone at 2.8). Useful as leads to capture as clean dedicated directions, not to ship raw.

**The "eeem"/sbuffo reproducer** (the one emergent paralinguistic worth saving — the closest the
model gets to a spontaneous sigh/filler; audio kept locally at `samples/emergent/`):
```bash
QWEN_EMOTION_DIR=presets/emotions/it_centered ./qwen_tts -d qwen3-tts-0.6b -j1 \
  --seed 42 -l Spanish -s ryan --emotion excited --steer-weight 2.6 \
  --text "La reunión comenzará mañana a las nueve y media en la sala grande." -o ES_sbuffo.wav
```
KNIFE-EDGE: `2.6` = sbuffo, `2.8` = calm. ES+ryan+centered-excited specific.

**Trying to capture it as a clean dedicated direction FAILED** (2026-06-06): a "hesitant/umm/bored"
instruct capture (1.7B) built a genuinely new direction (cos to excited only +0.3), but applied back
it either broke the language (ES h1 @2.2 sounded Chinese, pushed too far off-manifold) or did nothing
(IT/ES h2 = unchanged). So the sbuffo stays a fragile emergent artifact, not a portable preset. The
hesitant `.vec`s live at `/tmp/pal_hes/` (ephemeral) if anyone wants to retry at lower weights.

## Tools
- `tests/steer_center.py` — decorrelate a palette (β/γ + renorm).
- `tests/emotion_compare.sh` — A/B a palette across voices.
- `tests/steer_palette.sh` — capture a calibrated palette (1.7B instruct).
