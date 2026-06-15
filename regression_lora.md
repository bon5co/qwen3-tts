# Regression: cloned-voice emotion went FLAT — root cause + fix (2026-06-15)

**Symptom (user):** a few days ago `small ICL file + Italian expressivity LoRA + EN instruct + temp + CV model`
emoted well on cloned voices (SAD was great). Now both the small-ICL clone AND the full `.qvoice` clone
render FLAT. Suspected: (1) a C-engine regression, (2) the Italian LoRA retrained on a wider layer band
stealing the clone's timbre layers, (3) the same defect across other languages (FR/ES/...).

**Verdict: NOT a C-engine regression. The Italian (and every) emotion LoRA was retrained BROAD-BAND
(L0-27) instead of the emotion band (L16-26). The broad-band pack disturbs the early identity/timbre/
prosody layers the clone relies on → it flattens instruct-driven emotion, drifts the neutral timbre, and
distorts duration. Hypotheses 2 and 3 CONFIRMED; hypothesis 1 REJECTED.**

---

## 1. Evidence from git (engine is clean)

From the "THE win" commit `0d4a47a` (2026-06-09, clone-emotion `--icl-only` graft) to HEAD, the ONLY commits
that touch C-engine code are:

| commit | date | engine change | verdict |
|---|---|---|---|
| `de4fc4a` | 06-09 | ICL-lite ref_codes + ICL onset trim + asym WAV fade | additive, ICL-only |
| `dec9710` | 06-10 | top-p nucleus mask by sorted index (fix EOS-mask runaway) | benign bugfix, golden unchanged |
| `5950114` | 06-10 | `--graft` / `--icl-frames` (default-off fields) | additive, default bit-identical |
| `ae8340c` | 06-11 | `--expr` `.expr` LoRA loader (`apply_expr_file`) | new feature, applied AFTER voice-load |
| `68f12ae` | 06-13 | `--no-compose` (gated on `text_has_markup`) | no effect on `--instruct` path |

Everything from 06-13 on (paralinguistic, neutral-in-dataset, dataset-prep) is **Python/docs/tests only —
it does NOT touch the inference engine.** `main.c` voice-load → `apply_expr` ordering (line ~2573, after the
voice block) is unchanged since `.expr` landed. So the paralinguistic / neutral experiments did not break
anything in C.

## 2. Empirical A/B (the proof) — `samples/emo_diag/`, mel-corr sad-vs-neutral, lower = more emotion

Params identical everywhere: `-d qwen3-tts-1.7b -l Italian -T 1.1 --seed 42 -j1`,
SAD instruct = "Speak deeply sad and heartbroken, a slow broken voice on the verge of tears."

| case | mel-corr | dur neu→sad | verdict |
|---|---|---|---|
| preset **ryan** (no clone / no expr) | 0.494 | 4.16→3.52s | emotes ✅ |
| preset **vivian** | 0.315 | 4.08→4.24s | emotes strongly ✅ |
| **galatea graft** (`--icl-only`, **NO expr**) = "THE win" | **0.391** | 3.68→2.88s | **clone emotes ✅** |
| galatea graft + `italian_bb027_r32.expr` (L0-27) | 0.660 | 2.16→2.40s | **FLAT ❌** |
| galatea graft + `italian_bb027_ep5_r32.expr` (L0-27) | 0.576 | 4.88→2.48s | flat + duration blown (96%) ❌ |
| neutral-only: graft vs graft+bb027 | 0.575 | — | **the LoRA even moves the neutral timbre ❌** |

→ engine + preset + pure graft all emote (0.31–0.49). Adding the broad-band `bb027` LoRA is what flattens
the clone (0.391 → 0.660) and shifts the neutral voice.

## 3. The artifacts disagree with the repo (the trap)

- Committed `training/expressivity-lora/train_lora.py` and `docs/expressivity-lora.md` say band = **L16-26**.
- But EVERY `.expr` actually on disk that we now load tells a different story:

| `.expr` | layers | tensors |
|---|---|---|
| `paralinguistic_*` (old format) | **L16-26** | 55 |
| `italian_bb027_r32` / `_ep5_r32` | **L0-27 (all 28)** | 165 |
| `french_/german_/spanish_/korean_/portuguese_/russian_ _bb027_r32` | **L0-27** | 165 |

`training/expressivity-lora/DATASETS.md` confirms: *"All broad-band L00-27 r32"* and recommends LOW
`--expr-weight` (RU 0.2 "full rushes the pace", KO ~0.6) — a tell that broad-band at weight 1.0 distorts.
The old ear-validated L16-26 packs (`italian.expr`, `italian_lora_r32.expr`, `italian_lora_r64.expr`)
were **overwritten/deleted** (`.expr` are gitignored → not recoverable from git; no backup found).

## 4. Why broad-band flattens a clone (mechanism)

The good recipe = clone identity/timbre from the ICL prefix (ref_codes pin the early/prosody layers) +
emotion from the **L16-26** LoRA + EN instruct + temp. The L16-26 LoRA adds emotion at the late instruct/
expressivity layers **without touching the layers the clone uses for identity/prosody**. The broad-band
L0-27 LoRA changes q/k/v/o/gate on **every** layer including L0-12 (timbre/prosody) → it fights the ICL
anchor and regresses the model toward the LoRA's averaged (neutral-ish) training distribution → flat,
timbre-drifted, pace-distorted. Same defect on every language because every `bb027` pack is L0-27.

## 5. DGX state (192.168.1.94 = `dgx`, `/home/gabriele/qwen-ft/`)

- Datasets present, ready (with codec codes): `emovo/train_with_codes.jsonl` (IT/EMOVO), plus emodb (DE),
  cafe/esmatch (FR/ES), mesd, etc.
- Real trainer: `Qwen3-TTS/finetuning/dgx_sft_expr_lora.py` — has `--layers` (default **16-26**), `--lora_r`.
- The broad-band run came from `dgx_emovo_broadband.sh`, which **overrides `--layers 0-27`** (its header even
  mis-claims L0-27 "beats emotion-only L16-26" — contradicted by §2). The trainer itself is fine; only the
  invocation was broad-band.
- Export: `expr-lora/export_expr.py ADAPTER_DIR OUT.expr --lang Italian --hidden 2048` → QEXP factored (dtype 5).
- A single r32 run = ~9 min / 8 epochs on the GB10.

## 6. Fix

**Immediate (no retrain):** drop `--expr` and use the pure graft recipe (already emotes, 0.391):
```
./qwen_tts -d qwen3-tts-1.7b --load-voice voices/galatea_17b.qvoice --icl-only \
  -l Italian -T 1.1 --seed 42 -j1 --instruct "<EN emotion instruct>" --text "..." -o out.wav
```
or keep the broad-band pack only at low weight: `--expr italian_bb027_r32.expr --expr-weight ~0.3` (never 1.0).

**Real fix:** retrain the per-language emotion LoRA on **L16-26** (r32 + r64), export to `.expr`, restore the
weight-1.0 validated recipe. Keep `bb027` as a separate "broad-band/style" pack used only at low weight.
See the PLAN TODO. Start with Italian (EMOVO is ready), then FR/DE/ES/...

**Acceptance check:** with `galatea_icl.qvoice` (small ICL) + CV 1.7B + new L16-26 LoRA, SAD must move
(mel-corr meaningfully < neutral baseline), neutral timbre must NOT drift, duration must stay sane — and the
new `.expr` must parse as L16-26 only (ICL owns timbre layers, LoRA owns emotion layers).
