# Emotional-speech datasets for `.expr` training (per language)

A survey of public emotional-speech corpora to train a `<lang>.expr` pack. What we need (see the
[README](README.md)): **multiple emotions × multiple speakers**, audio resampleable to **24 kHz
mono**, and ideally **known transcripts** (acted corpora with fixed sentences are easiest — our
`prepare_manifest.py` needs `text` per clip; spontaneous corpora need ASR first).

> ⚖️ **License matters for shipping.** Many SER corpora are **research / non-commercial** (CC-BY-NC-SA).
> Those are fine for personal/experimental `.expr` files but **cannot be redistributed/shipped**
> as a derivative. Prefer CC-BY / permissive if you want to release the pack. Always verify the
> dataset's own license before training a pack you intend to publish.

## Ready-to-use hub: `confit` (same parquet loader as our EMOVO example)

The [`confit`](https://huggingface.co/confit) HF org hosts parquet versions of several corpora with
a uniform schema (audio + emotion label), so they drop into the same pipeline as EMOVO:

| dataset | lang | emotions | clips | sr | note |
|---|---|---|---|---|---|
| `confit/emovo-parquet` | **Italian** | 7 | 588 | 48k | our worked example |
| `confit/emodb-parquet` | **German** | 7 | 535 | 48k→16k | **DE pack is immediately doable** |
| `confit/ravdess-parquet` | English | 8 | 2880 | 48k | acted, 24 speakers |
| `confit/crema-d-parquet` | English | 6 | 7442 | 16k | 91 speakers (big) |
| `confit/iemocap-parquet` | English | 4 | 5531 | 16k | dyadic |

(confit clips carry emotion labels; for fixed-sentence corpora the transcripts are known and can be
mapped by clip id, like our EMOVO example.)

## Per target language

### 🇩🇪 German — **EmoDB (Berlin)** ✅ easiest
7 emotions (anger, boredom, anxiety, happiness, sadness, disgust, neutral), 10 actors (5M/5F),
535 utterances, 48k→16k, **CC-BY** (relatively permissive). On `confit` as parquet → reuse the
EMOVO flow directly. Small (535) — expect r32/r64 to work; more data would help.
Sources: [Emo-DB](http://emodb.bilderbar.info/start.html), [confit/emodb-parquet](https://huggingface.co/datasets/confit/emodb-parquet).

### 🇪🇸 Spanish — **EmoMatchSpanishDB** (main) + MESD (supplement)
- **EmoMatchSpanishDB**: 6 Ekman emotions + neutral, **50 speakers** (31M/19F), ~2,050 elicited
  clips — good speaker diversity for voice-agnostic training. Verify license (academic).
- **MESD** (Mexican Spanish): 6 emotions, 864 **single-word** utterances, adults + children —
  word-level limits prosody learning; use as a supplement, not the base.
- (In-the-wild: EMOVOME spontaneous voice messages — needs ASR transcripts.)
Sources: [EmoMatchSpanishDB (UPM)](https://oa.upm.es/80921/), [MESD](https://data.mendeley.com/).

### 🇫🇷 French — **CaFE** (quality) + **Oréau** (standard accent)
- **CaFE** (Canadian French): 6 emotions + neutral × 2 intensities, **12 actors** (6M/6F),
  192 kHz/24-bit (excellent), **CC-BY-NC-SA** (non-commercial). Accent = Canadian.
- **Oréau** (standard/European French): 7 emotions, **32 speakers**, ~79 utterances (few per
  emotion), non-commercial. Standard accent but small.
- Best: combine — CaFE for clean acted range, Oréau for European accent. Both NC → personal use.
Sources: [CaFE](https://dl.acm.org/doi/10.1145/3204949.3208121), Oréau (Zenodo/SER-datasets list).

### 🇵🇹 Portuguese — **VERBO** (Brazilian; main)
- **VERBO**: acted emotional speech, Brazilian Portuguese — the most usable acted SER corpus for PT.
- **CORAA SER** (BR, spontaneous): only 3 classes (neutral / non-neutral M/F) → too coarse for our
  per-emotion instructs; not ideal.
- **European Portuguese**: only small acted corpora in the literature; scarce — BR is the practical base.
Sources: [VERBO](https://thescipub.com/abstract/jcssp.2018.1420.1430), [CORAA SER](https://github.com/rmarcacini/ser-coraa-pt-br).

## Multilingual / aggregators (useful for scale or extra languages)
- **ESD** — Emotional Speech Database: EN + ZH, 10+10 speakers, 5 emotions (clean, acted).
- **CAMEO** — Collection of Multilingual Emotional Speech Corpora ([arXiv 2505.11051](https://arxiv.org/html/2505.11051)) — a curated multilingual set.
- **EmoBox** / [SuperKogito/SER-datasets](https://github.com/SuperKogito/SER-datasets) — big indexes of SER corpora by language + license.
- **nEMO** — Polish (9 actors, 6 emotions) if you want a Polish pack.

## Transcripts — published, not invented

Acted corpora use a **fixed sentence set** that's published in the dataset's paper/docs — you map
each clip's code to its sentence, you don't transcribe by hand:

- **EmoDB (DE)** — 10 sentences, codes `a01..b10` (source: audeering audformat reference + Burkhardt
  et al. 2005). **Already built into `prepare_manifest.py` (`--emodb`).** Use the *original* EmoDB
  download (filenames like `03a01Fa.wav` encode text+emotion); the `confit` parquet drops the codes/text.
- **EMOVO (IT)** — 14 sentences (the worked example, also built in).
- **CaFE (FR)** — 6 sentences (published with phonemic transcriptions in the CaFE paper).
- **RAVDESS (EN)** — 2 fixed statements ("Kids are talking by the door" / "Dogs are sitting by the door").
- **MESD (ES)** — single-word list (published).
- **EmoMatchSpanishDB (ES) / VERBO (PT)** — check the paper/repo for the prompt set; elicited/acted
  corpora ship their sentence list. Spontaneous corpora (EMOVOME, CORAA) have **no fixed script** → run ASR.

So for the acted EU corpora the transcripts come straight from the dataset's own documentation; only
the spontaneous ones need ASR.

## Practical recommendation (priority order)
1. **German** — EmoDB via `confit` (ready, CC-BY) — fastest second language after Italian.
2. **Spanish** — EmoMatchSpanishDB (50 speakers) — best speaker diversity.
3. **French** — CaFE + Oréau (note Canadian vs European accent; NC license = personal use).
4. **Portuguese** — VERBO (Brazilian).

For each: aim for ≥ a few hundred clips across ≥ several speakers and the full emotion set; map each
emotion to a vivid **English** instruct (see `prepare_manifest.py`'s `EMOTION_INSTRUCT`); then run the
4-step pipeline. More & varied data → richer expressivity and better language-prosody/timbre.
