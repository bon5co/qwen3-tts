---
title: "Emotions on a cloned voice: a 25 MB graft, a steering vector, and a lot of dead ends"
published: false
description: "How we got sad / happy / angry / fearful speech to work on cloned voices in a pure-C Qwen3-TTS engine — a 25 MB 'graft' clone that keeps the emotion levers alive, plus a mixed steering + fine-tune recipe hard-won after many abandoned experiments."
tags: machinelearning, tts, c, audio
---

*Part of [qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts) — a pure C inference engine for Qwen3-TTS.*

## TL;DR

Qwen3-TTS ships 9 neutral preset speakers. No emotion control, and cloning a voice used to mean a huge file that couldn't emote at all. We changed both:

- **Small clones.** A cloned voice is now a **~25 MB `.qvoice` "graft"** — small enough to share, and, crucially, built so the emotion machinery still works on it.
- **Emotions on *any* voice.** One flag — `--emotion <sad|joy|anger|fear|disgust|surprise>` — works on **presets and cloned voices**, in every supported language.

Getting there took an embarrassing number of dead ends. This post is the honest map of what failed and what finally worked.

---

## Why emotion on a clone is hard

The neutral clone was the easy part: feed 30 seconds of audio, an ECAPA-TDNN encoder extracts a speaker embedding, and the model reproduces the timbre. But that clone is **frozen and flat.** It says the words in the right voice with no feeling.

Two forces fight you when you try to add emotion to a clone:

1. **Emotion and timbre live in the same weights.** Push the model toward "angry" and the timbre drifts — you get an angry *stranger*, not an angry *you*.
2. **A cheap clone throws away the levers.** The smallest clone formats (a 4 KB x-vector, a KV-cache prefix) bolt a voice onto the model but lose the internal state the emotion tools need to hook into.

So the clone format and the emotion method are not independent problems. They have to be designed together.

---

## The 25 MB graft: small *and* emotable

We landed on a **graft** `.qvoice` (`--icl-only`): instead of shipping the whole retrained model, it keeps the CustomVoice transformer weights and stores just the delta needed to *be* your voice. About **25 MB**. That's the sweet spot:

- Small enough to attach to an email or check into a repo.
- Preserves full prosody (it's not a lossy 4 KB summary).
- **Keeps the emotion levers alive** — because the CustomVoice weights are still present, the steering and fine-tune hooks have something to grab.

(For reference: a bit-identical clone is ~785 MB, a shareable one ~16 MB, a postcard x-vector ~4 KB — but only the graft keeps the instruct/emotion controls working. Trade-offs, all measured.)

---

## The graveyard of methods that didn't work

Before the recipe that ships, we tried — and abandoned — a lot. Writing them down so nobody (including future us) re-derives them:

- **τ-vectors / task-arithmetic (`.vec`).** Compute an "emotion direction" by float-space arithmetic between neutral and emotional fine-tunes, add it at inference. Elegant on paper; muddy and timbre-shifting in the ear.
- **x-vector emotion injection.** Bolt an emotion onto the 4 KB speaker vector. Too little state to carry it.
- **Per-language dense fine-tunes.** Full FT of layers 16–26 per language. Big, brittle, and it averaged emotions together instead of letting you *select* one.
- **Seed palettes.** Curate "good seeds" per emotion. Fragile and non-portable across voices.
- **Graft-emotion, per-language EXPR-COMBINE variants.** Each solved one case and broke another.

Every one of these produced audio. None of them produced *reliable, selectable, timbre-preserving* emotion on a cloned voice. They're archived — the useful thing they left behind is the recipe below.

---

## What actually works: steer for presets, COMBINE for clones

The shipped system is two hooks used together:

1. **Activation steering.** A tiny, speaker-and-language-agnostic direction added to the residual stream at layers 21–25 at inference time. It nudges "emotion" without touching timbre. The vectors are a few KB and committed to the repo.
2. **CSP fine-tune (`.expr`).** A small weight-delta band (a few layers, LoRA-style) trained on real emotional speech — the **Emozionalmente** Italian emotion corpus (CC-BY 4.0; Catania, Wilke & Garzotto, PoliMi, IEEE TASLP 2025). Romance languages transfer from the Italian pack; other languages get their own small pack.

The one rule:

- **Preset voice → pure steering.** Clean in every language, nothing else needed.
- **Cloned voice → COMBINE** — the language `.expr` fine-tune **plus** the steering vector **plus** an English instruct prompt, applied *together*. Neither alone is enough on a clone; together they push emotion hard enough to be heard while the fine-tune keeps it in-distribution so the timbre survives.

That "together" is the whole trick. The steering vector supplies a clean emotional *direction*; the fine-tune supplies the emotional *texture* the clone lacks; the instruct prompt sets the *strength*. Remove any one and it degrades.

The result is one flag:

```bash
# preset voice
./qwen_tts --emotion sad -s ryan -l English --text "I can't believe he's gone."

# your own 25 MB cloned voice — same flag
./qwen_tts --qvoice me.qvoice --emotion joy -l Italian --text "Ce l'abbiamo fatta!"
```

Native preset per language under the hood (Japanese, Korean, Chinese, Romance, Russian…), so the emotion lands naturally instead of fighting the language.

---

## Bonus: paralinguistics without a splice

Emotion is prosody. **Paralinguistics** — a laugh, a sigh — is an *event*. We ship `[laugh]` and `[sigh]` as inline tags:

```
./qwen_tts --text "That's hilarious [laugh] I can't even."
```

The naive approach is to splice a separate laugh clip in — but a splice sounds like a *different person* laughing. Instead, the tag triggers the event **inline, in one generation, in the voice's own timbre**, so it's your clone laughing, not a stranger. That took its own round of onomatopoeia-by-seed hunting to make universal across voices and languages.

---

## Takeaways

- **Clone format and emotion method are one problem.** The 25 MB graft exists specifically so the steering/fine-tune hooks survive.
- **One clean method beats five clever ones.** τ-vectors, x-vector emotion, dense per-language FT, seed palettes — all archived in favor of *steer + fine-tune, together*.
- **On a clone, layer the levers.** Steering direction + fine-tune texture + instruct strength. Individually weak, together strong.
- **Keep events in-timbre.** A spliced laugh is a stranger; an inline one is you.

It's all pure C, CPU by default, and the emotional-expressivity `.expr` packs are fetched on demand from HuggingFace. Clone your voice once, then make it *feel* something — after, admittedly, a lot of experiments that didn't.
