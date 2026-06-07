# Inline expressive markup (audiobooks & podcasts)

Write one text with **inline tags** and the engine renders an expressive, multi-emotion
take in a single pass — different moods per sentence, paragraph-level pauses, and small
paralinguistic fillers (sighs, huffs). The tag style follows the modern AI-TTS convention
(ElevenLabs / Bark): **English tags in square brackets**, placed inline, switchable mid-text.

```bash
./qwen_tts -d qwen3-tts-0.6b -s ryan -l English \
  --text "I couldn't believe it. [excited] We actually won! [pause:500ms] [sad] But it meant leaving everyone behind. [sigh]" \
  -o out.wav
```

You don't need a special flag: if `--text` contains recognized tags it is rendered as
expressive markup automatically. (`--compose "..."` does the same explicitly.)

## Tags

| tag | effect |
|---|---|
| `[happy] [sad] [excited] [eager] [proud] [calm] [dramatic] [news] [annoyed] [stern] [angry] [joy] [gloomy]` | switch the emotion for the text that follows (full recipe: steering + roughness + volume + rate) |
| `[neutral]` (`[none]`, `[normal]`) | back to plain, unmodified delivery |
| `[sigh]` `[sighs]` `[groan]` `[hmm]` | weary paralinguistic filler (slow + soft) |
| `[huff]` `[ugh]` | clipped, irritated filler |
| `[pause:400ms]` `[pause:1s]` `[pause:0.5]` `[break:300ms]` `[0.5]` | insert a pause (ms or seconds; bare number = seconds) |

- Tags are **case-insensitive** and **always English** (`[sigh]`, not `[sospiro]`).
- An emotion tag stays active until the next emotion tag or `[neutral]`.
- Text before the first tag is spoken neutrally.
- **Unrecognized** `[...]` is left as literal text — a stray bracket won't break your script.
- The full emotion list is in [expressivity.md](expressivity.md); each mood's recipe is in
  [expressivity-recipes.md](expressivity-recipes.md).

## How it renders

Each span (a run of text under one emotion) is synthesized **separately** with its own
recipe, then all spans are concatenated into one WAV with the pauses you asked for. Because
every span is **model-generated** (same voice, same 24 kHz codec), the joins are seamless —
this is *not* audio splicing from a reference, so there are no phase/timbre artifacts.

Adjacent spoken spans get a small default gap (`--compose-pause`, default `0.12s`) so words
don't collide; explicit `[pause:…]` tags add exactly the silence you specify.

## Paralinguistic fillers — what they are (and aren't)

`[sigh]`/`[huff]`/etc. are **approximations**, not recorded breaths. The trick: the slow+soft
`sad` recipe stretches a short vowel (`"Ehh…"`, `"Uff…"`) into a weary-sounding filler. They
read convincingly as tiredness/exasperation in context, but they are synthesized vowels, not
true non-verbal breaths (the model has no real `<breath>` token — see the dead-ends in
[expressivity-recipes.md](expressivity-recipes.md)).

## Tuning

- Per-span moods use their **baked recipe** weight/rate/volume (calibrated by ear). To bias a
  whole render louder/slower, add global `--volume`/`--rate` (they post-process the final mix).
- For Italian, the language-aware resolver automatically uses the centered palette — just pass
  `-l Italian`.
- Want a custom default gap between spans? `--compose-pause 0.25`.

## Examples

```bash
# Audiobook beat: setup (neutral) -> reveal (excited) -> turn (sad + sigh)
./qwen_tts -d qwen3-tts-0.6b -s ryan -l English \
  --text "The letter sat on the table for days. [pause:400ms] [excited] When she finally opened it, she gasped. [pause:600ms] [sad] It was the goodbye she'd feared. [sigh]" \
  -o scene.wav

# Italian, explicit --compose form (| is an optional hard span break)
./qwen_tts -d qwen3-tts-0.6b -s ryan -l Italian \
  --compose "[annoyed] Te l'avevo detto! [pause:300ms] | [neutral] Va bene, ricominciamo. [sigh]" \
  -o dialogo.wav

# Tired character: huff, beat, resigned line
./qwen_tts -d qwen3-tts-0.6b -s ryan -l English \
  --text "[huff] [pause:300ms] [sad] Fine. I'll do it myself." -o tired.wav
```
