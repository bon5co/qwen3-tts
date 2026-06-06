# it_centered — decorrelated Italian emotion palette (experimental, 2026-06-06)

Same captures as `presets/emotions/it/` but MEAN-CENTERED: each vector has the shared
"instructed-vs-neutral" common mode removed (`vec' = (vec - mean_all)`, renorm to original
norm), so the tones are near-orthogonal (mean pairwise cosine +0.57 -> -0.09) and sound
distinct instead of all-alike. `angry.vec` = a captured "furious" direction (1.7B instruct),
also centered.

Use: `QWEN_EMOTION_DIR=presets/emotions/it_centered ./qwen_tts ... --emotion <name> --steer-weight 2.0`

VALIDATED RECIPES (Galatea 0.6B, by ear) — steering alone is not enough; pair with rate/volume:
| mood            | recipe                                                            |
|-----------------|-------------------------------------------------------------------|
| joy (real)      | --emotion excited --steer-weight 2.6  + atempo 1.10 + volume 1.10 |
| excited/proud   | --emotion <n> --steer-weight 1.8..2.6                             |
| sad / gloomy    | --emotion <n> --steer-weight ~2.0  + atempo 0.82..0.85 (+ vol down)|
| annoyed/stern   | --emotion angry --steer-weight 2.6 [+ roughness 0.25] + atempo 1.12 + volume 1.3 |
| (full rage)     | OUT OF REACH — model converts "furious" into proud/authoritative   |

`--roughness` is a TIMBRE knob (raspy/worn voice), NOT an emotion.
